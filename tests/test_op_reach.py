"""op_reach: shaders and instructions per base instruction name, on hand-written listings.

    python -I tests/test_op_reach.py

Every input is built here (five tiny .s listings, an index.json and FSSM headers with only the magic and the
wave flag). No game data.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "op_reach.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import op_reach  # noqa: E402

LISTINGS = {
    # both encodings of one instruction: one shader, two instructions
    "a_ps": "; a_ps  code 0x0..0x10, gfx1013, 0 invalid words\n"
            "\tv_cndmask_b32_e32 v0, v1, v2, vcc\n\tv_cndmask_b32_e64 v3, v1, v2, s[0:1]\n\ts_endpgm\n",
    # an SDWA and a DPP form; a comment line naming an instruction is not counted
    "b_vs": "; v_rcp_f64 in a comment\n"
            "\tv_add_f16_sdwa v0, v1, v2 dst_sel:WORD_1\n\tv_add_f16_dpp v0, v1, v2 quad_perm:[0,1,2,3]\n"
            "\tv_cndmask_b32_e32 v0, v1, v2, vcc\n\ts_endpgm\n",
    # a name that merely ends like a suffix keeps it: v_cvt_pkrtz_f16_f32 has no encoding suffix
    "c_cs": "\tv_cvt_pkrtz_f16_f32_e32 v0, v1, v2\n\ts_endpgm\n",
    # two programs, then data words decoded as instructions after the last s_endpgm: those are not counted
    "d_cs": "\tv_exp_f16_e32 v0, v1\n\ts_endpgm\n\tv_cos_f16_e32 v0, v1\n\ts_endpgm\n"
            "\ts_cmov_b32 s0, s1\n\tv_mac_f32_e32 v0, v1, v2\n",
    # no s_endpgm at all: the whole listing counts
    "e_cs": "\tv_cvt_f16_f32_e32 v0, v1\n",
}


def build(root):
    disasm = os.path.join(root, "disasm")
    os.makedirs(disasm)
    for stem, text in LISTINGS.items():
        with open(os.path.join(disasm, stem + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    index = os.path.join(root, "index.json")
    with open(index, "w", encoding="utf-8") as f:
        json.dump([{"name": s, "file": s + ".dxbc", "stage": s.split("_")[1]} for s in LISTINGS], f)
    return disasm, index


# Compiled wave size per listing; e_cs has no sidecar header.
WAVES = {"a_ps": 64, "b_vs": 64, "c_cs": 32, "d_cs": 64}


def build_fssm(root):
    fssm = os.path.join(root, "fssm")
    os.makedirs(fssm)
    for stem, wave in WAVES.items():
        header = bytearray(0x100)
        header[:4] = b"FSSM"
        struct.pack_into("<I", header, 0xD4, 0x20 if wave == 32 else 0)
        with open(os.path.join(fssm, stem + ".fssm"), "wb") as f:
            f.write(header)
    return fssm


def run(*args):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)


class OpReachTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.disasm, self.index = build(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_name_drops_only_the_encoding_suffix(self):
        self.assertEqual(op_reach.base_name("v_cndmask_b32_e64"), "v_cndmask_b32")
        self.assertEqual(op_reach.base_name("v_add_f16_sdwa"), "v_add_f16")
        self.assertEqual(op_reach.base_name("v_add_f16_dpp"), "v_add_f16")
        self.assertEqual(op_reach.base_name("v_cvt_pkrtz_f16_f32"), "v_cvt_pkrtz_f16_f32")
        self.assertEqual(op_reach.base_name("s_endpgm"), "s_endpgm")

    def test_encodings_merge_and_a_shader_counts_once(self):
        r = op_reach.census(self.disasm)
        self.assertEqual(r["listings"], 5)
        self.assertEqual(r["shaders"]["v_cndmask_b32"], 2)
        self.assertEqual(r["instructions"]["v_cndmask_b32"], 3)
        self.assertEqual(r["shaders"]["v_add_f16"], 1)
        self.assertEqual(r["instructions"]["v_add_f16"], 2)
        self.assertNotIn("v_rcp_f64", r["shaders"])        # named in a comment only

    def test_only_the_program_up_to_the_last_s_endpgm_counts(self):
        r = op_reach.census(self.disasm)
        self.assertEqual(r["shaders"]["v_exp_f16"], 1)      # first program
        self.assertEqual(r["shaders"]["v_cos_f16"], 1)      # second program, before the last s_endpgm
        self.assertNotIn("s_cmov_b32", r["shaders"])       # data words after it
        self.assertNotIn("v_mac_f32", r["shaders"])
        self.assertEqual(r["after_last_endpgm"], 2)
        self.assertEqual(r["shaders"]["v_cvt_f16_f32"], 1)  # a listing without s_endpgm counts whole

    def test_selected_names_report_zero_and_stages(self):
        r = op_reach.census(self.disasm, op_reach.stages_of(self.index))
        rows = op_reach.report(r, ["v_cndmask_b32_e64", "v_rcp_f64"])
        self.assertEqual(rows[0], {"op": "v_cndmask_b32", "shaders": 2, "instructions": 3,
                                   "by_stage": {"ps": 1, "vs": 1}})
        self.assertEqual(rows[1], {"op": "v_rcp_f64", "shaders": 0, "instructions": 0, "by_stage": {}})
        self.assertEqual(r["listings_by_stage"], {"cs": 3, "ps": 1, "vs": 1})

    def test_cli_json(self):
        out = os.path.join(self._tmp.name, "out.json")
        p = run(self.disasm, "v_cvt_pkrtz_f16_f32", "v_log_f16", "--index", self.index, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("5 listings", p.stdout)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["after_last_endpgm"], 2)
        self.assertEqual([(row["op"], row["shaders"]) for row in doc["rows"]],
                         [("v_cvt_pkrtz_f16_f32", 1), ("v_log_f16", 0)])

    def test_missing_folder_is_a_usage_error(self):
        p = run(os.path.join(self._tmp.name, "no-such-folder"))
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.startswith("ERROR:"), p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_empty_folder_fails(self):
        empty = os.path.join(self._tmp.name, "empty")
        os.makedirs(empty)
        p = run(empty)
        self.assertEqual(p.returncode, 1)
        self.assertTrue(p.stderr.startswith("ERROR:"), p.stderr)

    def test_patterns_match_printed_names(self):
        r = op_reach.census(self.disasm)
        rows = {row["op"]: row for row in op_reach.report(r, ["*_dpp", "*_sdwa", "v_cndmask_b32_e*", "v_add_f16_sdwa"])}
        self.assertEqual((rows["*_dpp"]["shaders"], rows["*_dpp"]["instructions"]), (1, 1))
        self.assertEqual((rows["*_sdwa"]["shaders"], rows["*_sdwa"]["instructions"]), (1, 1))
        self.assertEqual((rows["v_cndmask_b32_e*"]["shaders"], rows["v_cndmask_b32_e*"]["instructions"]), (2, 3))
        # without a wildcard a suffixed name is its base name: every encoding counts
        self.assertEqual((rows["v_add_f16"]["shaders"], rows["v_add_f16"]["instructions"]), (1, 2))
        # a pattern never matches the data words after the last s_endpgm
        self.assertEqual(op_reach.report(r, ["s_cmov*"])[0]["shaders"], 0)

    def test_wave_split_from_the_fssm_headers(self):
        fssm = build_fssm(self._tmp.name)
        r = op_reach.census(self.disasm, op_reach.stages_of(self.index), fssm)
        self.assertEqual(r["listings_by_wave"], {"w32": 1, "w64": 3, "w?": 1})
        self.assertEqual(r["listings_by_stage_wave"], {"cs/w32": 1, "cs/w64": 1, "cs/w?": 1, "ps/w64": 1, "vs/w64": 1})
        rows = {row["op"]: row for row in op_reach.report(r, ["v_cndmask_b32", "v_cvt_pkrtz_f16_f32", "v_cvt_f16_f32"])}
        self.assertEqual(rows["v_cndmask_b32"]["by_wave"], {"w64": 2})
        self.assertEqual(rows["v_cndmask_b32"]["by_stage_wave"], {"ps/w64": 1, "vs/w64": 1})
        self.assertEqual(rows["v_cvt_pkrtz_f16_f32"]["by_wave"], {"w32": 1})
        self.assertEqual(rows["v_cvt_f16_f32"]["by_wave"], {"w?": 1})      # no sidecar

    def test_cli_fssm(self):
        fssm = build_fssm(self._tmp.name)
        out = os.path.join(self._tmp.name, "w.json")
        p = run(self.disasm, "*_dpp", "--fssm", fssm, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["listings_by_wave"], {"w32": 1, "w64": 3, "w?": 1})
        self.assertEqual(doc["rows"][0]["by_wave"], {"w64": 1})
        p = run(self.disasm, "--fssm", os.path.join(self._tmp.name, "no-such-fssm"))
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.startswith("ERROR:"), p.stderr)


if __name__ == "__main__":
    unittest.main()
