"""tools/wave32_pixel_scan.py: families, wave32/stage selection and privacy, on hand-written listings; no game data.

    python -I tests/test_wave32_pixel_scan.py -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fssm_header  # noqa: E402
import wave32_pixel_scan as scan  # noqa: E402


def fixture(name, lines):
    """A synthetic disassembly: shader_disasm's '; name' header, then tab-prefixed instructions."""
    body = "\n".join("\t" + ln for ln in lines)
    return name, "; %s  code 0x0..0x10 (header), gfx1013, 0 invalid words\n%s\n" % (name[:-2], body)


PS_LANE_ID = fixture("ps_lane.s", ["v_mbcnt_lo_u32_b32 v0, -1, 0", "s_endpgm"])
PS_MASK_WORD = fixture("ps_bcnt.s", [
    "s_bcnt1_i32_b32 s0, exec_lo",                  # EXEC: the wave's lane count
    "v_cmp_gt_f32_e64 s4, v0, v1",                  # s4 holds a lane mask (lane_masks.py)
    "s_ff1_i32_b32 s5, s4",                         # so this is the wave's first lane
    "s_load_dword s6, s[0:1], 0x0",
    "s_bcnt1_i32_b32 s7, s6",                       # data from memory: not counted
    "s_endpgm",
])
PS_DATA_ONLY = fixture("ps_data.s", [
    "s_load_dword s6, s[0:1], 0x0",
    "s_bcnt1_i32_b32 s7, s6",
    "s_and_b32 s8, s6, s7",
    "s_cselect_b32 s9, 1, 0",                       # SCC from data: not a vote
    "v_add_f32_e32 v0, v1, v2",
    "s_endpgm",
])
PS_SELECT = fixture("ps_rfl.s", ["v_readfirstlane_b32 s0, v1", "v_readlane_b32 s1, v1, 5", "s_endpgm"])
PS_PARKED = fixture("ps_parked.s", [
    "v_cmp_gt_f32_e64 s4, v0, v1",
    "v_writelane_b32 v9, s4, 0",                    # parks the mask in lane 0 of v9: a lane write, counted
    "v_readlane_b32 s5, v9, 0",                     # fetches it back: the translator reads its own copy
    "v_mov_b32_e32 v8, exec_lo",                    # a mask through a VGPR
    "v_readfirstlane_b32 s6, v8",                   # each lane reads its own bit: not counted
    "s_endpgm",
])
PS_PERMUTE = fixture("ps_swz.s", ['ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,"pippi")', "s_endpgm"])
PS_QUAD = fixture("ps_quad.s", [
    "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,1,0,3,2)",
    "v_mov_b32_dpp v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "s_wqm_b32 exec_lo, exec_lo",
    "s_endpgm",
])
PS_DPP_ROW = fixture("ps_dpp.s", ["v_mov_b32_dpp v2, v3 row_shr:1 row_mask:0xf bank_mask:0xf", "s_endpgm"])
PS_VOTE = fixture("ps_vote.s", [
    "v_cmp_gt_f32_e32 vcc_lo, v0, v1",
    "s_and_saveexec_b32 s0, vcc_lo",                # SCC = the wave's new EXEC is not empty
    "s_cselect_b32 s1, 1, 0",                       # read: a vote over the subgroup
    "s_endpgm",
])
E4, E8 = " ; encoding: [0,0,0,0]", " ; encoding: [0,0,0,0,0,0,0,0]"   # only the lengths matter here
# Two basic blocks joined at pc 20. s4 is a lane mask on both paths in the first, on one path only in the second
# (data from memory on the other), so lane_masks.py drops it at the join and the s_bcnt1 there reads data.
PS_BRANCH_MASK = fixture("ps_brmask.s", [
    "v_cmp_gt_f32_e64 s4, v0, v1" + E8,             # pc 0
    "s_cbranch_execz 2" + E4,                       # pc 8 -> 20: a whole-wave branch (vote)
    "v_cmp_lt_f32_e64 s4, v0, v1" + E8,             # pc 12
    "s_bcnt1_i32_b32 s5, s4" + E4,                  # pc 20: a mask on both paths
    "s_endpgm" + E4,
])
PS_BRANCH_DATA = fixture("ps_brdata.s", [
    "s_load_dword s4, s[0:1], 0x0" + E8,            # pc 0
    "s_cbranch_scc0 2" + E4,                        # pc 8 -> 20 (SCC not from a mask: no vote)
    "v_cmp_lt_f32_e64 s4, v0, v1" + E8,             # pc 12
    "s_bcnt1_i32_b32 s5, s4" + E4,                  # pc 20: a mask on one path only
    "s_endpgm" + E4,
])
PS_W64 = fixture("ps_w64.s", ["s_and_saveexec_b64 s[0:1], vcc", "v_mbcnt_lo_u32_b32 v0, -1, 0", "s_endpgm"])
VS_W32 = fixture("vs_w32.s", ["v_mbcnt_lo_u32_b32 v0, -1, 0", "s_endpgm"])


class ScanTests(unittest.TestCase):
    def scan(self, *fixtures):
        with tempfile.TemporaryDirectory() as directory:
            for name, text in fixtures:
                with open(os.path.join(directory, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            return scan.scan_directory(directory)

    def test_families(self):
        r = self.scan(PS_LANE_ID, PS_MASK_WORD, PS_SELECT, PS_PERMUTE, PS_DPP_ROW, PS_VOTE)
        self.assertEqual(r["occurrences"], {"lane_id": 1, "mask_word": 2, "lane_select": 2, "permute": 1,
                                            "dpp_row": 1, "vote": 1, "quad": 0})
        self.assertEqual((r["ps_wave32"], r["crossing"], r["quad_only"]), (6, 6, 0))
        self.assertEqual(r["only_family"]["mask_word"], 1)

    def test_data_and_quads_do_not_cross(self):
        r = self.scan(PS_DATA_ONLY, PS_QUAD)
        self.assertEqual((r["ps_wave32"], r["crossing"], r["quad_only"]), (2, 0, 1))
        self.assertEqual(r["occurrences"]["quad"], 3)
        self.assertEqual(r["shaders"]["vote"] + r["shaders"]["mask_word"], 0)

    def test_parked_and_vgpr_masks_are_not_lane_selects(self):
        r = self.scan(PS_PARKED)
        self.assertEqual(r["occurrences"]["lane_select"], 1)

    def test_blocks_follow_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, PS_BRANCH_MASK[0])
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(PS_BRANCH_MASK[1])
            blocks, succ = scan.blocks_of(scan.parse_file(path)["insns"])
        self.assertEqual([len(b) for b in blocks], [2, 1, 2])
        self.assertEqual(succ, [[2, 1], [2], []])
        self.assertEqual(scan.scan_shader(blocks[0] + blocks[1] + blocks[2]), {"vote": 1, "mask_word": 1})
        r = self.scan(PS_BRANCH_MASK, PS_BRANCH_DATA)
        self.assertEqual(r["occurrences"]["mask_word"], 1)   # the join drops the one-path mask
        self.assertEqual(r["occurrences"]["vote"], 1)
        self.assertEqual((r["crossing"], r["only_family"]["vote"]), (1, 0))

    def test_scc_overwritten_by_a_scalar_op_is_not_a_vote(self):
        head = ["v_cmp_gt_f32_e32 vcc_lo, v0, v1", "s_and_saveexec_b32 s0, vcc_lo"]
        tail = ["s_cselect_b32 s1, 1, 0", "s_endpgm"]
        for middle, votes in (([], 1), (["s_mov_b32 s2, 0"], 1), (["s_mov_b32 s2, 0", "s_addk_i32 s2, 1"], 0),
                              (["s_add_i32 s2, 0, 1"], 0), (["s_cmpk_eq_i32 s2, 1"], 0)):
            with self.subTest(middle=middle):
                r = self.scan(fixture("ps_probe.s", head + middle + tail))
                self.assertEqual((r["occurrences"]["vote"], r["crossing"]), (votes, votes))

    def test_only_wave32_pixel_shaders(self):
        r = self.scan(PS_LANE_ID, PS_W64, VS_W32)
        self.assertEqual((r["ps"], r["ps_wave32"], r["crossing"]), (2, 1, 1))

    def test_fssm_header_decides_the_wave_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "disasm").mkdir()
            (root / "fssm").mkdir()
            for size, flags in ((32, fssm_header.FLAG_WAVE32), (64, 0)):
                name = f"ps_header{size}"
                listing = fixture(name + ".s", ["v_readfirstlane_b32 s0, v1", "s_endpgm"])[1]   # no wave64 marker
                (root / "disasm" / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")
                header = bytearray(fssm_header.WAVE_FLAGS_OFFSET + 4)
                header[:4] = fssm_header.MAGIC
                struct.pack_into("<I", header, fssm_header.WAVE_FLAGS_OFFSET, flags)
                (root / "fssm" / (name + ".fssm")).write_bytes(header)
            r = scan.scan_directory(str(root / "disasm"))
            self.assertEqual((r["ps"], r["ps_wave32"], r["crossing"]), (2, 1, 1))

    def test_report_has_no_names(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as outdir:
            for name, text in (PS_LANE_ID, PS_VOTE):
                with open(os.path.join(src, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            out = os.path.join(outdir, "scan.json")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(scan.main([src, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                blob = f.read()
            json.loads(blob)
            for text in (blob, printed.getvalue()):
                self.assertNotIn("ps_lane", text)
                self.assertNotIn("ps_vote", text)

    def test_missing_source_is_an_error(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            scan.main(["no_such_shaders_dir_xyz"])
        self.assertNotEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
