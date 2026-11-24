"""s_bfm_b64 in wave32 code: 64-bit data, not a lane mask.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA SOP2 S_BFM_B64: D = ((1 << (S0 & 63)) - 1) << (S1 & 63), SCC not written. In a 64-lane wave the
translator reads D as a lane mask (each invocation holds its own lane's bit; test_salu_bfm64_add_i32.py). A 32-lane
wave has no 64-bit lane masks, so there it is plain 64-bit data. The pipeline library's wave32 compute programs
(docs/shader_library.md) use it that way: `s_bfm_b64 s[88:89], 2, 54` and `s_mov_b64 s[90:91], 0x18500000` build
an S# in place, which image_sample then reads. Their kind-0 dispatch state says wave32, and until this change the
translator refused all 958 of them ("FSPX wave32 dispatch state contradicts wave64 code"), counting s_bfm_b64 as a
wave64 marker.

Each case writes fspx/<name>.json with the wave size (the sidecar tools/shader_library.py writes), translates a
compute listing, runs every invocation with run_workgroup and has each store both words at its own slot.
"""
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0x4000, 0x31016FAC]
RSRC2 = 0x88                    # 4 user SGPRs (the output V#), workgroup id x in s4, thread id x in v0
THREADS = 32


def translate(body, wave32, name="cs_bfm"):
    lines = list(body) + ["s_endpgm"]
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fspx"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write("; " + name + "\n" + "".join(f"\t{ln}\n" for ln in lines))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"stage": "cs", "rsrc2": RSRC2, "threads": [THREADS, 1, 1], "wave32": wave32}, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))


def words(size, offset):
    """Each invocation stores s8, s9 at byte 8 * i; returns the (lo, hi) of every invocation."""
    spv = translate([f"s_bfm_b64 s[8:9], {size}, {offset}", "v_lshlrev_b32 v3, 3, v0",
                     "v_mov_b32 v5, s8", "v_mov_b32 v6, s9", "buffer_store_dwordx2 v[5:6], v3, s[0:3], 0 offen"],
                    wave32=True)

    def inputs(i):
        return {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i}
    mem = si.run_workgroup(spv, THREADS, 32, inputs_for=inputs, push_constants=VSHARP + [0] * 28, memory={})
    return {(mem.get(8 * i), mem.get(8 * i + 4)) for i in range(THREADS)}


class Wave32BfmTests(unittest.TestCase):
    def test_value_in_every_invocation(self):
        for size, offset in ((2, 54), (1, 63), (32, 8), (32, 32), (32, 0), (0, 5), (64, 0), (65, 66), (8, 28)):
            with self.subTest(size=size, offset=offset):
                v = (((1 << (size & 63)) - 1) << (offset & 63)) & 0xFFFFFFFFFFFFFFFF
                self.assertEqual(words(size, offset), {(v & 0xFFFFFFFF, v >> 32)})

    def test_sampler_words_are_known_constants(self):
        # The S# built in place reaches the index's literal samplers, as one moved with s_mov_b64 does.
        spv = translate(["s_bfm_b64 s[88:89], 2, 54", "s_mov_b64 s[90:91], 0x18500000",
                         "s_load_dwordx8 s[80:87], s[0:1], 0x0", "v_mov_b32 v1, 0", "v_mov_b32 v2, 0",
                         "s_waitcnt lgkmcnt(0)",
                         "image_sample_lz v5, v[1:2], s[80:87], s[88:91] dmask:0x1 dim:SQ_RSRC_IMG_2D",
                         "s_waitcnt vmcnt(0)", "buffer_store_dword v5, v0, s[0:3], 0 offen"], wave32=True)
        self.assertIn("OpImageSample", spv)
        self.assertIn("OpConstant %uint 12582912", spv)         # word 1 = (3 << 54) >> 32, folded

    def test_fspx_wave32_with_bfm_is_not_refused(self):
        translate(["s_bfm_b64 s[8:9], 2, 54", "s_and_saveexec_b32 s10, vcc_lo", "s_mov_b32 exec_lo, s10"],
                  wave32=True)

    def test_other_wave64_markers_still_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["s_bfm_b64 s[8:9], 2, 54", "s_and_saveexec_b64 s[10:11], vcc"], wave32=True)

    def test_run_time_operand_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["s_mov_b32 s5, 2", "v_readfirstlane_b32 s6, v0", "s_bfm_b64 s[8:9], s6, 54"], wave32=True)

    def test_lane_masks_leaves_wave32_result_data(self):
        ins = {"op": "s_bfm_b64", "args": [{"k": "sgpr", "r": 8, "n": 2}, {"k": "imm", "v": 2},
                                            {"k": "imm", "v": 54}], "mods": []}
        st64, st32 = {}, {}
        lane_masks.step(st64, ins)
        lane_masks.step(st32, ins, wave64=False)
        self.assertEqual((st64.get(8), st64.get(9)), (lane_masks.MASK_LO, lane_masks.MASK_HI))
        self.assertEqual((st32.get(8), st32.get(9)), (None, None))


if __name__ == "__main__":
    unittest.main()
