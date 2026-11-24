"""The wave size of an FSPX program (the FidelityFX DLL's, docs/fspx.md) comes from its dispatch state: kind 0 word 4
bit 20 set = wave32, which tools/fspx_unpack.py writes to fspx/<name>.json as `wave32`. Before, a program without an
FSSM sidecar got the code heuristic, which also counts any _b64 op on vcc or exec as wave64; the DLL's wave32
programs move vcc as an ordinary SGPR pair, so they were translated as wave64, and a wave64 program without any
marker as wave32.

Every case is a hand-written listing in the FSPX layout (disasm/, fspx/<name>.json and .code, no fssm/), translated
by tools/rdna2spirv.py. The last class runs the result, where the two wave sizes give different answers.
Standard library only; no GPU, no LLVM, no game files.
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
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# 4 user SGPRs (the output V# in s[0:3]) and the workgroup id X; 64 threads.
RSRC2 = 0x88
VSHARP = [0, 0, 0x1000, 0x31016FAC]
# A _b64 op that names vcc but uses it as an ordinary SGPR pair: what the DLL's wave32 programs do.
VCC_AS_DATA = ["s_mov_b64 vcc, s[2:3]", "s_mov_b32 s6, vcc_lo"]


def translate(body, wave32=None, fssm=None, name="cs_wave"):
    """Translate `body` as an FSPX program whose dispatch state says `wave32` (True, False, or None: field absent),
    optionally with an FSSM sidecar too. Returns (spvasm text, wave64 chosen)."""
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fspx", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        meta = {"rsrc2": RSRC2, "threads": [64, 1, 1]}
        if wave32 is not None:
            meta["wave32"] = wave32
        with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(os.path.join(tmp, "fspx", name + ".code"), "wb") as f:
            f.write(b"\0" * 64)
        if fssm is not None:
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(fssm)
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                asm = rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        return asm, rdna2spirv.translate.last_gen.wave64


def fssm_header(wave32):
    """An FSSM header with the compute words the translator reads and the wave-size flag (0xD4 bit 5 = wave32)."""
    hdr = bytearray(0x148)
    hdr[:4] = b"FSSM"
    struct.pack_into("<4I", hdr, 0x138, RSRC2, 64, 1, 1)
    struct.pack_into("<I", hdr, 0xD4, 0x20 if wave32 else 0)
    return bytes(hdr)


class WaveSizeFromDispatchState(unittest.TestCase):
    def test_wave32_state_with_vcc_as_data_is_wave32(self):
        asm, wave64 = translate(VCC_AS_DATA, wave32=True)
        self.assertFalse(wave64)

    def test_wave64_state_without_any_marker_is_wave64(self):
        asm, wave64 = translate(["v_mov_b32_e32 v1, v0"], wave32=False)
        self.assertTrue(wave64)

    def test_wave32_state_against_a_strict_wave64_marker_is_refused(self):
        for marker in ("v_mbcnt_hi_u32_b32 v1, -1, v0", "s_and_saveexec_b64 s[4:5], vcc", "s_mov_b32 s4, exec_hi"):
            with self.subTest(marker=marker), self.assertRaises(rdna2spirv.Unsupported) as cm:
                translate([marker], wave32=True)
            self.assertIn("FSPX wave32 dispatch state contradicts wave64 code", str(cm.exception))

    def test_wave64_state_against_a_wave32_marker_is_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(["s_and_saveexec_b32 s4, vcc_lo"], wave32=False)
        self.assertIn("FSPX wave64 dispatch state contradicts wave32 code", str(cm.exception))

    def test_no_state_keeps_the_code_heuristic(self):
        """A kind 0 too short for word 4 gives no `wave32`: today's heuristic, where vcc in a _b64 op means wave64."""
        self.assertTrue(translate(VCC_AS_DATA, wave32=None)[1])
        self.assertFalse(translate(["v_mov_b32_e32 v1, v0"], wave32=None)[1])

    def test_an_fssm_sidecar_keeps_the_fssm_rules(self):
        """The game's shaders have an FSSM header: its flag decides, and its stricter check (any _b64 op on vcc counts
        as wave64) stays, whatever an fspx/<name>.json beside it says."""
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(VCC_AS_DATA, wave32=True, fssm=fssm_header(wave32=True))
        self.assertIn("FSSM wave32 header contradicts wave64 code", str(cm.exception))
        self.assertTrue(translate(["v_mov_b32_e32 v1, v0"], wave32=True, fssm=fssm_header(wave32=False))[1])


# v0 = local id x. Lane 0 stores readlane(3 * lane + 1, 40): the lane select is S1[4:0] in wave32 (lane 8, 25) and
# S1[5:0] in wave64 (lane 40, 121). No instruction here is a wave marker, so only the dispatch state decides.
READLANE_40 = [
    "v_mad_u32_u24 v2, v0, 3, 1",
    "v_readlane_b32 s6, v2, 40",
    "v_cmpx_eq_u32_e32 0, v0",
    "v_mov_b32_e32 v20, s6",
    "v_mov_b32_e32 v21, 0",
    "buffer_store_dword v20, v21, s[0:3], 0 offen",
]


def run(asm, subgroup_size):
    memory = si.run_workgroup(
        asm, 64, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
        push_constants=VSHARP + [0] * 32, memory={})
    return memory.get(0)


class TheChoiceMatters(unittest.TestCase):
    def test_readlane_40_reads_lane_8_in_wave32_and_lane_40_in_wave64(self):
        asm32, wave64 = translate(READLANE_40, wave32=True)
        self.assertFalse(wave64)
        self.assertEqual(run(asm32, 32), 3 * 8 + 1)
        asm64, wave64 = translate(READLANE_40, wave32=False)
        self.assertTrue(wave64)
        self.assertEqual(run(asm64, 64), 3 * 40 + 1)


if __name__ == "__main__":
    unittest.main()
