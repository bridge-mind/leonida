"""Semantic interpreter and spirv-val tests for RDNA2 v_cvt_norm_i16_f16 and v_cvt_norm_u16_f16.

v_cvt_norm_i16_f16 converts a binary16 input to snorm16: round_even(clamp(S0, -1, 1) * 32767), NaN -> 0.
v_cvt_norm_u16_f16 converts a binary16 input to unorm16: round_even(clamp(S0, 0, 1) * 65535), NaN -> 0.
Both preserve the unwritten 16-bit half of the destination and support VOP3 neg/abs and op_sel:[src,dst].
"""
from fractions import Fraction
import os
import random
import shutil
import struct
import subprocess
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SEED_DST = 0xA55A1234


def f16_to_float(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def float_to_f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def ref_norm16(h: int, snorm: bool) -> int:
    x = f16_to_float(h)
    if x != x:
        return 0
    lo = -1.0 if snorm else 0.0
    c = min(max(x, lo), 1.0)
    scale = 32767 if snorm else 65535
    return round(Fraction(c) * scale) & 0xFFFF


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_norm_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_norm_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    # Export v0 split into two safe 16-bit halves ORed into 0x40000000 so float export never alters NaN/denorm bits.
    full = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = "; test_cvt_norm_f16\n" + "".join(f"\t{ln}\n" for ln in full) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines) -> int:
    spv = translate_lines(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


class CvtNormF16Tests(unittest.TestCase):
    def test_edges_and_single_rounding_ties(self):
        # Includes the 4 f16 encodings near +-0.5 (0x3801, 0x37ff, 0xb801, 0xb7ff) where f32(x * scale) double-rounds.
        edges = (
            0x0000, 0x8000,                   # +0, -0
            0x0001, 0x8001, 0x03FF, 0x83FF,   # subnormals
            0x0400, 0x8400,                   # min normals
            0x3400, 0xB400,                   # +-0.25
            0x37FE, 0x37FF, 0x3800, 0x3801, 0x3802,  # around +0.5
            0xB7FE, 0xB7FF, 0xB800, 0xB801, 0xB802,  # around -0.5
            0x3BFF, 0x3C00, 0x3C01,           # around +1.0
            0xBBFF, 0xBC00, 0xBC01,           # around -1.0
            0x4000, 0xC000,                   # +-2.0 (clamped)
            0x7C00, 0xFC00,                   # +-inf
            0x7E00, 0xFE00, 0x7C01, 0xFC01,   # qNaN / sNaN
        )
        for op, snorm in (("v_cvt_norm_i16_f16", True), ("v_cvt_norm_u16_f16", False)):
            for h in edges:
                with self.subTest(op=op, h=hex(h)):
                    got = run_lines([
                        f"v_mov_b32 v1, {(0xDEAD0000 | h):#x}",
                        f"v_mov_b32 v0, {SEED_DST:#x}",
                        f"{op}_e32 v0, v1",
                    ])
                    self.assertEqual(got, (SEED_DST & 0xFFFF0000) | ref_norm16(h, snorm))

    def test_seeded_sweep_matches_exact_fraction(self):
        rng = random.Random(0xF16)
        samples = [rng.randrange(65536) for _ in range(60)]
        for op, snorm in (("v_cvt_norm_i16_f16", True), ("v_cvt_norm_u16_f16", False)):
            for h in samples:
                with self.subTest(op=op, h=hex(h)):
                    got = run_lines([
                        f"v_mov_b32 v1, {(0xBEEF0000 | h):#x}",
                        f"v_mov_b32 v0, {SEED_DST:#x}",
                        f"{op}_e32 v0, v1",
                    ])
                    self.assertEqual(got, (SEED_DST & 0xFFFF0000) | ref_norm16(h, snorm))

    def test_opsel_and_source_modifiers(self):
        lo_h = float_to_f16(-0.5)
        hi_h = float_to_f16(0.75)
        packed = (hi_h << 16) | lo_h

        # op_sel:[1,0]: read high half (0.75), write low half, preserve high half of v0
        got = run_lines([
            f"v_mov_b32 v1, {packed:#x}",
            f"v_mov_b32 v0, {SEED_DST:#x}",
            "v_cvt_norm_i16_f16_e64 v0, v1 op_sel:[1,0]",
        ])
        self.assertEqual(got, (SEED_DST & 0xFFFF0000) | ref_norm16(hi_h, True))

        # op_sel:[0,1]: read low half (-0.5), apply abs -> +0.5, write high half, preserve low half of v0
        got2 = run_lines([
            f"v_mov_b32 v1, {packed:#x}",
            f"v_mov_b32 v0, {SEED_DST:#x}",
            "v_cvt_norm_u16_f16_e64 v0, |v1| op_sel:[0,1]",
        ])
        self.assertEqual(got2, (ref_norm16(float_to_f16(0.5), False) << 16) | (SEED_DST & 0xFFFF))

        # op_sel:[1,1] with neg: read high half (0.75), negate -> -0.75, write high half
        got3 = run_lines([
            f"v_mov_b32 v1, {packed:#x}",
            f"v_mov_b32 v0, {SEED_DST:#x}",
            "v_cvt_norm_i16_f16_e64 v0, -v1 op_sel:[1,1]",
        ])
        self.assertEqual(got3, (ref_norm16(float_to_f16(-0.75), True) << 16) | (SEED_DST & 0xFFFF))

    def test_sdwa_word1_high_half_preserve(self):
        # GFX10/LLVM rejects op_sel on v_cvt_norm_*_f16; high-half access uses SDWA WORD_1.
        hi_h = float_to_f16(0.75)
        packed = (hi_h << 16) | float_to_f16(-0.25)
        for op, snorm in (("v_cvt_norm_i16_f16", True), ("v_cvt_norm_u16_f16", False)):
            with self.subTest(op=op):
                got = run_lines([
                    f"v_mov_b32 v1, {packed:#x}",
                    f"v_mov_b32 v0, {SEED_DST:#x}",
                    f"{op}_sdwa v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
                ])
                self.assertEqual(got, (ref_norm16(hi_h, snorm) << 16) | (SEED_DST & 0xFFFF))

    def test_sgpr_inline_and_spirv_val(self):
        spv = translate_lines([
            f"s_mov_b32 s4, {float_to_f16(0.5):#x}",
            f"v_mov_b32 v0, {SEED_DST:#x}",
            "v_cvt_norm_i16_f16_e32 v0, s4",
            "v_cvt_norm_u16_f16_e64 v0, 0.5 op_sel:[0,1]",
        ])
        validate_spirv(spv)
        lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
        got = (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)
        want = (ref_norm16(float_to_f16(0.5), False) << 16) | ref_norm16(float_to_f16(0.5), True)
        self.assertEqual(got, want)

    def test_output_modifiers_match_plain_and_reject_invalid_forms(self):
        h = float_to_f16(0.5)
        for op, snorm in (("v_cvt_norm_i16_f16", True), ("v_cvt_norm_u16_f16", False)):
            want = (SEED_DST & 0xFFFF0000) | ref_norm16(h, snorm)
            for mod in ("clamp", "mul:2", "mul:4", "div:2"):
                with self.subTest(op=op, mod=mod):
                    got = run_lines([
                        f"v_mov_b32 v1, {h:#x}",
                        f"v_mov_b32 v0, {SEED_DST:#x}",
                        f"{op}_e64 v0, v1 {mod}",
                    ])
                    self.assertEqual(got, want)
            with self.subTest(op=op, form="sdwa_clamp"):
                got_sdwa = run_lines([
                    f"v_mov_b32 v1, {(h << 16):#x}",
                    f"v_mov_b32 v0, {SEED_DST:#x}",
                    f"{op}_sdwa v0, v1 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
                ])
                self.assertEqual(got_sdwa, (ref_norm16(h, snorm) << 16) | (SEED_DST & 0xFFFF))
        for bad_line in (
            "v_cvt_norm_i16_f16_e64 v0, v1 mul:8",
            "v_cvt_norm_u16_f16_sdwa v0, v1 mul:2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
            "v_cvt_norm_i16_f16_e32 v0, v1 clamp",
        ):
            with self.subTest(line=bad_line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines([bad_line])


if __name__ == "__main__":
    unittest.main()
