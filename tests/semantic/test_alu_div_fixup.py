"""Semantic interpreter and spirv-val tests for RDNA2 v_div_fixup_f16 and v_div_fixup_f64.

v_div_fixup_{f16,f64} D, S0 (quot), S1 (denom), S2 (num) resolves IEEE-754 division special cases:
  - sign_out = sign(S1) ^ sign(S2)
  - NaN(S2) -> quiet(S2); NaN(S1) -> quiet(S1)
  - (0/0) or (inf/inf) -> canonical negative qNaN (0xFE00 / 0xFFF80000_00000000)
  - (x/inf) or (0/y) -> copysign(0, sign_out)
  - (x/0) or (inf/y) or NaN(S0) -> copysign(inf, sign_out)
  - otherwise copysign(S0, sign_out)
"""
import os
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

H_POS0, H_NEG0 = 0x0000, 0x8000
H_ONE, H_NEG2, H_HALF = 0x3C00, 0xC000, 0x3800
H_INF, H_NINF = 0x7C00, 0xFC00
H_SNAN1, H_QNAN2 = 0x7C05, 0x7E12
H_CANON_NAN = 0xFE00

D_POS0, D_NEG0 = 0x0000000000000000, 0x8000000000000000
D_ONE, D_NEG2, D_THREE = 0x3FF0000000000000, 0xC000000000000000, 0x4008000000000000
D_INF, D_NINF = 0x7FF0000000000000, 0xFFF0000000000000
D_SNAN1, D_QNAN2 = 0x7FF0000000000005, 0x7FF8000000000012
D_CANON_NAN = 0xFFF8000000000000


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_div_fixup.s")
    with open(os.path.join(_TMP_DIR.name, "ps_div_fixup.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines, safe_v0=False):
    if _TMP_DIR is None:
        setUpModule()
    body = list(lines)
    if safe_v0:
        body += [
            "v_and_b32 v8, 0xffff, v0",
            "v_lshrrev_b32 v9, 16, v0",
            "v_or_b32 v8, 0x40000000, v8",
            "v_or_b32 v9, 0x40000000, v9",
        ]
        exp = "\texp mrt0 v8, v9, v8, v8 done vm\n"
    else:
        exp = "\texp mrt0 v0, v1, v2, v3 done vm\n"
    asm = "; test_div_fixup\n" + "".join(f"\t{ln}\n" for ln in body) + exp + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


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


def run_f16(s0: int, s1: int, s2: int, insn: str = "v_div_fixup_f16 v0, v1, v2, v3") -> int:
    spv = translate_lines([
        f"v_mov_b32 v1, {s0 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v2, {s1 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {s2 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v0, {SEED_DST:#x}",
        insn,
    ], safe_v0=True)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_f64(s0: int, s1: int, s2: int, insn: str = "v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7]") -> int:
    spv = translate_lines([
        f"v_mov_b32 v2, {s0 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {(s0 >> 32) & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v4, {s1 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v5, {(s1 >> 32) & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v6, {s2 & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v7, {(s2 >> 32) & 0xFFFFFFFF:#x}",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0",
        insn,
    ])
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[1] << 32) | out[0]


class DivFixupF16Tests(unittest.TestCase):
    def test_all_special_and_normal_cases(self):
        cases = (
            ("normal_pos", H_HALF, H_ONE, H_ONE, H_HALF),
            ("normal_neg_sign_fix", H_HALF, H_NEG2, H_ONE, H_HALF | 0x8000),
            ("nan_s2_priority", H_ONE, H_SNAN1, H_QNAN2, H_QNAN2),
            ("nan_s1_snan_ieee0", H_ONE, H_SNAN1, H_ONE, H_SNAN1),
            ("nan_s2_snan_ieee0", H_ONE, H_ONE, 0x7C01, 0x7C01),
            ("zero_div_zero", H_ONE, H_POS0, H_NEG0, H_CANON_NAN),
            ("inf_div_inf", H_ONE, H_INF, H_NINF, H_CANON_NAN),
            ("finite_div_inf_neg", H_ONE, H_NINF, H_ONE, H_NEG0),
            ("zero_div_finite_pos", H_ONE, H_NEG2, H_NEG0, H_POS0),
            ("finite_div_zero_neg", H_ONE, H_NEG0, H_ONE, H_NINF),
            ("inf_div_finite_pos", H_ONE, H_NEG2, H_NINF, H_INF),
            ("nan_quot_to_pos_inf", 0x7E35, H_ONE, H_ONE, H_INF),
            ("nan_quot_to_neg_inf", 0x7E35, H_NEG2, H_ONE, H_NINF),
            ("f16_no_exp_underflow", H_ONE, 0x7800, 0x0400, H_ONE),       # gfx1036: e2-e1=-29 keeps S0
            ("f16_no_exp_overflow", H_ONE, 0x0400, 0x7800, H_ONE),        # gfx1036: e2-e1=+29 keeps S0
        )
        for name, s0, s1, s2, want in cases:
            with self.subTest(case=name):
                got = run_f16(0xBEEF0000 | s0, 0xCAFE0000 | s1, 0xDEAD0000 | s2)
                self.assertEqual(got, (SEED_DST & 0xFFFF0000) | want)

    def test_opsel_input_and_output_modifiers_and_spirv_val(self):
        # op_sel:[1,0,1,1] + neg on s1: s0=hi(0.5), s1=-lo(-2.0)=+2.0, s2=hi(1.0) -> +0.5 into high half
        got = run_f16(
            H_HALF << 16,
            H_NEG2,
            H_ONE << 16,
            "v_div_fixup_f16 v0, v1, -v2, v3 op_sel:[1,0,1,1]",
        )
        self.assertEqual(got, (H_HALF << 16) | (SEED_DST & 0xFFFF))

        # Output modifier mul:2 clamp on 0.5 -> on RDNA2 omod is ignored for f16 with denorms kept; clamp applies
        spv = translate_lines([
            f"v_mov_b32 v1, {H_HALF:#x}",
            f"v_mov_b32 v2, {H_ONE:#x}",
            f"v_mov_b32 v3, {H_ONE:#x}",
            f"v_mov_b32 v0, {SEED_DST:#x}",
            "v_div_fixup_f16 v0, v1, v2, v3 mul:2 clamp",
        ], safe_v0=True)
        validate_spirv(spv)
        lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
        self.assertEqual((lo & 0xFFFF) | ((hi & 0xFFFF) << 16), (SEED_DST & 0xFFFF0000) | H_HALF)


class DivFixupF64Tests(unittest.TestCase):
    def test_all_special_and_normal_cases(self):
        cases = (
            ("normal_pos", D_THREE, D_ONE, D_ONE, D_THREE),
            ("normal_neg_sign_fix", D_THREE, D_NEG2, D_ONE, D_THREE | (1 << 63)),
            ("nan_s2_priority", D_ONE, D_SNAN1, D_QNAN2, D_QNAN2),
            ("nan_s1_snan_ieee0", D_ONE, D_SNAN1, D_ONE, D_SNAN1),
            ("nan_s2_snan_ieee0", D_ONE, D_ONE, 0x7FF0000000000001, 0x7FF0000000000001),
            ("zero_div_zero", D_ONE, D_POS0, D_NEG0, D_CANON_NAN),
            ("inf_div_inf", D_ONE, D_INF, D_NINF, D_CANON_NAN),
            ("finite_div_inf_neg", D_ONE, D_NINF, D_ONE, D_NEG0),
            ("zero_div_finite_pos", D_ONE, D_NEG2, D_NEG0, D_POS0),
            ("finite_div_zero_neg", D_ONE, D_NEG0, D_ONE, D_NINF),
            ("inf_div_finite_pos", D_ONE, D_NEG2, D_NINF, D_INF),
            ("nan_quot_to_pos_inf", 0xFFF8123456789ABC, D_NEG2 & ~(1 << 63), D_ONE, D_INF),
            ("nan_quot_to_neg_inf", 0x7FF8123456789ABC, D_NEG2, D_ONE, D_NINF),
            ("exp_diff_m1075_keeps_s0", D_ONE, 2046 << 52, 971 << 52, D_ONE),    # gfx1036: e2-e1=-1075 keeps S0
            ("exp_diff_m1076_to_zero", D_ONE, 2046 << 52, 970 << 52, D_POS0),    # gfx1036: e2-e1=-1076 -> +0
            ("exp_underflow_m1122", D_ONE, 1123 << 52, 1 << 52, D_POS0),         # gfx1036: 2^-1022 / 2^100 -> +0
            ("f64_no_exp_overflow", D_ONE, 1 << 52, 1025 << 52, D_ONE),          # gfx1036: e2-e1=+1024 keeps S0
        )
        for name, s0, s1, s2, want in cases:
            with self.subTest(case=name):
                self.assertEqual(run_f64(s0, s1, s2), want)

    def test_modifiers_plain_no_float64_and_omod_float64_validates(self):
        # Plain form: no Float64 needed, supports abs/neg and SGPR/inline sources
        spv_plain = translate_lines([
            f"v_mov_b32 v2, {D_THREE & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v3, {D_THREE >> 32:#x}",
            f"s_mov_b32 s4, {D_NEG2 & 0xFFFFFFFF:#x}",
            f"s_mov_b32 s5, {D_NEG2 >> 32:#x}",
            "v_div_fixup_f64 v[0:1], v[2:3], |s[4:5]|, -1.0",
        ])
        self.assertNotIn("OpCapability Float64", spv_plain)
        validate_spirv(spv_plain)
        out = si.run_fragment(spv_plain)["%out_mrt0"]
        self.assertEqual((out[1] << 32) | out[0], D_THREE | (1 << 63))

        # With output modifiers: enables Float64 + clamp_mods_f64 and passes spirv-val
        spv_mod = translate_lines([
            f"v_mov_b32 v2, {D_THREE & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v3, {D_THREE >> 32:#x}",
            "v_div_fixup_f64 v[0:1], v[2:3], 1.0, 1.0 mul:2 clamp",
        ])
        self.assertIn("OpCapability Float64", spv_mod)
        self.assertIn("FClamp", spv_mod)
        validate_spirv(spv_mod)

    def test_scaling_modifiers_are_ignored_under_the_games_mode(self):
        # Measured by @fiestorra on a gfx1036 iGPU (driver 31.0.24002.92) through tools/hw_oracle,
        # tools/f64_omod_probe.py: on both ops the f16 and the binary64 result ignore mul:/div: while
        # denorm_mode_16_64 = 3, the MODE these rows assume, and scale only with IEEE=0 and
        # denorm_mode_16_64 = 0, while clamp applies in every MODE. See docs/design.md,
        # "v_div_fixup_f16, v_div_fixup_f64".
        two, onep5, three = 0x4000000000000000, 0x3FF8000000000000, 0x4008000000000000
        spv = translate_lines([
            f"v_mov_b32 v2, {two & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v3, {two >> 32:#x}",
            f"v_mov_b32 v4, {onep5 & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v5, {onep5 >> 32:#x}",
            f"v_mov_b32 v6, {three & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v7, {three >> 32:#x}",
            "v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] mul:4 div:2",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotRegex(spv, r"OpFMul %double")
        self.assertNotIn("FClamp", spv)
        validate_spirv(spv)
        # 2.0 is the fixup of (2.0, 1.5, 3.0); mul:4 div:2 would have made it 4.0.
        self.assertEqual(
            run_f64(two, onep5, three, insn="v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] mul:4 div:2"), two)

        # The f16 form keeps the low-half write: no unpack/pack round-trip, no clamp.
        spv16 = translate_lines([
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mov_b32 v3, 0x4200",
            "v_div_fixup_f16 v0, v1, v2, v3 mul:2",
        ])
        self.assertNotRegex(spv16, r"UnpackHalf2x16")
        self.assertNotIn("FClamp", spv16)
        validate_spirv(spv16)
        self.assertEqual(
            run_f16(0x4000, 0x3E00, 0x4200, insn="v_div_fixup_f16 v0, v1, v2, v3 mul:2") & 0xFFFF, 0x4000)

    def test_rejects_non_pair_destination(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_div_fixup_f64 v0, v[2:3], v[4:5], v[6:7]"])


if __name__ == "__main__":
    unittest.main()
