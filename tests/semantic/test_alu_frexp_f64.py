"""Semantic interpreter and spirv-val tests for RDNA2 v_frexp_mant_f64 and v_frexp_exp_i32_f64.

v_frexp_mant_f64 returns the binary64 mantissa of S0 in [0.5, 1) with its sign, and v_frexp_exp_i32_f64 returns the
matching signed 32-bit exponent (S0 = mant * 2^exp). Under default denorm16_64=3 (measured natively on gfx1036,
wave32 and wave64), non-zero subnormals normalize into [0.5, 1) with exponent -1073..-1022; signed zero gives signed
zero and exponent 0; infinities and NaNs pass through v_frexp_mant_f64 unchanged and give exponent 0.
"""
import math
import os
import random
import re
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
INF = 0x7FF0000000000000
NINF = 0xFFF0000000000000
QNAN = 0x7FF8000000000001
NQNAN = 0xFFF8000000000042
SNAN = 0x7FF0000000000001
MIN_NORMAL = 0x0010000000000000
MAX_FINITE = 0x7FEFFFFFFFFFFFFF
SUBNORMAL = 0x0000000000000001
MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF


def d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def u2d(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]


def is_normal(u: int) -> bool:
    return 0 < ((u >> 52) & 0x7FF) < 2047


def is_finite_nonzero(u: int) -> bool:
    exp_f = (u >> 52) & 0x7FF
    frac = u & ((1 << 52) - 1)
    return exp_f < 2047 and (exp_f > 0 or frac != 0)


def ref_mant(u: int) -> int:
    if not is_finite_nonzero(u):
        return u
    return d2u(math.frexp(u2d(u))[0])


def ref_exp(u: int) -> int:
    return math.frexp(u2d(u))[1] & 0xFFFFFFFF if is_finite_nonzero(u) else 0


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_frexp_f64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_frexp_f64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_frexp_f64\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
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


def run_mant(line: str, u: int) -> int:
    lo, hi = u & 0xFFFFFFFF, (u >> 32) & 0xFFFFFFFF
    spv = translate_lines([
        f"v_mov_b32 v2, {lo:#x}",
        f"v_mov_b32 v3, {hi:#x}",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0",
        line,
    ])
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[1] << 32) | out[0]


def run_exp(line: str, u: int) -> int:
    lo, hi = u & 0xFFFFFFFF, (u >> 32) & 0xFFFFFFFF
    spv = translate_lines([
        f"v_mov_b32 v2, {lo:#x}",
        f"v_mov_b32 v3, {hi:#x}",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0",
        line,
    ])
    return si.run_fragment(spv)["%out_mrt0"][0]


def sample_normals(seed: int, count: int):
    rng = random.Random(seed)
    out = [
        d2u(1.0), d2u(-1.0), d2u(0.5), d2u(0.75), d2u(8.0), d2u(-0.1), d2u(math.pi),
        MIN_NORMAL, MAX_FINITE, MAX_FINITE | (1 << 63), MIN_NORMAL | (1 << 63),
    ]
    while len(out) < count:
        u = rng.getrandbits(64)
        if is_normal(u):
            out.append(u)
    return out


class FrexpF64Tests(unittest.TestCase):
    def test_known_values(self):
        cases = (
            (1.0, 0.5, 1),
            (-1.0, -0.5, 1),
            (0.5, 0.5, 0),
            (3.0, 0.75, 2),
            (-0.1875, -0.75, -2),
            (1024.0, 0.5, 11),
            (2.0 ** -1022, 0.5, -1021),
        )
        for x, m, e in cases:
            with self.subTest(x=x):
                self.assertEqual(run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", d2u(x)), d2u(m))
                self.assertEqual(run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", d2u(x)), e & 0xFFFFFFFF)

    def test_matches_python_on_normals(self):
        for u in sample_normals(0xF64, 60):
            with self.subTest(x=hex(u)):
                self.assertEqual(run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", u), ref_mant(u))
                self.assertEqual(run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", u), ref_exp(u))

    def test_reconstruction_identity(self):
        for u in sample_normals(0xF65, 20):
            m = u2d(run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", u))
            e = run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", u)
            e = e - (1 << 32) if e & 0x80000000 else e
            self.assertEqual(d2u(math.ldexp(m, e)), u)
            self.assertTrue(0.5 <= abs(m) < 1.0)

    def test_zero_subnormals_and_specials(self):
        for u in (0x0000000000000000, 0x8000000000000000):
            with self.subTest(zero=hex(u)):
                self.assertEqual(run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", u), u)
                self.assertEqual(run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", u), 0)
        for u in (INF, NINF, QNAN, NQNAN, SNAN):
            with self.subTest(special=hex(u)):
                self.assertEqual(run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", u), u)
                self.assertEqual(run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", u), 0)

    def test_subnormals_match_gfx1036(self):
        # All 52 single-bit subnormal positions plus multi-bit patterns crossing bits 19/20/21/31/32/51.
        # Every case verified natively on gfx1036 (wave32 and wave64 identical, default denorm16_64=3).
        subnormals = [1 << b for b in range(52)] + [
            0x00000000FFFFFFFF,
            0x000FFFFF00000000,
            MAX_SUBNORMAL,
            0x0000000180000001,
            0x0000000000180005,
            0x0000000080000001,
            0x0000000100000001,
            0x0008000000000001,
            SUBNORMAL | (1 << 63),
            MAX_SUBNORMAL | (1 << 63),
            0x8000000180000001,
        ]
        for u in subnormals:
            with self.subTest(sub=hex(u)):
                got_m = run_mant("v_frexp_mant_f64_e32 v[0:1], v[2:3]", u)
                got_e = run_exp("v_frexp_exp_i32_f64_e32 v0, v[2:3]", u)
                self.assertEqual(got_m, ref_mant(u))
                self.assertEqual(got_e, ref_exp(u))
                e_signed = got_e - (1 << 32) if got_e & 0x80000000 else got_e
                self.assertEqual(d2u(math.ldexp(u2d(got_m), e_signed)), u)
                self.assertTrue(0.5 <= abs(u2d(got_m)) < 1.0)
        # VOP3 abs/neg on subnormals.
        self.assertEqual(
            run_mant("v_frexp_mant_f64_e64 v[0:1], |v[2:3]|", SUBNORMAL | (1 << 63)),
            0x3FE0000000000000,
        )
        self.assertEqual(
            run_mant("v_frexp_mant_f64_e64 v[0:1], -v[2:3]", SUBNORMAL),
            0xBFE0000000000000,
        )
        self.assertEqual(
            run_exp("v_frexp_exp_i32_f64_e64 v0, |v[2:3]|", SUBNORMAL | (1 << 63)),
            (-1073) & 0xFFFFFFFF,
        )

    def test_source_modifiers_sgpr_inline_and_aliasing(self):
        x = d2u(3.0)
        nx = d2u(-3.0)
        self.assertEqual(run_mant("v_frexp_mant_f64_e64 v[0:1], |v[2:3]|", nx), d2u(0.75))
        self.assertEqual(run_mant("v_frexp_mant_f64_e64 v[0:1], -v[2:3]", x), d2u(-0.75))
        self.assertEqual(run_mant("v_frexp_mant_f64_e64 v[0:1], -|v[2:3]|", nx), d2u(-0.75))
        self.assertEqual(run_exp("v_frexp_exp_i32_f64_e64 v0, -|v[2:3]|", nx), 2)

        # SGPR pair source
        spv_sgpr = translate_lines([
            f"s_mov_b32 s4, {x & 0xFFFFFFFF:#x}",
            f"s_mov_b32 s5, {(x >> 32) & 0xFFFFFFFF:#x}",
            "v_frexp_mant_f64_e32 v[0:1], s[4:5]",
            "v_frexp_exp_i32_f64_e32 v2, s[4:5]",
        ])
        out = si.run_fragment(spv_sgpr)["%out_mrt0"]
        self.assertEqual((out[1] << 32) | out[0], d2u(0.75))
        self.assertEqual(out[2], 2)

        # Inline float constant and integer 0
        spv_imm = translate_lines([
            "v_frexp_mant_f64_e32 v[0:1], -4.0",
            "v_frexp_exp_i32_f64_e32 v2, -4.0",
            "v_frexp_exp_i32_f64_e32 v3, 0",
        ])
        out_imm = si.run_fragment(spv_imm)["%out_mrt0"]
        self.assertEqual((out_imm[1] << 32) | out_imm[0], d2u(-0.5))
        self.assertEqual(out_imm[2], 3)
        self.assertEqual(out_imm[3], 0)

        # Destination aliasing source (partial overlap v[1:2] <- v[0:1])
        spv_alias = translate_lines([
            f"v_mov_b32 v0, {x & 0xFFFFFFFF:#x}",
            f"v_mov_b32 v1, {(x >> 32) & 0xFFFFFFFF:#x}",
            "v_frexp_mant_f64_e32 v[1:2], v[0:1]",
        ])
        out_alias = si.run_fragment(spv_alias)["%out_mrt0"]
        self.assertEqual((out_alias[2] << 32) | out_alias[1], d2u(0.75))

    def test_plain_needs_no_float64_and_validates(self):
        spv = translate_lines([
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40080000",
            "v_frexp_mant_f64_e32 v[0:1], v[2:3]",
            "v_frexp_exp_i32_f64_e32 v2, v[2:3]",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        validate_spirv(spv)

    def test_mant_output_modifiers_and_validation(self):
        # Measured on gfx1036 through tools/hw_oracle (wave32 and wave64 identical, tools/f64_omod_probe.py):
        # a binary64 result ignores mul:2/mul:4/div:2 while denorm_mode_16_64 = 3, the MODE the translator
        # assumes, so those forms keep the integer-only lowering. clamp applies in every MODE. See
        # docs/design.md, "v_frexp_mant_f64, v_frexp_exp_i32_f64".
        for mods, clamp_requested in (
            ("mul:2", False),
            ("mul:4", False),
            ("div:2", False),
            ("clamp", True),
            ("clamp div:2", True),
        ):
            with self.subTest(mods=mods):
                spv = translate_lines([
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0x40080000",
                    f"v_frexp_mant_f64_e64 v[0:1], v[2:3] {mods}",
                ])
                self.assertNotRegex(spv, r"OpFMul %double")
                if clamp_requested:
                    self.assertIn("OpCapability Float64", spv)
                    self.assertIn("FClamp", spv)
                else:
                    self.assertNotIn("OpCapability Float64", spv)
                    self.assertNotIn("FClamp", spv)
                validate_spirv(spv)

        for mods in ("mul:2", "mul:4", "div:2", "mul:2 div:2"):
            with self.subTest(ignored=mods):
                # 3.0 has mantissa 0.75; the scaling would make it 1.5, 3.0 or 0.375.
                self.assertEqual(run_mant(f"v_frexp_mant_f64_e64 v[0:1], v[2:3] {mods}", d2u(3.0)), d2u(0.75))
                self.assertEqual(run_mant(f"v_frexp_mant_f64_e64 v[0:1], v[2:3] {mods}", d2u(-1.75)), d2u(-0.875))

    def test_rejects_invalid_forms(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_frexp_exp_i32_f64_e64 v0, v[2:3] clamp"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_frexp_exp_i32_f64_e64 v0, v[2:3] mul:2"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_frexp_mant_f64_e32 v0, v[2:3]"])


if __name__ == "__main__":
    unittest.main()
