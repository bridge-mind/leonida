"""Semantic interpreter tests for RDNA2 v_frexp_mant_f32, v_frexp_exp_i32_f32 and v_ldexp_f32 (SHADER-89).

v_frexp_mant_f32 returns the mantissa of S0 in [0.5, 1) with its sign, v_frexp_exp_i32_f32 the matching exponent
(S0 = mant * 2^exp), and v_ldexp_f32 computes S0 * 2^S1 with S1 a signed integer (RDNA2 ISA). Zero gives a signed zero
mantissa and exponent 0; infinities and NaNs pass through the mantissa and ldexp, and give exponent 0. Each case
translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and
compares the exported word with a plain-Python reference (math.frexp / math.ldexp on the double value of the input).

Not covered because the translator does not model them (docs/design.md): f32 denormal inputs and denormal results.
ldexp results below the smallest normal are expected to be signed zero here, which is what the translator
produces; this was not measured on hardware.
"""
import math
import os
import random
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
INF, NINF, QNAN = 0x7F800000, 0xFF800000, 0x7FC00000
MIN_NORMAL = 0x00800000
MAX_FINITE = 0x7F7FFFFF


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_normal(u: int) -> bool:
    return 0 < ((u >> 23) & 0xFF) < 255


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_frexp_ldexp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_frexp_ldexp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int = 0) -> int:
    """Seed v1 and v2, clear v0, run ``line``, return v0."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1 & 0xFFFFFFFF:#x}", f"v_mov_b32 v2, {v2 & 0xFFFFFFFF:#x}", "v_mov_b32 v0, 0", line]
    asm = "; test_frexp_ldexp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def ref_mant(u: int) -> int:
    if not is_normal(u):
        return (u & 0x80000000) if (u >> 23) & 0xFF == 0 else u
    return f2u(math.frexp(u2f(u))[0])


def ref_exp(u: int) -> int:
    return math.frexp(u2f(u))[1] & 0xFFFFFFFF if is_normal(u) else 0


def ref_ldexp(u: int, n: int) -> int:
    e = (u >> 23) & 0xFF
    sgn = u & 0x80000000
    if e == 255:
        return u
    if e == 0:
        return sgn
    n &= 0xFFFFFFFF
    n = n - (1 << 32) if n & 0x80000000 else n
    n = max(-512, min(512, n))
    e2 = e + n
    if e2 >= 255:
        return sgn | INF
    if e2 <= 0:
        return sgn
    return f2u(math.ldexp(u2f(u), n))


def sample_normals(seed: int, count: int):
    rng = random.Random(seed)
    out = [f2u(1.0), f2u(-1.0), f2u(0.5), f2u(0.75), f2u(8.0), f2u(-0.1), f2u(3.14159), MIN_NORMAL, MAX_FINITE,
           MAX_FINITE | 0x80000000, MIN_NORMAL | 0x80000000]
    while len(out) < count:
        u = rng.getrandbits(32)
        if is_normal(u):
            out.append(u)
    return out


class FrexpTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_mant(f2u(8.0)), f2u(0.5))
        self.assertEqual(ref_exp(f2u(8.0)), 4)
        self.assertEqual(ref_exp(f2u(0.75)), 0)
        self.assertEqual(ref_exp(f2u(-0.1)), (-3) & 0xFFFFFFFF)

    def test_known_values(self):
        for x, m, e in ((1.0, 0.5, 1), (-1.0, -0.5, 1), (0.5, 0.5, 0), (3.0, 0.75, 2), (-0.1875, -0.75, -2)):
            with self.subTest(x=x):
                self.assertEqual(run("v_frexp_mant_f32_e32 v0, v1", f2u(x)), f2u(m))
                self.assertEqual(run("v_frexp_exp_i32_f32_e32 v0, v1", f2u(x)), e & 0xFFFFFFFF)

    def test_matches_python_on_normals(self):
        for u in sample_normals(0x89, 60):
            with self.subTest(x=hex(u)):
                self.assertEqual(run("v_frexp_mant_f32_e32 v0, v1", u), ref_mant(u))
                self.assertEqual(run("v_frexp_exp_i32_f32_e32 v0, v1", u), ref_exp(u))

    def test_mantissa_times_two_to_the_exponent_is_the_input(self):
        for u in sample_normals(0x90, 20):
            m, e = u2f(run("v_frexp_mant_f32_e32 v0, v1", u)), run("v_frexp_exp_i32_f32_e32 v0, v1", u)
            e = e - (1 << 32) if e & 0x80000000 else e
            self.assertEqual(f2u(math.ldexp(m, e)), u)
            self.assertTrue(0.5 <= abs(m) < 1.0)

    def test_zero_and_specials(self):
        for u in (0x00000000, 0x80000000):
            self.assertEqual(run("v_frexp_mant_f32_e32 v0, v1", u), u)       # signed zero
            self.assertEqual(run("v_frexp_exp_i32_f32_e32 v0, v1", u), 0)
        for u in (INF, NINF, QNAN, QNAN | 0x80000000):
            self.assertEqual(run("v_frexp_mant_f32_e32 v0, v1", u), u)       # passed through
            self.assertEqual(run("v_frexp_exp_i32_f32_e32 v0, v1", u), 0)

    def test_source_modifiers_and_output_modifiers(self):
        one = f2u(1.0)
        self.assertEqual(run("v_frexp_mant_f32_e64 v0, -v1", one), f2u(-0.5))
        self.assertEqual(run("v_frexp_mant_f32_e64 v0, |v1|", f2u(-1.0)), f2u(0.5))
        self.assertEqual(run("v_frexp_mant_f32_e64 v0, v1 mul:2", one), f2u(1.0))
        self.assertEqual(run("v_frexp_mant_f32_e64 v0, v1 mul:4 clamp", one), f2u(1.0))
        self.assertEqual(run("v_frexp_exp_i32_f32_e64 v0, -v1", f2u(8.0)), 4)


class LdexpTests(unittest.TestCase):
    def test_known_values(self):
        for x, n, want in ((1.0, 3, 8.0), (1.0, -1, 0.5), (-3.0, 2, -12.0), (0.75, 0, 0.75), (6.0, -2, 1.5)):
            with self.subTest(x=x, n=n):
                self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(x), n), f2u(want))

    def test_matches_python_on_normals(self):
        rng = random.Random(0x91)
        for u in sample_normals(0x92, 40):
            for n in (rng.randrange(-40, 40), rng.randrange(-300, 300)):
                with self.subTest(x=hex(u), n=n):
                    self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", u, n), ref_ldexp(u, n))

    def test_overflow_and_underflow_saturate(self):
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(1.0), 127), f2u(2.0 ** 127))
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(1.0), 128), INF)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(-1.0), 128), NINF)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(1.0), 0x7FFFFFFF), INF)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(1.0), -126), MIN_NORMAL)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(1.0), -127), 0)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(-1.0), -127), 0x80000000)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", f2u(-1.0), 0x80000000), 0x80000000)   # INT_MIN
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", MAX_FINITE, 1), INF)
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", MIN_NORMAL, -1), 0)

    def test_zero_and_specials_ignore_the_exponent(self):
        for u in (0x00000000, 0x80000000):
            for n in (0, 5, -5, 0x7FFFFFFF, 0x80000000):
                self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", u, n), u)
        for u in (INF, NINF, QNAN):
            for n in (0, 5, -200, 0x7FFFFFFF):
                self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2", u, n), u)

    def test_source_and_output_modifiers(self):
        self.assertEqual(run("v_ldexp_f32_e64 v0, -v1, v2", f2u(1.0), 1), f2u(-2.0))
        self.assertEqual(run("v_ldexp_f32_e64 v0, |v1|, v2", f2u(-1.0), 1), f2u(2.0))
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2 mul:2", f2u(1.0), 1), f2u(4.0))
        self.assertEqual(run("v_ldexp_f32_e64 v0, v1, v2 clamp", f2u(1.0), 3), f2u(1.0))


if __name__ == "__main__":
    unittest.main()
