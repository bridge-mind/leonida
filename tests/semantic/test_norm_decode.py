"""Typed-buffer UNORM/SNORM decode (Gen.norm_bits in tools/rdna2spirv.py) against exact values.

The RX 6700 returns the correctly rounded f32 of x / (2^n - 1) for UNORM and max(x / (2^(n-1) - 1), -1) for SNORM
(measured natively, #374). OpFDiv may be 2.5 ULP off and the AMD Windows and NVIDIA drivers return the reciprocal-
multiply result for it (#599), so the translation builds the quotient with integer ops. The expected values here come
from Fraction arithmetic; nothing in this file imports the translator's rounding.
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_buffer_load_format import BASE, FMT_U32, load_format  # noqa: E402


def f32_rne(x: Fraction) -> int:
    """Bits of the binary32 nearest to x (ties to even); |x| <= 1 and x is 0 or at least 2^-126 here."""
    if x == 0:
        return 0
    sign = 0x80000000 if x < 0 else 0
    a, e = abs(x), 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    m = a * (1 << 23)
    q, r = divmod(m.numerator, m.denominator)
    if 2 * r > m.denominator or (2 * r == m.denominator and q & 1):
        q += 1
    if q == 1 << 24:
        q >>= 1
        e += 1
    return sign | (e + 127) << 23 | (q - (1 << 23))


def norm_bits_on_interp(m: int, n: int) -> int:
    """Run the SPIR-V that Gen.norm_bits emits for the constant m in the mini interpreter."""
    gen = rdna2spirv.Gen("ps")
    res = gen.norm_bits(gen.cu(m), n)
    consts = "".join(f"{name} = OpConstant %uint {v}\n" for v, name in gen.consts.items())
    consts += "".join(f"{name} = OpConstant %ulong {v}\n" for v, name in gen.lconsts.items())
    text = ("OpCapability Shader\nOpCapability Int64\n%glsl = OpExtInstImport \"GLSL.std.450\"\n"
            "OpMemoryModel Logical GLSL450\nOpEntryPoint Fragment %main \"main\" %out\n"
            "%void = OpTypeVoid\n%fn = OpTypeFunction %void\n%bool = OpTypeBool\n%uint = OpTypeInt 32 0\n"
            "%int = OpTypeInt 32 1\n%ulong = OpTypeInt 64 0\n%_ptr_Output_uint = OpTypePointer Output %uint\n"
            "%out = OpVariable %_ptr_Output_uint Output\n" + consts
            + "%main = OpFunction %void None %fn\n%entry = OpLabel\n"
            + "".join(line.strip() + "\n" for line in gen.body)
            + f"OpStore %out {res}\nOpReturn\nOpFunctionEnd\n")
    return si.run_fragment(text)["%out"][0]


def recip_mul(m: int, n: int) -> int:
    """The decode before #599 (and what OpFDiv returns on those drivers): f32(m) * f32(1/(2^n-1)), rounded once."""
    c = Fraction(f32_rne_float(Fraction(1, (1 << n) - 1)))
    return f32_rne(m * c)


def f32_rne_float(x: Fraction) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<I", f32_rne(x)))[0]


class NormBitsExact(unittest.TestCase):
    def check(self, n, values):
        d = (1 << n) - 1
        for m in values:
            with self.subTest(n=n, m=m):
                self.assertEqual(norm_bits_on_interp(m, n), f32_rne(Fraction(m, d)))

    def test_every_value_up_to_8_bits(self):
        for n in range(1, 9):
            self.check(n, range(1 << n))

    def test_10_and_16_bits(self):
        rng = random.Random(599)
        for n in (10, 15, 16):
            d = (1 << n) - 1
            edge = [0, 1, 2, 3, d // 2, d // 2 + 1, d - 2, d - 1, d]
            # the inputs where the reciprocal multiply is off are the ones that matter most
            off = [m for m in range(1, d) if recip_mul(m, n) != f32_rne(Fraction(m, d))][:64]
            self.check(n, edge + off + [rng.randrange(d + 1) for _ in range(96)])

    def test_reciprocal_multiply_really_differs(self):
        # negative control: the values above include inputs the old decode got wrong (126 of 256 for 8 bits)
        self.assertEqual(sum(recip_mul(m, 8) != f32_rne(Fraction(m, 255)) for m in range(256)), 126)


class DecodeThroughBufferLoad(unittest.TestCase):
    """End to end through buffer_load_format (8_8_8_8_UNORM/SNORM and the formats this change adds)."""

    def test_unorm8x4_and_snorm8x4(self):
        mem = {BASE: 0x80FF7F03}          # bytes 0x03, 0x7F, 0xFF, 0x80
        self.assertEqual(load_format(56, 0, memory=mem),
                         tuple(f32_rne(Fraction(b, 255)) for b in (0x03, 0x7F, 0xFF, 0x80)))
        self.assertEqual(load_format(57, 0, memory=mem),
                         (f32_rne(Fraction(3, 127)), 0x3F800000, f32_rne(Fraction(-1, 127)), 0xBF800000))

    def test_fixed_snorm8x4_uses_the_instruction_format(self):
        # A U32 descriptor cannot supply the expected fractional values or the signed minimum.
        words = load_format(FMT_U32, 0, memory={BASE: 0x80FF7F03}, op="tbuffer_load_format",
                            mods=" format:[BUF_FMT_8_8_8_8_SNORM]")
        self.assertEqual(words, (f32_rne(Fraction(3, 127)), 0x3F800000,
                                 f32_rne(Fraction(-1, 127)), 0xBF800000))

    def test_added_snorm_formats(self):
        mem = {BASE: 0x8001FF81}
        # 8_SNORM (2): byte 0x81 = -127 -> -1.0
        self.assertEqual(load_format(2, 0, 1, memory=mem)[0], 0xBF800000)
        # 8_8_SNORM (15): 0x81, 0xFF -> -1.0, -1/127
        self.assertEqual(load_format(15, 0, 2, memory=mem)[:2], (0xBF800000, f32_rne(Fraction(-1, 127))))
        # 16_SNORM (8): 0xFF81 = -127 -> -127/32767; high half 0x8001 = -32767 -> -1.0
        self.assertEqual(load_format(8, 0, 1, memory=mem)[0], f32_rne(Fraction(-127, 32767)))
        self.assertEqual(load_format(8, 2, 1, memory=mem)[0], 0xBF800000)
        # 16_SNORM: the most negative code, -32768, is -1.0 too
        self.assertEqual(load_format(8, 0, 1, memory={BASE: 0x8000})[0], 0xBF800000)


if __name__ == "__main__":
    unittest.main()
