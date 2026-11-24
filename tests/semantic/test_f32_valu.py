"""Value tests for 32-bit float VALU rules found by tools/f32_fuzz.py (FUZZ-3), and for the interpreter operations
that tool needs.

ISA semantics used (AMD RDNA2 ISA reference, VOP1/VOP2/VOP3 chapters; LLVM AMDGPU as the cross-check):
  v_mac_f32             D = S0 * S1 + D with the product rounded before the add: the VOP2 form of v_mad_f32, not
                          of v_fma_f32 (LLVM SIInstrInfo.cpp getNewFMAInst: V_MAC_F32 -> V_MAD_F32_e64,
                          V_FMAC_F32 -> V_FMA_F32_e64).
  v_cvt_f32_ubyte1      D = float(S0[15:8]); the VOP3 form has clamp and omod (LLVM VOPProfileI2F: Ins64 is
  v_cvt_f32_u32           $src0, $clamp, $omod), applied to the float result: omod scales, then clamp to [0, 1].
  v_cvt_f32_f16         D = float(S0.f16); clamp and omod as for any float result (VOP_F32_F16).
  v_rcp_iflag_f32       D = 1 / S0; clamp and omod as above (VOP_F32_F32).

ps_f32_mods (runtime/test/shaders/ps_f32_mods.asm) exports one word per rule; the constants below are written by
hand and recomputed by the plain-Python references in ReferenceTests, which use neither the translator nor the
interpreter.
"""
import math
import struct
import unittest
from fractions import Fraction

import spirv_mini_interp as si
import support

EXPECTED_F32_MODS = (0xC0256A01, 0x3F800000, 0x41E00000, 0x3FC00000)


def f32_value(word):
    return Fraction(struct.unpack("<f", struct.pack("<I", word))[0])


def round_to_f32(value):
    """Exact rational -> binary32 bits, round to nearest even (normal range only, enough for these cases)."""
    sign = 0x80000000 if value < 0 else 0
    a = abs(value)
    e = math.floor(math.log2(a))
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    q = a / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    if n == 1 << 24:
        n, e = n >> 1, e + 1
    return sign | ((e + 127) << 23) | (n & 0x7FFFFF)


def reference_f32_mods():
    # v_mac_f32 v2, v0, v1: round(round(v0 * v1) + v2).
    v0, v1, v2 = 0xBC500C59, 0x435A2099, 0x3E3DB015
    product = round_to_f32(f32_value(v0) * f32_value(v1))
    mac = round_to_f32(f32_value(product) + f32_value(v2))
    # v_cvt_f32_ubyte1 of 0x12347856 = 120.0, clamped to 1.0.
    ubyte = min(Fraction((0x12347856 >> 8) & 0xFF), Fraction(1))
    # v_cvt_f32_u32 of 7, mul:4.
    u32 = Fraction(7) * 4
    # v_cvt_f32_f16 of 0x3c00 (1.0), div:2; v_rcp_iflag_f32 of 4.0, mul:4; summed.
    total = Fraction(1) / 2 + Fraction(1, 4) * 4
    return (mac, round_to_f32(ubyte), round_to_f32(u32), round_to_f32(total))


def reference_fused():
    """What a fused multiply-add gives for the v_mac_f32 operands: the word the test must not see."""
    v0, v1, v2 = 0xBC500C59, 0x435A2099, 0x3E3DB015
    return round_to_f32(f32_value(v0) * f32_value(v1) + f32_value(v2))


class ReferenceTests(unittest.TestCase):
    def test_constants_match_references(self):
        self.assertEqual(reference_f32_mods(), EXPECTED_F32_MODS)

    def test_mac_operands_tell_fused_from_unfused(self):
        self.assertNotEqual(reference_fused(), EXPECTED_F32_MODS[0])


class F32ValuTests(unittest.TestCase):
    def test_mac_unfused_and_conversion_output_modifiers(self):
        words = support.mrt0_words(support.translate_fixture("ps_f32_mods"))
        for i, (got, want) in enumerate(zip(words, EXPECTED_F32_MODS)):
            with self.subTest(component="xyzw"[i]):
                self.assertEqual(f"{got:#010x}", f"{want:#010x}")


HEADER = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%fm0 = OpConstant %float -0
%f0 = OpConstant %float 0
%f1 = OpConstant %float 1
%f4 = OpConstant %float 4
%f8 = OpConstant %float 8
%fm0p4 = OpConstant %float -0.4
%fm0p5 = OpConstant %float -0.5
%fm1p5 = OpConstant %float -1.5
%fm1 = OpConstant %float -1
%u_inf = OpConstant %uint 2139095040
%u_ninf = OpConstant %uint 4286578688
%u_nan = OpConstant %uint 2143289344
%main = OpFunction %void None %fn
%entry = OpLabel
%inf = OpBitcast %float %u_inf
%ninf = OpBitcast %float %u_ninf
%nan = OpBitcast %float %u_nan
"""
FOOTER = "OpReturn\nOpFunctionEnd\n"


def run_float(body):
    """Run `body` (which must define the float %r) and return the bits stored to %out."""
    text = HEADER + body + "%res = OpBitcast %uint %r\nOpStore %out %res\n" + FOOTER
    return si.run_fragment(text)["%out"][0]


def run_bool(body):
    text = HEADER + body + "%res = OpSelect %uint %b %c1 %c0\nOpStore %out %res\n" + FOOTER
    return si.run_fragment(text)["%out"][0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class InterpreterFloatOpTests(unittest.TestCase):
    """Expectations from GLSL.std.450 and the SPIR-V specification, not from the translator."""

    def ext(self, name, operand):
        return run_float(f"%r = OpExtInst %float %glsl {name} {operand}\n")

    def test_round_even_keeps_the_sign_of_zero(self):
        # roundEven(-0.4) is -0.0 (IEEE roundToIntegralTiesToEven); Python's round() returns the int 0.
        self.assertEqual(self.ext("RoundEven", "%fm0p4"), 0x80000000)
        self.assertEqual(self.ext("RoundEven", "%fm0p5"), 0x80000000)
        self.assertEqual(self.ext("RoundEven", "%fm1p5"), bits(-2.0))

    def test_floor_ceil_trunc(self):
        self.assertEqual(self.ext("Floor", "%fm0p5"), bits(-1.0))
        self.assertEqual(self.ext("Ceil", "%fm0p5"), 0x80000000)
        self.assertEqual(self.ext("Trunc", "%fm1p5"), bits(-1.0))
        self.assertEqual(self.ext("Trunc", "%fm0p4"), 0x80000000)
        self.assertEqual(self.ext("Floor", "%ninf"), 0xFF800000)

    def test_exp2_log2_sqrt_inversesqrt_where_defined(self):
        self.assertEqual(self.ext("Exp2", "%ninf"), 0)
        self.assertEqual(self.ext("Exp2", "%inf"), 0x7F800000)
        self.assertEqual(self.ext("Log2", "%f8"), bits(3.0))
        self.assertEqual(self.ext("Log2", "%inf"), 0x7F800000)
        self.assertEqual(self.ext("Sqrt", "%fm0"), 0x80000000)
        self.assertEqual(self.ext("Sqrt", "%f4"), bits(2.0))
        self.assertEqual(self.ext("InverseSqrt", "%f4"), bits(0.5))
        self.assertEqual(self.ext("InverseSqrt", "%inf"), 0)

    def test_undefined_inputs_raise(self):
        # GLSL.std.450: Log2 and InverseSqrt are undefined for x <= 0, Sqrt for x < 0.
        for name, operand in (("Log2", "%f0"), ("Log2", "%fm1"), ("InverseSqrt", "%fm0"), ("Sqrt", "%fm1")):
            with self.subTest(name=name, operand=operand):
                with self.assertRaises(si.UnsupportedConstruct):
                    self.ext(name, operand)

    def test_fdiv(self):
        self.assertEqual(run_float("%r = OpFDiv %float %f1 %f4\n"), bits(0.25))
        self.assertEqual(run_float("%r = OpFDiv %float %fm1 %inf\n"), 0x80000000)
        r = run_float("%r = OpFDiv %float %inf %inf\n")
        self.assertTrue((r >> 23) & 0xFF == 0xFF and r & 0x7FFFFF, f"{r:#x} is not NaN")
        with self.assertRaises(si.UnsupportedConstruct):        # SPIR-V: undefined if Operand 2 is 0
            run_float("%r = OpFDiv %float %f1 %f0\n")

    def test_fma_rounds_once(self):
        # (1 + 2**-12)**2 = 1 + 2**-11 + 2**-24 sits exactly halfway between two binary32 values; the tiny addend
        # 2**-70 decides the rounding (up). Rounding through a double first loses the addend and ties to even.
        text = HEADER.replace("%main = OpFunction", "%fa = OpConstant %float 1.000244140625\n"
                              "%fc = OpConstant %float 8.470329472543003e-22\n%main = OpFunction")
        body = "%r = OpExtInst %float %glsl Fma %fa %fa %fc\n%res = OpBitcast %uint %r\nOpStore %out %res\n"
        self.assertEqual(si.run_fragment(text + body + FOOTER)["%out"][0], 0x3F801001)
        self.assertEqual(si.f32_bits(si.fma_f32(-0.0, 1.0, -0.0)), 0x80000000)
        self.assertEqual(si.f32_bits(si.fma_f32(-0.0, 1.0, 0.0)), 0)

    def test_isnan_isinf(self):
        self.assertEqual(run_bool("%b = OpIsNan %bool %nan\n"), 1)
        self.assertEqual(run_bool("%b = OpIsNan %bool %inf\n"), 0)
        self.assertEqual(run_bool("%b = OpIsInf %bool %ninf\n"), 1)
        self.assertEqual(run_bool("%b = OpIsInf %bool %f1\n"), 0)


if __name__ == "__main__":
    unittest.main()
