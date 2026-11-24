"""tests/spirv_interp.py keeps the exact 32-bit pattern of a float made by OpBitcast (issue #509, as #533 did for
tests/semantic/spirv_mini_interp.py).

    python -I tests/test_spirv_interp_bits.py

Hand-written SPIR-V only, standard library only. No translator, no game data.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirv_interp as si  # noqa: E402

# Signalling NaNs (exponent all ones, bit 22 clear, mantissa not zero), a quiet NaN and two ordinary values.
# Going through a Python float set bit 22 of the first four.
WORDS = (0xFFA3A095, 0x7F800001, 0x7FBFFFFF, 0xFF800001, 0x7FC00001, 0x3F800000, 0x80000000)

MODULE = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%c0 = OpConstant %uint 0
%c4 = OpConstant %uint 4
%f1 = OpConstant %float 1
%t = OpConstantTrue %bool
%cw = OpConstant %uint {word}
%arr = OpTypeArray %uint %c4
%null = OpConstantNull %arr
%_ptr_Function_arr = OpTypePointer Function %arr
%_ptr_Function_uint = OpTypePointer Function %uint
%_ptr_Output_float = OpTypePointer Output %float
%vgpr = OpVariable %_ptr_Function_arr Function %null
%sgpr = OpVariable %_ptr_Function_arr Function %null
%out_mrt0 = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %fn
%entry = OpLabel
{body}OpReturn
OpFunctionEnd
"""


def run(word, body):
    return si.run(MODULE.format(word=word, body=body))


class BitcastKeepsBits(unittest.TestCase):
    def test_round_trip_through_every_pass_through(self):
        # uint -> float -> OpCopyObject -> composite -> OpSelect -> a float Output and back to a uint in v0.
        body = ("%x = OpBitcast %float %cw\n"
                "%y = OpCopyObject %float %x\n"
                "%v = OpCompositeConstruct %v2float %f1 %y\n"
                "%e = OpCompositeExtract %float %v 1\n"
                "%s = OpSelect %float %t %e %f1\n"
                "OpStore %out_mrt0 %s\n"
                "%w = OpBitcast %uint %s\n"
                "%p = OpAccessChain %_ptr_Function_uint %vgpr %c0\n"
                "OpStore %p %w\n")
        for word in WORDS:
            with self.subTest(word=f"{word:#010x}"):
                r = run(word, body)
                self.assertEqual(r.vgpr[0], word)
                self.assertEqual(si.f2u(r.outputs["out_mrt0"]), word)

    def test_still_a_nan_to_float_operations(self):
        r = run(0xFFA3A095, "%x = OpBitcast %float %cw\n%n = OpIsNan %bool %x\n"
                            "%k = OpSelect %uint %n %c4 %c0\n"
                            "%p = OpAccessChain %_ptr_Function_uint %vgpr %c0\nOpStore %p %k\n")
        self.assertEqual(r.vgpr[0], 4)

    def test_arithmetic_gives_a_nan(self):
        # Arithmetic drops the kept pattern: SPIR-V does not fix the bits of a NaN result, so only "a NaN" is
        # asserted (exponent all ones, mantissa not zero).
        r = run(0xFFA3A095, "%x = OpBitcast %float %cw\n%y = OpFAdd %float %x %f1\n%w = OpBitcast %uint %y\n"
                            "%p = OpAccessChain %_ptr_Function_uint %vgpr %c0\nOpStore %p %w\n")
        self.assertEqual(r.vgpr[0] & 0x7F800000, 0x7F800000)
        self.assertNotEqual(r.vgpr[0] & 0x007FFFFF, 0)

    def test_helpers(self):
        for word in WORDS:
            with self.subTest(word=f"{word:#010x}"):
                self.assertEqual(si.f2u(si.u2f(word)), word)
        self.assertEqual(si.f2u(1.5), 0x3FC00000)        # a plain float still converts as before


class HalfNanRuleTests(unittest.TestCase):
    """si.half_to_float / si.float_to_half pin one NaN rule on every Python version.

    struct's '<e' format drops NaN payloads before Python 3.14 and keeps them from 3.14 (CPython gh-130317), so
    the old struct-based helpers returned version-dependent words for NaN operands, while the suites feeding them
    (test_f16_op_sel_values and friends) are pinned to the pre-3.14 behaviour CI's 3.12 sees. The helpers are
    bit-level now with the chosen rule: a NaN keeps its sign and loses its payload to the canonical quiet NaN
    (f16 0x7E00/0xFE00, f32 0x7FC00000/0xFFC00000).
    """

    def test_nan_operands_keep_sign_and_drop_payload(self):
        for h, want in ((0x7E00, 0x7FC00000), (0xFE00, 0xFFC00000), (0x7C01, 0x7FC00000), (0x7FFF, 0x7FC00000),
                        (0xFE01, 0xFFC00000), (0xFFFF, 0xFFC00000), (0x7C00, 0x7F800000), (0xFC00, 0xFF800000)):
            with self.subTest(half=hex(h)):
                self.assertEqual(si.f2u(si.half_to_float(h)), want)
        for w, want in ((0x7FC00000, 0x7E00), (0xFFC00000, 0xFE00), (0x7F800000, 0x7C00), (0xFF800000, 0xFC00)):
            with self.subTest(f32=hex(w)):
                self.assertEqual(si.float_to_half(si.u2f(w)), want)
        # a double NaN that still carries a payload (bytes -> double keeps it on every version) canonicalizes too
        payload = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000123456))[0]
        self.assertEqual(si.float_to_half(payload), 0x7E00)
        self.assertEqual(si.float_to_half(-payload), 0xFE00)
        # non-NaN spot checks: exact grid values, the smallest half subnormal and both overflow boundaries
        self.assertEqual(si.f2u(si.half_to_float(0x0001)), 0x33800000)
        self.assertEqual(si.f2u(si.half_to_float(0x03FF)), 0x387FC000)
        self.assertEqual(si.float_to_half(si.u2f(0x477FE000)), 0x7BFF)      # 65504, max finite
        self.assertEqual(si.float_to_half(si.u2f(0x477FF000)), 0x7C00)      # 65520 RNE tie -> infinity


if __name__ == "__main__":
    unittest.main()
