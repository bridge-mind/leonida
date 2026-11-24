"""tests/spirv_interp.py holds an integer as its 32-bit pattern, whatever the signedness of its type.

    python -I tests/test_spirv_interp_signs.py

SPIR-V defines the signed and unsigned instructions on the bits of their operands, not on the operand types: OpSLess*
of two uint values compares them as signed, and OpULess*, OpShiftRightLogical and OpConvertUToF of an int value
read it as unsigned. The interpreter used to keep a value of a signed type as a negative Python int and a value of
an unsigned type as a non-negative one, and used both raw, so every case below where the instruction and the type
disagree gave the wrong word. Also here: x / -0.0, which returned +inf for a positive x.

Hand-written SPIR-V only, standard library only. No translator, no game data.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirv_interp as si  # noqa: E402

MODULE = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c4 = OpConstant %uint 4
%c8 = OpConstant %uint 8
%c31 = OpConstant %uint 31
%umin = OpConstant %uint 2147483648
%umax = OpConstant %uint 4294967295
%im8 = OpConstant %int -8
%i1 = OpConstant %int 1
%imin = OpConstant %int -2147483648
%f1 = OpConstant %float 1
%fm1 = OpConstant %float -1
%f0 = OpConstant %float 0
%fm0 = OpConstant %float -0.0
%fbig = OpConstant %float -3e9
%arr = OpTypeArray %uint %c4
%null = OpConstantNull %arr
%_ptr_Function_arr = OpTypePointer Function %arr
%_ptr_Function_uint = OpTypePointer Function %uint
%vgpr = OpVariable %_ptr_Function_arr Function %null
%sgpr = OpVariable %_ptr_Function_arr Function %null
%main = OpFunction %void None %fn
%entry = OpLabel
{body}OpReturn
OpFunctionEnd
"""
STORE = "%p = OpAccessChain %_ptr_Function_uint %vgpr %c0\nOpStore %p %r\n"


def word(body):
    """v0 after `body`, which leaves a uint, int or float in %r (a float is stored as its bits)."""
    return si.run(MODULE.format(body=body + STORE)).vgpr[0]


def flag(compare):
    return word(f"%c = {compare}\n%r = OpSelect %uint %c %c1 %c0\n")


def fbits(expr):
    return word(f"%d = {expr}\n%r = OpBitcast %uint %d\n")


class SignedInstructionsOnUnsignedValues(unittest.TestCase):
    def test_signed_compares_read_the_bits_as_signed(self):
        # 0x80000000 is INT_MIN and 0xffffffff is -1 to a signed compare, whatever the type says.
        self.assertEqual(flag("OpSLessThanEqual %bool %umin %c1"), 1)
        self.assertEqual(flag("OpSGreaterThanEqual %bool %umin %c1"), 0)
        self.assertEqual(flag("OpSLessThanEqual %bool %c1 %umax"), 0)
        self.assertEqual(flag("OpSGreaterThanEqual %bool %c0 %umax"), 1)
        self.assertEqual(flag("OpSLessThanEqual %bool %im8 %i1"), 1)          # unchanged: -8 <= 1
        self.assertEqual(flag("OpSGreaterThanEqual %bool %umax %umax"), 1)    # equal

    def test_smin_smax(self):
        # min(INT_MIN, 1) and max(-1, 1), as bit patterns.
        self.assertEqual(word("%r = OpExtInst %uint %glsl SMin %umin %c1\n"), 0x80000000)
        self.assertEqual(word("%r = OpExtInst %uint %glsl SMax %umax %c1\n"), 1)
        self.assertEqual(word("%r = OpExtInst %int %glsl SMin %im8 %i1\n"), 0xFFFFFFF8)
        self.assertEqual(word("%r = OpExtInst %uint %glsl UMax %umax %c1\n"), 0xFFFFFFFF)


class UnsignedInstructionsOnSignedValues(unittest.TestCase):
    def test_logical_shift_right(self):
        # -8 is 0xfffffff8: a logical shift brings in zeros.
        self.assertEqual(word("%r = OpShiftRightLogical %int %im8 %c1\n"), 0x7FFFFFFC)
        self.assertEqual(word("%r = OpShiftRightLogical %int %imin %c31\n"), 1)
        self.assertEqual(word("%r = OpShiftRightArithmetic %int %im8 %c1\n"), 0xFFFFFFFC)     # unchanged

    def test_unsigned_compares(self):
        self.assertEqual(flag("OpULessThan %bool %im8 %c1"), 0)                # 0xfffffff8 < 1 is false
        self.assertEqual(flag("OpUGreaterThan %bool %im8 %c1"), 1)
        self.assertEqual(flag("OpULessThanEqual %bool %imin %umax"), 1)
        self.assertEqual(flag("OpUGreaterThanEqual %bool %i1 %im8"), 0)
        self.assertEqual(flag("OpIEqual %bool %im8 %im8"), 1)

    def test_convert_to_float(self):
        self.assertEqual(fbits("OpConvertUToF %float %im8"), 0x4F800000)       # 4294967288 rounds to 2^32
        self.assertEqual(fbits("OpConvertSToF %float %im8"), 0xC1000000)       # -8.0
        self.assertEqual(fbits("OpConvertSToF %float %umax"), 0xBF800000)      # -1.0
        self.assertEqual(fbits("OpConvertUToF %float %umax"), 0x4F800000)

    def test_results_are_bit_patterns(self):
        # A negative result of a signed-typed instruction leaves as its 32-bit pattern and can feed any instruction.
        self.assertEqual(word("%r = OpConvertFToS %int %fm1\n"), 0xFFFFFFFF)
        self.assertEqual(word("%r = OpConvertFToS %int %fbig\n"), 0x80000000)  # saturates at INT_MIN
        self.assertEqual(word("%r = OpBitFieldSExtract %int %umax %c0 %c8\n"), 0xFFFFFFFF)
        self.assertEqual(word("%x = OpBitFieldSExtract %int %umax %c0 %c8\n"
                              "%r = OpShiftRightLogical %int %x %c1\n"), 0x7FFFFFFF)
        self.assertEqual(word("%x = OpBitcast %int %fm1\n%r = OpShiftRightLogical %int %x %c1\n"), 0x5FC00000)
        self.assertEqual(word("%x = OpConvertFToS %int %fm1\n%c = OpULessThan %bool %x %c1\n"
                              "%r = OpSelect %uint %c %c1 %c0\n"), 0)
        self.assertEqual(word("%r = OpIAdd %int %im8 %i1\n"), 0xFFFFFFF9)
        self.assertEqual(word("%r = OpNot %int %im8\n"), 7)


class DivisionByZero(unittest.TestCase):
    def test_sign_of_the_infinity(self):
        self.assertEqual(fbits("OpFDiv %float %f1 %f0"), 0x7F800000)
        self.assertEqual(fbits("OpFDiv %float %f1 %fm0"), 0xFF800000)
        self.assertEqual(fbits("OpFDiv %float %fm1 %f0"), 0xFF800000)
        self.assertEqual(fbits("OpFDiv %float %fm1 %fm0"), 0x7F800000)

    def test_zero_over_zero_is_a_nan(self):
        for expr in ("OpFDiv %float %f0 %f0", "OpFDiv %float %fm0 %f0", "OpFDiv %float %f0 %fm0"):
            with self.subTest(expr=expr):
                w = fbits(expr)
                self.assertEqual(w & 0x7F800000, 0x7F800000)
                self.assertNotEqual(w & 0x007FFFFF, 0)


if __name__ == "__main__":
    unittest.main()
