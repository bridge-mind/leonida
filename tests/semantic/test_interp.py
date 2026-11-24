"""Self-tests for the mini SPIR-V interpreter, using hand-written modules and expectations taken from the SPIR-V
and GLSL.std.450 specifications (not from the translator)."""
import math
import struct
import unittest

import spirv_mini_interp as si
import spirv_wave_interp

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
%v2float = OpTypeVector %float 2
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c4 = OpConstant %uint 4
%c8 = OpConstant %uint 8
%c31 = OpConstant %uint 31
%c32 = OpConstant %uint 32
%cF0 = OpConstant %uint 4294967280
%cFF = OpConstant %uint 4294967295
%cInf = OpConstant %uint 2139095040
%c80 = OpConstant %uint 128
%ch = OpConstant %uint 1073757184
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%f3 = OpConstant %float 3
%f70000 = OpConstant %float 70000
%f2p5 = OpConstant %float 2.5
%f3p5 = OpConstant %float 3.5
%f0p5 = OpConstant %float 0.5
%int = OpTypeInt 32 1
%t = OpConstantTrue %bool
%f = OpConstantFalse %bool
%_ptr_Function_uint = OpTypePointer Function %uint
%arr = OpTypeArray %uint %c4
%_ptr_Function_arr = OpTypePointer Function %arr
%null_arr = OpConstantNull %arr
%ud = OpTypeStruct %uint
%umul = OpTypeStruct %uint %uint
%_ptr_PC = OpTypePointer PushConstant %ud
%pc = OpVariable %_ptr_PC PushConstant
%_ptr_PC_uint = OpTypePointer PushConstant %uint
%main = OpFunction %void None %fn
%entry = OpLabel
"""
FOOTER = "OpReturn\nOpFunctionEnd\n"


def run_word(body):
    """Run `body` (instructions that must define %res as a uint) and return the word stored to %out."""
    return si.run_fragment(HEADER + body + "OpStore %out %res\n" + FOOTER)["%out"][0]


class InterpreterTests(unittest.TestCase):
    def test_bitfield_extract(self):
        # SPIR-V: Count 0 gives 0; SExtract replicates the field's top bit; UExtract zero-extends.
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %cF0 %c2 %c4\n"), 0xFFFFFFFC)   # 0b1100 -> -4
        self.assertEqual(run_word("%res = OpBitFieldUExtract %uint %cF0 %c2 %c4\n"), 0xC)
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %c80 %c0 %c8\n"), 0xFFFFFF80)
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %cFF %c2 %c0\n"), 0)

    def test_bitfield_insert(self):
        self.assertEqual(run_word("%res = OpBitFieldInsert %uint %cF0 %c1 %c0 %c4\n"), 0xFFFFFFF1)

    def test_half_pack_unpack(self):
        self.assertEqual(run_word("%v = OpCompositeConstruct %v2float %f1 %f2\n"
                                  "%res = OpExtInst %uint %glsl PackHalf2x16 %v\n"), 0x40003C00)
        self.assertEqual(run_word("%v = OpCompositeConstruct %v2float %f70000 %f1\n"
                                  "%res = OpExtInst %uint %glsl PackHalf2x16 %v\n"), 0x3C007C00)  # 70000 -> +inf
        self.assertEqual(run_word("%u = OpExtInst %v2float %glsl UnpackHalf2x16 %ch\n"          # %ch = 0x40003c00
                                  "%x = OpCompositeExtract %float %u 1\n"
                                  "%res = OpBitcast %uint %x\n"), 0x40000000)                   # 2.0f

    def test_float_sign_operations_use_result_width(self):
        # The low word in 1 + 2**-30 and the finite magnitude 2**130 both need binary64.
        text = """\
OpCapability Shader
OpCapability Float64
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %f32_control
OpExecutionMode %main OriginUpperLeft
OpDecorate %out Location 0
OpDecorate %f32_control Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%double = OpTypeFloat 64
%u2 = OpTypeVector %uint 2
%u4 = OpTypeVector %uint 4
%precise64 = OpConstant %double 1.000000000931322574615478515625
%large_negative64 = OpConstant %double -1361129467683753853853498429727072845824
%zero32 = OpConstant %float 0
%negative32 = OpConstant %float -1.25
%ptr_out_words = OpTypePointer Output %u4
%ptr_out_pair = OpTypePointer Output %u2
%out = OpVariable %ptr_out_words Output
%f32_control = OpVariable %ptr_out_pair Output
%main = OpFunction %void None %fn
%entry = OpLabel
%neg64 = OpFNegate %double %precise64
%abs64 = OpExtInst %double %glsl FAbs %large_negative64
%neg_words = OpBitcast %u2 %neg64
%abs_words = OpBitcast %u2 %abs64
%neg_lo = OpCompositeExtract %uint %neg_words 0
%neg_hi = OpCompositeExtract %uint %neg_words 1
%abs_lo = OpCompositeExtract %uint %abs_words 0
%abs_hi = OpCompositeExtract %uint %abs_words 1
%words = OpCompositeConstruct %u4 %neg_lo %neg_hi %abs_lo %abs_hi
OpStore %out %words
%neg32 = OpFNegate %float %zero32
%abs32 = OpExtInst %float %glsl FAbs %negative32
%neg32_bits = OpBitcast %uint %neg32
%abs32_bits = OpBitcast %uint %abs32
%control = OpCompositeConstruct %u2 %neg32_bits %abs32_bits
OpStore %f32_control %control
"""
        outputs = si.run_fragment(text + FOOTER)
        self.assertEqual(outputs["%f32_control"], (0x80000000, 0x3FA00000))
        self.assertEqual(outputs["%out"], (0x00400000, 0xBFF00000, 0, 0x48100000))

    def test_fma_select_bitcast(self):
        self.assertEqual(run_word("%x = OpExtInst %float %glsl Fma %f2 %f3 %f1\n%res = OpBitcast %uint %x\n"),
                         0x40E00000)                                                              # 7.0f
        self.assertEqual(run_word("%res = OpSelect %uint %t %c1 %c2\n"), 1)
        self.assertEqual(run_word("%res = OpSelect %uint %f %c1 %c2\n"), 2)

    def test_signed_unsigned_min_max(self):
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SMax %cFF %c1\n"), 1)        # -1 vs 1 signed
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl UMin %cFF %c1\n"), 1)        # 2^32-1 vs 1 unsigned

    def test_integer_ops_wrap(self):
        self.assertEqual(run_word("%res = OpIAdd %uint %cFF %c2\n"), 1)
        self.assertEqual(run_word("%res = OpISub %uint %c1 %c2\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpIMul %uint %cFF %c2\n"), 0xFFFFFFFE)
        self.assertEqual(run_word("%res = OpShiftLeftLogical %uint %cFF %c4\n"), 0xFFFFFFF0)
        self.assertEqual(run_word("%res = OpShiftRightLogical %uint %cF0 %c4\n"), 0x0FFFFFFF)
        self.assertEqual(run_word("%res = OpBitwiseXor %uint %cF0 %cFF\n"), 0xF)
        self.assertEqual(run_word("%res = OpNot %uint %cF0\n"), 0xF)
        self.assertEqual(run_word("%res = OpBitCount %uint %cFF\n"), 32)
        self.assertEqual(run_word("%res = OpBitCount %uint %c0\n"), 0)

    def test_float_ops(self):
        self.assertEqual(run_word("%x = OpFMul %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0x40C00000)
        self.assertEqual(run_word("%x = OpFSub %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0xBF800000)
        self.assertEqual(run_word("%x = OpFAdd %float %f1 %f2\n%res = OpBitcast %uint %x\n"), 0x40400000)
        self.assertEqual(run_word("%x = OpFNegate %float %f3\n%res = OpBitcast %uint %x\n"), 0xC0400000)
        self.assertEqual(run_word("%b = OpFOrdEqual %bool %f2 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpFOrdLessThan %bool %f1 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpFUnordNotEqual %bool %f1 %f1\n"
                                  "%res = OpSelect %uint %b %c1 %c0\n"), 0)

    def test_conversions(self):
        # SToF reads its operand as signed; FToU and FToS round toward zero.
        self.assertEqual(run_word("%x = OpConvertUToF %float %c8\n%res = OpBitcast %uint %x\n"), 0x41000000)
        self.assertEqual(run_word("%x = OpConvertUToF %float %cFF\n%res = OpBitcast %uint %x\n"), 0x4F800000)
        self.assertEqual(run_word("%x = OpConvertSToF %float %cFF\n%res = OpBitcast %uint %x\n"), 0xBF800000)
        self.assertEqual(run_word("%x = OpConvertSToF %float %cF0\n%res = OpBitcast %uint %x\n"), 0xC1800000)
        self.assertEqual(run_word("%x = OpFNegate %float %f2p5\n%res = OpConvertFToS %int %x\n"), 0xFFFFFFFE)
        self.assertEqual(run_word("%res = OpConvertFToU %uint %f2p5\n"), 2)
        self.assertEqual(run_word("%res = OpConvertFToU %uint %f70000\n"), 70000)

    def test_int32_to_float_uses_result_width(self):
        # 2**24 + 1 is exact in binary64 but rounds to 2**24 in binary32.
        text = """\
OpCapability Shader
OpCapability Float64
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %f32_control
OpExecutionMode %main OriginUpperLeft
OpDecorate %out Location 0
OpDecorate %f32_control Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%sint = OpTypeInt 32 1
%float = OpTypeFloat 32
%double = OpTypeFloat 64
%bool = OpTypeBool
%u2 = OpTypeVector %uint 2
%u4 = OpTypeVector %uint 4
%unsigned = OpConstant %uint 16777217
%signed = OpConstant %sint -16777217
%rounded_unsigned = OpConstant %float 16777216
%rounded_signed = OpConstant %float -16777216
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%ptr_out_words = OpTypePointer Output %u4
%ptr_out_uint = OpTypePointer Output %uint
%out = OpVariable %ptr_out_words Output
%f32_control = OpVariable %ptr_out_uint Output
%main = OpFunction %void None %fn
%entry = OpLabel
%du = OpConvertUToF %double %unsigned
%ds = OpConvertSToF %double %signed
%bu = OpBitcast %u2 %du
%bs = OpBitcast %u2 %ds
%ulo = OpCompositeExtract %uint %bu 0
%uhi = OpCompositeExtract %uint %bu 1
%slo = OpCompositeExtract %uint %bs 0
%shi = OpCompositeExtract %uint %bs 1
%words = OpCompositeConstruct %u4 %ulo %uhi %slo %shi
OpStore %out %words
%fu = OpConvertUToF %float %unsigned
%fs = OpConvertSToF %float %signed
%u_ok = OpFOrdEqual %bool %fu %rounded_unsigned
%s_ok = OpFOrdEqual %bool %fs %rounded_signed
%f_ok = OpLogicalAnd %bool %u_ok %s_ok
%control = OpSelect %uint %f_ok %one %zero
OpStore %f32_control %control
"""
        outputs = si.run_fragment(text + FOOTER)
        self.assertEqual(outputs["%f32_control"], (1,))
        self.assertEqual(outputs["%out"], (0x10000000, 0x41700000, 0x10000000, 0xC1700000))

    def test_floor_ceil_trunc_and_signed_zero(self):
        def word(name, const, neg=False):
            pre = f"%n = OpFNegate %float {const}\n" if neg else ""
            x = "%n" if neg else const
            return run_word(f"{pre}%x = OpExtInst %float %glsl {name} {x}\n%res = OpBitcast %uint %x\n")
        self.assertEqual(word("Floor", "%f2p5"), 0x40000000)                 # 2.5 -> 2
        self.assertEqual(word("Floor", "%f2p5", neg=True), 0xC0400000)       # -2.5 -> -3
        self.assertEqual(word("Ceil", "%f2p5"), 0x40400000)                  # 2.5 -> 3
        self.assertEqual(word("Ceil", "%f2p5", neg=True), 0xC0000000)        # -2.5 -> -2
        self.assertEqual(word("Trunc", "%f3p5"), 0x40400000)                 # 3.5 -> 3
        self.assertEqual(word("Trunc", "%f3p5", neg=True), 0xC0400000)       # -3.5 -> -3
        # A zero result keeps the operand's sign: -0.5 rounds to -0.0 (0x80000000), 0.5 to +0.0.
        self.assertEqual(word("Ceil", "%f0p5", neg=True), 0x80000000)
        self.assertEqual(word("Trunc", "%f0p5", neg=True), 0x80000000)
        self.assertEqual(word("RoundEven", "%f0p5", neg=True), 0x80000000)
        self.assertEqual(word("Floor", "%f0p5"), 0)
        self.assertEqual(word("Floor", "%f70000"), 0x4788B800)              # already integral: unchanged
        # A vector operand rounds component by component (the translator emits a %v2float Floor/Ceil).
        for name, want in (("Floor", (0xC0400000, 0x40000000)), ("Ceil", (0xC0000000, 0x40400000))):
            got = tuple(run_word(f"%n = OpFNegate %float %f2p5\n%v = OpCompositeConstruct %v2float %n %f2p5\n"
                                 f"%r = OpExtInst %v2float %glsl {name} %v\n%x = OpCompositeExtract %float %r {i}\n"
                                 "%res = OpBitcast %uint %x\n") for i in (0, 1))
            self.assertEqual(got, want, name)

    def test_wide_integer_conversion_is_rejected(self):
        # A 64-bit operand must not be silently truncated to 32 bits before the conversion.
        text = HEADER.replace("%c0 = OpConstant %uint 0",
                              "%ulong = OpTypeInt 64 0\n"
                              "%c64_big = OpConstant %ulong 4294967296\n"
                              "%c0 = OpConstant %uint 0")
        for op in ("OpConvertUToF", "OpConvertSToF"):
            with self.subTest(op=op):
                body = text + (f"%x = {op} %float %c64_big\n"
                               "%res = OpBitcast %uint %x\nOpStore %out %res\n" + FOOTER)
                with self.assertRaises(si.UnsupportedConstruct):
                    si.run_fragment(body)

    def test_64bit_operand_below_2_32_is_rejected(self):
        # A 64-bit operand is rejected by its type width, not its value: signed 2**31 in a 64-bit int would
        # otherwise convert as -2147483648.0 under OpConvertSToF (follow-up to #346).
        text = HEADER.replace("%c0 = OpConstant %uint 0",
                              "%ulong = OpTypeInt 64 0\n%long = OpTypeInt 64 1\n"
                              "%cl_one = OpConstant %ulong 1\n%cl_2p31 = OpConstant %long 2147483648\n"
                              "%c0 = OpConstant %uint 0")
        for op in ("OpConvertUToF", "OpConvertSToF"):
            for src in ("%cl_one", "%cl_2p31"):
                with self.subTest(op=op, src=src):
                    body = text + (f"%x = {op} %float {src}\n"
                                   "%res = OpBitcast %uint %x\nOpStore %out %res\n" + FOOTER)
                    with self.assertRaises(si.UnsupportedConstruct):
                        si.run_fragment(body)

    WIDE = HEADER.replace("%c0 = OpConstant %uint 0",
                          "%ulong = OpTypeInt 64 0\n"
                          "%cl_1 = OpConstant %ulong 1\n"
                          "%cl_2p31 = OpConstant %ulong 2147483648\n"
                          "%cl_2p32 = OpConstant %ulong 4294967296\n"
                          "%cl_2p32_5 = OpConstant %ulong 4294967301\n"
                          "%cl_hi = OpConstant %ulong 18446744069414584320\n"       # 0xFFFFFFFF00000000 = -2**32
                          "%cl_m1 = OpConstant %ulong 18446744073709551615\n"
                          "%c0 = OpConstant %uint 0")

    def test_bit_count_counts_every_bit_of_a_64bit_operand(self):
        # OpBitCount counts the set bits of its operand, whatever the width of the (32-bit) result.
        for src, want in (("%cl_hi", 32), ("%cl_m1", 64), ("%cl_2p32_5", 3), ("%cl_1", 1)):
            with self.subTest(src=src):
                out = si.run_fragment(self.WIDE + f"%res = OpBitCount %uint {src}\nOpStore %out %res\n" + FOOTER)
                self.assertEqual(out["%out"], (want,))
        self.assertEqual(run_word("%res = OpBitCount %uint %cF0\n"), 28)

    def test_signed_min_max_clamp_abs_use_the_width_of_a_64bit_type(self):
        # GLSL.std.450 SMin/SMax/SClamp/SAbs read their operands as signed at the width of their type. 2**31 is
        # positive in a 64-bit integer and 0xFFFFFFFF00000000 is -2**32; a 32-bit reading takes them as
        # -2**31 and 0.
        for inst, want in (("SMin %cl_2p31 %cl_1", "%cl_1"), ("SMax %cl_2p31 %cl_1", "%cl_2p31"),
                           ("SMin %cl_hi %cl_1", "%cl_hi"), ("SMax %cl_hi %cl_m1", "%cl_m1"),
                           ("SClamp %cl_2p31 %cl_1 %cl_2p32", "%cl_2p31"), ("SClamp %cl_hi %cl_m1 %cl_1", "%cl_m1"),
                           ("SAbs %cl_hi", "%cl_2p32"), ("SAbs %cl_2p32_5", "%cl_2p32_5"), ("SAbs %cl_m1", "%cl_1")):
            with self.subTest(inst=inst):
                out = si.run_fragment(self.WIDE + f"%r = OpExtInst %ulong %glsl {inst}\n"
                                      f"%eq = OpIEqual %bool %r {want}\n%res = OpSelect %uint %eq %c1 %c0\n"
                                      "OpStore %out %res\n" + FOOTER)
                self.assertEqual(out["%out"], (1,))

    def test_glsl_integer_operations_reject_mixed_component_widths(self):
        for inst in ("SAbs %cl_1", "SMin %c0 %cl_1", "SMax %cl_1 %c0",
                     "SClamp %c0 %c0 %cl_1", "FindILsb %cl_1", "FindSMsb %cl_1", "FindUMsb %cl_1"):
            with self.subTest(inst=inst):
                body = f"%res = OpExtInst %uint %glsl {inst}\n"
                with self.assertRaisesRegex(si.UnsupportedConstruct, "64-bit.*32-bit"):
                    si.run_fragment(self.WIDE + body + "OpStore %out %res\n" + FOOTER)

    def test_find_bit_of_a_64bit_operand_is_rejected(self):
        # GLSL.std.450 limits FindILsb, FindSMsb and FindUMsb to 32-bit components.
        for name in ("FindILsb", "FindSMsb", "FindUMsb"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(si.UnsupportedConstruct, "64-bit"):
                    si.run_fragment(self.WIDE + f"%r = OpExtInst %ulong %glsl {name} %cl_2p32\n"
                                    "%res = OpUConvert %uint %r\nOpStore %out %res\n" + FOOTER)

    def test_round_even_clamp_max(self):
        # RoundEven sends halves to the even neighbour: 2.5 -> 2, 3.5 -> 4.
        self.assertEqual(run_word("%x = OpExtInst %float %glsl RoundEven %f2p5\n%res = OpBitcast %uint %x\n"),
                         0x40000000)
        self.assertEqual(run_word("%x = OpExtInst %float %glsl RoundEven %f3p5\n%res = OpBitcast %uint %x\n"),
                         0x40800000)
        self.assertEqual(run_word("%x = OpExtInst %float %glsl FClamp %f3 %f1 %f2\n%res = OpBitcast %uint %x\n"),
                         0x40000000)
        self.assertEqual(run_word("%n = OpFNegate %float %f3\n%x = OpExtInst %float %glsl FClamp %n %f1 %f2\n"
                                  "%res = OpBitcast %uint %x\n"), 0x3F800000)
        self.assertEqual(run_word("%x = OpExtInst %float %glsl FClamp %f2p5 %f1 %f3\n%res = OpBitcast %uint %x\n"),
                         0x40200000)
        self.assertEqual(run_word("%x = OpExtInst %float %glsl FMax %f1 %f3\n%res = OpBitcast %uint %x\n"),
                         0x40400000)

    def test_64bit_arithmetic(self):
        text = HEADER.replace("%c0 = OpConstant %uint 0",
                              "%ulong = OpTypeInt 64 0\n"
                              "%c64_1 = OpConstant %ulong 1\n"
                              "%c64_32 = OpConstant %ulong 32\n"
                              "%c0 = OpConstant %uint 0")
        text += ("%sh = OpShiftLeftLogical %ulong %c64_1 %c64_32\n"
                 "%sum = OpIAdd %ulong %sh %c64_1\n"
                 "%prod = OpIMul %ulong %sum %c64_1\n"
                 "%res = OpUConvert %uint %prod\n" + "OpStore %out %res\n" + FOOTER)
        self.assertEqual(si.run_fragment(text)["%out"][0], 1)

    def test_64bit_pointer_load(self):
        text = HEADER.replace("OpMemoryModel Logical GLSL450",
                              'OpCapability PhysicalStorageBufferAddresses\n'
                              'OpExtension "SPV_KHR_physical_storage_buffer"\n'
                              "OpMemoryModel PhysicalStorageBuffer64 GLSL450")
        decls = ("%ulong = OpTypeInt 64 0\n"
                 "%rt = OpTypeRuntimeArray %uint\n"
                 "%_ptr_PSB_rt = OpTypePointer PhysicalStorageBuffer %rt\n"
                 "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint\n"
                 "%addr = OpConstant %ulong 4096\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        text += ("%p = OpConvertUToPtr %_ptr_PSB_rt %addr\n"
                 "%q = OpAccessChain %_ptr_PSB_uint %p %c1\n"
                 "%res = OpLoad %uint %q\n"
                 "OpStore %out %res\n" + FOOTER)
        self.assertEqual(si.run_fragment(text, memory={4100: 0x12345678})["%out"][0], 0x12345678)

    def test_physical_storage_buffer_access_is_one_aligned_32bit_word(self):
        """The flat memory holds one 32-bit word per 4-byte-aligned address, so a PSB load or store of another
        type, or at an address that is not 4-byte aligned (undefined under `Aligned 4`), must raise instead of
        reading or writing one dictionary entry (a 64-bit load would give only its low word, a load at 4098 the
        0 of a key nobody wrote)."""
        text = HEADER.replace("OpMemoryModel Logical GLSL450",
                              'OpCapability PhysicalStorageBufferAddresses\n'
                              'OpExtension "SPV_KHR_physical_storage_buffer"\n'
                              "OpMemoryModel PhysicalStorageBuffer64 GLSL450")
        decls = ("%ulong = OpTypeInt 64 0\n"
                 "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint\n"
                 "%_ptr_PSB_ulong = OpTypePointer PhysicalStorageBuffer %ulong\n"
                 "%_ptr_PSB_float = OpTypePointer PhysicalStorageBuffer %float\n"
                 "%a4096 = OpConstant %ulong 4096\n%a4098 = OpConstant %ulong 4098\n"
                 "%cl_big = OpConstant %ulong 4294967296\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        memory = {4096: 0x11223344, 4100: 0x55667788}

        def run(body):
            return si.run_fragment(text + body + "OpStore %out %res\n" + FOOTER, memory=dict(memory))

        # Control: an aligned 32-bit load and store work.
        self.assertEqual(run("%p = OpConvertUToPtr %_ptr_PSB_uint %a4096\n"
                             "%res = OpLoad %uint %p Aligned 4\n")["%out"][0], 0x11223344)
        mem = dict(memory)
        si.run_fragment(text + "%p = OpConvertUToPtr %_ptr_PSB_uint %a4096\nOpStore %p %c8 Aligned 4\n"
                        "%res = OpLoad %uint %p Aligned 4\nOpStore %out %res\n" + FOOTER, memory=mem)
        self.assertEqual(mem, {4096: 8, 4100: 0x55667788})
        cases = {
            "unaligned load": "%p = OpConvertUToPtr %_ptr_PSB_uint %a4098\n%res = OpLoad %uint %p Aligned 4\n",
            "unaligned store": "%p = OpConvertUToPtr %_ptr_PSB_uint %a4098\nOpStore %p %c8 Aligned 4\n"
                               "%res = OpCopyObject %uint %c0\n",
            "64-bit load": "%p = OpConvertUToPtr %_ptr_PSB_ulong %a4096\n%w = OpLoad %ulong %p Aligned 8\n"
                           "%res = OpUConvert %uint %w\n",
            "64-bit store": "%p = OpConvertUToPtr %_ptr_PSB_ulong %a4096\nOpStore %p %cl_big Aligned 8\n"
                            "%res = OpCopyObject %uint %c0\n",
            "float load": "%p = OpConvertUToPtr %_ptr_PSB_float %a4096\n%x = OpLoad %float %p Aligned 4\n"
                          "%res = OpBitcast %uint %x\n",
            "float store": "%p = OpConvertUToPtr %_ptr_PSB_float %a4096\nOpStore %p %f1 Aligned 4\n"
                           "%res = OpCopyObject %uint %c0\n",
        }
        for name, body in cases.items():
            with self.subTest(name):
                with self.assertRaises(si.UnsupportedConstruct):
                    run(body)

    def test_sconvert_sign_extends_and_truncates(self):
        """OpSConvert: sign-extend from the operand's width, wrap to the result's (the global_* offset path)."""
        text = HEADER.replace("OpCapability Shader", "OpCapability Shader\nOpCapability Int64")
        text = text.replace("%main = OpFunction", "%long = OpTypeInt 64 1\n%ulong = OpTypeInt 64 0\n"
                            "%int = OpTypeInt 32 1\n%neg = OpConstant %int -16\n%main = OpFunction")
        text += ("%w = OpSConvert %long %neg\n"           # -16 as 64 bits
                 "%u = OpBitcast %ulong %w\n"
                 "%hi64 = OpShiftRightLogical %ulong %u %c64_32\n"
                 "%hi = OpUConvert %uint %hi64\n"
                 "%back = OpSConvert %int %w\n"           # and back to 32 bits
                 "%lo = OpBitcast %uint %back\n"
                 "%res = OpBitwiseXor %uint %hi %lo\n"    # 0xFFFFFFFF ^ 0xFFFFFFF0
                 "OpStore %out %res\n" + FOOTER)
        text = text.replace("%main = OpFunction", "%c64_32 = OpConstant %ulong 32\n%main = OpFunction")
        self.assertEqual(si.run_fragment(text)["%out"][0], 0x0000000F)

    def test_physical_storage_buffer_atomics(self):
        """Atomics on PhysicalStorageBuffer memory return the original value and write the combined one."""
        text = HEADER.replace("OpMemoryModel Logical GLSL450",
                              'OpCapability PhysicalStorageBufferAddresses\n'
                              'OpExtension "SPV_KHR_physical_storage_buffer"\n'
                              "OpMemoryModel PhysicalStorageBuffer64 GLSL450")
        decls = ("%ulong = OpTypeInt 64 0\n"
                 "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint\n"
                 "%addr = OpConstant %ulong 4096\n%cm1 = OpConstant %uint 4294967295\n%c5 = OpConstant %uint 5\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        cases = [  # (instruction, memory before, expected memory after, expected result)
            ("OpAtomicIAdd %uint %p %c1 %c0 %cm1", 7, 6, 7),
            ("OpAtomicISub %uint %p %c1 %c0 %c2", 1, 0xFFFFFFFF, 1),
            ("OpAtomicSMin %uint %p %c1 %c0 %cm1", 3, 0xFFFFFFFF, 3),
            ("OpAtomicUMin %uint %p %c1 %c0 %cm1", 3, 3, 3),
            ("OpAtomicSMax %uint %p %c1 %c0 %c2", 0x80000000, 2, 0x80000000),
            ("OpAtomicXor %uint %p %c1 %c0 %c5", 6, 3, 6),
            ("OpAtomicExchange %uint %p %c1 %c0 %c5", 9, 5, 9),
            ("OpAtomicCompareExchange %uint %p %c1 %c0 %c0 %c5 %c2", 2, 5, 2),   # equal: swapped
            ("OpAtomicCompareExchange %uint %p %c1 %c0 %c0 %c5 %c2", 3, 3, 3),   # not equal: kept
        ]
        for ins, before, after, result in cases:
            with self.subTest(ins=ins, before=before):
                body = (text + "%p = OpConvertUToPtr %_ptr_PSB_uint %addr\n"
                        f"%res = {ins}\nOpStore %out %res\n" + FOOTER)
                memory = {4096: before}
                self.assertEqual(si.run_fragment(body, memory=memory)["%out"][0], result)
                self.assertEqual(memory[4096], after)

    def test_inputs_and_push_constants_can_be_seeded(self):
        text = HEADER.replace('OpEntryPoint Fragment %main "main" %out',
                              'OpEntryPoint Fragment %main "main" %out %in')
        decls = ("%v4float = OpTypeVector %float 4\n"
                 "%_ptr_Input_v4 = OpTypePointer Input %v4float\n"
                 "%_ptr_Input_float = OpTypePointer Input %float\n"
                 "%in = OpVariable %_ptr_Input_v4 Input\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        text += ("%ip = OpAccessChain %_ptr_Input_float %in %c2\n"
                 "%iv = OpLoad %float %ip\n"
                 "%ib = OpBitcast %uint %iv\n"
                 "%pp = OpAccessChain %_ptr_PC_uint %pc %c0\n"
                 "%pv = OpLoad %uint %pp\n"
                 "%res = OpIAdd %uint %ib %pv\n"
                 "OpStore %out %res\n" + FOOTER)
        out = si.run_fragment(text, inputs={"%in": (0.0, 0.0, 1.5, 0.0)}, push_constants=[0x100])
        self.assertEqual(out["%out"][0], 0x3FC00100)

    def test_sub_borrow_and_comparisons(self):
        self.assertEqual(run_word("%s = OpISubBorrow %umul %c1 %c2\n"
                                  "%d = OpCompositeExtract %uint %s 0\n"
                                  "%b = OpCompositeExtract %uint %s 1\n"
                                  "%res = OpBitwiseAnd %uint %d %b\n"), 1)
        self.assertEqual(run_word("%s = OpISubBorrow %umul %c2 %c1\n"
                                  "%res = OpCompositeExtract %uint %s 1\n"), 0)
        self.assertEqual(run_word("%b = OpIEqual %bool %c2 %c2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpINotEqual %bool %c2 %c1\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpSLessThan %bool %cFF %c0\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpLogicalAnd %bool %t %f\n%res = OpSelect %uint %b %c1 %c2\n"), 2)
        self.assertEqual(run_word("%b = OpLogicalOr %bool %f %t\n%res = OpSelect %uint %b %c1 %c2\n"), 1)

    def test_a_nan_the_operation_creates_is_the_same_on_every_host(self):
        floats = "%inf = OpBitcast %float %cInf\n%zero = OpBitcast %float %c0\n"
        for op in ("OpFMul %float %zero %inf", "OpFMul %float %inf %zero", "OpFSub %float %inf %inf",
                   "OpFAdd %float %inf %ninf", "OpExtInst %float %glsl Fma %zero %inf %f1"):
            with self.subTest(op=op):
                body = floats + "%ninf = OpFNegate %float %inf\n" + f"%x = {op}\n%res = OpBitcast %uint %x\n"
                self.assertEqual(run_word(body), 0xFFC00000)
        header = HEADER.replace("%main = OpFunction", "%cNaN = OpConstant %uint 2143289345\n%main = OpFunction")
        for op in ("OpFAdd %float %nan %f1", "OpFMul %float %nan %f2", "OpExtInst %float %glsl Fma %nan %f2 %f1"):
            with self.subTest(op=op, propagated=True):
                text = header + f"%nan = OpBitcast %float %cNaN\n%x = {op}\n%res = OpBitcast %uint %x\n"
                self.assertEqual(si.run_fragment(text + "OpStore %out %res\n" + FOOTER)["%out"][0], 0x7FC00001)

    def test_created_nan_is_pinned_whatever_the_host_gives(self):
        arm64_nan = math.copysign(math.nan, 1.0)
        self.assertEqual(struct.pack("<d", si.ieee_nan(arm64_nan, 0.0, math.inf)),
                         struct.pack("<Q", 0xFFF8000000000000))
        self.assertEqual(struct.pack("<d", si.ieee_nan(arm64_nan, arm64_nan, 1.0)), struct.pack("<d", arm64_nan))
        self.assertEqual(si.ieee_nan(2.5, 1.0, 1.5), 2.5)

    def test_ordered_integer_comparisons(self):
        # 2 vs 0xFFFFFFFF: unsigned 2 < 4294967295, signed 2 > -1
        for op, want in (("OpULessThan", 1), ("OpULessThanEqual", 1), ("OpUGreaterThan", 0),
                         ("OpUGreaterThanEqual", 0), ("OpSLessThan", 0), ("OpSLessThanEqual", 0),
                         ("OpSGreaterThan", 1), ("OpSGreaterThanEqual", 1)):
            with self.subTest(op=op):
                self.assertEqual(run_word(f"%b = {op} %bool %c2 %cFF\n%res = OpSelect %uint %b %c1 %c0\n"), want)
        for op, want in (("OpULessThanEqual", 1), ("OpUGreaterThan", 0), ("OpSLessThanEqual", 1),
                         ("OpSGreaterThan", 0), ("OpSGreaterThanEqual", 1)):
            with self.subTest(op=op, equal=True):
                self.assertEqual(run_word(f"%b = {op} %bool %c2 %c2\n%res = OpSelect %uint %b %c1 %c0\n"), want)

    def test_signed_comparisons_use_the_operand_width(self):
        # 0x80000000 as a 64-bit operand is positive; as a 32-bit one it is negative.
        types = ("%ulong = OpTypeInt 64 0\n%L80 = OpConstant %ulong 2147483648\n%L1 = OpConstant %ulong 1\n"
                 "%Lneg = OpConstant %ulong 18446744073709551615\n%c8000 = OpConstant %uint 2147483648\n")
        header = HEADER.replace("%main = OpFunction", types + "%main = OpFunction")
        for a, b, want in (("%L80", "%L1", 1), ("%Lneg", "%L1", 0), ("%c8000", "%c1", 0)):
            with self.subTest(a=a, b=b):
                body = f"%b = OpSGreaterThan %bool {a} {b}\n%res = OpSelect %uint %b %c1 %c0\n"
                self.assertEqual(si.run_fragment(header + body + "OpStore %out %res\n" + FOOTER)["%out"][0], want)
        body = ("%x = OpIAdd %ulong %L80 %L80\n%b = OpSGreaterThan %bool %x %L1\n"   # a result id keeps its width
                "%res = OpSelect %uint %b %c1 %c0\n")
        self.assertEqual(si.run_fragment(header + body + "OpStore %out %res\n" + FOOTER)["%out"][0], 1)
    def test_add_carry(self):
        # 0xFFFFFFFF + 2 = 1, carry 1; 1 + 2 = 3, carry 0
        self.assertEqual(run_word("%s = OpIAddCarry %umul %cFF %c2\n"
                                  "%d = OpCompositeExtract %uint %s 0\n"
                                  "%k = OpCompositeExtract %uint %s 1\n"
                                  "%res = OpBitwiseAnd %uint %d %k\n"), 1)
        self.assertEqual(run_word("%s = OpIAddCarry %umul %c1 %c2\n%res = OpCompositeExtract %uint %s 0\n"), 3)
        self.assertEqual(run_word("%s = OpIAddCarry %umul %c1 %c2\n%res = OpCompositeExtract %uint %s 1\n"), 0)

    def test_float_mul_ord_equal_findsmsb_sin(self):
        self.assertEqual(run_word("%x = OpFMul %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0x40C00000)
        self.assertEqual(run_word("%b = OpFOrdEqual %bool %f2 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%nf = OpBitcast %float %cFF\n"
                                  "%b = OpFOrdEqual %bool %nf %nf\n%res = OpSelect %uint %b %c1 %c0\n"), 0)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %c0\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %cFF\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %c80\n"), 7)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %cF0\n"), 3)
        # FindILsb / FindUMsb: bit number of the lowest / highest 1 bit, -1 for 0 (GLSL.std.450).
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindILsb %c0\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindILsb %cF0\n"), 4)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindUMsb %c0\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindUMsb %cF0\n"), 31)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindUMsb %c80\n"), 7)
        # UMax / SMin and the clamps compare as the name says: 0xffffffff is the largest unsigned, -1 signed.
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl UMax %cFF %c1\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SMin %cFF %c1\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl UClamp %cFF %c1 %c8\n"), 8)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SClamp %cFF %c1 %c8\n"), 1)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SClamp %c4 %cF0 %c8\n"), 4)
        with self.assertRaises(si.UnsupportedConstruct):                                  # minVal > maxVal
            run_word("%res = OpExtInst %uint %glsl UClamp %c4 %c8 %c1\n")
        self.assertEqual(run_word("%b = OpUGreaterThanEqual %bool %c8 %c8\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpUGreaterThanEqual %bool %c1 %cFF\n%res = OpSelect %uint %b %c1 %c0\n"), 0)
        self.assertEqual(run_word("%x = OpBitcast %float %c0\n%s = OpExtInst %float %glsl Sin %x\n"
                                  "%res = OpBitcast %uint %s\n"), 0)

    def test_bit_reverse_and_sabs(self):
        self.assertEqual(run_word("%res = OpBitReverse %uint %c1\n"), 0x80000000)
        self.assertEqual(run_word("%res = OpBitReverse %uint %cF0\n"), 0x0FFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SAbs %cFF\n"), 1)             # |-1|
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SAbs %c2\n"), 2)
        self.assertEqual(run_word("%m = OpShiftLeftLogical %uint %c1 %c31\n"                   # |INT_MIN| wraps
                                  "%res = OpExtInst %uint %glsl SAbs %m\n"), 0x80000000)

    def test_forward_branch_and_kill(self):
        live = (HEADER + "OpStore %out %c1\nOpSelectionMerge %live None\n"
                "OpBranchConditional %t %live %dead\n%dead = OpLabel\nOpKill\n%live = OpLabel\n" + FOOTER)
        dead = (HEADER + "OpStore %out %c1\nOpSelectionMerge %live None\n"
                "OpBranchConditional %f %live %dead\n%dead = OpLabel\nOpKill\n%live = OpLabel\n" + FOOTER)
        self.assertEqual(si.run_fragment(live)["%out"][0], 1)
        self.assertEqual(si.run_fragment(dead), {})

    def test_float_ordered_unordered_and_classify(self):
        nan = "%nan = OpBitcast %float %cFF\n"                       # 0xFFFFFFFF is a NaN
        for op, nan_result, ordered_result in (("OpFOrdNotEqual", 0, 1), ("OpFUnordNotEqual", 1, 1),
                                               ("OpFUnordLessThan", 1, 1), ("OpFUnordLessThanEqual", 1, 1),
                                               ("OpFUnordGreaterThan", 1, 0), ("OpFUnordGreaterThanEqual", 1, 0),
                                               ("OpFUnordEqual", 1, 0)):
            with self.subTest(op=op):   # 2 vs NaN, then 2 vs 3
                self.assertEqual(run_word(nan + f"%b = {op} %bool %f2 %nan\n%res = OpSelect %uint %b %c1 %c0\n"),
                                 nan_result)
                self.assertEqual(run_word(f"%b = {op} %bool %f2 %f3\n%res = OpSelect %uint %b %c1 %c0\n"),
                                 ordered_result)
        self.assertEqual(run_word(nan + "%b = OpIsNan %bool %nan\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpIsNan %bool %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 0)
        self.assertEqual(run_word("%i = OpBitcast %float %cInf\n%b = OpIsInf %bool %i\n"
                                  "%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word(nan + "%b = OpIsInf %bool %nan\n%res = OpSelect %uint %b %c1 %c0\n"), 0)

    def test_float_compares_with_equal_operands(self):
        # 2 vs 2 for every ordered and unordered relation (asked in review on #262): without this an interpreter
        # that dropped the equality part of an Equal / LessThanEqual / GreaterThanEqual compare, or that turned
        # NotEqual into "unordered only", would still pass the NaN and 2-vs-3 cases above.
        for op, want in (("OpFOrdEqual", 1), ("OpFOrdNotEqual", 0), ("OpFOrdLessThan", 0),
                         ("OpFOrdLessThanEqual", 1), ("OpFOrdGreaterThan", 0), ("OpFOrdGreaterThanEqual", 1),
                         ("OpFUnordEqual", 1), ("OpFUnordNotEqual", 0), ("OpFUnordLessThan", 0),
                         ("OpFUnordLessThanEqual", 1), ("OpFUnordGreaterThan", 0), ("OpFUnordGreaterThanEqual", 1)):
            with self.subTest(op=op):
                self.assertEqual(run_word(f"%b = {op} %bool %f2 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), want)
        # -0.0 equals +0.0 in both families.
        for op, want in (("OpFOrdEqual", 1), ("OpFUnordEqual", 1), ("OpFOrdNotEqual", 0), ("OpFUnordNotEqual", 0)):
            with self.subTest(op=op, operands="-0.0, +0.0"):
                self.assertEqual(run_word("%z = OpBitcast %float %c0\n%n = OpFNegate %float %z\n"
                                          f"%b = {op} %bool %n %z\n%res = OpSelect %uint %b %c1 %c0\n"), want)

    def test_phi_selects_the_value_from_the_taken_predecessor(self):
        for cond, want in (("%t", 1), ("%f", 2)):
            text = (HEADER + f"OpSelectionMerge %merge None\nOpBranchConditional {cond} %a %b\n"
                    "%a = OpLabel\nOpBranch %merge\n"
                    "%b = OpLabel\nOpBranch %merge\n"
                    "%merge = OpLabel\n%res = OpPhi %uint %c1 %a %c2 %b\n"
                    "OpStore %out %res\n" + FOOTER)
            with self.subTest(cond=cond):
                self.assertEqual(si.run_fragment(text)["%out"][0], want)

    def test_phi_without_a_matching_predecessor_names_the_result_and_edge(self):
        body = "OpBranch %next\n%next = OpLabel\n%answer = OpPhi %uint %c1 %other\n"
        with self.assertRaisesRegex(si.UnsupportedConstruct, r"%answer.*%entry"):
            si.run_fragment(HEADER + body + FOOTER)

    def test_phi_before_the_first_label(self):
        text = HEADER.replace("%entry = OpLabel\n", "") + "%answer = OpPhi %uint %c1 %other\nOpFunctionEnd\n"
        with self.assertRaisesRegex(si.InvalidModule, r"%answer.*start of the function"):
            si.run_fragment(text)

    def test_duplicate_phi_result_is_rejected(self):
        body = ("OpBranch %next\n%next = OpLabel\n%a = OpPhi %uint %c1 %entry\n"
                "%a = OpPhi %uint %c2 %entry\n")
        with self.assertRaisesRegex(si.UnsupportedConstruct, r"duplicate.*%a"):
            si.run_fragment(HEADER + body + FOOTER)

    def test_phis_of_a_block_take_their_values_together(self):
        # A loop header that swaps %a and %b on every back edge: each OpPhi reads the other's value from before
        # the block. Three iterations from (1, 2) leave (2, 1); assigning %a first would give %b the new %a, (2, 2).
        text = (HEADER.replace('"main" %out', '"main" %out %out2')
                .replace("%main = OpFunction", "%out2 = OpVariable %_ptr_Output_uint Output\n"
                         "%c3 = OpConstant %uint 3\n%main = OpFunction") +
                "OpBranch %head\n"
                "%head = OpLabel\n"
                "%a = OpPhi %uint %c1 %entry %b %cont\n"
                "%b = OpPhi %uint %c2 %entry %a %cont\n"
                "%i = OpPhi %uint %c0 %entry %inext %cont\n"
                "%more = OpULessThan %bool %i %c3\n"
                "OpLoopMerge %done %cont None\nOpBranchConditional %more %cont %done\n"
                "%cont = OpLabel\n%inext = OpIAdd %uint %i %c1\nOpBranch %head\n"
                "%done = OpLabel\nOpStore %out %a\nOpStore %out2 %b\n" + FOOTER)
        out = si.run_fragment(text)
        self.assertEqual((out["%out"], out["%out2"]), ((2,), (1,)))
        lane0 = spirv_wave_interp.run_wave(text, wave=32)["outputs"][0]     # the other interpreter agrees
        self.assertEqual((lane0["%out"], lane0["%out2"]), ((2,), (1,)))

    def test_function_array_access(self):
        self.assertEqual(run_word("%arr_v = OpVariable %_ptr_Function_arr Function %null_arr\n"
                                  "%p = OpAccessChain %_ptr_Function_uint %arr_v %c2\n"
                                  "OpStore %p %cF0\n"
                                  "%q = OpAccessChain %_ptr_Function_uint %arr_v %c2\n"
                                  "%res = OpLoad %uint %q\n"), 0xFFFFFFF0)

    def test_unsupported_constructs_are_rejected(self):
        for body in ("%res = OpUDiv %uint %c2 %c1\n",                                     # opcode not modelled
                     "%res = OpExtInst %uint %glsl Ldexp %c2 %c1\n",                     # ext inst not modelled
                     "%res = OpBitFieldUExtract %uint %cFF %c32 %c1\n",                  # undefined per spec
                     "%res = OpShiftLeftLogical %uint %cFF %c32\n",                     # undefined per spec
                     "%res = OpShiftRightLogical %uint %cFF %c32\n",                    # undefined per spec
                     "%x = OpBitcast %float %cFF\n%res = OpConvertFToU %uint %x\n",     # NaN: undefined per spec
                     "%x = OpFNegate %float %f1\n%res = OpConvertFToU %uint %x\n",      # negative: out of range
                     "%x = OpExtInst %float %glsl FClamp %f1 %f2 %f1\n%res = OpBitcast %uint %x\n",  # lo > hi
                     "%res = OpSelect %uint %c1 %c1 %c2\n",                              # condition not a bool
                     "OpBranch %entry\n%res = OpCopyObject %uint %c1\n",                 # unconditional branch
                     "OpBranchConditional %t %entry %entry\n%res = OpCopyObject %uint %c1\n"):  # backward branch
            with self.subTest(body=body.splitlines()[0]):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_word(body)

    def test_store_to_push_constant_is_rejected(self):
        text = HEADER + "%p = OpAccessChain %_ptr_PC_uint %pc %c0\nOpStore %p %c1\n" + FOOTER
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(text)

    # Signalling-NaN words (exponent all ones, bit 22 clear, mantissa not zero), a quiet NaN and two ordinary
    # values. Issue #509: going through a Python float set bit 22 of the first four.
    SNAN_WORDS = (0xFFA3A095, 0x7F800001, 0x7FBFFFFF, 0xFF800001, 0x7FC00001, 0x3F800000, 0x80000000)

    def test_bitcast_keeps_signalling_nan_bits(self):
        # SPIR-V OpBitcast: the bit pattern is unchanged. The word goes uint -> float and on through
        # OpCopyObject, a composite, OpSelect and a Function variable before it comes back as a uint and also
        # leaves through a float Output; both must be the word that went in.
        text = HEADER.replace('OpEntryPoint Fragment %main "main" %out',
                              'OpEntryPoint Fragment %main "main" %out %outf').replace(
            "%main = OpFunction",
            "%_ptr_Output_float = OpTypePointer Output %float\n%outf = OpVariable %_ptr_Output_float Output\n"
            "%_ptr_Function_float = OpTypePointer Function %float\n%main = OpFunction")
        body = ("%fv = OpVariable %_ptr_Function_float Function\n"
                "%p = OpAccessChain %_ptr_PC_uint %pc %c0\n"
                "%w = OpLoad %uint %p\n"
                "%x = OpBitcast %float %w\n"
                "%y = OpCopyObject %float %x\n"
                "%v = OpCompositeConstruct %v2float %f1 %y\n"
                "%e = OpCompositeExtract %float %v 1\n"
                "%s = OpSelect %float %t %e %f2\n"
                "OpStore %fv %s\n"
                "%l = OpLoad %float %fv\n"
                "OpStore %outf %l\n"
                "%res = OpBitcast %uint %l\n"
                "OpStore %out %res\n")
        for word in self.SNAN_WORDS:
            with self.subTest(word=f"{word:#010x}"):
                out = si.run_fragment(text + body + FOOTER, push_constants=[word])
                self.assertEqual(out["%out"], (word,))
                self.assertEqual(out["%outf"], (word,))

    def test_bitcast_signalling_nan_is_still_a_nan(self):
        # The value keeps its bits but is still a float NaN to the float operations (OpIsNan, OpFOrdEqual).
        text = HEADER.replace("%main = OpFunction", "%cS = OpConstant %uint 4288913557\n%main = OpFunction")
        for body, want in (("%x = OpBitcast %float %cS\n%n = OpIsNan %bool %x\n", 1),
                           ("%x = OpBitcast %float %cS\n%n = OpFOrdEqual %bool %x %x\n", 0)):
            with self.subTest(body=body.splitlines()[1]):
                out = si.run_fragment(text + body + "%res = OpSelect %uint %n %c1 %c0\nOpStore %out %res\n"
                                      + FOOTER)
                self.assertEqual(out["%out"], (want,))

    def test_float_arithmetic_on_signalling_nan_gives_a_nan(self):
        # Arithmetic drops the kept pattern: SPIR-V does not fix the bits of a NaN result, so only "a NaN" is
        # asserted (exponent all ones, mantissa not zero).
        text = HEADER.replace("%main = OpFunction", "%cS = OpConstant %uint 4288913557\n%main = OpFunction")
        for op in ("%y = OpFAdd %float %x %f1\n", "%y = OpFMul %float %x %f2\n", "%y = OpFNegate %float %x\n"):
            with self.subTest(op=op.split()[2]):
                word = si.run_fragment(text + "%x = OpBitcast %float %cS\n" + op +
                                       "%res = OpBitcast %uint %y\nOpStore %out %res\n" + FOOTER)["%out"][0]
                self.assertEqual(word & 0x7F800000, 0x7F800000)
                self.assertNotEqual(word & 0x007FFFFF, 0)

    def test_push_constant_reads_zero(self):
        text = HEADER + "%p = OpAccessChain %_ptr_PC_uint %pc %c0\n%res = OpLoad %uint %p\nOpStore %out %res\n" + FOOTER
        self.assertEqual(si.run_fragment(text)["%out"][0], 0)

    def test_a_module_run_twice_gives_the_same_words_and_the_same_parse_error(self):
        # run_workgroup runs one generator per invocation over one text: each must see the same module.
        text = HEADER + "%res = OpIAdd %uint %c1 %c2\nOpStore %out %res\n" + FOOTER
        self.assertEqual(si.run_fragment(text)["%out"][0], 3)
        self.assertEqual(si.run_fragment(text)["%out"][0], 3)
        broken = HEADER + "%res = OpIAdd %uint %c1 %c2\nthis is not an instruction\n" + FOOTER
        for _ in range(2):
            with self.assertRaises(si.UnsupportedConstruct) as ctx:
                si.run_fragment(broken)
            self.assertIn("this is not an instruction", str(ctx.exception))


class AggregateValueTests(unittest.TestCase):
    HEADER = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
OpExecutionMode %main OriginUpperLeft
OpDecorate %out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%seven = OpConstant %uint 7
%array = OpTypeArray %uint %one
%nested = OpTypeArray %array %one
%null_array = OpConstantNull %array
%null_nested = OpConstantNull %nested
%const_array = OpConstantComposite %array %zero
%ptr_array = OpTypePointer Function %array
%ptr_nested = OpTypePointer Function %nested
%ptr_word = OpTypePointer Function %uint
%ptr_output = OpTypePointer Output %v2uint
%out = OpVariable %ptr_output Output
%main = OpFunction %void None %fn
%entry = OpLabel
"""

    def test_whole_array_store_does_not_alias_its_operand(self):
        body = """\
%src = OpVariable %ptr_array Function %null_array
%dst = OpVariable %ptr_array Function %null_array
%saved = OpLoad %array %src
OpStore %dst %saved
%element = OpAccessChain %ptr_word %dst %zero
OpStore %element %seven
%old = OpCompositeExtract %uint %saved 0
%new = OpLoad %uint %element
%result = OpCompositeConstruct %v2uint %old %new
OpStore %out %result
"""
        self.assertEqual(si.run_fragment(self.HEADER + body + FOOTER)["%out"], (0, 7))

    def test_nested_array_load_keeps_its_original_elements(self):
        body = """\
%src = OpVariable %ptr_nested Function %null_nested
%saved = OpLoad %nested %src
%element = OpAccessChain %ptr_word %src %zero %zero
OpStore %element %seven
%old = OpCompositeExtract %uint %saved 0 0
%new = OpLoad %uint %element
%result = OpCompositeConstruct %v2uint %old %new
OpStore %out %result
"""
        self.assertEqual(si.run_fragment(self.HEADER + body + FOOTER)["%out"], (0, 7))

    def test_shared_nested_initializer_creates_separate_storage(self):
        body = """\
%src = OpVariable %ptr_nested Function %null_nested
%other = OpVariable %ptr_nested Function %null_nested
%element = OpAccessChain %ptr_word %src %zero %zero
OpStore %element %seven
%independent = OpAccessChain %ptr_word %other %zero %zero
%old = OpLoad %uint %independent
%new = OpLoad %uint %element
%result = OpCompositeConstruct %v2uint %old %new
OpStore %out %result
"""
        self.assertEqual(si.run_fragment(self.HEADER + body + FOOTER)["%out"], (0, 7))

    def test_array_constant_store_preserves_the_constant(self):
        body = """\
%dst = OpVariable %ptr_array Function
OpStore %dst %const_array
%element = OpAccessChain %ptr_word %dst %zero
OpStore %element %seven
%old = OpCompositeExtract %uint %const_array 0
%new = OpLoad %uint %element
%result = OpCompositeConstruct %v2uint %old %new
OpStore %out %result
"""
        self.assertEqual(si.run_fragment(self.HEADER + body + FOOTER)["%out"], (0, 7))


if __name__ == "__main__":
    unittest.main()
