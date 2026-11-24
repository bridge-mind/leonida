"""Grammar-table integrity and binary-encoding round-trip tests for tools/spirv_asm_tables.py."""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import spirv_asm  # noqa: E402
from spirv_asm_tables import BIT_ENUMS, GLSL_STD_450, INSTRUCTIONS, VALUE_ENUMS  # noqa: E402


class InstructionsTableTests(unittest.TestCase):
    def test_schema_and_non_empty(self):
        self.assertGreater(len(INSTRUCTIONS), 300)
        for name, spec in INSTRUCTIONS.items():
            self.assertIsInstance(name, str)
            self.assertTrue(name.startswith("Op"), name)
            self.assertIsInstance(spec, tuple, name)
            self.assertEqual(len(spec), 4, name)
            opcode, has_type, has_result, operands = spec
            self.assertIsInstance(opcode, int, name)
            self.assertTrue(0 <= opcode <= 0xFFFF, f"{name}: {opcode}")
            self.assertIsInstance(has_type, bool, name)
            self.assertIsInstance(has_result, bool, name)
            self.assertIsInstance(operands, tuple, name)
            for op_spec in operands:
                self.assertIsInstance(op_spec, tuple, name)
                self.assertEqual(len(op_spec), 2, name)
                kind, quant = op_spec
                self.assertIsInstance(kind, str, name)
                self.assertIn(quant, ("", "?", "*"), f"{name}: {op_spec}")

    def test_core_opcodes_and_flags(self):
        expected = {
            "OpNop": (0, False, False),
            "OpExtInst": (12, True, True),
            "OpTypeInt": (21, False, True),
            "OpTypeFloat": (22, False, True),
            "OpConstant": (43, True, True),
            "OpImageSampleExplicitLod": (88, True, True),
            "OpImageSampleDrefExplicitLod": (90, True, True),
            "OpImageDrefGather": (97, True, True),
        }
        for name, (opc, ht, hr) in expected.items():
            with self.subTest(op=name):
                got_opc, got_ht, got_hr, _ = INSTRUCTIONS[name]
                self.assertEqual((got_opc, got_ht, got_hr), (opc, ht, hr))


class GlslExtTableTests(unittest.TestCase):
    def test_schema_and_unique_opcodes(self):
        self.assertGreater(len(GLSL_STD_450), 50)
        seen = {}
        for name, opc in GLSL_STD_450.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(opc, int)
            self.assertGreater(opc, 0)
            self.assertNotIn(opc, seen, f"duplicate GLSL opcode {opc}: {name} vs {seen.get(opc)}")
            seen[opc] = name

    def test_expected_glsl_opcodes(self):
        expected = {
            "RoundEven": 2,
            "Trunc": 3,
            "FAbs": 4,
            "Floor": 8,
            "Ceil": 9,
            "Fract": 10,
            "Sqrt": 31,
            "InverseSqrt": 32,
            "FMin": 37,
            "FMax": 40,
            "FClamp": 43,
            "Fma": 50,
            "PackHalf2x16": 58,
            "UnpackHalf2x16": 62,
        }
        for name, opc in expected.items():
            with self.subTest(fn=name):
                self.assertEqual(GLSL_STD_450[name], opc)


class EnumsTableTests(unittest.TestCase):
    def test_value_enums_schema(self):
        self.assertGreater(len(VALUE_ENUMS), 10)
        for enum_name, members in VALUE_ENUMS.items():
            self.assertIsInstance(enum_name, str)
            self.assertIsInstance(members, dict)
            self.assertGreater(len(members), 0, enum_name)
            for k, spec in members.items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(spec, tuple)
                self.assertEqual(len(spec), 2)
                val, extra = spec
                self.assertIsInstance(val, int)
                self.assertIsInstance(extra, tuple)

    def test_bit_enums_power_of_two_masks(self):
        self.assertGreater(len(BIT_ENUMS), 5)
        for enum_name, members in BIT_ENUMS.items():
            self.assertIsInstance(enum_name, str)
            self.assertIsInstance(members, dict)
            for k, spec in members.items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(spec, tuple)
                self.assertEqual(len(spec), 2)
                mask, extra = spec
                self.assertIsInstance(mask, int)
                self.assertGreaterEqual(mask, 0)
                if mask != 0:
                    self.assertEqual(mask & (mask - 1), 0, f"{enum_name}.{k}: {mask:#x}")
                self.assertIsInstance(extra, tuple)

    def test_image_operands_bit_flags_and_order(self):
        img_ops = BIT_ENUMS["ImageOperands"]
        expected = {
            "None": (0, ()),
            "Bias": (1, ("IdRef",)),
            "Lod": (2, ("IdRef",)),
            "Grad": (4, ("IdRef", "IdRef")),
            "ConstOffset": (8, ("IdRef",)),
            "Offset": (16, ("IdRef",)),
            "ConstOffsets": (32, ("IdRef",)),
            "Sample": (64, ("IdRef",)),
            "MinLod": (128, ("IdRef",)),
            "Offsets": (65536, ("IdRef",)),
        }
        for name, want in expected.items():
            with self.subTest(flag=name):
                self.assertEqual(img_ops[name], want)
        self.assertLess(img_ops["Grad"][0], img_ops["Offset"][0])
        self.assertLess(img_ops["Offset"][0], img_ops["MinLod"][0])


# Every instruction alias in SPIRV-Headers' spirv.core.grammar.json (each instruction's "aliases"), alias -> opname.
# spirv-as accepts either spelling and writes the same words for both.
INSTRUCTION_ALIASES = {
        "OpArbitraryFloatAddINTEL": "OpArbitraryFloatAddALTERA",
        "OpArbitraryFloatCastFromIntINTEL": "OpArbitraryFloatCastFromIntALTERA",
        "OpArbitraryFloatCastINTEL": "OpArbitraryFloatCastALTERA",
        "OpArbitraryFloatCastToIntINTEL": "OpArbitraryFloatCastToIntALTERA",
        "OpArbitraryFloatCbrtINTEL": "OpArbitraryFloatCbrtALTERA",
        "OpArbitraryFloatDivINTEL": "OpArbitraryFloatDivALTERA", "OpArbitraryFloatEQINTEL": "OpArbitraryFloatEQALTERA",
        "OpArbitraryFloatGEINTEL": "OpArbitraryFloatGEALTERA", "OpArbitraryFloatGTINTEL": "OpArbitraryFloatGTALTERA",
        "OpArbitraryFloatHypotINTEL": "OpArbitraryFloatHypotALTERA",
        "OpArbitraryFloatLEINTEL": "OpArbitraryFloatLEALTERA", "OpArbitraryFloatLTINTEL": "OpArbitraryFloatLTALTERA",
        "OpArbitraryFloatMulINTEL": "OpArbitraryFloatMulALTERA",
        "OpArbitraryFloatRSqrtINTEL": "OpArbitraryFloatRSqrtALTERA",
        "OpArbitraryFloatRecipINTEL": "OpArbitraryFloatRecipALTERA",
        "OpArbitraryFloatSinCosPiINTEL": "OpArbitraryFloatSinCosPiALTERA",
        "OpArbitraryFloatSqrtINTEL": "OpArbitraryFloatSqrtALTERA",
        "OpArbitraryFloatSubINTEL": "OpArbitraryFloatSubALTERA",
        "OpControlBarrierArriveINTEL": "OpControlBarrierArriveEXT",
        "OpControlBarrierWaitINTEL": "OpControlBarrierWaitEXT",
        "OpCooperativeMatrixConvertNV": "OpCooperativeMatrixConvertUseEXT",
        "OpCooperativeMatrixPerElementOpNV": "OpCooperativeMatrixPerElementOpEXT",
        "OpCooperativeMatrixReduceNV": "OpCooperativeMatrixReduceEXT",
        "OpCrossWorkgroupCastToPtrINTEL": "OpCrossWorkgroupCastToPtrALTERA",
        "OpDecorateStringGOOGLE": "OpDecorateString", "OpDemoteToHelperInvocationEXT": "OpDemoteToHelperInvocation",
        "OpFPGARegINTEL": "OpFPGARegALTERA", "OpFixedCosINTEL": "OpFixedCosALTERA",
        "OpFixedCosPiINTEL": "OpFixedCosPiALTERA", "OpFixedExpINTEL": "OpFixedExpALTERA",
        "OpFixedLogINTEL": "OpFixedLogALTERA", "OpFixedRecipINTEL": "OpFixedRecipALTERA",
        "OpFixedRsqrtINTEL": "OpFixedRsqrtALTERA", "OpFixedSinCosINTEL": "OpFixedSinCosALTERA",
        "OpFixedSinCosPiINTEL": "OpFixedSinCosPiALTERA", "OpFixedSinINTEL": "OpFixedSinALTERA",
        "OpFixedSinPiINTEL": "OpFixedSinPiALTERA", "OpFixedSqrtINTEL": "OpFixedSqrtALTERA",
        "OpGroupNonUniformPartitionNV": "OpGroupNonUniformPartitionEXT",
        "OpMemberDecorateStringGOOGLE": "OpMemberDecorateString",
        "OpPtrCastToCrossWorkgroupINTEL": "OpPtrCastToCrossWorkgroupALTERA",
        "OpRayQueryGetClusterIdNV": "OpRayQueryGetIntersectionClusterIdNV",
        "OpReadPipeBlockingINTEL": "OpReadPipeBlockingALTERA", "OpReportIntersectionNV": "OpReportIntersectionKHR",
        "OpSDotAccSatKHR": "OpSDotAccSat", "OpSDotKHR": "OpSDot", "OpSUDotAccSatKHR": "OpSUDotAccSat",
        "OpSUDotKHR": "OpSUDot", "OpTaskSequenceAsyncINTEL": "OpTaskSequenceAsyncALTERA",
        "OpTaskSequenceCreateINTEL": "OpTaskSequenceCreateALTERA", "OpTaskSequenceGetINTEL": "OpTaskSequenceGetALTERA",
        "OpTaskSequenceReleaseINTEL": "OpTaskSequenceReleaseALTERA",
        "OpTypeAccelerationStructureNV": "OpTypeAccelerationStructureKHR",
        "OpTypeCooperativeVectorNV": "OpTypeVectorIdEXT", "OpTypeTaskSequenceINTEL": "OpTypeTaskSequenceALTERA",
        "OpUDotAccSatKHR": "OpUDotAccSat", "OpUDotKHR": "OpUDot",
        "OpWritePipeBlockingINTEL": "OpWritePipeBlockingALTERA",
}

ALIAS_MODULE = """
OpCapability Shader
OpCapability DemoteToHelperInvocation
OpCapability DotProduct
OpCapability DotProductInput4x8BitPacked
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_user_type"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %target
OpExecutionMode %main OriginUpperLeft
OpDecorate %target Location 0
OpDecorateStringGOOGLE %target UserSemantic "SV_Target0"
OpMemberDecorateStringGOOGLE %pair 0 UserSemantic "PAIR0"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%pair = OpTypeStruct %int
%out_int = OpTypePointer Output %int
%target = OpVariable %out_int Output
%a = OpConstant %uint 16909060
%b = OpConstant %uint 84281096
%main = OpFunction %void None %fn
%entry = OpLabel
%dot = OpSDotKHR %int %a %b PackedVectorFormat4x8Bit
OpStore %target %dot
OpDemoteToHelperInvocationEXT
OpReturn
OpFunctionEnd
"""


class InstructionAliasTests(unittest.TestCase):
    def test_every_alias_has_its_canonical_entry(self):
        for alias, canonical in INSTRUCTION_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIn(alias, INSTRUCTIONS)
                self.assertEqual(INSTRUCTIONS[alias], INSTRUCTIONS[canonical])

    def test_alias_spellings_assemble_like_canonical_ones(self):
        # The same module with each alias opname replaced by the instruction it names.
        canonical = "\n".join(" ".join(INSTRUCTION_ALIASES.get(token, token) for token in line.split(" "))
                              for line in ALIAS_MODULE.splitlines())
        for alias, opname in (("OpDecorateStringGOOGLE", "OpDecorateString"),
                              ("OpMemberDecorateStringGOOGLE", "OpMemberDecorateString"),
                              ("OpSDotKHR", "OpSDot"),
                              ("OpDemoteToHelperInvocationEXT", "OpDemoteToHelperInvocation")):
            self.assertIn(alias, ALIAS_MODULE.split())
            self.assertNotIn(alias, canonical.split())
            self.assertIn(opname, canonical.split())
        self.assertEqual(spirv_asm.assemble(ALIAS_MODULE, target_env="vulkan1.3"),
                         spirv_asm.assemble(canonical, target_env="vulkan1.3"))


class AssembleRoundTripTests(unittest.TestCase):
    def test_assemble_module_matches_table_opcodes(self):
        asm = "\n".join(
            [
                "OpCapability Shader",
                '%glsl = OpExtInstImport "GLSL.std.450"',
                "OpMemoryModel Logical GLSL450",
                'OpEntryPoint Fragment %main "main"',
                "OpExecutionMode %main OriginUpperLeft",
                "%void = OpTypeVoid",
                "%fn_t = OpTypeFunction %void",
                "%float = OpTypeFloat 32",
                "%c_neg = OpConstant %float -2.5",
                "%main = OpFunction %void None %fn_t",
                "%entry = OpLabel",
                "%res = OpExtInst %float %glsl FAbs %c_neg",
                "OpReturn",
                "OpFunctionEnd",
            ]
        )
        spv = spirv_asm.assemble(asm, target_env="vulkan1.2")
        self.assertEqual(len(spv) % 4, 0)
        words = struct.unpack(f"<{len(spv) // 4}I", spv)
        self.assertEqual(words[0], spirv_asm.SPIRV_MAGIC)
        self.assertEqual(words[1], spirv_asm.TARGET_ENV_VERSION["vulkan1.2"])
        self.assertGreater(words[3], 1)
        self.assertEqual(words[4], 0)

        decoded = []
        idx = 5
        while idx < len(words):
            header = words[idx]
            wc = header >> 16
            opc = header & 0xFFFF
            self.assertGreater(wc, 0)
            decoded.append((opc, words[idx : idx + wc]))
            idx += wc
        self.assertEqual(idx, len(words))

        expected_opcodes = [
            INSTRUCTIONS["OpCapability"][0],
            INSTRUCTIONS["OpExtInstImport"][0],
            INSTRUCTIONS["OpMemoryModel"][0],
            INSTRUCTIONS["OpEntryPoint"][0],
            INSTRUCTIONS["OpExecutionMode"][0],
            INSTRUCTIONS["OpTypeVoid"][0],
            INSTRUCTIONS["OpTypeFunction"][0],
            INSTRUCTIONS["OpTypeFloat"][0],
            INSTRUCTIONS["OpConstant"][0],
            INSTRUCTIONS["OpFunction"][0],
            INSTRUCTIONS["OpLabel"][0],
            INSTRUCTIONS["OpExtInst"][0],
            INSTRUCTIONS["OpReturn"][0],
            INSTRUCTIONS["OpFunctionEnd"][0],
        ]
        self.assertEqual([opc for opc, _ in decoded], expected_opcodes)
        ext_inst_words = decoded[11][1]
        self.assertEqual(ext_inst_words[4], GLSL_STD_450["FAbs"])


if __name__ == "__main__":
    unittest.main()
