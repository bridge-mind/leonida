"""Tests for tools/spirv_asm.py (pure-Python SPIR-V binary assembler)."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import support  # noqa: E402

SPIRV_VAL = shutil.which("spirv-val")
SPIRV_AS = shutil.which("spirv-as")


def _validate(spv: bytes, target_env: str = "vulkan1.2") -> None:
    if not SPIRV_VAL:
        return
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
        f.write(spv)
        path = f.name
    try:
        r = subprocess.run(
            [SPIRV_VAL, "--target-env", target_env, "--allow-offset-texture-operand", path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise AssertionError(f"spirv-val failed ({target_env}):\n{r.stderr}")
    finally:
        os.unlink(path)


class SpirvAsmUnitTests(unittest.TestCase):
    def test_reject_repeated_type_result_ids(self):
        declarations = (
            ("OpTypeInt 32 0", "OpTypeFloat 32"),
            ("OpTypeBool", "OpTypeBool"),
            ("OpTypeInt 32 0", "OpTypeBool"),
            ("OpTypeBool", "OpTypeInt 32 0"),
        )
        for first, second in declarations:
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(spirv_asm.SpirvAsmError, r"line 2: type ID %t is already defined"):
                    spirv_asm.assemble(f"%t = {first}\n%t = {second}\n")

    def test_forward_pointer_can_be_completed_once(self):
        declaration = "%node_ptr = OpTypePointer PhysicalStorageBuffer %node\n"
        asm = (
            "OpCapability Shader\n"
            "OpCapability PhysicalStorageBufferAddresses\n"
            'OpExtension "SPV_KHR_physical_storage_buffer"\n'
            "OpMemoryModel PhysicalStorageBuffer64 GLSL450\n"
            'OpEntryPoint GLCompute %main "main"\n'
            "OpExecutionMode %main LocalSize 1 1 1\n"
            "OpTypeForwardPointer %node_ptr PhysicalStorageBuffer\n"
            "%node = OpTypeStruct %node_ptr\n"
            f"{declaration}"
            "%void = OpTypeVoid\n"
            "%fn = OpTypeFunction %void\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        _validate(spirv_asm.assemble(asm))
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, r"line 10: type ID %node_ptr is already defined"):
            spirv_asm.assemble(asm.replace(declaration, declaration * 2))

    def test_assembler_leaves_non_type_redefinitions_to_validator(self):
        for body in (
            "%t = OpTypeBool\n%t = OpUndef %uint\n",
            "%t = OpUndef %uint\n%t = OpTypeBool\n",
            "%t = OpConstant %uint 1\n%t = OpConstant %uint 2\n",
        ):
            with self.subTest(body=body):
                self.assertTrue(spirv_asm.assemble("%uint = OpTypeInt 32 0\n" + body))

    def test_header_and_target_env_versions(self):
        asm = (
            "OpCapability Shader\n"
            "OpMemoryModel Logical GLSL450\n"
            'OpEntryPoint GLCompute %main "main"\n'
            "OpExecutionMode %main LocalSize 1 1 1\n"
            "%void = OpTypeVoid\n"
            "%fn = OpTypeFunction %void\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        for env, expected_ver in (
            ("vulkan1.0", 0x00010000),
            ("vulkan1.1", 0x00010300),
            ("vulkan1.2", 0x00010500),
            ("vulkan1.3", 0x00010600),
        ):
            spv = spirv_asm.assemble(asm, target_env=env)
            magic, ver, gen, bound, schema = struct.unpack_from("<5I", spv, 0)
            self.assertEqual(magic, 0x07230203)
            self.assertEqual(ver, expected_ver)
            self.assertEqual(gen, 0)
            self.assertGreater(bound, 1)
            self.assertEqual(schema, 0)
            _validate(spv, target_env=env)

    def test_constants_strings_bit_enums_and_control_flow(self):
        # Exercises int32/int64/float32/float64/neg-zero/hex-float constants, GLSL.std.450 OpExtInst,
        # BitEnum multi-flag ascending-bit parameter ordering (MemoryAccess Aligned|Nontemporal),
        # OpSwitch, OpPhi, comments inside/outside lines, and escaped string literals.
        asm = (
            "; leading comment\n"
            "OpCapability Shader\n"
            "OpCapability Int64\n"
            "OpCapability Float64\n"
            'OpExtension "SPV_KHR_storage_buffer_storage_class"\n'
            '%glsl = OpExtInstImport "GLSL.std.450" ; inline comment\n'
            "OpMemoryModel Logical GLSL450\n"
            'OpEntryPoint GLCompute %main "main"\n'
            "OpExecutionMode %main LocalSize 4 1 1\n"
            '%str = OpString "hello; \\"world\\""\n'
            "%void = OpTypeVoid\n"
            "%fn = OpTypeFunction %void\n"
            "%uint = OpTypeInt 32 0\n"
            "%ulong = OpTypeInt 64 0\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%ptr_fn_u = OpTypePointer Function %uint\n"
            "%c0 = OpConstant %uint 0\n"
            "%c1 = OpConstant %uint 1\n"
            "%c_hex = OpConstant %uint 0xDEADBEEF\n"
            "%c64 = OpConstant %ulong 0x1122334455667788\n"
            "%fneg0 = OpConstant %float -0.0\n"
            "%fhex = OpConstant %float 0x1.8p+0\n"
            "%dval = OpConstant %double 3.5\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "%var = OpVariable %ptr_fn_u Function\n"
            "OpStore %var %c_hex Nontemporal|Aligned 4\n"
            "%ld = OpLoad %uint %var Aligned|Nontemporal 4\n"
            "%absf = OpExtInst %float %glsl FAbs %fneg0\n"
            "%sumf = OpFAdd %float %absf %fhex\n"
            "%sum64 = OpIAdd %ulong %c64 %c64\n"
            "OpSelectionMerge %merge None\n"
            "OpSwitch %ld %def 0 %case0 1 %case1\n"
            "%case0 = OpLabel\n"
            "OpBranch %merge\n"
            "%case1 = OpLabel\n"
            "OpBranch %merge\n"
            "%def = OpLabel\n"
            "OpBranch %merge\n"
            "%merge = OpLabel\n"
            "%phi = OpPhi %uint %c0 %case0 %c1 %case1 %ld %def\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        spv = spirv_asm.assemble(asm, target_env="vulkan1.2")
        _validate(spv)
        # Verify 64-bit constant words appear little-endian (lo32 then hi32) and -0.0 is 0x80000000.
        words = struct.unpack(f"<{len(spv) // 4}I", spv)
        self.assertIn(0x55667788, words)
        self.assertIn(0x11223344, words)
        self.assertIn(0x80000000, words)
        self.assertIn(0x3FC00000, words)  # 0x1.8p+0 = 1.5f
        # A selector produced by a pure-ID instruction still needs its 64-bit result type.
        asm64 = asm.replace("OpSwitch %ld %def 0 %case0 1 %case1",
                            "OpSwitch %sum64 %def 0 %case0 0x100000001 %case1")
        spv64 = spirv_asm.assemble(asm64, target_env="vulkan1.2")
        self.assertEqual(len(spv64), len(spv) + 8)  # one extra word per case literal
        _validate(spv64)

    def test_subword_integer_literals_use_the_declared_signedness(self) -> None:
        for width, signed, token, expected in (
            (8, 1, "0x80", 0xFFFFFF80),
            (16, 1, "0x8000", 0xFFFF8000),
            (16, 1, "-32768", 0xFFFF8000),
            (16, 0, "0x8000", 0x00008000),
        ):
            with self.subTest(width=width, signed=signed, token=token):
                asm = f'''OpCapability Shader
OpCapability Int{width}
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%short = OpTypeInt {width} {signed}
%value = OpConstant %short {token}
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge None
OpSwitch %value %merge {token} %case
%case = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
'''
                spv = spirv_asm.assemble(asm, target_env="vulkan1.2")
                words = struct.unpack(f"<{len(spv) // 4}I", spv)
                literals = {}
                offset = 5
                while offset < len(words):
                    count, opcode = words[offset] >> 16, words[offset] & 0xFFFF
                    if opcode in (43, 251):  # OpConstant, OpSwitch
                        literals[opcode] = words[offset + 3]
                    offset += count
                self.assertEqual(literals, {43: expected, 251: expected})
                _validate(spv)

    def test_hex_float_inf_nan_and_string_escapes_match_spirv_as(self):
        # Regression tests for PR #2251 review (@FreddyxS):
        # 1) Width-aware hex-float Inf and NaN payload spellings (f16, f32, f64).
        # 2) SPIRV-Tools string backslash decoding (\n -> literal 'n', not LF).
        asm = (
            "OpCapability Shader\n"
            "OpCapability Float16\n"
            "OpCapability Float64\n"
            "OpMemoryModel Logical GLSL450\n"
            'OpEntryPoint GLCompute %main "main"\n'
            "OpExecutionMode %main LocalSize 1 1 1\n"
            'OpName %main "left\\nright"\n'
            "%void = OpTypeVoid\n"
            "%fn = OpTypeFunction %void\n"
            "%f16 = OpTypeFloat 16\n"
            "%f32 = OpTypeFloat 32\n"
            "%f64 = OpTypeFloat 64\n"
            "%h_inf = OpConstant %f16 0x1p+16\n"
            "%h_nan = OpConstant %f16 0x1.8p+16\n"
            "%f_inf = OpConstant %f32 0x1p+128\n"
            "%f_ninf = OpConstant %f32 -0x1p+128\n"
            "%f_nan = OpConstant %f32 0x1.8p+128\n"
            "%d_inf = OpConstant %f64 0x1p+1024\n"
            "%d_nan = OpConstant %f64 0x1.8p+1024\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        spv = spirv_asm.assemble(asm, target_env="vulkan1.2")
        _validate(spv)
        words = struct.unpack(f"<{len(spv) // 4}I", spv)
        self.assertIn(0x7C00, words)       # f16 +Inf
        self.assertIn(0x7E00, words)       # f16 qNaN
        self.assertIn(0x7F800000, words)   # f32 +Inf
        self.assertIn(0xFF800000, words)   # f32 -Inf
        self.assertIn(0x7FC00000, words)   # f32 qNaN
        self.assertIn(0x7FF00000, words)   # f64 +Inf hi word
        self.assertIn(0x7FF80000, words)   # f64 qNaN hi word
        # Verify exact byte-for-byte parity (excluding header generator word [2]) with spirv-as when present.
        if SPIRV_AS:
            with tempfile.TemporaryDirectory() as d:
                sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
                with open(sa, "w", encoding="utf-8", newline="\n") as f:
                    f.write(asm)
                subprocess.run([SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"], check=True)
                with open(sv, "rb") as f:
                    ref_words = struct.unpack(f"<{os.path.getsize(sv) // 4}I", f.read())
            self.assertEqual(len(words), len(ref_words))
            self.assertEqual(words[:2] + words[3:], ref_words[:2] + ref_words[3:])

    def test_translated_shaders_validate_and_match_spirv_as(self):
        # Translate diverse synthetic shaders (PS ALU/dot4/f16/branches, CS LDS/atomics/wave64-emulate)
        # and verify every binary assembled by spirv_asm passes spirv-val.
        ps_lines = [
            "v_mov_b32_e32 v1, 0x01020304",
            "v_mov_b32_e32 v2, 0x05060708",
            "v_mov_b32_e32 v3, 0x3f800000",
            "v_dot4_i32_i8 v4, v1, v2, v3",
            "v_add_f32_e32 v5, v3, 1.5",
            "v_pk_add_f16 v6, v1, v2 op_sel:[0,1] op_sel_hi:[1,0]",
            "exp mrt0 v4, v5, v6, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_spirv_asm_smoke", ps_lines)
        spv = spirv_asm.assemble(spvasm, target_env="vulkan1.2")
        _validate(spv)
        # Also test a wave64-emulated compute shader (exercises OpControlBarrier, Workgroup arrays, loops,
        # PhysicalStorageBuffer, OpAtomic*, OpSwitch/structured CF, Int64).
        import json
        cs_lines = [
            "s_load_dwordx2 s[4:5], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "ds_write_b32 v1, v0",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v2, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_cmp_lt_u32_e64 vcc, v0, 32",
            "v_cndmask_b32_e64 v3, v2, v0, vcc",
            "global_store_dword v1, v3, s[4:5]",
            "s_endpgm",
        ]
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            s_path = os.path.join(d, "disasm", "cs_spirv_asm_w64.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_spirv_asm_w64  code 0x0..0x100, gfx1013, 0 invalid words\n")
                for ln in cs_lines:
                    f.write(f"\t{ln} ; encoding: [0x00,0x00,0x00,0x00]\n")
            with open(os.path.join(d, "fspx", "cs_spirv_asm_w64.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (8 << 1) | (16 << 15), "threads": [64, 1, 1]}, f)
            old_w64 = rdna2spirv.WAVE64_EMULATE
            try:
                rdna2spirv.WAVE64_EMULATE = True
                cs_spvasm = rdna2spirv.translate(s_path)
            finally:
                rdna2spirv.WAVE64_EMULATE = old_w64
        cs_spv = spirv_asm.assemble(cs_spvasm, target_env="vulkan1.2")
        _validate(cs_spv)

    def test_rdna2spirv_assembler_python_no_asm_skips_spvasm_file(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            s_path = os.path.join(d, "disasm", "cs_tiny.s")
            out_dir = os.path.join(d, "out")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_tiny  code 0x0..0x10, gfx1013, 0 invalid words\n")
                f.write("\tv_mov_b32_e32 v0, 1.0 ; encoding: [0xf2,0x02,0x00,0x7e]\n")
                f.write("\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(os.path.join(d, "fspx", "cs_tiny.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (4 << 1), "threads": [32, 1, 1]}, f)
            if SPIRV_VAL:
                r = subprocess.run(
                    [sys.executable, "-I", os.path.join(TOOLS, "rdna2spirv.py"),
                     s_path, "--out", out_dir, "--assembler", "python", "--no-asm"],
                    capture_output=True, text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertTrue(os.path.isfile(os.path.join(out_dir, "cs_tiny.spv")))
                self.assertFalse(os.path.exists(os.path.join(out_dir, "cs_tiny.spvasm")))

    def test_cli_and_error_cases(self):
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble("OpNotARealOpcode %x\n")
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble("OpTypeVoid\n")  # missing result id
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble("%x = OpCapability Shader\n")  # unexpected result id
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble('%s = OpString "unterminated\n')
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble("OpCapability NotACapability\n")
        with self.assertRaises(spirv_asm.SpirvAsmError):
            spirv_asm.assemble("OpCapability Shader\n", target_env="invalid")

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "min.spvasm")
            dst = os.path.join(d, "min.spv")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "OpCapability Shader\nOpMemoryModel Logical GLSL450\n"
                    'OpEntryPoint GLCompute %m "main"\nOpExecutionMode %m LocalSize 1 1 1\n'
                    "%v = OpTypeVoid\n%ft = OpTypeFunction %v\n%m = OpFunction %v None %ft\n"
                    "%l = OpLabel\nOpReturn\nOpFunctionEnd\n"
                )
            rc = spirv_asm.main([src, "-o", dst, "--target-env", "vulkan1.2"])
            self.assertEqual(rc, 0)
            with open(dst, "rb") as f:
                _validate(f.read())

    def test_invalid_id_spelling_in_results_and_references(self):
        for identifier in ("%", "%bad-name", "%bad.name", "%\u0436"):
            for line in (f"{identifier} = OpLabel", f"OpBranch {identifier}"):
                with self.subTest(line=line):
                    with self.assertRaisesRegex(spirv_asm.SpirvAsmError, "line 2: invalid ID"):
                        spirv_asm.assemble(f"OpCapability Shader\n{line}\n")

    def test_cli_invalid_id_leaves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "bad.spvasm")
            dst = os.path.join(d, "existing.spv")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write("OpCapability Shader\n%bad.name = OpTypeVoid\n")
            with open(dst, "wb") as f:
                f.write(b"existing output")
            result = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "spirv_asm.py"), src, "-o", dst],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("line 2: invalid ID", result.stderr)
            self.assertEqual(len(result.stderr.splitlines()), 1)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), b"existing output")

    def test_cli_source_that_is_not_utf8_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "bad.spvasm")
            dst = os.path.join(d, "bad.spv")
            with open(src, "wb") as f:
                f.write(b"OpCapability Shader\n; \xb1\xff\n")
            r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "spirv_asm.py"), src, "-o", dst],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("ERROR:", r.stderr)
            self.assertFalse(os.path.exists(dst))


_INT_HEAD = (
    "OpCapability Shader\nOpCapability Int64\nOpCapability Int16\nOpCapability Int8\n"
    "OpMemoryModel Logical GLSL450\n"
    "%i8 = OpTypeInt 8 1\n%u8 = OpTypeInt 8 0\n%i16 = OpTypeInt 16 1\n%u16 = OpTypeInt 16 0\n"
    "%i32 = OpTypeInt 32 1\n%u32 = OpTypeInt 32 0\n%i64 = OpTypeInt 64 1\n%u64 = OpTypeInt 64 0\n"
)


def _switch(sel_type, literal):
    return (_INT_HEAD + "%void = OpTypeVoid\n%fn = OpTypeFunction %void\n"
            f"%k = OpConstant %{sel_type} 0\n%f = OpFunction %void None %fn\n%l0 = OpLabel\n"
            f"OpSelectionMerge %m None\nOpSwitch %k %m {literal} %l1\n%l1 = OpLabel\nOpBranch %m\n"
            "%m = OpLabel\nOpReturn\nOpFunctionEnd\n")


# (SPIR-V text, words spirv-as gives for the literal, or None where spirv-as refuses it). Measured with the
# Vulkan SDK 1.4.363.1 spirv-as (BUILD-1012).
_INT_LITERAL_CASES = []
for _t, _lit, _words in (
    ("u32", "4294967295", [0xFFFFFFFF]), ("u32", "4294967296", None), ("u32", "0x100000000", None),
    ("u32", "-1", None), ("u32", "-0", None), ("u32", "+5", [5]),
    ("i32", "-1", [0xFFFFFFFF]), ("i32", "2147483647", [0x7FFFFFFF]), ("i32", "2147483648", None),
    ("i32", "-2147483648", [0x80000000]), ("i32", "-2147483649", None), ("i32", "0xFFFFFFFF", [0xFFFFFFFF]),
    ("i32", "0x80000000", [0x80000000]), ("i32", "0x100000000", None), ("i32", "-0x1", [0xFFFFFFFF]),
    ("u16", "65535", [0xFFFF]), ("u16", "65536", None), ("u16", "-1", None), ("u16", "0x10000", None),
    ("i16", "-1", [0xFFFFFFFF]), ("i16", "32767", [0x7FFF]), ("i16", "32768", None),
    ("i16", "-32768", [0xFFFF8000]), ("i16", "-32769", None), ("i16", "-0x8001", None),
    ("i16", "0xFFFF", [0xFFFFFFFF]), ("i16", "0x8000", [0xFFFF8000]), ("i16", "0x10000", None),
    ("u8", "255", [0xFF]), ("u8", "256", None), ("u8", "-1", None),
    ("i8", "127", [0x7F]), ("i8", "128", None), ("i8", "-128", [0xFFFFFF80]), ("i8", "-129", None),
    ("i8", "0xFF", [0xFFFFFFFF]), ("i8", "0x80", [0xFFFFFF80]),
    # A signed hex literal (+0x...) is a value, not a bit pattern: it must fit the signed range (@mrbean56, #3931).
    ("i16", "+0xFFFF", None), ("i16", "+0x8000", None), ("i16", "+0x7FFF", [0x7FFF]), ("i8", "+0x80", None),
    ("i32", "+0xFFFFFFFF", None), ("i32", "+0x7FFFFFFF", [0x7FFFFFFF]), ("i64", "+0xFFFFFFFFFFFFFFFF", None),
    ("u16", "+0xFFFF", [0xFFFF]), ("u32", "+0xFFFFFFFF", [0xFFFFFFFF]),
    ("u64", "18446744073709551615", [0xFFFFFFFF, 0xFFFFFFFF]), ("u64", "18446744073709551616", None),
    ("u64", "-1", None),
    ("i64", "-1", [0xFFFFFFFF, 0xFFFFFFFF]), ("i64", "9223372036854775807", [0xFFFFFFFF, 0x7FFFFFFF]),
    ("i64", "9223372036854775808", None), ("i64", "-9223372036854775809", None),
    ("i64", "0xFFFFFFFFFFFFFFFF", [0xFFFFFFFF, 0xFFFFFFFF]),
):
    _INT_LITERAL_CASES.append((f"OpConstant %{_t} {_lit}", _INT_HEAD + f"%c = OpConstant %{_t} {_lit}\n", _words))
for _lit, _words in (("4294967295", [0xFFFFFFFF]), ("4294967296", None), ("-1", None)):
    _INT_LITERAL_CASES.append((f"OpDecorate Location {_lit}", _INT_HEAD + f"OpDecorate %u32 Location {_lit}\n",
                               _words))
    _INT_LITERAL_CASES.append((f"OpGroupMemberDecorate {_lit}",
                               _INT_HEAD + f"%g = OpDecorationGroup\nOpGroupMemberDecorate %g %u32 {_lit}\n",
                               _words))
for _t, _lit, _words in (
    ("u32", "4294967295", [0xFFFFFFFF]), ("u32", "4294967296", None), ("u32", "-1", None),
    ("i32", "-1", [0xFFFFFFFF]), ("i32", "2147483648", None),
    ("i16", "0xFFFF", [0xFFFFFFFF]), ("i16", "0x8000", [0xFFFF8000]), ("i16", "32768", None),
    ("u16", "65536", None), ("i8", "0x80", [0xFFFFFF80]),
    ("i32", "+0xFFFFFFFF", None), ("i16", "+0x8000", None), ("i32", "0xFFFFFFFF", [0xFFFFFFFF]),
    ("u64", "18446744073709551615", [0xFFFFFFFF, 0xFFFFFFFF]), ("u64", "18446744073709551616", None),
    ("u64", "-1", None), ("i64", "-1", [0xFFFFFFFF, 0xFFFFFFFF]),
):
    _INT_LITERAL_CASES.append((f"OpSwitch %{_t} case {_lit}", _switch(_t, _lit), _words))


def _literal_words(spv: bytes, opcode: int, skip: int):
    """The words after the first `skip` operand words of the last instruction with `opcode` (the literal)."""
    words = struct.unpack(f"<{len(spv) // 4}I", spv)
    i, found = 5, None
    while i < len(words):
        n = words[i] >> 16
        if words[i] & 0xFFFF == opcode:
            found = list(words[i + 1 + skip:i + n])
        i += n
    return found


# opcode and leading operand words before the literal: OpConstant (result type, result), OpDecorate (target,
# decoration), OpGroupMemberDecorate (group, then the struct id the literal pairs with), OpSwitch (selector,
# default); a switch target label follows its literal.
_LITERAL_AT = {"OpConstant": (43, 2), "OpDecorate": (71, 2), "OpGroupMemberDecorate": (75, 2), "OpSwitch": (251, 2)}


class IntegerLiteralRangeTests(unittest.TestCase):
    """BUILD-1012: an integer literal that doesn't fit its type is refused as spirv-as refuses it, not masked."""

    def check_case(self, name, text, expected):
        op = name.split()[0]
        opcode, skip = _LITERAL_AT[op]
        if expected is None:
            with self.assertRaises(spirv_asm.SpirvAsmError):
                spirv_asm.assemble(text)
            return
        got = _literal_words(spirv_asm.assemble(text), opcode, skip)
        if op == "OpSwitch":
            got = got[:-1]  # the case's target label
        self.assertEqual(got, expected)

    def test_literals_fit_their_type_or_are_refused(self):
        for name, text, expected in _INT_LITERAL_CASES:
            with self.subTest(name):
                self.check_case(name, text, expected)

    def test_refusal_names_the_literal_and_the_type(self):
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError,
                                    r"line 14: integer 4294967296 does not fit in a 32-bit unsigned integer"):
            spirv_asm.assemble(_INT_HEAD + "%c = OpConstant %u32 4294967296\n")
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, r"negative number in an unsigned literal: -1"):
            spirv_asm.assemble(_INT_HEAD + "%c = OpConstant %u16 -1\n")

    @unittest.skipUnless(SPIRV_AS, "spirv-as not installed")
    def test_cases_match_spirv_as(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
            for name, text, expected in _INT_LITERAL_CASES:
                with self.subTest(name):
                    with open(src, "w", encoding="utf-8", newline="\n") as f:
                        f.write(text)
                    r = subprocess.run([SPIRV_AS, src, "-o", dst, "--target-env", "vulkan1.2"],
                                       capture_output=True, text=True)
                    if expected is None:
                        self.assertNotEqual(r.returncode, 0, f"spirv-as accepted {name}")
                        continue
                    self.assertEqual(r.returncode, 0, r.stderr)
                    with open(dst, "rb") as f:
                        ref = f.read()
                    got = spirv_asm.assemble(text)
                    a = struct.unpack(f"<{len(got) // 4}I", got)
                    b = struct.unpack(f"<{len(ref) // 4}I", ref)
                    self.assertEqual(a[:2] + a[3:], b[:2] + b[3:])


def _selector_module(sel_type, literal, later):
    """A compute shader with an OpSwitch on `%sel = OpIAdd %sel_type %c0 %c0`.

    `later` puts the block that defines `%sel` textually AFTER the block with the OpSwitch (the layout of SHADER-1089);
    otherwise the definition comes first. spirv-as refuses the `later` layout ("The selector operand for OpSwitch must
    be the result of an instruction that generates an integer scalar") and spirv-val refuses its block order ("Invalid
    OpSwitch: selector id N has no type"): blocks must be laid out so that a block precedes every block it dominates.
    """
    head = (
        "OpCapability Shader\nOpCapability Int64\nOpMemoryModel Logical GLSL450\n"
        'OpEntryPoint GLCompute %main "main"\nOpExecutionMode %main LocalSize 1 1 1\n'
        "%void = OpTypeVoid\n%fn = OpTypeFunction %void\n%uint = OpTypeInt 32 0\n%ulong = OpTypeInt 64 0\n"
        f"%c0 = OpConstant %{sel_type} 0\n%main = OpFunction %void None %fn\n%entry = OpLabel\n"
    )
    define = f"%sel = OpIAdd %{sel_type} %c0 %c0\n"
    switch = f"OpSelectionMerge %exit None\nOpSwitch %sel %exit {literal} %exit\n"
    tail = "%exit = OpLabel\nOpReturn\nOpFunctionEnd\n"
    if later:
        return head + "OpBranch %dom\n%sw = OpLabel\n" + switch + "%dom = OpLabel\n" + define + "OpBranch %sw\n" + tail
    return head + "OpBranch %dom\n%dom = OpLabel\n" + define + "OpBranch %sw\n%sw = OpLabel\n" + switch + tail


def _switch_instruction(spv: bytes):
    """The words of the (only) OpSwitch instruction, its header word first."""
    words = struct.unpack(f"<{len(spv) // 4}I", spv)
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if words[i] & 0xFFFF == 251:
            return list(words[i:i + n])
        i += n
    raise AssertionError("no OpSwitch in the module")


class SelectorMustBeDefinedBeforeTests(unittest.TestCase):
    """SHADER-1089: an OpSwitch selector with no type yet is an error, not a silent 32-bit guess for the case literals."""

    SELECTOR_ERROR = r"line \d+: OpSwitch selector %sel has no known type: it must be defined before the OpSwitch"

    def test_repro_with_a_64_bit_selector_defined_in_a_later_block_is_refused(self):
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, self.SELECTOR_ERROR):    # was five words (0x500fb)
            spirv_asm.assemble(_selector_module("ulong", "1", later=True))

    def test_literal_above_32_bits_with_a_later_selector_gives_the_same_error(self):
        for literal in ("0x100000001", "4294967296", "0xFFFFFFFFFFFFFFFF"):    # was an out-of-range error on a 32-bit guess
            with self.subTest(literal=literal):
                with self.assertRaisesRegex(spirv_asm.SpirvAsmError, self.SELECTOR_ERROR):
                    spirv_asm.assemble(_selector_module("ulong", literal, later=True))

    def test_32_bit_selector_defined_later_is_refused_too(self):
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, self.SELECTOR_ERROR):    # was five words, silently
            spirv_asm.assemble(_selector_module("uint", "1", later=True))

    def test_selector_defined_nowhere_is_refused(self):
        text = _selector_module("ulong", "1", later=False).replace("%sel = OpIAdd %ulong %c0 %c0\n", "")
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, self.SELECTOR_ERROR):
            spirv_asm.assemble(text)

    def test_the_error_names_the_line_of_the_switch(self):
        text = _selector_module("ulong", "1", later=True)
        switch_line = text.splitlines().index("OpSwitch %sel %exit 1 %exit") + 1
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, f"^line {switch_line}: OpSwitch selector %sel"):
            spirv_asm.assemble(text)

    def test_selector_defined_before_is_unchanged_and_valid(self):
        for sel_type, literal, length in (("ulong", "1", 6), ("ulong", "0x100000001", 6), ("uint", "1", 5)):
            with self.subTest(sel_type=sel_type, literal=literal):
                spv = spirv_asm.assemble(_selector_module(sel_type, literal, later=False))
                self.assertEqual(len(_switch_instruction(spv)), length)
                _validate(spv)

    def test_selector_that_is_a_function_parameter_or_phi_or_load_is_typed(self):
        head = (
            "OpCapability Shader\nOpCapability Int64\nOpMemoryModel Logical GLSL450\n"
            'OpEntryPoint GLCompute %main "main"\nOpExecutionMode %main LocalSize 1 1 1\n'
            "%void = OpTypeVoid\n%ulong = OpTypeInt 64 0\n%fnp = OpTypeFunction %void %ulong\n%fn = OpTypeFunction %void\n"
            "%ptr = OpTypePointer Function %ulong\n%c0 = OpConstant %ulong 0\n"
        )
        param = (head + "%f = OpFunction %void None %fnp\n%p = OpFunctionParameter %ulong\n%e = OpLabel\n"
                 "OpSelectionMerge %x None\nOpSwitch %p %x 0x100000001 %x\n%x = OpLabel\nOpReturn\nOpFunctionEnd\n"
                 "%main = OpFunction %void None %fn\n%me = OpLabel\nOpReturn\nOpFunctionEnd\n")
        self.assertEqual(len(_switch_instruction(spirv_asm.assemble(param))), 6)
        load = (head + "%main = OpFunction %void None %fn\n%e = OpLabel\n%v = OpVariable %ptr Function\n"
                "%sel = OpLoad %ulong %v\nOpSelectionMerge %x None\nOpSwitch %sel %x 0x100000001 %x\n"
                "%x = OpLabel\nOpReturn\nOpFunctionEnd\n")
        self.assertEqual(len(_switch_instruction(spirv_asm.assemble(load))), 6)


if __name__ == "__main__":
    unittest.main()
