"""Semantic interpreter tests for flat interpolation constant copying (v_interp_mov_f32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    python -I tests/semantic/test_interp_mov_flat.py
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Architecture:
    - Instruction: `v_interp_mov_f32` (and `v_interp_mov_f32_e32`).
    - Format: `v_interp_mov_f32 v_dst, p0, attrN.C`
    - In pixel shaders, parameter `p0` represents the provoking vertex (vertex 0)
      for flat interpolation.
    - Unlike perspective interpolation (`v_interp_p1_f32` / `v_interp_p2_f32`) which
      evaluates P = P0 + i*(P1 - P0) + j*(P2 - P0), `v_interp_mov_f32` copies the
      constant attribute value from the provoking vertex verbatim to every pixel.
    - Preserves integer bit patterns without floating-point distortion. Flat inputs
      typically carry integer IDs (instance ID, material ID, primitive ID, bitmasks).
      Smooth interpolation of integer bit patterns corrupts them, whereas flat
      copying delivers exact bitwise identity.
  * SPIR-V Translation (tools/rdna2spirv.py):
    - Declares input attributes as vec4 inputs:
      `%in_attrN = OpVariable %_ptr_Input_v4 Input`
    - Input attributes read only through `v_interp_mov_f32` are marked `Flat`:
      `OpDecorate %in_attrN Location N`
      `OpDecorate %in_attrN Flat`
    - Attributes read via `v_interp_p1` / `v_interp_p2` are interpolated and must
      NOT receive the `Flat` decoration.
    - Loaded via `OpAccessChain %_ptr_Input_float %in_attrN %c{comp}`, `OpLoad %float`,
      and bitcast to uint `OpBitcast %uint` to preserve all 32 bits intact.
    - Valid only in pixel shaders (`stage == "ps"`). Use in other stages or with
      missing `attr` operands raises `Unsupported`.
"""
from __future__ import annotations

import gc
import os
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


def f32_round(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def f32_to_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def ref_flat_copy(p0_vec4, comp: int) -> int:
    """Reference oracle for flat interpolation: exact component copy from provoking vertex."""
    if not (0 <= comp <= 3):
        raise ValueError(f"Invalid component index: {comp}")
    val = p0_vec4[comp]
    if isinstance(val, float):
        return f32_to_bits(val)
    return val & MASK32


def ref_barycentric_interp(p0_f: float, p1_f: float, p2_f: float, i: float, j: float) -> float:
    """Reference oracle for perspective barycentric interpolation: P0 + i*(P1-P0) + j*(P2-P0)."""
    return f32_round(p0_f + i * (p1_f - p0_f) + j * (p2_f - p0_f))


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel", "OpReturn",
    "OpFunctionEnd", "OpName", "OpSource",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


class FlatSpirvInterp:
    """Mini SPIR-V interpreter with Input vector attribute support for flat constant interpolation."""

    def __init__(self, spvasm: str, flat_inputs=None):
        self.spvasm = spvasm
        self.flat_inputs = flat_inputs or {}
        self.types = {}
        self.constants = {}
        self.ptr_types = {}
        self.var_class = {}
        self.ptr_class = {}
        self.var_box = {}
        self.pointers = {}
        self.values = {}
        self.outputs = []
        self.body = []
        self._parse()

    def _parse(self):
        in_body = False
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"Cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                self.body.append((res, op, a))
                continue
            if op == "OpFunction":
                in_body = True
                continue
            if op in _IGNORED:
                continue
            if op == "OpTypeInt":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit int supported")
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeFloat":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit float supported")
                self.types[res] = ("float",)
            elif op == "OpTypeBool":
                self.types[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                size = int(self.constants[a[1]])
                self.types[res] = ("arr", a[0], size)
            elif op == "OpTypeStruct":
                self.types[res] = ("struct", a)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                k = self.types[a[0]]
                if k[0] == "float":
                    self.constants[res] = self.values[res] = f32_round(float(a[1]))
                elif k[0] == "int":
                    self.constants[res] = self.values[res] = int(a[1], 0) & MASK32
            elif op == "OpConstantTrue":
                self.constants[res] = self.values[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = self.values[res] = False
            elif op == "OpConstantNull":
                k = self.types[a[0]]
                if k[0] == "arr":
                    self.constants[res] = self.values[res] = [0] * k[2]
                elif k[0] == "vec":
                    self.constants[res] = self.values[res] = (0.0,) * k[2]
                else:
                    self.constants[res] = self.values[res] = 0
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                self.var_class[res] = cls
                self.ptr_class[res] = cls
                if cls == "PushConstant":
                    self.var_box[res] = [None]
                elif cls == "Input":
                    m_idx = re.search(r"\d+", res)
                    attr_idx = int(m_idx.group()) if m_idx else 0
                    raw_vec = self.flat_inputs.get(attr_idx, (0, 0, 0, 0))
                    floats = [bits_to_f32(w) if isinstance(w, int) else f32_round(w) for w in raw_vec]
                    self.var_box[res] = [list(floats)]
                elif cls in ("Function", "Output"):
                    if len(a) > 2:
                        init = self.values[a[2]]
                        self.var_box[res] = [list(init) if isinstance(init, (list, tuple)) else init]
                    else:
                        pty = self.types[pointee]
                        if pty[0] == "vec":
                            self.var_box[res] = [[0.0] * pty[2]]
                        elif pty[0] == "arr":
                            self.var_box[res] = [[0] * pty[2]]
                        else:
                            self.var_box[res] = [0]
                    if cls == "Output":
                        self.outputs.append(res)
                else:
                    raise UnsupportedConstruct(f"Unsupported storage class {cls}")
                self.pointers[res] = (self.var_box[res], 0)

    def run(self):
        for res, op, a in self.body:
            if op in _IGNORED:
                continue
            if op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("non-Function variable in body")
                self.var_class[res] = cls
                self.ptr_class[res] = cls
                if len(a) > 2:
                    init = self.values[a[2]]
                    self.var_box[res] = [list(init) if isinstance(init, (list, tuple)) else init]
                else:
                    self.var_box[res] = [0]
                self.pointers[res] = (self.var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                b_cls = self.ptr_class.get(base, self.var_class.get(base))
                self.ptr_class[res] = b_cls
                if b_cls == "PushConstant":
                    self.pointers[res] = ("pc", None)
                    continue
                p = self.pointers[base]
                for idx_id in a[2:]:
                    idx = self.values[idx_id]
                    container = p[0][p[1]]
                    if not isinstance(container, list) or not 0 <= idx < len(container):
                        raise UnsupportedConstruct("OpAccessChain index out of range")
                    p = (container, idx)
                self.pointers[res] = p
            elif op == "OpLoad":
                p = self.pointers[a[1]]
                if p[0] == "pc":
                    self.values[res] = 0
                else:
                    v = p[0][p[1]]
                    self.values[res] = list(v) if isinstance(v, list) else v
            elif op == "OpStore":
                p_cls = self.ptr_class.get(a[0], self.var_class.get(a[0]))
                if p_cls in ("Input", "PushConstant"):
                    raise UnsupportedConstruct(f"Cannot store to {p_cls} variable")
                p = self.pointers[a[0]]
                p[0][p[1]] = self.values[a[1]]
            elif op == "OpSelect":
                cond = self.values[a[1]]
                if not isinstance(cond, bool):
                    raise UnsupportedConstruct("OpSelect condition must be boolean")
                self.values[res] = self.values[a[2]] if cond else self.values[a[3]]
            elif op == "OpBitcast":
                target_type = self.types[a[0]]
                val = self.values[a[1]]
                if target_type[0] == "float" and isinstance(val, int):
                    self.values[res] = bits_to_f32(val)
                elif target_type[0] == "int" and isinstance(val, float):
                    self.values[res] = f32_to_bits(val)
                elif target_type[0] == "int" and isinstance(val, int):
                    self.values[res] = val & MASK32
                else:
                    self.values[res] = val
            elif op == "OpIAdd":
                self.values[res] = (self.values[a[1]] + self.values[a[2]]) & MASK32
            elif op == "OpISub":
                self.values[res] = (self.values[a[1]] - self.values[a[2]]) & MASK32
            elif op == "OpBitwiseAnd":
                self.values[res] = (self.values[a[1]] & self.values[a[2]]) & MASK32
            elif op == "OpBitwiseOr":
                self.values[res] = (self.values[a[1]] | self.values[a[2]]) & MASK32
            elif op == "OpBitwiseXor":
                self.values[res] = (self.values[a[1]] ^ self.values[a[2]]) & MASK32
            elif op == "OpShiftLeftLogical":
                sh = self.values[a[2]]
                self.values[res] = (self.values[a[1]] << sh) & MASK32 if sh < 32 else 0
            elif op == "OpShiftRightLogical":
                sh = self.values[a[2]]
                self.values[res] = (self.values[a[1]] >> sh) & MASK32 if sh < 32 else 0
            elif op == "OpCompositeConstruct":
                self.values[res] = tuple(self.values[x] for x in a[1:])
            elif op == "OpCompositeExtract":
                v = self.values[a[1]]
                for i in a[2:]:
                    v = v[int(i)]
                self.values[res] = v
            else:
                raise UnsupportedConstruct(f"Unhandled opcode in body: {op}")

        result = {}
        for out in self.outputs:
            raw = self.var_box[out][0]
            flat = raw if isinstance(raw, (tuple, list)) else (raw,)
            result[out] = tuple(f32_to_bits(x) if isinstance(x, float) else x & MASK32 for x in flat)
        return result


def translate_synthetic_ps(asm_instructions: str, shader_name: str = "ps_synthetic") -> str:
    """Translates a straight-line fragment shader into SPIR-V assembly text."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm_dir = os.path.join(tmp, "disasm")
        os.mkdir(disasm_dir)
        s_path = os.path.join(disasm_dir, f"{shader_name}.s")
        dxbc_path = os.path.join(tmp, f"{shader_name}.dxbc")
        with open(dxbc_path, "wb") as f:
            f.write(STUB_DXBC)

        full_asm = f"; {shader_name}\n" + asm_instructions
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(full_asm)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        return spv


class FlatInterpolationReferenceTests(unittest.TestCase):
    """Verifies that flat copying preserves raw bits while barycentric interpolation alters them."""

    def test_flat_copy_oracle_all_channels(self):
        p0 = (0x101, 0x202, 0x303, 0x404)
        for comp, name in enumerate("xyzw"):
            with self.subTest(component=name):
                self.assertEqual(ref_flat_copy(p0, comp), p0[comp])

    def test_flat_copy_preserves_integer_ids_unmangled(self):
        integer_ids = [0x0, 0x1, 0x101, 0x102, 0xCAFEBABE, 0xDEADBEEF, 0xFFFFFFFF]
        for val in integer_ids:
            with self.subTest(val=hex(val)):
                vec = (val, 0, 0, 0)
                self.assertEqual(ref_flat_copy(vec, 0), val)

    def test_barycentric_interpolation_distorts_integer_bits(self):
        p0_int = 0x101
        p1_int = 0x102
        p2_int = 0x103
        p0_f = bits_to_f32(p0_int)
        p1_f = bits_to_f32(p1_int)
        p2_f = bits_to_f32(p2_int)
        interp_f = ref_barycentric_interp(p0_f, p1_f, p2_f, 0.333, 0.333)
        interp_bits = f32_to_bits(interp_f)
        self.assertNotEqual(interp_bits, p0_int)
        self.assertEqual(ref_flat_copy((p0_int, p1_int, p2_int, 0), 0), p0_int)


class FlatSpirvDirectInterpTests(unittest.TestCase):
    """Direct tests of FlatSpirvInterp on hand-written SPIR-V modules."""

    DIRECT_HEADER = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0 %in_attr1
OpExecutionMode %main OriginUpperLeft
OpDecorate %out_mrt0 Location 0
OpDecorate %in_attr1 Location 1
OpDecorate %in_attr1 Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4 = OpTypePointer Output %v4float
%_ptr_Input_v4 = OpTypePointer Input %v4float
%_ptr_Input_float = OpTypePointer Input %float
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%out_mrt0 = OpVariable %_ptr_Output_v4 Output
%in_attr1 = OpVariable %_ptr_Input_v4 Input
%main = OpFunction %void None %fn
%entry = OpLabel
"""

    def test_direct_flat_input_read_all_channels(self):
        body = self.DIRECT_HEADER + """\
%p0 = OpAccessChain %_ptr_Input_float %in_attr1 %c0
%f0 = OpLoad %float %p0
%p1 = OpAccessChain %_ptr_Input_float %in_attr1 %c1
%f1 = OpLoad %float %p1
%p2 = OpAccessChain %_ptr_Input_float %in_attr1 %c2
%f2 = OpLoad %float %p2
%p3 = OpAccessChain %_ptr_Input_float %in_attr1 %c3
%f3 = OpLoad %float %p3
%out_vec = OpCompositeConstruct %v4float %f0 %f1 %f2 %f3
OpStore %out_mrt0 %out_vec
OpReturn
OpFunctionEnd
"""
        inputs = {1: (0x11111111, 0x22222222, 0x33333333, 0x44444444)}
        interp = FlatSpirvInterp(body, inputs)
        res = interp.run()
        self.assertEqual(res["%out_mrt0"], (0x11111111, 0x22222222, 0x33333333, 0x44444444))

    def test_direct_flat_input_integer_bitcast_arithmetic(self):
        body = self.DIRECT_HEADER + """\
%p0 = OpAccessChain %_ptr_Input_float %in_attr1 %c0
%f0 = OpLoad %float %p0
%u0 = OpBitcast %uint %f0
%u1 = OpIAdd %uint %u0 %c1
%f1 = OpBitcast %float %u1
%out_vec = OpCompositeConstruct %v4float %f0 %f1 %f0 %f1
OpStore %out_mrt0 %out_vec
OpReturn
OpFunctionEnd
"""
        inputs = {1: (0x101, 0, 0, 0)}
        interp = FlatSpirvInterp(body, inputs)
        res = interp.run()
        self.assertEqual(res["%out_mrt0"], (0x101, 0x102, 0x101, 0x102))

    def test_store_to_input_variable_is_rejected(self):
        body = self.DIRECT_HEADER + """\
%p0 = OpAccessChain %_ptr_Input_float %in_attr1 %c0
%f0 = OpLoad %float %p0
OpStore %p0 %f0
OpReturn
OpFunctionEnd
"""
        interp = FlatSpirvInterp(body, {1: (0x1, 0x2, 0x3, 0x4)})
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_out_of_bounds_input_component_access_is_rejected(self):
        body = self.DIRECT_HEADER + """\
%c4 = OpConstant %uint 4
%p4 = OpAccessChain %_ptr_Input_float %in_attr1 %c4
%f4 = OpLoad %float %p4
OpReturn
OpFunctionEnd
"""
        interp = FlatSpirvInterp(body, {1: (0x1, 0x2, 0x3, 0x4)})
        with self.assertRaises(UnsupportedConstruct):
            interp.run()


class RdnaFlatTranslationTests(unittest.TestCase):
    """Verifies that rdna2spirv translates v_interp_mov_f32 into flat decorated SPIR-V."""

    def test_v_interp_mov_emits_flat_decoration(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        self.assertIn("OpDecorate %in_attr1 Location 1", spv)
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertIn("%in_attr1 = OpVariable %_ptr_Input_v4 Input", spv)
        self.assertIn("%_ptr_Input_float %in_attr1 %c0", spv)

    def test_all_four_components_map_to_correct_constants(self):
        asm = (
            "\tv_interp_mov_f32 v2, p0, attr2.x\n"
            "\tv_interp_mov_f32 v3, p0, attr2.y\n"
            "\tv_interp_mov_f32 v4, p0, attr2.z\n"
            "\tv_interp_mov_f32 v5, p0, attr2.w\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        self.assertIn("OpDecorate %in_attr2 Flat", spv)
        self.assertIn("%_ptr_Input_float %in_attr2 %c0", spv)
        self.assertIn("%_ptr_Input_float %in_attr2 %c1", spv)
        self.assertIn("%_ptr_Input_float %in_attr2 %c2", spv)
        self.assertIn("%_ptr_Input_float %in_attr2 %c3", spv)

    def test_mixed_flat_and_interpolated_attributes(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\tv_interp_p1_f32_e32 v4, v0, attr0.x\n"
            "\tv_interp_p2_f32_e32 v4, v1, attr0.x\n"
            "\texp mrt0 v2, v2, v4, v4 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        self.assertIn("OpDecorate %in_attr1 Location 1", spv)
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertIn("OpDecorate %in_attr0 Location 0", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)

    def test_attribute_read_both_flat_and_interpolated_is_not_flat(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr3.x\n"
            "\tv_interp_p1_f32_e32 v4, v0, attr3.y\n"
            "\tv_interp_p2_f32_e32 v4, v1, attr3.y\n"
            "\texp mrt0 v2, v2, v4, v4 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        self.assertIn("OpDecorate %in_attr3 Location 3", spv)
        self.assertNotIn("OpDecorate %in_attr3 Flat", spv)

    def test_v_interp_mov_in_vertex_shader_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm_dir = os.path.join(tmp, "disasm")
            os.mkdir(disasm_dir)
            s_path = os.path.join(disasm_dir, "vs_test.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write("; vs\n\tv_interp_mov_f32 v2, p0, attr1.x\n\texp pos0 v2, v2, v2, v2 done\n\ts_endpgm\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaises(rdna2spirv.Unsupported):
                    rdna2spirv.translate(s_path)
                gc.collect()

    def test_v_interp_mov_without_attr_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm_dir = os.path.join(tmp, "disasm")
            os.mkdir(disasm_dir)
            s_path = os.path.join(disasm_dir, "ps_test.s")
            dxbc_path = os.path.join(tmp, "ps_test.dxbc")
            with open(dxbc_path, "wb") as f:
                f.write(STUB_DXBC)
            with open(s_path, "w", encoding="utf-8") as f:
                f.write("; ps\n\tv_interp_mov_f32 v2, p0, v1\n\texp mrt0 v2, v2, v2, v2 done vm\n\ts_endpgm\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaises(rdna2spirv.Unsupported):
                    rdna2spirv.translate(s_path)
                gc.collect()


class EndToEndFlatExecutionTests(unittest.TestCase):
    """End-to-end translation and execution of RDNA2 flat pixel shaders."""

    def test_p0_and_interpolated_read_of_one_f32_location(self) -> None:
        with open(os.path.join(HERE, "fixtures", "mixed_f32_varying.asm"), encoding="ascii") as fixture:
            lines = fixture.read().splitlines()
        shader = support.translate_lines("ps_mixed_f32_varying", lines)
        attrs = {0: [(f32_to_bits(x), 0, 0, 0) for x in (2.0, 6.0, 10.0)]}
        for weights, interpolated in (((0.25, 0.25, 0.5), 7.0), ((0.5, 0.5, 0.0), 4.0)):
            with self.subTest(weights=weights):
                inputs = support.varyings(shader, attrs, weights)
                self.assertEqual(si.run_fragment(shader, inputs)["%out_mrt0"],
                                 (f32_to_bits(2.0), f32_to_bits(interpolated)) * 2)

        for fallback in ("BARY_FALLBACK", "F16_HALVES"):
            with self.subTest(fallback=fallback), mock.patch.object(rdna2spirv, fallback, True):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "need per-vertex inputs"):
                    support.translate_lines("ps_mixed_f32_varying", lines)

    def test_ps_flat_model_execution(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\tv_add_nc_u32_e32 v3, 1, v2\n"
            "\texp mrt0 v2, v3, v2, v3 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        inputs = {1: (0x101, 0, 0, 0)}
        interp = FlatSpirvInterp(spv, inputs)
        res = interp.run()
        self.assertEqual(res["%out_mrt0"], (0x101, 0x102, 0x101, 0x102))

    def test_four_channel_vector_passthrough(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\tv_interp_mov_f32_e32 v3, p0, attr1.y\n"
            "\tv_interp_mov_f32_e32 v4, p0, attr1.z\n"
            "\tv_interp_mov_f32_e32 v5, p0, attr1.w\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        vec = (0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321)
        interp = FlatSpirvInterp(spv, {1: vec})
        res = interp.run()
        self.assertEqual(res["%out_mrt0"], vec)

    def test_flat_boundary_bit_patterns(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        patterns = [
            0x00000000,
            0x00000001,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
        ]
        for val in patterns:
            with self.subTest(pattern=hex(val)):
                interp = FlatSpirvInterp(spv, {1: (val, 0, 0, 0)})
                res = interp.run()
                self.assertEqual(res["%out_mrt0"], (val, val, val, val))

    def test_multiple_flat_attributes_interaction(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\tv_interp_mov_f32_e32 v3, p0, attr2.y\n"
            "\tv_add_nc_u32_e32 v4, v2, v3\n"
            "\tv_subrev_nc_u32_e32 v5, v2, v3\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        inputs = {
            1: (0x50, 0, 0, 0),
            2: (0, 0x30, 0, 0),
        }
        interp = FlatSpirvInterp(spv, inputs)
        res = interp.run()
        self.assertEqual(res["%out_mrt0"], (0x50, 0x30, 0x80, 0xFFFFFFE0))

    def test_flat_bitwise_masking_arithmetic(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\tv_and_b32_e32 v3, 0x00FF00FF, v2\n"
            "\tv_lshlrev_b32_e32 v4, 8, v3\n"
            "\tv_xor_b32_e32 v5, v2, v4\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        inputs = {1: (0x12345678, 0, 0, 0)}
        interp = FlatSpirvInterp(spv, inputs)
        res = interp.run()
        v2 = 0x12345678
        v3 = v2 & 0x00FF00FF
        v4 = (v3 << 8) & MASK32
        v5 = v2 ^ v4
        self.assertEqual(res["%out_mrt0"], (v2, v3, v4, v5))


class FlatNegativeControlTests(unittest.TestCase):
    """Negative controls proving sensitivity to flat decoration and access sabotage."""

    def test_sabotaged_flat_component_access_detected(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        good_spv = translate_synthetic_ps(asm)
        inputs = {1: (0x101, 0x202, 0x303, 0x404)}
        good_res = FlatSpirvInterp(good_spv, inputs).run()
        self.assertEqual(good_res["%out_mrt0"], (0x101, 0x101, 0x101, 0x101))

        bad_spv = good_spv.replace("%_ptr_Input_float %in_attr1 %c0", "%_ptr_Input_float %in_attr1 %c1")
        bad_res = FlatSpirvInterp(bad_spv, inputs).run()
        self.assertNotEqual(bad_res["%out_mrt0"], good_res["%out_mrt0"])
        self.assertEqual(bad_res["%out_mrt0"], (0x202, 0x202, 0x202, 0x202))

    def test_sabotaged_attribute_source_detected(self):
        asm = (
            "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        spv = translate_synthetic_ps(asm)
        inputs_a = {1: (0xAAAA, 0, 0, 0)}
        inputs_b = {1: (0xBBBB, 0, 0, 0)}
        res_a = FlatSpirvInterp(spv, inputs_a).run()["%out_mrt0"]
        res_b = FlatSpirvInterp(spv, inputs_b).run()["%out_mrt0"]
        self.assertNotEqual(res_a, res_b)


if __name__ == "__main__":
    unittest.main()
