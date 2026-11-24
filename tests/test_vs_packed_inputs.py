"""SHADER-644: packed VS ISG1 elements that share a register.

The fetch-call path replaces a driver fetch with one vec4 Vulkan input per ISG1
register. Two elements in the same register (for example xy and zw) must not
create two OpVariable declarations with the same name.

Run from the repository root:  python -m pytest tests/test_vs_packed_inputs.py
Every shader and DXBC container here is synthetic; no game data is used.
"""
import re
import struct

import pytest

import rdna2spirv
from conftest import make_dxbc, requires_tools

pytestmark = requires_tools

# s_swappc_b64 marks the driver fetch. All eight components of registers 0 and 1
# are read before being written, matching two ISG1 elements in register 0.
VS_PACKED = """
	s_swappc_b64 s[4:5], s[4:5]
	v_add_f32 v1, v4, v5
	v_add_f32 v2, v6, v7
	exp pos0 v1, v2, v1, v1 done
	exp param0 v4, v5, v6, v7
	s_endpgm
"""

VS_TWO_REGISTERS = """
	s_swappc_b64 s[4:5], s[4:5]
	v_add_f32 v1, v4, v8
	v_add_f32 v2, v7, v11
	exp pos0 v1, v2, v1, v1 done
	exp param0 v4, v5, v6, v7
	exp param1 v8, v9, v10, v11
	s_endpgm
"""


def isg1(elements):
    """ISG1 chunk: elements are (register, system value, component mask, component type)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, sysv, mask, ctype in elements:
        element = bytearray(32)
        struct.pack_into("<6I", element, 0, 0, 0, 0, sysv, ctype, reg)
        element[24], element[25] = mask, 1
        payload += element
    return b"ISG1", bytes(payload)


def test_packed_register_emits_one_input_variable(make_shader):
    shader = make_shader(VS_PACKED, "vs_packed_input",
                         dxbc=make_dxbc([isg1([(0, 0, 0x3, 3), (0, 0, 0xC, 3)])]))
    spvasm = shader.translate()
    shader.validate(spvasm)

    assert len(re.findall(r"^%vin0 = OpVariable ", spvasm, re.M)) == 1
    assert spvasm.count("OpDecorate %vin0 Location 0") == 1
    gen = rdna2spirv.translate.last_gen
    assert gen.vertex_inputs == [(0, 3)]
    assert gen.vertex_signature == [
        {"register": 0, "mask": 0x3, "system_value": 0, "component_type": 3},
        {"register": 0, "mask": 0xC, "system_value": 0, "component_type": 3},
    ]
    meta = rdna2spirv.shader_meta(gen, "")
    assert meta["vertex_inputs"] == [{"location": 0, "type": "float"}]
    assert len(meta["vertex_input_signature"]) == 2


def test_distinct_integer_registers_keep_their_component_types(make_shader):
    shader = make_shader(VS_TWO_REGISTERS, "vs_typed_inputs",
                         dxbc=make_dxbc([isg1([(0, 0, 0xF, 1), (1, 0, 0xF, 2)])]))
    spvasm = shader.translate()
    shader.validate(spvasm)

    gen = rdna2spirv.translate.last_gen
    assert gen.vertex_inputs == [(0, 1), (1, 2)]
    assert "%vin0 = OpVariable %_ptr_Input_v4u Input" in spvasm
    assert "%vin1 = OpVariable %_ptr_Input_v4i Input" in spvasm
    assert rdna2spirv.shader_meta(gen, "")["vertex_inputs"] == [
        {"location": 0, "type": "uint"}, {"location": 1, "type": "sint"}]


def test_packed_register_with_conflicting_types_is_rejected(make_shader):
    shader = make_shader(VS_PACKED, "vs_conflicting_input",
                         dxbc=make_dxbc([isg1([(0, 0, 0x3, 1), (0, 0, 0xC, 3)])]))
    with pytest.raises(rdna2spirv.Unsupported, match="mixed component types"):
        shader.translate()
