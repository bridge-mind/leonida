"""rdna2spirv.py --pc-tail-ubo: s0..s31 stay in push constants, the tail moves to a uniform block (D3D-14).

Devices with maxPushConstantsSize 128 (Vulkan's minimum: AMD's Windows driver on RDNA2 iGPUs, some Intel Arc
drivers) can't take the default 144-byte block (152 with --mesh-compute). With the flag, the push-constant block is
s0..s31 alone, and the tail that follows them by default (heap base, code base, the --mesh-compute output address)
is a uniform block at set 1, binding 0, in the same order. The inputs are the hand-written GPU tests in
runtime/test/shaders with conftest's synthetic containers (no game data). Here:
* without the flag the module and its metadata are what they were;
* with it the push-constant block is 128 bytes, every tail read goes through %pc_tail, the block is in the entry
  point's interface, the module passes spirv-val, and the metadata records "pc_tail" without counting its binding
  as a set-0 binding;
* a shader that reads no part of the tail declares no block;
* --mesh-compute's output address is the block's dwords 4..5.
"""
import os
import re
import sys

import pytest

from conftest import ROOT, SAMPLE_DXBC, requires_tools

sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

TAIL_READERS = ("cs_store", "ps_m15")     # heap base (memory); ps_m15 also the code base (s_getpc_b64)


def _translate(make_shader, name=None, asm=None, tail=True, mesh_compute=False):
    if asm is None:
        with open(os.path.join(ROOT, "runtime", "test", "shaders", name + ".asm"), encoding="utf-8") as f:
            asm = f.read()
    kw = {"dxbc": SAMPLE_DXBC[name]} if name in SAMPLE_DXBC else {}
    shader = make_shader(asm, name or "ps_test", **kw)
    old = rdna2spirv.PC_TAIL_UBO, rdna2spirv.MESH_COMPUTE
    rdna2spirv.PC_TAIL_UBO, rdna2spirv.MESH_COMPUTE = tail, mesh_compute
    try:
        spvasm = shader.translate()
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    finally:
        rdna2spirv.PC_TAIL_UBO, rdna2spirv.MESH_COMPUTE = old
    return shader, spvasm, meta


def _entry_point(spvasm):
    return re.search(r"OpEntryPoint \w+ %main \"main\"([^\n]*)", spvasm).group(1).split()


def test_mesh_compute_layout():
    old = rdna2spirv.PC_TAIL_UBO
    try:
        rdna2spirv.PC_TAIL_UBO = False
        assert rdna2spirv.mesh_compute_layout(3, 1, 0, 0, [0])["address_dwords"] == [36, 37]
        rdna2spirv.PC_TAIL_UBO = True
        lay = rdna2spirv.mesh_compute_layout(3, 1, 0, 0, [0])
    finally:
        rdna2spirv.PC_TAIL_UBO = old
    assert lay["push_constant_bytes"] == 128
    assert lay["address_dwords"] is None and lay["pc_tail_address_dwords"] == [4, 5]


@requires_tools
@pytest.mark.parametrize("name", TAIL_READERS)
def test_default_is_unchanged(make_shader, name):
    shader, spvasm, meta = _translate(make_shader, name, tail=False)
    shader.validate(spvasm)
    assert "%pc_tail" not in spvasm and "PcTail" not in spvasm
    assert "OpMemberDecorate %UserData 1 Offset 128" in spvasm
    assert "OpMemberDecorate %UserData 4 Offset 140" in spvasm
    assert meta["push_constant_bytes"] == 144 and "pc_tail" not in meta


@requires_tools
@pytest.mark.parametrize("name", TAIL_READERS)
def test_tail_is_a_uniform_block(make_shader, name):
    _, plain, plain_meta = _translate(make_shader, name, tail=False)
    shader, spvasm, meta = _translate(make_shader, name)
    shader.validate(spvasm)
    # Push constants: s0..s31 and nothing after them.
    assert re.search(r"^ *%UserData = OpTypeStruct %arr_ud$", spvasm, re.M)
    assert "OpMemberDecorate %UserData 1 " not in spvasm
    assert not re.search(r"OpAccessChain %_ptr_PC_uint %user_data %c[1-6]\b", spvasm)
    # The tail: four dwords at offsets 0..12, set 1, binding 0, listed in the interface.
    assert re.search(r"^ *%PcTail = OpTypeStruct %uint %uint %uint %uint$", spvasm, re.M)
    for k in range(4):
        assert f"OpMemberDecorate %PcTail {k} Offset {4 * k}" in spvasm
    assert "OpDecorate %PcTail Block" in spvasm
    assert "OpDecorate %pc_tail DescriptorSet 1" in spvasm and "OpDecorate %pc_tail Binding 0" in spvasm
    assert "%pc_tail" in _entry_point(spvasm)
    assert "%ud_base_lo_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c0" in spvasm
    assert "%ud_base_hi_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c1" in spvasm
    assert meta["push_constant_bytes"] == 128
    assert meta["pc_tail"] == {"set": 1, "binding": 0, "bytes": 16}
    # Set-0 bindings are what they were: %pc_tail's binding 0 is set 1's.
    assert meta["bindings"] == plain_meta["bindings"]
    # Apart from the tail, the function is the same.
    body = lambda s: s[s.index("%main = OpFunction"):]  # noqa: E731
    strip = lambda s: re.sub(r"%ud_(base|code)_(lo|hi)_p = OpAccessChain [^\n]*", "", body(s))  # noqa: E731
    assert strip(spvasm) == strip(plain)


@requires_tools
def test_code_base_reads_the_block(make_shader):
    _, spvasm, meta = _translate(make_shader, "ps_m15")
    assert meta["uses_code_base"]
    assert "%ud_code_lo_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c2" in spvasm
    assert "%ud_code_hi_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c3" in spvasm


@requires_tools
def test_no_block_without_a_tail_read(make_shader):
    asm = "v_mov_b32_e32 v2, 1.0\nexp mrt0 v2, v2, v2, v2 done vm\ns_endpgm\n"
    shader, spvasm, meta = _translate(make_shader, asm=asm)
    shader.validate(spvasm)
    assert "%pc_tail" not in spvasm and "PcTail" not in spvasm
    assert meta["push_constant_bytes"] == 128 and "pc_tail" not in meta


@requires_tools
def test_mesh_compute_address_is_in_the_block(make_shader):
    shader, spvasm, meta = _translate(make_shader, "ms_tri", mesh_compute=True)
    shader.validate(spvasm)
    assert re.search(r'OpEntryPoint GLCompute %main "main"', spvasm)
    assert re.search(r"^ *%PcTail = OpTypeStruct %uint %uint %uint %uint %uint %uint$", spvasm, re.M)
    assert "OpMemberDecorate %PcTail 5 Offset 20" in spvasm
    assert "%moa_lo_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c4" in spvasm
    assert "%moa_hi_p = OpAccessChain %_ptr_Uniform_uint_tail %pc_tail %c5" in spvasm
    assert meta["push_constant_bytes"] == 128
    assert meta["pc_tail"] == {"set": 1, "binding": 0, "bytes": 24}
    assert meta["mesh_compute"]["push_constant_bytes"] == 128
    assert meta["mesh_compute"]["pc_tail_address_dwords"] == [4, 5]
