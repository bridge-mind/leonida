"""Regression tests for tools/rdna2spirv.py (RDNA2 -> SPIR-V).

Run from the repository root:  python -m pytest tests
Needs llvm-mc (AMDGPU target) and spirv-tools; the tests skip themselves when those are missing.
"""
import glob
import math
import os
import re
import struct
from pathlib import Path

import pytest

import rdna2spirv
from conftest import (EMPTY_DXBC, KNOWN_VAL_ISSUES, ROOT, SAMPLE_DXBC, Shader, make_dxbc, mesh_psv0_chunk,
                      requires_tools, signature_chunk)

pytestmark = requires_tools

SAMPLES = sorted(glob.glob(os.path.join(ROOT, "runtime", "test", "shaders", "*.asm")))
SAMPLES += sorted(glob.glob(os.path.join(ROOT, "tests", "semantic", "fixtures", "ps_*.asm")))

# One minimal pixel shader per instruction family listed in the "Supported" section of rdna2spirv.py's docstring.
# Each body is wrapped in a program that exports its result so nothing is optimised away.
SUPPORTED = {
    "v_mov": "v_mov_b32 v0, 1.0",
    "v_add_f32": "v_mov_b32 v1, 1.0\nv_add_f32 v0, v1, v1",
    "v_sub_f32": "v_mov_b32 v1, 2.0\nv_sub_f32 v0, v1, 1.0",
    "v_mul_f32": "v_mov_b32 v1, 2.0\nv_mul_f32 v0, v1, v1",
    "v_fma_f32": "v_mov_b32 v1, 2.0\nv_fma_f32 v0, v1, v1, v1",
    "v_max_f32": "v_mov_b32 v1, 2.0\nv_max_f32 v0, v1, 1.0",
    "v_min_f32": "v_mov_b32 v1, 2.0\nv_min_f32 v0, v1, 1.0",
    "v_rcp_f32": "v_mov_b32 v1, 4.0\nv_rcp_f32 v0, v1",
    "v_sqrt_f32": "v_mov_b32 v1, 4.0\nv_sqrt_f32 v0, v1",
    "v_floor_f32": "v_mov_b32 v1, 1.5\nv_floor_f32 v0, v1",
    "v_fract_f32": "v_mov_b32 v1, 1.5\nv_fract_f32 v0, v1",
    "v_cvt_f32_u32": "v_mov_b32 v1, 7\nv_cvt_f32_u32 v0, v1",
    "v_cvt_u32_f32": "v_mov_b32 v1, 7.0\nv_cvt_u32_f32 v0, v1",
    "v_add_nc_u32": "v_mov_b32 v1, 7\nv_add_nc_u32 v0, v1, v1",
    "v_and_b32": "v_mov_b32 v1, 7\nv_and_b32 v0, 3, v1",
    "v_lshlrev_b32": "v_mov_b32 v1, 7\nv_lshlrev_b32 v0, 2, v1",
    "v_lshrrev_b32": "v_mov_b32 v1, 7\nv_lshrrev_b32 v0, 1, v1",
    "v_bfe_u32": "v_mov_b32 v1, 0xf0\nv_bfe_u32 v0, v1, 4, 4",
    "v_med3_f32": "v_mov_b32 v1, 2.0\nv_med3_f32 v0, v1, 1.0, 4.0",
    "s_mov_b32": "s_mov_b32 s40, 5\nv_mov_b32 v0, s40",
    "s_add_u32": "s_mov_b32 s40, 5\ns_add_u32 s41, s40, 3\nv_mov_b32 v0, s41",
}


def program(body):
    return f"{body}\nv_mov_b32 v8, v0\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm\n"


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_gpu_samples_translate_and_validate(path, make_shader):
    """The hand-written GPU-harness and semantic shader samples must translate to valid SPIR-V."""
    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem if stem.startswith(("vs_", "ps_", "cs_", "gs_", "ms_")) else "ps_" + stem   # the prefix selects the stage
    if name.startswith(("gs_", "ms_")) and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic DXBC (ISG1/OSG1 for gs, PSV0 for ms) defined for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), name, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    shader.validate(shader.translate(), KNOWN_VAL_ISSUES.get(stem))


@pytest.mark.parametrize("case", sorted(SUPPORTED))
def test_supported_instruction(case, make_shader):
    shader = make_shader(program(SUPPORTED[case]))
    shader.validate(shader.translate())


# VOP1/VOP2 DPP source controls. The source lane is a shuffle followed by row/bank masking; a lane that the
# instruction leaves unwritten keeps its old destination, including for non-mov ops.
DPP_CASES = {
    "quad_perm": "v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "quad_perm_masked": "v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0x1 bank_mask:0x3",
    "row_shl": "v_mov_b32_dpp v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf",
    "row_shr_bound": "v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "row_ror": "v_mov_b32_dpp v1, v0 row_ror:4 row_mask:0xf bank_mask:0xf",
    "row_share": "v_mov_b32_dpp v1, v0 row_share:2 row_mask:0xf bank_mask:0xf",
    "row_xmask": "v_mov_b32_dpp v1, v0 row_xmask:4 row_mask:0xf bank_mask:0xf",
    "row_mirror": "v_mov_b32_dpp v1, v0 row_mirror row_mask:0xf bank_mask:0xf",
    "row_half_mirror": "v_mov_b32_dpp v1, v0 row_half_mirror row_mask:0xf bank_mask:0xf",
    "dpp8": "v_mov_b32_dpp v1, v0 dpp8:[7,6,5,4,3,2,1,0] fi:1",
    "vop1_not_bank": "v_not_b32_dpp v1, v0 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0x1",
    "vop2_add_shr": "v_add_nc_u32_dpp v1, v0, v2 row_shr:1 row_mask:0xf bank_mask:0xf",
}


@pytest.mark.parametrize("case", sorted(DPP_CASES))
def test_dpp_controls_translate_and_validate(case, make_shader):
    shader = make_shader(program("v_mov_b32 v0, v0\n" + DPP_CASES[case]))
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpGroupNonUniformShuffle" in spvasm


@pytest.mark.parametrize("mods", [[], ["row_share:20"], ["row_shl:0"], ["row_shl:1", "row_shr:2"]])
def test_dpp_bad_control_is_rejected(mods):
    """A missing, out-of-range or duplicate DPP control must fail instead of guessing."""
    gen = rdna2spirv.Gen("ps")
    ins = {"op": "v_mov_b32_dpp", "args": [{"k": "vgpr", "r": 1}, {"k": "vgpr", "r": 0}], "mods": mods}
    with pytest.raises(rdna2spirv.Unsupported):
        gen.insn(ins)


def test_dpp_wave64_emulation_goes_through_workgroup_memory(make_shader, monkeypatch):
    """--wave64-emulate must not leave a subgroup shuffle in a DPP shader."""
    shader = make_shader("s_mov_b64 exec, -1\n"
                         "v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
                         "s_endpgm\n", name="cs_dpp_w64")
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpGroupNonUniform" not in spvasm and "%sg_lane" not in spvasm
    assert "%x_buf" in spvasm and "OpControlBarrier" in spvasm


def test_translation_is_deterministic(make_shader):
    shader = make_shader(program(SUPPORTED["v_fma_f32"]))
    assert shader.translate() == shader.translate()


def test_unsupported_instruction_is_reported_not_guessed(make_shader):
    """Anything outside the supported set must raise Unsupported rather than emit a wrong program."""
    shader = make_shader(program("v_mov_b32 v1, 1.0\nv_sat_pk_u8_i16 v0, v1"))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("a16", [False, True], ids=["a32", "a16"])
@pytest.mark.parametrize("d16", [False, True], ids=["f32", "d16"])
def test_explicit_mip_depth_compare(make_shader, a16, d16):
    # Under A16 the reference is still f32; XY share v1 and LOD is the low half of v2.
    if a16:
        moves = "v_mov_b32 v1, 0x3a003400\nv_mov_b32 v2, 0x40003c00\n"
        address, modifier = "v[0:2]", " a16"
    else:
        moves = "v_mov_b32 v1, 0.25\nv_mov_b32 v2, 0.75\nv_mov_b32 v3, 1.0\n"
        address, modifier = "v[0:3]", ""
    shader = make_shader("v_mov_b32 v0, 0.5\n" + moves +
                         f"image_sample_c_l v4, {address}, s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D" +
                         modifier + (" d16" if d16 else "") + "\ns_waitcnt vmcnt(0)\nexp mrt0 v4, v4, v4, v4 done vm\ns_endpgm\n")
    spv = shader.translate()
    shader.validate(spv)
    sample = re.search(r"(%\w+) = OpImageSampleDrefExplicitLod %float %\w+ %\w+ (%\w+) Lod (%\w+)", spv)
    assert sample
    defs = dict(re.findall(r"(%\w+) = (.+)", spv))

    def float_source_register(value: str) -> list[str]:
        load = defs[value].split()[-1]  # f32 bitcast
        ptr = defs[load].split()[-1]    # raw VGPR word
        return defs[ptr].split()[-2:]

    result, ref, lod = sample.groups()
    assert float_source_register(ref) == ["%vgpr", "%c0"]
    if a16:
        unpacked_half = lod
        while defs[unpacked_half].startswith("OpBitcast "):   # a bitcast round trip is the original id
            unpacked_half = defs[unpacked_half].split()[-1]
        op, _, pair, half = defs[unpacked_half].split()
        assert (op, half) == ("OpCompositeExtract", "0")
        raw = defs[pair].split()[-1]
        assert defs[defs[raw].split()[-1]].endswith("%vgpr %c2")
    else:
        assert float_source_register(lod) == ["%vgpr", "%c3"]
    bits = next(name for name, code in defs.items() if code == f"OpBitcast %uint {result}")
    assert any(code == f"OpShiftRightLogical %uint {bits} %c10" for code in defs.values()) == d16


@pytest.mark.parametrize("sample", [
    "v[4:5], v[0:3], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D",
    "v4, v[0:4], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D",
])
def test_explicit_mip_depth_compare_unmodelled_forms(make_shader, sample):
    shader = make_shader(f"image_sample_c_l {sample}\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match="scalar 2D depth comparison"):
        shader.translate()


@pytest.mark.parametrize("d16", [False, True], ids=["f32", "d16"])
def test_depth_compare_sample_result_width(make_shader, d16):
    """image_sample_c_lz returns one value: f32 bits, or with d16 an f16 in the low half like every other d16 sample
    result. Written as f32 bits under d16, every compare result would read back as 0 (1.0 = 0x3f800000)."""
    shader = make_shader("v_mov_b32 v0, 0.5\nv_mov_b32 v1, 0.5\nv_mov_b32 v2, 0.25\n"
                         "image_sample_c_lz v4, v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                         + (" d16" if d16 else "") + "\nexp mrt0 v4, v4, v4, v4 done vm\ns_endpgm\n", "ps_dref")
    spvasm = shader.translate()
    shader.validate(spvasm)
    dref = re.search(r"^\s*(%\w+) = OpImageSampleDrefExplicitLod %float ", spvasm, re.M)
    assert dref, "the depth compare sample is missing"
    d = re.escape(dref.group(1))
    bitcast = re.search(rf"^\s*(%\w+) = OpBitcast %uint {d}$", spvasm, re.M)
    assert bitcast, "the depth compare bitcast is missing"
    bc = re.escape(bitcast.group(1))
    as_f16 = re.search(rf"^\s*%\w+ = OpShiftRightLogical %uint {bc} %c10$", spvasm, re.M)
    assert bool(as_f16) == d16



@pytest.mark.parametrize("export,depth", [
    ("exp mrtz v2, off, off, off vm", True),               # Z only (SPI_SHADER_32_R)
    ("exp mrtz v2, v3, off, off vm", True),                # Z and stencil (32_GR)
    ("exp mrtz off, v3, off, off vm", False),              # stencil only (32_GR)
    ("exp mrtz off, off, v3, off vm", False),              # sample mask only, 32-bit
    ("exp mrtz v2, v2, v3, v3 compr vm", False),           # stencil and sample mask, 16-bit (UINT16_ABGR)
], ids=["z", "z-stencil", "stencil", "mask", "compr"])
def test_mrtz_depth_only_from_an_enabled_x(make_shader, export, depth):
    """exp mrtz carries Z in x only when x is on in an uncompressed export; the 16-bit form holds stencil and sample
    mask, never Z (Mesa ac_nir_lower_ps_late.c emit_ps_mrtz_export). Without a depth the shader must not replace it."""
    shader = make_shader("v_mov_b32 v2, 0.5\nv_mov_b32 v3, 1\n" + export + "\nexp mrt0 v2, v2, v2, v2 done vm\n"
                         "s_endpgm\n", "ps_mrtz")
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert ("BuiltIn FragDepth" in spvasm) == depth
    assert ("DepthReplacing" in spvasm) == depth


@pytest.mark.parametrize("body", [
    "",                                                              # straight line (SHADER-350)
    "s_cmp_eq_u32 s20, 0\ns_cbranch_scc1 1\nv_mov_b32 v0, 1.0\n",      # branching control flow
], ids=["straight-line", "branch"])
def test_pops_interlock_begins_and_ends_once(make_shader, body):
    """A PS that reads src_pops_exiting_wave_id runs in one fragment-shader interlock: one
    OpBeginInvocationInterlockEXT, then one OpEndInvocationInterlockEXT (SPV_EXT_fragment_shader_interlock: each
    exactly once, begin before end). Validated against spirv-val."""
    shader = make_shader("v_mov_b32 v0, 0.5\ns_mov_b32 s20, src_pops_exiting_wave_id\n" + body +
                         "exp mrt0 v0, v0, v0, v0 done vm\ns_endpgm\n", "ps_pops")
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "PixelInterlockOrderedEXT" in spvasm
    assert spvasm.count("OpBeginInvocationInterlockEXT") == 1
    assert spvasm.count("OpEndInvocationInterlockEXT") == 1
    assert spvasm.index("OpBeginInvocationInterlockEXT") < spvasm.index("OpEndInvocationInterlockEXT")

def test_constants_asked_for_late_are_defined(make_shader):
    """assemble writes the uint constants before the format tables, which then ask gen.cu for their own. In a shader
    whose only typed access is a d16 store nothing else used 128, so fmt_size_arr's length was %c80 with no
    definition and the module failed spirv-val."""
    shader = make_shader("v_mov_b32 v1, 0x9c004000\nv_mov_b32 v2, 0\n"
                         "buffer_store_format_d16_xy v1, v2, s[0:3], 0 offen\n"
                         "exp mrt0 v0, v0, v0, v0 done vm\ns_endpgm\n", "ps_late_constants")
    spvasm = shader.translate()
    assert re.search(r"%fmt_size_arr = OpTypeArray %uint %c80$", spvasm, re.M)
    defs = [m.start() for m in re.finditer(r"^\s*%c80 = OpConstant %uint 128$", spvasm, re.M)]
    assert len(defs) == 1 and defs[0] < spvasm.index("%fmt_size_arr = OpTypeArray")
    shader.validate(spvasm)


def test_a16_sample_gradients_use_full_vgprs(make_shader):
    path = os.path.join(ROOT, "runtime", "test", "shaders", "ps_texture_grad_a16.asm")
    with open(path, encoding="utf-8") as fixture:
        shader = make_shader(fixture.read(), "ps_texture_grad_a16")
    spvasm = shader.translate()
    sample = re.search(r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad (%\w+) (%\w+)$", spvasm, re.M)
    assert sample, "the explicit-gradient sample is missing"
    definitions = dict(re.findall(r"^\s*(%\w+) = (Op[^\n]+)", spvasm, re.M))

    # Trace dPdx and dPdy back to the fixture's full 32-bit derivative words, in operand order.
    for vector, registers in zip(sample.groups(), ((8, 9), (10, 11))):
        assert definitions[vector].startswith("OpCompositeConstruct %v2float ")
        components = definitions[vector].split()[2:]
        assert len(components) == 2
        for component, register in zip(components, registers):
            cast, float_type, word = definitions[component].split()
            assert (cast, float_type) == ("OpBitcast", "%float")
            load, word_type, pointer = definitions[word].split()
            assert (load, word_type) == ("OpLoad", "%uint")
            *access, index = definitions[pointer].split()
            assert access == ["OpAccessChain", "%_ptr_Function_uint", "%vgpr"]
            assert definitions[index] == f"OpConstant %uint {register}"
    shader.validate(spvasm)


def test_unknown_stage_is_rejected(make_shader):
    shader = make_shader(program("v_mov_b32 v0, 1.0"), name="xx_test")
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_compute_translates_and_validates(make_shader):
    shader = make_shader("v_mov_b32 v1, 1\ns_endpgm\n", name="cs_test")
    shader.validate(shader.translate())


def test_cs_store_packed_atomic_pointers(make_shader):
    """Packed stores need element zero of the physical dword array for Metal atomics."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_store.asm"), encoding="utf-8") as f:
        spvasm = make_shader(f.read(), "cs_store").translate()
    defs = dict(re.findall(r"^\s*(%\w+) = (Op[^\n]+)", spvasm, re.M))
    atomics = re.findall(r"= (OpAtomicAnd|OpAtomicOr) %uint (%\w+) ", spvasm)
    assert {op for op, ptr in atomics} == {"OpAtomicAnd", "OpAtomicOr"}
    for _, ptr in atomics:
        access = defs[ptr].split()
        assert access[:2] == ["OpAccessChain", "%_ptr_PSB_uint"]
        _, _, base, index = access
        assert defs[base].startswith("OpConvertUToPtr %_ptr_PSB_arr ")
        assert defs[index] == "OpConstant %uint 0"


def test_compute_thread_group_size_sgpr_is_rejected(make_shader):
    """TG_SIZE_EN (RSRC2 bit 10) adds an SGPR the translator doesn't model."""
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088 | 1 << 10, (64, 1, 1)))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("group", [(0, 1, 1), (2048, 1, 1)], ids=["empty", "over-1024"])
def test_compute_bad_group_size_is_rejected(make_shader, group):
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088, group))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_geometry_emit_cut_starts_a_new_triangle_strip(tmp_path: Path) -> None:
    asm = (Path(__file__).parent / "semantic" / "fixtures" / "gs_emit_cut.asm").read_text(encoding="utf-8")
    position = [(0, 1, 0xf)]
    dxbc = make_dxbc([signature_chunk(b"ISG1", position), signature_chunk(b"OSG1", position)])
    shader = Shader(tmp_path, "gs_emit_cut", asm, dxbc=dxbc)
    module = shader.translate()
    shader.validate(module)
    assert "OpExecutionMode %main OutputTriangleStrip" in module
    # EXEC stays on throughout the fixture. Each end terminates the current strip after its last vertex.
    strips: list[list[int]] = [[]]
    vertex = 0
    for event in re.findall(r"^\s*(OpEmitVertex|OpEndPrimitive)$", module, re.MULTILINE):
        if event == "OpEndPrimitive":
            strips.append([])
        else:
            strips[-1].append(vertex)
            vertex += 1
    assert strips == [[0, 1, 2], [3, 4, 5], []]
    assert len(re.findall(r"OpEmitVertex\s+OpEndPrimitive", module)) == 2  # both run under the same EXEC guard


def test_geometry_without_signatures_is_rejected(make_shader):
    """A gs needs ISG1/OSG1 to know which registers LDS holds and which outputs to write."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "gs_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_gs_packed_input_register_emits_one_variable(make_shader):
    """Packed ISG1 elements sharing a register must emit one %gs_inN variable and pass spirv-val."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8") as f:
        asm = f.read()
    dxbc = make_dxbc([
        signature_chunk(b"ISG1", [(0, 1, 0xF), (1, 0, 0x3), (1, 0, 0xC)]),
        signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)]),
    ])
    shader = make_shader(asm, "gs_tri_packed", dxbc=dxbc)
    spvasm = shader.translate()
    assert spvasm.count("%gs_in1 = OpVariable") == 1
    assert spvasm.count("OpDecorate %gs_in1 Location 1") == 1
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert meta["gs_inputs"] == [0, 1]
    assert meta["gs_input_locations"] == [1]
    shader.validate(spvasm)


def test_gs_conflicting_position_and_user_in_same_register_rejected(make_shader):
    """Sharing one GS input register between SV_Position and a user varying cannot map to one variable."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8") as f:
        asm = f.read()
    dxbc = make_dxbc([
        signature_chunk(b"ISG1", [(0, 1, 0x3), (0, 0, 0xC)]),
        signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)]),
    ])
    shader = make_shader(asm, "gs_tri_conflict", dxbc=dxbc)
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("kw", [dict(topology=1), dict(threads=(2048, 1, 1)), dict(max_vertices=0)],
                         ids=["not-triangles", "over-1024-threads", "no-vertices"])
def test_mesh_bad_psv0_is_rejected(make_shader, kw):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri", dxbc=make_dxbc([mesh_psv0_chunk(**kw)]))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_mesh_without_psv0_is_rejected(make_shader):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_wave32_readlane_writelane_mask_31(make_shader):
    """SHADER-52: wave32 lane select is & 31; wave64 keeps & 63."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_lane_wrap.asm"), encoding="utf-8") as f:
        w32 = make_shader(f.read(), name="cs_w32_lane_wrap", cs_header=(0x8088, (32, 1, 1)))
    w32_spv = w32.translate()
    w32.validate(w32_spv)
    assert "OpBitwiseAnd %uint" in w32_spv and "%c1f" in w32_spv  # 31
    # Every readlane/writelane mask in this wave32 program is 31, never 63.
    assert not re.search(r"OpBitwiseAnd %uint %\w+ %c3f\b", w32_spv)
    assert rdna2spirv.translate.last_gen.wave64 is False

    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w64x_readlane.asm"), encoding="utf-8") as f:
        w64 = make_shader(f.read(), name="cs_w64x_readlane")
    w64_spv = w64.translate()
    assert rdna2spirv.translate.last_gen.wave64 is True
    assert re.search(r"OpBitwiseAnd %uint %\w+ %c3f\b", w64_spv)  # 63


@pytest.mark.parametrize("threads", [96, 128], ids=["96-threads", "128-threads"])
def test_mesh_v5_seed_and_threads_constant(make_shader, threads):
    """Mesh prologue computes v5 = %wlin * NumThreads + %lidx and registers NumThreads before assembly (7b82460)."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        asm = f.read()
    dxbc = make_dxbc([mesh_psv0_chunk(threads=(threads, 1, 1), max_vertices=64, max_primitives=64)])
    shader = make_shader(asm, name=f"ms_v5_seed_{threads}", dxbc=dxbc)
    spvasm = shader.translate()
    cid = f"%c{threads:x}"
    assert f"{cid} = OpConstant %uint {threads}" in spvasm
    assert f"%wbase = OpIMul %uint %wlin {cid}" in spvasm
    assert "%v5v = OpIAdd %uint %wbase %lidx" in spvasm
    assert "%v5p = OpAccessChain %_ptr_Function_uint %vgpr %c5" in spvasm
    assert "OpStore %v5p %v5v" in spvasm
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
def test_mesh_cull_primitive_is_block_bool(make_shader):
    """CullPrimitiveEXT must be a Block bool member (VUID-07036); array-of-bool trips SPIRV-Tools ≤2025.1."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_v5_workgroup.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_v5_workgroup", dxbc=SAMPLE_DXBC["ms_v5_workgroup"])
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpDecorate %ms_cull_blk Block" in spvasm
    assert "OpMemberDecorate %ms_cull_blk 0 BuiltIn CullPrimitiveEXT" in spvasm
    assert "OpMemberDecorate %ms_cull_blk 0 PerPrimitiveEXT" in spvasm
    assert "%ms_cull_blk = OpTypeStruct %bool" in spvasm
    assert re.search(r"%ms_arr_cull = OpTypeArray %ms_cull_blk %c\w+", spvasm)
    assert "OpDecorate %ms_cull BuiltIn CullPrimitiveEXT" not in spvasm
    assert re.search(r"OpAccessChain %_ptr_Out_bool_ms %ms_cull %\w+ %c0", spvasm)


# ---- pos1..pos3: ClipDistance / CullDistance / misc exports (VS & MS) ----

def _vgpr_to_index_map(spvasm, array_var, ptr_type):
    """Trace each OpStore to `array_var[idx]` back to the VGPR index loaded from `%vgpr`."""
    import re
    vgpr_of = {}
    val_of = {}
    ptr_idx = {}
    stores = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c([0-9a-f]+)", line)
        if m:
            vgpr_of[m.group(1)] = int(m.group(2), 16)
            continue
        m = re.match(r"\s*(%\w+) = OpLoad %uint (%\w+)", line)
        if m and m.group(2) in vgpr_of:
            val_of[m.group(1)] = vgpr_of[m.group(2)]
            continue
        m = re.match(r"\s*(%\w+) = OpBitcast %float (%\w+)", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(rf"\s*(%\w+) = OpAccessChain {re.escape(ptr_type)} {re.escape(array_var)} %c(\d+)", line)
        if m:
            ptr_idx[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\s*(%\w+) = OpSelect %float %\w+ (%\w+) %\w+", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(r"\s*OpStore (%\w+) (%\w+)", line)
        if m and m.group(1) in ptr_idx and m.group(2) in val_of:
            stores[ptr_idx[m.group(1)]] = val_of[m.group(2)]
    return stores


def _ms_header_asm(pos_exports):
    return (
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n"
        "s_mov_b32 m0, 0x1003\n"
        "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)\n"
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "v_mov_b32 v9, 0x200400\n"
        "exp prim v9, off, off, off done\n"
        "s_endpgm\n"
    )


def _vs_header_asm(pos_exports):
    return (
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "s_endpgm\n"
    )


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_vs_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_vs_header_asm(exp_line), name="vs_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" not in spvasm
    assert "OpDecorate %out_clip BuiltIn ClipDistance" in spvasm
    assert f"%arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp))
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {i: 10 + i for i in range(n_comp)}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert meta["clip_distance_count"] == n_comp
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" not in meta["features"]
    assert 1 not in meta.get("outputs", [])


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_ms_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_ms_header_asm(exp_line), name="ms_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %ms_vtx_blk Block" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert f"%ms_arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Function_float %ms_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp)) * 2   # the export writes, then the copy to the output
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {i: 10 + i for i in range(n_comp)}
    # Copied to the mesh output one float at a time: SPIRV-Cross turns a whole-array store there into a C array
    # assignment that Metal rejects (SHADER-155).
    out_chains = re.findall(r"OpAccessChain %_ptr_Out_float_ms %ms_out_vtx %mli %c1 %c(\d+)", spvasm)
    assert [int(x) for x in out_chains] == list(range(n_comp))
    assert "OpLoad %ms_arr_clip" not in spvasm
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert 1 not in meta.get("outputs", [])
    assert 1 not in meta.get("params", [])


def test_vs_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """2 clip distances + 3 cull distances pack contiguously: pos1.xy=clip[0..1], pos1.zw=cull[0..1], pos2.x=cull[2]."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _vs_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="vs_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%arr_cull = OpTypeArray %float %c3" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    cull_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_cull %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]
    assert [int(x) for x in cull_chains] == [0, 1, 2]
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%out_cull", "%_ptr_Output_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert meta["clip_distance_count"] == 2
    assert meta["cull_distance_count"] == 3
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_vs_misc_pos1_shifts_clip_to_pos2(make_shader):
    """When OSG1 declares SV_RenderTargetArrayIndex (sysv=4), pos1 is the misc vector and ClipDistance starts at pos2."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 4, 0x1), (2, 2, 0x3)])])
    asm = _vs_header_asm("exp pos1 off, off, v14, off\nexp pos2 v10, v11, off, off done")
    shader = make_shader(asm, name="vs_misc_clip", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ShaderLayer" in spvasm
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %out_layer BuiltIn Layer" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]


def test_pos1_without_osg1_is_ignored(make_shader):
    """Without OSG1 declaring SV_ClipDistance/SV_CullDistance, pos1 exports are discarded in both VS and MS."""
    vs = make_shader(_vs_header_asm("exp pos1 v10, off, off, off done"), name="vs_no_osg1_pos1")
    vs_spv = vs.translate()
    vs.validate(vs_spv)
    assert "OpCapability ClipDistance" not in vs_spv
    assert "BuiltIn ClipDistance" not in vs_spv
    vs_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, vs_spv)
    assert "clip_distance_count" not in vs_meta
    assert "shaderClipDistance" not in vs_meta["features"]
    assert 1 not in vs_meta.get("outputs", [])

    ms = make_shader(_ms_header_asm("exp pos1 v10, off, off, off done"), name="ms_no_osg1_pos1",
                     dxbc=make_dxbc([mesh_psv0_chunk()]))
    ms_spv = ms.translate()
    ms.validate(ms_spv, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in ms_spv
    assert "BuiltIn ClipDistance" not in ms_spv
    assert "%ms_out_pos" in ms_spv and "%ms_vtx_blk" not in ms_spv
    ms_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, ms_spv)
    assert "shaderClipDistance" not in ms_meta["features"]
    assert 1 not in ms_meta.get("params", [])


def test_ms_dummy_all_off_pos1_does_not_emit_clip(make_shader):
    """An all-off dummy pos1 export in a mesh shader must not emit ClipDistance or change the output block."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x1)])])
    shader = make_shader(_ms_header_asm("exp pos1 off, off, off, off done"), name="ms_dummy_pos1", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in spvasm
    assert "%ms_out_pos" in spvasm
    assert "%ms_vtx_blk" not in spvasm


def test_ms_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """Mesh shader with 2 clip + 3 cull distances emits both arrays in %ms_vtx_blk and maps v10..v14."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _ms_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="ms_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 2 BuiltIn CullDistance" in spvasm
    assert "%ms_arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%ms_arr_cull_d = OpTypeArray %float %c3" in spvasm
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%ms_cull_dist", "%_ptr_Function_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_wave64_detected_from_mask_ops_without_exec_markers(make_shader):
    """#210 part 2 / SHADER-28: s_bcnt1_i32_b64 on ordinary SGPRs marks the shader wave64 (no exec_hi / saveexec)."""
    shader = make_shader(
        "; mattr: +wavefrontsize64,-wavefrontsize32\n"
        "v_and_b32_e32 v5, 5, v0\nv_cmp_ne_u32_e64 s[20:21], 0, v5\n"
        "s_bcnt1_i32_b64 s30, s[20:21]\nv_mov_b32 v1, s30\ns_endpgm\n",
        name="cs_mask_detect_ci",
    )
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64


def test_wave64_bfm64_and_half_cmp_emit_wave_ops(make_shader):
    """Wave64 s_bfm_b64 must use wave_lane() and reconstruct the 64-bit mask; 32-bit half ops must guard their half."""
    bfm = make_shader("s_mov_b64 exec, -1\ns_bfm_b64 s[6:7], 16, 8\ns_bcnt1_i32_b64 s8, s[6:7]\ns_endpgm\n",
                      name="cs_bfm64_ci")
    asm_bfm = bfm.translate()
    bfm.validate(asm_bfm)
    assert "OpGroupNonUniformBitwiseOr" in asm_bfm
    assert "OpLoad %v3uint %local_id" in asm_bfm and "OpLoad %uint %sg_lane" not in asm_bfm

    hcmp = make_shader("s_mov_b32 exec_lo, 0\ns_mov_b32 exec_hi, 0xffff\ns_mov_b64 s[6:7], exec\n"
                       "s_cmp_eq_u32 s6, 0\ns_cmp_lg_u32 s7, 0\ns_endpgm\n", name="cs_halfcmp_ci")
    asm_hcmp = hcmp.translate()
    hcmp.validate(asm_hcmp)
    # Low-half s_cmp_eq_u32: lane < 32 (%c20) guards OpSelect(..., %true) feeding OpGroupNonUniformAll.
    assert re.search(
        r"(%t\d+) = OpULessThan %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpSelect %bool \1 %t\d+ %true\n\s*"
        r"%t\d+ = OpGroupNonUniformAll %bool %c3 \2",
        asm_hcmp,
    ), "low-half s_cmp_eq_u32 did not guard OpGroupNonUniformAll with lane < 32"
    # High-half s_cmp_lg_u32: lane >= 32 (%c20) guards OpLogicalAnd feeding OpGroupNonUniformAny.
    assert re.search(
        r"(%t\d+) = OpUGreaterThanEqual %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpLogicalAnd %bool \1 %t\d+\n\s*"
        r"%t\d+ = OpGroupNonUniformAny %bool %c3 \2",
        asm_hcmp,
    ), "high-half s_cmp_lg_u32 did not guard OpGroupNonUniformAny with lane >= 32"

    # Copying vcc_lo / vcc_hi to a 32-bit SGPR preserves half identity for wave64 s_bcnt1_i32_b32.
    vcc_half = make_shader(
        "s_mov_b64 exec, -1\nv_cmp_gt_u32_e32 vcc_lo, 16, v0\n"
        "s_mov_b32 s6, vcc_lo\ns_bcnt1_i32_b32 s8, s6\n"
        "s_mov_b32 s7, vcc_hi\ns_bcnt1_i32_b32 s9, s7\ns_endpgm\n",
        name="cs_vcchalf_ci",
    )
    asm_vcc = vcc_half.translate()
    vcc_half.validate(asm_vcc)
    assert asm_vcc.count("OpBitCount %uint %c0") == 2


def test_wave64_bpermute_reads_zero_from_lanes_that_are_not_running(make_shader):
    """Native wave64 ds_bpermute: a source lane missing from a partial wave must read 0, not an undefined shuffle."""
    shader = make_shader("s_mov_b64 exec, -1\nv_lshlrev_b32_e32 v1, 2, v0\nds_bpermute_b32 v2, v1, v0\n"
                         "s_waitcnt lgkmcnt(0)\ns_endpgm\n", name="cs_bperm64_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64
    assert re.search(
        r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 %true\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \1 (%t\d+)\n\s*"
        r"(%t\d+) = OpSelect %uint \2 \3 %t\d+\n\s*"
        r"(%t\d+) = OpGroupNonUniformShuffle %uint %c3 %t\d+ \4\n\s*"
        r"(%t\d+) = OpSelect %uint \2 %cffffffff %c0\n\s*"
        r"%t\d+ = OpBitwiseAnd %uint \5 \6\n",
        asm,
    ), "wave64 ds_bpermute did not keep the shuffle only where the source lane is running"



def test_wave32_bpermute_zeroes_inactive_and_missing_source_lanes(make_shader):
    """Wave32 ds_bpermute (RDNA ISA 10.4.4): the source value is 0 where the source lane is off in EXEC, and the
    shuffle result is kept only where the source lane is running, as in the wave64 path."""
    shader = make_shader(
        "v_cmp_ne_u32_e32 vcc_lo, 0, v0\ns_and_saveexec_b32 s4, vcc_lo\n"
        "v_lshlrev_b32_e32 v1, 2, v0\nds_bpermute_b32 v2, v1, v0\ns_waitcnt lgkmcnt(0)\ns_endpgm\n",
        name="cs_bperm32_ci",
    )
    asm = shader.translate()
    shader.validate(asm)
    assert not rdna2spirv.translate.last_gen.wave64
    shuffle = re.search(
        r"(%t\d+) = OpSelect %uint (%t\d+) %t\d+ %c0\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 %true\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \3 (%t\d+)\n\s*"
        r"(%t\d+) = OpSelect %uint \4 \5 %t\d+\n\s*"
        r"(%t\d+) = OpGroupNonUniformShuffle %uint %c3 \1 \6\n\s*"
        r"(%t\d+) = OpSelect %uint \4 %cffffffff %c0\n\s*"
        r"(?P<result>%t\d+) = OpBitwiseAnd %uint \7 \8\n",
        asm,
    )
    assert shuffle, "wave32 ds_bpermute must mask inactive and missing source lanes"
    active_id = shuffle.group(2)
    assert (f"{active_id} = OpLoad %bool %active" in asm or f"OpStore %active {active_id}" in asm), \
        "source mask must read the active lane predicate"
    assert f"OpBitwiseAnd %uint {shuffle['result']} " in asm, "the destination write must also mask the shuffle"

WQM_MASK = "s_mov_b64 exec, -1\nv_cmp_gt_u32_e32 vcc_lo, 16, v0\ns_wqm_b64 s[6:7], vcc\ns_bcnt1_i32_b64 s8, s[6:7]\ns_endpgm\n"
WQM_DATA = "s_mov_b32 s20, 0x00800010\ns_mov_b32 s21, 1\ns_wqm_b64 s[22:23], s[20:21]\ns_add_u32 s24, s22, s23\ns_endpgm\n"


def test_wqm_on_a_lane_mask_gathers_the_quad_and_votes_scc(make_shader, monkeypatch):
    """s_wqm on a lane mask ORs the lane's bit with its quad's (two OpGroupNonUniformQuadSwap, horizontal then
    vertical) and SCC is the wave-wide any. Emulated, the four lanes of the quad come through %x_buf instead."""
    shader = make_shader(WQM_MASK, name="cs_wqm_mask_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "OpCapability GroupNonUniformQuad" in asm
    assert asm.count("OpGroupNonUniformQuadSwap %uint %c3") == 2
    assert re.search(r"OpGroupNonUniformQuadSwap %uint %c3 %t\d+ %c0\n(?:.*\n)*?.*OpGroupNonUniformQuadSwap %uint %c3 %t\d+ %c1\n", asm)
    assert "OpGroupNonUniformAny" in asm                   # SCC: any bit of the result in the wave
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    emulated = shader.translate()
    shader.validate(emulated)
    assert "OpGroupNonUniform" not in emulated and "GroupNonUniformQuad" not in emulated
    assert "%x_buf" in emulated


def test_wqm_on_data_is_a_bit_operation(make_shader):
    """A register that is not a lane mask is data: per 32-bit word (x | x>>1 | x>>2 | x>>3) & 0x11111111, times 15."""
    shader = make_shader(WQM_DATA, name="cs_wqm_data_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "OpGroupNonUniform" not in asm
    for k in (1, 2, 3):                                  # three shifts per word, two words (s_wqm_b64)
        assert len(re.findall(rf"OpShiftRightLogical %uint %t\d+ %c{k}\n", asm)) == 2
    assert len(re.findall(r"OpBitwiseAnd %uint %t\d+ %c11111111\n", asm)) == 2
    assert len(re.findall(r"OpIMul %uint %t\d+ %cf\n", asm)) == 2
    assert re.search(r"OpBitwiseOr %uint %t\d+ %t\d+\n(?:.*\n)*?.*OpStore %scc", asm)   # SCC = low | high != 0


def test_wqm_without_quad_operations_outside_fragment_and_compute(make_shader):
    """Quad operations are only guaranteed in fragment and compute shaders: a vertex shader shuffles from the four
    lanes of the quad instead."""
    shader = make_shader("v_cmp_gt_u32_e32 vcc_lo, 1, v0\ns_wqm_b32 s8, vcc_lo\nv_cndmask_b32_e64 v6, 0, 1, s8\n"
                         "exp pos0 v6, v6, v6, v6 done\ns_endpgm\n", name="vs_wqm_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "QuadSwap" not in asm and "GroupNonUniformQuad" not in asm
    assert asm.count("OpGroupNonUniformShuffle") == 4


def test_wave64_wqm_b32_changes_and_votes_only_its_half(make_shader):
    """s_wqm_b32 exec_hi in wave64 writes the lanes 32-63 of EXEC only (exec_half_write) and SCC votes over them."""
    shader = make_shader("s_mov_b32 exec_hi, 0x102\ns_wqm_b32 exec_hi, exec_hi\ns_cselect_b32 s9, 1, 0\ns_endpgm\n",
                         name="cs_wqm_half_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64
    assert re.search(r"(%t\d+) = OpUGreaterThanEqual %bool %t\d+ %c20\n(?:.*\n)*?.*OpLogicalAnd %bool \1 %t\d+\n"
                     r"(?:.*\n)*?.*OpGroupNonUniformAny", asm)


def test_wqm_in_a_pixel_shader_uses_the_fragment_quad(make_shader):
    shader = make_shader("v_cmp_gt_u32_e32 vcc_lo, 1, v0\ns_wqm_b32 s8, vcc_lo\nv_cndmask_b32_e64 v6, 0, 1, s8\n"
                         "exp mrt0 v6, v6, v6, v6 done vm\ns_endpgm\n", name="ps_wqm_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert asm.count("OpGroupNonUniformQuadSwap") == 2 and "OpCapability GroupNonUniformQuad" in asm


FLOAT_CONTROLS_CS = """\
v_mov_b32_e32 v1, 0x80000000
v_mul_f32_e32 v2, v1, v1
v_lshlrev_b32_e32 v3, 2, v0
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm
"""


@pytest.mark.parametrize("stage", ["ps", "cs"])
def test_float_controls_declared(stage, make_shader):
    """Every module declares SignedZeroInfNanPreserve for 32-bit floats, so the driver keeps NaN, inf and -0 as
    RDNA2 computes them (AMD folded x != x to false and turned -0 products into +0 without it; see
    runtime/test/shaders/cs_signed_zero_nan.asm), and DenormPreserve for 16-bit floats, so it keeps an f16 denormal
    instead of flushing it (NVIDIA read the 0x0040 of ps_sdwa_f16_const's `src0_sel:BYTE_1` as +0 without it,
    SHADER-526). Its index entry names the device properties the runtime checks."""
    body = FLOAT_CONTROLS_CS if stage == "cs" else program("v_mul_f32 v1, v0, v0")
    shader = make_shader(body, name=f"{stage}_float_controls")
    spv = shader.translate()
    shader.validate(spv)
    assert "OpCapability SignedZeroInfNanPreserve" in spv
    assert "OpExecutionMode %main SignedZeroInfNanPreserve 32" in spv
    assert "OpCapability DenormPreserve" in spv
    assert "OpExecutionMode %main DenormPreserve 16" in spv
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    assert meta["float_controls"] == ["shaderSignedZeroInfNanPreserveFloat32", "shaderDenormPreserveFloat16"]


def test_no_float_controls_leaves_them_out(make_shader, monkeypatch):
    """--no-float-controls (FLOAT_CONTROLS = False), for a driver without the properties: no capability, no
    execution mode, no index entry, and the module still validates."""
    monkeypatch.setattr(rdna2spirv, "FLOAT_CONTROLS", False)
    shader = make_shader(FLOAT_CONTROLS_CS, name="cs_no_float_controls")
    spv = shader.translate()
    shader.validate(spv)
    assert "SignedZeroInfNanPreserve" not in spv
    assert "DenormPreserve" not in spv
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    assert "float_controls" not in meta


EMULATE_SAMPLES = [p for p in SAMPLES if os.path.basename(p).startswith(("cs_", "ms_"))]


@pytest.mark.parametrize("path", EMULATE_SAMPLES, ids=[os.path.basename(p) for p in EMULATE_SAMPLES])
def test_wave64_emulate_uses_no_subgroup_ops(path, make_shader, monkeypatch):
    """--wave64-emulate: a wave64 compute or mesh shader validates and uses no subgroup operation (everything
    cross-lane goes through %x_buf); a wave32 one comes out exactly as without the flag."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("ms_") and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic PSV0 for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), stem, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    plain = shader.translate()
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    emulated = shader.translate()
    if not rdna2spirv.translate.last_gen.wave64:
        assert emulated == plain
        return
    shader.validate(emulated, KNOWN_VAL_ISSUES.get(stem))
    assert "OpGroupNonUniform" not in emulated and "%sg_lane" not in emulated
    assert "%x_buf" in emulated and emulated.count("OpControlBarrier") >= 2


# SHADER-91: a native wave32 compute shader whose s_barrier isn't in a top-level block runs in barrier rounds.
BARRIER_SAMPLES = ("cs_barrier_divergent", "cs_barrier_early_end", "cs_barrier_loop")


@pytest.mark.parametrize("stem", BARRIER_SAMPLES)
def test_nested_barrier_runs_in_barrier_rounds(stem, make_shader):
    """No OpControlBarrier is left among the blocks (the cases of the switch over %pc): s_barrier ends a segment
    there. The only two are before the round loop and at its end, where the whole workgroup is."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", stem + ".asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), stem)
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.control_flow == "barrier rounds"
    blocks = asm[asm.index("OpSwitch %pcv"):asm.index("%sw_default = OpLabel")]
    assert "OpControlBarrier" not in blocks and "OpStore %x_bar %true" in blocks
    assert asm.count("OpControlBarrier") == 2
    head, tail = asm.index("OpLoopMerge %loop_exit %loop_cont"), asm.index("%loop_cont = OpLabel")
    assert asm.index("OpControlBarrier") < head and asm.rindex("OpControlBarrier") > tail
    assert "OpAtomicIAdd" in asm[tail - 2000:tail] and "%x_buf = OpVariable" in asm


def test_top_level_barrier_keeps_the_plain_translation(make_shader):
    """A barrier in a block every invocation runs (the top level of the structured tree) stays one OpControlBarrier
    in place: no rounds for a shader that doesn't need them."""
    shader = make_shader("v_lshlrev_b32_e32 v2, 2, v0\nds_write_b32 v2, v0\ns_waitcnt lgkmcnt(0)\ns_barrier\n"
                         "v_readfirstlane_b32 s5, v0\ns_cmp_lg_u32 s5, 0\ns_cbranch_scc1 2\n"   # skips the 2-dword read
                         "ds_read_b32 v4, v2\ns_endpgm\n", "cs_barrier_top")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.control_flow == "structured"
    assert asm.count("OpControlBarrier") == 1 and "%x_bar" not in asm and "%x_buf" not in asm


def test_top_level_barrier_over_the_depth_bound_runs_in_barrier_rounds(make_shader, monkeypatch):
    """SHADER-427 review: the barrier-rounds decision uses the same capped tree as assemble(). A top-level barrier
    is uniform only while the tree is structured; once the tree nests deeper than MAX_STRUCTURED_DEPTH it is
    emitted in the dispatch loop, where the barrier sits in a switch case, so the shader has to run in rounds."""
    n = 4
    chain = "".join(f"s_cmp_eq_u32 s0, {i}\ns_cbranch_scc0 {3 + 2 * (n - i - 1)}\n" for i in range(n))
    shader = make_shader("v_lshlrev_b32_e32 v2, 2, v0\nds_write_b32 v2, v0\ns_waitcnt lgkmcnt(0)\ns_barrier\n"
                         + chain + "v_lshlrev_b32_e32 v3, 2, v0\nds_write_b32 v3, v0\ns_endpgm\n",
                         "cs_barrier_top_deep")
    structured = shader.translate()
    shader.validate(structured)
    gen = rdna2spirv.translate.last_gen
    assert gen.control_flow == "structured" and not gen.bar_rounds and gen.nesting > 1
    assert structured.count("OpControlBarrier") == 1 and "%x_bar" not in structured

    monkeypatch.setattr(rdna2spirv, "MAX_STRUCTURED_DEPTH", gen.nesting - 1)
    rounds = shader.translate()
    shader.validate(rounds)
    gen = rdna2spirv.translate.last_gen
    assert gen.bar_rounds and gen.control_flow == "barrier rounds"
    blocks = rounds[rounds.index("OpSwitch %pcv"):rounds.index("%sw_default = OpLabel")]
    assert "OpControlBarrier" not in blocks and "OpStore %x_bar %true" in blocks


def test_barrier_rounds_are_wave32_compute_only(make_shader, monkeypatch):
    """Native wave64 (AMD's 64-wide subgroups) keeps the barrier where s_barrier is; under --wave64-emulate the
    emulated round loop already holds the only barrier."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_barrier_divergent.asm"), encoding="utf-8") as f:
        src = f.read()
    shader = make_shader("s_mov_b64 exec, -1\n" + src, "cs_barrier_divergent_w64")   # a 64-bit EXEC write: wave64
    asm = shader.translate()
    gen = rdna2spirv.translate.last_gen
    assert gen.wave64 and not gen.bar_rounds and gen.control_flow == "structured"
    assert asm.count("OpControlBarrier") == 2 and "%x_bar" not in asm
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    shader.translate()
    gen = rdna2spirv.translate.last_gen
    # #2089: --wave64-emulate keeps structured control flow where it can (was a round loop per barrier)
    assert gen.emulate and not gen.bar_rounds and gen.control_flow in ("emulated wave64 structured",
                                                                       "emulated wave64 rounds")


# SHADER-36: image_gather4_l(_o) runs every texel of its 2x2 footprint through the S#'s CLAMP_X/CLAMP_Y.
def _gather_l(s_word0, dynamic=False, s_word1=0):
    return program(
        "s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 0\ns_mov_b32 s11, 0\n"
        "s_mov_b32 s12, 0\ns_mov_b32 s13, 0\ns_mov_b32 s14, 0\ns_mov_b32 s15, 0\n"
        f"s_mov_b32 s16, {s_word0}\ns_mov_b32 s17, {s_word1}\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        # A loaded word, not s16 + 0: SHADER-315 folds scalar ALU on constants, which made the sampler a literal.
        + ("s_load_dword s16, s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n" if dynamic else "")
        + "v_mov_b32 v10, 0.5\nv_mov_b32 v11, 0.5\nv_mov_b32 v12, 0\n"
        "image_gather4_l v[0:3], v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D")


def _count(text, op):
    return len(re.findall(rf"= {op} ", text))


def _gather_address_bit_offsets(text: str) -> list[int]:
    constants = {name: int(value) for name, value in re.findall(r"(%\w+) = OpConstant %uint (\d+)", text)}
    fields = re.findall(r"OpBitFieldUExtract %uint %\w+ (%\w+) (%\w+)", text)
    return [constants[offset] for offset, width in fields if constants[width] == 3]


@pytest.mark.parametrize("mx", range(4))
@pytest.mark.parametrize("my", range(4))
def test_gather4_l_literal_sampler_emits_only_the_code_its_modes_need(mx, my, make_shader):
    """CLAMP_X = bits [2:0], CLAMP_Y = [5:3]; 0 wrap, 1 mirror, 2 clamp, 3 mirror once. Each axis maps its two
    texel coordinates: wrap = 1 OpSRem + 1 OpSLessThan (the modulo's fix-up), mirror = the same + 1 OpSLessThan
    (the reflection), mirror once = 1 OpSGreaterThanEqual. No OpSMod: AMD's Windows driver gave it the dividend's
    sign for a negative coordinate (Gen.gather_addr)."""
    shader = make_shader(_gather_l(mx | my << 3))
    text = shader.translate()
    shader.validate(text)
    assert _count(text, "OpSMod") == 0
    assert _count(text, "OpSRem") == 2 * (mx in (0, 1)) + 2 * (my in (0, 1))
    assert _count(text, "OpSLessThan") == 2 * (mx in (0, 1)) + 2 * (mx == 1) + 2 * (my in (0, 1)) + 2 * (my == 1)
    assert _count(text, "OpSGreaterThanEqual") == 2 * (mx == 3) + 2 * (my == 3)
    assert _gather_address_bit_offsets(text) == []      # the sampler modes are literals


def _gather_rounded_lod(defs):
    """The LOD the explicit-LOD gather rounds to a mip: the operand of the Floor whose fraction (an FSub from the same
    LOD) is compared with %fhalf_g."""
    cmp, = (e for e in defs.values() if e.startswith("OpFOrdGreaterThanEqual %bool ") and e.endswith(" %fhalf_g"))
    sub = defs[cmp.split()[2]].split()
    assert sub[:2] == ["OpFSub", "%float"], sub
    lod, floor = sub[2], sub[3]
    assert defs[floor].split() == ["OpExtInst", "%float", "%glsl", "Floor", lod]
    return lod


def test_gather4_l_literal_lod_bounds_feed_mip_selection(make_shader):
    shader = make_shader(_gather_l(0x12, s_word1=(192 << 12) | 64))  # [0.25, 0.75] in unsigned 4.8
    text = shader.translate()
    shader.validate(text)
    defs = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
    clamp = defs[_gather_rounded_lod(defs)].split()
    assert clamp[:4] == ["OpExtInst", "%float", "%glsl", "FClamp"]
    for limit, bits in zip(clamp[-2:], (0x3e800000, 0x3f400000)):
        cast, scalar, value = defs[limit].split()
        assert (cast, scalar) == ("OpBitcast", "%float")
        assert defs[value] == f"OpConstant %uint {bits}"


def test_gather4_l_run_time_lod_bounds_clamp_without_fclamp(make_shader):
    # A run-time S# here (the real game shape) must not use FClamp: LLPC 26.10.1 miscompiles it on this path and
    # leaves the gather at mip 0 when MIN_LOD should raise it (#1032). FMax then FMin is the same clamp.
    asm = ("s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 0\ns_mov_b32 s11, 0\n"
           "s_mov_b32 s12, 0\ns_mov_b32 s13, 0\ns_mov_b32 s14, 0\ns_mov_b32 s15, 0\n"
           "s_load_dwordx4 s[16:19], s[0:1], 0\n"
           "v_mov_b32 v10, 0.5\nv_mov_b32 v11, 0.5\nv_mov_b32 v12, 0\n"
           "image_gather4_l v[0:3], v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D")
    shader = make_shader(program(asm))
    text = shader.translate()
    shader.validate(text)
    defs = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
    # The LOD rounds to the nearest mip, ties up (SHADER-152); the lod it rounds is the clamp's FMin.
    clamp = defs[_gather_rounded_lod(defs)].split()
    assert clamp[:4] == ["OpExtInst", "%float", "%glsl", "FMin"]
    assert "FMax" in text
    assert "FClamp" not in text


def test_gather4_l_run_time_sampler_selects_the_mode_from_the_sgpr(make_shader):
    shader = make_shader(_gather_l(0x12, dynamic=True))
    text = shader.translate()
    shader.validate(text)
    assert _gather_address_bit_offsets(text) == [0, 3]      # CLAMP_X and CLAMP_Y
    # Four coordinates, each through all four modes: wrap and mirror an OpSRem and a fix-up OpSLessThan each (8),
    # mirror one more OpSLessThan (4), SHADER-348 border-mode range tests two OpSLessThan each (8), mirror once an
    # OpSGreaterThanEqual (4).
    assert _count(text, "OpSMod") == 0 and _count(text, "OpSRem") == 8
    assert _count(text, "OpSLessThan") == 20 and _count(text, "OpSGreaterThanEqual") == 4


@pytest.mark.parametrize("word0", [4, 0x28, 0x8000])
def test_gather4_l_literal_half_border_or_unnormalized_sampler_is_unsupported(word0, make_shader):
    with pytest.raises(rdna2spirv.Unsupported):
        make_shader(_gather_l(word0)).translate()


# SHADER-348: modes 6 (clamp border) and 7 (mirror once border) give a tap outside the image the border colour's
# channel, BORDER_COLOR_TYPE = S# word 3 [31:30] (0 transparent black, 1 opaque black, 2 opaque white, 3 table).
def _gather_l_border(word0, word3, dynamic=False, dmask=1):
    return program(
        "s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 0\ns_mov_b32 s11, 0\n"
        "s_mov_b32 s12, 0\ns_mov_b32 s13, 0\ns_mov_b32 s14, 0\ns_mov_b32 s15, 0\n"
        f"s_mov_b32 s16, {word0}\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, {word3}\n"
        + ("s_load_dword s16, s[0:1], 0x0\ns_load_dword s19, s[0:1], 0x4\ns_waitcnt lgkmcnt(0)\n"
           if dynamic else "")
        + "v_mov_b32 v10, 0.5\nv_mov_b32 v11, 0.5\nv_mov_b32 v12, 0\n"
        f"image_gather4_l v[0:3], v[10:12], s[8:15], s[16:19] dmask:{dmask:#x} dim:SQ_RSRC_IMG_2D")


def _float_bits(text, name):
    defs = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
    expr = defs[name].split()
    if expr[:2] == ["OpBitcast", "%float"]:
        expr = defs[expr[2]].split()
    assert expr[:2] in (["OpConstant", "%uint"], ["OpConstant", "%float"]), expr
    return int(expr[2]) if expr[1] == "%uint" else struct.unpack("<I", struct.pack("<f", float(expr[2])))[0]


@pytest.mark.parametrize("word0, taps", [(0x6, 4), (0x30, 4), (0x36, 4), (0x7, 4), (0x38, 4)])
# Word 3 keeps BORDER_COLOR_PTR = 0x100 so llvm-mc prints it as an integer: 0x40000000 alone is the inline float 2.0,
# which the translator reads at run time (covered by the run-time test below).
@pytest.mark.parametrize("word3, dmask, one", [(0x100, 1, False), (1 << 30 | 0x100, 1, False), (1 << 30 | 0x100, 8, True),
                                               (2 << 30 | 0x100, 1, True)])
def test_gather4_l_literal_border_mode_selects_the_border_colour(word0, taps, word3, dmask, one, make_shader):
    shader = make_shader(_gather_l_border(word0, word3, dmask=dmask))
    text = shader.translate()
    shader.validate(text)
    selects = re.findall(r"= OpSelect %float %\w+ (%\w+) %\w+$", text, re.M)
    assert len(selects) == taps                     # each of the four taps can leave the image
    assert {_float_bits(text, s) for s in selects} == {0x3F800000 if one else 0}


def test_gather4_l_literal_border_colour_from_the_table_is_unsupported(make_shader):
    with pytest.raises(rdna2spirv.Unsupported):
        make_shader(_gather_l_border(0x36, 3 << 30 | 0x100)).translate()


def test_gather4_l_run_time_sampler_reads_the_border_type_from_word3(make_shader):
    shader = make_shader(_gather_l_border(0x36, 2 << 30, dynamic=True))
    text = shader.translate()
    shader.validate(text)
    assert re.search(r"= OpShiftRightLogical %uint %\w+ (%\w+)$", text, re.M)
    assert len(re.findall(r"= OpSelect %float %\w+ %\w+ %\w+$", text, re.M)) >= 4


# SHADER-152: the explicit-LOD gather's mip is the nearest one, ties up, like the native gather on Radeon
# (LOD 0.5 with bounds [0,1] picked mip 1); the old RoundEven picked mip 0 there. It is computed as
# floor(LOD) + (fraction >= 0.5), exact in binary32, not Floor(LOD + 0.5), whose sum rounds 0x3effffff up to 1.0.
def _gather_l_at_lod(lod, word1=0x100000):
    return program(
        "s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 0\ns_mov_b32 s11, 0\n"
        "s_mov_b32 s12, 0\ns_mov_b32 s13, 0\ns_mov_b32 s14, 0\ns_mov_b32 s15, 0\n"
        # S# word 1: by default MIN_LOD 0, MAX_LOD 1.0 (unsigned 4.8), so the clamp below is [0, 1].
        f"s_mov_b32 s16, 0\ns_mov_b32 s17, {word1:#x}\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        f"v_mov_b32 v10, 0.5\nv_mov_b32 v11, 0.5\nv_mov_b32 v12, {lod}\n"
        "image_gather4_l v[0:3], v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D")


def test_gather4_l_rounds_the_lod_tie_up(make_shader):
    """The LOD is clamped to the S# bounds first, then floored, and its fraction compared with 0.5; the result is
    not rounded to even (which would send a 0.5 tie to mip 0), and no +0.5 is added in binary32."""
    shader = make_shader(_gather_l_at_lod(0.5))
    text = shader.translate()
    shader.validate(text)
    defs = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
    clamp = re.search(r"(%\w+) = OpExtInst %float %glsl FClamp (%\w+) (%\w+) (%\w+)", text)
    assert clamp, text
    assert _gather_rounded_lod(defs) == clamp.group(1)  # the clamped LOD, not the raw one
    assert "= OpExtInst %float %glsl RoundEven" not in text
    assert not re.search(r"= OpFAdd %float %\w+ %fhalf_g", text), text


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _translated_gather_mip(text, lod, levels):
    """Evaluate the translated mip choice (the level operand of OpImageQuerySizeLod) for a raw LOD, with every float
    result rounded to binary32 as the GPU computes it. The raw LOD is the first operand of the literal S#'s FClamp;
    QueryLevels gives levels."""
    defs = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
    raw, = re.findall(r"= OpExtInst %float %glsl FClamp (%\w+) %\w+ %\w+$", text, re.M)
    env = {raw: _f32(lod), "%fhalf_g": 0.5}

    def ev(name):
        if name in env:
            return env[name]
        e = defs[name].split()
        op, args = e[0], e[2:]
        if op == "OpConstant":
            v = float(args[0]) if e[1] == "%float" else int(args[0], 0)
            return v - (1 << 32) if e[1] == "%int" and v >= 1 << 31 else v
        if op == "OpBitcast":
            v = ev(args[0])
            if e[1] == "%float":
                return _f32_bits(v & 0xFFFFFFFF)
            return v if e[1] == "%int" else v & 0xFFFFFFFF
        if op == "OpImageQueryLevels":
            return levels
        if op == "OpExtInst":
            fn, a = args[1], [ev(x) for x in args[2:]]
            out = {"Floor": lambda: float(math.floor(a[0])), "FClamp": lambda: min(max(a[0], a[1]), a[2]),
                   "FMax": lambda: max(a), "FMin": lambda: min(a), "SMax": lambda: max(a), "SMin": lambda: min(a),
                   "SClamp": lambda: min(max(a[0], a[1]), a[2])}[fn]()
            return _f32(out) if isinstance(out, float) else out
        a = [ev(x) for x in args]
        return {"OpFAdd": lambda: _f32(a[0] + a[1]), "OpFSub": lambda: _f32(a[0] - a[1]),
                "OpIAdd": lambda: a[0] + a[1], "OpISub": lambda: a[0] - a[1],
                "OpConvertFToS": lambda: int(a[0]), "OpFOrdGreaterThanEqual": lambda: a[0] >= a[1],
                "OpSelect": lambda: a[1] if a[0] else a[2]}[op]()

    level, = re.findall(r"= OpImageQuerySizeLod %v2int %\w+ (%\w+)$", text, re.M)
    return ev(level)


@pytest.mark.parametrize("word1, lod, mip", [
    (0x100000, _f32_bits(0x3effffff), 0),   # just below 0.5: Floor(lod + 0.5) rounded the sum to 1.0 and gave mip 1
    (0x100000, 0.5, 1), (0x100000, _f32_bits(0x3f000001), 1), (0x100000, 0.25, 0), (0x100000, 0.75, 1),
    (0x100000, -1.0, 0), (0x100000, 2.0, 1),                    # clamped to the S# bounds [0, 1] first
    (0xfff000, _f32_bits(0x3fbfffff), 1), (0xfff000, 1.5, 2), (0xfff000, 2.5, 3),
    (0xfff000, _f32_bits(0x401fffff), 2), (0xfff000, 9.0, 3),   # the last clamped to levels - 1
])
def test_gather4_l_mip_is_the_nearest_level_in_binary32(word1, lod, mip, make_shader):
    shader = make_shader(_gather_l_at_lod(0, word1=word1))
    text = shader.translate()
    shader.validate(text)
    assert _translated_gather_mip(text, lod, 4) == mip


def test_gather4_l_binary32_sum_rounds_just_below_half_up():
    # Negative control: the old Floor(lod + 0.5) in binary32 gives mip 1 for 0x3effffff (the sum is halfway between
    # 1 - 2^-24 and 1.0 and rounds to even); computed exactly it is mip 0.
    lod = _f32_bits(0x3effffff)
    assert math.floor(_f32(lod + 0.5)) == 1
    assert math.floor(lod) + (lod - math.floor(lod) >= 0.5) == 0


def _gather_mip(lod, levels):
    """Reference for the translated sequence: clamp to [0, levels-1] after floor(lod + 0.5)."""
    return min(max(math.floor(lod + 0.5), 0), levels - 1)


@pytest.mark.parametrize("lod, levels, mip", [
    (-2.0, 4, 0), (-1.5, 4, 0), (-0.5, 4, 0), (-0.1, 4, 0),
    (0.0, 4, 0), (0.25, 4, 0), (0.499, 4, 0),
    (0.5, 4, 1), (0.501, 4, 1), (0.75, 4, 1),
    (1.5, 4, 2), (2.5, 4, 3), (3.5, 4, 3), (3.6, 4, 3),     # the last two are clamped to levels-1
    (0.5, 2, 1), (1.5, 2, 1), (10.0, 1, 0),
])
def test_gather_mip_rounds_ties_up_and_clamps_to_the_level_range(lod, levels, mip):
    assert _gather_mip(lod, levels) == mip


def test_gather_mip_old_round_even_differs_only_at_ties():
    # Negative control for the rule: Python's round() is round-half-to-even, the old translator's choice.
    assert (_gather_mip(0.5, 4), round(0.5)) == (1, 0)
    assert (_gather_mip(2.5, 4), round(2.5)) == (3, 2)
    assert (_gather_mip(1.5, 4), round(1.5)) == (2, 2)
# ---- Irreducible CFG node splitting (compute shaders) ----

IRREDUCIBLE_PATTERNS = {
    # 2-entry SCC with s_branch + s_cbranch_scc0 (all 1-dword insns; offsets are relative dwords from next_pc)
    "scc0_branch": """\
s_mov_b32 s1, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc0 3
s_add_u32 s1, s1, 1
s_cmp_lt_u32 s1, 4
s_cbranch_scc0 4
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 8
s_cbranch_scc0 1
s_branch -7
v_mov_b32 v1, s1
s_endpgm
""",
    # 2-entry SCC with s_cbranch_scc1
    "scc1": """\
s_mov_b32 s1, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 3
s_add_u32 s1, s1, 1
s_cmp_ge_u32 s1, 4
s_cbranch_scc1 3
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 8
s_cbranch_scc1 -6
v_mov_b32 v1, s1
s_endpgm
""",
    # 2-entry SCC with s_cbranch_vccnz
    "vccnz": """\
s_mov_b32 s1, 0
v_cmp_gt_u32_e32 vcc_lo, 4, v0
s_cbranch_vccnz 3
s_add_u32 s1, s1, 1
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_cbranch_vccnz 3
s_add_u32 s1, s1, 2
v_cmp_lt_u32_e32 vcc_lo, 2, v0
s_cbranch_vccnz -6
v_mov_b32 v1, s1
s_endpgm
""",
    # Nested irreducible cycle {A, B} inside an outer reducible loop headed by H (pc=2)
    "nested_in_reducible": """\
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_add_u32 s2, s2, 1
s_cmp_eq_u32 s0, 0
s_cbranch_scc0 3
s_add_u32 s1, s1, 1
s_cmp_eq_u32 s1, 3
s_cbranch_scc1 3
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 6
s_cbranch_scc1 -6
s_cmp_lt_u32 s2, 4
s_cbranch_scc1 -11
v_mov_b32 v1, s1
s_endpgm
""",
}


@pytest.mark.parametrize("pattern", sorted(IRREDUCIBLE_PATTERNS))
def test_irreducible_cfg_node_splitting(pattern, make_shader, monkeypatch):
    """Irreducible CFGs (2-entry SCCs and nested irreducible cycles) structure via node splitting and pass spirv-val."""
    asm = IRREDUCIBLE_PATTERNS[pattern]
    shader = make_shader(asm, name=f"cs_irred_{pattern}")

    # Without node splitting, the shader genuinely falls back to the dispatch loop.
    monkeypatch.setattr(rdna2spirv.cfg_structure, "split_irreducible", lambda succ, **kw: None)
    spv_fallback = shader.translate()
    meta_fallback = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv_fallback)
    assert meta_fallback["control_flow"] == "dispatch loop"
    assert "OpSwitch" in spv_fallback

    # With node splitting enabled, the shader becomes structured and passes spirv-val.
    monkeypatch.undo()
    spv_struct = shader.translate()
    shader.validate(spv_struct)
    meta_struct = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv_struct)
    assert meta_struct["control_flow"] == "structured"
    assert "OpLoopMerge" in spv_struct
    assert "OpSwitch" not in spv_struct


def test_deep_structured_nesting_uses_the_dispatch_loop(make_shader, monkeypatch):
    """SHADER-299: a tree nested deeper than MAX_STRUCTURED_DEPTH is emitted in the flat dispatch-loop form, which
    validates at the same depth in a fraction of the time; raising the bound brings the structured form back."""
    # cfg_structure hoists tails past 16 guards deep (#1904), so a chain never structures deeper than 16; the bound is
    # set below that here so the chain still exceeds it.
    monkeypatch.setattr(rdna2spirv, "MAX_STRUCTURED_DEPTH", 8)
    n = 24
    body = "".join(f"s_cmp_eq_u32 s0, {i}\ns_cbranch_scc0 {3 + 2 * (n - i - 1)}\n" for i in range(n))
    shader = make_shader(body + "v_mov_b32 v8, s0\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm\n",
                         name="ps_nesting_bound")
    asm = shader.translate()
    shader.validate(asm)
    gen = rdna2spirv.translate.last_gen
    assert gen.nesting > rdna2spirv.MAX_STRUCTURED_DEPTH
    assert gen.control_flow == "dispatch loop" and "OpSwitch %pcv" in asm

    monkeypatch.setattr(rdna2spirv, "MAX_STRUCTURED_DEPTH", gen.nesting)
    structured = shader.translate()
    shader.validate(structured)
    assert rdna2spirv.translate.last_gen.control_flow == "structured"
    assert "OpSwitch" not in structured

    # SHADER-427: 0 (--max-structured-depth 0) never falls back, whatever the depth.
    monkeypatch.setattr(rdna2spirv, "MAX_STRUCTURED_DEPTH", 0)
    unbounded = shader.translate()
    shader.validate(unbounded)
    assert rdna2spirv.translate.last_gen.control_flow == "structured"
    assert rdna2spirv.translate.last_gen.nesting == gen.nesting
    assert unbounded == structured


MASK_REUSE_HEAD = "; mattr: +wavefrontsize64,-wavefrontsize32\nv_mbcnt_lo_u32_b32 v1, -1, 0\nv_mbcnt_hi_u32_b32 v1, -1, v1\n"


def emulated_exchanges(make_shader, monkeypatch, body, name="cs_reuse"):
    """Translate `body` (wave64 compute) with --wave64-emulate; return (SPIR-V, number of mask-word ORs). Every
    whole-mask exchange ORs two words (lanes 0-31 and 32-63), so the count is two per exchange."""
    shader = make_shader(MASK_REUSE_HEAD + body + "s_endpgm\n", name)
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    asm = shader.translate()
    shader.validate(asm)
    return asm, asm.count("OpAtomicOr")


def test_wave64_emulate_reuses_a_mask_exchange(make_shader, monkeypatch):
    """The same register value (or EXEC) read as a whole mask twice in a block is exchanged once; a rewrite of the
    register or of EXEC gets a new exchange."""
    cmp = "v_cmp_eq_u32_e64 s[4:5], 5, v1\n"
    _, same = emulated_exchanges(make_shader, monkeypatch, cmp + "s_bcnt1_i32_b64 s6, s[4:5]\ns_bcnt1_i32_b64 s7, s[4:5]\n")
    _, rewritten = emulated_exchanges(
        make_shader, monkeypatch, cmp + "s_bcnt1_i32_b64 s6, s[4:5]\nv_cmp_gt_u32_e64 s[4:5], 40, v1\ns_bcnt1_i32_b64 s7, s[4:5]\n")
    _, same_exec = emulated_exchanges(make_shader, monkeypatch, "s_bcnt1_i32_b64 s6, exec\ns_bcnt1_i32_b64 s7, exec\n")
    _, new_exec = emulated_exchanges(
        make_shader, monkeypatch, "s_bcnt1_i32_b64 s6, exec\nv_cmpx_gt_u32_e32 10, v1\ns_bcnt1_i32_b64 s7, exec\n")
    _, pair = emulated_exchanges(  # s4 and s5 hold the same value: one exchange serves v_mbcnt_lo(s4) and _hi(s5)
        make_shader, monkeypatch, cmp + "v_mbcnt_lo_u32_b32 v3, s4, 0\nv_mbcnt_hi_u32_b32 v3, s5, v3\n")
    _, half = emulated_exchanges(  # s5 rewritten, s4 kept: two exchanges
        make_shader, monkeypatch,
        cmp + "v_mbcnt_lo_u32_b32 v3, s4, 0\nv_cmp_gt_u32_e64 s[6:7], 40, v1\ns_and_b32 s5, s5, s7\n"
        "v_mbcnt_hi_u32_b32 v3, s5, v3\n")
    assert (same, rewritten, same_exec, new_exec, pair, half) == (2, 4, 2, 4, 2, 4)


def test_mesh_s3_wave_id_emits_local_index_shift_and_bfe(make_shader):
    """Mesh shader prologue initializes s3[27:24] from LocalInvocationIndex >> 6 and s_bfe_u32 extracts it."""
    path = os.path.join(ROOT, "runtime", "test", "shaders", "ms_wave_info_s3.asm")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_wave_info_s3", dxbc=SAMPLE_DXBC["ms_wave_info_s3"])
    asm = shader.translate()
    shader.validate(asm)
    assert "%wave = OpShiftRightLogical %uint %lidx %c6" in asm
    assert "%mwi = OpShiftLeftLogical %uint %wave %c18" in asm
    assert "OpStore %s3p %mwi" in asm
    assert "OpBitFieldUExtract %uint" in asm


PK_F16_REPRO_513 = """\
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v27, 0xbdd53e00
v_mov_b32_e32 v32, 0x3d3d32a5
v_and_b32_e32 v7, 7, v0
v_lshlrev_b32_e32 v7, 7, v7
v_or_b32_e32 v20, 0x3c00, v7
v_pk_add_f16 v31, v32, v20 op_sel:[0,0] op_sel_hi:[0,0]
v_pk_mul_f16 v34, v27, v31 op_sel:[0,0] op_sel_hi:[0,0]
v_lshlrev_b32_e32 v1, 2, v0
buffer_store_dword v34, v1, s[0:3], 0 offen
v_add_nc_u32_e32 v2, 256, v1
buffer_store_dword v31, v2, s[0:3], 0 offen
s_endpgm
"""


@pytest.mark.parametrize("emulate", [False, True], ids=["wave64", "wave64-emulate"])
def test_pk_f16_int_pack_repro_513(emulate, make_shader, monkeypatch):
    """SHADER-70 (#513): v_pk_*_f16 packs with integer IEEE-754 RTE instead of PackHalf2x16 so drivers cannot fold
    UnpackHalf2x16(PackHalf2x16(x)) back to unrounded f32, and produces exact RTE bits on the #513 repro."""
    import sys

    semantic_dir = os.path.join(ROOT, "tests", "semantic")
    if semantic_dir not in sys.path:
        sys.path.insert(0, semantic_dir)
    import spirv_mini_interp

    shader = make_shader(PK_F16_REPRO_513, name="cs_pk_f16_repro_513")
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", emulate)
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    sg_size = 32 if emulate else 64
    mem = spirv_mini_interp.run_workgroup(
        spv, 64, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
        memory={},
    )
    assert [mem[256 + 4 * i] for i in range(3)] == [0x3CD53CD5, 0x3D553D55, 0x3DD53DD5]
    assert [mem[4 * i] for i in range(3)] == [0x3F403F40, 0x40004000, 0x40604060]


@pytest.mark.parametrize("op_asm", [
    "v_pk_add_f16 v3, v1, v2",
    "v_pk_mul_f16 v3, v1, v2",
    "v_pk_max_f16 v3, v1, v2",
    "v_pk_min_f16 v3, v1, v2",
    "v_pk_fma_f16 v3, v1, v2, v4",
    "v_add_f16_e64 v3, v1, v2",
])
def test_pk_and_scalar_f16_ops_emit_no_packhalf2x16(op_asm, make_shader):
    """All v_pk_*_f16 instructions and scalar f16_out ops validate and avoid GLSL PackHalf2x16."""
    body = (
        "v_mov_b32 v1, 0x3c003800\n"
        "v_mov_b32 v2, 0x40003c00\n"
        "v_mov_b32 v3, 0x00000000\n"
        "v_mov_b32 v4, 0x38003400\n"
        f"{op_asm}\n"
        "v_mov_b32 v0, v3"
    )
    shader = make_shader(program(body))
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv


@pytest.mark.parametrize("body", [
    "buffer_load_format_d16_xyzw v[2:3], v1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)\nexp mrt0 v2, v3, v2, v3 done vm",
    "v_mov_b32 v2, 1.0\nv_mov_b32 v3, 0.5\nv_mov_b32 v4, 0.25\nv_mov_b32 v5, 2.0\n"
    "buffer_store_format_xyzw v[2:5], v1, s[0:3], 0 offen\nexp mrt0 v2, v3, v4, v5 done vm",
], ids=["d16-load", "store"])
def test_buffer_format_f16_narrowing_avoids_packhalf2x16(make_shader, body):
    """The f16 of a d16 typed buffer load, and of an f16 format a typed buffer store writes, narrow with integer RTE
    like the d16 image results (#489): PackHalf2x16 rounds toward zero on AMD drivers."""
    shader = make_shader("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0x100\ns_mov_b32 s3, 0x31001fac\n"
                         "v_mov_b32 v1, 0\n" + body + "\ns_endpgm\n", "ps_buf_f16")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv


def test_repeated_tbuffer_loads_share_normalization(make_shader):
    """A packed fixed format with 10- and 2-bit fields needs two exact quotients, regardless of load count."""
    load = "tbuffer_load_format_xyzw v[4:7], v0, s[0:3], 0 format:[BUF_FMT_2_10_10_10_UNORM] offen"
    shader = make_shader((load + "\ns_waitcnt vmcnt(0)\n") * 8 + "exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n",
                         "ps_repeated_tbuffer")
    spv = shader.translate()
    shader.validate(spv)
    assert spv.count("FindUMsb") == 2


def test_array_resinfo_queries_width_height_layers_and_mip_count(make_shader):
    shader = make_shader(
        "v_mov_b32_e32 v1, 1\n"
        "image_get_resinfo v[4:7], v1, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY\n"
        "s_waitcnt vmcnt(0)\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n",
        name="ps_array_resinfo",
    )
    asm = shader.translate()
    shader.validate(asm)
    assert "OpDecorate %images_arr Binding 3" in asm
    size = re.search(r"(%\w+) = OpImageQuerySizeLod %v3int (%\w+) %\w+", asm)
    assert size and f"OpImageQueryLevels %int {size[2]}" in asm
    for axis in range(3):
        assert f"OpCompositeExtract %int {size[1]} {axis}" in asm


class _ResinfoStub:
    """Just enough of the generator for Gen.resinfo_words; the lines are evaluated by _eval_resinfo."""
    def __init__(self):
        self.n, self.lines = 0, []

    def tmp(self):
        self.n += 1
        return f"%t{self.n}"

    def cu(self, v):
        return f"%c{v & 0xFFFFFFFF:x}"

    def emit(self, s):
        self.lines.append(s)

    def as_int(self, u):
        i = self.tmp()
        self.emit(f"{i} = OpBitcast %int {u}")
        return i


def _eval_resinfo(lines, view, level, word3):
    """view: (width, height, layers, levels) of the Vulkan view, which starts at the T#'s BASE_LEVEL. Values are
    Python ints (uint words); OpImageQuerySizeLod refuses a level the view doesn't have, as Vulkan leaves it undefined."""
    width, height, layers, levels = view
    env = {"%img": None, "%level": level, "%word3": word3}

    def val(x):
        return int(x[2:], 16) if x.startswith("%c") else env[x]
    for line in lines:
        dst, _, op, _type, *args = line.split()
        if op == "OpImageQueryLevels":
            env[dst] = levels
        elif op == "OpImageQuerySizeLod":
            lod = val(args[1])
            assert 0 <= lod < levels, f"OpImageQuerySizeLod at level {lod} of a {levels}-level view"
            env[dst] = (max(1, width >> lod), max(1, height >> lod), layers)
        elif op == "OpCompositeExtract":
            env[dst] = env[args[0]][int(args[1])]
        elif op == "OpExtInst":
            assert args[1] == "UMax"
            env[dst] = max(val(args[2]), val(args[3]))
        elif op == "OpBitFieldUExtract":
            env[dst] = (val(args[0]) >> val(args[1])) & ((1 << val(args[2])) - 1)
        else:
            a = [val(x) for x in args]
            env[dst] = {"OpBitcast": lambda: a[0], "OpULessThan": lambda: a[0] < a[1],
                        "OpSelect": lambda: a[1] if a[0] else a[2], "OpIAdd": lambda: (a[0] + a[1]) & 0xFFFFFFFF,
                        "OpShiftRightLogical": lambda: a[0] >> a[1] if a[1] < 32 else 0}[op]()
    return env


@pytest.mark.parametrize("array", [False, True])
def test_resinfo_words_follow_rdna2(array):
    """SHADER-388 (tools/image_query_probe.py on gfx1036): a 2D depth word is 0; a 2D array's layer count is
    minified from BASE_LEVEL; past the last level width, height and depth are 0 and the level count stays; a level
    with bit 31 set is past the last level too (the address VGPR is unsigned)."""
    cases = {  # (T# full size w, h, layers, BASE_LEVEL, LAST_LEVEL, level) -> words, as measured
        (16, 4, 1, 0, 4, 0): [16, 4, 0, 5], (16, 4, 1, 0, 4, 3): [2, 1, 0, 5], (16, 4, 1, 0, 4, 5): [0, 0, 0, 5],
        (16, 4, 1, 1, 4, 0): [8, 2, 0, 4], (16, 4, 1, 0, 4, 0x80000000): [0, 0, 0, 5],
    } if not array else {
        (256, 256, 6, 0, 8, 1): [128, 128, 3, 9], (256, 256, 7, 0, 8, 2): [64, 64, 1, 9],
        (256, 256, 100, 1, 8, 3): [16, 16, 6, 8], (256, 256, 100, 0, 8, 0): [256, 256, 100, 9],
        (16, 4, 6, 0, 2, 3): [0, 0, 0, 3], (16, 4, 6, 0, 2, 9): [0, 0, 0, 3],
    }
    for (w, h, layers, base, last, level), want in cases.items():
        stub = _ResinfoStub()
        ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", array, "%word3")
        view = (max(1, w >> base), max(1, h >> base), layers, last - base + 1)
        env = _eval_resinfo(stub.lines, view, level, base << 12 | last << 16 | (10 if array else 9) << 28)
        assert [env[i] & 0xFFFFFFFF for i in ids] == want, (w, h, layers, base, last, level)


def test_image_sample_d16_avoids_packhalf2x16(make_shader):
    """d16 image_sample packs f32->f16 via integer ops (no PackHalf2x16) and pass spirv-val."""
    sample_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        "image_sample v[4:5], v[0:1], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\nexp mrt0 v4, v5, v4, v5 done vm\ns_endpgm\n"
    )
    sh1 = make_shader(sample_asm, name="ps_d16_sample_ci")
    asm1 = sh1.translate()
    sh1.validate(asm1)
    assert "PackHalf2x16" not in asm1
    assert "OpBitwiseOr %uint" in asm1 and "OpSelect %uint" in asm1


@pytest.mark.parametrize("suffix,last_addr", [("lz", 1), ("l", 2), ("l_o", 3)])
def test_gather_d16_avoids_packhalf2x16(make_shader, suffix, last_addr):
    gather_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        f"image_gather4_{suffix} v[4:5], v[0:{last_addr}], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\nexp mrt0 v4, v5, v4, v5 done vm\ns_endpgm\n"
    )
    sh2 = make_shader(gather_asm, name=f"ps_d16_gather_{suffix}_ci")
    asm2 = sh2.translate()
    sh2.validate(asm2)
    assert "PackHalf2x16" not in asm2
    assert "OpBitwiseOr %uint" in asm2 and "OpSelect %uint" in asm2
    for ty in ("%v4float", "%v4uint_st", "%v4int_st"):
        assert asm2.count(f"OpImageFetch {ty} ") == (0 if suffix == "lz" else 4)


def _load_mini_interp():
    import sys
    semantic_dir = os.path.join(ROOT, "tests", "semantic")
    if semantic_dir not in sys.path:
        sys.path.insert(0, semantic_dir)
    import spirv_mini_interp
    return spirv_mini_interp


def test_fma_mix_decodes_fimm_as_binary16_and_preserves_other_half(make_shader):
    """v_fma_mix{lo_f16,hi_f16,_f32} with op_sel_hi=1 decode fimm literals (e.g. 1.5, -2.0) as
    f16 instead of f32 bit patterns, preserve the unwritten dst half, and emit no PackHalf2x16."""
    si = _load_mini_interp()
    ps = (
        "; fma_mix_fimm\n"
        "v_mov_b32 v1, 0x40000000\n"  # f32 2.0
        "v_mov_b32 v2, 0x3e800000\n"  # f32 0.25
        "v_mov_b32 v3, 0xbeef0000\n"  # sentinel high half
        # 1.5 (f16) * 2.0 (f32) + 0.25 (f32) = 3.25 -> f16 0x4280; high half 0xbeef preserved
        "v_fma_mixlo_f16 v3, 1.5, v1, v2 op_sel:[0,0,0] op_sel_hi:[1,0,0]\n"
        "v_mov_b32 v4, 0x0000cafe\n"  # sentinel low half
        "v_mov_b32 v5, 0x3f800000\n"  # f32 1.0
        # (-2.0 (f16)) * 1.5 (f16) + 1.0 (f32) = -2.0 -> f16 0xc000; low half 0xcafe preserved
        "v_fma_mixhi_f16 v4, -2.0, 1.5, v5 op_sel:[0,0,0] op_sel_hi:[1,1,0]\n"
        # (-2.0 (f16)) * 1.5 (f16) + 0.25 (f32) = -2.75 -> f32 0xc0300000
        "v_fma_mix_f32 v6, -2.0, 1.5, v2 op_sel:[0,0,0] op_sel_hi:[1,1,0]\n"
        "exp mrt0 v3, v4, v6, v3 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name="ps_fma_mix_fimm")
    spv = shader.translate()
    shader.validate(spv)

    out = si.run_fragment(spv)["%out_mrt0"]
    assert out[0] == 0xBEEF4280
    assert out[1] == 0xC000CAFE
    assert out[2] == 0xC0300000
    assert "PackHalf2x16" not in spv


@pytest.mark.parametrize("op,sentinel,shift,f32_bits,expected_half", [
    # RTE tie-to-even: 1.0 + 2^-11 -> 0x3c00 (even), 1.0 + 3*2^-11 -> 0x3c02 (even)
    ("v_fma_mixlo_f16", 0xBEEF0000, 0, 0x3F801000, 0x3C00),
    ("v_fma_mixlo_f16", 0xBEEF0000, 0, 0x3F803000, 0x3C02),
    ("v_fma_mixhi_f16", 0x0000CAFE, 16, 0x3F801000, 0x3C00),
    ("v_fma_mixhi_f16", 0x0000CAFE, 16, 0x3F803000, 0x3C02),
    # Subnormals & subnormal ties: 2^-15 -> 0x0200, 2^-24 -> 0x0001, 2^-25 -> 0x0000, 3*2^-25 -> 0x0002
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x38000000, 0x0200),
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x33800000, 0x0001),
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x33000000, 0x0000),
    ("v_fma_mixhi_f16", 0x00005678, 16, 0x33C00000, 0x0002),
    # Overflow to +-Inf, -0.0, quiet NaN
    ("v_fma_mixlo_f16", 0xAAAA0000, 0, 0x477FF000, 0x7C00),
    ("v_fma_mixhi_f16", 0x0000BBBB, 16, 0xC77FF000, 0xFC00),
    ("v_fma_mixlo_f16", 0xCCCC0000, 0, 0x80000000, 0x8000),
    ("v_fma_mixhi_f16", 0x0000DDDD, 16, 0x7FC00000, 0x7E00),
])
def test_fma_mix_f16_rte_edge_cases_and_half_preservation(
    op, sentinel, shift, f32_bits, expected_half, make_shader
):
    """v_fma_mixlo_f16 / v_fma_mixhi_f16 round f32 results with exact IEEE-754 binary16 RTE
    (ties-to-even, subnormals, overflow, -0.0, quiet NaN) while preserving the other dst half."""
    si = _load_mini_interp()
    # Fma(1.0, x, -0.0) yields x (preserving -0.0 when x == -0.0)
    ps = (
        "; fma_mix_f16_rte\n"
        "v_mov_b32 v1, 0x3f800000\n"
        f"v_mov_b32 v2, {f32_bits:#010x}\n"
        "v_mov_b32 v3, 0x80000000\n"
        f"v_mov_b32 v4, {sentinel:#010x}\n"
        f"{op} v4, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[0,0,0]\n"
        "exp mrt0 v4, v4, v4, v4 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name="ps_fma_mix_f16_rte")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    out = si.run_fragment(spv)["%out_mrt0"]
    expected_word = sentinel | (expected_half << shift)
    assert out[0] == expected_word


@pytest.mark.parametrize("suffix,fmt_code,expected_word", [
    ("xy", 29, 0x3C013C00),  # format 29 = (4, "f16", (16, 16)): lo=0x3c00, hi=0x3c01 (toward zero, SHADER-358)
    ("x", 13, 0x00003C00),   # format 13 = (2, "f16", (16,)): lo=0x3c00
])
def test_encode_format_f16_uses_f16_bits_rte(suffix, fmt_code, expected_word, make_shader):
    """buffer_store_format_{x,xy} with f16 V# formats packs without PackHalf2x16, validates with spirv-val, and
    stores the f16 bits RDNA2 does: rounded toward zero, as measured in SHADER-358 (was tie-to-even before it)."""
    si = _load_mini_interp()
    base = 0x1000
    vdata = "v2" if suffix == "x" else "v[2:3]"
    sel = 0x22C if suffix == "xy" else 0x204  # XY01 for 2-comp, X001 for 1-comp (RDNA2 store COMP_SWAP)
    ps = (
        "; buffer_store_format_f16_rte\n"
        f"s_mov_b32 s0, {base:#x}\n"
        "s_mov_b32 s1, 0\n"
        "s_mov_b32 s2, 0x100\n"
        f"s_mov_b32 s3, {fmt_code << 12 | 0x30000000 | sel:#x}\n"
        "v_mov_b32 v1, 0\n"
        "v_mov_b32 v2, 0x3f801000\n"  # 1.0 + 2^-11 -> 0x3c00 (tie to even)
        "v_mov_b32 v3, 0x3f803000\n"  # 1.0 + 3*2^-11 -> 0x3c01 (toward zero)
        f"buffer_store_format_{suffix} {vdata}, v1, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v2, v3, v2, v3 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name=f"ps_encode_format_f16_{suffix}")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    mem = {base: 0}
    si.run_fragment(spv, memory=mem)
    assert mem[base] == expected_word


@pytest.mark.parametrize("fmt, expected", [(56, 0x80000000), (57, 0x40810000)])
@pytest.mark.parametrize("packed", [False, True])
def test_normalized_buffer_store_nan(fmt, expected, packed, make_shader):
    # RGBA = positive signaling NaN, negative quiet NaN, -infinity, +0.5.
    # gfx1036 stored zero for both NaNs, clamped infinity and rounded 0.5 to even.
    if packed:
        inputs = "v_mov_b32 v4, 0xfe017c01\nv_mov_b32 v5, 0x3800fc00\n"
        store = "buffer_store_format_d16_xyzw v[4:5]"
    else:
        inputs = ("v_mov_b32 v4, 0x7f800001\nv_mov_b32 v5, 0xffc01000\n"
                  "v_mov_b32 v6, 0xff800000\nv_mov_b32 v7, 0.5\n")
        store = "buffer_store_format_xyzw v[4:7]"
    src = ("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 4\n"
           f"s_mov_b32 s3, {0x31000fac | fmt << 12:#x}\n" + inputs +
           f"{store}, off, s[0:3], 0\ns_waitcnt vmcnt(0)\ns_endpgm\n")
    shader = make_shader(src, name="ps_store_nan")
    spv = shader.translate()
    shader.validate(spv)
    memory = {0x1000: 0xDEADBEEF}
    _load_mini_interp().run_fragment(spv, memory=memory)
    assert memory[0x1000] == expected


@pytest.mark.parametrize("fmt", [22, 20])  # 32_FLOAT, 32_UINT
def test_unnormalized_buffer_store_keeps_nan_bits(fmt, make_shader):
    src = ("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 4\n"
           f"s_mov_b32 s3, {0x31000fac | fmt << 12:#x}\n"
           "v_mov_b32 v4, 0xffc01000\nbuffer_store_format_x v4, off, s[0:3], 0\ns_endpgm\n")
    shader = make_shader(src, name="ps_raw_store_nan")
    memory = {0x1000: 0}
    _load_mini_interp().run_fragment(shader.translate(), memory=memory)
    assert memory[0x1000] == 0xFFC01000


# GFX10 32-bit integer formats by component count: UINT code, SINT code (gfx10_formats.json).
INT32_FORMATS = [(1, 20, 21), (2, 62, 63), (3, 72, 73), (4, 75, 76)]


@pytest.mark.parametrize("n, uint_fmt, sint_fmt", INT32_FORMATS)
def test_int32_buffer_formats_load_and_store_raw_bits(n, uint_fmt, sint_fmt, make_shader):
    """SHADER-257: the 32-bit SINT typed-buffer formats (and 32_32_32_UINT) load and store each dword unchanged, as
    their UINT twins do. Negative values and NaN-like patterns must survive both ways; a load's missing channels
    repeat the format's own, as for every other format (SHADER-86)."""
    si = _load_mini_interp()
    words = [0xFFFFFFFF, 0x80000000, 0x7FC01000, 0x00000005][:n]
    for fmt in (uint_fmt, sint_fmt):
        desc = (f"s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, {4 * n}\n"
                f"s_mov_b32 s3, {0x31000fac | fmt << 12:#x}\n")
        load = desc + ("buffer_load_format_xyzw v[4:7], off, s[0:3], 0\ns_waitcnt vmcnt(0)\n"
                       "exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
        shader = make_shader(load, name=f"ps_int32_load_{fmt}")
        spv = shader.translate()
        shader.validate(spv)
        memory = {0x1000 + 4 * i: w for i, w in enumerate(words)}
        out = si.run_fragment(spv, memory=memory)["%out_mrt0"]
        assert list(out) == [words[c % n] for c in range(4)], fmt

        store_sel = 0x22C if n == 2 else 0xFAC  # XY01 for 2-comp RG (XYZW selects (x, w) on RDNA2 MUBUF stores)
        store_desc = (f"s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, {4 * n}\n"
                      f"s_mov_b32 s3, {0x31000000 | store_sel | fmt << 12:#x}\n")
        setup = "".join(f"v_mov_b32 v{4 + i}, {w:#x}\n" for i, w in enumerate(words))
        regs = "v4" if n == 1 else f"v[4:{3 + n}]"
        store = store_desc + setup + f"buffer_store_format_{'xyzw'[:n]} {regs}, off, s[0:3], 0\ns_endpgm\n"
        shader = make_shader(store, name=f"ps_int32_store_{fmt}")
        spv = shader.translate()
        shader.validate(spv)
        memory = {0x1000 + 4 * i: 0xDEADBEEF for i in range(5)}
        si.run_fragment(spv, memory=memory)
        assert [memory[0x1000 + 4 * i] for i in range(5)] == words + [0xDEADBEEF] * (5 - n), fmt


@pytest.mark.parametrize("fmt, shifts, channels", [(56, (0, 8, 16, 24), (64, 128, 191, 255)),
                                                (50, (0, 10, 20, 30), (256, 512, 767, 3))])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("components", [1, 4])
@pytest.mark.parametrize("bgra", [False, True])
def test_buffer_store_format_bgra(fmt, shifts, channels, packed, components, bgra, make_shader):
    # gfx1036 writes the red source into the third field of the BGRA view, even for store_format_x.
    values = (0x38003400, 0x3c003a00) if packed else (0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000)
    order = (2, 1, 0, 3) if bgra else (0, 1, 2, 3)
    selectors = sum((4 + channel) << (3 * i) for i, channel in enumerate(order))
    inputs = "".join(f"v_mov_b32 v{4 + i}, {value:#x}\n" for i, value in enumerate(values))
    regs = "v4" if components == 1 else ("v[4:5]" if packed else "v[4:7]")
    op = "buffer_store_format_" + ("d16_" if packed else "") + "xyzw"[:components]
    src = ("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 4\n"
           f"s_mov_b32 s3, {0x31000000 | fmt << 12 | selectors:#x}\n" + inputs +
           f"{op} {regs}, off, s[0:3], 0\ns_endpgm\n")
    shader = make_shader(src, name="ps_store_bgra")
    spv = shader.translate()
    shader.validate(spv)
    memory = {0x1000: 0xdeadbeef}
    _load_mini_interp().run_fragment(spv, memory=memory)
    expected = sum(channels[c] << shift for c, shift in zip(order, shifts) if c < components)
    assert memory[0x1000] == expected


@pytest.mark.parametrize("fmt, packed, inputs, expected", [
    (60, True, (0x01000101, 0x00fe00ff), (0xfeffffff,)),
    (27, False, (65536, 65537), (0xffffffff,)),
    (69, False, (65534, 65535, 65536, 0xffffffff), (0xfffffffe, 0xffffffff)),
    (75, False, (65534, 65535, 65536, 0xffffffff), (65534, 65535, 65536, 0xffffffff)),
])
def test_uint_buffer_store_saturation(fmt, packed, inputs, expected, make_shader):
    # Native gfx1036 clips each channel to the destination's width. The 32-bit control keeps every bit.
    setup = "".join(f"v_mov_b32 v{4 + i}, {v:#x}\n" for i, v in enumerate(inputs))
    selectors = 0x22c if fmt == 27 else 0xfac  # XY01 for RG16, XYZW for RGBA
    components = 4 if packed else len(inputs)
    op = "buffer_store_format_" + ("d16_" if packed else "") + "xyzw"[:components]
    src = ("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 16\n"
           f"s_mov_b32 s3, {0x31000000 | fmt << 12 | selectors:#x}\n" + setup +
           f"{op} v[4:{3 + len(inputs)}], off, s[0:3], 0\ns_endpgm\n")
    shader = make_shader(src, name="ps_store_uint")
    spv = shader.translate()
    shader.validate(spv)
    memory = {0x1000 + 4 * i: 0xdeadbeef for i in range(4)}
    _load_mini_interp().run_fragment(spv, memory=memory)
    assert tuple(memory[0x1000 + 4 * i] for i in range(4)) == expected + (0xdeadbeef,) * (4 - len(expected))


def test_remaining_d16_paths_avoid_packhalf2x16(make_shader):
    """buffer_format d16 and image_sample_c_lz d16 avoid PackHalf2x16 and validate with spirv-val."""
    tbuf_asm = (
        "s_mov_b32 s0, 0x1000\n"
        "s_mov_b32 s1, 0\n"
        "s_mov_b32 s2, 0x100\n"
        f"s_mov_b32 s3, {77 << 12 | 0x30000000:#x}\n"  # format 77 = f32x4
        "v_mov_b32 v0, 0\n"
        "buffer_load_format_d16_xy v2, v0, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v2, v2, v2, v2 done vm\n"
        "s_endpgm\n"
    )
    sh1 = make_shader(tbuf_asm, name="ps_tbuffer_d16_nopack")
    spv1 = sh1.translate()
    sh1.validate(spv1)
    assert "PackHalf2x16" not in spv1

    cmplz_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        "image_sample_c_lz v4, v[0:2], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v4, v4, v4, v4 done vm\n"
        "s_endpgm\n"
    )
    sh2 = make_shader(cmplz_asm, name="ps_sample_c_lz_d16_nopack")
    spv2 = sh2.translate()
    sh2.validate(spv2)
    assert "PackHalf2x16" not in spv2


WAVE64_SCMP_MASK_CASES = """\
; mattr: +wavefrontsize64,-wavefrontsize32
v_cmp_lt_u32_e64 vcc, v0, 17
s_and_b64 s[6:7], vcc, exec
s_cmp_eq_u32 vcc_lo, vcc_hi
s_cmp_lg_u32 exec_lo, exec_hi
s_cmp_eq_u32 s6, s7
s_cmp_eq_u32 vcc_lo, 0x80000003
s_cmp_lt_u32 exec_hi, s4
s_cmp_gt_i32 s6, 0
s_cmp_eq_u64 vcc, s[4:5]
s_cmp_lg_u64 exec, 0x12345678
s_endpgm
"""


@pytest.mark.parametrize("emulate", [False, True], ids=["wave64", "wave64-emulate"])
def test_wave64_scmp_mask_halves_and_scalars_validate(emulate, make_shader, monkeypatch):
    """Opposite-half s_cmp_*_u32, ordering compares on mask halves, and mask-vs-scalar/imm u32/u64
    compares translate and pass spirv-val in both native wave64 and --wave64-emulate modes."""
    shader = make_shader(WAVE64_SCMP_MASK_CASES, name="cs_w64_scmp_masks")
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", emulate)
    spv = shader.translate()
    shader.validate(spv)
    if emulate:
        assert "OpGroupNonUniform" not in spv
    else:
        assert "OpGroupNonUniformBitwiseOr" in spv
