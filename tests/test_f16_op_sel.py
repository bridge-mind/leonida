"""op_sel on f16 VOP3 ops in tools/rdna2spirv.py.

These tests feed the translator hand-written ``.s`` listings, so they need neither llvm-mc nor spirv-tools and run
everywhere. They check the shape of the generated SPIR-V (which half is read, where the result goes), not computed
values; values are covered on a GPU by runtime/test/vk_harness.

Semantics (Mesa ACO ``can_use_opsel`` / AMD RDNA2 ISA, VOP3 ``op_sel``): bit i < nsrc selects the high half of source i,
bit nsrc selects the high half of the destination. The half of the destination that is not written is kept.
"""
import re

import pytest

import rdna2spirv
from listing import translate_listing

# Result register v3 is initialised first: an f16 write keeps the other half, so it reads the old value, and the
# translator rejects pixel shaders that read v0/v1 before writing them.
PROLOGUE = "v_mov_b32 v3, 0x12345678\nv_mov_b32 v1, 0x3c004000"


def translate(tmp_path, instruction):
    return translate_listing(tmp_path, PROLOGUE + "\n" + instruction)


def high_extracts(spv):
    """Number of UnpackHalf2x16 results read at component 1 (the high half)."""
    return len(re.findall(r"OpCompositeExtract %float %\w+ 1$", spv, re.M))


def low_extracts(spv):
    return len(re.findall(r"OpCompositeExtract %float %\w+ 0$", spv, re.M))


def left_shifts(spv):
    return spv.count("OpShiftLeftLogical %uint")


# (id, instruction without modifiers, number of sources)
OPS = [
    ("v_fma_f16", "v_fma_f16 v3, v1, v1, v1", 3),
    ("v_med3_f16", "v_med3_f16 v3, v1, v1, v1", 3),
    ("v_min3_f16", "v_min3_f16 v3, v1, v1, v1", 3),
    ("v_max3_f16", "v_max3_f16 v3, v1, v1, v1", 3),
    ("v_add_f16", "v_add_f16_e64 v3, v1, v1", 2),
    ("v_sub_f16", "v_sub_f16_e64 v3, v1, v1", 2),
    ("v_subrev_f16", "v_subrev_f16_e64 v3, v1, v1", 2),
    ("v_mul_f16", "v_mul_f16_e64 v3, v1, v1", 2),
    ("v_max_f16", "v_max_f16_e64 v3, v1, v1", 2),
    ("v_min_f16", "v_min_f16_e64 v3, v1, v1", 2),
    ("v_sqrt_f16", "v_sqrt_f16_e64 v3, v1", 1),
    ("v_rcp_f16", "v_rcp_f16_e64 v3, v1", 1),
]


@pytest.mark.parametrize("name,asm,nsrc", OPS, ids=[o[0] for o in OPS])
def test_default_reads_low_halves_and_writes_low_half(name, asm, nsrc, tmp_path):
    spv = translate(tmp_path, asm)
    assert high_extracts(spv) == 0
    assert low_extracts(spv) == nsrc


@pytest.mark.parametrize("name,asm,nsrc", OPS, ids=[o[0] for o in OPS])
def test_source_select_reads_high_half(name, asm, nsrc, tmp_path):
    base = translate(tmp_path, asm)
    for i in range(nsrc):
        bits = [1 if j == i else 0 for j in range(nsrc + 1)]
        spv = translate(tmp_path, f"{asm} op_sel:[{','.join(map(str, bits))}]")
        assert high_extracts(spv) == 1, f"source {i}"
        assert low_extracts(spv) == nsrc - 1, f"source {i}"
        assert left_shifts(spv) == left_shifts(base)       # destination is still the low half


@pytest.mark.parametrize("name,asm,nsrc", OPS, ids=[o[0] for o in OPS])
def test_destination_select_writes_high_half(name, asm, nsrc, tmp_path):
    base = translate(tmp_path, asm)
    bits = [0] * nsrc + [1]
    spv = translate(tmp_path, f"{asm} op_sel:[{','.join(map(str, bits))}]")
    assert high_extracts(spv) == 0
    assert left_shifts(spv) == left_shifts(base) + 1       # result << 16, low half of the old value kept


def test_all_bits_set(tmp_path):
    spv = translate(tmp_path, "v_fma_f16 v3, v1, v1, v1 op_sel:[1,1,1,1]")
    assert high_extracts(spv) == 3
    assert low_extracts(spv) == 0


def test_op_sel_with_neg_and_clamp_still_translates(tmp_path):
    spv = translate(tmp_path, "v_fma_f16 v3, -v1, |v1|, v1 op_sel:[1,0,1,1] clamp")
    assert high_extracts(spv) == 2


def test_translation_is_deterministic(tmp_path):
    asm = "v_med3_f16 v3, v1, v1, v1 op_sel:[1,0,1,1]"
    assert translate(tmp_path, asm) == translate(tmp_path, asm)


def test_other_unsupported_instructions_are_still_rejected(tmp_path):
    with pytest.raises(rdna2spirv.Unsupported):
        translate(tmp_path, "v_sat_pk_u8_i16 v3, v1")
