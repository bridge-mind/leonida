"""16-bit high-half reads must not follow ``OpShiftRightLogical %uint u 16`` with a 16-bit extract (#1506).

AMD's Windows LLPC compiler (observed on RDNA2 drivers 22.40.84.06 in #1506 and 24.10.24) drops the ``>> 16`` when
either ``OpBitwiseAnd %uint (u >> 16) 0xFFFF`` or ``OpBitFieldSExtract %int (u >> 16) 0 16`` follows two partial-half
writes to ``u`` and feeds ``OpIAdd``, ``OpIMul``, or ``OpShiftLeftLogical``. On a 32-bit value ``OpShiftRightLogical``
already clears bits 16..31 and ``OpShiftRightArithmetic`` already sign-extends bits 16..31, so ``Gen.u16_half`` and
``Gen.i16_half`` emit those single shifts for high halves and narrow in place only for low halves.
"""
import os
import re
import subprocess
import sys

import pytest

import spirv_interp as si
from conftest import HAVE_SPIRV
from listing import translate_listing

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic"))
import spirv_mini_interp as mini  # noqa: E402


def _shr16_then_extract16(spvasm):
    """Return any ``(%shr, %use)`` where ``%shr = OpShiftRightLogical %uint ... %c10`` feeds either
    ``OpBitwiseAnd %uint %shr %cffff`` or ``OpBitFieldSExtract %int (OpBitcast %int %shr) %c0 %c10``."""
    shr_ids = set(re.findall(r"(%\w+)\s*=\s*OpShiftRightLogical %uint %\w+ %c10\b", spvasm))
    # Map OpBitcast %int %shr -> %shr so signed extracts of a shifted value are caught too.
    cast_to_shr = {dst: src for dst, src in re.findall(r"(%\w+)\s*=\s*OpBitcast %int (%\w+)\b", spvasm)
                   if src in shr_ids}
    bad = [(m.group(2), m.group(1))
           for m in re.finditer(r"(%\w+)\s*=\s*OpBitwiseAnd %uint (%\w+) %cffff\b", spvasm)
           if m.group(2) in shr_ids]
    bad += [(cast_to_shr[m.group(2)], m.group(1))
            for m in re.finditer(r"(%\w+)\s*=\s*OpBitFieldSExtract %int (%\w+) %c0 %c10\b", spvasm)
            if m.group(2) in cast_to_shr]
    return bad


PROLOGUE = (
    "v_mov_b32 v1, 0x41b1af48\n"
    "v_mov_b32 v2, 0x9e00ccd9\n"
    "v_mov_b32 v4, 0x40004200\n"
    "v_mov_b32 v3, 0xaaaa5555\n"
    "s_mov_b32 s4, 0x41b1138b\n"
)

U16_HIGH_HALF_OPS = [
    ("v_lshlrev_b16", "v_lshlrev_b16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_add_u16", "v_add_u16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_mul_lo_u16", "v_mul_lo_u16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_mad_u16", "v_mad_u16 v3, v1, v2, v4 op_sel:[1,1,1,0]"),
    ("v_min3_u16", "v_min3_u16 v3, v1, v2, v4 op_sel:[1,1,1,0]"),
    ("v_cvt_f16_u16", "v_cvt_f16_u16_e64 v3, v1 op_sel:[1,0]"),
    ("v_pack_b32_f16", "v_pack_b32_f16 v3, v1, v2 op_sel:[1,1]"),
    ("v_frexp_mant_f16", "v_frexp_mant_f16_e64 v3, v1 op_sel:[1,0]"),
    ("v_ldexp_f16", "v_ldexp_f16 v3, v1, v2 op_sel:[1,0,0]"),
    ("s_bitreplicate_b64_b32", "s_bitreplicate_b64_b32 s[6:7], s4"),
]

I16_HIGH_HALF_OPS = [
    ("v_ashrrev_i16", "v_ashrrev_i16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_add_i16", "v_add_i16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_sub_i16", "v_sub_i16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_min_i16", "v_min_i16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_max_i16", "v_max_i16_e64 v3, v1, v2 op_sel:[1,1,0]"),
    ("v_mad_i16", "v_mad_i16 v3, v1, v2, v4 op_sel:[1,1,1,0]"),
    ("v_min3_i16", "v_min3_i16 v3, v1, v2, v4 op_sel:[1,1,1,0]"),
    ("v_cvt_f16_i16", "v_cvt_f16_i16_e64 v3, v2 op_sel:[1,0]"),
    ("v_ldexp_f16_exp", "v_ldexp_f16 v3, v1, v2 op_sel:[0,1,0]"),
]


def _validate_spv(tmp_path, spvasm):
    if not HAVE_SPIRV:
        return
    asm_path = tmp_path / "p.spvasm"
    spv_path = tmp_path / "p.spv"
    asm_path.write_text(spvasm, encoding="utf-8", newline="\n")
    for cmd in (["spirv-as", "--target-env", "vulkan1.2", str(asm_path), "-o", str(spv_path)],
                ["spirv-val", "--target-env", "vulkan1.2", str(spv_path)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("name,asm", U16_HIGH_HALF_OPS, ids=[o[0] for o in U16_HIGH_HALF_OPS])
def test_u16_no_redundant_extract_after_shr16(name, asm, tmp_path):
    spvasm = translate_listing(tmp_path, PROLOGUE + asm)
    assert re.search(r"OpShiftRightLogical %uint %\w+ %c10\b", spvasm), asm
    assert _shr16_then_extract16(spvasm) == [], f"{name} narrowed OpShiftRightLogical >> 16 to 16 bits"


@pytest.mark.parametrize("name,asm", I16_HIGH_HALF_OPS, ids=[o[0] for o in I16_HIGH_HALF_OPS])
def test_i16_uses_ashr16_not_shr16_sextract(name, asm, tmp_path):
    spvasm = translate_listing(tmp_path, PROLOGUE + asm)
    assert re.search(r"OpShiftRightArithmetic %int %\w+ %c10\b", spvasm), asm
    assert _shr16_then_extract16(spvasm) == [], f"{name} narrowed OpShiftRightLogical >> 16 via SExtract"
    _validate_spv(tmp_path, spvasm)


def test_d16_buffer_store_no_redundant_extract_after_shr16(tmp_path):
    asm = (
        "v_mov_b32 v1, 0x9c004000\n"
        "v_mov_b32 v2, 0\n"
        "buffer_store_format_d16_xy v1, v2, s[0:3], 0 offen\n"
    )
    spvasm = translate_listing(tmp_path, asm)
    assert re.search(r"OpShiftRightLogical %uint %\w+ %c10\b", spvasm)
    assert re.search(r"OpShiftRightArithmetic %int %\w+ %c10\b", spvasm)
    assert _shr16_then_extract16(spvasm) == []
    _validate_spv(tmp_path, spvasm)


def test_issue_1506_sequence_computes_exact_shift_and_validates(tmp_path):
    """Exact minimized instruction sequence from #1506 (valu_fuzz --family f16 --show 2:163:64)."""
    asm = "\n".join([
        "v_mov_b32 v24, 0x41b1af48",
        "v_mov_b32 v26, 0x3e00ccd9",
        "v_mov_b32 v31, 0x40004200",
        "v_mov_b32 v34, 0xc09946b2",
        "v_mov_b32 v7, 3",
        "v_or_b32 v20, 0x3c00, v7",
        "v_pk_max_f16 v30, v26, v24 op_sel:[0,0] op_sel_hi:[1,1]",
        "v_min_i16_e64 v20, v24, v31 op_sel:[1,0,1]",
        "v_fma_f16_e64 v20, v34, v26, v26 op_sel:[0,0,1,0]",
        "v_lshlrev_b16_e64 v30, v30, v20 op_sel:[1,1,0]",
    ])
    spvasm = translate_listing(tmp_path, asm)
    assert _shr16_then_extract16(spvasm) == []
    state = si.run(spvasm)
    assert state.vgpr[20] == 0x41B1D803
    assert state.vgpr[30] == 0x41B18362
    _validate_spv(tmp_path, spvasm)


@pytest.mark.parametrize("asm,want", [
    ("v_add_u16_e64 v3, v1, v2 op_sel:[1,1,0]", 0xAAAA0000 | ((0x41B1 + 0x9E00) & 0xFFFF)),
    ("v_mul_lo_u16_e64 v3, v1, v2 op_sel:[1,1,0]", 0xAAAA0000 | ((0x41B1 * 0x9E00) & 0xFFFF)),
    ("v_mad_u16 v3, v1, v2, v4 op_sel:[1,1,1,0]", 0xAAAA0000 | ((0x41B1 * 0x9E00 + 0x4000) & 0xFFFF)),
    ("v_min3_u16 v3, v1, v2, v4 op_sel:[1,1,1,0]", 0xAAAA4000),
    ("v_pack_b32_f16 v3, v1, v2 op_sel:[1,1]", 0x9E0041B1),
    # Signed high-half operations with negative high half 0x9E00 (-25088):
    ("v_ashrrev_i16_e64 v3, v1, v2 op_sel:[1,1,0]", 0xAAAA0000 | (((-25088) >> (0x41B1 & 15)) & 0xFFFF)),
    ("v_add_i16_e64 v3, v1, v2 op_sel:[1,1,0] clamp", 0xAAAA0000 | ((0x41B1 - 25088) & 0xFFFF)),
    ("v_min_i16_e64 v3, v1, v2 op_sel:[1,1,0]", 0xAAAA9E00),
    ("v_max_i16_e64 v3, v1, v2 op_sel:[1,1,0]", 0xAAAA41B1),
    ("v_mad_i16 v3, v1, v2, v4 op_sel:[1,1,1,0] clamp", 0xAAAA8000),
    ("v_min3_i16 v3, v1, v2, v4 op_sel:[1,1,1,0]", 0xAAAA9E00),
])
def test_16bit_high_half_values(asm, want, tmp_path):
    spvasm = translate_listing(tmp_path, PROLOGUE + asm)
    assert mini.run_fragment(spvasm)["%out_mrt0"][0] == want


def test_s_bitreplicate_high_half_value(tmp_path):
    spvasm = translate_listing(tmp_path, PROLOGUE + "s_bitreplicate_b64_b32 s[6:7], s4")
    state = si.run(spvasm)

    def rep16(x):
        out = 0
        for b in range(16):
            if (x >> b) & 1:
                out |= 3 << (2 * b)
        return out

    assert state.sgpr[6] == rep16(0x138B)
    assert state.sgpr[7] == rep16(0x41B1)
