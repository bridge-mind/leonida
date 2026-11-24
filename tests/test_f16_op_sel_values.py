"""Computed values for op_sel on f16 VOP3 ops, checked with the SPIR-V interpreter in tests/spirv_interp.py.

The reference is plain Python: pick the selected half of each source, compute in double precision, round to f16
(round to nearest even), and merge into the destination keeping the half that is not written. Inputs are chosen so
every intermediate is exact, which makes the translator's f32 detour and the reference's single rounding agree.
"""
import itertools

import pytest

import spirv_interp as si
from listing import translate_listing

SRC = {1: 0x44003E00, 2: 0x40003800, 4: 0x3400BC00}     # v1: 1.5|4.0  v2: 0.5|2.0  v4: -1.0|0.25  (lo|hi)
OLD = 0xAAAA5555


def median(a, b, c):
    return sorted((a, b, c))[1]


# name -> (asm with {s} operand slots, number of sources, reference on floats)
OPS = {
    "v_add_f16": ("v_add_f16_e64 v3, {0}, {1}", 2, lambda a, b: a + b),
    "v_sub_f16": ("v_sub_f16_e64 v3, {0}, {1}", 2, lambda a, b: a - b),
    "v_subrev_f16": ("v_subrev_f16_e64 v3, {0}, {1}", 2, lambda a, b: b - a),
    "v_mul_f16": ("v_mul_f16_e64 v3, {0}, {1}", 2, lambda a, b: a * b),
    "v_max_f16": ("v_max_f16_e64 v3, {0}, {1}", 2, max),
    "v_min_f16": ("v_min_f16_e64 v3, {0}, {1}", 2, min),
    "v_fma_f16": ("v_fma_f16 v3, {0}, {1}, {2}", 3, lambda a, b, c: a * b + c),
    "v_med3_f16": ("v_med3_f16 v3, {0}, {1}, {2}", 3, median),
    "v_min3_f16": ("v_min3_f16 v3, {0}, {1}, {2}", 3, min),
    "v_max3_f16": ("v_max3_f16 v3, {0}, {1}, {2}", 3, max),
    "v_sqrt_f16": ("v_sqrt_f16_e64 v3, {0}", 1, lambda a: a ** 0.5),
    "v_rcp_f16": ("v_rcp_f16_e64 v3, {0}", 1, lambda a: 1 / a),
}
REGS = ["v1", "v2", "v4"]


def half(reg_value, hi):
    return si.half_to_float(reg_value >> 16 if hi else reg_value)


def expected(fn, nsrc, sel, dst_hi, regs=(1, 2, 4)):
    vals = [half(SRC[regs[i]], sel[i]) for i in range(nsrc)]
    h = si.float_to_half(fn(*vals))
    return (h << 16 | OLD & 0xFFFF) if dst_hi else (OLD & 0xFFFF0000 | h)


def execute(tmp_path, asm):
    prologue = "".join(f"v_mov_b32 v{r}, {v:#x}\n" for r, v in SRC.items()) + f"v_mov_b32 v3, {OLD:#x}\n"
    return si.run(translate_listing(tmp_path, prologue + asm)).vgpr[3]


CASES = [(name, bits) for name, (_, n, _) in OPS.items() for bits in itertools.product((0, 1), repeat=n + 1)]


@pytest.mark.parametrize("name,bits", CASES, ids=[f"{n}-{''.join(map(str, b))}" for n, b in CASES])
def test_op_sel_value(name, bits, tmp_path):
    template, nsrc, fn = OPS[name]
    asm = template.format(*REGS) + f" op_sel:[{','.join(map(str, bits))}]"
    assert execute(tmp_path, asm) == expected(fn, nsrc, bits[:nsrc], bits[nsrc]), asm


def test_neg_applies_to_the_selected_half(tmp_path):
    got = execute(tmp_path, "v_add_f16_e64 v3, -v1, v2 op_sel:[1,0,0]")
    assert got == OLD & 0xFFFF0000 | si.float_to_half(-4.0 + 0.5)


def test_abs_applies_to_the_selected_half(tmp_path):
    got = execute(tmp_path, "v_add_f16_e64 v3, |v4|, v2 op_sel:[0,1,0]")
    assert got == OLD & 0xFFFF0000 | si.float_to_half(1.0 + 2.0)


def test_clamp_with_destination_high(tmp_path):
    got = execute(tmp_path, "v_add_f16_e64 v3, v1, v2 op_sel:[0,0,1] clamp")      # 1.5 + 0.5 = 2.0 -> clamped to 1.0
    assert got == si.float_to_half(1.0) << 16 | OLD & 0xFFFF


def test_fmac_destination_is_the_accumulator(tmp_path):
    # v_fmac_f16 v3, v1, v2: v3.lo = v1.lo * v2.lo + v3.lo, high half kept
    prologue = f"v_mov_b32 v1, {SRC[1]:#x}\nv_mov_b32 v2, {SRC[2]:#x}\nv_mov_b32 v3, 0xaaaa3c00\n"     # accumulator 1.0
    got = si.run(translate_listing(tmp_path, prologue + "v_fmac_f16_e32 v3, v1, v2")).vgpr[3]
    assert got == 0xAAAA0000 | si.float_to_half(1.5 * 0.5 + 1.0)
