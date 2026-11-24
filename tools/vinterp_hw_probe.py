r"""SHADER-430: RDNA2 hardware probe for 32-bit and 16-bit v_interp reads on a packed parameter channel.

Measures on native RDNA2 hardware (tools/hw_oracle, wave32 and wave64) how the VINTRP unit executes 32-bit
(`v_interp_p1_f32`, `v_interp_p2_f32`, `v_interp_mov_f32`) and 16-bit (`v_interp_p1ll_f16`, `v_interp_p1lv_f16`,
`v_interp_p2_f16`, with and without `high`) reads against the same packed LDS parameter-cache channel, and compares the
non-tie cases against Leonida's live translator (`tools/rdna2spirv.py` -> `tests/semantic/spirv_mini_interp.py`).
Standard library only; no game files.

Parameter-cache layout in LDS (`M0` base, 48 bytes / 12 dwords per attribute `attrN`, component `c` in `0..3` for
`x..w`), confirmed on AMD Radeon 680M (`gfx1035`, RDNA2):
    P0   dword index = 12 * N + 2 * c       (byte offset 48 * N + 8 * c)
    P10  dword index = 12 * N + 2 * c + 1   (byte offset 48 * N + 8 * c + 4)
    P20  dword index = 12 * N + 8 + c       (byte offset 48 * N + 32 + 4 * c)

Questions answered by the probe (docs/design.md, "Pixel-shader interpolants" and "16-bit interpolation" rows):
  1. Mixed 32-bit and 16-bit reads of one packed channel (SHADER-430):
     A 32-bit `v_interp_mov_f32` (`p0`, `p10`, `p20`) or `v_interp_p1_f32` + `v_interp_p2_f32` read of a channel that
     `v_interp_p1ll_f16` / `v_interp_p2_f16` also reads has no hidden 16-bit unpacking in the VINTRP unit: `mov` returns
     the raw 32-bit parameter-cache word (`P0`, `P10`, `P20`) unchanged, and `p1_f32` / `p2_f32` evaluate
     `fma_f32(j, P20_f32, fma_f32(i, P10_f32, P0_f32))` on those raw 32-bit words reinterpreted as binary32, matching the
     translator's `Gen.attribute(..., half=None)` model.
  2. `v_interp_p1ll_f16` / `v_interp_p1lv_f16` intermediate and `v_interp_p2_f16` destination high half:
     `p1ll_f16` widens the selected f16 half (`[15:0]` or `[31:16]` under `high`) of `P0` and `P10` to binary32 and
     writes the fused binary32 `fma_f32(i, f32(P10.half), f32(P0.half))` (`p1lv_f16` reads `P0.half` from its `vsrc2`
     VGPR instead of LDS `P0`). `v_interp_p2_f16` writes its f16 result to `vdst[15:0]` and preserves `vdst[31:16]`.
  3. f16 denormal and NaN halves:
     Under the game's inferred MODE (`ieee=0`, `dx10_clamp=1`, `denorm32=0`, `denorm16_64=3`), f16 subnormal halves in
     both the low and high half of `P0`/`P10`/`P20` are preserved (`denorm16_64=0` flushes them to signed zero), and
     quiet or signalling f16 NaN halves propagate as a quiet f16 NaN with their payload bits preserved (`0x7c01` ->
     `0x7e01`, `0x7e05` -> `0x7e05`).
  4. `v_interp_p2_f16` rounding:
     Like `v_fma_mixlo_f16`, `v_interp_p2_f16` rounds the exact `partial_f32 + j * f32(P20.half)` once to binary16
     following `MODE.FP_ROUND.16_64` (nearest-even when `0`, toward-zero when `3`), whereas the translator's current
     two-step `OpFMul` + `OpFAdd` in f32 followed by `f16_res` rounds twice on near-ties.

Usage:
    python -I tools/vinterp_hw_probe.py                # translator + analytical reference check (CPU)
    python -I tools/vinterp_hw_probe.py --oracle hw    # + native AMD RDNA2 GPU (wave32 and wave64)
"""
import argparse
from fractions import Fraction
import gc
import os
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for p in (HERE, os.path.join(ROOT, "tests", "semantic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SENTINEL_HI = 0x5A5A0000
GAME_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round32": 0, "round16_64": 0}
RTZ16_MODE = dict(GAME_MODE, round16_64=3)
FLUSH16_MODE = dict(GAME_MODE, denorm16_64=0)


def lds_param_offset(attr, comp, vertex_select):
    """Byte offset in LDS (from M0) of VINTRP parameter slot `vertex_select` ('p0', 'p10', 'p20') for `attrN.comp`."""
    if not (0 <= comp < 4):
        raise ValueError(f"component {comp} out of range 0..3")
    base = 48 * attr
    if vertex_select == "p0":
        return base + 8 * comp
    if vertex_select == "p10":
        return base + 8 * comp + 4
    if vertex_select == "p20":
        return base + 32 + 4 * comp
    raise ValueError(f"unknown vertex_select {vertex_select!r}")


def pack_f16_pair(lo_bits, hi_bits):
    return (lo_bits & 0xFFFF) | ((hi_bits & 0xFFFF) << 16)


def f32_to_bits(val):
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def bits_to_f32(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def hbits_to_fraction(h):
    h &= 0xFFFF
    s = -1 if (h & 0x8000) else 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0x1F:
        raise ValueError(f"non-finite f16 0x{h:04x}")
    if e == 0:
        return Fraction(0) if m == 0 else s * Fraction(m, 1 << 24)
    return s * Fraction(0x400 | m, 1 << 10) * (Fraction(2) ** (e - 15))


def fbits_to_fraction(w):
    w &= 0xFFFFFFFF
    s = -1 if (w >> 31) else 1
    e = (w >> 23) & 0xFF
    m = w & 0x7FFFFF
    if e == 0xFF:
        raise ValueError(f"non-finite f32 0x{w:08x}")
    if e == 0:
        return Fraction(0)
    return s * Fraction(0x800000 | m, 1 << 23) * (Fraction(2) ** (e - 127))


def round_rne_fraction(x, bits, emin):
    if x == 0:
        return Fraction(0)
    a, e = abs(x), emin
    while a >= Fraction(2) ** (e + 1):
        e += 1
    q = Fraction(2) ** (e - bits + 1)
    n = a / q
    k = n.numerator // n.denominator
    rem = n - k
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (k & 1)):
        k += 1
    return (k * q) if x > 0 else -(k * q)


def fraction_to_f16_rne(x):
    r = round_rne_fraction(x, 11, -14)
    s = 0x8000 if x < 0 else 0
    a = abs(r)
    if a > 65504:
        return s | 0x7C00
    if a < Fraction(1, 1 << 14):
        return s | int(a * (1 << 24))
    e = 0
    while a >= Fraction(2) ** (e + 1):
        e += 1
    while a < Fraction(2) ** e:
        e -= 1
    return s | ((e + 15) << 10) | int((a / (Fraction(2) ** e) - 1) * 1024)


def fraction_to_f32_rne(x):
    if x == 0:
        return 0
    r = round_rne_fraction(x, 24, -126)
    s = 0x80000000 if x < 0 else 0
    a = abs(r)
    e = 0
    while a >= Fraction(2) ** (e + 1):
        e += 1
    while a < Fraction(2) ** e:
        e -= 1
    return s | ((e + 127) << 23) | int((a / (Fraction(2) ** e) - 1) * (1 << 23))


class ProbeCase:
    """One 4-dword hardware probe kernel (wave32 & wave64)."""

    __slots__ = ("name", "mode", "lds_writes", "body", "expected_hw", "translator_check")

    def __init__(self, name, mode, lds_writes, body, expected_hw, translator_check=None):
        self.name = name
        self.mode = dict(mode)
        self.lds_writes = tuple(lds_writes)
        self.body = tuple(body)
        self.expected_hw = tuple(expected_hw)
        self.translator_check = translator_check

    def kernel_lines(self):
        lines = ["s_mov_b32 m0, 0", "v_mov_b32 v2, 0"]
        for off, word in self.lds_writes:
            lines.append(f"v_mov_b32 v3, 0x{word & 0xFFFFFFFF:08x}")
            lines.append(f"ds_write_b32 v2, v3 offset:{off}")
        lines.append("s_waitcnt lgkmcnt(0)")
        lines.extend(self.body)
        lines.extend([
            "s_getreg_b32 s4, hwreg(HW_REG_HW_ID1)",
            "v_mov_b32 v12, s4",
            "v_mov_b32 v0, 0",
            "global_store_dwordx4 v0, v[8:11], s[2:3]",
            "global_store_dword v0, v12, s[2:3] offset:16",
            "s_nop 0",
            "s_endpgm",
        ])
        return lines


# Case 1: LDS layout probe — confirms P0, P10, P20 byte offsets for attr0.x..w and attr1.x.
CASE_LDS_LAYOUT = ProbeCase(
    "lds_param_layout",
    GAME_MODE,
    [(4 * idx, 0x3F800000 + idx) for idx in range(24)],
    [
        "v_interp_mov_f32 v8, p0, attr0.x",     # idx 0  (off 0)
        "v_interp_mov_f32 v9, p10, attr0.x",    # idx 1  (off 4)
        "v_interp_mov_f32 v10, p20, attr0.x",   # idx 8  (off 32)
        "v_interp_mov_f32 v11, p20, attr1.x",   # idx 20 (off 80 = 48 + 32)
    ],
    (0x3F800000, 0x3F800001, 0x3F800008, 0x3F800014),
)

# Case 2 (core SHADER-430): Mixed 16-bit and 32-bit reads on the same packed attr0.x channel.
# Per-vertex packed words:
#   P0  = lo 1.5 (0x3e00) | hi -2.0 (0xc000) -> 0xc0003e00 (as f32: -2.0037841796875)
#   P1  = lo 2.5 (0x4100) | hi  8.0 (0x4800) -> P10 in LDS:
#         For f16 halves: P10_lo = 1.0 (0x3c00), P10_hi = 10.0 (0x4900) -> 0x49003c00 (as f32: 525248.0)
#   P2  = lo 3.5 (0x4300) | hi 16.0 (0x4c00) -> P20 in LDS:
#         For f16 halves: P20_lo = 2.0 (0x4000), P20_hi = 18.0 (0x4c80) -> 0x4c804000 (as f32: 67239936.0)
# At i = 0.25 (0x3e800000), j = 0.50 (0x3f000000):
#   - f16 low:  1.5 + 0.25 * 1.0 + 0.50 * 2.0  = 2.75 (0x4180), high half preserved (0x5a5a) -> 0x5a5a4180
#   - f16 high: -2.0 + 0.25 * 10.0 + 0.50 * 18.0 = 9.5 (0x48c0), high half preserved (0x5a5a) -> 0x5a5a48c0
#   - mov_f32 p0: raw 32-bit word P0 = 0xc0003e00
#   - p1_f32 + p2_f32: f32(-2.0037841796875 + 0.25 * 525248.0 + 0.50 * 67239936.0) = 33751276.0 (0x4c00c03c)
CASE_MIXED_F16_F32 = ProbeCase(
    "mixed_f16_and_f32_same_channel",
    GAME_MODE,
    [
        (lds_param_offset(0, 0, "p0"), 0xC0003E00),
        (lds_param_offset(0, 0, "p10"), 0x49003C00),
        (lds_param_offset(0, 0, "p20"), 0x4C804000),
    ],
    [
        "v_mov_b32 v6, 0x3e800000",
        "v_mov_b32 v7, 0x3f000000",
        "v_mov_b32 v8, 0x5a5a0000",
        "v_interp_p1ll_f16 v12, v6, attr0.x",
        "v_interp_p2_f16 v8, v7, attr0.x, v12",
        "v_mov_b32 v9, 0x5a5a0000",
        "v_interp_p1ll_f16 v13, v6, attr0.x high",
        "v_interp_p2_f16 v9, v7, attr0.x, v13 high",
        "v_interp_mov_f32 v10, p0, attr0.x",
        "v_interp_p1_f32 v11, v6, attr0.x",
        "v_interp_p2_f32 v11, v7, attr0.x",
    ],
    (0x5A5A4180, 0x5A5A48C0, 0xC0003E00, 0x4C00C03C),
    translator_check="mixed",
)

# Case 3: v_interp_p1ll_f16 and v_interp_p1lv_f16 f32 intermediates + v_interp_p2_f16 with p1lv.
# P0 = lo 1.0 (0x3c00) | hi 2.0 (0x4000), P10 = lo 4.0 (0x4400) | hi 8.0 (0x4800), P20 = lo -1.0 (0xbc00) | hi -1.0 (0xbc00)
# i = 0.25, j = 0.50, vsrc2 = lo 2.0 (0x4000) | hi 3.0 (0x4200).
#   - v8:  p1ll_f16 low  = 1.0 + 0.25 * 4.0 = 2.0 (f32 0x40000000)
#   - v9:  p1ll_f16 high = 2.0 + 0.25 * 8.0 = 4.0 (f32 0x40800000)
#   - v10: p1lv_f16 low (vsrc2.lo=2.0)  -> 2.0 + 0.25*4.0 = 3.0 -> p2_f16 (+ 0.5*-1.0 = 2.5 = 0x4100), kept hi -> 0x5a5a4100
#   - v11: p1lv_f16 high (vsrc2.hi=3.0) -> 3.0 + 0.25*8.0 = 5.0 -> p2_f16 (+ 0.5*-1.0 = 4.5 = 0x4480), kept hi -> 0x5a5a4480
CASE_P1LL_P1LV = ProbeCase(
    "p1ll_and_p1lv_intermediates",
    GAME_MODE,
    [
        (lds_param_offset(0, 0, "p0"), 0x40003C00),
        (lds_param_offset(0, 0, "p10"), 0x48004400),
        (lds_param_offset(0, 0, "p20"), 0xBC00BC00),
    ],
    [
        "v_mov_b32 v6, 0x3e800000",
        "v_mov_b32 v7, 0x3f000000",
        "v_mov_b32 v14, 0x42004000",
        "v_interp_p1ll_f16 v8, v6, attr0.x",
        "v_interp_p1ll_f16 v9, v6, attr0.x high",
        "v_mov_b32 v10, 0x5a5a0000",
        "v_interp_p1lv_f16 v12, v6, attr0.x, v14",
        "v_interp_p2_f16 v10, v7, attr0.x, v12",
        "v_mov_b32 v11, 0x5a5a0000",
        "v_interp_p1lv_f16 v13, v6, attr0.x, v14 high",
        "v_interp_p2_f16 v11, v7, attr0.x, v13 high",
    ],
    (0x40000000, 0x40800000, 0x5A5A4100, 0x5A5A4480),
    translator_check="p1ll_p1lv",
)

# Case 4: f16 denormal and NaN halves under GAME_MODE (denorm16_64=3, ieee=0).
# attr0.x: P0 = lo +2^-24 (0x0001) | hi -2^-24 (0x8001), P10 = P20 = lo +2^-15 (0x0200) | hi -2^-15 (0x8200)
#   At i = j = 0.5: lo = +2^-24 + 2^-15 = 0x0201, hi = -2^-24 - 2^-15 = 0x8201.
# attr0.y: P0 = lo qNaN (0x7e05) | hi sNaN (0x7c01), P10 = 1.0, P20 = 2.0
#   lo = 0x7e05, hi = quieted 0x7e01.
CASE_DENORM_NAN_KEPT = ProbeCase(
    "f16_denorm_kept_and_nan_quieting",
    GAME_MODE,
    [
        (lds_param_offset(0, 0, "p0"), 0x80010001),
        (lds_param_offset(0, 0, "p10"), 0x82000200),
        (lds_param_offset(0, 0, "p20"), 0x82000200),
        (lds_param_offset(0, 1, "p0"), 0x7C017E05),
        (lds_param_offset(0, 1, "p10"), 0x3C003C00),
        (lds_param_offset(0, 1, "p20"), 0x40004000),
    ],
    [
        "v_mov_b32 v6, 0x3f000000",
        "v_mov_b32 v7, 0x3f000000",
        "v_mov_b32 v8, 0x5a5a0000",
        "v_interp_p1ll_f16 v12, v6, attr0.x",
        "v_interp_p2_f16 v8, v7, attr0.x, v12",
        "v_mov_b32 v9, 0x5a5a0000",
        "v_interp_p1ll_f16 v13, v6, attr0.x high",
        "v_interp_p2_f16 v9, v7, attr0.x, v13 high",
        "v_mov_b32 v10, 0x5a5a0000",
        "v_interp_p1ll_f16 v14, v6, attr0.y",
        "v_interp_p2_f16 v10, v7, attr0.y, v14",
        "v_mov_b32 v11, 0x5a5a0000",
        "v_interp_p1ll_f16 v15, v6, attr0.y high",
        "v_interp_p2_f16 v11, v7, attr0.y, v15 high",
    ],
    (0x5A5A0201, 0x5A5A8201, 0x5A5A7E05, 0x5A5A7E01),
)

# Case 5: f16 denormals flushed when denorm16_64=0.
CASE_DENORM_FLUSHED = ProbeCase(
    "f16_denorm_flushed_when_mode_0",
    FLUSH16_MODE,
    CASE_DENORM_NAN_KEPT.lds_writes,
    CASE_DENORM_NAN_KEPT.body,
    (0x5A5A0000, 0x5A5A8000, 0x5A5A7E05, 0x5A5A7E01),
)

# Case 6: Single-rounding ties in v_interp_p2_f16 (nearest-even mode):
#   v8 (tie A, low half):  1 + 2^-11 + (-2^-40)*(-1) = 1 + 2^-11 + 2^-40 -> single 0x3c01, double 0x3c00
#   v9 (tie B, high half): 1 + 3*2^-11 + (2^-30)*(-1) = 1 + 3*2^-11 - 2^-30 -> single 0x3c01, double 0x3c02
#   v10 (tie E, overflow): 65504 + 16 + (2^-20)*(-1) = 65520 - 2^-20 -> single 0x7bff, double 0x7c00
#   v11 (tie D, subnorm):  0 + (2^-11)*2^-14 + (2^-46)*2^-14 = 2^-25 + 2^-60 -> single 0x0001, double 0x0000
CASE_SINGLE_ROUND_TIES = ProbeCase(
    "p2_f16_single_rounding_ties_rne",
    GAME_MODE,
    [
        (lds_param_offset(0, 0, "p0"), 0x3C003C00),
        (lds_param_offset(0, 0, "p10"), 0x3C003C00),
        (lds_param_offset(0, 0, "p20"), 0xBC00BC00),
        (lds_param_offset(0, 1, "p0"), 0x7BFF7BFF),
        (lds_param_offset(0, 1, "p10"), 0x4C004C00),
        (lds_param_offset(0, 1, "p20"), 0xBC00BC00),
        (lds_param_offset(0, 2, "p0"), 0x00000000),
        (lds_param_offset(0, 2, "p10"), 0x04000400),
        (lds_param_offset(0, 2, "p20"), 0x04000400),
    ],
    [
        "v_mov_b32 v6, 0x3a000000",
        "v_mov_b32 v7, 0xab800000",
        "v_mov_b32 v8, 0x5a5a0000",
        "v_interp_p1ll_f16 v12, v6, attr0.x",
        "v_interp_p2_f16 v8, v7, attr0.x, v12",
        "v_mov_b32 v6, 0x3ac00000",
        "v_mov_b32 v7, 0x30800000",
        "v_mov_b32 v9, 0x5a5a0000",
        "v_interp_p1ll_f16 v13, v6, attr0.x high",
        "v_interp_p2_f16 v9, v7, attr0.x, v13 high",
        "v_mov_b32 v6, 0x3f800000",
        "v_mov_b32 v7, 0x35800000",
        "v_mov_b32 v10, 0x5a5a0000",
        "v_interp_p1ll_f16 v14, v6, attr0.y",
        "v_interp_p2_f16 v10, v7, attr0.y, v14",
        "v_mov_b32 v6, 0x3a000000",
        "v_mov_b32 v7, 0x28800000",
        "v_mov_b32 v11, 0x5a5a0000",
        "v_interp_p1ll_f16 v15, v6, attr0.z",
        "v_interp_p2_f16 v11, v7, attr0.z, v15",
    ],
    (0x5A5A3C01, 0x5A5A3C01, 0x5A5A7BFF, 0x5A5A0001),
)

# Case 7: v_interp_p2_f16 under round16_64 = 3 (round toward zero) + VOP3 clamp.
#   v8: 1.0 + (3*2^-12)*1.0 + 0 = 1.0 + 0.75 ulp -> RTZ gives 0x3c00 (RNE gives 0x3c01)
#   v9: 1.0 + 1.0*1.0 + (2^-30)*(-1.0) = 2.0 - 2^-30 -> RTZ single rounding gives 0x3fff (double f32->f16 gives 0x4000)
#   v10: 1.0 + 1.0*1.0 + 0 with clamp -> 1.0 (0x3c00)
#   v11: 1.0 + 0 + 2.0*(-1.0) = -1.0 with clamp -> +0.0 (0x0000)
CASE_RTZ_AND_CLAMP = ProbeCase(
    "p2_f16_rtz_and_clamp",
    RTZ16_MODE,
    [
        (lds_param_offset(0, 0, "p0"), 0x3C003C00),
        (lds_param_offset(0, 0, "p10"), 0x3C003C00),
        (lds_param_offset(0, 0, "p20"), 0xBC00BC00),
        (lds_param_offset(0, 1, "p0"), 0x3C003C00),
        (lds_param_offset(0, 1, "p10"), 0x3C003C00),
        (lds_param_offset(0, 1, "p20"), 0xBC00BC00),
    ],
    [
        "v_mov_b32 v6, 0x3a400000",
        "v_mov_b32 v7, 0",
        "v_mov_b32 v8, 0x5a5a0000",
        "v_interp_p1ll_f16 v12, v6, attr0.x",
        "v_interp_p2_f16 v8, v7, attr0.x, v12",
        "v_mov_b32 v6, 0x3f800000",
        "v_mov_b32 v7, 0x30800000",
        "v_mov_b32 v9, 0x5a5a0000",
        "v_interp_p1ll_f16 v13, v6, attr0.x high",
        "v_interp_p2_f16 v9, v7, attr0.x, v13 high",
        "v_mov_b32 v6, 0x3f800000",
        "v_mov_b32 v7, 0",
        "v_mov_b32 v10, 0x5a5a0000",
        "v_interp_p1ll_f16 v14, v6, attr0.y",
        "v_interp_p2_f16 v10, v7, attr0.y, v14 clamp",
        "v_mov_b32 v6, 0",
        "v_mov_b32 v7, 0x40000000",
        "v_mov_b32 v11, 0x5a5a0000",
        "v_interp_p1ll_f16 v15, v6, attr0.y high",
        "v_interp_p2_f16 v11, v7, attr0.y, v15 high clamp",
    ],
    (0x5A5A3C00, 0x5A5A3FFF, 0x5A5A3C00, 0x5A5A0000),
)

CASES = (
    CASE_LDS_LAYOUT,
    CASE_MIXED_F16_F32,
    CASE_P1LL_P1LV,
    CASE_DENORM_NAN_KEPT,
    CASE_DENORM_FLUSHED,
    CASE_SINGLE_ROUND_TIES,
    CASE_RTZ_AND_CLAMP,
)


def _translate_ps(name, lines):
    text = f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines)
    with tempfile.TemporaryDirectory(prefix="vinterp_hw_probe_") as work:
        os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
        src = os.path.join(work, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(work, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
            return spv


def run_translator_checks():
    """Verify the translator's PerVertexKHR mixed f16/f32 and p1ll/p1lv paths match the hardware words."""
    # 1. Mixed f16 low, f16 high, mov_f32 p0, and p1_f32 + p2_f32 on the same attr0.x channel.
    # For the f32 read, per-vertex words P0, P1, P2 whose differences P1 - P0 and P2 - P0 are 0x49003c00 and 0x4c804000:
    p0_u = 0xC0003E00
    p10_f = bits_to_f32(0x49003C00)
    p20_f = bits_to_f32(0x4C804000)
    p0_f = bits_to_f32(p0_u)
    p1_u = f32_to_bits(p0_f + p10_f)
    p2_u = f32_to_bits(p0_f + p20_f)
    # And for the f16 reads, per-vertex packed words with P0=(1.5, -2.0), P1=(2.5, 8.0), P2=(3.5, 16.0):
    v0_f16 = pack_f16_pair(0x3E00, 0xC000)
    v1_f16 = pack_f16_pair(0x4100, 0x4800)
    v2_f16 = pack_f16_pair(0x4300, 0x4C00)
    spv_mix = _translate_ps("ps_vinterp_mixed", [
        "v_interp_p1ll_f16 v6, v0, attr0.x",
        "v_interp_p2_f16 v2, v1, attr0.x, v6",
        "v_interp_p1ll_f16 v7, v0, attr0.x high",
        "v_interp_p2_f16 v3, v1, attr0.x, v7 high",
        "v_interp_mov_f32_e32 v4, p0, attr0.x",
        "v_interp_p1_f32_e32 v5, v0, attr0.y",
        "v_interp_p2_f32_e32 v5, v1, attr0.y",
        "exp mrt0 v2, v3, v4, v5 done vm",
        "s_endpgm",
    ])
    pv_attr0 = (
        (bits_to_f32(v0_f16), bits_to_f32(p0_u), 0.0, 0.0),
        (bits_to_f32(v1_f16), bits_to_f32(p1_u), 0.0, 0.0),
        (bits_to_f32(v2_f16), bits_to_f32(p2_u), 0.0, 0.0),
    )
    got_mix = tuple(spirv_mini_interp.run_fragment(
        spv_mix, inputs={"%bary": (0.25, 0.25, 0.5), "%in_attr0": pv_attr0}
    )["%out_mrt0"])

    # 2. p1ll and p1lv pairs on the same packed PerVertexKHR channel.
    # P0 = (1.0, 2.0), P1 = (5.0, 10.0) [so P10 = (4.0, 8.0)], P2 = (0.0, 1.0) [so P20 = (-1.0, -1.0)]
    pv_p1ll = (
        (bits_to_f32(pack_f16_pair(0x3C00, 0x4000)), 0.0, 0.0, 0.0),
        (bits_to_f32(pack_f16_pair(0x4500, 0x4900)), 0.0, 0.0, 0.0),
        (bits_to_f32(pack_f16_pair(0x0000, 0x3C00)), 0.0, 0.0, 0.0),
    )
    spv_p1 = _translate_ps("ps_vinterp_p1ll_p2", [
        "v_interp_p1ll_f16 v6, v0, attr0.x",
        "v_interp_p2_f16 v2, v1, attr0.x, v6",
        "v_interp_p1ll_f16 v7, v0, attr0.x high",
        "v_interp_p2_f16 v3, v1, attr0.x, v7 high",
        "exp mrt0 v2, v3, v6, v7 done vm",
        "s_endpgm",
    ])
    got_p1 = tuple(spirv_mini_interp.run_fragment(
        spv_p1, inputs={"%bary": (0.25, 0.25, 0.5), "%in_attr0": pv_p1ll}
    )["%out_mrt0"])
    return got_mix, got_p1


def run_hw(arch=None, waves=(32, 64)):
    """Run all probe cases on native AMD hardware.

    In a WGP (Workgroup Processor) on RDNA2, each pair of CUs shares 128 KiB of LDS (64 KiB at CU0 base 0 and 64 KiB
    at CU1 base 64 KiB). While ds_write_b32 adds the workgroup's assigned LDS_BASE, v_interp indexes LDS by the 16-bit
    M0[15:0] byte offset directly (always in the low 64 KiB, CU0). Allocating 64 KiB of LDS and dispatching a batch for
    each case ensures at least one kernel instance runs on CU0 (HW_ID1 bit 8 == 0, where LDS_BASE == 0).
    """
    from hw_oracle import asm, runner  # noqa: E402
    oracle = runner.open_oracle(arch=arch)
    try:
        device = oracle.device
        hw_arch = oracle.arch
        per_wave = {}
        for wave in waves:
            per_wave[wave] = {}
            for c in CASES:
                k = asm.Kernel(c.kernel_lines(), wave=wave, threads=1, words=5, raw=True, lds=65536, mode=c.mode)
                batch = oracle.run([k] * 4, words=5)
                picked = None
                for r in batch:
                    if not isinstance(r, list) or len(r) < 5:
                        continue
                    if (r[4] & 0x100) == 0:
                        picked = tuple(r[:4])
                        break
                if picked is None:
                    first = batch[0]
                    picked = tuple(first[:4]) if isinstance(first, list) else first
                per_wave[wave][c.name] = picked
        return device, hw_arch, per_wave
    finally:
        oracle.close()


def _fmt(words):
    if isinstance(words, (list, tuple)):
        return " ".join(f"0x{w:08x}" for w in words)
    return str(words)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("translator", "hw", "both"), default="translator")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    args = ap.parse_args(argv)

    got_mix, got_p1 = run_translator_checks()
    want_mix = (
        CASE_MIXED_F16_F32.expected_hw[0] & 0xFFFF,
        CASE_MIXED_F16_F32.expected_hw[1] & 0xFFFF,
        CASE_MIXED_F16_F32.expected_hw[2],
        CASE_MIXED_F16_F32.expected_hw[3],
    )
    if got_mix != want_mix:
        print(f"FAIL translator mixed f16/f32: got {_fmt(got_mix)}, expected {_fmt(want_mix)}")
        return 1
    # In got_p1, mrt0.xy = p2_f16 low/high results (1.5 = 0x3e00, 3.5 = 0x4300)
    if got_p1[:2] != (0x00003E00, 0x00004300):
        print(f"FAIL translator p1ll/p2 halves: got {_fmt(got_p1[:2])}")
        return 1
    print(f"translator checks: ok (mixed f16/f32 = {_fmt(got_mix)}, p2_f16 halves = {_fmt(got_p1[:2])})")

    if args.oracle == "translator":
        return 0

    from hw_oracle import runner  # noqa: E402
    try:
        device, hw_arch, per_wave = run_hw(arch=args.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    print(f"hardware oracle: {device} ({hw_arch})")
    failures = 0
    for c in CASES:
        w32 = per_wave[32][c.name]
        w64 = per_wave[64][c.name]
        ok = (w32 == c.expected_hw) and (w64 == c.expected_hw)
        status = "ok" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"  {c.name:<36} {status:<8} hw32=[{_fmt(w32)}] hw64=[{_fmt(w64)}] expected=[{_fmt(c.expected_hw)}]")
    return 1 if failures else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
