"""Semantic and translation tests for v_interp_p1lv_f16, f16 --bary-fallback, and stage/attr guards.

Covers the remaining 16-bit interpolation behaviour in tools/rdna2spirv.py not exercised by
test_interp_f16.py (clamp), test_interp_f16_high.py (p1ll low/high), or test_f16_interp_fallback.py:
- v_interp_p1lv_f16: PerVertexKHR low/high half decoding, standalone f32 intermediate, paired v_interp_p2_f16
  f16 narrowing and VOP3 clamp, nearest-even tie rounding, SV_Position ISG1 layout coexistence,
  --f16-interp-fallback (F16_HALVES) execution and location-clash rejection, and SPIR-V negative controls.
- --bary-fallback (BARY_FALLBACK) with v_interp_p1ll_f16, v_interp_p1lv_f16, and v_interp_p2_f16: omits
  PerVertexKHR / BaryCoordKHR, reads ordinary smooth vec4 inputs (where `high` cannot be distinguished),
  places %bary at the lowest free location when raw v0/v1 reads are also present, and yields to
  --f16-interp-fallback when both flags are enabled.
- I-source guards (interp_pair_src) for v_interp_p1lv_f16 and v_interp_p1ll_f16, plus Unsupported("<op> form")
  rejection on non-PS stages (vs_, cs_) and missing `attr` operands for all three *_f16 opcodes.
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support     # noqa: E402

BARY = (0.25, 0.25, 0.5)   # vertex 0, 1, 2 weights (AMD i = 0.25, j = 0.5)
LO = (0.5, 1.5, 2.5)       # exact interpolated value: 1.75 (0x3f00)
HI = (-4.0, 8.0, 16.0)     # exact interpolated value: 9.0  (0x4880)


def pack_f16(lo, hi):
    """Pack two floats as (low_f16 | high_f16 << 16) in a 32-bit word."""
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


def f16_bits(val):
    return struct.unpack("<H", struct.pack("<e", float(val)))[0]


def f32_bits(val):
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def f16_round(val: float) -> float:
    """Round a float to binary16 precision."""
    try:
        return struct.unpack("<e", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return float("inf") if val > 0 else float("-inf")


def f16_to_bits(val: float) -> int:
    """Return uint16 bits of half-precision float."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return 0x7C00 if val > 0 else 0xFC00


def bits_to_f16(bits: int) -> float:
    """Convert uint16 bits to float."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def f32_round(val: float) -> float:
    """Round a float to binary32 precision."""
    try:
        return struct.unpack("<f", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return float("inf") if val > 0 else float("-inf")


def f32_to_bits(val: float) -> int:
    """Return uint32 bits of single-precision float."""
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return 0x7F800000 if val > 0 else 0xFF800000


def bits_to_f32(bits: int) -> float:
    """Convert uint32 bits to float."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def unpack_f16_halves(word: int) -> tuple[float, float]:
    """Unpack low and high f16 halves from a 32-bit packed word."""
    lo = bits_to_f16(word & 0xFFFF)
    hi = bits_to_f16((word >> 16) & 0xFFFF)
    return lo, hi


def compute_perspective_barycentrics(l0: float, l1: float, l2: float,
                                     w0: float, w1: float, w2: float) -> tuple[float, float]:
    """Compute perspective-corrected barycentric coordinates (i_persp, j_persp)."""
    inv_w0 = 1.0 / w0
    inv_w1 = 1.0 / w1
    inv_w2 = 1.0 / w2
    inv_w_total = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2
    i_persp = (l1 * inv_w1) / inv_w_total
    j_persp = (l2 * inv_w2) / inv_w_total
    return f32_round(i_persp), f32_round(j_persp)


def v_interp_p1lv_f16_ref(i_coord: float, p0: float, p1: float) -> float:
    """Reference oracle for v_interp_p1lv_f16: D = P0 + i * (P1 - P0) in f32 intermediate precision."""
    d1 = f32_round(p1 - p0)
    m1 = f32_round(d1 * i_coord)
    return f32_round(p0 + m1)


def v_interp_p2_f16_ref(j_coord: float, p0: float, p2: float,
                        p1_accum: float, clamp: bool = False) -> int:
    """Reference oracle for v_interp_p2_f16: D = p1_accum + j * (P2 - P0), optional clamp, narrowed to f16 bits."""
    d2 = f32_round(p2 - p0)
    m2 = f32_round(d2 * j_coord)
    res_f32 = f32_round(p1_accum + m2)
    if clamp:
        res_f32 = max(0.0, min(1.0, res_f32))
    return f16_to_bits(res_f32)


def v_interp_p1lv_eval_ref(i_coord: float, j_coord: float,
                           p0: float, p1: float, p2: float,
                           clamp: bool = False) -> tuple[float, int]:
    """Evaluate paired v_interp_p1lv_f16 and v_interp_p2_f16: returns (f32_intermediate, f16_bits)."""
    p1_accum = v_interp_p1lv_f16_ref(i_coord, p0, p1)
    p2_res = v_interp_p2_f16_ref(j_coord, p0, p2, p1_accum, clamp=clamp)
    return p1_accum, p2_res


def analytical_perspective_interp_f16(p0: float, p1: float, p2: float,
                                      l0: float, l1: float, l2: float,
                                      w0: float, w1: float, w2: float,
                                      clamp: bool = False) -> int:
    """Analytical perspective interpolation ground truth narrowed to f16 bits."""
    inv_w0 = 1.0 / w0
    inv_w1 = 1.0 / w1
    inv_w2 = 1.0 / w2
    inv_w_total = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2
    numer = l0 * p0 * inv_w0 + l1 * p1 * inv_w1 + l2 * p2 * inv_w2
    val = numer / inv_w_total
    if clamp:
        val = max(0.0, min(1.0, val))
    return f16_to_bits(val)


def lerp(values, bary=BARY):
    return sum(b * v for b, v in zip(bary, values))


def isg1_dxbc(elements):
    """Synthetic DXBC container with one ISG1 chunk: elements = (register, mask, rw_mask, sysv, ctype)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, rw_mask, sysv, ctype in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, 0, sysv, ctype, reg)
        elem[24], elem[25] = mask, rw_mask
        payload += elem
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def cs_fssm(rsrc2, group):
    """Minimal compute FSSM header carrying COMPUTE_PGM_RSRC2 and local thread-group dimensions."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# Default compute FSSM (4 user SGPRs, 64x1x1 threads) and PS ISG1 (attr0.xyzw + attr1.x, float32).
CS_FSSM = cs_fssm(0x8088, (64, 1, 1))
PS_ATTR01_DXBC = isg1_dxbc([(0, 0xF, 0xF, 0, 3), (1, 0x1, 0x1, 0, 3)])
PS_POS_ATTR0_DXBC = isg1_dxbc([(0, 0xF, 0x3, 1, 3), (1, 0xF, 0xF, 0, 3)])


def translate_with_flags(name, lines, *, dxbc=support.STUB_DXBC, fssm=None,
                         bary_fallback=False, f16_halves=False):
    """Translate `lines` with explicit BARY_FALLBACK / F16_HALVES flags and return (spvasm, meta)."""
    old_bf, old_fh = rdna2spirv.BARY_FALLBACK, rdna2spirv.F16_HALVES
    rdna2spirv.BARY_FALLBACK = bary_fallback
    rdna2spirv.F16_HALVES = f16_halves
    try:
        spv = support.translate_lines(name, lines, dxbc=dxbc, fssm=fssm)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.BARY_FALLBACK = old_bf
        rdna2spirv.F16_HALVES = old_fh
    return spv, meta


# Primary v_interp_p1lv_f16 PS exercising low half, high half, clamp, standalone f32 intermediate,
# and f32/flat controls on attr0 (PerVertexKHR) and attr1 (smooth f32).
P1LV_SHADER = [
    "v_mov_b32_e32 v10, 0",
    "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
    "v_interp_p2_f16 v2, v1, attr0.x, v2",
    "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
    "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
    "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
    "v_interp_p2_f16 v4, v1, attr0.y, v4 clamp",
    "v_interp_p1lv_f16 v5, v0, attr0.y, v10 high",
    "v_interp_p2_f16 v5, v1, attr0.y, v5 high clamp",
    "v_interp_p1lv_f16 v6, v0, attr0.z, v10",       # standalone f32 intermediate
    "v_interp_p1_f32_e32 v7, v0, attr1.x",
    "v_interp_p2_f32_e32 v7, v1, attr1.x",
    "v_interp_mov_f32_e32 v8, p0, attr0.w",
    "exp mrt0 v2, v3, v4, v5",
    "s_nop 0",
    "exp mrt1 v6, v7, v8, v8 done vm",
    "s_endpgm",
]

P1LV_CLAMP_PAIR = [
    "v_mov_b32_e32 v10, 0",
    "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
    "v_interp_p2_f16 v2, v1, attr0.x, v2 clamp",
    "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
    "v_interp_p2_f16 v3, v1, attr0.x, v3 high clamp",
    "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
    "v_interp_p2_f16 v4, v1, attr0.y, v4",
    "v_interp_p1lv_f16 v5, v0, attr0.z, v10",
    "exp mrt0 v2, v3, v4, v5 done vm",
    "s_endpgm",
]


class InterpP1lvF16DefaultTests(unittest.TestCase):
    """Default PerVertexKHR translation and execution for v_interp_p1lv_f16."""

    def _default_attrs(self):
        # attr0.x: (lo=1.75, hi=9.0); attr0.y: (lo=-0.25 -> clamp 0.0, hi=1.75 -> clamp 1.0);
        # attr0.z: lo=(1.0, 3.0, 5.0) -> 3.5 as f32 intermediate; attr0.w: raw flat words.
        y_lo = (-1.0, -1.0, 0.5)    # 0.25*(-1.0) + 0.25*(-1.0) + 0.5*0.5 = -0.25 -> clamped to 0.0
        y_hi = (0.5, 1.5, 2.5)      # 1.75 -> clamped to 1.0
        z_lo = (1.0, 3.0, 5.0)      # 3.5
        w_raw = (0x1234, 0x5678, 0x9ABC)
        attr0 = [(pack_f16(lo, hi), pack_f16(yl, yh), pack_f16(zl, 0.0), wr)
                 for lo, hi, yl, yh, zl, wr in zip(LO, HI, y_lo, y_hi, z_lo, w_raw)]
        attr1 = [(f32_bits(x), 0, 0, 0) for x in (-2.0, 2.0, 4.0)]   # 0.25*(-2)+0.25*2+0.5*4 = 2.0
        return {0: attr0, 1: attr1}

    def test_p1lv_low_high_clamp_and_intermediate_end_to_end(self):
        spv, meta = translate_with_flags("ps_p1lv", P1LV_SHADER, dxbc=PS_ATTR01_DXBC)

        self.assertIn("OpCapability FragmentBarycentricKHR", spv)
        self.assertIn('OpExtension "SPV_KHR_fragment_shader_barycentric"', spv)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        self.assertIn("OpDecorate %in_attr1 Location 1", spv)
        self.assertNotIn("OpDecorate %in_attr1 PerVertexKHR", spv)
        self.assertNotIn("OpDecorate %in_attr1 Flat", spv)
        self.assertIn("UnpackHalf2x16", spv)
        self.assertEqual(meta["features"], ["fragmentShaderBarycentric"])
        self.assertEqual(meta["inputs"], [0, 1])

        attrs = self._default_attrs()
        out = support.run_fragment(spv, inputs=support.varyings(spv, attrs, BARY))
        mrt0, mrt1 = out["%out_mrt0"], out["%out_mrt1"]
        self.assertEqual([w & 0xFFFF for w in mrt0], [0x3F00, 0x4880, 0x0000, 0x3C00])
        # Standalone v_interp_p1lv_f16 exports the f32 intermediate (3.5), not f16 bits.
        self.assertEqual(f"{mrt1[0]:#010x}", f"{f32_bits(3.5):#010x}")
        self.assertEqual(f"{mrt1[1]:#010x}", f"{f32_bits(2.0):#010x}")
        self.assertEqual(mrt1[2], 0x1234)

    def test_p1lv_clamp_vs_unclamped_and_in_range(self):
        # attr0.x: lo=1.5 (clamp->1.0), hi=-0.25 (clamp->+0.0); attr0.y: lo=0.5 (in-range); attr0.z: lo=3.5 (p1lv)
        attrs = {0: [(pack_f16(1.5, -0.25), pack_f16(0.5, 0.0), pack_f16(3.5, 0.0), 0)] * 3}
        spv_clamp, _ = translate_with_flags("ps_p1lv_clamp", P1LV_CLAMP_PAIR, dxbc=PS_ATTR01_DXBC)
        self.assertIn("FClamp", spv_clamp)
        words_clamp = support.mrt0_words(spv_clamp, inputs=support.varyings(spv_clamp, attrs, BARY))
        self.assertEqual([w & 0xFFFF for w in words_clamp[:3]], [0x3C00, 0x0000, 0x3800])
        self.assertEqual(words_clamp[3], f32_bits(3.5))

        no_clamp_lines = [ln.replace(" clamp", "") for ln in P1LV_CLAMP_PAIR]
        spv_raw, _ = translate_with_flags("ps_p1lv_noclamp", no_clamp_lines, dxbc=PS_ATTR01_DXBC)
        self.assertNotIn("FClamp", spv_raw)
        words_raw = support.mrt0_words(spv_raw, inputs=support.varyings(spv_raw, attrs, BARY))
        self.assertEqual([w & 0xFFFF for w in words_raw[:3]], [f16_bits(1.5), f16_bits(-0.25), 0x3800])

    def test_p1lv_nearest_even_rounding_and_subnormal_f16(self):
        # Midpoint ties:
        # 1.0 (0x3c00) and 1.0 + 2^-10 (0x3c01) averaged at (0.5, 0.5, 0.0) -> 1.0 + 2^-11 -> 0x3c00 (even)
        # 1.0 + 2^-10 (0x3c01) and 1.0 + 2^-9 (0x3c02) averaged -> 1.0 + 3*2^-11 -> 0x3c02 (even)
        # Subnormals 0x0002 and 0x0004 averaged -> 0x0003 (subnormal preserved by f16_bits)
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
            "v_interp_p1lv_f16 v4, v0, attr0.y, v10",
            "v_interp_p2_f16 v4, v1, attr0.y, v4",
            "exp mrt0 v2, v3, v4, v4 done vm",
            "s_endpgm",
        ]
        h0 = 0x3C00
        h1 = 0x3C01
        h2 = 0x3C02
        attr0 = [
            ((h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16), 0x0002, 0, 0),
            ((h1 & 0xFFFF) | ((h2 & 0xFFFF) << 16), 0x0004, 0, 0),
            ((h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16), 0x0002, 0, 0),
        ]
        spv, _ = translate_with_flags("ps_p1lv_ties", lines, dxbc=PS_ATTR01_DXBC)
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: attr0}, (0.5, 0.5, 0.0)))
        self.assertEqual([w & 0xFFFF for w in words[:3]], [0x3C00, 0x3C02, 0x0003])

    def test_p1lv_with_sv_position_in_isg1(self):
        # SV_Position at ISG1 reg 0 occupies v2..v5; v_interp_p1lv_f16 reads v0 and vsrc2=v10 without conflict.
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v6, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v6, v1, attr0.x, v6 high",
            "v_add_f32_e32 v7, v2, v3",
            "exp mrt0 v6, v7, v6, v7 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with_flags("ps_p1lv_pos", lines, dxbc=PS_POS_ATTR0_DXBC)
        self.assertEqual(meta["ps_system_inputs"], ["0", "1"])
        attr0 = [(pack_f16(0.5, hi), 0, 0, 0) for hi in HI]
        inputs = support.varyings(spv, {0: attr0}, BARY)
        inputs["%frag_coord"] = (1.25, 2.5, 0.0, 1.0)
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4880)
        self.assertEqual(words[1], f32_bits(3.75))

    def test_negative_controls_p1lv(self):
        attrs = {0: [(pack_f16(1.5, -0.25), pack_f16(0.5, 0.0), pack_f16(3.5, 0.0), 0)] * 3}
        spv, _ = translate_with_flags("ps_p1lv_neg", P1LV_CLAMP_PAIR, dxbc=PS_ATTR01_DXBC)
        inputs = support.varyings(spv, attrs, BARY)
        baseline = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual([w & 0xFFFF for w in baseline[:3]], [0x3C00, 0x0000, 0x3800])

        # Sabotage 1: force high-half extract (component 1) to low half (component 0).
        bad_half = re.sub(r"OpCompositeExtract %float (%\S+) 1", r"OpCompositeExtract %float \1 0", spv)
        self.assertNotEqual(bad_half, spv)
        self.assertNotEqual(support.mrt0_words(bad_half, inputs=inputs), baseline)

        # Sabotage 2: drop FClamp by replacing FClamp %v %f0 %f1 with FMax %v %f0.
        bad_clamp = re.sub(r"OpExtInst %float %glsl FClamp (%\S+) %f0 %f1", r"OpExtInst %float %glsl FMax \1 %f0", spv)
        self.assertNotEqual(bad_clamp, spv)
        self.assertNotEqual(support.mrt0_words(bad_clamp, inputs=inputs), baseline)


class InterpP1lvF16HalvesFallbackTests(unittest.TestCase):
    """--f16-interp-fallback (F16_HALVES) behaviour with v_interp_p1lv_f16."""

    def test_p1lv_f16_halves_translation_and_execution(self):
        spv, meta = translate_with_flags("ps_p1lv_fh", P1LV_CLAMP_PAIR,
                                         dxbc=PS_ATTR01_DXBC, f16_halves=True)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("BaryCoordKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("fragmentShaderBarycentric", meta["features"])
        self.assertIn("OpDecorate %in_f16h0_0 Location 28", spv)
        self.assertIn("OpDecorate %in_f16h0_1 Location 29", spv)
        self.assertNotIn("OpDecorate %in_f16h0_0 Flat", spv)
        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})
        self.assertEqual(meta["inputs"], [28, 29])

        inputs = {
            "%in_f16h0_0": (lerp(LO), 0.5, 3.5, 0.0),   # attr0 low: 1.75 (clamped->1.0), 0.5, 3.5 (p1lv f32)
            "%in_f16h0_1": (lerp(HI), 0.0, 0.0, 0.0),   # attr0 high: 9.0 (clamped->1.0)
        }
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual([w & 0xFFFF for w in words[:3]], [0x3C00, 0x3C00, 0x3800])
        self.assertEqual(words[3], f32_bits(3.5))

    def test_p1lv_f16_halves_location_clash_packs_and_overflow_rejects(self):
        clash = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1_f32_e32 v3, v0, attr29.x",
            "v_interp_p2_f32_e32 v3, v1, attr29.x",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with_flags("ps_p1lv_fh_packed", clash, f16_halves=True)
        self.assertEqual(meta["f16_halves"], {"0": [27, 28]})
        self.assertIn("OpDecorate %in_f16h0_0 Location 27", spv)
        self.assertIn("OpDecorate %in_attr29 Location 29", spv)

        overflow = ["v_mov_b32_e32 v10, 0"]
        for a in range(29):
            overflow += [f"v_interp_p1_f32_e32 v{20 + a}, v0, attr{a}.x",
                         f"v_interp_p2_f32_e32 v{20 + a}, v1, attr{a}.x"]
        overflow += [
            "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "exp mrt0 v2, v20, v20, v20 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_with_flags("ps_p1lv_fh_overflow", overflow, f16_halves=True)
        self.assertIn("decoded-half locations", str(cm.exception))


class InterpF16BaryFallbackTests(unittest.TestCase):
    """--bary-fallback (BARY_FALLBACK) translation and execution for *_f16 ops."""

    def test_bary_fallback_for_p1ll_p1lv_p2_uses_smooth_inputs_without_per_vertex(self):
        lines = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.y, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.y, v3 high clamp",
            "v_interp_p1lv_f16 v4, v0, attr0.z, v10",       # standalone f32 intermediate
            "exp mrt0 v2, v3, v4, v2 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with_flags("ps_f16_bf", lines, dxbc=PS_ATTR01_DXBC, bary_fallback=True)

        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("BaryCoordKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("SPV_KHR_fragment_shader_barycentric", spv)
        self.assertNotIn("%bary", spv)
        self.assertIn("OpDecorate %in_attr0 Location 0", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)
        self.assertEqual(meta["features"], [])
        self.assertEqual(meta["inputs"], [0])
        self.assertNotIn("bary_fallback_input", meta)
        self.assertNotIn("f16_halves", meta)

        # Under --bary-fallback there are no per-vertex inputs, so high reads the same smooth f32 channel.
        attrs = {0: [(f32_bits(x), f32_bits(y), f32_bits(z), 0)
                     for x, y, z in zip((0.5, 1.5, 2.5), (-1.0, -1.0, 0.5), (1.0, 3.0, 5.0))]}
        words = support.mrt0_words(spv, inputs=support.varyings(spv, attrs, BARY))
        self.assertEqual(words[0] & 0xFFFF, 0x3F00)      # 1.75
        self.assertEqual(words[1] & 0xFFFF, 0x0000)      # -0.25 clamped to +0.0
        self.assertEqual(words[2], f32_bits(3.5))        # standalone p1lv f32 intermediate

        # Confirm `high` and non-`high` read identically under --bary-fallback.
        lines_same = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "v_interp_p1lv_f16 v3, v0, attr0.x, v10 high",
            "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv_same, _ = translate_with_flags("ps_f16_bf_same", lines_same, dxbc=PS_ATTR01_DXBC, bary_fallback=True)
        words_same = support.mrt0_words(spv_same, inputs={"%in_attr0": (1.75, 0.0, 0.0, 0.0)})
        self.assertEqual([w & 0xFFFF for w in words_same[:2]], [0x3F00, 0x3F00])

    def test_bary_fallback_with_raw_bary_read_and_sparse_attr(self):
        for attr_idx, want_bary_loc in ((0, 1), (30, 0)):
            with self.subTest(attr=attr_idx):
                lines = [
                    "v_mov_b32_e32 v10, 0",
                    f"v_interp_p1lv_f16 v2, v0, attr{attr_idx}.x, v10",
                    f"v_interp_p2_f16 v2, v1, attr{attr_idx}.x, v2",
                    "v_mov_b32_e32 v4, v0",
                    "v_mov_b32_e32 v5, v1",
                    "exp mrt0 v2, v4, v5, v2 done vm",
                    "s_endpgm",
                ]
                spv, meta = translate_with_flags(f"ps_f16_bf_raw_{attr_idx}", lines, bary_fallback=True)
                self.assertNotIn("PerVertexKHR", spv)
                self.assertNotIn("BaryCoordKHR", spv)
                self.assertNotIn("FragmentBarycentricKHR", spv)
                self.assertIn(f"OpDecorate %in_attr{attr_idx} Location {attr_idx}", spv)
                self.assertIn(f"OpDecorate %bary Location {want_bary_loc}", spv)
                self.assertIsNone(re.search(r"OpDecorate %bary (Flat|NoPerspective)", spv))
                self.assertEqual(meta["bary_fallback_input"], {"location": want_bary_loc})
                self.assertEqual(meta["ps_system_inputs"], ["bary_i", "bary_j"])

                words = support.mrt0_words(spv, inputs={
                    f"%in_attr{attr_idx}": (1.75, 0.0, 0.0, 0.0),
                    "%bary": BARY,
                })
                self.assertEqual(words[0] & 0xFFFF, 0x3F00)
                self.assertEqual(words[1], f32_bits(BARY[1]))
                self.assertEqual(words[2], f32_bits(BARY[2]))

    def test_f16_halves_takes_precedence_over_bary_fallback(self):
        spv, meta = translate_with_flags("ps_both_flags", P1LV_CLAMP_PAIR,
                                         dxbc=PS_ATTR01_DXBC, bary_fallback=True, f16_halves=True)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertIn("OpDecorate %in_f16h0_0 Location 28", spv)
        self.assertIn("OpDecorate %in_f16h0_1 Location 29", spv)
        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})


class InterpF16GuardsAndRejectionsTests(unittest.TestCase):
    """interp_pair_src guards and Unsupported('<op> form') stage/attr rejections."""

    def test_p1lv_and_p1ll_i_source_guards(self):
        bad_cases = [
            ("v_interp_p1lv_f16 I source v4", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v4, attr0.x, v10",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("v_interp_p1lv_f16 I source v1", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v1, attr0.x, v10",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("v_interp_p1ll_f16 I source v1", [
                "v_interp_p1ll_f16 v2, v1, attr0.x",
                "v_interp_p2_f16 v2, v1, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            # An I source holding a zeroed v0 ("v_mov v0, 0; v_mov v6, v0") is a computed I of 0, which the translator
            # models since #1545 (computed I/J barycentrics) instead of rejecting; it is no longer a bad case here.
            ("v_interp_p2_f16 J source v0", [
                "v_mov_b32_e32 v10, 0",
                "v_interp_p1lv_f16 v2, v0, attr0.x, v10",
                "v_interp_p2_f16 v2, v0, attr0.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
        ]
        for msg, lines in bad_cases:
            with self.subTest(msg=msg):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags("ps_guard_bad", lines, dxbc=PS_ATTR01_DXBC)
                self.assertIn(msg, str(cm.exception))

    def test_p1lv_accepts_saved_copy_of_v0_even_if_v0_later_zeroed(self):
        lines = [
            "v_mov_b32_e32 v6, v0",
            "v_mov_b32_e32 v7, v1",
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1lv_f16 v2, v6, attr0.x, v10 high",
            "v_interp_p2_f16 v2, v7, attr0.x, v2 high",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv, _ = translate_with_flags("ps_p1lv_copy", lines, dxbc=PS_ATTR01_DXBC)
        attrs = {0: [(pack_f16(lo, hi), 0, 0, 0) for lo, hi in zip(LO, HI)]}
        words = support.mrt0_words(spv, inputs=support.varyings(spv, attrs, BARY))
        self.assertEqual(words[0] & 0xFFFF, 0x4880)

    def test_non_ps_stages_rejected_for_all_f16_interp_ops(self):
        ops = [
            ("v_interp_p1ll_f16", "v_interp_p1ll_f16 v2, v0, attr0.x"),
            ("v_interp_p1lv_f16", "v_interp_p1lv_f16 v2, v0, attr0.x, v3"),
            ("v_interp_p2_f16", "v_interp_p2_f16 v2, v1, attr0.x, v2"),
        ]
        for op, insn in ops:
            with self.subTest(stage="vs", op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"vs_{op}", [insn, "exp pos0 v2, v2, v2, v2 done", "s_endpgm"])
                self.assertIn(f"{op} form", str(cm.exception))
            with self.subTest(stage="cs", op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"cs_{op}", [insn, "s_endpgm"], fssm=CS_FSSM)
                self.assertIn(f"{op} form", str(cm.exception))

    def test_missing_attr_operand_rejected_for_all_f16_interp_ops(self):
        ops = [
            ("v_interp_p1ll_f16", "v_interp_p1ll_f16 v2, v0, v3"),
            ("v_interp_p1lv_f16", "v_interp_p1lv_f16 v2, v0, v3, v4"),
            ("v_interp_p2_f16", "v_interp_p2_f16 v2, v1, v3, v2"),
        ]
        for op, insn in ops:
            with self.subTest(op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate_with_flags(f"ps_noattr_{op}",
                                         [insn, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                                         dxbc=PS_ATTR01_DXBC)
                self.assertIn(f"{op} form", str(cm.exception))


class InterpP1lvF16ReferenceOracleTests(unittest.TestCase):
    """Mathematical reference oracle tests for half-precision perspective interpolation."""

    def test_vertex_boundary_conditions(self):
        p0, p1, p2 = 2.0, 6.0, 14.0
        p1_acc, p2_bits = v_interp_p1lv_eval_ref(0.0, 0.0, p0, p1, p2)
        self.assertEqual(p1_acc, 2.0)
        self.assertEqual(p2_bits, f16_to_bits(2.0))

        p1_acc, p2_bits = v_interp_p1lv_eval_ref(1.0, 0.0, p0, p1, p2)
        self.assertEqual(p1_acc, 6.0)
        self.assertEqual(p2_bits, f16_to_bits(6.0))

        p1_acc, p2_bits = v_interp_p1lv_eval_ref(0.0, 1.0, p0, p1, p2)
        self.assertEqual(p1_acc, 2.0)
        self.assertEqual(p2_bits, f16_to_bits(14.0))

    def test_edge_midpoints_and_centroid(self):
        p0, p1, p2 = 4.0, 8.0, 16.0
        _, res_01 = v_interp_p1lv_eval_ref(0.5, 0.0, p0, p1, p2)
        self.assertEqual(res_01, f16_to_bits(6.0))

        _, res_02 = v_interp_p1lv_eval_ref(0.0, 0.5, p0, p1, p2)
        self.assertEqual(res_02, f16_to_bits(10.0))

        _, res_12 = v_interp_p1lv_eval_ref(0.5, 0.5, p0, p1, p2)
        self.assertEqual(res_12, f16_to_bits(12.0))

        third = f32_round(1.0 / 3.0)
        _, res_c = v_interp_p1lv_eval_ref(third, third, p0, p1, p2)
        expected_c = (4.0 + 8.0 + 16.0) / 3.0
        self.assertAlmostEqual(bits_to_f16(res_c), expected_c, delta=0.01)

    def test_perspective_parameter_calculation_equal_vs_varying_depth(self):
        p0, p1, p2 = 10.0, 50.0, 100.0
        i_eq, j_eq = compute_perspective_barycentrics(0.2, 0.5, 0.3, 4.0, 4.0, 4.0)
        self.assertAlmostEqual(i_eq, 0.5, places=5)
        self.assertAlmostEqual(j_eq, 0.3, places=5)
        _, res_eq = v_interp_p1lv_eval_ref(i_eq, j_eq, p0, p1, p2)
        self.assertEqual(res_eq, analytical_perspective_interp_f16(p0, p1, p2, 0.2, 0.5, 0.3, 4.0, 4.0, 4.0))

        cases = [
            (0.333333, 0.333333, 0.333334, 1.0, 2.0, 4.0, 1.0, 5.0, 10.0),
            (0.1, 0.7, 0.2, 0.5, 5.0, 2.5, 0.0, 1.0, 0.5),
            (0.6, 0.2, 0.2, 10.0, 1.0, 5.0, -10.0, 20.0, 50.0),
        ]
        for l0, l1, l2, w0, w1, w2, a0, a1, a2 in cases:
            with self.subTest(w=(w0, w1, w2), a=(a0, a1, a2)):
                i_p, j_p = compute_perspective_barycentrics(l0, l1, l2, w0, w1, w2)
                _, got_bits = v_interp_p1lv_eval_ref(i_p, j_p, a0, a1, a2)
                want_bits = analytical_perspective_interp_f16(a0, a1, a2, l0, l1, l2, w0, w1, w2)
                self.assertAlmostEqual(bits_to_f16(got_bits), bits_to_f16(want_bits), delta=0.05)

    def test_standalone_p1lv_f32_precision(self):
        p0 = 1.0009765625
        p1 = 1.001953125
        i_c = 0.3333333432674408
        p1_accum = v_interp_p1lv_f16_ref(i_c, p0, p1)
        self.assertNotEqual(p1_accum, f16_round(p1_accum))

    def test_unpack_f16_halves_and_pack(self):
        lo_in, hi_in = 1.75, -9.5
        packed = pack_f16(lo_in, hi_in)
        lo_out, hi_out = unpack_f16_halves(packed)
        self.assertEqual(lo_out, lo_in)
        self.assertEqual(hi_out, hi_in)

    def test_vop3_clamp_oracle(self):
        _, clamped_neg = v_interp_p1lv_eval_ref(0.25, 0.5, -2.0, -4.0, 1.0, clamp=True)
        self.assertEqual(clamped_neg, 0x0000)

        _, clamped_high = v_interp_p1lv_eval_ref(0.25, 0.5, 2.0, 4.0, 3.0, clamp=True)
        self.assertEqual(clamped_high, 0x3C00)

        _, unclamped = v_interp_p1lv_eval_ref(0.25, 0.5, 0.0, 0.5, 1.0, clamp=False)
        _, clamped_mid = v_interp_p1lv_eval_ref(0.25, 0.5, 0.0, 0.5, 1.0, clamp=True)
        self.assertEqual(unclamped, clamped_mid)

    def test_nearest_even_tie_rounding_oracle(self):
        tie_even = 1.0 + 2.0**-11
        self.assertEqual(f16_to_bits(tie_even), 0x3C00)

        tie_odd = 1.0 + 2.0**-10 + 2.0**-11
        self.assertEqual(f16_to_bits(tie_odd), 0x3C02)


class SimdWaveInterpP1lv:
    """Simulates an RDNA2 SIMD wave (Wave32 or Wave64) executing perspective interpolation instructions.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - Per-lane VGPR bank (256 registers per lane).
      - Shared SGPR bank (128 scalar registers).
      - EXEC mask (32-bit or 64-bit) gating lane evaluation and register writes.
      - Inactive lane preservation: disabled lanes keep their prior VGPR state.
      - v_interp_p1lv_f16: evaluates first-axis perspective interpolation intermediate in f32.
      - v_interp_p2_f16: accumulates second axis, applies optional clamp, narrows to f16 (nearest-even).
    """

    def __init__(self, wave_size: int = 32, exec_mask: int | None = None):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size: {wave_size}")
        self.wave_size = wave_size
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask = self.mask_all if exec_mask is None else (exec_mask & self.mask_all)
        self.sgprs = [0] * 128
        self.vgprs = [[0] * 256 for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_lane_barycentrics(self, lane: int, i_coord: float, j_coord: float,
                              v_i: int = 0, v_j: int = 1):
        """Set barycentric I and J coordinates in given VGPRs for lane."""
        self.vgprs[lane][v_i] = f32_to_bits(i_coord)
        self.vgprs[lane][v_j] = f32_to_bits(j_coord)

    def v_interp_p1lv_f16(self, vdst: int, vsrc_i: int,
                          attr_packed: tuple[int, int, int],
                          high: bool = False):
        """Execute v_interp_p1lv_f16 across all active SIMD lanes."""
        vals = [unpack_f16_halves(p)[1 if high else 0] for p in attr_packed]
        p0, p1 = vals[0], vals[1]

        for lane in range(self.wave_size):
            if not self.is_lane_active(lane):
                continue
            i_coord = bits_to_f32(self.vgprs[lane][vsrc_i])
            accum_f32 = v_interp_p1lv_f16_ref(i_coord, p0, p1)
            self.vgprs[lane][vdst] = f32_to_bits(accum_f32)

    def v_interp_p2_f16(self, vdst: int, vsrc_j: int,
                        attr_packed: tuple[int, int, int],
                        partial_src: int,
                        high: bool = False, clamp: bool = False):
        """Execute v_interp_p2_f16 across all active SIMD lanes."""
        vals = [unpack_f16_halves(p)[1 if high else 0] for p in attr_packed]
        p0, p2 = vals[0], vals[2]

        for lane in range(self.wave_size):
            if not self.is_lane_active(lane):
                continue
            j_coord = bits_to_f32(self.vgprs[lane][vsrc_j])
            accum_f32 = bits_to_f32(self.vgprs[lane][partial_src])
            res_bits = v_interp_p2_f16_ref(j_coord, p0, p2, accum_f32, clamp=clamp)
            self.vgprs[lane][vdst] = res_bits


class InterpP1lvSimdMultiLaneTests(unittest.TestCase):
    """SIMD multi-lane wave evaluation tests for half-precision perspective interpolation."""

    def test_wave32_uniform_triangle_raster_grid(self):
        wave = SimdWaveInterpP1lv(wave_size=32)
        attr0 = (pack_f16(1.0, 2.0), pack_f16(3.0, 6.0), pack_f16(5.0, 10.0))

        bary_coords = []
        for lane in range(32):
            i_val = (lane % 8) * 0.1
            j_val = (lane // 8) * 0.15
            bary_coords.append((i_val, j_val))
            wave.set_lane_barycentrics(lane, i_val, j_val)

        wave.v_interp_p1lv_f16(vdst=2, vsrc_i=0, attr_packed=attr0, high=False)
        wave.v_interp_p2_f16(vdst=2, vsrc_j=1, attr_packed=attr0, partial_src=2, high=False)

        wave.v_interp_p1lv_f16(vdst=3, vsrc_i=0, attr_packed=attr0, high=True)
        wave.v_interp_p2_f16(vdst=3, vsrc_j=1, attr_packed=attr0, partial_src=3, high=True)

        for lane, (i_val, j_val) in enumerate(bary_coords):
            _, want_lo = v_interp_p1lv_eval_ref(i_val, j_val, 1.0, 3.0, 5.0)
            _, want_hi = v_interp_p1lv_eval_ref(i_val, j_val, 2.0, 6.0, 10.0)
            self.assertEqual(wave.vgprs[lane][2], want_lo)
            self.assertEqual(wave.vgprs[lane][3], want_hi)

    def test_wave64_divergent_exec_masking(self):
        wave = SimdWaveInterpP1lv(wave_size=64)
        attr0 = (pack_f16(2.0, -1.0), pack_f16(4.0, 3.0), pack_f16(8.0, 7.0))

        for lane in range(64):
            wave.vgprs[lane][2] = 0xDEAD0000 | lane

        active_mask = 0
        for lane in range(64):
            if lane % 3 != 0:
                active_mask |= (1 << lane)
        wave.exec_mask = active_mask

        for lane in range(64):
            wave.set_lane_barycentrics(lane, 0.25, 0.5)

        wave.v_interp_p1lv_f16(vdst=2, vsrc_i=0, attr_packed=attr0, high=False)
        wave.v_interp_p2_f16(vdst=2, vsrc_j=1, attr_packed=attr0, partial_src=2, high=False)

        _, want_bits = v_interp_p1lv_eval_ref(0.25, 0.5, 2.0, 4.0, 8.0)
        for lane in range(64):
            if lane % 3 != 0:
                self.assertEqual(wave.vgprs[lane][2], want_bits)
            else:
                self.assertEqual(wave.vgprs[lane][2], 0xDEAD0000 | lane)

    def test_checkerboard_and_half_wave_divergence(self):
        wave = SimdWaveInterpP1lv(wave_size=32, exec_mask=0x55555555)
        for lane in range(32):
            wave.vgprs[lane][4] = 0xCAFE0000 | lane
            wave.set_lane_barycentrics(lane, 0.2, 0.4)

        attr0 = (pack_f16(10.0, 0.0), pack_f16(20.0, 0.0), pack_f16(30.0, 0.0))
        wave.v_interp_p1lv_f16(vdst=4, vsrc_i=0, attr_packed=attr0, high=False)
        wave.v_interp_p2_f16(vdst=4, vsrc_j=1, attr_packed=attr0, partial_src=4, high=False)

        _, want_bits = v_interp_p1lv_eval_ref(0.2, 0.4, 10.0, 20.0, 30.0)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][4], want_bits)
            else:
                self.assertEqual(wave.vgprs[lane][4], 0xCAFE0000 | lane)

        wave_odd = SimdWaveInterpP1lv(wave_size=32, exec_mask=0xAAAAAAAA)
        for lane in range(32):
            wave_odd.vgprs[lane][4] = 0xBEEF0000 | lane
            wave_odd.set_lane_barycentrics(lane, 0.1, 0.7)

        wave_odd.v_interp_p1lv_f16(vdst=4, vsrc_i=0, attr_packed=attr0, high=False)
        wave_odd.v_interp_p2_f16(vdst=4, vsrc_j=1, attr_packed=attr0, partial_src=4, high=False)

        _, want_bits_odd = v_interp_p1lv_eval_ref(0.1, 0.7, 10.0, 20.0, 30.0)
        for lane in range(32):
            if lane % 2 == 1:
                self.assertEqual(wave_odd.vgprs[lane][4], want_bits_odd)
            else:
                self.assertEqual(wave_odd.vgprs[lane][4], 0xBEEF0000 | lane)

    def test_multi_lane_high_vs_low_packed_independence(self):
        wave = SimdWaveInterpP1lv(wave_size=32)
        attr0 = (pack_f16(-2.0, 100.0), pack_f16(0.0, 200.0), pack_f16(4.0, 300.0))
        for lane in range(32):
            wave.set_lane_barycentrics(lane, lane / 64.0, (31 - lane) / 64.0)

        wave.v_interp_p1lv_f16(vdst=2, vsrc_i=0, attr_packed=attr0, high=False)
        wave.v_interp_p2_f16(vdst=2, vsrc_j=1, attr_packed=attr0, partial_src=2, high=False)

        wave.v_interp_p1lv_f16(vdst=3, vsrc_i=0, attr_packed=attr0, high=True)
        wave.v_interp_p2_f16(vdst=3, vsrc_j=1, attr_packed=attr0, partial_src=3, high=True)

        for lane in range(32):
            i_c = lane / 64.0
            j_c = (31 - lane) / 64.0
            _, want_lo = v_interp_p1lv_eval_ref(i_c, j_c, -2.0, 0.0, 4.0)
            _, want_hi = v_interp_p1lv_eval_ref(i_c, j_c, 100.0, 200.0, 300.0)
            self.assertEqual(wave.vgprs[lane][2], want_lo)
            self.assertEqual(wave.vgprs[lane][3], want_hi)

    def test_multi_sample_shader_translation_evaluation(self):
        spv, _ = translate_with_flags("ps_p1lv_multi", P1LV_SHADER, dxbc=PS_ATTR01_DXBC)
        y_lo = (-1.0, -1.0, 0.5)
        y_hi = (0.5, 1.5, 2.5)
        z_lo = (1.0, 3.0, 5.0)
        w_raw = (0x1234, 0x5678, 0x9ABC)
        attr0 = [(pack_f16(lo, hi), pack_f16(yl, yh), pack_f16(zl, 0.0), wr)
                 for lo, hi, yl, yh, zl, wr in zip(LO, HI, y_lo, y_hi, z_lo, w_raw)]
        attr1 = [(f32_bits(x), 0, 0, 0) for x in (-2.0, 2.0, 4.0)]

        sample_points = [
            (0.1, 0.1),
            (0.2, 0.3),
            (0.5, 0.2),
            (0.0, 0.8),
            (0.33, 0.33),
            (0.7, 0.1),
            (0.1, 0.6),
            (0.25, 0.5),
        ]
        for i_c, j_c in sample_points:
            with self.subTest(i=i_c, j=j_c):
                bary_w = (1.0 - i_c - j_c, i_c, j_c)
                inputs = support.varyings(spv, {0: attr0, 1: attr1}, bary_w)
                out = support.run_fragment(spv, inputs=inputs)
                mrt0, mrt1 = out["%out_mrt0"], out["%out_mrt1"]

                _, want_x_lo = v_interp_p1lv_eval_ref(i_c, j_c, LO[0], LO[1], LO[2], clamp=False)
                _, want_x_hi = v_interp_p1lv_eval_ref(i_c, j_c, HI[0], HI[1], HI[2], clamp=False)
                _, want_y_lo = v_interp_p1lv_eval_ref(i_c, j_c, y_lo[0], y_lo[1], y_lo[2], clamp=True)
                _, want_y_hi = v_interp_p1lv_eval_ref(i_c, j_c, y_hi[0], y_hi[1], y_hi[2], clamp=True)
                want_z_lo_f32 = v_interp_p1lv_f16_ref(i_c, z_lo[0], z_lo[1]) + (z_lo[2] - z_lo[0]) * j_c

                self.assertEqual(mrt0[0] & 0xFFFF, want_x_lo)
                self.assertEqual(mrt0[1] & 0xFFFF, want_x_hi)
                self.assertEqual(mrt0[2] & 0xFFFF, want_y_lo)
                self.assertEqual(mrt0[3] & 0xFFFF, want_y_hi)
                self.assertAlmostEqual(bits_to_f32(mrt1[0]), want_z_lo_f32, delta=1e-4)


class InterpP1lvSubnormalAndClippingTests(unittest.TestCase):
    """Subnormal weights, denormal attribute preservation, and clipping boundary tests."""

    def test_subnormal_f16_attribute_interpolation(self):
        p0 = bits_to_f16(0x0002)
        p1 = bits_to_f16(0x0004)
        p2 = bits_to_f16(0x0008)

        _, res_mid = v_interp_p1lv_eval_ref(0.5, 0.0, p0, p1, p2)
        self.assertEqual(res_mid, 0x0003)

        _, res_v0 = v_interp_p1lv_eval_ref(0.0, 0.0, p0, p1, p2)
        self.assertEqual(res_v0, 0x0002)

        _, res_v2 = v_interp_p1lv_eval_ref(0.0, 0.5, p0, p1, p2)
        self.assertEqual(res_v2, 0x0005)

    def test_subnormal_barycentric_weights(self):
        p0, p1, p2 = 5.0, 25.0, 50.0
        subnormal_w = 1e-15
        _, res = v_interp_p1lv_eval_ref(subnormal_w, subnormal_w, p0, p1, p2)
        self.assertEqual(res, f16_to_bits(5.0))

    def test_subnormal_to_normal_transition(self):
        p0 = 0.0
        p1 = bits_to_f16(0x0400)
        p2 = 0.0

        _, res_half = v_interp_p1lv_eval_ref(0.5, 0.0, p0, p1, p2)
        self.assertEqual(res_half, 0x0200)

        _, res_quarter = v_interp_p1lv_eval_ref(0.25, 0.0, p0, p1, p2)
        self.assertEqual(res_quarter, 0x0100)

        _, res_one = v_interp_p1lv_eval_ref(1.0, 0.0, p0, p1, p2)
        self.assertEqual(res_one, 0x0400)

    def test_hypotenuse_boundary_clipping(self):
        p0, p1, p2 = 10.0, 20.0, 30.0
        for i_c, j_c in [(0.5, 0.5), (0.25, 0.75), (0.0, 1.0), (1.0, 0.0)]:
            with self.subTest(i=i_c, j=j_c):
                _, got_bits = v_interp_p1lv_eval_ref(i_c, j_c, p0, p1, p2)
                expected_val = i_c * p1 + j_c * p2
                self.assertEqual(got_bits, f16_to_bits(expected_val))

    def test_extreme_extrapolation_and_clamp_boundaries(self):
        p0, p1, p2 = 0.5, 1.5, 2.5
        _, clamped_low = v_interp_p1lv_eval_ref(-2.0, 0.5, p0, p1, p2, clamp=True)
        self.assertEqual(clamped_low, 0x0000)

        _, unclamped_low = v_interp_p1lv_eval_ref(-2.0, 0.5, p0, p1, p2, clamp=False)
        self.assertLess(bits_to_f16(unclamped_low), 0.0)

        _, clamped_high = v_interp_p1lv_eval_ref(5.0, 2.0, p0, p1, p2, clamp=True)
        self.assertEqual(clamped_high, 0x3C00)

        _, unclamped_high = v_interp_p1lv_eval_ref(5.0, 2.0, p0, p1, p2, clamp=False)
        self.assertGreater(bits_to_f16(unclamped_high), 1.0)

    def test_negative_zero_clamping(self):
        p0, p1, p2 = -0.1, -0.2, -0.5
        _, res_bits = v_interp_p1lv_eval_ref(0.3, 0.3, p0, p1, p2, clamp=True)
        self.assertEqual(res_bits, 0x0000)
        self.assertNotEqual(res_bits, 0x8000)


class InterpP1lvNegativeControlsTests(unittest.TestCase):
    """Negative controls and mutation detection tests for half-precision interpolation."""

    def test_mutation_axis_swap_detection(self):
        p0, p1, p2 = 1.0, 5.0, 20.0
        i_c, j_c = 0.2, 0.7
        _, normal_bits = v_interp_p1lv_eval_ref(i_c, j_c, p0, p1, p2)
        _, swapped_bits = v_interp_p1lv_eval_ref(j_c, i_c, p0, p1, p2)
        self.assertNotEqual(normal_bits, swapped_bits)

    def test_mutation_half_selection_detection(self):
        packed = pack_f16(1.5, 45.0)
        lo, hi = unpack_f16_halves(packed)
        self.assertEqual(lo, 1.5)
        self.assertEqual(hi, 45.0)
        self.assertNotEqual(lo, hi)

    def test_mutation_clamp_removal_detection(self):
        p0, p1, p2 = 2.0, 4.0, 6.0
        _, clamped = v_interp_p1lv_eval_ref(0.5, 0.5, p0, p1, p2, clamp=True)
        _, unclamped = v_interp_p1lv_eval_ref(0.5, 0.5, p0, p1, p2, clamp=False)
        self.assertEqual(clamped, 0x3C00)
        self.assertNotEqual(unclamped, clamped)

    def test_mutation_rtz_vs_nearest_even(self):
        val = 1.0 + 2.0**-11
        rne_bits = f16_to_bits(val)
        self.assertEqual(rne_bits, 0x3C00)

        val_odd = 1.0 + 2.0**-10 + 2.0**-11
        rne_odd_bits = f16_to_bits(val_odd)
        self.assertEqual(rne_odd_bits, 0x3C02)
        rtz_odd_bits = 0x3C01
        self.assertNotEqual(rne_odd_bits, rtz_odd_bits)


if __name__ == "__main__":
    unittest.main()
