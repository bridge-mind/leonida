from __future__ import annotations

"""Float min/max/min3/max3/med3 with quiet/signaling NaN, payload preservation, and v_fract near 1.0.

The expected words were measured on RDNA2 hardware: AMD Radeon RX 6800 (gfx1030), with the hw-oracle tool from
the AnyPS5 project, in all four IEEE/DX10_CLAMP mode combinations and in wave32 and wave64 (docs/design.md,
"Float min/max/med3 with NaN, v_fract"). The rules:
  * IEEE 754 binary32 NaN layout:
    - Sign bit: [31]
    - Exponent: [30:23] = 0xFF (all 1s)
    - Mantissa: [22:0] != 0
    - Quiet NaN (qNaN): MSB of mantissa (bit 22) is 1. Payload in bits [21:0]. Canonical qNaN is 0x7FC00000.
    - Signaling NaN (sNaN): MSB of mantissa (bit 22) is 0, payload in bits [21:0] is non-zero.
  * RDNA2 ALU NaN handling & quieting:
    - When an sNaN enters a float ALU operation, it is converted to a quiet NaN by setting bit 22 to 1.
    - The custom mantissa payload in bits [21:0] is preserved where possible by unary rounding / transcendental
      ops (v_floor_f32, v_ceil_f32, v_trunc_f32, v_rndne_f32).
  * v_min/v_max (f32, f16, each half of v_pk_min/max_f16): one NaN operand gives the other operand, bits
    unchanged; -0 < +0. Both NaN: returns src0's exact bits (preserving quiet/signalling NaN payloads).
  * v_min3/v_max3 = f(f(s0, s1), s2) with that min/max.
  * v_med3: with any NaN operand the result is min3(s0, s1, s2); otherwise the median.
  * v_fract_f32: a result that rounds to 1.0 is 0x3f7fffff instead (f16: 0x3bff); fract(-0) = +0,
    fract(+-inf) = NaN.
Inputs come from memory through s_buffer_load (V# in s[0:3] -> 0x1000), as in the GPU fixtures
runtime/test/shaders/ps_fnan_med3.asm and ps_fnan_fract.asm, which are run here too.
"""
import gc
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402  (puts LEONIDA_TOOLS_DIR or tools/ on the path)
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

MASK32 = 0xFFFFFFFF
EXP_MASK32 = 0x7F800000
MANTISSA_MASK32 = 0x007FFFFF
QUIET_BIT32 = 0x00400000
PAYLOAD_MASK32 = 0x003FFFFF
SIGN_BIT32 = 0x80000000

NAN_A, NAN_B = 0x7FC10001, 0xFFC20002          # two quiet NaNs with different signs and payloads
M25, P35, P50, P70, ONE = 0xC0200000, 0x40600000, 0x40A00000, 0x40E00000, 0x3F800000   # -2.5 3.5 5 7 1
PZ, NZ = 0x00000000, 0x80000000
PUSH = [0x1000, 0, 32, 0] + [0] * 28            # V#: base 0x1000, num_records 32


def is_nan32(w: int) -> bool:
    return (w & EXP_MASK32) == EXP_MASK32 and bool(w & MANTISSA_MASK32)


def is_qnan32(w: int) -> bool:
    return is_nan32(w) and bool(w & QUIET_BIT32)


def is_snan32(w: int) -> bool:
    return is_nan32(w) and not bool(w & QUIET_BIT32)


def nan_payload32(w: int) -> int:
    return w & PAYLOAD_MASK32


def nan_sign32(w: int) -> int:
    return (w >> 31) & 1


def make_qnan32(sign: int = 0, payload: int = 0) -> int:
    return ((sign & 1) << 31) | EXP_MASK32 | QUIET_BIT32 | (payload & PAYLOAD_MASK32)


def make_snan32(sign: int = 0, payload: int = 1) -> int:
    pay = payload & PAYLOAD_MASK32
    if pay == 0:
        pay = 1
    return ((sign & 1) << 31) | EXP_MASK32 | pay


def quiet_nan32(w: int) -> int:
    """RDNA2 quieting: forces bit 22 to 1, preserving sign and mantissa payload."""
    return (w & MASK32) | QUIET_BIT32


def ref_quiet_snan(w: int) -> int:
    """If w is sNaN, return quieted NaN; otherwise return w unchanged."""
    if is_snan32(w):
        return quiet_nan32(w)
    return w & MASK32


def is_nan16(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and bool(h & 0x3FF)


# Hand-worked test vectors for IEEE 754 NaN bitfields and quieting.
NAN_BITFIELD_SAMPLES: tuple[tuple[str, int, bool, bool, int, int], ...] = (
    # name, bits, is_qnan, is_snan, sign, payload
    ("pos_canonical_qnan", 0x7FC00000, True, False, 0, 0),
    ("neg_canonical_qnan", 0xFFC00000, True, False, 1, 0),
    ("pos_qnan_min_payload", 0x7FC00001, True, False, 0, 1),
    ("neg_qnan_min_payload", 0xFFC00001, True, False, 1, 1),
    ("pos_qnan_alt_bits1", 0x7FD55555, True, False, 0, 0x155555),
    ("neg_qnan_alt_bits2", 0xFFEAAAAA, True, False, 1, 0x2AAAAA),
    ("pos_qnan_max_payload", 0x7FFFFFFF, True, False, 0, PAYLOAD_MASK32),
    ("neg_qnan_max_payload", 0xFFFFFFFF, True, False, 1, PAYLOAD_MASK32),
    ("pos_snan_min_payload", 0x7F800001, False, True, 0, 1),
    ("neg_snan_min_payload", 0xFF800001, False, True, 1, 1),
    ("pos_snan_alt_bits1", 0x7F955555, False, True, 0, 0x155555),
    ("neg_snan_alt_bits2", 0xFFAAAAAA, False, True, 1, 0x2AAAAA),
    ("pos_snan_max_payload", 0x7FBFFFFF, False, True, 0, PAYLOAD_MASK32),
    ("neg_snan_max_payload", 0xFFBFFFFF, False, True, 1, PAYLOAD_MASK32),
    ("pos_snan_custom_tag", 0x7FBA5A5A, False, True, 0, 0x3A5A5A),
    ("neg_snan_custom_tag", 0xFFBA5A5A, False, True, 1, 0x3A5A5A),
)

NON_NAN_BOUNDARY_SAMPLES: tuple[tuple[str, int], ...] = (
    ("pos_zero", 0x00000000),
    ("neg_zero", 0x80000000),
    ("pos_subnorm_min", 0x00000001),
    ("neg_subnorm_min", 0x80000001),
    ("pos_norm_min", 0x00800000),
    ("pos_one", 0x3F800000),
    ("neg_one", 0xBF800000),
    ("pos_max_normal", 0x7F7FFFFF),
    ("neg_max_normal", 0xFF7FFFFF),
    ("pos_infinity", 0x7F800000),
    ("neg_infinity", 0xFF800000),
)


def translate_text(body: str) -> str:
    """SPIR-V text for a pixel shader whose instructions are ``body`` (one per line)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_fnan.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; ps_fnan\n" + "".join("\t" + ln.strip() + "\n" for ln in body.strip().splitlines()))
        with open(os.path.join(tmp, "ps_fnan.dxbc"), "wb") as f:
            f.write(support.STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


LOAD = """s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v0, 0
"""


def run_op(line: str, inputs: tuple[int, ...]) -> int:
    """Run ``line`` (writing v0, reading v1..v4 = inputs) and return the v0 word."""
    spv = translate_text(LOAD + line + "\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
    mem = {0x1000 + 4 * i: w for i, w in enumerate((list(inputs) + [0] * 4)[:4])}
    return support.mrt0_words(spv, push_constants=PUSH, memory=mem)[0]


class NanBitfieldPayloadOracleTests(unittest.TestCase):
    """Reference oracle tests for IEEE 754 NaN bitfield classification, payload preservation, and quieting."""

    def test_nan_classification_and_payload_extraction(self):
        for name, bits, want_qnan, want_snan, want_sign, want_payload in NAN_BITFIELD_SAMPLES:
            with self.subTest(sample=name, bits=f"{bits:#010x}"):
                self.assertTrue(is_nan32(bits))
                self.assertEqual(is_qnan32(bits), want_qnan)
                self.assertEqual(is_snan32(bits), want_snan)
                self.assertEqual(nan_sign32(bits), want_sign)
                self.assertEqual(nan_payload32(bits), want_payload)

    def test_non_nan_boundary_rejection(self):
        for name, bits in NON_NAN_BOUNDARY_SAMPLES:
            with self.subTest(sample=name, bits=f"{bits:#010x}"):
                self.assertFalse(is_nan32(bits))
                self.assertFalse(is_qnan32(bits))
                self.assertFalse(is_snan32(bits))

    def test_make_qnan_and_snan_constructors(self):
        for sign in (0, 1):
            for payload in (0, 1, 0x1234, 0x155555, 0x2AAAAA, PAYLOAD_MASK32):
                q = make_qnan32(sign, payload)
                self.assertTrue(is_qnan32(q))
                self.assertFalse(is_snan32(q))
                self.assertEqual(nan_sign32(q), sign)
                self.assertEqual(nan_payload32(q), payload)

                s_pay = payload if payload != 0 else 1
                s = make_snan32(sign, s_pay)
                self.assertTrue(is_snan32(s))
                self.assertFalse(is_qnan32(s))
                self.assertEqual(nan_sign32(s), sign)
                self.assertEqual(nan_payload32(s), s_pay)

    def test_snan_quieting_preserves_sign_and_payload(self):
        for name, bits, _, is_snan, want_sign, want_payload in NAN_BITFIELD_SAMPLES:
            if not is_snan:
                continue
            with self.subTest(sample=name, bits=f"{bits:#010x}"):
                quieted = quiet_nan32(bits)
                self.assertTrue(is_qnan32(quieted))
                self.assertFalse(is_snan32(quieted))
                self.assertEqual(nan_sign32(quieted), want_sign)
                self.assertEqual(nan_payload32(quieted), want_payload)
                self.assertEqual(quieted, bits | QUIET_BIT32)
                # Quieting an already quiet NaN must be idempotent.
                self.assertEqual(quiet_nan32(quieted), quieted)
                self.assertEqual(ref_quiet_snan(bits), quieted)
                self.assertEqual(ref_quiet_snan(quieted), quieted)

    def test_all_single_bit_payloads_roundtrip(self):
        for bit_idx in range(22):
            payload = 1 << bit_idx
            for sign in (0, 1):
                snan = make_snan32(sign, payload)
                qnan = make_qnan32(sign, payload)
                self.assertEqual(nan_payload32(snan), payload)
                self.assertEqual(nan_payload32(qnan), payload)
                self.assertEqual(quiet_nan32(snan), qnan)


class MinMaxNaN(unittest.TestCase):
    ROWS = [  # (x, y, min, max); None = a NaN
        (NAN_A, M25, M25, M25),
        (M25, NAN_B, M25, M25),
        (NAN_A, NAN_B, None, None),
        (PZ, NZ, NZ, PZ),
        (NZ, PZ, NZ, PZ),
    ]

    def check(self, got, want, what):
        if want is None:
            self.assertTrue(is_nan32(got), f"{what}: {got:#010x} is not a NaN")
        else:
            self.assertEqual(got, want, f"{what}: got {got:#010x}, want {want:#010x}")

    def test_v_min_v_max_f32(self):
        for x, y, mn, mx in self.ROWS:
            for form in ("e32", "e64"):
                with self.subTest(x=hex(x), y=hex(y), form=form):
                    self.check(run_op(f"v_min_f32_{form} v0, v1, v2", (x, y)), mn, f"min({x:#x}, {y:#x})")
                    self.check(run_op(f"v_max_f32_{form} v0, v1, v2", (x, y)), mx, f"max({x:#x}, {y:#x})")

    def test_v_min3_v_max3_f32(self):
        self.assertEqual(run_op("v_min3_f32 v0, v1, v2, v3", (NAN_A, M25, P35)), M25)
        self.assertEqual(run_op("v_max3_f32 v0, v1, v2, v3", (NAN_A, M25, P35)), P35)

    def test_v_min_v_max_f16(self):
        # The other half of the destination (v0 = 0) is kept.
        for x, y in ((0x7E11, 0xC100), (0xC100, 0xFE22)):
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op("v_min_f16_e32 v0, v1, v2", (x, y)), 0xC100)
                self.assertEqual(run_op("v_max_f16_e32 v0, v1, v2", (x, y)), 0xC100)

    def test_v_pk_min_v_pk_max_f16(self):
        # Low halves (0x7e11, 0xc100), high halves (0xc100, 0xfe22): both give 0xc100.
        x, y = 0xC1007E11, 0xFE22C100
        self.assertEqual(run_op("v_pk_min_f16 v0, v1, v2", (x, y)), 0xC100C100)
        self.assertEqual(run_op("v_pk_max_f16 v0, v1, v2", (x, y)), 0xC100C100)


class Med3NaN(unittest.TestCase):
    def test_v_med3_f32_rows(self):
        rows = [
            ((NAN_A, P70, P50), P50),
            ((NAN_A, P35, M25), M25),
            ((M25, NAN_B, P35), M25),
            ((P70, P50, NAN_A), P50),          # NaN in S2 (z): previously hit undefined FClamp
            ((P50, P70, NAN_B), P50),
            ((NAN_A, NAN_B, P50), P50),        # two NaNs in S0, S1: previously hit undefined FClamp
            ((NAN_A, P50, NAN_B), P50),
            ((P50, NAN_A, NAN_B), P50),
            ((NZ, PZ, NZ), NZ),
            ((ONE, P35, M25), ONE),            # no NaN: the median, unchanged
        ]
        for ins, want in rows:
            with self.subTest(ins=[hex(w) for w in ins]):
                self.assertEqual(run_op("v_med3_f32 v0, v1, v2, v3", ins), want)
        # All three NaN: must return a NaN without triggering undefined FClamp.
        self.assertTrue(is_nan32(run_op("v_med3_f32 v0, v1, v2, v3", (NAN_A, NAN_B, NAN_A))))
        spv = translate_text(LOAD + "v_med3_f32 v0, v1, v2, v3\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertNotIn("FClamp", spv)

    def test_v_med3_f32_with_inline_constants(self):
        # med3(0, NaN, 1.0) = 0 on hardware. Only the register operand can be a NaN, so only it is tested.
        self.assertEqual(run_op("v_med3_f32 v0, 0, v1, 1.0", (NAN_A,)), PZ)
        self.assertEqual(run_op("v_med3_f32 v0, 0, 1.0, v1", (NAN_A,)), PZ)
        spv = translate_text(LOAD + "v_med3_f32 v0, 0, v1, 1.0\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertEqual(spv.count("OpIsNan"), 1)

    def test_v_med3_f32_all_constants_needs_no_select(self):
        spv = translate_text("v_med3_f32 v0, 0, 1.0, 2.0\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertNotIn("OpIsNan", spv)
        self.assertNotIn("OpSelect %float", spv)
        self.assertEqual(support.mrt0_words(spv)[0], ONE)

    def test_v_med3_f32_nan_literal_is_checked(self):
        # A literal can be a NaN, so it keeps its OpIsNan (and med3 gives min3 = the smallest of the others).
        spv = translate_text(LOAD + "v_med3_f32 v0, 0x7fc00000, v1, v2\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertEqual(spv.count("OpIsNan"), 3)
        self.assertEqual(support.mrt0_words(spv, push_constants=PUSH, memory={0x1000: P70, 0x1004: P50})[0], P50)

    def test_v_med3_f16_follows_the_same_rule(self):
        # All 1-, 2-, and 3-NaN operand positions for v_med3_f16 (measured on gfx1035 wave32 & wave64).
        nan16, h5, h7 = 0x7E11, 0x4500, 0x4700
        for ins, want in (
            ((nan16, h7, h5), h5),
            ((h7, nan16, h5), h5),
            ((h7, h5, nan16), h5),
            ((nan16, nan16, h5), h5),
            ((nan16, h5, nan16), h5),
            ((h5, nan16, nan16), h5),
        ):
            with self.subTest(ins=[hex(w) for w in ins]):
                self.assertEqual(run_op("v_med3_f16 v0, v1, v2, v3", ins), want)
        self.assertTrue(is_nan16(run_op("v_med3_f16 v0, v1, v2, v3", (nan16, nan16, nan16)) & 0xFFFF))


class FractNearOne(unittest.TestCase):
    def test_v_fract_f32_rows(self):
        for x, want in ((0xAEDBE6FF, 0x3F7FFFFF), (0xB2800000, 0x3F7FFFFF), (NZ, PZ),
                        (0x3FC00000, 0x3F000000), (0xBF99999A, 0x3F4CCCCC)):
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op("v_fract_f32_e32 v0, v1", (x,)), want)

    def test_v_fract_f32_of_infinity_is_nan(self):
        for x in (0x7F800000, 0xFF800000):
            with self.subTest(x=hex(x)):
                self.assertTrue(is_nan32(run_op("v_fract_f32_e32 v0, v1", (x,))))

    def test_v_fract_f16_rows(self):
        for x in (0x8800, 0x8001, 0x9000):
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op("v_fract_f16_e32 v0, v1", (x,)), 0x3BFF)
        got = run_op("v_fract_f16_e32 v0, v1", (0x7C00,))
        self.assertTrue(is_nan16(got & 0xFFFF) and got >> 16 == 0, hex(got))

    def test_v_fract_f16_mul2_keeps_a_true_one(self):
        # On RDNA2 with denorms kept, mul:2 is ignored on f16: fract(0.5) remains 0.5 (0x3800).
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:2", (0x3800,)), 0x3800)

    def test_fract_alone_rounds_to_one(self):
        # Without the select, fract(-1e-10) is exactly 1.0: the select is what gives 0x3f7fffff.
        self.assertEqual(si.f32_bits(si._fract(si.bits_f32(0xAEDBE6FF))), ONE)


class GpuFixtures(unittest.TestCase):
    """The GPU fixtures with the heap words check.py seeds (the same words the .asm headers give)."""

    def test_ps_fnan_med3(self):
        heap = (NAN_A, P70, P50, NAN_A, P35, M25)
        spv = support.translate_fixture("ps_fnan_med3")
        words = support.mrt0_words(spv, push_constants=PUSH,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(heap)})
        self.assertEqual(words, [P50, M25, M25, P35])

    def test_ps_fnan_fract(self):
        heap = (0xAEDBE6FF, 0x00008800, NAN_A, M25)
        spv = support.translate_fixture("ps_fnan_fract")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 16, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(heap)})
        self.assertEqual(words, [0x3F7FFFFF, 0x00003BFF, M25, M25])

    def test_ps_fnan_med3_s2(self):
        # med3(5, 7, NaN), med3(NaN, NaN, 5), med3(NaN, -2.5, NaN), med3(7, NaN, NaN): min3 with the NaNs ignored
        heap = (P50, P70, NAN_A, M25)
        spv = support.translate_fixture("ps_fnan_med3_s2")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 16, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(heap)})
        self.assertEqual(words, [P50, P50, M25, P70])

    def test_ps_f16_fnan_med3(self):
        # the same four rows as f16 (5.0, 7.0, NaN, -2.5); the high half of each destination is kept
        heap = (0x00004500, 0x00004700, 0x00007E01, 0x0000C100)
        spv = support.translate_fixture("ps_f16_fnan_med3")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 16, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(heap)})
        self.assertEqual(words, [0xAAAA4500, 0xAAAA4500, 0xAAAAC100, 0xAAAA4700])


class AllNaNSrc0PayloadTests(unittest.TestCase):
    """All-NaN operands in f16 and 3-input min/max/med3 preserve src0's NaN bits (including sNaN under IEEE=0)."""

    @staticmethod
    def _run_with_old(line: str, inputs: tuple[int, ...], old: int = 0xDEADBEEF) -> int:
        spv = translate_text(LOAD + f"v_mov_b32_e32 v0, {old:#x}\n" + line + "\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        mem = {0x1000 + 4 * i: w for i, w in enumerate((list(inputs) + [0] * 4)[:4])}
        return support.mrt0_words(spv, push_constants=PUSH, memory=mem)[0]

    def test_f16_all_nan_keeps_src0_and_mods(self):
        snan, qnan_a, qnan_b = 0x7C01, 0x7E02, 0xFE22
        for op in ("v_min_f16_e32", "v_max_f16_e32"):
            with self.subTest(op=op):
                self.assertEqual(self._run_with_old(f"{op} v0, v1, v2", (0xAAAA0000 | snan, 0xBBBB0000 | qnan_a)),
                                 0xDEAD0000 | snan)
                self.assertEqual(self._run_with_old(f"{op} v0, v1, v2", (0xAAAA0000 | qnan_b, 0xBBBB0000 | snan)),
                                 0xDEAD0000 | qnan_b)
        self.assertEqual(
            self._run_with_old("v_min_f16_e64 v0, -v1, v2 op_sel:[1,0,1]", ((snan << 16) | 0x3C00, qnan_a)),
            ((snan ^ 0x8000) << 16) | 0xBEEF,
        )
        self.assertEqual(
            self._run_with_old("v_max_f16_e64 v0, |v1|, v2", (snan | 0x8000, qnan_a)),
            0xDEAD0000 | snan,
        )
        self.assertEqual(
            self._run_with_old(
                "v_min_f16_sdwa v0, -v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0",
                ((snan << 16) | 0x1111, 0x22220000 | qnan_a),
            ),
            ((snan ^ 0x8000) << 16) | 0xBEEF,
        )
        for op in ("v_min3_f16", "v_max3_f16", "v_med3_f16"):
            with self.subTest(op=op):
                self.assertEqual(self._run_with_old(f"{op} v0, v1, v2, v3", (snan, qnan_a, qnan_b)), 0xDEAD0000 | snan)
                self.assertEqual(
                    self._run_with_old(f"{op} v0, -|v1|, v2, v3 op_sel:[1,0,0,1]", ((snan << 16), qnan_a, qnan_b)),
                    ((snan ^ 0x8000) << 16) | 0xBEEF,
                )
                self.assertEqual(
                    self._run_with_old(f"{op} v0, v1, v2, v3", (snan, qnan_a, 0x3C00)),
                    0xDEAD3C00,
                )
                self.assertEqual(
                    self._run_with_old(f"{op} v0, 0x7c01, v1, v2", (qnan_a, qnan_b)),
                    0xDEAD7C01,
                )

    def test_pk_minmax_f16_all_nan_per_lane(self):
        # Lane 0 all-NaN keeps src0 sNaN (0x7C01), lane 1 picks finite -2.5 (0xC100) over qNaN.
        for op in ("v_pk_min_f16", "v_pk_max_f16"):
            with self.subTest(op=op):
                self.assertEqual(run_op(f"{op} v0, v1, v2", (0xFE117C01, 0xC1007E02)), 0xC1007C01)
                self.assertEqual(run_op(f"{op} v0, v1, v2", (0xFC053C00, 0x7E007C01)), 0xFC053C00)
        self.assertEqual(
            run_op("v_pk_min_f16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[0,1]",
                   (0x7C01FE22, 0x7E037E04)),
            ((0xFE22) << 16) | (0x7C01 ^ 0x8000),
        )
        # NaN literal in high half + non-NaN in low half: high preserves 0x7C01, low keeps 0x3C00.
        self.assertEqual(run_op("v_pk_min_f16 v0, 0x7c013c00, v1", (0x7E027E03,)), 0x7C013C00)

    def test_f32_three_input_all_nan_keeps_src0(self):
        snan_a, snan_b = 0x7F800001, 0xFF800005
        for op in ("v_min3_f32", "v_max3_f32", "v_med3_f32"):
            with self.subTest(op=op):
                self.assertEqual(run_op(f"{op} v0, v1, v2, v3", (snan_a, NAN_A, NAN_B)), snan_a)
                self.assertEqual(run_op(f"{op} v0, v1, v2, v3", (NAN_A, NAN_B, snan_b)), NAN_A)
                self.assertEqual(run_op(f"{op} v0, -v1, v2, v3", (snan_a, NAN_A, NAN_B)), snan_a ^ SIGN_BIT32)
                for args in ((snan_a, NAN_B, M25), (snan_a, M25, NAN_B), (M25, snan_a, NAN_B)):
                    self.assertEqual(run_op(f"{op} v0, v1, v2, v3", args), M25)

    def test_f32_two_input_minmax_modifiers_keep_signaling_nan(self):
        snan = 0x7F800001
        for op in ("v_min_f32_e64", "v_max_f32_e64"):
            with self.subTest(op=op):
                self.assertEqual(run_op(f"{op} v0, -v1, v2", (snan, NAN_A)), snan ^ SIGN_BIT32)
                self.assertEqual(run_op(f"{op} v0, |v1|, v2", (snan, NAN_A)), snan)

    def test_non_nan_constant_fast_path(self):
        # f16_bits emits 1 OpUGreaterThan (%float |x| > inf); all-NaN tracking adds 0 when a constant is non-NaN.
        for line, reg_line, extra_gt in (
            ("v_min_f16_e32 v0, v1, 1.0", "v_min_f16_e32 v0, v1, v2", 2),
            ("v_min3_f16 v0, v1, v2, 1.0", "v_min3_f16 v0, v1, v2, v3", 3),
        ):
            spv_const = translate_text(LOAD + line + "\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
            spv_reg = translate_text(LOAD + reg_line + "\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
            self.assertEqual(spv_const.count("OpUGreaterThan "), 1, line)
            self.assertEqual(spv_reg.count("OpUGreaterThan "), 1 + extra_gt, reg_line)
        # v_pk_min_f16 emits 2 f16_bits conversions -> 2 OpUGreaterThan when fast-pathed vs 6 with 2 registers x 2 lanes.
        self.assertEqual(
            translate_text(LOAD + "v_pk_min_f16 v0, v1, 1.0\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm").count("OpUGreaterThan "),
            2,
        )
        self.assertEqual(
            translate_text(LOAD + "v_pk_min_f16 v0, v1, v2\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm").count("OpUGreaterThan "),
            6,
        )
        spv_min3_f32 = translate_text(LOAD + "v_min3_f32 v0, v1, v2, 1.0\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm")
        self.assertEqual(spv_min3_f32.count("OpIsNan"), 1)

    def test_spirv_val_all_nan_variants(self):
        import shutil
        import subprocess
        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if not (spirv_as and spirv_val):
            self.skipTest("spirv-as / spirv-val not found")
        spv = translate_text(LOAD + "\n".join([
            "v_min_f16_e32 v0, v1, v2",
            "v_min3_f16 v0, -|v1|, v2, v3 op_sel:[1,0,0,1]",
            "v_med3_f16 v0, v1, v2, v3",
            "v_pk_max_f16 v0, v1, v2 neg_lo:[1,0] op_sel:[0,1]",
            "v_min3_f32 v0, v1, v2, v3",
            "v_med3_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]))
        bin_spv = subprocess.run([spirv_as, "--target-env", "vulkan1.2", "-o", "-", "-"],
                                 input=spv.encode("utf-8"), check=True, stdout=subprocess.PIPE).stdout
        subprocess.run([spirv_val, "--target-env", "vulkan1.2", "-"], input=bin_spv, check=True)


class InterpreterOps(unittest.TestCase):
    """NMin/NMax, OpIsNan and Fract in spirv_mini_interp, against GLSL.std.450."""

    def test_nmin_nmax(self):
        nan = float("nan")
        self.assertEqual(si._nmin_nmax("NMin", nan, -2.5), -2.5)
        self.assertEqual(si._nmin_nmax("NMax", -2.5, nan), -2.5)
        self.assertTrue(math.isnan(si._nmin_nmax("NMin", nan, nan)))
        self.assertEqual(math.copysign(1, si._nmin_nmax("NMin", 0.0, -0.0)), -1)
        self.assertEqual(math.copysign(1, si._nmin_nmax("NMax", -0.0, 0.0)), 1)

    def test_fract_exact(self):
        self.assertEqual(si._fract(-0.75), 0.25)
        self.assertEqual(math.copysign(1, si._fract(-0.0)), 1)
        self.assertTrue(math.isnan(si._fract(float("inf"))))
        self.assertEqual(si._fract(si.bits_f32(0x80000001)), 1.0)      # 1 - 2**-149 rounds to 1.0

    def test_fmin_of_nan_still_raises(self):
        body = ("OpCapability Shader\n%glsl = OpExtInstImport \"GLSL.std.450\"\nOpMemoryModel Logical GLSL450\n"
                "OpEntryPoint Fragment %main \"main\" %out\n%void = OpTypeVoid\n%fn = OpTypeFunction %void\n"
                "%uint = OpTypeInt 32 0\n%float = OpTypeFloat 32\n%_ptr_Output_uint = OpTypePointer Output %uint\n"
                "%out = OpVariable %_ptr_Output_uint Output\n%n = OpConstant %uint 2143289344\n"
                "%main = OpFunction %void None %fn\n%entry = OpLabel\n%x = OpBitcast %float %n\n"
                "%r = OpExtInst %float %glsl FMin %x %x\n%u = OpBitcast %uint %r\nOpStore %out %u\nOpReturn\n"
                "OpFunctionEnd\n")
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(body)


if __name__ == "__main__":
    unittest.main()
