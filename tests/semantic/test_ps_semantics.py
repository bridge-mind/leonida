"""Value-level regression tests for the RDNA2 -> SPIR-V translator on six straight-line pixel shaders.

Each shader ends with `exp mrt0 v.., v.., v.., v..`, i.e. exactly four 32-bit words. Six shaders x four words =
the 24 values checked here. Every word is asserted individually (see the subTests), so no output component
is left unchecked, and the output variable set is asserted to be exactly {mrt0}.

The expected words are written out as constants AND recomputed by plain-Python reference functions that
implement the ISA semantics below; a separate test checks the two agree. Neither depends on the SPIR-V
interpreter or on the translator.

ISA semantics relied on (AMD RDNA2 "Instruction Set Architecture" reference, VALU/SDWA chapters):
  SDWA source select  BYTE_n / WORD_n: take that field of the 32-bit source, zero-extended; sext(x) sign-extends it.
  SDWA dest select    dst_sel BYTE_n / WORD_n: the 32-bit result is truncated to that width and written there;
                      dst_unused UNUSED_PRESERVE keeps the other destination bits, UNUSED_PAD zeroes them,
                      UNUSED_SEXT sign-extends the written field into the bits above it and zeroes those below.
  v_subrev_nc_u32     D = S1 - S0.        v_pk_fma_f16   per 16-bit half: D = S0 * S1 + S2 (IEEE half).
  v_min3_u32 / v_max3_i32   minimum (unsigned) / maximum (signed) of three sources.
  v_bfe_i32           D = sign_extend(S0[S1[4:0] +: S2[4:0]])  (bit-field extract, width in bits).
  v_fma_f16 op_sel    reads chosen 16-bit half of each source; writes chosen dst half while preserving the other.
  v_subrev_co_ci_u32  D = S1 - S0 - borrow_in (where borrow_in = lane bit of carry-in mask).
  v_cmp/cmpx_eq_u64   64-bit equality compare across register pairs; v_cmpx updates EXEC (active lane mask).
  v_mul_i32_i24       sign-extends low 24 bits of S0 and S1, returns low 32 bits of signed product.
  v_alignbyte_b32     (({S0, S1} as 64-bit) >> (8 * (S2 & 3))) low 32 bits.
  v_xnor_b32          ~(S0 ^ S1).         v_swap_b32  exchanges two VGPRs.
  v_ffbh_i32          MSB-first index (0..31) of first bit differing from the sign bit, or -1 if all equal.
  v_pk_lshlrev/lshrrev_b16  per-16-bit-half shift by the matching half of S0 (& 15).
  v_add_nc_u16        16-bit add into low half of dst, preserving high half on GFX10.
  v_sin_f16           sin(2*pi * f16(S0)) in turns, preserving high half of dst on GFX10.
  v_mul_legacy_f32    IEEE f32 multiply except 0 * anything (including inf and NaN) == +0.0.

Expectations were also recorded as GPU results in the header comments of runtime/test/shaders/ps_*.asm
(not re-confirmed on GPU here).
"""
import math
import struct
import unittest

import support

M = 0xFFFFFFFF
FINITE_EXPORT_XOR = 0x40000000

# Independent constants, written down by hand (see derivations in the reference functions below).
EXPECTED_PS_SDWA = [0x000000F1, 0xBFFEFFF2, 0x3C003344, 0xBFF70000]
EXPECTED_PS_OPS = [0x48804300, 0x00000005, 0x00000007, 0xBFFFFFFC]
EXPECTED_PS_OPS2 = [0xABCD4400, 0x46001234, 0xBFFFFFFA, 0x00000007]
EXPECTED_PS_OPS3 = [0xBFFFFFFA, 0x44660066, 0x44AABBCC, 0x00000010]
EXPECTED_PS_OPS4 = [0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00]
EXPECTED_PS_LEGACY_MUL = [0x00000000, 0x00000000, 0x00000000, 0x40C00000]


def _half(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _unhalf(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def _s32(x):
    return x - (1 << 32) if x >> 31 else x


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & M))[0]


def reference_ps_sdwa():
    v0, v1 = 0x8081F0F7, 0x00010005
    r0 = ((v0 >> 8) & 0xFF) + ((v1 >> 16) & 0xFFFF)                  # BYTE_1 of v0 + WORD_1 of v1 = 0xf0 + 1
    byte0 = v0 & 0xFF
    sext_byte0 = byte0 - 0x100 if byte0 & 0x80 else byte0            # sext(v0) with BYTE_0 = -9
    r1 = (sext_byte0 - v1) & M                                       # subrev: S1 - S0 = -9 - 0x10005
    r2 = (_half(1.0) << 16) | (0x11223344 & 0xFFFF)                  # WORD_1 <- f16(1.0), low half preserved
    field = (v0 & 0xFF) | (v1 & 0xFF)                                # BYTE_0 | BYTE_0 = 0xf7
    r3 = (field << 16) | (0xFF000000 if field & 0x80 else 0)         # BYTE_2, UNUSED_SEXT: sign above, zero below
    return [r0, r1 ^ FINITE_EXPORT_XOR, r2, r3 ^ FINITE_EXPORT_XOR]


def reference_ps_ops():
    lo = _unhalf(0x3C00) * _unhalf(0x4200) + _unhalf(0x3800)         # 1.0 * 3.0 + 0.5
    hi = _unhalf(0x4000) * _unhalf(0x4400) + _unhalf(0x3C00)         # 2.0 * 4.0 + 1.0
    r0 = (_half(hi) << 16) | _half(lo)
    r1 = min(7, 0xFFFFFFF0, 5)                                       # v_min3_u32 (unsigned)
    r2 = max(7, _s32(0xFFFFFFF0), 5) & M                             # v_max3_i32 (signed)
    f = (0xFFFFFFF0 >> 2) & 0xF                                      # bits [5:2]
    r3 = (f - 16 if f & 8 else f) & M                                # v_bfe_i32 sign extension of a 4-bit field
    return [r0, r1, r2, r3 ^ FINITE_EXPORT_XOR]


def reference_ps_ops2():
    v0, v1 = 0x40003C00, 0x44004200                                  # (1.0, 2.0) and (3.0, 4.0)
    r0 = (0xABCD0000 & 0xFFFF0000) | _half(_unhalf(v0 & 0xFFFF) * _unhalf(v1 >> 16) + 0.0)   # op_sel:[0,1,0,0]
    r1 = (_half(_unhalf(v0 >> 16) * _unhalf(v1 & 0xFFFF) + 0.0) << 16) | (0x00001234 & 0xFFFF)  # op_sel:[1,0,0,1]
    r2 = (5 - 10 - 1) & M                                            # v_subrev_co_ci_u32: v4 - 10 - borrow(1)
    r3 = 7 if (0 == ((0 << 32) | 0)) else 0                          # v_cmp_eq_u64 -> v_cndmask_b32
    return [r0, r1, r2 ^ FINITE_EXPORT_XOR, r3]


def reference_ps_ops3():
    s24 = 0x00FFFFFE - (1 << 24)                                     # signed 24-bit -2
    r0 = (s24 * 3) & M                                               # v_mul_i32_i24
    v3, v4 = 0x11223344, 0xAABBCCDD
    v5 = (((v3 << 32) | v4) >> 8) & M                                # v_alignbyte_b32(..., 1)
    v6 = (~(v3 ^ v4)) & M                                            # v_xnor_b32
    v5, v6 = v6, v5                                                  # v_swap_b32
    v7 = 0x0000FFFF
    sign = (v7 >> 31) & 1
    msb_diff = next((bit for bit in range(30, -1, -1) if ((v7 >> bit) & 1) != sign), None)
    r3 = M if msb_diff is None else (31 - msb_diff)                  # v_ffbh_i32
    return [r0 ^ FINITE_EXPORT_XOR, v5, v6, r3]


def reference_ps_ops4():
    v0, v2 = 0x12348001, 0x00080004
    lo0, hi0 = v0 & 0xFFFF, (v0 >> 16) & 0xFFFF
    sh_lo, sh_hi = v2 & 0xF, (v2 >> 16) & 0xF
    r0 = (((hi0 << sh_hi) & 0xFFFF) << 16) | ((lo0 << sh_lo) & 0xFFFF)  # v_pk_lshlrev_b16
    r1 = ((hi0 >> sh_hi) << 16) | (lo0 >> sh_lo)                        # v_pk_lshrrev_b16
    r2 = (0xBEEF0000 & 0xFFFF0000) | ((0x0001 + 0xFFFF) & 0xFFFF)       # v_add_nc_u16 keeps high half
    r3 = (0x12340000 & 0xFFFF0000) | _half(math.sin(_unhalf(0x3400) * 2.0 * math.pi))  # v_sin_f16 (turns)
    return [r0, r1, r2, r3]


def reference_ps_legacy_mul():
    def legacy(a_bits, b_bits):
        a, b = _bits_f32(a_bits), _bits_f32(b_bits)
        if (not math.isnan(a) and a == 0.0) or (not math.isnan(b) and b == 0.0):
            return 0x00000000
        return _f32_bits(a * b)

    return [
        legacy(0x00000000, 0x7F800000),   # +0.0 * +inf
        legacy(0x00000000, 0x7FC00000),   # +0.0 * qNaN
        legacy(0x80000000, 0xFF800000),   # -0.0 * -inf
        legacy(_f32_bits(2.0), _f32_bits(3.0)),  # 2.0 * 3.0
    ]


class OracleTests(unittest.TestCase):
    """The hand-written constants and the plain-Python reference agree (so the oracle is self-consistent)."""

    def test_reference_matches_constants(self):
        self.assertEqual(reference_ps_sdwa(), EXPECTED_PS_SDWA)
        self.assertEqual(reference_ps_ops(), EXPECTED_PS_OPS)
        self.assertEqual(reference_ps_ops2(), EXPECTED_PS_OPS2)
        self.assertEqual(reference_ps_ops3(), EXPECTED_PS_OPS3)
        self.assertEqual(reference_ps_ops4(), EXPECTED_PS_OPS4)
        self.assertEqual(reference_ps_legacy_mul(), EXPECTED_PS_LEGACY_MUL)


class TranslatorSemantics(unittest.TestCase):
    def check(self, name, expected):
        words = support.mrt0_words(support.translate_fixture(name))
        self.assertEqual(len(words), 4, "mrt0 has four components")
        for i, (got, want) in enumerate(zip(words, expected)):
            with self.subTest(shader=name, component="xyzw"[i]):
                self.assertEqual(f"{got:#010x}", f"{want:#010x}")

    def test_ps_sdwa(self):
        self.check("ps_sdwa", EXPECTED_PS_SDWA)

    def test_ps_ops(self):
        self.check("ps_ops", EXPECTED_PS_OPS)

    def test_ps_ops2(self):
        self.check("ps_ops2", EXPECTED_PS_OPS2)

    def test_ps_ops3(self):
        self.check("ps_ops3", EXPECTED_PS_OPS3)

    def test_ps_ops4(self):
        self.check("ps_ops4", EXPECTED_PS_OPS4)

    def test_ps_legacy_mul(self):
        self.check("ps_legacy_mul", EXPECTED_PS_LEGACY_MUL)


if __name__ == "__main__":
    unittest.main()
