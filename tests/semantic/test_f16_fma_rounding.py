"""Semantic tests for single-rounded f16 FMAs (SHADER-55).

RDNA2's v_fma_f16, v_fmac_f16 and v_pk_fma_f16 round the exact a*b+c once, to f16 (an RX 6700 matched a correctly
rounded f16 FMA in all 12,612 measured cases, #374). The translator used GLSL Fma (rounded to f32) and then narrowed
to f16, which double-rounds. These tests compare the translated ops with an exact rational reference on random
triples, plus triples built so that the exact result sits just off an f16 halfway point: those are the cases
where f32-then-f16 rounding goes the wrong way. Results are exported with a zero high half (#509).
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def round_f16(x):
    """Exact rational -> f16 bits, round to nearest even (no NaN inputs here)."""
    if x == 0:
        return 0
    sign = 0x8000 if x < 0 else 0
    x = abs(x)
    lo, hi = 0, 0x7C00                     # positive f16 values are ordered like their bits; 0x7C00 = inf
    while lo < hi:                         # smallest h with value(h) >= x
        mid = (lo + hi) // 2
        if (Fraction(h2f(mid)) if mid < 0x7C00 else Fraction(10 ** 9)) >= x:
            hi = mid
        else:
            lo = mid + 1
    if lo == 0x7C00:                       # beyond 65504: round to 65504 or inf by the 65520 boundary
        return sign | (0x7C00 if x >= 65520 else 0x7BFF)
    above = lo
    if lo == 0 or Fraction(h2f(above)) == x:
        return sign | above
    below = lo - 1
    da, db = Fraction(h2f(above)) - x, x - Fraction(h2f(below))
    if above == 0x7C00:
        return sign | (0x7C00 if x >= 65520 else 0x7BFF)
    pick = above if da < db else below if db < da else (above if above % 2 == 0 else below)
    return sign | pick


def ref_fma(a, b, c):
    return round_f16(Fraction(h2f(a)) * Fraction(h2f(b)) + Fraction(h2f(c)))


def double_rounded(a, b, c):
    """What f32 Fma then f16 gives (the old translation)."""
    x = Fraction(h2f(a)) * Fraction(h2f(b)) + Fraction(h2f(c))
    f = struct.unpack("<f", struct.pack("<f", float(x)))[0]   # float(x) is correctly rounded to double first
    return round_f16(Fraction(f))


_TMP = None


def run(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_f16fma.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    path = os.path.join(_TMP.name, "disasm", "ps_f16fma.s")
    with open(path, "w", encoding="utf-8") as f:
        f.write("; ps_f16fma\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    return list(si.run_fragment(spv)["%out_mrt0"])


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def finite_f16(rng, lo_exp=1, hi_exp=30):
    return rng.getrandbits(1) << 15 | rng.randint(lo_exp, hi_exp) << 10 | rng.getrandbits(10)


def f16_bits_of(x):
    """Exact f16 bits of a rational that is representable (normal or subnormal)."""
    h = round_f16(x)
    assert Fraction(h2f(h)) == x, x
    return h


def near_tie_triples(rng, n):
    """a*b lands exactly on an f16 halfway point and c is a nudge smaller than half an f32 ulp of it.

    The exact a*b + c is just off the tie, so the correct f16 result goes towards c. Rounding to f32 first loses c,
    leaves an exact tie, and tie-to-even then picks a side that is wrong about half the time. Construction: the tie
    between two adjacent f16 values has a 12-bit odd significand M; when 3 divides M, a = M/3 (at most 11 bits) and
    b = 3.0 give it exactly.
    """
    out = []
    while len(out) < n:
        h = rng.randint(0x4400, 0x7AFF) & ~0          # f16 values from 4.0 up, so an f32 ulp of a*b is >= 2^-21
        mid = (Fraction(h2f(h)) + Fraction(h2f(h + 1))) / 2
        e = 0
        m = mid
        while m.denominator != 1 or m >= 4096:       # mid = M * 2^e with M an odd 12-bit integer
            if m.denominator != 1:
                m *= 2
                e -= 1
            else:
                m /= 2
                e += 1
        m = int(m)
        if m % 3 or m < 2048:
            continue
        a = f16_bits_of(Fraction(m // 3) * Fraction(2) ** e)
        b = 0x4200                                    # 3.0
        k = rng.randint(1, 6)                         # nudge: min subnormal .. 2^-19, all below half an f32 ulp
        c = (1 << (k - 1)) | (0x8000 if rng.getrandbits(1) else 0)
        if rng.getrandbits(1):
            a ^= 0x8000
        out.append((a, b, c))
    return out


class F16FmaRoundingTests(unittest.TestCase):
    def check_ops(self, triples, op):
        for i in range(0, len(triples), 4):
            chunk = (triples[i:i + 4] + [(0, 0, 0)] * 4)[:4]
            lines = []
            for k, (a, b, c) in enumerate(chunk):
                lines += [f"v_mov_b32 v{4 + k}, {a:#x}", f"v_mov_b32 v{8 + k}, {b:#x}", f"v_mov_b32 v{12 + k}, {c:#x}"]
                if op == "v_fma_f16":
                    lines += [f"v_mov_b32 v{k}, 0", f"v_fma_f16 v{k}, v{4 + k}, v{8 + k}, v{12 + k}"]
                elif op == "v_fmac_f16":
                    lines += [f"v_mov_b32 v{k}, v{12 + k}", f"v_fmac_f16 v{k}, v{4 + k}, v{8 + k}",
                              f"v_and_b32 v{k}, 0xffff, v{k}"]
                else:   # v_pk_fma_f16 in the high half, then moved down: checks the second lane of the packed op
                    lines += [f"v_lshlrev_b32 v{4 + k}, 16, v{4 + k}", f"v_lshlrev_b32 v{8 + k}, 16, v{8 + k}",
                              f"v_lshlrev_b32 v{12 + k}, 16, v{12 + k}",
                              f"v_pk_fma_f16 v{k}, v{4 + k}, v{8 + k}, v{12 + k}", f"v_lshrrev_b32 v{k}, 16, v{k}"]
            got = run(lines)
            for (a, b, c), g in zip(chunk, got):
                want = ref_fma(a, b, c)
                with self.subTest(op=op, a=hex(a), b=hex(b), c=hex(c)):
                    self.assertEqual(g, want, f"{op}: got {g:#06x}, want {want:#06x}")

    def test_reference_sees_double_rounding(self):
        triples = near_tie_triples(random.Random(55), 300)
        diff = [t for t in triples if ref_fma(*t) != double_rounded(*t)]
        self.assertGreater(len(diff), 5, "the near-tie generator must produce double-rounding cases")

    def test_fma_near_ties(self):
        triples = near_tie_triples(random.Random(55), 300)
        diff = [t for t in triples if ref_fma(*t) != double_rounded(*t)]
        self.check_ops(diff[:40] + triples[:40], "v_fma_f16")

    def test_fmac_and_pk_fma_near_ties(self):
        triples = near_tie_triples(random.Random(56), 300)
        diff = [t for t in triples if ref_fma(*t) != double_rounded(*t)]
        self.check_ops(diff[:20], "v_fmac_f16")
        self.check_ops(diff[:20], "v_pk_fma_f16")

    def test_fma_random(self):
        rng = random.Random(0x55)
        triples = [(finite_f16(rng, 1, 22), finite_f16(rng, 1, 22), finite_f16(rng, 1, 26)) for _ in range(120)]
        triples += [(0x0001, 0x3C00, 0x8000), (0x3C00, 0x3C00, 0xBC00), (0x7BFF, 0x4000, 0x0000), (0x0400, 0x0400, 0x0001)]
        self.check_ops(triples, "v_fma_f16")

    def test_fmac_output_modifiers(self):
        # LLVM gfx10_asm_vop2.s encodes clamp, mul:2, mul:4 and div:2 on v_fmac_f16_e64.
        # On RDNA2 with denormals kept, omod is ignored on f16 results; clamp applies.
        # High half (0xAAAA) of destination VGPR must remain untouched.
        got = run([
            "v_mov_b32 v0, 0xaaaa3c00",           # acc = 1.0h, hi = 0xAAAA
            "v_fmac_f16_e64 v0, 1.0, 1.0 clamp",  # 1*1 + 1 = 2.0 -> clamp -> 1.0 (0x3c00)
            "v_mov_b32 v1, 0xaaaa3800",           # acc = 0.5h
            "v_fmac_f16_e64 v1, 0.5, 0.5 mul:2",  # (0.25 + 0.5) = 0.75 (0x3a00) (mul:2 ignored)
            "v_mov_b32 v2, 0xaaaa3400",           # acc = 0.25h
            "v_fmac_f16_e64 v2, 0.5, 0.5 mul:4",  # (0.25 + 0.25) = 0.5 (0x3800) (mul:4 ignored)
            "v_mov_b32 v3, 0xaaaa4000",           # acc = 2.0h
            "v_fmac_f16_e64 v3, 1.0, 2.0 div:2",  # (2 + 2) = 4.0 (0x4400) (div:2 ignored)
        ])
        self.assertEqual(got, [0xAAAA3C00, 0xAAAA3A00, 0xAAAA3800, 0xAAAA4400])

        # Clamp applies, negative values clamp to 0.0, and omod is ignored.
        got2 = run([
            "v_mov_b32 v0, 0xaaaa3400",                  # acc = 0.25h
            "v_fmac_f16_e64 v0, 0.5, 0.5 mul:2 clamp",   # (0.25 + 0.25) = 0.5 -> clamp -> 0.5 (0x3800)
            "v_mov_b32 v1, 0xaaaabc00",                  # acc = -1.0h
            "v_fmac_f16_e64 v1, 0.5, 0.5 clamp",         # (0.25 - 1.0) = -0.75 -> clamp -> 0.0 (0x0000)
            "v_mov_b32 v2, 0xaaaa3400",                  # acc = 0.25h
            "v_fmac_f16_e64 v2, 0.5, 0.5 mul:4 clamp",   # (0.25 + 0.25) = 0.5 -> clamp -> 0.5 (0x3800)
            "v_mov_b32 v3, 0xaaaa3800",                  # acc = 0.5h
            "v_fmac_f16_e64 v3, 1.0, 0.5 div:2 clamp",   # (0.5 + 0.5) = 1.0 -> clamp -> 1.0 (0x3c00)
        ])
        self.assertEqual(got2, [0xAAAA3800, 0xAAAA0000, 0xAAAA3800, 0xAAAA3C00])


if __name__ == "__main__":
    unittest.main()
