"""f16 add/sub narrowing without the subnormal rounding (Gen.f16_bits exact_tiny).

The sum of two f16 values is a multiple of 2^-24, so below 2^-14 it already is an f16 value and needs no rounding.
These tests run v_add_f16, v_sub_f16, v_subrev_f16 and v_pk_add_f16 through the translator and the mini interpreter
and compare with the exact sum, rounded once, over the range boundaries and a seeded sweep. An omod can break that
property (div:2), so the last test checks that such a result is still rounded.
"""
import random
import struct
import unittest
from fractions import Fraction

import support


def value(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def reference(x, y):
    """f16 bits of x + y rounded once to nearest even; None for a NaN."""
    total = x + y
    if total != total:
        return None
    if total in (float("inf"), float("-inf")) or total == 0:
        return struct.unpack("<H", struct.pack("<e", total))[0]
    exact = Fraction(x) + Fraction(y)
    assert Fraction(total) == exact     # two f16 values always add exactly in binary64
    try:
        return struct.unpack("<H", struct.pack("<e", total))[0]
    except OverflowError:
        return 0xFC00 if total < 0 else 0x7C00


def run(op, pairs, mods=""):
    lines = []
    for i, (a, b) in enumerate(pairs):
        lines += [f"v_mov_b32 v{i}, 0", f"v_mov_b32 v{4 + i}, {a:#x}", f"v_mov_b32 v{8 + i}, {b:#x}",
                  f"{op} v{i}, v{4 + i}, v{8 + i}{mods}", f"v_and_b32 v{i}, 0xffff, v{i}"]
    lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_f16sum", lines))


EDGE = [0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x01FF, 0x0200, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x0401,
        0x3C00, 0xBC00, 0x3C01, 0x7BFE, 0x7BFF, 0xFBFF, 0x5000, 0x7C00]


def cases():
    rng = random.Random(0xF165)
    out = [(a, b) for a in EDGE for b in EDGE]
    for _ in range(200):
        out.append((rng.randrange(0x0C00) | rng.getrandbits(1) << 15,       # around and below 2^-14
                    rng.randrange(0x0C00) | rng.getrandbits(1) << 15))
        out.append((rng.randrange(0x7C00) | rng.getrandbits(1) << 15,       # any finite value
                    rng.randrange(0x7C00) | rng.getrandbits(1) << 15))
    return out


class F16SumNarrowingTests(unittest.TestCase):
    def check(self, op, combine):
        wanted = [(a, b, reference(*combine(value(a), value(b)))) for a, b in cases()]
        wanted = [c for c in wanted if c[2] is not None]
        wanted += [(0, 0, 0)] * (-len(wanted) % 4)
        for i in range(0, len(wanted), 4):
            chunk = wanted[i:i + 4]
            got = run(op, [c[:2] for c in chunk])
            for (a, b, want), g in zip(chunk, got):
                self.assertEqual(g, want, f"{op}({a:#06x}, {b:#06x}) = {g:#06x}, want {want:#06x}")

    def test_add(self):
        self.check("v_add_f16", lambda x, y: (x, y))

    def test_sub(self):
        self.check("v_sub_f16", lambda x, y: (x, -y))

    def test_subrev(self):
        self.check("v_subrev_f16", lambda x, y: (y, -x))

    def test_pk_add(self):
        self.check("v_pk_add_f16", lambda x, y: (x, y))

    def test_omod_result_is_still_rounded(self):
        # (1 + 2) * 2^-24 with div:2 is a tie that rounds to 2; 3 if the omod is ignored because denormals are kept.
        # Truncating without rounding would give 1.
        for got in run("v_add_f16_e64", [(1, 2)] * 4, " div:2"):
            self.assertIn(got, (2, 3))


if __name__ == "__main__":
    unittest.main()
