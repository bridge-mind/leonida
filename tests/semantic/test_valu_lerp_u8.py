"""v_lerp_u8 (RDNA2 VOP3): per byte i, D.b[i] = (S0.b[i] + S1.b[i] + S2.b[i][0]) >> 1: the unsigned average of the
two bytes, rounded up when bit 0 of that byte of S2 is set and down otherwise; bits 7:1 of S2's bytes are ignored and
no byte can carry into its neighbour (255 + 255 + 1 = 511, shifted 255).

Each case is translated by tools/rdna2spirv.py (support.translate_lines) and executed with
tests/semantic/spirv_mini_interp.py; the expected words are written by hand from that formula and also recomputed by a
plain-Python reference, with VGPR, SGPR and literal sources and seeded random operands. Standard library only: no GPU,
no LLVM, no game data. (The old file modelled the instruction in its own classes and never ran the translator;
test_alu_lerp_u8.py is the older real test. `clamp` is refused by the translator and is not pinned here.)
"""
import os
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402


def pk(*b):
    return sum((x & 0xFF) << (8 * i) for i, x in enumerate(b))


def ref(s0, s1, s2):
    out = 0
    for i in range(4):
        b0, b1, r = (s0 >> 8 * i) & 0xFF, (s1 >> 8 * i) & 0xFF, (s2 >> 8 * i) & 1
        out |= ((b0 + b1 + r) >> 1) << 8 * i
    return out


def run(s0, s1, s2, form="vgpr"):
    if form == "vgpr":
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {s2:#x}", "v_lerp_u8 v0, v1, v2, v3"]
    elif form == "sgpr":
        lines = [f"s_mov_b32 s1, {s0:#x}", f"s_mov_b32 s2, {s1:#x}", f"s_mov_b32 s3, {s2:#x}", "v_lerp_u8 v0, s1, s2, s3"]
    elif form == "literal":
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v3, {s2:#x}", f"v_lerp_u8 v0, v1, {s1:#x}, v3"]
    else:
        raise ValueError(form)
    lines += ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_lerp_u8", lines))[0]


# (S0, S1, S2, expected word), written by hand, byte 0 first in pk().
CASES = [
    (pk(0, 0, 0, 0), pk(0, 0, 0, 0), 0, 0),
    (pk(10, 20, 30, 40), pk(20, 40, 60, 80), 0, pk(15, 30, 45, 60)),            # exact averages
    (pk(1, 2, 3, 4), pk(2, 3, 4, 5), 0, pk(1, 2, 3, 4)),                        # .5 rounds down with bit 0 clear
    (pk(1, 2, 3, 4), pk(2, 3, 4, 5), 0x01010101, pk(2, 3, 4, 5)),               # and up with it set
    (pk(1, 2, 3, 4), pk(2, 3, 4, 5), pk(1, 0, 1, 0), pk(2, 2, 4, 4)),           # per byte rounding
    (pk(1, 2, 3, 4), pk(2, 3, 4, 5), 0xFEFEFEFE, pk(1, 2, 3, 4)),               # bits 7:1 of S2 ignored
    (pk(255, 255, 255, 255), pk(255, 255, 255, 255), 0x01010101, 0xFFFFFFFF),   # 511 >> 1, no carry out of a byte
    (pk(255, 0, 255, 0), pk(255, 0, 0, 255), 0x01010101, pk(255, 0, 128, 128)),
    (pk(0, 0, 0, 0), pk(1, 1, 1, 1), 0x01010101, pk(1, 1, 1, 1)),
    (pk(0x80, 0x7F, 0x81, 0xFE), pk(0x80, 0x80, 0x7F, 0xFF), 0, pk(0x80, 0x7F, 0x80, 0xFE)),
]


class LerpU8(unittest.TestCase):
    def test_hand_table_matches_reference(self):
        for s0, s1, s2, want in CASES:
            self.assertEqual(ref(s0, s1, s2), want, (hex(s0), hex(s1), hex(s2)))

    def test_translated_forms(self):
        for form in ("vgpr", "sgpr", "literal"):
            for s0, s1, s2, want in CASES:
                with self.subTest(form=form, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(run(s0, s1, s2, form), want)

    def test_symmetric_and_identity(self):
        rng = random.Random(0x1E8)
        for _ in range(20):
            a, b, c = (rng.getrandbits(32) for _ in range(3))
            self.assertEqual(run(a, b, c), run(b, a, c))
            self.assertEqual(run(a, a, c), a)

    def test_random_operands(self):
        rng = random.Random(0x1E8B)
        for _ in range(60):
            s0, s1, s2 = (rng.getrandbits(32) for _ in range(3))
            for form in ("vgpr", "sgpr", "literal"):
                with self.subTest(form=form, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(run(s0, s1, s2, form), ref(s0, s1, s2))


if __name__ == "__main__":
    unittest.main()
