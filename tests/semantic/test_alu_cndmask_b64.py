"""64-bit conditional select on RDNA2: there is no `v_cndmask_b64`; a 64-bit select is two v_cndmask_b32, low and high
dword, under the same condition bit (D = cond ? S1 : S0 per lane, for the lane's bit of VCC or of an SGPR mask).

Each case is translated by tools/rdna2spirv.py (support.translate_lines) and executed with
tests/semantic/spirv_mini_interp.py (one lane: the translator models a lane mask as that lane's own bit, 0 or 1); the expected words are
written by hand, and seeded random 64-bit operands are checked against a plain-Python select. `v_cndmask_b64` itself
stays Unsupported rather than guessed. Standard library only: no GPU, no LLVM, no game data. (The old file ran its own
SIMD simulator and never called the translator.)
"""
import os
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

M32 = 0xFFFFFFFF


def select64(cond, s0, s1):
    return s1 if cond else s0


def run(cond_mask, s0, s1, form="vcc"):
    """Select a 64-bit value with two v_cndmask_b32; returns (low, high) of v[0:1]. cond_mask is the lane's bit, 0 or 1."""
    setup = [f"v_mov_b32 v2, {s0 & M32:#x}", f"v_mov_b32 v3, {s0 >> 32:#x}",
             f"v_mov_b32 v4, {s1 & M32:#x}", f"v_mov_b32 v5, {s1 >> 32:#x}"]
    if form == "vcc":
        sel = [f"s_mov_b32 vcc_lo, {cond_mask:#x}", "v_cndmask_b32_e32 v0, v2, v4, vcc_lo",
               "v_cndmask_b32_e32 v1, v3, v5, vcc_lo"]
    elif form == "sgpr":
        sel = [f"s_mov_b32 s8, {cond_mask:#x}", "v_cndmask_b32_e64 v0, v2, v4, s8", "v_cndmask_b32_e64 v1, v3, v5, s8"]
    else:
        raise ValueError(form)
    lines = setup + sel + ["exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm"]
    w = support.mrt0_words(support.translate_lines("ps_cndmask64", lines))
    assert w[0] == w[2] and w[1] == w[3]
    return w[0], w[1]


A = 0x1111111122222222
B = 0xAAAAAAAABBBBBBBB


class CndmaskPair(unittest.TestCase):
    def test_by_hand(self):
        for form in ("vcc", "sgpr"):
            with self.subTest(form=form):
                self.assertEqual(run(0, A, B, form), (0x22222222, 0x11111111))      # bit clear: S0
                self.assertEqual(run(1, A, B, form), (0xBBBBBBBB, 0xAAAAAAAA))      # bit set: S1
                self.assertEqual(run(1, A, A, form), (0x22222222, 0x11111111))
                self.assertEqual(run(1, 0, 0xFFFFFFFFFFFFFFFF, form), (M32, M32))

    def test_both_halves_follow_the_same_bit(self):
        for cond in (0, 1):
            lo, hi = run(cond, 0x00000000FFFFFFFF, 0xFFFFFFFF00000000)
            self.assertEqual((hi << 32) | lo, select64(cond, 0x00000000FFFFFFFF, 0xFFFFFFFF00000000))

    def test_random(self):
        rng = random.Random(0xC0DE64)
        for _ in range(30):
            s0, s1, mask = rng.getrandbits(64), rng.getrandbits(64), rng.getrandbits(1)
            want = select64(mask, s0, s1)
            for form in ("vcc", "sgpr"):
                with self.subTest(form=form, s0=hex(s0), s1=hex(s1), mask=hex(mask)):
                    lo, hi = run(mask, s0, s1, form)
                    self.assertEqual((hi << 32) | lo, want)

    def test_v_cndmask_b64_is_refused(self):
        with self.assertRaises(Unsupported):
            support.translate_lines("ps_cndmask64_op", ["s_mov_b32 vcc_lo, 1", "v_cndmask_b64 v[0:1], v[2:3], v[4:5], vcc_lo",
                                                        "exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
