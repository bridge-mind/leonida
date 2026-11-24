"""s_and_saveexec_b64 through the translator, on every lane of a wave64.

RDNA2 ISA, SOP1:
  s_and_saveexec_b64: D = EXEC; EXEC = S0 & EXEC; SCC = (EXEC != 0).

Each case translates a hand-written compute program with tools/rdna2spirv.py and runs 64 lanes in
tests/semantic/spirv_mini_interp.py (harness: saveexec_run.py), with S0 from VCC, an SGPR pair written by v_cmp, an
SGPR pair loaded with literals, and EXEC itself, against the ISA formula in plain Python. Until this file, the test of
this name ran a self-built interpreter and never called the translator.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saveexec_run  # noqa: E402  (support puts tools/ on sys.path)


class Saveexec(unittest.TestCase):
    def test_and_by_hand(self):
        # S0 = lanes 0..31, EXEC = lanes 16..47: EXEC = S0 & EXEC written out by hand.
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        s0, e = lanes(0, 32), lanes(16, 48)
        want = {"and": lanes(16, 32), "andn2": lanes(0, 16), "or": lanes(0, 48), "orn2": lanes(0, 32) | lanes(48, 64) | lanes(16, 32),
                "xor": lanes(0, 16) | lanes(32, 48), "nand": lanes(0, 16) | lanes(32, 64),
                "nor": lanes(48, 64), "xnor": lanes(16, 32) | lanes(48, 64)}["and"]
        self.assertEqual(saveexec_run.model("and", s0, e), (want, e, 1))

    def test_and_every_lane(self):
        saveexec_run.check(self, "and")


if __name__ == "__main__":
    unittest.main()
