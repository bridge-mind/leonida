"""ds_wrxchg_rtn_b32 through the translator (the two-dword form is in test_lds_wrxchg2.py).

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": ds_wrxchg_rtn_b32 vdst, vaddr, vdata is the atomic exchange: vdst receives the
old dword at (vaddr + offset) / 4 and the dword becomes vdata. There is no plain (non-returning) form of the
single-dword exchange.

Each case is a compute shader translated by tools/rdna2spirv.py (support.translate_lines) and run in
tests/semantic/spirv_mini_interp.py (one lane: that interpreter does not share LDS between lanes, see lds_run.py).
It writes a few LDS dwords, runs exchanges on them and stores each returned value and the final dwords. The tables
are written out by hand; seeded random sequences are checked against a plain-Python LDS. Until this file, the LDS
tests of this name ran a self-built interpreter that checked only the shape of the SPIR-V.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

XCHG = "ds_wrxchg_rtn_b32"


def model(ops, init):
    lds = dict(init)
    rtns = []
    for _, addr, off, (d,) in ops:
        rtns.append(lds[addr + off])
        lds[addr + off] = d
    return rtns, {a: lds[a] for a in sorted({addr + off for _, addr, off, _ in ops})}


class TestExchange(unittest.TestCase):
    def test_by_hand(self):
        for m, d in ((0, 0xFFFFFFFF), (0x12345678, 0), (0x80000000, 0x80000000), (0xFFFFFFFF, 0x7FFFFFFF), (5, 5)):
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(L.run_chain([(XCHG, 0x40, 0, (d,))], {0x40: m}), ([m], {0x40: d}))

    def test_offset_adds_to_the_address_in_bytes(self):
        rtns, fin = L.run_chain([(XCHG, 0x20, 0x24, (9,))], {0x44: 4, 0x40: 100, 0x48: 200})
        self.assertEqual((rtns, fin), ([4], {0x44: 9}))

    def test_a_chain_passes_each_value_on(self):
        # Exchanging 2, 3, 4 into a dword that holds 1 returns 1, 2, 3 and leaves 4.
        ops = [(XCHG, 8, 0, (v,)) for v in (2, 3, 4)]
        self.assertEqual(L.run_chain(ops, {8: 1}), ([1, 2, 3], {8: 4}))

    def test_other_dwords_are_untouched(self):
        ops = [(XCHG, 0x10, 0, (0xAAAA,)), (XCHG, 0x14, 0, (0xBBBB,))]
        self.assertEqual(L.run_chain(ops, {0x10: 1, 0x14: 2, 0x18: 3}), ([1, 2], {0x10: 0xAAAA, 0x14: 0xBBBB}))

    def test_seeded_sequences(self):
        rng = random.Random(0x7C6)
        for trial in range(8):
            addrs = [0x10, 0x30, 0x7C, 0x100]
            init = {a: rng.getrandbits(32) for a in addrs}
            ops = []
            for _ in range(12):
                eff, off = rng.choice(addrs), rng.choice((0, 0, 0x10, 0x20))
                ops.append((XCHG, eff - off, off, (rng.getrandbits(32),)))
            with self.subTest(trial=trial):
                self.assertEqual(L.run_chain(ops, init), model(ops, init))


class TestOutsideCompute(unittest.TestCase):
    def test_pixel_shader_refuses_the_exchange(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_lds_xchg", [f"{XCHG} v1, v0, v2", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
