"""ds_min_u32, ds_max_u32, ds_min_i32, ds_max_i32 (and their _rtn_ forms) through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": the atomic M = min(M, D) or max(M, D) on the dword at (vaddr + offset) / 4,
comparing as unsigned (_u32) or two's-complement signed (_i32); the _rtn_ form also returns the old M in vdst, the
plain form leaves vdst alone. The two comparisons differ exactly when the operands differ in bit 31: 0x80000000 is the
largest signed-negative and the 2^31-th unsigned value.

Each case is a compute shader translated by tools/rdna2spirv.py (support.translate_lines) and run in
tests/semantic/spirv_mini_interp.py (one lane: that interpreter does not share LDS between lanes, see lds_run.py).
It writes a few LDS dwords, runs a sequence of atomics and stores each returned value and the final dwords. The
tables are written out by hand; seeded random sequences are checked against a plain-Python LDS. Until this file, the
LDS tests of this name ran a self-built interpreter that checked only the shape of the SPIR-V.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)

M32 = L.M32


def s32(x):
    return x - (1 << 32) if x >> 31 else x


FN = {"min_u32": min, "max_u32": max,
      "min_i32": lambda m, d: m if s32(m) <= s32(d) else d, "max_i32": lambda m, d: m if s32(m) >= s32(d) else d}


def model(ops, init):
    lds = dict(init)
    rtns = []
    for name, addr, off, (d,) in ops:
        a = addr + off
        rtns.append(lds[a] if "_rtn_" in name else 0xDEAD)
        lds[a] = FN[name.replace("_rtn_", "_")[3:]](lds[a], d)
    return rtns, {a: lds[a] for a in sorted({addr + off for _, addr, off, _ in ops})}


class TestHandTable(unittest.TestCase):
    #       M           D           min_u       max_u       min_i       max_i
    TABLE = ((0, 0, 0, 0, 0, 0),
             (1, 2, 1, 2, 1, 2),
             (5, 3, 3, 5, 3, 5),
             (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000, 0x80000000, 0x7FFFFFFF),
             (0x80000000, 0x7FFFFFFF, 0x7FFFFFFF, 0x80000000, 0x80000000, 0x7FFFFFFF),
             (0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
             (0xFFFFFFFF, 1, 1, 0xFFFFFFFF, 0xFFFFFFFF, 1),
             (0x80000000, 0x80000001, 0x80000000, 0x80000001, 0x80000000, 0x80000001),
             (7, 7, 7, 7, 7, 7))

    def test_one_operation(self):
        for m, d, *want in self.TABLE:
            for kind, w in zip(("min_u32", "max_u32", "min_i32", "max_i32"), want):
                for rtn in (True, False):
                    name = f"ds_{kind[:3]}_rtn_{kind[4:]}" if rtn else f"ds_{kind}"
                    with self.subTest(op=name, m=hex(m), d=hex(d)):
                        rtns, fin = L.run_chain([(name, 0x40, 0, (d,))], {0x40: m})
                        self.assertEqual((rtns, fin), ([m if rtn else 0xDEAD], {0x40: w}))

    def test_signed_and_unsigned_diverge_on_the_same_operands(self):
        ops = [("ds_min_rtn_u32", 0x10, 0, (0x80000000,)), ("ds_min_rtn_i32", 0x14, 0, (0x80000000,))]
        rtns, fin = L.run_chain(ops, {0x10: 1, 0x14: 1})
        self.assertEqual(rtns, [1, 1])
        self.assertEqual(fin, {0x10: 1, 0x14: 0x80000000})

    def test_offset_adds_to_the_address_in_bytes(self):
        rtns, fin = L.run_chain([("ds_max_rtn_u32", 0x20, 0x24, (9,))], {0x44: 4, 0x40: 100})
        self.assertEqual((rtns, fin), ([4], {0x44: 9}))

    def test_running_reduction_in_program_order(self):
        # min of 50, 20, 70, 20, 5 into one dword starting at 100: each _rtn_ shows the minimum so far.
        ops = [("ds_min_rtn_u32", 8, 0, (d,)) for d in (50, 20, 70, 20, 5)]
        self.assertEqual(L.run_chain(ops, {8: 100}), ([100, 50, 20, 20, 20], {8: 5}))


class TestSeeded(unittest.TestCase):
    def test_random_sequences(self):
        rng = random.Random(0x3141)
        edge = (0, 1, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF)
        for trial in range(8):
            addrs = [0x10, 0x30, 0x7C, 0x100]
            init = {a: rng.choice((rng.getrandbits(32), *edge)) for a in addrs}
            ops = []
            for _ in range(12):
                kind = rng.choice(tuple(FN))
                name = f"ds_{kind[:3]}_rtn_{kind[4:]}" if rng.random() < 0.6 else f"ds_{kind}"
                eff = rng.choice(addrs)
                off = rng.choice((0, 0, 0x10, 0x20))
                ops.append((name, eff - off, off, (rng.choice((rng.getrandbits(32), *edge)),)))
            with self.subTest(trial=trial):
                self.assertEqual(L.run_chain(ops, init), model(ops, init))


if __name__ == "__main__":
    unittest.main()
