"""ds_add_u32, ds_add_rtn_u32, ds_sub_u32 and ds_sub_rtn_u32 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": ds_add_u32 is the atomic M = M + D (mod 2^32) on the dword at
(vaddr + offset) / 4, ds_sub_u32 is M = M - D; the _rtn_ form also returns the old M in vdst, the plain form leaves
vdst alone. The offset is added to the address in bytes.

Each case is a compute shader translated by tools/rdna2spirv.py (support.translate_lines) and run in
tests/semantic/spirv_mini_interp.py (one lane: that interpreter does not share LDS between lanes, see lds_run.py).
It writes a few LDS dwords, runs a sequence of these atomics on them and stores each returned value and the final
dwords. The one-operation tables are written out by hand from the formulas above; seeded random sequences are checked
against a plain-Python LDS. Until this file, the LDS tests of this name ran a self-built interpreter that checked
only the shape of the SPIR-V.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)

M32 = L.M32
FN = {"add": lambda m, d: (m + d) & M32, "sub": lambda m, d: (m - d) & M32}


def model(ops, init):
    lds = dict(init)
    rtns = []
    for name, addr, off, (d,) in ops:
        a = addr + off
        old = lds[a]
        rtns.append(old if "_rtn_" in name else 0xDEAD)
        lds[a] = FN[name.split("_")[1]](old, d)
    return rtns, {a: lds[a] for a in sorted({addr + off for _, addr, off, _ in ops})}


class TestHandTable(unittest.TestCase):
    TABLE = (            # (M, D, M + D, M - D)
        (0, 0, 0, 0), (1, 2, 3, 0xFFFFFFFF), (0xFFFFFFFF, 1, 0, 0xFFFFFFFE), (0x7FFFFFFF, 1, 0x80000000, 0x7FFFFFFE),
        (0x80000000, 0x80000000, 0, 0), (5, 0xFFFFFFFF, 4, 6), (0, 0xFFFFFFFF, 0xFFFFFFFF, 1),
        (0x12345678, 0x9ABCDEF0, 0xACF13568, 0x77777788),
    )

    def test_one_operation(self):
        for m, d, add, sub in self.TABLE:
            for name, want in (("ds_add_rtn_u32", add), ("ds_sub_rtn_u32", sub), ("ds_add_u32", add),
                               ("ds_sub_u32", sub)):
                with self.subTest(op=name, m=hex(m), d=hex(d)):
                    rtns, fin = L.run_chain([(name, 0x40, 0, (d,))], {0x40: m})
                    self.assertEqual((rtns, fin), ([m if "_rtn_" in name else 0xDEAD], {0x40: want}))

    def test_offset_adds_to_the_address_in_bytes(self):
        # vaddr 0x20 + offset 0x24 is the dword at byte 0x44; the dwords on either side stay as seeded.
        init = {0x40: 1, 0x44: 40, 0x48: 3}
        rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x20, 0x24, (2,))], init)
        self.assertEqual(rtns, [40])
        self.assertEqual(fin, {0x44: 42})
        rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x20, 0x24, (2,)), ("ds_sub_u32", 0x40, 0, (5,)),
                                 ("ds_add_u32", 0x48, 0, (7,))], init)
        self.assertEqual(rtns, [40, 0xDEAD, 0xDEAD])
        self.assertEqual(fin, {0x40: 0xFFFFFFFC, 0x44: 42, 0x48: 10})

    def test_chain_reuses_the_returned_value_order(self):
        # Three adds to the same dword: each returns the running total, in program order.
        ops = [("ds_add_rtn_u32", 8, 0, (10,)), ("ds_add_rtn_u32", 8, 0, (20,)), ("ds_sub_rtn_u32", 8, 0, (5,))]
        self.assertEqual(L.run_chain(ops, {8: 100}), ([100, 110, 130], {8: 125}))


class TestSeeded(unittest.TestCase):
    def test_random_sequences(self):
        rng = random.Random(0xADD5)
        for trial in range(8):
            addrs = [0x10, 0x30, 0x7C, 0x100]
            init = {a: rng.getrandbits(32) for a in addrs}
            ops = []
            for _ in range(12):
                name = rng.choice(("ds_add_rtn_u32", "ds_sub_rtn_u32", "ds_add_u32", "ds_sub_u32"))
                eff = rng.choice(addrs)
                off = rng.choice((0, 0, 0x10, 0x20))          # the same dword, split between vaddr and offset
                a = eff - off
                ops.append((name, a, off, (rng.choice((rng.getrandbits(32), 0xFFFFFFFF, 0x80000000, 1)),)))
            with self.subTest(trial=trial):
                self.assertEqual(L.run_chain(ops, init), model(ops, init))


if __name__ == "__main__":
    unittest.main()
