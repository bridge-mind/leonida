"""global_atomic_csub / buffer_atomic_csub (GFX10.3), run through spirv_mini_interp on the CPU.

RDNA2 ISA: tmp = M; M = (DATA > tmp) ? 0 : tmp - DATA (unsigned: a subtract clamped at 0, never wrapping); with glc
the old M comes back. The translator emits the OpAtomicCompareExchange loop of the inc / dec atomics
(Gen.atomic_inc_dec_cas). The 32-lane case contends on one dword and must equal 32 sequential
subtracts. Vonsek measured the formula on gfx1036 (#3193: 162/162 memory/data pairs, both forms). Standard library
only; no GPU, no game files.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_global_atomics import OUT, TARGET, run, si, translate  # noqa: E402  (also puts tools/ on sys.path)

M32 = 0xFFFFFFFF


def global_op(kind, m, d, glc=True):
    body = [f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {d:#x}", "v_mov_b32_e32 v5, 0xdead"]
    body.append(f"global_atomic_{kind} v5, v1, v2, s[0:1] glc" if glc else f"global_atomic_{kind} v1, v2, s[0:1]")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
    mem = run(body, {TARGET: m})
    return mem.get(OUT), mem.get(TARGET)


def buffer_op(kind, m, d, offset=0x40, records=0x80):
    body = ["s_mov_b32 s8, 0x100", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {d:#x}",
            f"buffer_atomic_{kind} v2, v1, s[8:11], 0 offen glc", "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v2, s[0:1]"]
    mem = run(body, {0x100 + offset: m})
    return mem.get(OUT), mem.get(0x100 + offset)


def csub(m, d):
    return 0 if d > m else m - d


VALUES = [0, 1, 2, 5, 6, 0x7FFFFFFF, 0x80000000, M32 - 1, M32]


class AtomicCsub(unittest.TestCase):
    def test_reference(self):
        self.assertEqual([csub(m, 5) for m in (0, 4, 5, 6, M32)], [0, 0, 0, 1, M32 - 5])

    def test_global_values(self):
        for m in VALUES:
            for d in VALUES:
                with self.subTest(m=hex(m), d=hex(d)):
                    self.assertEqual(global_op("csub", m, d), (m, csub(m, d)))

    def test_global_without_glc_keeps_vdst(self):
        self.assertEqual(global_op("csub", 3, 5, glc=False), (0xDEAD, 0))
        self.assertEqual(global_op("csub", 9, 5, glc=False), (0xDEAD, 4))

    def test_buffer_values_and_bounds(self):
        for m, d in ((0, 5), (5, 5), (9, 5), (0x80000000, M32)):
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(buffer_op("csub", m, d), (m, csub(m, d)))
        self.assertEqual(buffer_op("csub", 9, 5, offset=0x80), (0, 9))      # out of range: untouched, glc 0

    def test_32_lanes_on_one_dword(self):
        for m, d in ((100, 3), (50, 3), (0, 1), (M32, 0x10000000)):
            with self.subTest(m=m, d=d):
                spv = translate([f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {d:#x}",
                                 "global_atomic_csub v1, v2, s[0:1]"])
                mem = {TARGET: m}
                si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                     "%sg_lane": i},
                                 push_constants=[0] * 36, memory=mem)
                want = m
                for _ in range(32):
                    want = csub(want, d)
                self.assertEqual(mem[TARGET], want)

    def test_seeded(self):
        rng = random.Random(0xC5B)
        for _ in range(20):
            m, d = rng.getrandbits(32), rng.choice((rng.getrandbits(32), rng.getrandbits(8)))
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(global_op("csub", m, d), (m, csub(m, d)))


if __name__ == "__main__":
    unittest.main()
