"""ds_cmpst_f32 / ds_cmpst_rtn_f32, global_atomic_fcmpswap and buffer_atomic_fcmpswap, run through spirv_mini_interp.

RDNA2 ISA: tmp = M; M = (tmp == CMP compared as binary32) ? DATA : tmp; the returning forms give tmp. A float compare
is not a bit compare: -0 == +0 matches, and a NaN never matches, not even the same NaN bits. The translator emits an
OpAtomicCompareExchange loop (Gen.float_cmpswap_loop). Operand order: ds_cmpst_f32 takes CMP then DATA (as
ds_cmpst_b32); the global / buffer forms take vdata as the pair {DATA, CMP} (as cmpswap). Standard library only.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_atomic  # noqa: E402  (also puts tools/ on sys.path)
from test_global_atomics import OUT, TARGET, run, si, translate  # noqa: E402
import rdna2spirv  # noqa: E402


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def ref(m, cmp, new):
    return new if f32(m) == f32(cmp) else m          # Python float ==: -0 == +0, NaN != anything


ONE, TWO, PZ, NZ, NAN, NAN2, INF = bits(1.0), bits(2.0), bits(0.0), 0x80000000, 0x7FC00000, 0x7FC00001, 0x7F800000
CASES = [(ONE, ONE, TWO), (ONE, TWO, 0x12345678), (PZ, NZ, TWO), (NZ, PZ, TWO), (NAN, NAN, TWO), (NAN2, NAN2, ONE),
         (ONE, NAN, TWO), (INF, INF, ONE), (bits(-3.5), bits(-3.5), PZ), (0x00000001, 0x00000001, ONE)]


def global_op(m, cmp, new, glc=True):
    body = [f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {new:#x}", f"v_mov_b32_e32 v3, {cmp:#x}",
            "v_mov_b32_e32 v5, 0xdead",
            "global_atomic_fcmpswap v5, v1, v[2:3], s[0:1] glc" if glc else "global_atomic_fcmpswap v1, v[2:3], s[0:1]",
            "s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
    mem = run(body, {TARGET: m})
    return mem.get(OUT), mem.get(TARGET)


def buffer_op(m, cmp, new, offset=0x40):
    body = ["s_mov_b32 s8, 0x100", "s_mov_b32 s9, 0", "s_mov_b32 s10, 0x80", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {new:#x}", f"v_mov_b32_e32 v3, {cmp:#x}",
            "buffer_atomic_fcmpswap v[2:3], v1, s[8:11], 0 offen glc", "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v2, s[0:1]"]
    mem = run(body, {0x100 + offset: m})
    return mem.get(OUT), mem.get(0x100 + offset)


class FloatCmpswap(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(PZ, NZ, TWO), TWO)              # -0 == +0
        self.assertEqual(ref(NAN, NAN, TWO), NAN)            # NaN never matches
        self.assertEqual(ref(ONE, TWO, 7), ONE)

    def test_ds_cmpst_f32(self):
        for m, cmp, new in CASES:
            with self.subTest(m=hex(m), cmp=hex(cmp), new=hex(new)):
                self.assertEqual(run_atomic("ds_cmpst_rtn_f32 v5, v1, v3, v4", m, data=(cmp, new)), (m, ref(m, cmp, new)))
        self.assertEqual(run_atomic("ds_cmpst_f32 v1, v3, v4", PZ, data=(NZ, TWO)), (0xDEAD, TWO))

    def test_global_fcmpswap(self):
        for m, cmp, new in CASES:
            with self.subTest(m=hex(m), cmp=hex(cmp), new=hex(new)):
                self.assertEqual(global_op(m, cmp, new), (m, ref(m, cmp, new)))
        self.assertEqual(global_op(NZ, PZ, ONE, glc=False), (0xDEAD, ONE))

    def test_buffer_fcmpswap_and_bounds(self):
        for m, cmp, new in CASES[:6]:
            with self.subTest(m=hex(m), cmp=hex(cmp), new=hex(new)):
                self.assertEqual(buffer_op(m, cmp, new), (m, ref(m, cmp, new)))
        self.assertEqual(buffer_op(ONE, ONE, TWO, offset=0x80), (0, ONE))     # out of range: untouched, glc 0

    def test_32_lanes_on_one_dword(self):
        # Every lane swaps 0.0 -> 1.0 if it still sees +-0: one wins, the rest see 1.0 and keep it.
        spv = translate([f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {ONE:#x}", f"v_mov_b32_e32 v3, {NZ:#x}",
                         "global_atomic_fcmpswap v1, v[2:3], s[0:1]"])
        mem = {TARGET: PZ}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=[0] * 36, memory=mem)
        self.assertEqual(mem[TARGET], ONE)

    def test_rejected_forms(self):
        for line in ("global_atomic_fcmpswap v5, v1, v2, s[0:1] glc", "global_atomic_fcmpswap_x2 v[4:5], v1, v[2:5], s[0:1] glc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])


if __name__ == "__main__":
    unittest.main()
