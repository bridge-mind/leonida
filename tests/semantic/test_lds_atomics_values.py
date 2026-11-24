"""Value tests for the LDS atomics the translator already emits, run through spirv_mini_interp on the CPU.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions" (one lane, so an atomic is read, combine, write):
  ds_add_u32  M = M + D        ds_sub_u32  M = M - D        (mod 2^32)
  ds_min_u32 / ds_max_u32      unsigned min / max of M and D
  ds_min_i32 / ds_max_i32      signed min / max of M and D
  ds_and_b32 / ds_or_b32 / ds_xor_b32
  ds_wrxchg_rtn_b32            M = D
  ds_cmpst_rtn_b32 vdst, vaddr, data0, data1:  M = (M == data0) ? data1 : M
The _rtn_ forms also return the old M in vdst. ADDR = vaddr + offset (bytes), the dword at ADDR / 4.

Each case is a compute shader run as one invocation (lane 0), with LDS as a Workgroup variable (#592): it seeds an LDS
dword with ds_write_b32, runs the atomic, reads the dword back with ds_read_b32 and stores (returned value, new
memory word) to a buffer. Those words are compared with the plain-Python model above. Until now these atomics were
only checked structurally (test_lds_atomic_add.py models OpAtomicIAdd itself); spirv_mini_interp ran atomics on
physical-storage-buffer pointers only.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# RSRC2 0x40088: 4 user SGPRs (the output V#), workgroup id in s4, 4 KiB LDS; 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


MODEL = {
    "add_u32": lambda m, d: (m + d) & M32,
    "sub_u32": lambda m, d: (m - d) & M32,
    "min_u32": min,
    "max_u32": max,
    "min_i32": lambda m, d: m if s32(m) <= s32(d) else d,
    "max_i32": lambda m, d: m if s32(m) >= s32(d) else d,
    "and_b32": lambda m, d: m & d,
    "or_b32": lambda m, d: m | d,
    "xor_b32": lambda m, d: m ^ d,
}
VALUES = [0, 1, 7, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678]


def run_listing(body: list) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        name = "cs_lds_atomics"
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write("; " + name + "\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(FSSM)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
    return si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=VSHARP + [0] * 28, memory={})


def run_atomic(line: str, init: int, addr: int = 0x40, offset: int = 0, data=()) -> tuple:
    """LDS[addr + offset] = init; run `line` (v1 = addr, v3.. = data); return (v5, LDS[addr + offset])."""
    body = [f"v_mov_b32_e32 v1, {addr:#x}", f"v_mov_b32_e32 v2, {init:#x}", "v_mov_b32_e32 v5, 0xdead",
            f"ds_write_b32 v1, v2 offset:{offset}", "s_waitcnt lgkmcnt(0)"]
    body += [f"v_mov_b32_e32 v{3 + k}, {d:#x}" for k, d in enumerate(data)]
    body += [line + (f" offset:{offset}" if offset else ""), "s_waitcnt lgkmcnt(0)",
             f"ds_read_b32 v6, v1 offset:{offset}", "s_waitcnt lgkmcnt(0)",
             "v_mov_b32_e32 v31, 0", "buffer_store_dword v5, v31, s[0:3], 0 offen",
             "v_mov_b32_e32 v31, 4", "buffer_store_dword v6, v31, s[0:3], 0 offen"]
    mem = run_listing(body)
    return mem.get(0), mem.get(4)


class LdsAtomicValues(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in MODEL.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=f"ds_{kind[:-4]}_rtn_{kind[-3:]}", m=hex(m), d=hex(d)):
                        got = run_atomic(f"ds_{kind[:-4]}_rtn_{kind[-3:]} v5, v1, v3", m, data=(d,))
                        self.assertEqual(got, (m, fn(m, d)))

    def test_non_returning_forms_leave_vdst_alone(self):
        for kind, fn in MODEL.items():
            for m, d in ((5, 3), (0x80000000, 0x7FFFFFFF), (0xFFFFFFFF, 1)):
                with self.subTest(op=f"ds_{kind}", m=hex(m), d=hex(d)):
                    self.assertEqual(run_atomic(f"ds_{kind} v1, v3", m, data=(d,)), (0xDEAD, fn(m, d)))

    def test_wrxchg(self):
        for m, d in ((0, 0xFFFFFFFF), (0x12345678, 0), (0x80000000, 0x80000000)):
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(run_atomic("ds_wrxchg_rtn_b32 v5, v1, v3", m, data=(d,)), (m, d))

    def test_cmpst_rtn(self):
        # data0 (v3) is the comparator, data1 (v4) the value written on a match.
        for m, cmp_, new in ((5, 5, 9), (5, 6, 9), (0xFFFFFFFF, 0xFFFFFFFF, 0), (0, 1, 2)):
            with self.subTest(m=hex(m), cmp=hex(cmp_), new=hex(new)):
                got = run_atomic("ds_cmpst_rtn_b32 v5, v1, v3, v4", m, data=(cmp_, new))
                self.assertEqual(got, (m, new if m == cmp_ else m))

    def test_cmpst_without_return(self):
        # ds_cmpst_b32 vaddr, data0, data1: the same compare-and-store, vdst untouched.
        for m, cmp_, new in ((5, 5, 9), (5, 6, 9), (0x80000000, 0x80000000, 1)):
            with self.subTest(m=hex(m), cmp=hex(cmp_), new=hex(new)):
                got = run_atomic("ds_cmpst_b32 v1, v3, v4", m, data=(cmp_, new))
                self.assertEqual(got, (0xDEAD, new if m == cmp_ else m))

    def test_offset_addresses_the_same_dword(self):
        self.assertEqual(run_atomic("ds_add_rtn_u32 v5, v1, v3", 40, addr=0x20, offset=0x24, data=(2,)), (40, 42))

    def test_seeded(self):
        rng = random.Random(0x1D5)
        kinds = list(MODEL)
        for _ in range(20):
            kind, m, d = rng.choice(kinds), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(kind=kind, m=hex(m), d=hex(d)):
                got = run_atomic(f"ds_{kind[:-4]}_rtn_{kind[-3:]} v5, v1, v3", m, data=(d,))
                self.assertEqual(got, (m, MODEL[kind](m, d)))


if __name__ == "__main__":
    unittest.main()
