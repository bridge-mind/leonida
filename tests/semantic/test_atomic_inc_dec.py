"""Value tests for RDNA2 32-bit global_atomic_{inc,dec} and buffer_atomic_{inc,dec} via spirv_mini_interp.

RDNA2 ISA (FLAT/GLOBAL and MUBUF):
  inc: old = M; M = (old >= DATA) ? 0 : old + 1        (unsigned compare)
  dec: old = M; M = (old == 0 || old > DATA) ? DATA : old - 1  (unsigned compare)
With `glc`, the pre-op value `old` is returned (in `vdst` for global_atomic_*, in `vdata` for buffer_atomic_*);
without `glc`, the destination/data register is preserved. Inactive lanes never touch memory; out-of-bounds MUBUF
lanes leave memory untouched and return 0 when `glc` is set. Both lower to a structured OpAtomicCompareExchange CAS
loop so no extra SPIR-V capability is required. Standard library only; no GPU, no LLVM, no game files.
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
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from shader_ir import parse_line as parse  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)
TARGET, OUT = 0x2000, 0x3000
ELEM, BOUT, RECORDS = 0x40, 0x80, 0x100

VALUES = [0, 1, 2, 3, 4, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, M32]


def model_inc(m, d):
    return 0 if m >= d else (m + 1) & M32


def model_dec(m, d):
    return d if (m == 0 or m > d) else (m - 1) & M32


MODELS = {"inc": model_inc, "dec": model_dec}


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_incdec.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_incdec.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_incdec.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_incdec\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, memory, nlanes=1):
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def run_global(kind, m, d, glc=True, offset=0, vaddr=TARGET):
    body = [f"v_mov_b32_e32 v1, {vaddr - offset:#x}", f"v_mov_b32_e32 v2, {d:#x}", "v_mov_b32_e32 v5, 0xdead"]
    off = f" offset:{offset}" if offset else ""
    body.append(f"global_atomic_{kind} v5, v1, v2, s[0:1]{off} glc" if glc else
                f"global_atomic_{kind} v1, v2, s[0:1]{off}")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
    mem = run(body, {TARGET: m})
    return mem.get(OUT), mem.get(TARGET)


def run_buffer(kind, m, d, glc=True, offset=ELEM, records=RECORDS):
    body = [
        "s_mov_b32 s8, 0", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
        f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {d:#x}",
        f"buffer_atomic_{kind} v2, v1, s[8:11], 0 offen" + (" glc" if glc else ""),
        "s_waitcnt vmcnt(0)",
        f"v_mov_b32_e32 v4, {BOUT:#x}", "s_mov_b32 s10, 0x1000", "buffer_store_dword v2, v4, s[8:11], 0 offen",
    ]
    mem = run(body, {offset: m})
    return mem.get(BOUT), mem.get(offset)


class CasLoop(unittest.TestCase):
    def test_initial_read_is_atomic_at_the_exchange_scope(self):
        for kind in ("inc", "dec"):
            for instruction in (f"global_atomic_{kind} v5, v1, v2, s[0:1] glc",
                                f"buffer_atomic_{kind} v2, v1, s[8:11], 0 offen glc"):
                with self.subTest(instruction=instruction):
                    lines = translate([instruction]).splitlines()
                    exchange, = (line.split("OpAtomicCompareExchange ")[1].split()
                                 for line in lines if "OpAtomicCompareExchange " in line)
                    loads = [line.split("OpAtomicLoad ")[1].split() for line in lines if "OpAtomicLoad " in line]
                    self.assertEqual(loads, [exchange[:4]])  # type, pointer, scope, memory semantics


class GlobalAtomicIncDec(unittest.TestCase):
    def test_edge_grid_returning_and_non_returning(self):
        for kind, fn in MODELS.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(run_global(kind, m, d, glc=True), (m, fn(m, d)))
                        self.assertEqual(run_global(kind, m, d, glc=False), (0xDEAD, fn(m, d)))

    def test_seeded_values(self):
        rng = random.Random(0x1CED)
        for _ in range(24):
            kind = rng.choice(("inc", "dec"))
            m, d = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(run_global(kind, m, d), (m, MODELS[kind](m, d)))

    def test_signed_offsets_and_off_addressing(self):
        for offset in (-2048, -4, 4, 2044):
            with self.subTest(offset=offset):
                self.assertEqual(run_global("inc", 2, 5, offset=offset), (2, 3))
                self.assertEqual(run_global("dec", 0, 5, offset=offset), (0, 5))
        body = [
            f"v_mov_b32_e32 v8, {TARGET:#x}", "v_mov_b32_e32 v9, 0",
            "v_mov_b32_e32 v2, 3", "v_mov_b32_e32 v5, 0xdead",
            "global_atomic_inc v5, v[8:9], v2, off glc",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run(body, {TARGET: 3})
        self.assertEqual((mem.get(OUT), mem.get(TARGET)), (3, 0))

    def test_inactive_lane_leaves_memory_and_vdst_alone(self):
        for kind, init, bound, want in (("inc", 3, 3, 0), ("dec", 0, 7, 7)):
            with self.subTest(op=kind):
                body = [
                    f"v_mov_b32_e32 v1, {TARGET:#x}",
                    f"v_mov_b32_e32 v2, {bound:#x}", "v_mov_b32_e32 v5, 0xdead",
                    "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                    "s_and_saveexec_b32 s4, vcc_lo",
                    f"global_atomic_{kind} v5, v1, v2, s[0:1] glc",
                    "s_mov_b32 exec_lo, s4",
                    "s_waitcnt vmcnt(0)",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run(body, {TARGET: init}, nlanes=2)
                self.assertEqual(mem.get(TARGET), want)
                self.assertEqual(mem.get(OUT), init)
                self.assertEqual(mem.get(OUT + 4), 0xDEAD)

    def test_multi_lane_cas_contention_wrapping(self):
        # 4 active lanes increment modulo 3 (DATA = 2) starting at 0 -> states 0 -> 1 -> 2 -> 0 -> 1.
        body_inc = [
            f"v_mov_b32_e32 v1, {TARGET:#x}", "v_mov_b32_e32 v2, 2",
            "global_atomic_inc v5, v1, v2, s[0:1] glc",
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32_e32 v6, 2, v0",
            f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run(body_inc, {TARGET: 0}, nlanes=4)
        self.assertEqual(mem.get(TARGET), 1)
        self.assertEqual(sorted(mem.get(OUT + 4 * i) for i in range(4)), [0, 0, 1, 2])

        # 4 active lanes decrement modulo 3 (DATA = 2) starting at 1 -> states 1 -> 0 -> 2 -> 1 -> 0.
        body_dec = [ln.replace("global_atomic_inc", "global_atomic_dec") for ln in body_inc]
        mem = run(body_dec, {TARGET: 1}, nlanes=4)
        self.assertEqual(mem.get(TARGET), 0)
        self.assertEqual(sorted(mem.get(OUT + 4 * i) for i in range(4)), [0, 1, 1, 2])

    def test_first_read_analysis_and_rejected_forms(self):
        for kind in ("inc", "dec"):
            self.assertEqual(rdna2spirv.read_before_write_vgprs([parse(f"\tglobal_atomic_{kind} v1, v2, s[0:1]")]),
                             {1, 2})
            self.assertEqual(
                rdna2spirv.read_before_write_vgprs([parse(f"\tglobal_atomic_{kind} v5, v1, v2, s[0:1] glc")]),
                {1, 2},
            )
        for line in (
            "global_atomic_inc_x2 v[4:5], v1, v[2:3], s[0:1] glc",
            "global_atomic_dec_x2 v[4:5], v1, v[2:3], s[0:1] glc",
            "global_atomic_inc v5, v1, v2, s[0:1] glc slc",
            "global_atomic_dec v5, v1, v2, s[0:1] glc dlc",
            "global_atomic_inc v1, v2, s[0:1] glc",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])


class BufferAtomicIncDec(unittest.TestCase):
    def test_edge_grid_returning_and_non_returning(self):
        for kind, fn in MODELS.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(run_buffer(kind, m, d, glc=True), (m, fn(m, d)))
                        self.assertEqual(run_buffer(kind, m, d, glc=False), (d, fn(m, d)))

    def test_bounds_and_oob_behavior(self):
        for kind, init, bound, want in (("inc", 5, 5, 0), ("dec", 0, 5, 5)):
            with self.subTest(op=kind):
                # Last in-bounds dword (ends exactly at RECORDS).
                self.assertEqual(run_buffer(kind, init, bound, offset=RECORDS - 4), (init, want))
                # Out-of-bounds: memory untouched, glc returns 0, non-glc preserves vdata.
                self.assertEqual(run_buffer(kind, init, bound, glc=True, offset=RECORDS), (0, init))
                self.assertEqual(run_buffer(kind, init, bound, glc=False, offset=RECORDS), (bound, init))

    def test_multi_lane_cas_contention_and_inactive_lane(self):
        body = [
            "s_mov_b32 s8, 0", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {RECORDS:#x}", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {ELEM:#x}", "v_mov_b32_e32 v2, 2",
            "v_cmp_lt_u32_e32 vcc_lo, v0, 3",
            "s_and_saveexec_b32 s4, vcc_lo",
            "buffer_atomic_inc v2, v1, s[8:11], 0 offen glc",
            "s_mov_b32 exec_lo, s4",
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32_e32 v4, 2, v0",
            f"v_add_nc_u32_e32 v4, {BOUT:#x}, v4",
            "s_mov_b32 s10, 0x1000", "buffer_store_dword v2, v4, s[8:11], 0 offen",
        ]
        # 3 active lanes increment from 0 with DATA=2 -> 0->1->2->0; lane 3 is inactive and keeps v2=2.
        mem = run(body, {ELEM: 0}, nlanes=4)
        self.assertEqual(mem.get(ELEM), 0)
        self.assertEqual(sorted(mem.get(BOUT + 4 * i) for i in range(3)), [0, 1, 2])
        self.assertEqual(mem.get(BOUT + 12), 2)

    def test_rejected_forms(self):
        for line in (
            "buffer_atomic_inc_x2 v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_dec_x2 v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_inc v[2:3], v1, s[8:11], 0 offen glc",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])


if __name__ == "__main__":
    unittest.main()
