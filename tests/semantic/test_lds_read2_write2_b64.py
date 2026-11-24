"""ds_read2_b64, ds_write2_b64, ds_read2st64_b64, ds_write2st64_b64: two 64-bit LDS elements per instruction.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": the 2-address forms access two elements at
    ADDR0 = vaddr + offset0 * unit,  ADDR1 = vaddr + offset1 * unit
where unit is the element size in bytes (8 for b64), times 64 for the st64 ("stride 64") forms. A b64 element is two
consecutive dwords, low first:
  * ds_write2_b64      vaddr, vdata0[0:1], vdata1[0:1]: MEM[ADDR0] = vdata0, MEM[ADDR1] = vdata1
  * ds_read2_b64       vdst[0:3], vaddr:                vdst[0:1] = MEM[ADDR0], vdst[2:3] = MEM[ADDR1]
The b32 forms (ds_read2_b32 / ds_write2_b32 and their st64 variants) use unit 4 and are already translated; this
checks that the b64 forms use unit 8 and move both dwords of each element.

Each case is a compute shader run as one invocation (lane 0) through tests/semantic/spirv_mini_interp.py, with LDS
as a Workgroup variable (#592). It writes LDS with one form, reads it back with plain ds_read_b32 (and with
ds_read2_b64), and stores the dwords it read to a buffer; those words are compared with a plain-Python model of LDS.
"""
import os
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# RSRC2 0x40088: 4 user SGPRs (the output V#), workgroup id in s4, LDS_SIZE = 8 x 512 bytes = 4 KiB; 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
LDS_BYTES = 4096
A = (0x11111111, 0x22222222)        # vdata0 (lo, hi)
B = (0x33333333, 0x44444444)        # vdata1 (lo, hi)


def model_write2(base: int, off0: int, off1: int, unit: int) -> dict:
    """LDS dword index -> value after ds_write2_* of A at element offset0 and B at offset1."""
    lds = {}
    for (lo, hi), off in ((A, off0), (B, off1)):
        addr = (base + off * unit) % LDS_BYTES
        lds[addr // 4] = lo
        lds[(addr // 4 + 1) % (LDS_BYTES // 4)] = hi
    return lds


def run_listing(body: list) -> dict:
    """Translate the compute listing `body` (it ends with buffer stores) and run lane 0; return buffer words."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        name = "cs_lds_read2_b64"
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


def store_lines(regs) -> list:
    out = []
    for k, r in enumerate(regs):
        out += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    return out


def write_then_read_b32(op: str, base: int, off0: int, off1: int, unit: int) -> list:
    """ds_write2(st64)_b64 with A, B; read the four dwords back with ds_read_b32; return them."""
    lines = [f"v_mov_b32_e32 v2, {base:#x}", f"v_mov_b32_e32 v4, {A[0]:#x}", f"v_mov_b32_e32 v5, {A[1]:#x}",
             f"v_mov_b32_e32 v6, {B[0]:#x}", f"v_mov_b32_e32 v7, {B[1]:#x}",
             f"{op} v2, v[4:5], v[6:7] offset0:{off0} offset1:{off1}", "s_waitcnt lgkmcnt(0)"]
    regs = []
    for k, (off, w) in enumerate(((off0, 0), (off0, 4), (off1, 0), (off1, 4))):
        addr = (base + off * unit + w) % LDS_BYTES
        lines += [f"v_mov_b32_e32 v8, {addr:#x}", f"ds_read_b32 v{10 + k}, v8"]
        regs.append(10 + k)
    lines.append("s_waitcnt lgkmcnt(0)")
    mem = run_listing(lines + store_lines(regs))
    return [mem.get(4 * k) for k in range(4)]


class LdsWrite2B64(unittest.TestCase):
    def test_model(self):
        lds = model_write2(16, 1, 3, 8)
        self.assertEqual(lds, {6: A[0], 7: A[1], 10: B[0], 11: B[1]})

    def test_ds_write2_b64_unit_is_8_bytes(self):
        for base, off0, off1 in ((0, 0, 1), (16, 1, 3), (8, 2, 255), (0x100, 0, 7)):
            with self.subTest(base=base, offset0=off0, offset1=off1):
                self.assertEqual(write_then_read_b32("ds_write2_b64", base, off0, off1, 8), [*A, *B])

    def test_ds_write2st64_b64_unit_is_512_bytes(self):
        for base, off0, off1 in ((0, 0, 1), (8, 1, 3), (0x40, 2, 7)):
            with self.subTest(base=base, offset0=off0, offset1=off1):
                self.assertEqual(write_then_read_b32("ds_write2st64_b64", base, off0, off1, 8 * 64), [*A, *B])


class LdsRead2B64(unittest.TestCase):
    def read2(self, op: str, base: int, off0: int, off1: int, unit: int) -> list:
        """Seed LDS with ds_write_b32 at the two elements' dwords, ds_read2(st64)_b64 them into v[10:13]."""
        lines = []
        for k, (off, w) in enumerate(((off0, 0), (off0, 4), (off1, 0), (off1, 4))):
            addr = (base + off * unit + w) % LDS_BYTES
            lines += [f"v_mov_b32_e32 v8, {addr:#x}", f"v_mov_b32_e32 v9, {(A + B)[k]:#x}", "ds_write_b32 v8, v9"]
        lines += ["s_waitcnt lgkmcnt(0)", f"v_mov_b32_e32 v2, {base:#x}",
                  f"{op} v[10:13], v2 offset0:{off0} offset1:{off1}", "s_waitcnt lgkmcnt(0)"]
        mem = run_listing(lines + store_lines([10, 11, 12, 13]))
        return [mem.get(4 * k) for k in range(4)]

    def test_ds_read2_b64(self):
        for base, off0, off1 in ((0, 0, 1), (16, 1, 3), (8, 2, 255), (0x100, 7, 0)):
            with self.subTest(base=base, offset0=off0, offset1=off1):
                self.assertEqual(self.read2("ds_read2_b64", base, off0, off1, 8), [*A, *B])

    def test_ds_read2st64_b64(self):
        for base, off0, off1 in ((0, 0, 1), (8, 1, 3), (0x40, 2, 7)):
            with self.subTest(base=base, offset0=off0, offset1=off1):
                self.assertEqual(self.read2("ds_read2st64_b64", base, off0, off1, 8 * 64), [*A, *B])

    def test_round_trip(self):
        # write2_b64 then read2_b64 at the same offsets gives back both pairs.
        lines = ["v_mov_b32_e32 v2, 0x20", f"v_mov_b32_e32 v4, {A[0]:#x}", f"v_mov_b32_e32 v5, {A[1]:#x}",
                 f"v_mov_b32_e32 v6, {B[0]:#x}", f"v_mov_b32_e32 v7, {B[1]:#x}",
                 "ds_write2_b64 v2, v[4:5], v[6:7] offset0:3 offset1:9", "s_waitcnt lgkmcnt(0)",
                 "ds_read2_b64 v[10:13], v2 offset0:9 offset1:3", "s_waitcnt lgkmcnt(0)"]
        mem = run_listing(lines + store_lines([10, 11, 12, 13]))
        self.assertEqual([mem.get(4 * k) for k in range(4)], [*B, *A])


if __name__ == "__main__":
    unittest.main()
