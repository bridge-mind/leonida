"""Value tests for RDNA2 8/16-bit MUBUF loads and stores, run through spirv_mini_interp on the CPU.

buffer_load_{u,s}byte / {u,s}short zero- or sign-extend to 32 bits; the _d16 loads write one 16-bit half of vdata (the
_hi forms the high half; ubyte / sbyte extended to 16 bits) and keep the other; buffer_store_byte / short write the low
8 / 16 bits of vdata (the _d16_hi forms its high half) and leave the neighbouring bytes alone. The V# is raw
(OOB_SELECT 3): an access is in range when offset + its size <= num_records; out of range a load gives 0 and a store
writes nothing. Every byte offset in a dword is tried, so a 16-bit access at byte 3 crosses into the next dword.
Standard library only; no GPU, no game files.
"""
import os
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE, OUT, RECORDS = 0x100, 0x200, 0x10C            # buffer bytes BASE .. BASE + 11 are seeded; num_records
SEED = [0x44332211, 0x88776655, 0xCCBBAA99]


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_bsub.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_bsub.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_bsub.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_bsub\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, invocations=1):
    """s[8:11] is a raw V# (base 0, num_records RECORDS); s[0:3] (the push-constant V#) covers OUT for results."""
    vsharp = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {RECORDS:#x}", "s_mov_b32 s11, 0x31016fac"]
    mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
    si.run_workgroup(translate(vsharp + body), invocations, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory=mem)
    return mem


def seed_bytes():
    return bytearray(b"".join(struct.pack("<I", w) for w in SEED))


def words(buf):
    return [struct.unpack_from("<I", buf, 4 * k)[0] for k in range(len(SEED))]


STORES = {"buffer_store_byte": (1, 0), "buffer_store_byte_d16_hi": (1, 16),
          "buffer_store_short": (2, 0), "buffer_store_short_d16_hi": (2, 16)}
LOADS = {  # op: (bytes, signed, half: None = whole register)
    "buffer_load_ubyte": (1, False, None), "buffer_load_sbyte": (1, True, None),
    "buffer_load_ushort": (2, False, None), "buffer_load_sshort": (2, True, None),
    "buffer_load_ubyte_d16": (1, False, 0), "buffer_load_ubyte_d16_hi": (1, False, 1),
    "buffer_load_sbyte_d16": (1, True, 0), "buffer_load_sbyte_d16_hi": (1, True, 1),
    "buffer_load_short_d16": (2, False, 0), "buffer_load_short_d16_hi": (2, False, 1)}
KEEP = 0xBEEF1234


def load_ref(op, byte, buf=None):
    n, signed, half = LOADS[op]
    buf = seed_bytes() if buf is None else buf
    if BASE + byte + n > RECORDS:
        v = 0
    else:
        v = int.from_bytes(buf[byte:byte + n], "little")
        if signed and v >> (8 * n - 1):
            v -= 1 << (8 * n)
    if half is None:
        return v & M32
    return (KEEP & ~(0xFFFF << (16 * half)) | (v & 0xFFFF) << (16 * half)) & M32


def load(op, byte):
    body = [f"v_mov_b32_e32 v1, {BASE + byte:#x}", f"v_mov_b32_e32 v5, {KEEP:#x}", f"{op} v5, v1, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "buffer_store_dword v5, v6, s[0:3], 0 offen"]
    return run(body)[OUT]


class BufferSubDword(unittest.TestCase):
    def test_stores_keep_neighbouring_bytes(self):
        data = 0xF1E2D3C4
        for op, (n, shift) in STORES.items():
            for byte in range(0, 8):                 # byte 3 and 7: a short crosses into the next dword
                with self.subTest(op=op, byte=byte):
                    mem = run([f"v_mov_b32_e32 v1, {BASE + byte:#x}", f"v_mov_b32_e32 v2, {data:#x}",
                               f"{op} v2, v1, s[8:11], 0 offen"])
                    want = seed_bytes()
                    want[byte:byte + n] = ((data >> shift) & ((1 << 8 * n) - 1)).to_bytes(n, "little")
                    self.assertEqual([mem[BASE + 4 * k] for k in range(3)], words(want))

    def test_loads(self):
        for op in LOADS:
            for byte in range(0, 8):
                with self.subTest(op=op, byte=byte):
                    self.assertEqual(load(op, byte), load_ref(op, byte))

    def test_bounds_cover_the_access_size(self):
        # num_records = BASE + 12: the last byte is in range for a byte load, a short at BASE + 11 is not.
        for op, byte in (("buffer_load_ubyte", 11), ("buffer_load_ushort", 10), ("buffer_load_ushort", 11),
                         ("buffer_load_sbyte_d16_hi", 12), ("buffer_load_short_d16", 11)):
            with self.subTest(op=op, byte=byte):
                self.assertEqual(load(op, byte), load_ref(op, byte))
        # a store out of range writes nothing
        mem = run([f"v_mov_b32_e32 v1, {BASE + 11:#x}", "v_mov_b32_e32 v2, 0x7777",
                   "buffer_store_short v2, v1, s[8:11], 0 offen"])
        self.assertEqual([mem[BASE + 4 * k] for k in range(3)], SEED)

    def test_inactive_lane_does_nothing(self):
        body = ["v_cmpx_eq_u32_e32 1, v0", f"v_mov_b32_e32 v1, {BASE + 1:#x}", "v_add_nc_u32_e32 v2, 0x70, v0",
                "buffer_store_byte v2, v1, s[8:11], 0 offen"]
        mem = run(body, invocations=2)
        self.assertEqual(mem[BASE], (SEED[0] & ~0xFF00) | 0x71 << 8)

    def test_rejected_forms(self):
        for line in ("buffer_load_ubyte v[4:5], v1, s[8:11], 0 offen", "buffer_store_short v1, v2, s[8:11], 0 offen lds"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])

    def test_first_read_analysis_counts_the_kept_half(self):
        from shader_ir import parse_line
        self.assertEqual(rdna2spirv.read_before_write_vgprs(
            [parse_line("\tbuffer_load_short_d16_hi v5, v1, s[8:11], 0 offen")]), {1, 5})
        self.assertEqual(rdna2spirv.read_before_write_vgprs(
            [parse_line("\tbuffer_load_ushort v5, v1, s[8:11], 0 offen")]), {1})


if __name__ == "__main__":
    unittest.main()
