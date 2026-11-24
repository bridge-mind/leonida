"""ds_read_u8_d16, ds_read_u8_d16_hi, ds_read_i8_d16, ds_read_i8_d16_hi: 8-bit LDS reads into one half of a VGPR.

Standard library only; no GPU, no LLVM, no game files.

    python -m unittest discover -s tests/semantic -p "test_lds_read_u8_d16.py" -v

RDNA2 ISA, "Data Share Instructions" (DS opcodes 162-165), the d16 forms of the byte reads:
  * ds_read_u8_d16     D[15:0]  = zero-extend(MEM[ADDR][7:0]), D[31:16] unchanged
  * ds_read_u8_d16_hi  D[31:16] = zero-extend(MEM[ADDR][7:0]), D[15:0] unchanged
  * ds_read_i8_d16     D[15:0]  = sign-extend 8 -> 16 of MEM[ADDR][7:0], D[31:16] unchanged
  * ds_read_i8_d16_hi  D[31:16] = sign-extend 8 -> 16 of MEM[ADDR][7:0], D[15:0] unchanged
ADDR = vaddr + offset (bytes). The "other half kept" rule is the one the translator already applies to
ds_read_u16_d16 / ds_read_u16_d16_hi.

The input is the GPU test runtime/test/shaders/cs_w32_lds_u8_d16.asm (fixture from regen_fixtures.py): it stores
LDS dword 0 = 0x7F80FF01, presets every destination to 0xBEEF1234 and reads single bytes with a non-zero offset
or base. It is translated for wave32 and run as one invocation (lane 0); the eight words it stores are compared
with values derived here from the ISA text above (the same values check.py / run_gpu_tests.py expect).
"""
import struct
import unittest

import support
import spirv_mini_interp as si

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

LDS_BYTES = (0x01, 0xFF, 0x80, 0x7F)         # bytes at byte address 0..3 of LDS dword 0 = 0x7F80FF01
OLD = 0xBEEF1234                             # every destination before the read


def sext16(b):
    return b | 0xFF00 if b & 0x80 else b


def to_lo(old, h):
    return (old & 0xFFFF0000) | h


def to_hi(old, h):
    return (old & 0xFFFF) | (h << 16)


# In the order the shader stores them.
EXPECTED = [
    to_lo(OLD, LDS_BYTES[1]),           # u8_d16     offset:1 -> 0xFF
    to_hi(OLD, LDS_BYTES[1]),           # u8_d16_hi  offset:1
    to_lo(OLD, sext16(LDS_BYTES[2])),   # i8_d16     offset:2 -> 0x80
    to_hi(OLD, sext16(LDS_BYTES[2])),   # i8_d16_hi  offset:2
    to_lo(OLD, sext16(LDS_BYTES[3])),   # i8_d16     offset:3 -> 0x7F
    to_hi(OLD, sext16(LDS_BYTES[3])),   # i8_d16_hi  offset:3
    to_lo(OLD, LDS_BYTES[2]),           # u8_d16     offset:2 -> 0x80 stays 0x0080
    to_hi(OLD, sext16(LDS_BYTES[1])),   # i8_d16_hi  v1 = 1, no offset -> 0xFF
]


def run():
    spv = support.translate_fixture("cs_w32_lds_u8_d16", fssm=FSSM)
    memory = si.run_workgroup(
        spv, 1, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    stray = sorted(a for a in memory if not 0 <= a < 32 or a % 4)
    if stray:
        raise AssertionError(f"stores outside out[0..7]: {stray}")
    return [memory.get(4 * k) for k in range(8)]


class LdsReadU8I8D16(unittest.TestCase):
    def test_reference_values(self):
        self.assertEqual(EXPECTED, [0xBEEF00FF, 0x00FF1234, 0xBEEFFF80, 0xFF801234,
                                    0xBEEF007F, 0x007F1234, 0xBEEF0080, 0xFFFF1234])

    def test_translated_shader(self):
        got = run()
        for k, (g, w) in enumerate(zip(got, EXPECTED)):
            with self.subTest(word=k):
                self.assertEqual(g, w, f"out[{k}] = {g if g is None else hex(g)}, want {hex(w)}")


if __name__ == "__main__":
    unittest.main()
