"""s_flbit_i32_b32 and s_flbit_i32 of a lane mask (vcc_lo, vcc_hi), run across every lane of the translated SPIR-V.

RDNA2 ISA: a lane mask is one bit per lane of the wave, so a scalar op that reads vcc_lo as a number sees lanes 0..31
(vcc_hi: lanes 32..63) and gives the same answer in every lane:
  s_flbit_i32_b32  D = the zeros above the highest set bit, -1 (0xFFFFFFFF) when the value is 0;
  s_flbit_i32      D = the bits above the first bit that differs from the sign bit, -1 when there is none.
The translator holds a mask as this lane's 0 / ~0, so reading it as data gave each lane its own 0 or -1 (the
s_bcnt1 and s_ff1 forms already gather the whole wave). Every lane stores its D; the expected values are computed
here from the two lines above, not by the translator. Programs are hand-written (no game data) and run with
spirv_mini_interp.run_workgroup. s_ff1_i32_b32 of the same mask is stored as a control for the gather.
"""
import struct
import unittest

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
M32 = 0xFFFFFFFF

# (multiplier, addend, and-mask): the lanes l with ((l * multiplier + addend) & and-mask) == 0 are in the mask.
CASES = (
    (1, 0, 8),       # lanes 0..7, 16..23, 32..39, ...
    (1, 0, 1),       # even lanes
    (1, 1, 1),       # odd lanes
    (1, 0, 0),       # every lane
    (1, 1, 0xFFFF),  # no lane (l + 1 is never 0 mod 2^16 below lane 64)
    (1, 0, 63),      # lane 0 only
    (1, 0, 32),      # the low half of a wave64, none of a wave32
    (1, 32, 32),     # the high half of a wave64, all of a wave32 (lane + 32 & 32 == 0 for lanes 32..63)
    (3, 5, 14),
)


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def flbit_u(x):
    return 31 - (x.bit_length() - 1) if x else M32


def flbit_i(x):
    sign = x >> 31
    for n in range(1, 32):
        if (x >> (31 - n)) & 1 != sign:
            return n
    return M32


def ff1(x):
    return (x & -x).bit_length() - 1 if x else M32


def mask_words(lanes, mul, add, mask):
    m = sum(1 << l for l in range(lanes) if ((l * mul + add) & mask) == 0)
    return m & M32, m >> 32


def program(lanes, reg, mul, add, mask):
    def store(d, at):
        return [f"v_mov_b32_e32 v20, {d}", f"v_add_nc_u32_e32 v21, {at}, v22",
                "buffer_store_dword v20, v21, s[0:3], 0 offen"]

    return (["v_mbcnt_lo_u32_b32 v1, -1, 0"] + (["v_mbcnt_hi_u32_b32 v1, -1, v1"] if lanes == 64 else []) +
            [f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3", "v_cmp_eq_u32_e32 vcc, 0, v3",
             f"s_flbit_i32_b32 s6, {reg}", f"s_flbit_i32 s7, {reg}", f"s_ff1_i32_b32 s8, {reg}",
             "v_lshlrev_b32_e32 v22, 2, v1"] +
            store("s6", 0) + store("s7", 256) + store("s8", 512) + ["s_endpgm"])


def run(spv, lanes):
    return si.run_workgroup(
        spv, lanes, lanes, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})


class FlbitOfLaneMask(unittest.TestCase):
    def check(self, lanes, reg):
        for mul, add, mask in CASES:
            with self.subTest(lanes=lanes, reg=reg, case=(mul, add, mask)):
                lo, hi = mask_words(lanes, mul, add, mask)
                word = hi if reg == "vcc_hi" else lo
                spv = support.translate_lines("cs_flbit_mask", program(lanes, reg, mul, add, mask),
                                              fssm=fssm(lanes))
                memory = run(spv, lanes)
                for lane in range(lanes):
                    self.assertEqual(memory[4 * lane], flbit_u(word), f"s_flbit_i32_b32, lane {lane}")
                    self.assertEqual(memory[256 + 4 * lane], flbit_i(word), f"s_flbit_i32, lane {lane}")
                    self.assertEqual(memory[512 + 4 * lane], ff1(word), f"s_ff1_i32_b32, lane {lane}")

    def test_model(self):
        self.assertEqual((flbit_u(1), flbit_u(0x80000000), flbit_u(0)), (31, 0, M32))
        self.assertEqual((flbit_i(1), flbit_i(0x40000000), flbit_i(M32), flbit_i(0)), (31, 1, M32, M32))

    def test_wave32_vcc_lo(self):
        self.check(32, "vcc_lo")

    def test_wave64_vcc_lo(self):
        self.check(64, "vcc_lo")

    def test_wave64_vcc_hi(self):
        self.check(64, "vcc_hi")


if __name__ == "__main__":
    unittest.main()
