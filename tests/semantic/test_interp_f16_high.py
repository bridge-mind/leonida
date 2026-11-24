"""v_interp_p1ll_f16 / v_interp_p2_f16 read the low or, with `high`, the high f16 of a packed attribute channel.

The producer packs two 16-bit varyings in one 32-bit param channel, low half and high half (Mesa
ac_nir_prerast_utils.c, nir_pack_32_2x16_split), and the PS picks a half with the `high` bit, passed to both ops of
the pair (Mesa ACO aco_isel_helpers.cpp emit_interp_instr; llvm-mc gfx1013 sets byte 5 bit 0 on both ops). A translated VS exports the channel's raw 32 bits, so each half has to be decoded per vertex and then
interpolated: interpolating the packed word as one f32 gives neither. support.varyings feeds the module what Vulkan
would give it for those raw per-vertex words.
"""
import struct
import unittest

import support

# Barycentric weights of vertices 0, 1, 2 (AMD i = 0.25, j = 0.5).
BARY = (0.25, 0.25, 0.5)
LO, HI = (0.5, 1.5, 2.5), (-4.0, 8.0, 16.0)       # per vertex; interpolated: 1.75 and 9.0, both exact in f16
INT_Y = (0x101, 0x102, 0x103)                    # attr0.y raw bits, read flat with v_interp_mov_f32
CTRL = (0.25, 0.5, 1.0)                          # attr1.x, f32, read with p1/p2_f32 only: 0.6875


def pack(lo, hi):
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


ATTR0 = [(pack(lo, hi), y, 0, 0) for lo, hi, y in zip(LO, HI, INT_Y)]
ATTR1 = [(struct.unpack("<I", struct.pack("<f", x))[0], 0, 0, 0) for x in CTRL]

SHADER = """v_interp_p1ll_f16 v2, v0, attr0.x
v_interp_p2_f16 v2, v1, attr0.x, v2
v_interp_p1ll_f16 v3, v0, attr0.x high
v_interp_p2_f16 v3, v1, attr0.x, v3 high
v_interp_p1_f32_e32 v4, v0, attr1.x
v_interp_p2_f32_e32 v4, v1, attr1.x
v_interp_mov_f32_e32 v5, p0, attr0.y
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""


def run(asm):
    spv = support.translate_lines("ps_interp_f16_high", asm.splitlines())
    return spv, support.mrt0_words(spv, inputs=support.varyings(spv, {0: ATTR0, 1: ATTR1}, BARY))


class InterpF16High(unittest.TestCase):
    def test_low_and_high_halves(self):
        _, words = run(SHADER)
        self.assertEqual(f"{words[0] & 0xFFFF:#06x}", "0x3f00")      # 1.75 from the low halves
        self.assertEqual(f"{words[1] & 0xFFFF:#06x}", "0x4880")      # 9.0 from the high halves

    def test_f32_location_control(self):
        """A location no f16 op reads stays a smooth f32 input (the same on the old translation)."""
        spv, words = run(SHADER)
        self.assertEqual(struct.unpack("<f", struct.pack("<I", words[2]))[0], 0.6875)
        self.assertNotIn("%in_attr1 PerVertexKHR", spv)

    def test_flat_read_of_a_per_vertex_location(self):
        """v_interp_mov_f32 of a location the f16 ops also read gets vertex 0's raw word."""
        _, words = run(SHADER)
        self.assertEqual(words[3], 0x101)

    def test_p2_f16_keeps_the_high_half_of_vdst(self):
        """v_interp_p2_f16 writes only the low half, like the other f16 results; the high half of vdst is kept
        (SHADER-430: measured on gfx1036 with a preset 0xdead high half, which survived every case)."""
        asm = SHADER.replace("exp mrt0 v2, v3, v4, v5 done vm",
                             "v_mov_b32_e32 v6, 0xdead0000\n"
                             "v_interp_p1ll_f16 v7, v0, attr0.x high\n"
                             "v_interp_p2_f16 v6, v1, attr0.x, v7 high\n"
                             "exp mrt0 v2, v3, v4, v6 done vm")
        _, words = run(asm)
        self.assertEqual(f"{words[3]:#010x}", "0xdead4880")            # 9.0 under the kept 0xdead

    def test_without_high_both_reads_are_the_low_half(self):
        _, words = run(SHADER.replace(" high", ""))
        self.assertEqual([f"{w & 0xFFFF:#06x}" for w in words[:2]], ["0x3f00", "0x3f00"])


if __name__ == "__main__":
    unittest.main()
