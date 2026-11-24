"""A constant lane mask that reaches a mask consumer through an SGPR: lane L takes bit L of the constant.

`s_mov_b32 s20, 0x3ffffc00; s_and_saveexec_b32 s6, s20` leaves lanes 10..29 active on hardware. Gen.mask_src read the
SGPR's raw value, which is the same non-zero constant in every invocation, so every lane stayed on (found writing
#1771's GPU test; the immediate form `s_and_saveexec_b32 s6, 0x3ffffc00` was already right). Each case runs a
translated pixel shader in tests/semantic/spirv_wave_interp.py as one wave and compares every lane with the
immediate form and with the bits of the constant. Masks built by compares still work (they hold 0 / ~0 per lane).
Standard library only; no GPU, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_salu64_bitwise_data import run_ps_wave  # noqa: E402  (also puts tools/ on sys.path)

EPILOGUE = ["v_mov_b32 v0, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]


def active_lanes(lines, wave):
    """Lanes whose v1 ends as 0x22: the ones the mask under test left active."""
    outs, _ = run_ps_wave(["v_mov_b32 v1, 0x11"] + lines + ["v_mov_b32 v1, 0x22", f"s_mov_b{wave} exec, -1"] +
                          EPILOGUE, wave=wave)
    return [lane for lane in range(wave) if outs[lane][1] == 0x22]


def bits(v, wave):
    return [lane for lane in range(wave) if v >> lane & 1]


class SgprConstantMask(unittest.TestCase):
    def test_saveexec_wave32(self):
        for c in (0x3FFFFC00, 0x00000001, 0x80000000, 0x0000FF00, 0x55555555):
            with self.subTest(c=hex(c)):
                via_sgpr = active_lanes([f"s_mov_b32 s20, {c:#x}", "s_and_saveexec_b32 s6, s20"], 32)
                via_imm = active_lanes([f"s_and_saveexec_b32 s6, {c:#x}"], 32)
                self.assertEqual(via_sgpr, bits(c, 32))
                self.assertEqual(via_sgpr, via_imm)

    def test_exec_write_and_cndmask(self):
        c = 0x0F0F00FF
        self.assertEqual(active_lanes([f"s_mov_b32 s20, {c:#x}", "s_and_b32 exec_lo, exec_lo, s20"], 32), bits(c, 32))
        outs, _ = run_ps_wave([f"s_mov_b32 s20, {c:#x}", "v_cndmask_b32_e64 v1, 0, 1, s20"] + EPILOGUE, wave=32)
        self.assertEqual([lane for lane in range(32) if outs[lane][1] == 1], bits(c, 32))

    def test_sop2_with_a_constant_and_exec(self):
        # 32-bit bitwise SOP2 on EXEC with a constant, inline or in an SGPR (both were taken whole before).
        c, full = 0x0F0F00FF, 0xFFFFFFFF            # EXEC starts with every lane on
        for kind, want in (("and", full & c), ("andn2", full & ~c), ("or", full | c), ("xor", full ^ c)):
            for src in (f"{c:#x}", "s20"):
                with self.subTest(op=kind, src=src):
                    lines = [f"s_mov_b32 s20, {c:#x}", f"s_{kind}_b32 exec_lo, exec_lo, {src}"]
                    self.assertEqual(active_lanes(lines, 32), bits(want & full, 32))

    def test_exec_lo_from_an_sgpr_constant(self):
        c = 0x00FF00F0
        self.assertEqual(active_lanes([f"s_mov_b32 s20, {c:#x}", "s_mov_b32 exec_lo, s20"], 32), bits(c, 32))

    def test_wave64_pair_takes_each_half_from_its_own_sgpr(self):
        lo, hi = 0x0000FFFF, 0xF0000000
        lanes = active_lanes([f"s_mov_b32 s20, {lo:#x}", f"s_mov_b32 s21, {hi:#x}",
                              "s_and_saveexec_b64 s[6:7], s[20:21]"], 64)
        self.assertEqual(lanes, bits(lo | hi << 32, 64))

    def test_masks_from_compares_still_work(self):
        lines = ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_cmp_gt_u32_e64 s20, 12, v4", "s_and_saveexec_b32 s6, s20"]
        self.assertEqual(active_lanes(lines, 32), list(range(12)))
        lines = ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_mbcnt_hi_u32_b32 v4, -1, v4", "v_cmp_lt_u32_e64 s[20:21], 40, v4",
                 "s_and_saveexec_b64 s[6:7], s[20:21]"]
        self.assertEqual(active_lanes(lines, 64), list(range(41, 64)))


class MaskAndConstantIntoAnSgpr(unittest.TestCase):
    """@Pierre-Kookis's case on #1792 (GPU-reproduced on AMD RDNA2 and an RTX 4070 Ti SUPER): an SOP2 of EXEC with a
    constant into a plain SGPR is data, and s_bcnt1 counts the wave's bits. The first version of this fix put each
    lane's bit there and got 32 / 0 instead of 1 / 8."""

    def test_s_and_of_exec_with_a_constant_counts_the_wave_bits(self):
        import struct
        import tempfile
        import warnings
        import spirv_mini_interp as si
        from support import STUB_DXBC, rdna2spirv
        lines = ["s_and_b32 s20, exec_lo, 1", "s_bcnt1_i32_b32 s21, s20", "s_and_b32 s22, exec_lo, 0xff00",
                 "s_bcnt1_i32_b32 s23, s22", "v_lshlrev_b32 v3, 3, v0", "v_mov_b32_e32 v1, s21",
                 "buffer_store_dword v1, v3, s[0:3], 0 offen", "v_add_nc_u32 v3, 4, v3", "v_mov_b32_e32 v1, s23",
                 "buffer_store_dword v1, v3, s[0:3], 0 offen", "s_endpgm"]
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "cs_bcnt.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            hdr = bytearray(0x148)
            struct.pack_into("<4I", hdr, 0x138, 0x88, 32, 1, 1)
            with open(os.path.join(tmp, "fssm", "cs_bcnt.fssm"), "wb") as f:
                f.write(hdr)
            src = os.path.join(tmp, "disasm", "cs_bcnt.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write("; cs_bcnt\n" + "".join(f"\t{ln}\n" for ln in lines))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
        for lane in range(32):
            self.assertEqual((mem[8 * lane], mem[8 * lane + 4]), (1, 8), f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
