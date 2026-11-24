"""s_movrels_b32 / s_movrels_b64 reading an SGPR that holds a lane mask (#1548, SHADER-284).

A lane-mask SGPR (a saved EXEC, a v_cmp result) holds only this lane's bit in the translation, so a plain dynamic
load copied that bit as data: with EXEC saved as 5, `s_movrels_b32 s12, s10` then `s_bcnt1_i32_b32 s13, s12` gave
32 where the ISA gives 2. The translator now collects the whole wave's mask word for every mask SGPR and uses it
when the index lands there. Each case runs every lane of a synthetic compute shader in
spirv_mini_interp.run_workgroup (wave32 and native wave64) and stores D and the bit count per lane. Standard
library only; no GPU, no game files.
"""
import gc
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw)


def run_cs(name, body, wave_size):
    """Translate `body` as a compute shader (s[0:3] = output V#) and run all lanes; return the memory."""
    prev = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = False
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            path = os.path.join(tmp, "disasm", name + ".s")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gc.collect()
    finally:
        rdna2spirv.WAVE64_EMULATE = prev
    return si.run_workgroup(spv, wave_size, wave_size,
                            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=VSHARP + [0] * 28, memory={})


def store(regs):
    """Store the four SGPRs `regs` for each lane at lane * 16 (v0 = lane)."""
    return (["s_mov_b32 exec_lo, -1", "v_lshlrev_b32 v20, 4, v0"] +
            [f"v_mov_b32 v{21 + k}, s{s}" for k, s in enumerate(regs)] +
            ["buffer_store_dwordx4 v[21:24], v20, s[0:3], 0 offen"])


def lanes(mem, wave_size, count):
    return [tuple(mem.get(lane * 16 + 4 * k, 0) for k in range(count)) for lane in range(wave_size)]


class MovrelsLaneMaskWave32(unittest.TestCase):
    def test_finding_reproducer(self):
        # The listing from the finding: EXEC = lanes 0 and 2, saved in s10, read back through s_movrels.
        body = ["s_mov_b32 exec_lo, 5", "s_mov_b32 s10, exec_lo", "s_mov_b32 exec_lo, -1", "s_mov_b32 m0, 0",
                "s_movrels_b32 s12, s10", "s_bcnt1_i32_b32 s13, s12",
                "s_mov_b32 s14, s10", "s_bcnt1_i32_b32 s15, s14"]       # the control: a plain copy
        got = lanes(run_cs("cs_movrels_mask_w32", body + store([12, 13, 14, 15]), 32), 32, 4)
        for lane, (d, bits, ctl_d, ctl_bits) in enumerate(got):
            with self.subTest(lane=lane):
                self.assertEqual((d, bits), (5, 2))
                self.assertEqual(ctl_bits, 2)
                self.assertEqual(ctl_d, 0xFFFFFFFF if lane in (0, 2) else 0)   # a copy stays a mask

    def test_index_from_m0_hits_mask_and_data_neighbours(self):
        # s10 data, s11 a v_cmp mask (lanes 1, 4, 9), s12 data; M0 picks each in turn from a base of s9.
        body = ["s_mov_b32 exec_lo, -1", "s_mov_b32 s10, 0x1234", "s_mov_b32 s12, 0xabcd",
                "v_cmp_eq_u32_e64 s11, v0, 1", "v_cmp_eq_u32_e64 s16, v0, 4", "s_or_b32 s11, s11, s16",
                "v_cmp_eq_u32_e64 s16, v0, 9", "s_or_b32 s11, s11, s16", "s_mov_b32 s9, 0"]
        reads = []
        for k, m in enumerate((1, 2, 3)):
            reads += [f"s_mov_b32 s17, {m}", "s_mov_b32 m0, s17", f"s_movrels_b32 s{20 + k}, s9"]
        reads += ["s_bcnt1_i32_b32 s23, s21"]
        got = lanes(run_cs("cs_movrels_mix_w32", body + reads + store([20, 21, 22, 23]), 32), 32, 4)
        want_mask = (1 << 1) | (1 << 4) | (1 << 9)
        for lane, row in enumerate(got):
            with self.subTest(lane=lane):
                self.assertEqual(row, (0x1234, want_mask, 0xABCD, 3))


class MovrelsLaneMaskWave64(unittest.TestCase):
    def test_b64_pair_of_saved_exec(self):
        # EXEC = lanes 0, 2 and 40, saved in s[10:11]; the b64 read gives both halves as data.
        body = ["s_mov_b64 exec, -1", "v_cmp_eq_u32_e64 s[4:5], v0, 0", "v_cmp_eq_u32_e64 s[6:7], v0, 2",
                "s_or_b64 s[4:5], s[4:5], s[6:7]", "v_cmp_eq_u32_e64 s[6:7], v0, 40",
                "s_or_b64 s[4:5], s[4:5], s[6:7]", "s_mov_b64 exec, s[4:5]", "s_mov_b64 s[10:11], exec",
                "s_mov_b64 exec, -1", "s_mov_b32 m0, 0", "s_movrels_b64 s[12:13], s[10:11]",
                "s_bcnt1_i32_b64 s14, s[12:13]", "s_mov_b32 m0, 1", "s_movrels_b32 s15, s10"]
        got = lanes(run_cs("cs_movrels_mask_w64", body + ["s_mov_b64 exec, -1"] + store([12, 13, 14, 15]), 64),
                    64, 4)
        for lane, row in enumerate(got):
            with self.subTest(lane=lane):
                self.assertEqual(row, (5, 1 << 8, 3, 1 << 8))   # M0 = 1 from s10 lands on s11, the high half


if __name__ == "__main__":
    unittest.main()
