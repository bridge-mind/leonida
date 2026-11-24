"""Ballots and votes on a lane mask, through the translator (TEST-603).

Each test translates a short RDNA2 compute listing with tools/rdna2spirv.py and runs every lane of it with
tests/semantic/spirv_mini_interp.run_workgroup (which models OpGroupNonUniformBallot/All/Any/BitwiseOr over the
running lanes). The listing builds a lane mask in VCC from two push constants (lane i's bit is bit i of s4:s5),
then reads it the ways the translator turns into subgroup operations:

  s_bcnt1_i32_b64/_b32 vcc    population count (ballot bit count)
  s_ff1_i32_b64/_b32 vcc      lowest set lane, -1 when none (ballot find-LSB)
  s_cmp_eq_u64 vcc, exec      every active lane true (vote all)
  s_cmp_lg_u64 vcc, 0         some lane true (vote any)
  v_cndmask_b32 ..., vcc      each lane's own bit (ballot bit extract)
  s_and_saveexec + v_readfirstlane / s_bcnt1 exec, then EXEC restored: the mask as the new EXEC

The expected values come from the s4:s5 pattern alone, so a translator that reads the wrong half, the wrong lane
or a stale mask fails here. This replaces a file that modelled the SPIR-V operations in Python without
translating anything. Standard library only; no GPU, llvm-mc or game files.
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
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# COMPUTE_PGM_RSRC2 0x8090: 8 user SGPRs (s[0:3] the V#, s4..s7 push constants), TGID_X.
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
LANE_BASE = 0x100      # per-lane results start here, one dword per lane
RESTORED_BASE = 0x200  # and each lane's mark written after EXEC is restored
NONE = 0xFFFFFFFF


def make_fssm(threads):
    return bytes(0x138) + struct.pack("<4I", 0x8090, threads, 1, 1)


def translate_cs(name, asm, threads):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for ln in asm.splitlines():
                st = ln.strip()
                if st and not st.startswith(";"):
                    f.write(f"\t{st}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        return spv


def run_cs(spv, threads, lo, hi=0):
    pc = VSHARP + [lo & NONE, hi & NONE, 0, 0] + [0] * 24
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


def store_sgpr(sgpr, slot):
    return (f"v_mov_b32_e32 v4, {sgpr}\n"
            f"v_mov_b32_e32 v5, {slot * 4}\n"
            "buffer_store_dword v4, v5, s[0:3], 0 offen\n")


def store_scc(slot):
    return "s_cselect_b32 s30, 1, 0\n" + store_sgpr("s30", slot)


# Lane i's bit of VCC is ((s4 if i < 32 else s5) >> (i & 31)) & 1.
BUILD_VCC64 = """\
v_mov_b32_e32 v11, s5
v_cmp_lt_u32_e64 vcc, v0, 32
v_cndmask_b32_e64 v1, v11, s4, vcc
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc, 0, v3
"""
BUILD_VCC32 = """\
v_mov_b32_e32 v1, s4
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc_lo, 0, v3
"""
PER_LANE_BIT = f"""\
v_cndmask_b32_e64 v6, 0, 1, {{vcc}}
v_lshlrev_b32_e32 v7, 2, v0
v_add_nc_u32_e32 v7, 0x{LANE_BASE:x}, v7
buffer_store_dword v6, v7, s[0:3], 0 offen
"""


def listing(wave):
    if wave == 64:
        vcc, exec_, b, build = "vcc", "exec", "b64", BUILD_VCC64
        cmp_all, cmp_any = "s_cmp_eq_u64 vcc, exec", "s_cmp_lg_u64 vcc, 0"
        save, restore = "s_and_saveexec_b64 s[14:15], vcc", "s_mov_b64 exec, s[14:15]"
    else:
        vcc, exec_, b, build = "vcc_lo", "exec_lo", "b32", BUILD_VCC32
        cmp_all, cmp_any = "s_cmp_eq_u32 vcc_lo, exec_lo", "s_cmp_lg_u32 vcc_lo, 0"
        save, restore = "s_and_saveexec_b32 s14, vcc_lo", "s_mov_b32 exec_lo, s14"
    return (build
            + f"s_bcnt1_i32_{b} s20, {vcc}\n" + store_sgpr("s20", 0)
            + f"s_ff1_i32_{b} s21, {vcc}\n" + store_sgpr("s21", 1)
            + cmp_all + "\n" + store_scc(2)
            + cmp_any + "\n" + store_scc(3)
            + PER_LANE_BIT.format(vcc=vcc)
            # The mask as EXEC: the first lane left running, and how many there are.
            + save + "\n"
            + "v_readfirstlane_b32 s22, v0\n"
            + f"s_bcnt1_i32_{b} s23, {exec_}\n"
            + restore + "\n"
            # Every lane writes its own slot after the restore: a partial restore leaves the other slots unwritten.
            + "v_mov_b32_e32 v6, 1\n"
            + "v_lshlrev_b32_e32 v7, 2, v0\n"
            + f"v_add_nc_u32_e32 v7, 0x{RESTORED_BASE:x}, v7\n"
            + "buffer_store_dword v6, v7, s[0:3], 0 offen\n"
            + store_sgpr("s22", 4) + store_sgpr("s23", 5)
            + "s_endpgm\n")


def expected(mask, wave):
    full = (1 << wave) - 1
    mask &= full
    first = (mask & -mask).bit_length() - 1 if mask else NONE
    return {"count": mask.bit_count(), "ff1": first, "all": int(mask == full), "any": int(mask != 0),
            "bits": [(mask >> lane) & 1 for lane in range(wave)], "first_active": first,
            "exec_count": mask.bit_count()}


MASKS64 = [0, (1 << 64) - 1, 1, 1 << 63, 1 << 31, 1 << 32, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA,
           0xFFFFFFFF00000000, 0x00000000FFFFFFFF, 0x8000000180000001, 0x0123456789ABCDEF]
MASKS32 = [0, 0xFFFFFFFF, 1, 1 << 31, 0x55555555, 0xAAAAAAAA, 0xFFFF0000, 0x80000001, 0x89ABCDEF]


class BallotVoteTests(unittest.TestCase):
    spv = {}

    @classmethod
    def setUpClass(cls):
        cls.spv = {64: translate_cs("cs_w64_ballot_votes", listing(64), 64),
                   32: translate_cs("cs_w32_ballot_votes", listing(32), 32)}

    def check(self, wave, mask):
        mem = run_cs(self.spv[wave], wave, mask & NONE, mask >> 32)
        want = expected(mask, wave)
        self.assertEqual(mem[0], want["count"], "s_bcnt1 vcc")
        self.assertEqual(mem[4], want["ff1"], "s_ff1 vcc")
        self.assertEqual(mem[8], want["all"], "vote all (vcc == exec)")
        self.assertEqual(mem[12], want["any"], "vote any (vcc != 0)")
        self.assertEqual([mem[LANE_BASE + 4 * lane] for lane in range(wave)], want["bits"], "per-lane bit")
        if mask:  # with no lane left in EXEC, v_readfirstlane has no lane to read
            self.assertEqual(mem[16], want["first_active"], "v_readfirstlane under the mask as EXEC")
        self.assertEqual(mem[20], want["exec_count"], "s_bcnt1 exec under the mask as EXEC")
        self.assertEqual([mem.get(RESTORED_BASE + 4 * lane) for lane in range(wave)], [1] * wave,
                         "every lane runs again after EXEC is restored")

    def test_wave64(self):
        for mask in MASKS64:
            with self.subTest(mask=f"0x{mask:016x}"):
                self.check(64, mask)

    def test_wave32(self):
        for mask in MASKS32:
            with self.subTest(mask=f"0x{mask:08x}"):
                self.check(32, mask)

    def test_the_translation_uses_subgroup_operations(self):
        # The reads go through the subgroup, not a per-lane shortcut that would pass a one-lane run.
        for wave, spv in self.spv.items():
            with self.subTest(wave=wave):
                self.assertIn("OpGroupNonUniform", spv)


if __name__ == "__main__":
    unittest.main()
