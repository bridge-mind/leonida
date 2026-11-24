"""Wave64: s_cbranch_vccz/s_cbranch_vccnz must test all 64 bits of VCC when it holds data.

RDNA2 ISA: VCCZ/VCCNZ test the whole 64-bit register. A shader that writes vcc_lo and
vcc_hi with s_mov_b32 is using VCC as ordinary scalar data, not as a lane mask: in the
translation each lane holds the same low word in %vcc and the same high word in %vcc_hi.
The old branch_condition() loaded only %vcc, so a high-only non-zero VCC looked zero.
Both SGPR-source and constant writes are covered: @ray3242's original reproducer uses
`s_mov_b32 vcc_lo, 0` / `s_mov_b32 vcc_hi, 0x8000000a`. A VCC that was a lane mask
earlier and is then overwritten with data is a separate
dataflow limitation of #2544 and is not covered here.

Each case is a synthetic 64-thread compute shader translated by tools/rdna2spirv.py and
run with tests/semantic/spirv_mini_interp.run_workgroup; output word 0 is 7 when the
branch was taken and 9 when it fell through. No llvm-mc, GPU or game data is used.
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # s[0:3]: raw buffer (OOB_SELECT 3), any offset in range

# An EXEC-only 64-bit op selects wave64 without making VCC a lane mask.
WAVE64 = [("s_and_b64 s[12:13], exec, exec", 4)]
BUILD_DATA = [
    ("s_mov_b32 vcc_lo, s4", 4),
    ("s_mov_b32 vcc_hi, s5", 4),
]
BUILD_MASK = [
    ("v_cmp_eq_u32_e32 vcc, 33, v0", 8),  # one lane true, a certified VCC lane mask
]


def observe(branch):
    return [
        ("s_mov_b32 s20, 7", 4),
        (branch + " 1", 4),
        ("s_mov_b32 s20, 9", 4),
        ("v_mov_b32_e32 v6, s20", 4),
        ("v_lshlrev_b32_e32 v5, 2, v0", 4),
        ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
        ("s_endpgm", 4),
    ]


def translate_cs(name, lines):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def run(spv, lo, hi):
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                            "%sg_lane": i}, push_constants=pc, memory={})
    return [mem[4 * l] for l in range(64)]


CASES = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 0x8000000A),
    (0xA, 0x8000000A),
    (M32, M32),
]
CONST_CASES = CASES  # @ray3242's reproducer uses constant writes, not SGPR sources


class Wave64VccDataBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.progs = {}
        for op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
            suffix = "z" if op.endswith("z") else "nz"
            cls.progs[("data", op)] = translate_cs(
                "cs_w64_vcc_data_branch_" + suffix, WAVE64 + BUILD_DATA + observe(op))
            cls.progs[("mask", op)] = translate_cs(
                "cs_w64_vcc_mask_branch_" + suffix, WAVE64 + BUILD_MASK + observe(op))
            for lo, hi in CONST_CASES:
                cls.progs[("const", lo, hi, op)] = translate_cs(
                    f"cs_w64_vcc_const_branch_{suffix}_{lo:x}_{hi:x}",
                    WAVE64 + [("s_mov_b32 vcc_lo, " + hex(lo), 4),
                              ("s_mov_b32 vcc_hi, " + hex(hi), 4)] + observe(op))

    def test_vcc_data_branch_checks_both_halves(self):
        for lo, hi in CASES:
            nz = bool((lo | hi) & M32)
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(run(self.progs[("data", "s_cbranch_vccz")], lo, hi),
                                 [9 if nz else 7] * 64)
                self.assertEqual(run(self.progs[("data", "s_cbranch_vccnz")], lo, hi),
                                 [7 if nz else 9] * 64)

    def test_vcc_constant_writes_check_both_halves(self):
        for lo, hi in CONST_CASES:
            nz = bool((lo | hi) & M32)
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(run(self.progs[("const", lo, hi, "s_cbranch_vccz")], 0, 0),
                                 [9 if nz else 7] * 64)
                self.assertEqual(run(self.progs[("const", lo, hi, "s_cbranch_vccnz")], 0, 0),
                                 [7 if nz else 9] * 64)

    def test_certified_vcc_lane_mask_keeps_whole_wave_vote(self):
        for op, taken in (("s_cbranch_vccz", 9), ("s_cbranch_vccnz", 7)):
            with self.subTest(op=op):
                self.assertEqual(run(self.progs[("mask", op)], 0, 0), [taken] * 64)


if __name__ == "__main__":
    unittest.main()
