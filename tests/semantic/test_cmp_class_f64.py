"""Semantic tests for RDNA2 v_cmp_class_f64 and v_cmpx_class_f64 via spirv_mini_interp.

Tests all 10 IEEE-754 binary64 classes (including payloads living solely in the low 32 bits), mask combinations and
ignored upper mask bits, VGPR/SGPR pairs, float inline / inline integer / 32-bit upper-half literal constants, VOP3
abs/neg sign-bit modifiers on src0, v_cmpx_class_f64 EXEC narrowing, inactive lanes, absence of Float64 capability,
and rejection of malformed operands/modifiers. Standard library only; no GPU, no LLVM, no game files.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# IEEE-754 binary64 bit patterns -> expected class bit index (0..9).
CLASS_SAMPLES = [
    # bit 0: sNaN (exp=0x7ff, bit 51=0, mantissa != 0 — including payload in lo word only, both signs)
    (0x00000001, 0x7FF00000, 0),
    (0x00000000, 0x7FF40000, 0),
    (0x00000001, 0xFFF00000, 0),
    (0xFFFFFFFF, 0xFFF7FFFF, 0),
    # bit 1: qNaN (exp=0x7ff, bit 51=1, both signs)
    (0x00000000, 0x7FF80000, 1),
    (0x00000001, 0x7FF80000, 1),
    (0x00000000, 0xFFF80000, 1),
    (0xFFFFFFFF, 0xFFFFFFFF, 1),
    # bit 2: -inf
    (0x00000000, 0xFFF00000, 2),
    # bit 3: -normal (min, -1.0, max)
    (0x00000000, 0x80100000, 3),
    (0x00000000, 0xBFF00000, 3),
    (0xFFFFFFFF, 0xFFEFFFFF, 3),
    # bit 4: -subnormal (payload in lo only and in hi only)
    (0x00000001, 0x80000000, 4),
    (0x00000000, 0x80080000, 4),
    (0xFFFFFFFF, 0x800FFFFF, 4),
    # bit 5: -0.0
    (0x00000000, 0x80000000, 5),
    # bit 6: +0.0
    (0x00000000, 0x00000000, 6),
    # bit 7: +subnormal (payload in lo only and in hi only)
    (0x00000001, 0x00000000, 7),
    (0x00000000, 0x00080000, 7),
    (0xFFFFFFFF, 0x000FFFFF, 7),
    # bit 8: +normal (min, +1.0, max)
    (0x00000000, 0x00100000, 8),
    (0x00000000, 0x3FF00000, 8),
    (0xFFFFFFFF, 0x7FEFFFFF, 8),
    # bit 9: +inf
    (0x00000000, 0x7FF00000, 9),
]


def translate(lines, export="v0, v1, v2, v3"):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_cmp_class_f64.s")
        with open(os.path.join(tmp, "ps_cmp_class_f64.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; ps_cmp_class_f64\n" + "".join(f"\t{ln}\n" for ln in lines)
                    + f"\texp mrt0 {export} done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(lines, export="v0, v1, v2, v3"):
    return list(si.run_fragment(translate(lines, export=export))["%out_mrt0"])


class CmpClassF64Tests(unittest.TestCase):
    def test_all_ten_classes_exhaustive_masks(self):
        # For every sample, verify exact class bit hit, complementary mask miss, and high-bit ignore.
        for lo, hi, bit in CLASS_SAMPLES:
            hit_mask = 1 << bit
            miss_mask = 0x3FF ^ hit_mask
            hi_only_mask = 0xFFFFFC00
            with self.subTest(lo=hex(lo), hi=hex(hi), bit=bit):
                got = run([
                    f"v_mov_b32_e32 v4, {lo:#x}",
                    f"v_mov_b32_e32 v5, {hi:#x}",
                    f"v_mov_b32_e32 v6, {hit_mask:#x}",
                    f"v_mov_b32_e32 v7, {miss_mask:#x}",
                    f"v_mov_b32_e32 v8, {hi_only_mask:#x}",
                    "v_cmp_class_f64_e32 vcc_lo, v[4:5], v6",
                    "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
                    "v_cmp_class_f64_e64 s4, v[4:5], v7",
                    "v_cndmask_b32_e64 v1, 0, 1, s4",
                    "v_cmp_class_f64_e64 s5, v[4:5], v8",
                    "v_cndmask_b32_e64 v2, 0, 1, s5",
                    f"v_cmp_class_f64_e64 s6, v[4:5], {hit_mask | 0xFFFF0000:#x}",
                    "v_cndmask_b32_e64 v3, 0, 1, s6",
                ])
                self.assertEqual(got, [1, 0, 0, 1])

    def test_sgpr_pair_and_constants(self):
        # SGPR pair (-inf), fimm (-2.0 -> bit 3), inline ints (0 -> +0 bit 6, 1 -> +subnorm bit 7, -1 -> qNaN bit 1),
        # and 32-bit upper-half literals (0x3ff00000 -> +1.0 bit 8, 0x7ff00000 -> +inf bit 9).
        got1 = run([
            "s_mov_b32 s8, 0",
            "s_mov_b32 s9, 0xfff00000",
            "v_cmp_class_f64_e64 s4, s[8:9], 0x4",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, -2.0, 0x8",
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0, 0x40",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 1, 0x80",
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got1, [1, 1, 1, 1])

        got2 = run([
            "v_cmp_class_f64_e64 s4, -1, 0x2",          # sign-extended 0xffffffff_ffffffff = qNaN
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x3ff00000, 0x100",  # +1.0 in upper 32 bits -> +normal
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x7ff00000, 0x200",  # +inf in upper 32 bits -> +inf
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x80000000, 0x20",   # -0.0 in upper 32 bits -> -0.0
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got2, [1, 1, 1, 1])

    def test_vop3_abs_and_neg_modifiers_on_src0(self):
        # abs clears bit 63 first, neg flips bit 63 second; sNaN stays sNaN (not quieted).
        got = run([
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0xbff00000",                # -1.0
            "v_cmp_class_f64_e64 s4, |v[4:5]|, 0x100",     # -> +normal (bit 8)
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, -|1.0|, 0x8",         # -> -normal (bit 3)
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, neg(0), 0x20",        # -> -0.0 (bit 5)
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_mov_b32_e32 v6, 1",
            "v_mov_b32_e32 v7, 0x7ff00000",                # +sNaN (lo=1)
            "v_cmp_class_f64_e64 s4, -v[6:7], 0x1",        # -> -sNaN, still bit 0 (not quieted to bit 1)
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got, [1, 1, 1, 1])

    def test_cmpx_exec_narrowing_and_inactive_lanes(self):
        # v_cmpx_class_f64 keeps active lane on match and clears EXEC on mismatch; inactive lane writes 0 to sdst.
        got = run([
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v1, 20",
            "v_mov_b32_e32 v2, 30",
            "v_mov_b32_e32 v3, 40",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0x7ff00000",                # +inf (bit 9)
            "s_mov_b32 s10, exec_lo",
            "v_cmpx_class_f64_e64 v[4:5], 0x200",          # matches -> lane stays active
            "v_mov_b32_e32 v0, 11",
            "v_cmpx_class_f64_e64 v[4:5], 0x4",            # -inf mismatch -> lane turns off
            "v_mov_b32_e32 v1, 21",                        # skipped
            "v_cmp_class_f64_e64 s6, v[4:5], 0x200",       # inactive -> writes 0 to s6
            "s_mov_b32 exec_lo, s10",
            "v_cndmask_b32_e64 v2, 31, 32, s6",            # s6 is 0 -> selects 31
            "v_cmp_class_f64_e64 s7, v[4:5], 0x200",       # active again -> writes 1
            "v_cndmask_b32_e64 v3, 41, 42, s7",            # s7 is 1 -> selects 42
        ])
        self.assertEqual(got, [11, 20, 31, 42])

    def test_no_float64_capability_required(self):
        spv = translate([
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0x3ff00000",
            "v_cmp_class_f64_e64 s4, v[0:1], 0x100",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_mov_b32_e32 v3, 0",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotIn("OpTypeFloat 64", spv)

    def test_rejected_forms(self):
        for line in (
            "v_cmp_class_f64_e32 vcc_lo, v0, v1",          # single VGPR instead of pair
            "v_cmp_class_f64_e64 s4, s0, 0x1",             # single SGPR instead of pair
            "v_cmp_class_f64_e64 s4, v[0:1], v[2:3]",      # pair mask
            "v_cmp_class_f64_e64 s4, v[0:1], -v2",         # neg on mask
            "v_cmp_class_f64_e64 v2, v[0:1], 0x1",         # VGPR destination
            "v_cmp_class_f64_e64 s4, v[0:1], 0x1 clamp",   # unsupported modifier
            "v_cmpx_class_f64_e64 s4, v[0:1], 0x1",        # non-exec 3-operand cmpx
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0",
                               "v_mov_b32_e32 v3, 0", line])


if __name__ == "__main__":
    unittest.main()
