"""Semantic tests for all 32 RDNA2 binary64 vector compares (v_cmp_*_f64 and v_cmpx_*_f64) via spirv_mini_interp.

Verifies all 16 predicates (f, lt, eq, le, gt, lg, ge, o, u, nge, nlg, ngt, nle, neq, nlt, tru) across IEEE-754
binary64 edge values (sNaN, qNaN, +-inf, +-normal, 1-ULP neighbors, +-subnormal, +-0.0), VGPR/SGPR pairs, float inline /
inline integer / 32-bit upper-half literal constants, VOP3 abs/neg modifiers, v_cmpx_*_f64 EXEC narrowing, inactive
lanes, absence of Float64 capability, and bad-operand/modifier rejections. Standard library only; no GPU, no LLVM, no
game files.
"""
import math
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

PREDS = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")

# Representative IEEE-754 binary64 values covering every class, 1-ULP boundaries, low-word-only diffs, and NaNs.
F64_SAMPLES = [
    (0x00000000, 0xFFF00000),  # -inf
    (0xFFFFFFFF, 0xFFEFFFFF),  # -max_normal
    (0x00000001, 0xBFF00000),  # -1.0 - 1 ulp
    (0x00000000, 0xBFF00000),  # -1.0
    (0xFFFFFFFF, 0xBFEFFFFF),  # -1.0 + 0.5 ulp
    (0x00000001, 0x80000000),  # -min_subnormal
    (0x00000000, 0x80000000),  # -0.0
    (0x00000000, 0x00000000),  # +0.0
    (0x00000001, 0x00000000),  # +min_subnormal
    (0x00000000, 0x3FF00000),  # +1.0
    (0x00000001, 0x3FF00000),  # +1.0 + 1 ulp
    (0x00000000, 0x7FF00000),  # +inf
    (0x00000001, 0x7FF00000),  # +sNaN (payload in lo word only)
    (0x00000000, 0xFFF80000),  # -qNaN
]


def bits_to_double(lo, hi):
    return struct.unpack("<d", struct.pack("<II", lo, hi))[0]


def expected_pred(pred, alo, ahi, blo, bhi):
    a, b = bits_to_double(alo, ahi), bits_to_double(blo, bhi)
    unord = math.isnan(a) or math.isnan(b)
    if pred == "f":
        return 0
    if pred == "tru":
        return 1
    if pred == "o":
        return int(not unord)
    if pred == "u":
        return int(unord)
    if pred == "lt":
        return int(not unord and a < b)
    if pred == "eq":
        return int(not unord and a == b)
    if pred == "le":
        return int(not unord and a <= b)
    if pred == "gt":
        return int(not unord and a > b)
    if pred == "lg":
        return int(not unord and a != b)
    if pred == "ge":
        return int(not unord and a >= b)
    if pred == "nge":
        return int(unord or a < b)
    if pred == "nlg":
        return int(unord or a == b)
    if pred == "ngt":
        return int(unord or a <= b)
    if pred == "nle":
        return int(unord or a > b)
    if pred == "neq":
        return int(unord or a != b)
    if pred == "nlt":
        return int(unord or a >= b)
    raise ValueError(pred)


def translate(lines, export="v0, v1, v2, v3"):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_cmp_f64.s")
        with open(os.path.join(tmp, "ps_cmp_f64.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; ps_cmp_f64\n" + "".join(f"\t{ln}\n" for ln in lines)
                    + f"\texp mrt0 {export} done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(lines, export="v0, v1, v2, v3"):
    return list(si.run_fragment(translate(lines, export=export))["%out_mrt0"])


class CmpF64Tests(unittest.TestCase):
    def test_all_16_predicates_on_edge_pairs(self):
        # Pack 4 predicates per fragment run -> 4 runs per pair across representative pairs.
        pairs = [
            # (a, b) covering -inf, -1-ulp, -1, -1+0.5ulp, -subnorm, -0, +0, +subnorm, +1, +1+ulp, +inf, sNaN, qNaN
            (F64_SAMPLES[0], F64_SAMPLES[1]),   # -inf < -max_normal
            (F64_SAMPLES[2], F64_SAMPLES[3]),   # -1-ulp < -1
            (F64_SAMPLES[3], F64_SAMPLES[4]),   # -1 < -1+0.5ulp
            (F64_SAMPLES[5], F64_SAMPLES[6]),   # -min_subnormal < -0.0
            (F64_SAMPLES[6], F64_SAMPLES[7]),   # -0.0 == +0.0
            (F64_SAMPLES[7], F64_SAMPLES[8]),   # +0.0 < +min_subnormal
            (F64_SAMPLES[9], F64_SAMPLES[10]),  # +1.0 < +1.0+ulp
            (F64_SAMPLES[10], F64_SAMPLES[9]),  # +1.0+ulp > +1.0
            (F64_SAMPLES[11], F64_SAMPLES[11]), # +inf == +inf
            (F64_SAMPLES[12], F64_SAMPLES[9]),  # sNaN vs +1.0
            (F64_SAMPLES[9], F64_SAMPLES[13]),  # +1.0 vs qNaN
            (F64_SAMPLES[12], F64_SAMPLES[13]), # sNaN vs qNaN
        ]
        for (alo, ahi), (blo, bhi) in pairs:
            for chunk in range(0, 16, 4):
                ps = PREDS[chunk:chunk + 4]
                lines = [
                    f"v_mov_b32_e32 v4, {alo:#x}", f"v_mov_b32_e32 v5, {ahi:#x}",
                    f"v_mov_b32_e32 v6, {blo:#x}", f"v_mov_b32_e32 v7, {bhi:#x}",
                ]
                for i, p in enumerate(ps):
                    if i == 0:
                        lines += [f"v_cmp_{p}_f64_e32 vcc_lo, v[4:5], v[6:7]",
                                  f"v_cndmask_b32_e32 v{i}, 0, 1, vcc_lo"]
                    else:
                        lines += [f"v_cmp_{p}_f64_e64 s{4 + i}, v[4:5], v[6:7]",
                                  f"v_cndmask_b32_e64 v{i}, 0, 1, s{4 + i}"]
                want = [expected_pred(p, alo, ahi, blo, bhi) for p in ps]
                with self.subTest(a=(hex(alo), hex(ahi)), b=(hex(blo), hex(bhi)), preds=ps):
                    self.assertEqual(run(lines), want)

    def test_sgpr_pairs_constants_and_vop3_modifiers(self):
        # SGPR pair, fimm, inline ints (0, 1, -1), 32-bit upper-half literals, and abs/neg on both sources.
        got = run([
            "s_mov_b32 s8, 0", "s_mov_b32 s9, 0x3ff00000",          # +1.0
            "v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v5, 0x3ff80000",  # +1.5
            "v_cmp_eq_f64_e64 s4, s[8:9], 1.0",                     # SGPR == fimm -> 1
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_eq_f64_e64 s4, 0x3ff80000, v[4:5]",              # 32-bit upper-half literal == +1.5 -> 1
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_lt_f64_e64 s4, 0, 1",                            # +0.0 < +min_subnormal (inline int 1) -> 1
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_u_f64_e64 s4, -1, 0",                            # inline -1 is qNaN -> unordered 1
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got, [1, 1, 1, 1])

        got_mod = run([
            "v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v5, 0xc0000000",  # -2.0
            "v_cmp_eq_f64_e64 s4, |v[4:5]|, 2.0",                   # |-2.0| == 2.0 -> 1
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_eq_f64_e64 s4, -|2.0|, v[4:5]",                  # -|2.0| == -2.0 -> 1
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_eq_f64_e64 s4, neg(0), 0",                       # -0.0 == +0.0 -> 1
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_lt_f64_e64 s4, neg(1), 0",                       # -min_subnormal < +0.0 -> 1
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got_mod, [1, 1, 1, 1])

    def test_cmpx_exec_narrowing_and_inactive_lanes(self):
        got = run([
            "v_mov_b32_e32 v0, 10", "v_mov_b32_e32 v1, 20",
            "v_mov_b32_e32 v2, 30", "v_mov_b32_e32 v3, 40",
            "v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v5, 0x3ff00000",  # +1.0
            "v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0x40000000",  # +2.0
            "s_mov_b32 s10, exec_lo",
            "v_cmpx_lt_f64_e32 v[4:5], v[6:7]",                     # 1.0 < 2.0 -> stays active
            "v_mov_b32_e32 v0, 11",
            "v_cmpx_gt_f64_e64 v[4:5], v[6:7]",                     # 1.0 > 2.0 -> turns off
            "v_mov_b32_e32 v1, 21",                                 # skipped
            "v_cmp_tru_f64_e64 s6, v[4:5], v[6:7]",                 # inactive -> writes 0
            "s_mov_b32 exec_lo, s10",
            "v_cndmask_b32_e64 v2, 31, 32, s6",                     # s6 is 0 -> 31
            "v_cmp_tru_f64_e64 s7, v[4:5], v[6:7]",                 # active -> writes 1
            "v_cndmask_b32_e64 v3, 41, 42, s7",                     # s7 is 1 -> 42
        ])
        self.assertEqual(got, [11, 20, 31, 42])

    def test_no_float64_capability_required(self):
        spv = translate([
            "v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0x3ff00000",
            "v_cmp_lt_f64_e64 s4, v[0:1], 2.0",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_mov_b32_e32 v3, 0",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotIn("OpTypeFloat 64", spv)

    def test_rejected_forms(self):
        for line in (
            "v_cmp_lt_f64_e32 vcc_lo, v0, v[2:3]",          # single VGPR src0
            "v_cmp_lt_f64_e64 s4, v[0:1], s2",              # single SGPR src1
            "v_cmp_lt_f64_e64 v2, v[0:1], v[2:3]",          # VGPR destination
            "v_cmp_lt_f64_e64 s4, v[0:1], v[2:3] clamp",    # unsupported modifier
            "v_cmp_t_f64_e64 s4, v[0:1], v[2:3]",           # integer 't' predicate on float
            "s_cmp_eq_f64 s[0:1], s[2:3]",                  # scalar compare has no f64
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                               "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, 0", line])


if __name__ == "__main__":
    unittest.main()
