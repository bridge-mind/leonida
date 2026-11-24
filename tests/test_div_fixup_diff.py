"""tools/div_fixup_diff.py: the reference against the words measured on RDNA2 hardware (gfx1036), the generator's
encoding rules, the sweep and edge grid, and short seeded differential runs. Standard library only; no GPU, no game
files.

The f32 rows are tests/semantic/test_alu_div_fixup_f32.py's MEASURED list, imported as is. The f16 and f64 rows below
are the ones tests/semantic/test_alu_div_fixup.py marks as gfx1036 measurements, plus the cases of its
test_all_special_and_normal_cases, copied by value (they live inside a test method there).
"""
import io
import os
import random
import re
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import div_fixup_diff as d  # noqa: E402  (puts tests/semantic on sys.path)
import test_alu_div_fixup_f32 as mf  # noqa: E402  (the gfx1036-measured v_div_fixup_f32 words)

F16, F32, F64 = (d.FMT[op] for op in d.OPS)
H_ONE, H_NEG2, H_HALF, H_INF, H_NINF = 0x3C00, 0xC000, 0x3800, 0x7C00, 0xFC00
D_ONE, D_NEG2, D_THREE = 0x3FF0000000000000, 0xC000000000000000, 0x4008000000000000
D_INF, D_NINF = 0x7FF0000000000000, 0xFFF0000000000000
# (S0, S1, S2, result) from tests/semantic/test_alu_div_fixup.py
F16_ROWS = (
    (H_HALF, H_ONE, H_ONE, H_HALF), (H_HALF, H_NEG2, H_ONE, H_HALF | 0x8000), (H_ONE, 0x7C01, 0x7E01, 0x7E01),
    (H_ONE, 0x7C01, H_ONE, 0x7C01), (H_ONE, H_ONE, 0x7C01, 0x7C01), (H_ONE, 0, 0x8000, 0xFE00),
    (H_ONE, H_INF, H_NINF, 0xFE00), (H_ONE, H_NINF, H_ONE, 0x8000), (H_ONE, H_NEG2, 0x8000, 0),
    (H_ONE, 0x8000, H_ONE, H_NINF), (H_ONE, H_NEG2, H_NINF, H_INF), (0x7E35, H_ONE, H_ONE, H_INF),
    (0x7E35, H_NEG2, H_ONE, H_NINF),
    (H_ONE, 0x7800, 0x0400, H_ONE),            # gfx1036: e2 - e1 = -29 keeps S0 (no f16 cutoff)
    (H_ONE, 0x0400, 0x7800, H_ONE),            # gfx1036: e2 - e1 = +29 keeps S0
)
F64_ROWS = (
    (D_THREE, D_ONE, D_ONE, D_THREE), (D_THREE, D_NEG2, D_ONE, D_THREE | 1 << 63),
    (D_ONE, 0x7FF0000000000001, 0x7FF8000000000002, 0x7FF8000000000002),
    (D_ONE, 0x7FF0000000000001, D_ONE, 0x7FF0000000000001), (D_ONE, D_ONE, 0x7FF0000000000001, 0x7FF0000000000001),
    (D_ONE, 0, 1 << 63, 0xFFF8000000000000), (D_ONE, D_INF, D_NINF, 0xFFF8000000000000),
    (D_ONE, D_NINF, D_ONE, 1 << 63), (D_ONE, D_NEG2, 1 << 63, 0), (D_ONE, 1 << 63, D_ONE, D_NINF),
    (D_ONE, D_NEG2, D_NINF, D_INF), (0xFFF8123456789ABC, 0x4000000000000000, D_ONE, D_INF),
    (0x7FF8123456789ABC, D_NEG2, D_ONE, D_NINF),
    (D_ONE, 2046 << 52, 971 << 52, D_ONE),     # gfx1036: e2 - e1 = -1075 keeps S0
    (D_ONE, 2046 << 52, 970 << 52, 0),         # gfx1036: e2 - e1 = -1076 gives +0
    (D_ONE, 1123 << 52, 1 << 52, 0),           # gfx1036: 2^-1022 / 2^100 gives +0
    (D_ONE, 1 << 52, 1025 << 52, D_ONE),       # gfx1036: e2 - e1 = +1024 keeps S0 (no overflow cutoff)
)
NONE = [None, None]


def run(line, **regs):
    """(the Machine, its v8 word) after the given v<n>/s<n> words and ``line``."""
    mc = d.Machine()
    mc.write_word(8, 0x5A5A5A5A)
    for name, w in regs.items():
        if name[0] == "s":
            mc.s[int(name[1:])] = w
        else:
            mc.write_word(int(name[1:]), w)
    d.step(line, mc)
    return mc, mc.v[8]


def pair(n, x):
    return {f"v{n}": x & 0xFFFFFFFF, f"v{n + 1}": x >> 32}


class Reference(unittest.TestCase):
    def test_f32_matches_every_measured_gfx1036_row(self):
        self.assertEqual(len(mf.MEASURED), 73)
        for insn, v1, v2, v3, want in mf.MEASURED:
            with self.subTest(insn=insn, s=(hex(v1), hex(v2), hex(v3))):
                mc = d.Machine()
                for r, w in ((1, v1), (2, v2), (3, v3), (4, 0xDEADBEEF)):
                    mc.write_word(r, w)
                d.step(insn, mc)
                self.assertIsNone(d.compare([mc.v[4], 0, 0, 0], [mc.state(4), NONE, NONE, NONE], [want, 0, 0, 0]))

    def test_f16_and_f64_semantic_rows(self):
        for s0, s1, s2, want in F16_ROWS:
            with self.subTest(f16=(hex(s0), hex(s1), hex(s2))):
                self.assertEqual(d.fixup(F16, s0, s1, s2)[1], want)
                _, v8 = run("v_div_fixup_f16 v8, v12, v13, v14", v12=0xBEEF0000 | s0, v13=0xCAFE0000 | s1,
                            v14=0xDEAD0000 | s2)
                self.assertEqual(v8, 0x5A5A0000 | want)
        for s0, s1, s2, want in F64_ROWS:
            with self.subTest(f64=(hex(s0), hex(s1), hex(s2))):
                mc, lo = run("v_div_fixup_f64 v[8:9], v[10:11], v[12:13], v[14:15]",
                             **pair(10, s0), **pair(12, s1), **pair(14, s2))
                self.assertEqual(lo | mc.v[9] << 32, want)

    def test_rule_order_and_cutoffs(self):
        # f32: e2 - e1 = -151 gives a zero even against a NaN or inf S0; -150 does not
        self.assertEqual(d.fixup(F32, 0x7FC00055, 0x64000000, 0x00800000), (5, 0))
        self.assertEqual(d.fixup(F32, 0x7F800000, 0xE4000000, 0x00800000), (5, 0x80000000))
        self.assertEqual(d.fixup(F32, 0x7FC00055, 0x4B800000, 0x00800000), (6, 0x7F800000))
        # f64 likewise at -1076 / -1075, and the NaN S0 rule comes after the cutoff
        self.assertEqual(d.fixup(F64, 0x7FF8000000000001, 2046 << 52, 970 << 52), (5, 0))
        self.assertEqual(d.fixup(F64, 0x7FF8000000000001, 2046 << 52, 971 << 52), (6, D_INF))
        # f16: no cutoff at all
        self.assertEqual(d.fixup(F16, 0x7E00, 0x7BFF, 0x0001), (6, H_INF))
        # f32 S1/S2 denormals are zeros (rules 3 to 5), an S0 denormal is kept; f16/f64 denormals are not zeros
        self.assertEqual(d.fixup(F32, 0x00000003, 0x80000001, 0x3F800000), (4, 0xFF800000))
        self.assertEqual(d.fixup(F16, 0x0003, 0x8001, H_ONE), (7, 0x8003))
        self.assertEqual(d.fixup(F64, 3, (1 << 63) | 1, D_ONE)[0], 7)

    def test_f64_denormal_exponent_is_not_judged_inside_the_band(self):
        # S2 = 2^-1074: its field 0, 1, or normalized -51; the readings agree below e1 = 1025 and above 1076
        self.assertEqual(d.fixup(F64, D_ONE, 1000 << 52, 1), (7, D_ONE))
        self.assertEqual(d.fixup(F64, D_ONE, 1100 << 52, 1), (5, 0))
        for e1 in (1025, 1050, 1076):      # 1 - e1 > -1076 >= -51 - e1 here
            with self.assertRaises(d.NotJudged, msg=e1):
                d.fixup(F64, D_ONE, e1 << 52, 1)

    def test_f16_op_sel_and_inline_constants(self):
        # op_sel:[1,0,1,1]: S0 and S2 from the high halves, the result to the high half, the low half kept
        _, v8 = run("v_div_fixup_f16 v8, v12, -v13, v14 op_sel:[1,0,1,1]", v12=H_HALF << 16, v13=H_NEG2,
                    v14=H_ONE << 16)
        self.assertEqual(v8, H_HALF << 16 | 0x5A5A)
        # a float inline constant is its f16 value; an integer one is the low half of its word (-1: a NaN)
        self.assertEqual(run("v_div_fixup_f16 v8, 0.5, 1.0, -2.0")[1], 0x5A5A0000 | 0xB800)
        self.assertEqual(run("v_div_fixup_f16 v8, 0.5, 1.0, -1")[1], 0x5A5AFFFF)
        self.assertEqual(run("v_div_fixup_f16 v8, 0x12343c00, 1.0, 1.0")[1], 0x5A5A3C00)
        _, v8 = run("v_div_fixup_f16 v8, v12, v13, v14 clamp", v12=0x4000, v13=H_ONE, v14=H_ONE)
        self.assertEqual(v8, 0x5A5A3C00)

    def test_f64_sources(self):
        # an integer inline constant is sign-extended (-1 is a NaN), 1/(2 pi) is the binary64 value
        mc, lo = run("v_div_fixup_f64 v[8:9], 1.0, 1.0, -1")
        self.assertEqual(lo | mc.v[9] << 32, (1 << 64) - 1)
        mc, lo = run("v_div_fixup_f64 v[8:9], 0.15915494309189532, -2.0, 1.0")
        self.assertEqual(lo | mc.v[9] << 32, 0xBFC45F306DC9C882)
        # abs and neg act on bit 63 only; SGPR pairs and odd VGPR pairs
        mc, lo = run("v_div_fixup_f64 v[9:10], -|v[11:12]|, s[4:5], v[13:14]", **pair(11, D_THREE),
                     **pair(13, D_ONE), s4=0, s5=0x40000000)
        self.assertEqual((lo, mc.v[9], mc.v[10]), (0x5A5A5A5A, 0, 0x40080000))

    def test_f32_output_modifiers(self):
        self.assertEqual(d.out_f32(0x7F7FFFFF, ["mul:2"]), (0x7F800000, None))
        self.assertEqual(d.out_f32(0x3F800000, ["div:2"]), (0x3F000000, None))
        self.assertEqual(d.out_f32(0xFF800000, ["mul:4"]), (0xFF800000, None))
        self.assertEqual(d.out_f32(0x3E800000, ["clamp", "mul:4"]), (0x3F800000, None))
        self.assertEqual(d.out_f32(0x80000000, ["clamp", "mul:2"]), (0, None))     # -0 under clamp: +0
        self.assertEqual(d.out_f32(0x80000000, ["clamp"]), (0, None))
        self.assertEqual(d.out_f32(0x807FFFFF, ["clamp"]), (0, None))
        self.assertEqual(d.out_f32(0x7F800001, ["clamp", "div:2"]), (0, None))
        self.assertEqual(d.out_f32(0x7F800001, ["mul:2"]), (0x7FC00000, "w32"))
        for w, mods in ((0x80000000, ["mul:2"]), (0x00000001, ["mul:2"]), (0x00800000, ["div:2"]),
                        (0x00000001, ["clamp"])):
            with self.assertRaises(d.NotJudged, msg=(hex(w), mods)):
                d.out_f32(w, mods)

    def test_unmodelled_nan_payloads(self):
        mc, _ = run("v_div_fixup_f32 v8, v12, v13, v14 mul:2", v12=H_ONE, v13=0x3F800000, v14=0xFFC00005)
        self.assertEqual(mc.state(8), ["w32", "w32"])
        d.step("v_div_fixup_f32 v9, 1.0, 1.0, -|v8|", mc)            # an unmodelled NaN passes through as one
        self.assertEqual(mc.state(9), ["w32", "w32"])
        d.step("v_div_fixup_f32 v10, v8, 1.0, 1.0", mc)              # as S0 it gives +inf whatever its payload
        self.assertEqual((mc.v[10], mc.state(10)), (0x7F800000, NONE))
        d.step("v_div_fixup_f32 v11, 1.0, 1.0, v8 clamp", mc)        # clamp gives +0
        self.assertEqual((mc.v[11], mc.state(11)), (0, NONE))
        for line in ("v_div_fixup_f16 v10, v8, 1.0, 1.0", "v_div_fixup_f64 v[10:11], v[8:9], 1.0, 1.0"):
            with self.assertRaises(d.NotJudged, msg=line):
                d.step(line, mc)

    def test_compare(self):
        self.assertIsNone(d.compare([0, 0, 0, 0], [["w32", "w32"], NONE, NONE, NONE], [0xFFC00001, 0, 0, 0]))
        self.assertIn("f32 NaN", d.compare([0, 0, 0, 0], [["w32", "w32"], NONE, NONE, NONE], [0x7F800000, 0, 0, 0]))
        self.assertIsNone(d.compare([0x12345678, 0, 0, 0], [["w32", None], NONE, NONE, NONE], [0x1234ABCD, 0, 0, 0]))
        self.assertIn("expected 0x00000000, actual 0x80000000",
                      d.compare([0, 0, 0, 0], [NONE] * 4, [0, 0x80000000, 0, 0]))

    def test_harness_errors(self):
        for line in ("v_div_fixup_f32 v8, v9, v10, v11 op_sel:[1,0,0,0]", "v_div_fixup_f16 v8, v9, v10, v11 mul:2 div:2",
                     "v_div_fixup_f64 v[8:9], v[10:11], v[10:11], v[10:11] clamp",
                     "v_div_fixup_f64 v[8:9], 0x3ff00000, v[10:11], v[10:11]", "v_div_fixup_f64 v8, 1.0, 1.0, 1.0",
                     "v_div_fixup_f16 v8, 1.0, v9, v9 op_sel:[1,0,0,0]", "v_div_fixup_f32 v8, 65, v9, v9",
                     "v_div_scale_f32 v8, vcc_lo, v9, v9, v9"):
            with self.assertRaises(d.HarnessError, msg=line):
                run(line, v9=0x3F800000, v10=0x3F800000, v11=0x3FF00000)


class OutputMultipliers(unittest.TestCase):
    """#6314 measured on gfx1036: under denorm_mode_16_64 = 3 an f16 or binary64 result ignores mul:2/mul:4/div:2,
    and an f16 clamp still applies."""

    def test_f16_result_is_not_scaled(self):
        for s0, s1, s2 in ((0x3E00, 0x3C00, 0x3C00), (0x7BFF, 0x3C00, 0x3C00), (0x0001, 0x3C00, 0x3C00),
                           (0x3C00, 0x0000, 0x3C00), (0x3C00, 0x7C01, 0x3C00), (0x3800, 0xBC00, 0x3C00)):
            regs = {"v12": s0 | 0xABCD0000, "v13": s1, "v14": s2}
            for sel in ("", " op_sel:[0,0,0,1]"):
                plain = run(f"v_div_fixup_f16 v8, v12, v13, v14{sel}", **regs)[1]
                clamped = run(f"v_div_fixup_f16 v8, v12, v13, v14{sel} clamp", **regs)[1] if s1 != 0x7C01 else None
                for om in ("mul:2", "mul:4", "div:2"):
                    with self.subTest(s0=hex(s0), s1=hex(s1), sel=sel, om=om):
                        self.assertEqual(run(f"v_div_fixup_f16 v8, v12, v13, v14{sel} {om}", **regs)[1], plain)
                        if clamped is not None:
                            self.assertEqual(run(f"v_div_fixup_f16 v8, v12, v13, v14{sel} clamp {om}", **regs)[1],
                                             clamped)
        # 1.5 under clamp div:2: 1.0 (clamped, not halved to 0.75)
        self.assertEqual(run("v_div_fixup_f16 v8, v12, v13, v14 clamp div:2", v12=0x3E00, v13=0x3C00, v14=0x3C00)[1]
                         & 0xFFFF, 0x3C00)

    def test_f64_result_is_not_scaled(self):
        for s0 in (0x3FF8000000000000, 0x7FEFFFFFFFFFFFFF, 0x0010000000000000, 3, 1 << 63):
            regs = {"v12": s0 & d.M32, "v13": s0 >> 32, "v14": 0, "v15": 0x3FF00000}
            mc, _ = run("v_div_fixup_f64 v[8:9], v[12:13], v[14:15], v[14:15]", **regs)
            plain = (mc.v[8], mc.v[9])
            self.assertEqual(plain, (s0 & d.M32, (s0 >> 32) & 0x7FFFFFFF))    # |S0| with sign(S1) ^ sign(S2) = +
            for om in ("mul:2", "mul:4", "div:2"):
                with self.subTest(s0=hex(s0), om=om):
                    mc, _ = run(f"v_div_fixup_f64 v[8:9], v[12:13], v[14:15], v[14:15] {om}", **regs)
                    self.assertEqual((mc.v[8], mc.v[9]), plain)


class Generator(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(d.program(random.Random(5), 6), d.program(random.Random(5), 6))
        self.assertEqual(d.grid_cases(), d.grid_cases())

    def test_ops_subset_is_respected(self):
        lines, used = d.program(random.Random(2), 20, ["v_div_fixup_f64"])
        self.assertEqual(set(used), {"v_div_fixup_f64"})
        self.assertTrue(all(line.split()[0] in ("v_mov_b32", "s_mov_b32", "v_div_fixup_f64") for line in lines))

    def test_sweep_covers_every_encodable_kind_combination(self):
        by_op = Counter(c[0] for c in d.sweep_cases())
        # 5^3 kind triples, less 13 with two or three literals and 4 with three constant-bus reads
        self.assertEqual(by_op, {"v_div_fixup_f16": 108, "v_div_fixup_f32": 108, "v_div_fixup_f64": 63})
        for op in d.OPS:
            kinds = {k for c in d.sweep_cases([op]) for k in c[1]}
            self.assertEqual(kinds, set(d.KINDS64 if op.endswith("f64") else d.KINDS), op)

    def test_programs_are_encodable(self):
        rng = random.Random(9)
        seen = Counter()
        for _ in range(500):
            lines, _ = d.program(rng, 6)
            for line in lines:
                op, args, mods = d.split_line(line)
                if op in ("v_mov_b32", "s_mov_b32"):
                    continue
                srcs = [a.strip("-|") for a in args[1:]]
                lits = [a for a in srcs if a.startswith("0x")]
                bus = [a for a in srcs if a.startswith("s")]
                self.assertLessEqual(len(lits), 1, line)
                self.assertLessEqual(len(lits) + len(bus), 2, line)
                sel = d._opsel(mods)
                for i, a in enumerate(srcs):
                    if sel[i]:
                        self.assertRegex(a, r"^[sv]\d+$", line)
                if op == "v_div_fixup_f64":
                    self.assertFalse(lits, line)
                    self.assertLessEqual(set(mods), {"mul:2", "mul:4", "div:2"}, line)
                    seen[op + " " + " ".join(mods)] += 1
                    self.assertRegex(args[0], r"^v\[(8:9|9:10|10:11)\]$", line)
                    seen["f64 odd pair"] += any(re.match(r"v\[(9|11|13):", a) for a in srcs)
                    seen["f64 sgpr pair"] += any(a.startswith("s[") for a in srcs)
                else:
                    self.assertIn(args[0], ("v8", "v9", "v10", "v11"), line)
                    seen[op + " " + " ".join(mods)] += 1
        for want in ("f64 odd pair", "f64 sgpr pair", "v_div_fixup_f32 clamp mul:4", "v_div_fixup_f32 div:2",
                     "v_div_fixup_f64 mul:2", "v_div_fixup_f64 mul:4", "v_div_fixup_f64 div:2"):
            self.assertGreater(seen[want], 0, want)
        self.assertTrue(any(k.startswith("v_div_fixup_f16 ") and k.endswith(" clamp mul:4") for k in seen))
        self.assertTrue(any(k.startswith("v_div_fixup_f16 op_sel:") and k.endswith("clamp") for k in seen))

    def test_edge_grid_reaches_every_rule(self):
        cases = d.grid_cases()
        self.assertEqual(Counter(c[0] for c in cases),
                         {"v_div_fixup_f16": 1450, "v_div_fixup_f32": 1397, "v_div_fixup_f64": 1179})
        for op in d.OPS:
            f, rules, unjudged = d.FMT[op], Counter(), 0
            for _, words, mods, _, _ in (c for c in cases if c[0] == op):
                bits = [(x & f.mag if "|" in m else x) ^ (f.sign if m.startswith("-") else 0)
                        for x, m in zip(words, mods)]
                try:
                    rules[d.fixup(f, *bits)[0]] += 1
                except d.NotJudged:
                    unjudged += 1
            want = set(range(8)) - ({5} if op == "v_div_fixup_f16" else set())
            self.assertEqual(set(rules), want, op)
            self.assertEqual(unjudged, 8 if op == "v_div_fixup_f64" else 0, op)
        every = {"", "clamp", "mul:2", "mul:4", "div:2", "clamp mul:2", "clamp mul:4", "clamp div:2"}
        omods = {op: {c[4] for c in cases if c[0] == op} for op in d.OPS}
        self.assertEqual(omods, {"v_div_fixup_f16": every, "v_div_fixup_f32": every,
                                 "v_div_fixup_f64": {"", "mul:2", "mul:4", "div:2"}})
        self.assertTrue(any(c[3][3] for c in cases if c[0] == "v_div_fixup_f16"))
        # the f16 multipliers reach both destination halves
        self.assertEqual({c[3][3] for c in cases if c[0] == "v_div_fixup_f16" and "mul:4" in c[4]}, {0, 1})


class Translated(unittest.TestCase):
    def test_harness_canaries(self):
        with redirect_stdout(io.StringIO()) as out:
            ok = d.self_test()
        self.assertTrue(ok, out.getvalue())

    def test_short_seeded_run_matches(self):
        counts, failure = d.run_suite(20, 3, 5)
        self.assertEqual((failure, counts["skip"]), (None, {}))
        self.assertEqual(counts["pass"] + sum(counts["unjudged"].values()), 20)
        self.assertGreater(counts["pass"], 12)

    def test_sweep_matches_for_f64(self):
        counts, failure = d.run_sweep(4, ["v_div_fixup_f64"])
        self.assertEqual((failure, counts["skip"]), (None, {}))
        self.assertGreater(counts["pass"], 110)

    def test_edge_grid_matches_for_f32(self):
        counts, failure = d.run_grid(["v_div_fixup_f32"])
        self.assertEqual((failure, counts["skip"], counts["mismatch"]), (None, {}, 0))
        self.assertGreater(counts["events"]["v_div_fixup_f32"]["exponent cutoff"], 30)

    def test_each_program_is_checked_before_the_next_is_made(self):
        # a run holds one program's text at a time, not the whole --programs list
        made, seen = [], []
        real = d.program, d.sweep_program, d.grid_program, d.run_one

        def count(f):
            return lambda *a: made.append(1) or f(*a)
        try:
            d.program, d.sweep_program, d.grid_program = (count(f) for f in real[:3])
            d.run_one = lambda lines, name, tmp, events: seen.append(len(made)) or ("match", None)
            d.run_suite(5, 1)
            d.run_sweep(1, ["v_div_fixup_f16"])
            d.run_grid(["v_div_fixup_f64"])
        finally:
            d.program, d.sweep_program, d.grid_program, d.run_one = real
        self.assertGreater(len(seen), 5 + 2 * 36)
        self.assertEqual(seen, list(range(1, len(seen) + 1)))

    def test_measured_rows_through_the_translator(self):
        for insn, v1, v2, v3, want in mf.MEASURED[::6]:
            line = insn.replace("v4,", "v8,").replace("v1", "v12").replace("v2", "v13").replace("v3", "v14")
            lines = [f"v_mov_b32 v{r}, 0x5a5a5a5a" for r in d.VGPRS] + [f"s_mov_b32 s{r}, 0" for r in d.SGPRS]
            lines += [f"v_mov_b32 v12, 0x{v1:08x}", f"v_mov_b32 v13, 0x{v2:08x}", f"v_mov_b32 v14, 0x{v3:08x}", line]
            with self.subTest(line=line):
                self.assertEqual(d.reference(lines)[0][0], want)
                self.assertIsNone(d.check(lines))

    def test_output_multipliers_through_the_translator(self):
        lines = [f"v_mov_b32 v{r}, 0x5a5a5a5a" for r in d.VGPRS] + [f"s_mov_b32 s{r}, 0" for r in d.SGPRS]
        lines += ["v_mov_b32 v12, 0x3e00", "v_mov_b32 v13, 0x3c00", "v_mov_b32 v14, 0x3c00", "v_mov_b32 v15, 0x3ff80000",
                  "v_div_fixup_f16 v8, v12, v13, v14 mul:2",
                  "v_div_fixup_f16 v9, v12, v13, v14 op_sel:[0,0,0,1] clamp div:2",
                  "v_div_fixup_f64 v[10:11], v[14:15], v[12:13], v[12:13] mul:4"]
        want = d.reference(lines)[0]
        self.assertEqual(want, [0x5A5A3E00, 0x3C005A5A, 0x3C00, 0x3FF80000])
        self.assertIsNone(d.check(lines))
        real = d.translate_listing
        try:   # a translator that applied mul:2 to the f16 result: 3.0, not 1.5
            d.translate_listing = lambda l, n, t: [0x5A5A4200] + real(l, n, t)[1:]
            self.assertIn("expected 0x5a5a3e00, actual 0x5a5a4200", d.check(lines))
            # ... or mul:4 to the f64 one: 6.0, not 1.5
            d.translate_listing = lambda l, n, t: real(l, n, t)[:3] + [0x40180000]
            self.assertIn("expected 0x3ff80000, actual 0x40180000", d.check(lines))
        finally:
            d.translate_listing = real

    def test_a_wrong_translation_is_caught(self):
        lines = [f"v_mov_b32 v{r}, 0x5a5a5a5a" for r in d.VGPRS] + [f"s_mov_b32 s{r}, 0" for r in d.SGPRS]
        lines += ["v_mov_b32 v12, 0x3f800000", "v_mov_b32 v13, 0x4c000000", "v_mov_b32 v14, 0x00800000",
                  "v_div_fixup_f32 v8, v12, v13, v14"]
        self.assertIsNone(d.check(lines))
        real = d.translate_listing
        try:   # the result a cutoff of -150 would give: S0, not +0
            d.translate_listing = lambda l, n, t: [0x3F800000] + real(l, n, t)[1:]
            self.assertIn("expected 0x00000000, actual 0x3f800000", d.check(lines))
        finally:
            d.translate_listing = real


if __name__ == "__main__":
    unittest.main()
