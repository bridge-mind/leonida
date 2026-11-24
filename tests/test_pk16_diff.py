"""tools/pk16_diff.py: the packed and 16-bit integer reference against hand-worked values, a short seeded fuzz run,
and canaries. Standard library only; no GPU."""
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "pk16_diff.py"
sys.path.insert(0, str(TOOL.parent))
import pk16_diff as d  # noqa: E402

M = d.M32


def prefix(v=None, s=None):
    return ([f"v_mov_b32_e32 v{r}, 0x{(v or {}).get(r, 0):x}" for r in d.VREGS]
            + [f"s_mov_b32 s{r}, 0x{(s or {}).get(r, 0):x}" for r in d.SREGS])


def run(*body, v=None, s=None, regs=(1, 2, 3, 4)):
    """The reference's four exported words; v / s map register -> seed value (others are 0)."""
    return d.reference(prefix(v, s) + list(body) + d.observe(list(regs)))


class Reference(unittest.TestCase):
    def test_packed_halves_follow_op_sel_and_op_sel_hi(self):
        v = {2: 0x00050003, 3: 0x00070002}
        self.assertEqual(run("v_pk_add_u16 v1, v2, v3", v=v)[0], 0x000C0005)
        # low result from (hi v2, lo v3) = 5 + 2, high result from (lo v2, lo v3) = 3 + 2
        self.assertEqual(run("v_pk_add_u16 v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]", v=v)[0], 0x00050007)
        self.assertEqual(run("v_pk_mul_lo_u16 v1, v2, v3", v={2: 0x01000100, 3: 0x01000101})[0], 0x00000100)
        self.assertEqual(run("v_pk_max_i16 v1, v2, v3", v={2: 0x7FFF8000, 3: 0x80000001})[0], 0x7FFF0001)
        self.assertEqual(run("v_pk_min_u16 v1, v2, v3", v={2: 0x7FFF8000, 3: 0x80000001})[0], 0x7FFF0001)

    def test_clamp_saturates_and_its_absence_wraps(self):
        v = {2: 0x7FFFFFFF, 3: 0x00010001}
        self.assertEqual(run("v_pk_add_i16 v1, v2, v3", v=v)[0], 0x80000000)
        self.assertEqual(run("v_pk_add_i16 v1, v2, v3 clamp", v=v)[0], 0x7FFF0000)    # 0x7fff+1 sat; -1+1 = 0
        self.assertEqual(run("v_pk_add_u16 v1, v2, v3 clamp", v=v)[0], 0x8000FFFF)  # 0xffff+1 sat
        self.assertEqual(run("v_pk_sub_u16 v1, v3, v2 clamp", v=v)[0], 0)
        self.assertEqual(run("v_pk_mad_i16 v1, v2, v3, v3 clamp", v={2: 0x80008000, 3: 0xFFFFFFFF})[0], 0x7FFF7FFF)
        self.assertEqual(run("v_mad_u16 v1, v2, v2, 1 clamp", v={2: 0x100})[0], 0xFFFF)
        self.assertEqual(run("v_sub_nc_i16 v1, v2, v3 clamp", v={2: 0x8000, 3: 1})[0], 0x8000)

    def test_packed_shifts_use_the_low_four_bits(self):
        self.assertEqual(run("v_pk_lshlrev_b16 v1, v2, v3", v={2: 0x00110001, 3: 0x00030003})[0], 0x00060006)
        self.assertEqual(run("v_pk_ashrrev_i16 v1, v2, v3", v={2: 0x000F0004, 3: 0x8000F000})[0], 0xFFFFFF00)

    def test_16bit_ops_keep_the_other_half(self):
        v = {1: 0xAAAABBBB, 2: 0x00050003, 3: 0x00070002}
        self.assertEqual(run("v_add_nc_u16 v1, v2, v3", v=v)[0], 0xAAAA0005)
        # op_sel:[1,1,1]: high halves in, result to the high half, low half kept
        self.assertEqual(run("v_add_nc_u16 v1, v2, v3 op_sel:[1,1,1]", v=v)[0], 0x000CBBBB)
        self.assertEqual(run("v_max_u16_e64 v1, v2, v3 op_sel:[0,1,0]", v=v)[0], 0xAAAA0007)
        self.assertEqual(run("v_med3_i16 v1, v2, -1, v3 op_sel:[0,0,0,1]", v=v)[0], 0x0002BBBB)
        self.assertEqual(run("v_lshrrev_b16_e64 v1, 17, v2", v=v)[0], 0xAAAA0001)

    def test_constants_are_their_32_bit_value(self):
        # measured on gfx1036: an inline constant is split like a register; a float constant is its f32 bits; a
        # 16-bit op's literal has 16 bits
        self.assertEqual(run("v_add_nc_u16 v1, 1.0, 0 op_sel:[1,0,0]")[0], 0x3F80)
        self.assertEqual(run("v_add_nc_u16 v1, 1.0, 0")[0], 0)
        self.assertEqual(run("v_add_nc_u16 v1, -16, 0 op_sel:[1,0,0]")[0], 0xFFFF)
        self.assertEqual(run("v_max_u16_e64 v1, 0.15915494, 0")[0], 0xF983)
        self.assertEqual(run("v_add_nc_u16 v1, 0xffff, 0 op_sel:[1,0,0]")[0], 0)
        self.assertEqual(run("v_pk_add_u16 v1, 0x12345678, 0")[0], 0x12345678)
        self.assertEqual(run("v_pk_add_u16 v1, 1.0, 0")[0], 0x3F800000)


class Generator(unittest.TestCase):
    def lines(self, known=False, literals=True, n=3000, seed=1):
        real = d.KNOWN, d.LITERALS
        d.KNOWN, d.LITERALS = known, literals
        try:
            gen = d.Gen(d.random.Random(seed))
            return [line for _ in range(n) for line in gen.op(d.ALL_OPS)]
        finally:
            d.KNOWN, d.LITERALS = real

    @staticmethod
    def split(line):
        op, _, rest = line.partition(" ")
        args, mods = d.parse(rest)
        return op, args[1:], mods

    def test_every_op_is_generated_with_its_modifiers(self):
        lines = self.lines()
        seen = {re.sub(r"_e64$", "", line.split()[0]) for line in lines}
        self.assertEqual(set(d.ALL_OPS) - seen, set())
        for word in (" op_sel:", " op_sel_hi:", " clamp"):
            self.assertTrue(any(word in line for line in lines), word)
        for line in lines:
            op, srcs, mods = self.split(line)
            base = op[:-4] if op.endswith("_e64") else op
            self.assertTrue(op.endswith("_e64") == (base in d.E64), line)
            table = d.OPS_PK if base in d.OPS_PK else d.OPS_16
            n, _, cl = table[base]
            self.assertEqual(len(srcs), n, line)
            if "clamp" in mods:
                self.assertIsNotNone(cl, line)
            self.assertEqual(len(mods.get("op_sel", [0] * (n if base in d.OPS_PK else n + 1))),
                             n if base in d.OPS_PK else n + 1, line)

    def test_sources_respect_the_constant_bus(self):
        for line in self.lines():
            _, srcs, _ = self.split(line)
            lits = [x for x in srcs if x.startswith("0x")]
            self.assertLessEqual(len(lits), 1, line)
            self.assertLessEqual(len(set(x for x in srcs if x.startswith("s")) | set(lits)), 2, line)
            if not line.startswith("v_pk_"):
                self.assertTrue(all(int(x, 16) <= 0xFFFF for x in lits), line)
        self.assertFalse(any("0x" in line for line in self.lines(literals=False)))

    def test_known_divergences_are_left_out_by_default(self):
        def const_hi(lines):
            for line in lines:
                op, srcs, mods = self.split(line)
                if op not in d.OPS_PK or op in d.PK_SHIFTS:
                    continue
                n = len(srcs)
                sel, hi = mods.get("op_sel", [0] * n), mods.get("op_sel_hi", [1] * n)
                if any(d.is_const(x) and (sel[i] or hi[i]) for i, x in enumerate(srcs)):
                    return True
            return False
        self.assertFalse(const_hi(self.lines()))
        self.assertTrue(const_hi(self.lines(known=True)))


class Translated(unittest.TestCase):
    def test_harness_canaries(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(d.self_test())

    def test_short_seeded_run_matches(self):
        counts, failures = d.run_suite(80, 5, 5, d.ALL_OPS)
        self.assertEqual(failures, [], failures[:1])
        self.assertEqual(counts["skip"], {})
        self.assertEqual(counts["pass"], 80)

    def test_hand_cases_match_the_translator(self):
        seeds = {1: 0xAAAABBBB, 2: 0x7FFF8000, 3: 0x80000001, 4: 0x00050003, 5: 0x00070002}
        cases = ["v_pk_add_i16 v1, v2, v3 clamp", "v_pk_sub_u16 v1, v3, v2 op_sel:[1,0] op_sel_hi:[0,1] clamp",
                 "v_pk_mad_i16 v1, v2, v3, s4 clamp", "v_pk_lshlrev_b16 v1, 17, v4",
                 "v_pk_mad_u16 v1, v4, 3, v5 op_sel_hi:[1,0,1]", "v_add_nc_u16 v1, v4, v5 op_sel:[1,1,1]",
                 "v_mad_i16 v1, v2, v3, -1 op_sel:[1,0,0,1] clamp", "v_med3_u16 v1, v2, 0x1234, v4 op_sel:[0,0,1,0]",
                 "v_max_i16_e64 v1, v2, v3 op_sel:[1,1,0]", "v_add_nc_u16 v1, 1.0, s5 op_sel:[0,0,1]",
                 "v_ashrrev_i16_e64 v1, -16, v2", "v_sub_nc_i16 v1, v2, v3 clamp"]
        for case in cases:
            with self.subTest(case=case):
                self.assertIsNone(d.check(prefix(seeds, {4: 0x00030004, 5: 0xFFFF0001}) + [case]
                                          + d.observe([1, 2, 3, 4])))

    def test_a_wrong_translation_is_caught(self):
        lines = prefix({2: 5}) + ["v_pk_add_u16 v1, v2, 1 op_sel_hi:[1,0]"] + d.observe([1, 2, 3, 4])
        self.assertIsNone(d.check(lines))
        want = d.reference(lines)
        real = d.translated
        try:
            d.translated = lambda _lines: [want[0] ^ 0x10000] + want[1:]
            self.assertIn("word 0 (v1)", d.check(lines))
        finally:
            d.translated = real

    def test_shrinking_keeps_the_instruction_that_matters(self):
        lines = d.generate(d.random.Random(2), 5, ["v_pk_add_u16", "v_add_nc_u16", "v_med3_i16"])
        body = lines[d.N_SEED:-d.N_OBS]
        target = body[len(body) // 2]
        real = d.translated
        try:
            d.translated = lambda l: [w ^ (1 if target in l else 0) for w in d.reference(l)]
            small = d.shrink(lines)
        finally:
            d.translated = real
        kept = [line for line in small if not line.startswith(("v_mov_b32_e32", "s_mov_b32"))]
        self.assertEqual(kept, [target])


class Cli(unittest.TestCase):
    def main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = d.main(list(argv))
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue()

    def test_exit_codes(self):
        rc, out = self.main("--list-ops")
        self.assertEqual(rc, 0)
        self.assertIn("v_pk_mad_i16", out)
        self.assertEqual(self.main("--ops", "v_nope")[0], 2)
        self.assertEqual(self.main("--oracle", "gpu")[0], 2)
        rc, out = self.main("--programs", "5", "--seed", "1")
        self.assertEqual(rc, 0, out)
        real = d.translated
        try:
            d.translated = lambda l: [w ^ 1 for w in d.reference(l)]
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_pk_max_u16")
        finally:
            d.translated = real
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out)

    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        def refuse(lines, *args):
            raise d.Skip("unsupported form")
        real = d.translated
        try:
            d.translated = refuse
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_pk_max_u16")
        finally:
            d.translated = real
        self.assertEqual(rc, 2, out)
        self.assertIn("no program ran: every one was skipped", out)

    def test_hardware_oracle_without_the_tools_exits_77(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOL.parent / "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", str(TOOL), "--oracle", "hw", "--programs", "3"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)


if __name__ == "__main__":
    unittest.main()
