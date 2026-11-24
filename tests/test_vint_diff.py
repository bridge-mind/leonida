"""tools/vint_diff.py: the integer VALU reference against hand-worked values, a short seeded fuzz run, and canaries.
Standard library only; no GPU."""
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "vint_diff.py"
sys.path.insert(0, str(TOOL.parent))
import vint_diff as d  # noqa: E402


def prefix(v=None, s=None):
    return ([f"v_mov_b32_e32 v{r}, 0x{(v or {}).get(r, 0):x}" for r in d.VREGS]
            + [f"s_mov_b32 s{r}, 0x{(s or {}).get(r, 0):x}" for r in d.SREGS])


def run(*body, v=None, s=None, regs=(1, 2, 3, 4)):
    """The reference's four exported words; v / s map register -> seed value (others are 0)."""
    return d.reference(prefix(v, s) + list(body) + d.observe(list(regs)))


class Reference(unittest.TestCase):
    def test_vop3_only_ops(self):
        self.assertEqual(run("v_add3_u32 v1, v2, s4, 0x10", v={2: 0xFFFFFFF0}, s={4: 4})[0], 4)
        self.assertEqual(run("v_alignbit_b32 v1, v2, v3, 8", v={2: 0x12345678, 3: 0x9ABCDEF0})[0], 0x789ABCDE)
        self.assertEqual(run("v_bfm_b32 v1, 5, 4")[0], 0x1F0)
        self.assertEqual(run("v_bcnt_u32_b32 v1, v2, 3", v={2: 0xF0F0})[0], 11)
        self.assertEqual(run("v_med3_i32 v1, v2, v3, 5", v={2: 0x80000000, 3: 7})[0], 5)
        self.assertEqual(run("v_xad_u32 v1, v2, v3, 1", v={2: 0xFF, 3: 0x0F})[0], 0xF1)
        # saturating packs: (-40000, 40000) -> 0x8000 | 0x7fff << 16; unsigned 0x12345 -> 0xffff
        self.assertEqual(run("v_cvt_pk_i16_i32 v1, v2, v3", v={2: (-40000) & d.M32, 3: 40000})[0], 0x7FFF8000)
        self.assertEqual(run("v_cvt_pk_u16_u32 v1, v2, 7", v={2: 0x12345})[0], 0x0007FFFF)

    def test_bitfield_extract_stops_at_bit_31(self):
        # offset 28, width 8: only bits 28..31 exist, 0x8 sign-extends from bit 3 of the field
        self.assertEqual(run("v_bfe_i32 v1, v2, 28, 8", v={2: 0x80000000})[0], 0xFFFFFFF8)
        self.assertEqual(run("v_bfe_u32 v1, v2, 28, 8", v={2: 0x80000000})[0], 0x8)
        self.assertEqual(run("v_bfe_u32 v1, v2, 4, 0", v={2: M})[0], 0)

    def test_perm_selectors(self):
        # {src0, src1} = 0x80000000_00008000: 8 -> bit 15 (set), 9 -> bit 31, 10 -> bit 47, 11 -> bit 63 (set)
        self.assertEqual(run("v_perm_b32 v1, v2, v3, 0x0b0a0908", v={2: 0x80000000, 3: 0x8000})[0], 0xFF0000FF)
        # 12 -> 0x00, 13 and up -> 0xff, 0..7 -> bytes of src1 then src0
        self.assertEqual(run("v_perm_b32 v1, v2, v3, 0x070d0c00", v={2: 0xAB000000, 3: 0x11})[0], 0xABFF0011)

    def test_24_and_16_bit_multiplies(self):
        self.assertEqual(run("v_mul_hi_i32_i24_e64 v1, v2, 2", v={2: 0x800000})[0], M)       # -2^23 * 2 < 0
        self.assertEqual(run("v_mul_u32_u24_e32 v1, v2, v3", v={2: 0xFF000003, 3: 5})[0], 15)
        # op_sel takes the high halves of the two 16-bit sources: 5 * 2 + 1
        self.assertEqual(run("v_mad_u32_u16 v1, v2, v3, 1 op_sel:[1,0,0,0]", v={2: 0x00050003, 3: 0x00070002})[0],
                         11)
        self.assertEqual(run("v_mad_i32_i16 v1, v2, v3, 0", v={2: 0xFFFF, 3: 3})[0], M - 2)  # -1 * 3

    def test_find_first_bit(self):
        self.assertEqual(run("v_ffbh_i32_e64 v1, v2", v={2: 0xFFFFFFFE})[0], 31)
        self.assertEqual(run("v_ffbh_u32_e32 v1, v2", v={2: 0x00010000})[0], 15)
        self.assertEqual(run("v_ffbl_b32_e32 v1, 0")[0], M)

    def test_constant_sources(self):
        # a float inline constant in an integer op is its f32 bits; a literal goes in src0 of an e32 VOP2
        self.assertEqual(run("v_add_nc_u32_e64 v1, 1.0, 0")[0], 0x3F800000)
        self.assertEqual(run("v_or_b32_e64 v1, 0.15915494, 0")[0], 0x3E22F983)
        self.assertEqual(run("v_xor_b32_e32 v1, 0x12345678, v2", v={2: 0xFF})[0], 0x12345687)
        self.assertEqual(run("v_lshlrev_b32_e64 v1, s4, -16", s={4: 36})[0], 0xFFFFFF00)    # shift by 36 & 31

    def test_compare_and_select(self):
        self.assertEqual(run("v_cmp_lt_i32_e64 s20, v2, v3", "v_cndmask_b32_e64 v1, 1, 2, s20",
                             v={2: 0x80000000, 3: 0})[0], 2)
        self.assertEqual(run("v_cmp_lt_u32_e64 vcc_lo, v2, v3", "v_cndmask_b32_e64 v1, 1, 2, vcc_lo",
                             v={2: 0x80000000, 3: 0})[0], 1)


M = d.M32


class Generator(unittest.TestCase):
    def lines(self, known=False, literals=True, n=3000, seed=1):
        real = d.KNOWN, d.LITERALS
        d.KNOWN, d.LITERALS = known, literals
        try:
            rng = d.random.Random(seed)
            gen = d.Gen(rng)
            return [line for _ in range(n) for line in gen.op(d.ALL_OPS)]
        finally:
            d.KNOWN, d.LITERALS = real

    def test_every_op_is_generated_in_every_form(self):
        lines = self.lines()
        seen = {re.sub(r"_e(32|64)$", "", line.split()[0]) for line in lines}
        self.assertEqual(set(d.OPS) - seen, set())
        forms = {line.split()[0] for line in lines}
        for name, (enc, _, _) in d.OPS.items():
            if enc != "vop3":
                self.assertTrue({name + "_e32", name + "_e64"} <= forms, name)
        self.assertTrue(any(line.startswith("v_cmp_") for line in lines))

    def test_sources_respect_the_encoding_rules(self):
        for line in self.lines():
            op, _, rest = line.partition(" ")
            srcs = [a.strip() for a in rest.split(" op_sel:")[0].split(",")][1:]
            if op == "v_cndmask_b32_e64":
                srcs = srcs[:-1]
            lits = [x for x in srcs if x.startswith("0x")]
            scalars = set(x for x in srcs if x.startswith("s")) | set(lits)
            if op.endswith("_e32"):
                self.assertTrue(len(srcs) == 1 or re.fullmatch(r"v\d+", srcs[1]), line)
            self.assertLessEqual(len(lits), 1, line)
            self.assertLessEqual(len(scalars), 2 if not op.startswith("v_cndmask") else 0, line)
        self.assertFalse(any("0x" in line.split(",", 1)[1] for line in self.lines(literals=False)))

    def test_16bit_sources_take_16bit_literals(self):
        for line in self.lines():
            if line.startswith(d.OP_SEL_OPS):
                srcs = [a.strip() for a in line.split(" op_sel:")[0].split(",")][1:3]
                self.assertTrue(all(int(x, 16) <= 0xFFFF for x in srcs if x.startswith("0x")), line)
        # measured on gfx1036: such a literal is zero-extended, a float constant is its f32 bits
        self.assertEqual(run("v_mad_u32_u16 v1, 0xffff, 1, 0 op_sel:[1,0,0,0]")[0], 0)
        self.assertEqual(run("v_mad_u32_u16 v1, 0.5, 1, 0 op_sel:[1,0,0,0]")[0], 0x3F00)

    def test_known_divergences_are_left_out_by_default(self):
        def mask_as_data(lines):
            for line in lines:
                op, _, rest = line.partition(" ")
                srcs = [a.strip() for a in rest.split(",")][1:]
                if op == "v_cndmask_b32_e64":
                    srcs = srcs[:-1]
                if any(x in d.MASKS for x in srcs):
                    return True
            return False
        self.assertFalse(mask_as_data(self.lines()))
        self.assertTrue(mask_as_data(self.lines(known=True)))


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
        seeds = {2: 0x80000000, 3: 0x00008000, 4: 0x00050003, 5: 0x00070002, 6: 0xFFFFFFFE}
        cases = ["v_bfe_i32 v1, v2, 28, 8", "v_perm_b32 v1, v2, v3, 0x0b0a0908", "v_perm_b32 v1, v2, v3, 0x070d0c00",
                 "v_mad_u32_u16 v1, v4, v5, 1 op_sel:[1,0,0,0]", "v_ffbh_i32_e64 v1, v6",
                 "v_add_nc_u32_e64 v1, 1.0, s4", "v_xor_b32_e32 v1, 0x12345678, v2",
                 "v_lshlrev_b32_e64 v1, s5, -16", "v_cvt_pk_i16_i32 v1, v6, v4", "v_mul_hi_i32_i24_e32 v1, -2, v3",
                 "v_alignbit_b32 v1, v4, v6, 8", "v_mad_u32_u16 v1, 0xffff, 0.5, v3 op_sel:[1,1,0,0]"]
        for case in cases:
            with self.subTest(case=case):
                self.assertIsNone(d.check(prefix(seeds, {4: 7, 5: 36}) + [case] + d.observe([1, 2, 3, 4])))

    def test_a_wrong_translation_is_caught(self):
        lines = prefix({2: 5}) + ["v_add3_u32 v1, v2, 1, 2"] + d.observe([1, 2, 3, 4])
        self.assertIsNone(d.check(lines))
        want = d.reference(lines)
        real = d.translated
        try:
            d.translated = lambda _lines: [want[0] ^ 1] + want[1:]
            self.assertIn("word 0 (v1)", d.check(lines))
        finally:
            d.translated = real

    def test_shrinking_keeps_the_instruction_that_matters(self):
        lines = d.generate(d.random.Random(2), 5, ["v_add_nc_u32", "v_xor_b32", "cmp"])
        body = lines[d.N_SEED:-d.N_OBS]
        target = body[len(body) // 2]
        if target.startswith("v_cmp"):
            target = body[len(body) // 2 + 1]
        real = d.translated
        try:
            d.translated = lambda l: [w ^ (1 if target in l else 0) for w in d.reference(l)]
            small = d.shrink(lines)
        finally:
            d.translated = real
        kept = [line for line in small if not line.startswith(("v_mov_b32_e32", "s_mov_b32"))]
        self.assertIn(target, kept)
        self.assertLessEqual(len(kept), 2)                       # the target (and its compare, for a v_cndmask)


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
        self.assertIn("v_perm_b32", out)
        self.assertEqual(self.main("--ops", "v_nope")[0], 2)
        self.assertEqual(self.main("--oracle", "gpu")[0], 2)
        rc, out = self.main("--programs", "5", "--seed", "1")
        self.assertEqual(rc, 0, out)
        real = d.translated
        try:
            d.translated = lambda l: [w ^ 1 for w in d.reference(l)]
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_xor_b32")
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
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_xor_b32")
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
