"""tools/salu_diff.py: the scalar-ALU reference against hand-worked values, and a short seeded fuzz run (FUZZ-4).
Standard library only; no GPU."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
import unittest.mock

TOOL = Path(__file__).resolve().parents[1] / "tools" / "salu_diff.py"
sys.path.insert(0, str(TOOL.parent))
import salu_diff as d  # noqa: E402


def run(*body, regs=(4, 5, 6), seeds=None):
    """The reference's four exported words for a program; seeds maps SGPR -> value."""
    lines = [f"s_mov_b32 s{r}, 0x{v:x}" for r, v in (seeds or {}).items()] + list(body)
    return d.reference(lines + d.observe(list(regs)))


class Reference(unittest.TestCase):
    def test_carries_borrows_and_signed_overflow(self):
        self.assertEqual(run("s_add_u32 s4, -1, 1"), [0, 0, 0, 1])                         # carry out
        self.assertEqual(run("s_sub_u32 s4, 0, 1"), [0xFFFFFFFF, 0, 0, 1])                  # borrow
        self.assertEqual(run("s_add_i32 s4, 0x7fffffff, 1"), [0x80000000, 0, 0, 1])         # signed overflow
        self.assertEqual(run("s_add_i32 s4, -1, 1"), [0, 0, 0, 0])                          # no overflow
        self.assertEqual(run("s_sub_u32 s4, 0, 1", "s_addc_u32 s5, 1, 1"), [0xFFFFFFFF, 3, 0, 0])
        self.assertEqual(run("s_sub_u32 s4, 0, 1", "s_subb_u32 s5, 1, 1"), [0xFFFFFFFF, 0xFFFFFFFF, 0, 1])

    def test_min_max_scc_is_strict(self):
        # SCC = 1 only when S0 is the one chosen; a tie picks S1 (SHADER-40).
        self.assertEqual(run("s_min_i32 s4, 5, 5")[3], 0)
        self.assertEqual(run("s_min_i32 s4, -1, 1")[::3], [0xFFFFFFFF, 1])
        self.assertEqual(run("s_min_u32 s4, -1, 1")[::3], [1, 0])
        self.assertEqual(run("s_max_i32 s4, 7, 7")[3], 0)

    def test_bit_scans_and_counts(self):
        self.assertEqual(run("s_ff0_i32_b32 s4, 3", "s_ff1_i32_b32 s5, 0x100", "s_ff1_i32_b32 s6, 0")[:3],
                         [2, 8, 0xFFFFFFFF])
        self.assertEqual(run("s_flbit_i32_b32 s4, 1", "s_flbit_i32_b32 s5, 0", "s_bcnt0_i32_b32 s6, 0xff")[:3],
                         [31, 0xFFFFFFFF, 24])
        self.assertEqual(run("s_ff1_i32_b64 s4, s[6:7]", seeds={6: 0, 7: 2})[0], 33)

    def test_field_ops(self):
        self.assertEqual(run("s_bfe_u32 s4, 0xf0, 0x40004")[0], 0xF)
        self.assertEqual(run("s_bfe_i32 s4, 0xf0, 0x40004")[0], 0xFFFFFFFF)
        self.assertEqual(run("s_bfm_b32 s4, 4, 8")[0], 0xF00)
        # a field past bit 31 stops there (gfx1150 hardware, LLVM constantFoldBFE): width 127 at offset 28 is 4 bits
        self.assertEqual(run("s_bfe_i32 s4, 0xd2f90159, 0x7f001c")[0], 0xFFFFFFFD)
        self.assertEqual(run("s_bfe_u32 s4, 0xd2f90159, 0x7f001c")[0], 0xD)
        self.assertEqual(run("s_bitset1_b32 s4, 3", seeds={4: 0x10})[0], 0x18)

    def test_sopk_extends_its_immediate(self):
        self.assertEqual(run("s_movk_i32 s4, 0x8000")[0], 0xFFFF8000)
        self.assertEqual(run("s_cmpk_lt_u32 s4, 0x8000", seeds={4: 5})[3], 1)        # zero-extended
        self.assertEqual(run("s_cmpk_lt_i32 s4, 0x8000", seeds={4: 5})[3], 0)        # -32768
        self.assertEqual(run("s_addk_i32 s4, 1", seeds={4: 0x7FFFFFFF}), [0x80000000, 0, 0, 1])

    def test_scc_survives_ops_that_do_not_write_it(self):
        self.assertEqual(run("s_cmp_eq_u32 1, 1", "s_mov_b32 s4, 0", "s_bfm_b32 s5, 1, 1")[3], 1)

    def test_absdiff_mul_hi_and_lshl_add(self):
        self.assertEqual(run("s_absdiff_i32 s4, 0x7FFFFFFF, 0x80000000")[::3], [1, 1])
        self.assertEqual(run("s_absdiff_i32 s4, 5, 5")[::3], [0, 0])
        self.assertEqual(run("s_mul_hi_i32 s4, -1, 1")[::3], [0xFFFFFFFF, 0])
        self.assertEqual(run("s_cmp_eq_u32 1, 1", "s_mul_hi_i32 s4, -1, 1")[::3], [0xFFFFFFFF, 1])
        self.assertEqual(run("s_lshl1_add_u32 s4, 0x80000000, 0")[::3], [0, 1])
        self.assertEqual(run("s_lshl2_add_u32 s4, 3, 4")[::3], [16, 0])
        self.assertEqual(run("s_lshl4_add_u32 s4, 0x0FFFFFFF, 0x10")[::3], [0, 1])

    def test_sext_flbit_and_pack(self):
        self.assertEqual(run("s_sext_i32_i8 s4, 0x80")[0], 0xFFFFFF80)
        self.assertEqual(run("s_sext_i32_i8 s4, 0x7F")[0], 0x7F)
        self.assertEqual(run("s_sext_i32_i16 s4, 0x8000")[0], 0xFFFF8000)
        self.assertEqual(run("s_sext_i32_i16 s4, 0x7FFF")[0], 0x7FFF)
        self.assertEqual(run("s_flbit_i32 s4, 0")[0], 0xFFFFFFFF)
        self.assertEqual(run("s_flbit_i32 s4, 1")[0], 31)
        self.assertEqual(run("s_flbit_i32 s4, 0x80000000")[0], 1)
        self.assertEqual(run("s_pack_ll_b32_b16 s4, 0x12345678, 0xABCDEF01")[0], 0xEF015678)
        self.assertEqual(run("s_pack_lh_b32_b16 s4, 0x12345678, 0xABCDEF01")[0], 0xABCD5678)
        self.assertEqual(run("s_pack_hh_b32_b16 s4, 0x12345678, 0xABCDEF01")[0], 0xABCD1234)


class Translated(unittest.TestCase):
    def test_harness_canaries(self):
        self.assertTrue(d.self_test())

    def test_short_seeded_run_matches(self):
        # 32-bit data ops only: 64-bit ones and s_wqm_b32 follow the translator's lane-mask model. s_bfe_*
        # control words stay inside offset + width <= 32 by default (past it is an open hardware question in
        # #156). s_ff0/ff1_i32_b32 wait for SHADER-46 (#349): main reads a constant source as a lane mask.
        ops = sorted(o for o, (form, _) in d.OPS.items()
                     if "64" not in form and o not in ("s_wqm_b32", "s_ff0_i32_b32", "s_ff1_i32_b32"))
        counts, failures = d.run_suite(60, 4, 5, ops)
        self.assertEqual(failures, [], failures[:1])
        self.assertEqual(counts["skip"], {})
        self.assertEqual(counts["pass"], 60)

    def test_bfe_control_words_stay_in_range_by_default(self):
        rng = d.random.Random(1)
        for _ in range(300):
            ctl = int(d.emit(rng, "s_bfe_i32").rsplit(", ", 1)[1], 0)
            self.assertLessEqual((ctl & 31) + (ctl >> 16 & 0x7F), 32)

    def test_min_max_ties_match_since_shader_40(self):
        # #332 is on main: s_min/max ties leave SCC 0 in the translator too.
        counts, failures = d.run_suite(40, 5, 3, ["s_max_i32", "s_max_u32", "s_min_i32", "s_min_u32"])
        self.assertEqual((counts["pass"], failures), (40, []))

    def test_a_wrong_translation_is_caught(self):
        lines = ["s_mov_b32 s4, 5", "s_min_i32 s5, s4, s4"] + d.observe([4, 5, 6])
        want = d.reference(lines)
        real = d.translated
        try:
            d.translated = lambda _lines: want[:3] + [want[3] ^ 1]
            self.assertIn("word 3 (SCC)", d.check(lines))
            d.translated = lambda _lines: [want[0] ^ 1] + want[1:]
            self.assertIn("word 0 (s4)", d.check(lines))
        finally:
            d.translated = real


class Generator(unittest.TestCase):
    def test_two_source_forms_share_one_literal_word(self) -> None:
        for op in ("s_add_u32", "s_bfm_b64", "s_cmp_eq_u32"):
            for operands in (("0x12345", "0x23456"), ("0x12345", "0x12345")):
                with self.subTest(op=op, operands=operands):
                    with unittest.mock.patch.object(d, "_src32", side_effect=operands):
                        line = d.emit(d.random.Random(2), op)
                    sources = line.split(" ", 1)[1].split(", ")[-2:]
                    literals = [source for source in sources if source.startswith("0x")]
                    self.assertEqual(len(set(literals)), 1, line)
                    self.assertEqual(len(literals), 2 if operands[0] == operands[1] else 1, line)


class Branches(unittest.TestCase):
    """--branches: SCC consumer chains and s_cbranch_scc0/1 forward over 1..3 instructions."""

    def test_reference_takes_the_branch_on_the_right_scc(self):
        body = ("s_cmp_eq_u32 1, 1", "s_cbranch_scc1 +2", "s_mov_b32 s4, 7", "s_mov_b32 s5, 8", "s_mov_b32 s6, 9")
        self.assertEqual(run(*body), [0, 0, 9, 1])                                  # taken: s4, s5 skipped
        self.assertEqual(run(*(b.replace("scc1", "scc0") for b in body)), [7, 8, 9, 1])   # not taken
        self.assertEqual(run("s_cmp_eq_u32 0, 1", "s_cbranch_scc0 +1", "s_mov_b32 s4, 7")[::3], [0, 0])

    def test_a_consumer_chain_carries_scc(self):
        # 0 - 1 borrows; s_addc adds it in and carries out of 0xffffffff + 0 + 1; s_cselect keeps that SCC
        self.assertEqual(run("s_sub_u32 s4, 0, 1", "s_addc_u32 s5, s4, 0", "s_cselect_b32 s6, 5, 6"), [0xFFFFFFFF, 0, 5, 1])

    def test_listing_offsets_count_literals(self):
        lines = ["s_cbranch_scc0 +3", "s_add_u32 s4, s5, 0x12345", "s_movk_i32 s4, 0x8000", "s_cmp_lt_u32 s4, -1"]
        self.assertEqual(d.listing(lines)[0], ("s_cbranch_scc0 4", 4))             # 8 + 4 + 4 bytes
        self.assertEqual([n for _, n in d.listing(lines)], [4, 8, 4, 4])

    def test_shrinking_keeps_the_skipped_range_consistent(self):
        body = ["s_cbranch_scc1 +2", "s_mov_b32 s4, 1", "s_mov_b32 s5, 2", "s_mov_b32 s6, 3"]
        self.assertEqual(d.drop(body, 1), ["s_cbranch_scc1 +1", "s_mov_b32 s5, 2", "s_mov_b32 s6, 3"])
        self.assertEqual(d.drop(body, 3), body[:3])

    def test_every_producer_meets_every_consumer_and_both_branches(self):
        ops = sorted(o for o in d.OPS if o not in d.LANE_MASK_OPS)
        producers = {o for o in ops if d.writes_scc(o)}
        seen = set()
        rng = d.random.Random(1)
        for _ in range(3000):
            lines = d.generate(rng, rng.randrange(1, 7), ops, branches=True)
            for prev, nxt in zip(lines, lines[1:]):
                op, after = prev.split()[0], nxt.split()[0]
                if op in producers and (after in d.SCC_CONSUMERS or after in d.SCC_BRANCHES):
                    seen.add((op, after))
        want = {(p, c) for p in producers for c in d.SCC_CONSUMERS + d.SCC_BRANCHES}
        self.assertEqual(want - seen, set())

    def test_short_seeded_run_matches(self):
        ops = sorted(o for o in d.OPS if o not in d.LANE_MASK_OPS)
        counts, failures = d.run_suite(60, 3, 5, ops, branches=True)
        self.assertEqual((counts["pass"], counts["skip"], failures), (60, {}, []))

    def test_a_misplaced_branch_target_is_caught(self):
        # One dword too far: the instruction after the skipped range is skipped too.
        lines = (["s_mov_b32 s4, 1", "s_cmp_eq_u32 s4, s4", "s_cbranch_scc1 +1", "s_mov_b32 s5, 0x12345678",
                  "s_mov_b32 s6, 3"] + d.observe([4, 5, 6]))
        self.assertIsNone(d.check(lines))
        real = d.listing
        try:
            d.listing = lambda ls: [(f"{t.rsplit(' ', 1)[0]} {int(t.rsplit(' ', 1)[1]) + 1}"
                                     if t.startswith("s_cbranch") else t, n) for t, n in real(ls)]
            self.assertIn("word 2 (s6)", d.check(lines))
        finally:
            d.listing = real


class CliOps(unittest.TestCase):
    def test_ops_whitespace_and_duplicates_are_normalized(self):
        cases = [
            ("s_add_u32, s_sub_u32", ["s_add_u32", "s_sub_u32"]),
            ("s_add_u32,s_add_u32", ["s_add_u32"]),
        ]
        for raw, expected in cases:
            with unittest.mock.patch.object(d, "run_suite", wraps=d.run_suite) as spy:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = d.main(["--programs", "2", "--max-ops", "2", "--ops", raw])
                self.assertEqual(rc, 0, out.getvalue())
                spy.assert_called_once()
                self.assertEqual(spy.call_args.args[3], expected)

    def test_ops_unknown_and_empty_tokens_report_on_stderr(self):
        cases = [
            ("nope", "unknown ops: nope"),
            ("s_add_u32,", "--ops must be a comma-separated list of opcodes"),
            (",", "--ops must be a comma-separated list of opcodes"),
            (" , ", "--ops must be a comma-separated list of opcodes"),
            ("", "--ops must be a comma-separated list of opcodes"),
        ]
        for raw, message in cases:
            proc = subprocess.run(
                [sys.executable, "-I", str(TOOL), "--programs", "1", "--ops", raw],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn(message, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)


class EverythingSkipped(unittest.TestCase):
    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        import contextlib
        import io
        from unittest import mock

        def refuse(lines, *args):
            raise d.Skip("unsupported form")
        out = io.StringIO()
        with mock.patch.object(d, "translated", refuse), contextlib.redirect_stdout(out):
            rc = d.main(["--programs", "3", "--seed", "1"])
        self.assertEqual(rc, 2, out.getvalue())
        self.assertIn("no program ran: every one was skipped", out.getvalue())


if __name__ == "__main__":
    unittest.main()
