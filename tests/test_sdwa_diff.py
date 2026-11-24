"""tools/sdwa_diff.py: the SDWA reference against hand-worked values, a short seeded fuzz run, and canaries.
Standard library only; no GPU."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "sdwa_diff.py"
sys.path.insert(0, str(TOOL.parent))
import sdwa_diff as d  # noqa: E402


def run(*body, v=None, s=None, vcc=0, regs=(1, 2, 3)):
    """The reference's four exported words; v / s map register -> seed value (others are 0)."""
    lines = ([f"v_mov_b32 v{r}, 0x{(v or {}).get(r, 0):x}" for r in d.VREGS]
             + [f"s_mov_b32 s{r}, 0x{(s or {}).get(r, 0):x}" for r in d.SREGS] + [f"s_mov_b32 vcc_lo, {vcc}"])
    return d.reference(lines + list(body) + d.observe(list(regs)))


class Reference(unittest.TestCase):
    def test_source_select_and_sext(self):
        # BYTE_1 of 0x8000 is 0x80: zero-extended 0x80, sign-extended 0xffffff80
        mov = "v_mov_b32_sdwa v1, {} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"
        self.assertEqual(run(mov.format("v2"), v={2: 0x8000})[0], 0x80)
        self.assertEqual(run(mov.format("sext(v2)"), v={2: 0x8000})[0], 0xFFFFFF80)
        # a constant is narrowed too: WORD_1 of -16 (0xfffffff0) is 0xffff
        self.assertEqual(run(mov.format("-16").replace("BYTE_1", "WORD_1"))[0], 0xFFFF)

    def test_destination_select_and_unused_bits(self):
        line = "v_mov_b32_sdwa v1, v2 dst_sel:BYTE_2 dst_unused:{} src0_sel:DWORD"
        seeds = {1: 0x11223344, 2: 0xA5}
        self.assertEqual(run(line.format("UNUSED_PAD"), v=seeds)[0], 0x00A50000)
        self.assertEqual(run(line.format("UNUSED_SEXT"), v=seeds)[0], 0xFFA50000)       # bits below stay 0
        self.assertEqual(run(line.format("UNUSED_PRESERVE"), v=seeds)[0], 0x11A53344)

    def test_integer_clamp_saturates(self):
        self.assertEqual(run("v_sub_nc_u32_sdwa v1, 1, 2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD "
                             "src0_sel:DWORD src1_sel:DWORD")[0], 0)
        self.assertEqual(run("v_add_nc_u32_sdwa v1, v2, 1 clamp dst_sel:DWORD dst_unused:UNUSED_PAD "
                             "src0_sel:DWORD src1_sel:DWORD", v={2: 0xFFFFFFFF})[0], 0xFFFFFFFF)

    def test_f16_neg_abs_on_the_selected_half(self):
        # WORD_1 of 0x40000000 is 2.0; -|2.0| * 3.0 = -6.0 = 0xc600
        self.assertEqual(run("v_mul_f16_sdwa v1, -|v2|, v3 dst_sel:WORD_0 dst_unused:UNUSED_PAD "
                             "src0_sel:WORD_1 src1_sel:WORD_0", v={2: 0x40000000, 3: 0x4200})[0], 0xC600)

    def test_float_constants(self):
        # measured on gfx1036: in an f16 op 1.0 is 0x3c00 over a zero high half, so WORD_1 of it is +0.0
        add = "v_add_f16_sdwa v1, 1.0, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:{} src1_sel:WORD_0"
        self.assertEqual(run(add.format("DWORD"), v={2: 0x3800})[0], 0x3E00)        # 1.0 + 0.5
        self.assertEqual(run(add.format("WORD_1"), v={2: 0x3800})[0], 0x3800)       # 0.0 + 0.5
        # in an integer op (v_cvt_f16_u16 included) it is the f32 bits: WORD_1 of -2.0 is 0xc000 = 49152 -> 0x7a00
        self.assertEqual(run("v_add_nc_u32_sdwa v1, 1.0, 0 dst_sel:DWORD dst_unused:UNUSED_PAD "
                             "src0_sel:DWORD src1_sel:DWORD")[0], 0x3F800000)
        self.assertEqual(run("v_cvt_f16_u16_sdwa v1, -2.0 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1")[0],
                         0x7A00)
        # v_cmp_class_f16's mask is an f16 operand too: 0.5 is 0x00003800 (BYTE_3=0, BYTE_1=0x38),
        # 1.0 is 0x00003C00 (WORD_1=0), and -2.0 is 0x0000C000 (BYTE_3=0), not f32 0x3F000000/0x3F800000/0xC0000000
        self.assertEqual(run("v_cmp_class_f16_sdwa vcc_lo, v2, 0.5 src0_sel:WORD_0 src1_sel:BYTE_3",
                             v={2: 0xBC00}, vcc=1)[3], 0)
        self.assertEqual(run("v_cmp_class_f16_sdwa vcc_lo, v2, 0.5 src0_sel:WORD_0 src1_sel:BYTE_1",
                             v={2: 0xBC00})[3], 1)                                   # 0x38 has bit 3 (-normal)
        self.assertEqual(run("v_cmp_class_f16_sdwa vcc_lo, 2.0, 1.0 src0_sel:DWORD src1_sel:WORD_1",
                             vcc=1)[3], 0)
        self.assertEqual(run("v_cmp_class_f16_sdwa vcc_lo, v2, -2.0 src0_sel:DWORD src1_sel:BYTE_3",
                             v={2: 0x0001}, vcc=1)[3], 0)                             # f32 BYTE_3=0xC0 falsely sets bit 7
        self.assertEqual(run("v_cmp_class_f16_sdwa vcc_lo, v2, 0x8 src0_sel:WORD_0 src1_sel:DWORD",
                             v={2: 0xBC00})[3], 1)                                   # -1.0 is -normal (bit 3)

    def test_16bit_result_with_dword_dst_zeroes_the_high_half(self):
        # measured on gfx1036 whatever dst_unused says; the non-SDWA f16 forms keep the high half instead
        for unused in d.UNUSED:
            self.assertEqual(run(f"v_add_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:{unused} "
                                 "src0_sel:WORD_0 src1_sel:WORD_0", v={1: 0xDEAD0000, 2: 0x3C00, 3: 0x3C00})[0],
                             0x4000)

    def test_fract_of_negative_zero_is_positive_zero(self):
        self.assertEqual(run("v_fract_f16_sdwa v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                             v={2: 0x8000})[0], 0)
        self.assertEqual(run("v_fract_f16_sdwa v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                             v={2: 0x8001})[0], 0x3BFF)                             # 1 - tiny rounds to 1.0, stops

    def test_compares_and_cndmask(self):
        cmp = "v_cmp_lt_{}16_sdwa vcc_lo, v2, v3 src0_sel:WORD_1 src1_sel:WORD_0"
        seeds = {2: 0x80000000, 3: 1}             # WORD_1 = 0x8000: -32768 signed, 32768 unsigned
        self.assertEqual(run(cmp.format("i"), v=seeds)[3], 1)
        self.assertEqual(run(cmp.format("u"), v=seeds, vcc=1)[3], 0)
        self.assertEqual(run(cmp.format("i"), "v_cndmask_b32_sdwa v1, v2, v3, vcc_lo dst_sel:DWORD "
                             "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0", v=seeds)[0], 1)

    def test_constant_shift_amount_is_selected(self):
        # the #2005 case: WORD_1 of 0x40001 is 4
        self.assertEqual(run("v_lshlrev_b32_sdwa v1, s4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD "
                             "src0_sel:WORD_1 src1_sel:DWORD", v={2: 1}, s={4: 0x40001})[0], 0x10)

    def test_nan_operands_are_skipped_not_guessed(self):
        with self.assertRaises(d.Skip):
            run("v_max_f16_sdwa v1, v2, v3 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
                v={2: 0x7E00})
        # a compare is defined for NaN: unordered
        self.assertEqual(run("v_cmp_u_f16_sdwa vcc_lo, v2, v3 src0_sel:WORD_0 src1_sel:WORD_0", v={2: 0x7E00})[3], 1)


class Generator(unittest.TestCase):
    def lines(self, n=2500, known=False, seed=1):
        real, d.KNOWN = d.KNOWN, known
        try:
            rng = d.random.Random(seed)
            return [d.emit(rng, rng.choice(sorted(d.OPS))) for _ in range(n)]
        finally:
            d.KNOWN = real

    def test_every_op_is_generated(self):
        seen = {line.split()[0][:-len("_sdwa")] for line in self.lines()}
        self.assertEqual(set(d.OPS) - seen, set())

    def test_known_divergences_are_left_out_by_default(self):
        def known_form(line):
            op, args, mods, _ = d.parse(line)
            family, form, width = d.OPS[op][:3]
            src0 = d.parse_src(args[1])
            if op in d.SHIFTS and src0["tok"][0] != "v" and (mods["src0_sel"] != "DWORD" or src0["sext"]):
                return "shift-const-sel"
            if form != "vopc" and width == 16 and mods["dst_sel"] == "DWORD":
                return "dword-16bit"
            srcs = args[1:3] if form != "vop1" else args[1:2]
            if family == "f16" and any(a in d.FLOAT_CONSTS and mods.get(f"src{i}_sel") != "DWORD"
                                       for i, a in enumerate(srcs)):
                return "f16-const-sel"
            return None
        self.assertEqual({known_form(line) for line in self.lines()} - {None}, {"dword-16bit", "f16-const-sel"})
        self.assertEqual({known_form(line) for line in self.lines(known=True)} - {None},
                         set(d.KNOWN_DIVERGENCES) | {"dword-16bit", "f16-const-sel"})

    def test_forms_main_translates_again_are_generated_by_default(self):
        # #2187 (dst_sel:DWORD zeroes a 16-bit result's high half) and #2209 (src_sel on an f16 float constant) are
        # on main, so a default run must keep both forms in, or a regression of either goes unseen
        self.assertEqual(set(d.KNOWN_DIVERGENCES), {"shift-const-sel"})
        lines = self.lines()
        self.assertTrue(any(d.OPS[line.split()[0][:-5]][2] == 16 and "dst_sel:DWORD" in line for line in lines))
        # each source on its own: the same operand is a float constant and its own selector is not DWORD
        selected = set()
        for line in lines:
            op, args, mods, _ = d.parse(line)
            family, form = d.OPS[op][:2]
            if family != "f16":
                continue
            srcs = args[1:3] if form != "vop1" else args[1:2]
            for i, a in enumerate(srcs):
                if a in d.FLOAT_CONSTS and mods.get(f"src{i}_sel", "DWORD") != "DWORD":
                    selected.add(i)
        self.assertEqual(selected, {0, 1})


class Translated(unittest.TestCase):
    ALLOWED_SKIPS = ("reference: ", "translator unsupported: v_cmp_class_f16",
                     "translator unsupported: v_cvt_f16_u16 output modifier",
                     "translator unsupported: v_cvt_f16_i16 output modifier")

    def test_harness_canaries(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(d.self_test())

    def test_short_seeded_run_matches(self):
        counts, failures = d.run_suite(60, 5, 4, sorted(d.OPS))
        self.assertEqual(failures, [], failures[:1])
        self.assertEqual([r for r in counts["skip"] if not r.startswith(self.ALLOWED_SKIPS)], [])
        self.assertGreaterEqual(counts["pass"], 50)

    def test_hand_cases_match_the_translator(self):
        # the Reference cases outside KNOWN_DIVERGENCES, so a sabotaged sdwa_src / sdwa_dst fails here directly
        seeds = {1: 0x11223344, 2: 0x400080A5, 3: 0x80004200}
        prefix = ([f"v_mov_b32 v{r}, 0x{seeds.get(r, 0):x}" for r in d.VREGS]
                  + [f"s_mov_b32 s{r}, 0x40001" for r in d.SREGS] + ["s_mov_b32 vcc_lo, 0"])
        cases = [f"v_mov_b32_sdwa v1, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_{n}" for n in (0, 1)]
        cases += [f"v_mov_b32_sdwa v1, v2 dst_sel:{sel} dst_unused:{u} src0_sel:BYTE_0"
                  for sel in ("BYTE_1", "BYTE_2", "WORD_1") for u in d.UNUSED]
        cases += ["v_sub_nc_u32_sdwa v1, 1, 2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                  "v_mul_f16_sdwa v1, -|v2|, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 "
                  "src1_sel:WORD_0",
                  "v_fract_f16_sdwa v1, v3 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1",
                  "v_cmp_lt_i16_sdwa vcc_lo, v3, v2 src0_sel:WORD_1 src1_sel:BYTE_0",
                  "v_lshlrev_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 "
                  "src1_sel:WORD_0"]
        for case in cases:
            with self.subTest(case=case):
                self.assertIsNone(d.check(prefix + [case, "v_cmp_lt_u16_sdwa vcc_lo, v1, v2 src0_sel:BYTE_2 "
                                                    "src1_sel:BYTE_0"] + d.observe([1, 2, 3])))

    def test_a_wrong_translation_is_caught(self):
        lines = (["v_mov_b32 v2, 0x8000", "s_mov_b32 vcc_lo, 0",
                  "v_mov_b32_sdwa v1, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
                  "v_cmp_eq_u32_sdwa vcc_lo, v1, v1 src0_sel:DWORD src1_sel:DWORD"] + d.observe([1, 2, 3]))
        self.assertIsNone(d.check(lines))
        want = d.reference(lines)
        real = d.translated
        try:
            d.translated = lambda _lines: want[:3] + [want[3] ^ 1]
            self.assertIn("word 3 (VCC)", d.check(lines))
            d.translated = lambda _lines: [0x80] + want[1:]          # sext() ignored
            self.assertIn("word 0 (v1)", d.check(lines))
        finally:
            d.translated = real

    def test_shrinking_keeps_the_instruction_that_matters(self):
        rng = d.random.Random(2)
        lines = d.generate(rng, 4, ["v_add_nc_u32", "v_xor_b32"])
        target = lines[d.N_SEED + 2]
        real = d.translated
        try:
            # a fake translator that is wrong exactly when `target` is in the program
            d.translated = lambda l: [w ^ (1 if target in l else 0) for w in d.reference(l)]
            small = d.shrink(lines)
        finally:
            d.translated = real
        self.assertEqual(small[d.N_SEED:-d.N_OBS], [target])


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
        self.assertIn("v_cmp_class_f16", out)
        self.assertEqual(self.main("--ops", "v_nope")[0], 2)
        self.assertEqual(self.main("--oracle", "gpu")[0], 2)
        rc, out = self.main("--programs", "5", "--seed", "1", "--family", "int")
        self.assertEqual(rc, 0, out)
        real = d.translated
        try:
            d.translated = lambda l: [w ^ 1 for w in d.reference(l)]
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_xor_b32")
        finally:
            d.translated = real
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out)

    def test_hardware_oracle_without_the_tools_exits_77(self):
        import os
        import subprocess
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOL.parent / "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", str(TOOL), "--oracle", "hw", "--programs", "3"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_a_gpu_without_sdwa_exits_77(self):
        import unittest.mock
        from hw_oracle import runner

        class NoSdwa:
            arch, device, closed = "gfx1150", "stub", False

            def close(self):
                NoSdwa.closed = True
        with unittest.mock.patch.object(runner, "open_oracle", return_value=NoSdwa()):
            rc, out = self.main("--oracle", "hw", "--programs", "3")
        self.assertEqual(rc, 77)
        self.assertIn("no SDWA", out)
        self.assertTrue(NoSdwa.closed)


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
