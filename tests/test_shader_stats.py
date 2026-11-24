"""Unit tests for scripts/shader_stats.py (standard library only, no GPU).

    python -I tests/test_shader_stats.py
"""
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "shader_stats.py")

# ps_flat's listing as llvm-mc prints it: 8 instructions, highest VGPR v5.
PS_FLAT = """; ps_flat  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words
\tv_interp_mov_f32_e32 v2, p0, attr1.x    ; encoding: [0x02,0x04,0x0a,0xc8]
\tv_add_nc_u32_e32 v3, 1, v2              ; encoding: [0x81,0x04,0x06,0x4a]
\tv_interp_p1_f32_e32 v4, v0, attr0.x     ; encoding: [0x00,0x00,0x10,0xc8]
\tv_interp_p2_f32_e32 v4, v1, attr0.x     ; encoding: [0x01,0x00,0x11,0xc8]
\tv_interp_p1_f32_e32 v5, v0, attr0.y     ; encoding: [0x00,0x01,0x14,0xc8]
\tv_interp_p2_f32_e32 v5, v1, attr0.y     ; encoding: [0x01,0x01,0x15,0xc8]
\texp mrt0 v2, v3, v4, v5 done vm         ; encoding: [0x0f,0x18,0x00,0xf8,0x02,0x03,0x04,0x05]
\ts_endpgm                                ; encoding: [0x00,0x00,0x81,0xbf]
"""

# Every RADV statistic reported: this full output must stay byte for byte as it was.
COMPLETE = ["ps_flat,ps,Instructions,16", "ps_flat,ps,VGPRs,8", "ps_flat,ps,SGPRs,20", "ps_flat,ps,Scratch size,1024",
            "ps_flat,ps,Spilled VGPRs,2", "ps_flat,ps,Spilled SGPRs,1", "ps_flat,ps,Subgroups per SIMD,16",
            "ps_flat,ps,Driver pipeline hash,abc"]
COMPLETE_SUMMARY = ("ps: 1 shaders, code growth median 2.00x (p90 2.00x, max 2.00x); VGPRs median 8 vs original 6; "
                    "scratch in 1, spills in 1; waves/SIMD median 16\n")
COMPLETE_DETAILS = """shader                     stage  orig   aco growth v_orig VGPRs SGPRs scratch spills waves
ps_flat                    ps        8    16   2.00      6     8    20    1024      3    16

"""
NOTHING = "ps: 1 shaders, code growth n/a; VGPRs n/a; scratch n/a, spills n/a; waves/SIMD n/a\n"


class ShaderStats(unittest.TestCase):
    def run_stats(self, rows, details=True, listings=None, header="name,stage,statistic,value", extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.makedirs(disasm)
            for name, text in (listings or {"ps_flat": PS_FLAT}).items():
                with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8") as f:
                    f.write(text)
            csv_path = os.path.join(tmp, "stats.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(header + "\n" + "".join(f"{r}\n" for r in rows))
            cmd = [sys.executable, "-I", SCRIPT, csv_path, disasm] + (["--details"] if details else []) + list(extra)
            return subprocess.run(cmd, capture_output=True, text=True)

    def detail(self, r, name="ps_flat"):
        return next(ln for ln in r.stdout.splitlines() if ln.startswith(name + " ")).split()

    def test_growth_registers_and_scratch(self):
        r = self.run_stats(["ps_flat,ps,Instructions,16", "ps_flat,ps,VGPRs,8", "ps_flat,ps,SGPRs,20",
                            "ps_flat,ps,Scratch size,1024", "ps_flat,ps,Spilled VGPRs,2",
                            "ps_flat,ps,Subgroups per SIMD,16", "ps_flat,ps,Driver pipeline hash,abc"])
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("ps_flat"))
        self.assertEqual(line.split(), ["ps_flat", "ps", "8", "16", "2.00", "6", "8", "20", "1024", ">=2", "16"])
        self.assertIn("ps: 1 shaders, code growth median 2.00x", r.stdout)
        self.assertIn("scratch in 1, spills in 1", r.stdout)

    def test_empty_listing_has_no_growth_denominator(self):
        rows = ["ps_empty,ps,Instructions,16"]
        listings = {"ps_empty": "; no instructions\n"}
        result = self.run_stats(rows, listings=listings)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.detail(result, "ps_empty")[2:5], ["0", "16", "-"])
        self.assertIn("code growth n/a", result.stdout)

        result = self.run_stats(rows + ["ps_flat,ps,Instructions,16"], details=False,
                                listings={**listings, "ps_flat": PS_FLAT})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("code growth median 2.00x (p90 2.00x, max 2.00x) (1 of 2)", result.stdout)

    def test_original_vgpr_count_includes_non_contiguous_image_addresses(self) -> None:
        listing = ("\timage_sample v[6:9], [v1, v31, v2], s[0:7], s[8:11] dmask:0xf\n"
                   "\ts_endpgm\n")
        result = self.run_stats(["ps_nsa,ps,VGPRs,24"], listings={"ps_nsa": listing})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.detail(result, "ps_nsa")[5:7], ["32", "24"])
        self.assertIn("VGPRs median 24 vs original 32", result.stdout)

    def test_unreadable_listing_has_a_path_specific_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = os.path.join(tmp, "stats.csv")
            listing = os.path.join(tmp, "ps_flat.s")
            with open(stats, "w", encoding="utf-8", newline="\n") as f:
                f.write("name,stage,statistic,value\n" + COMPLETE[0] + "\n")
            with open(listing, "w", encoding="utf-8", newline="\n") as f:
                f.write(PS_FLAT)
            open_file = open

            def locked(path, *args, **kwargs):
                if path == listing:
                    raise PermissionError(13, "Permission denied", path)
                return open_file(path, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=locked), \
                    mock.patch.object(sys, "argv", [SCRIPT, stats, tmp]), self.assertRaises(SystemExit) as caught:
                runpy.run_path(SCRIPT, run_name="__main__")
            self.assertIn(listing, str(caught.exception))
            self.assertIn("Permission denied", str(caught.exception))

    def test_shaders_without_a_listing_are_left_out(self):
        r = self.run_stats(["ps_other,ps,Instructions,10"], details=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no shader is in both", r.stderr)

    def test_complete_radv_output_is_unchanged(self):
        r = self.run_stats(COMPLETE)
        self.assertEqual((r.returncode, r.stderr, r.stdout), (0, "", COMPLETE_DETAILS + COMPLETE_SUMMARY))
        r = self.run_stats(COMPLETE, details=False)
        self.assertEqual((r.returncode, r.stderr, r.stdout), (0, "", COMPLETE_SUMMARY))

    def test_other_driver_statistics_are_shown_raw_not_interpreted(self):
        rows = ["ps_flat,ps,Foo count (bytes),123", 'ps_flat,ps,"Bar, total",18446744073709551615',
                "ps_flat,ps,Register Count,24", "ps_flat,ps,stage,99"]
        r = self.run_stats(rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.detail(r), ["ps_flat", "ps", "8", "-", "-", "6", "-", "-", "-", "-", "-"])
        lines = r.stdout.splitlines()
        i = lines.index("    Bar, total = 18446744073709551615")
        self.assertEqual(lines[i + 1:i + 3], ["    Foo count (bytes) = 123", "    Register Count = 24"])
        self.assertIn("    stage = 99", lines)
        self.assertTrue(r.stdout.endswith("\n" + NOTHING), r.stdout)
        r = self.run_stats(rows, details=False)
        self.assertEqual((r.returncode, r.stdout), (0, NOTHING))
        self.assertEqual(len(r.stderr.splitlines()), 1, r.stderr)
        self.assertEqual(r.stderr, "additional driver statistics (see --details): Bar, total; Foo count (bytes); Register Count; stage\n")

    def test_partial_statistics_state_their_subset(self):
        rows = ["ps_flat,ps,Instructions,16", "ps_flat,ps,VGPRs,8", "ps_flat,ps,SGPRs,20", "ps_flat,ps,Scratch size,1024",
                "ps_flat,ps,Spilled VGPRs,2", "ps_flat,ps,Spilled SGPRs,0", "ps_flat,ps,Subgroups per SIMD,16",
                "ps_part,ps,Instructions,12"]
        listings = {"ps_flat": PS_FLAT, "ps_part": PS_FLAT}
        r = self.run_stats(rows, listings=listings)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.detail(r, "ps_part"), ["ps_part", "ps", "8", "12", "1.50", "6", "-", "-", "-", "-", "-"])
        r = self.run_stats(rows, details=False, listings=listings)
        self.assertEqual((r.returncode, r.stderr), (0, ""))
        self.assertEqual(r.stdout, "ps: 2 shaders, code growth median 1.75x (p90 1.50x, max 2.00x); "
                                   "VGPRs median 8 vs original 6 (1 of 2); scratch in 1 (1 of 2), spills in 1 (1 of 2); "
                                   "waves/SIMD median 16 (1 of 2)\n")

    def test_measured_zero_differs_from_missing_and_one_spill_counter_is_a_lower_bound(self):
        cases = [(["Scratch size,0", "Spilled VGPRs,0", "Spilled SGPRs,0"], "0", "0", "scratch in 0, spills in 0;"),
                 (["Spilled SGPRs,3"], "-", ">=3", "scratch n/a, spills in 1;"),
                 (["Spilled VGPRs,0"], "-", "-", "scratch n/a, spills n/a;")]
        for stats, scratch, spilled, summary in cases:
            with self.subTest(stats=stats):
                r = self.run_stats(["ps_flat,ps,Instructions,16"] + ["ps_flat,ps," + s for s in stats])
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(self.detail(r)[8:10], [scratch, spilled])
                self.assertIn(summary, r.stdout)

    def test_values_are_exact_or_unavailable(self):
        r = self.run_stats(["ps_flat,ps,Instructions,18446744073709551615"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.detail(r)[3], "18446744073709551615")
        r = self.run_stats(["ps_flat,ps,Instructions,12.0"])
        self.assertEqual(self.detail(r)[3], "12")
        for stat, column in [("Instructions,nan", 3), ("Instructions,", 3), ("Instructions,inf", 3),
                             ("Instructions," + "9" * 400, 3), ("VGPRs,abc", 6)]:
            with self.subTest(stat=stat[:20]):
                r = self.run_stats(["ps_flat,ps," + stat])
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                self.assertEqual(self.detail(r)[column], "-")

    def test_missing_required_header_is_rejected_with_path_and_line(self):
        r = self.run_stats(["ps_flat,ps,Instructions,16"], header="name,stage,statistic")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("stats.csv: line 1:", r.stderr)
        self.assertIn("missing required header column(s): value", r.stderr)

    def test_duplicate_required_header_is_rejected_before_rows(self):
        r = self.run_stats(["ps_flat,ps_flat,ps,Instructions,16"], header="name,name,stage,statistic,value")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("stats.csv: line 1:", r.stderr)
        self.assertIn("duplicated required header column(s): name", r.stderr)

    def test_broken_quoted_record_does_not_report_partial_statistics(self):
        result = self.run_stats([
            "ps_flat,ps,Instructions,16",
            'ps_flat,ps,VGPRs,"8',
            "ps_flat,ps,Spilled VGPRs,2",
        ])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("stats.csv:", result.stderr)
        self.assertIn("unexpected end of data", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_short_row_missing_a_cell_is_rejected_with_its_line(self):
        r = self.run_stats(["ps_flat,ps,Instructions"])
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("stats.csv: line 2:", r.stderr)
        self.assertIn("row missing required column(s): value", r.stderr)

    def test_extra_headed_columns_and_explicitly_blank_value_remain_accepted(self):
        r = self.run_stats(["ps_flat,ps,Instructions,,abc", "ps_flat,ps,VGPRs,8,xyz"],
                           header="name,stage,statistic,value,extra")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = self.detail(r)
        self.assertEqual((d[3], d[6]), ("-", "8"))

    def test_row_wider_than_the_header_is_rejected_with_its_line(self):
        r = self.run_stats(["ps_flat,ps,Instructions,16", "ps_flat,ps,VGPRs,8,unheaded"])
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("stats.csv: line 3: row has more fields than the header", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def run_args(self, *args):
        return subprocess.run([sys.executable, "-I", SCRIPT, *args], capture_output=True, text=True)

    def test_help_goes_to_stdout_with_exit_code_0(self):
        for flag in ("-h", "--help"):
            r = self.run_args(flag)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("<stats.csv> <disasm dir> [--details]", r.stdout)
            self.assertEqual(r.stderr, "")

    def test_help_wins_over_other_arguments(self):
        r = self.run_args("stats.csv", "disasm", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--details", r.stdout)

    def test_wrong_argument_count_is_a_usage_error(self):
        for args in ((), ("stats.csv",), ("stats.csv", "disasm", "extra")):
            r = self.run_args(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn("usage: shader_stats.py", r.stderr)
            self.assertEqual(r.stdout, "")

    def test_unknown_option_is_refused_not_ignored(self):
        # --detail (a typo) used to print the summary with exit code 0, as if no option had been given.
        r = self.run_stats(COMPLETE, details=False, extra=["--detail"])
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown option --detail", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_details_is_still_accepted_anywhere(self):
        r = self.run_stats(COMPLETE, details=False, extra=["--details"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.detail(r)[0], "ps_flat")

    def test_header_only_report_is_a_valid_empty_report(self):
        r = self.run_stats([], details=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("missing required", r.stderr)
        self.assertIn("no shader is in both", r.stderr)


if __name__ == "__main__":
    unittest.main()
