"""tools/root_spill_census.py on hand-written index data (no game files, no GPU).

    python -I tests/test_root_spill_census.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "root_spill_census.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import root_spill_census as rsc  # noqa: E402


def reads(*pairs):
    return [{"sgpr": s, "kind": k} for s, k in pairs]


# Register-based ABI, nothing spilled: constants below the base, pointers from the base up.
NO_SPILL = {
    "a.ps": {"stage": "ps", "user_sgpr_reads": reads((1, "const32"), (8, "ptr32_lo"), (9, "ptr32_lo"), (31, "const32"))},
    "b.ps": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (10, "const32"))},
    "c.cs": {"stage": "cs", "user_sgprs": 6, "user_sgpr_reads": reads((1, "const32"), (4, "ptr32_lo"), (5, "const32"))},
    "d.cs": {"stage": "cs", "user_sgprs": 14, "user_sgpr_reads": reads((4, "ptr32_lo"), (13, "ptr32_lo"))},
    "e.gs": {"stage": "gs", "user_sgpr_reads": reads((2, "const32"), (8, "ptr64_lo"), (9, "ptr64_hi"))},
    "f.vs": {"stage": "vs", "user_sgpr_reads": []},
}

# A compute shader that spilled the way section 7 measured it: a 64-bit pointer in s[0:1].
SPILLED = {
    "s.cs": {"stage": "cs", "user_sgprs": 32,
             "user_sgpr_reads": reads((0, "ptr64_lo"), (1, "ptr64_hi"), (4, "ptr32_lo"), (31, "const32"))},
}


class CensusTest(unittest.TestCase):
    def test_no_spill_corpus(self):
        s = rsc.census(NO_SPILL)["stages"]
        for stage in s.values():
            self.assertEqual(stage["shaders_with_spill_pointer_candidate"], 0)
            self.assertEqual(stage["spill_pointer_candidates"], {})
        self.assertEqual(s["ps"]["highest_sgpr_read"], {"s10": 1, "s31": 1})
        self.assertEqual(s["ps"]["highest_read_kind"], {"const32": 2})
        self.assertEqual(s["ps"]["max_highest_sgpr_read"], 31)
        self.assertEqual(s["cs"]["highest_read_kind"], {"const32": 1, "pointer": 1})
        self.assertEqual(s["vs"]["highest_sgpr_read"], {})
        self.assertIsNone(s["vs"]["max_highest_sgpr_read"])

    def test_user_sgprs_budget(self):
        cs = rsc.census(NO_SPILL)["stages"]["cs"]
        self.assertEqual(cs["max_user_sgprs"], 14)
        self.assertEqual(cs["max_root_dwords"], 10)          # 14 - base s4
        self.assertEqual(cs["root_dwords_available"], 28)    # 32 - base s4
        self.assertEqual(cs["shaders_using_all_user_sgprs"], 0)
        self.assertNotIn("max_user_sgprs", rsc.census(NO_SPILL)["stages"]["ps"])  # not recorded for ps

    def test_spill_pointer_below_base(self):
        cs = rsc.census(SPILLED)["stages"]["cs"]
        self.assertEqual(cs["shaders_with_spill_pointer_candidate"], 1)
        self.assertEqual(cs["spill_pointer_candidates"], {"s0:ptr64_lo": 1, "s1:ptr64_hi": 1})
        self.assertEqual(cs["shaders_using_all_user_sgprs"], 1)

    def test_base_decides_what_is_below(self):
        # The same GS pointer at s8 is a candidate once the base is moved above it.
        gs = rsc.census(NO_SPILL, base={"gs": 10})["stages"]["gs"]
        self.assertEqual(gs["spill_pointer_candidates"], {"s8:ptr64_lo": 1, "s9:ptr64_hi": 1})
        # A stage with no base reports no candidates rather than guessing one.
        self.assertIsNone(rsc.census(NO_SPILL, base={})["stages"]["ps"]["base"])
        self.assertEqual(rsc.census(NO_SPILL, base={})["stages"]["ps"]["spill_pointer_candidates"], {})

    def test_boundary_at_each_default_base(self):
        # A pointer one SGPR below the base is a candidate; at the base and one above it is not.
        for stage, base in rsc.DEFAULT_BASE.items():
            for sgpr, expected in ((base - 1, 1), (base, 0), (base + 1, 0)):
                index = {"x": {"stage": stage, "user_sgpr_reads": reads((sgpr, "ptr32_lo"))}}
                got = rsc.census(index)["stages"][stage]["shaders_with_spill_pointer_candidate"]
                self.assertEqual(got, expected, f"{stage} s{sgpr}")

    def test_rts0_parts_counted(self):
        carve = [{"name": "x", "parts": {"DXIL": 1, "PRIV": 2}},
                 {"name": "y", "parts": {"DXIL": 1, "RTS0": 3}},
                 {"name": "z"}]
        out = rsc.census(NO_SPILL, carve=carve)
        self.assertEqual(out["containers"], 3)
        self.assertEqual(out["containers_with_rts0"], 1)
        self.assertNotIn("containers", rsc.census(NO_SPILL))

    def test_malformed_reads_are_skipped(self):
        index = {"m": {"stage": "cs", "user_sgpr_reads": [{"kind": "ptr32_lo"}, {"sgpr": "0", "kind": "ptr64_lo"}]}}
        cs = rsc.census(index)["stages"]["cs"]
        self.assertEqual(cs["shaders"], 1)
        self.assertEqual(cs["spill_pointer_candidates"], {})


class CliTest(unittest.TestCase):
    def run_tool(self, index, *args):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(index, f)
            report = os.path.join(tmp, "report.json")
            result = subprocess.run([sys.executable, "-I", TOOL, path, "--json", report, *args],
                                    capture_output=True, text=True)
            summary = None
            if os.path.exists(report):
                with open(report, encoding="utf-8") as f:
                    summary = json.load(f)
            return result, summary

    def test_expect_no_spill_passes_and_fails(self):
        ok, summary = self.run_tool(NO_SPILL, "--expect-no-spill")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(summary["stages"]["cs"]["max_root_dwords"], 10)
        self.assertIn("cs: 2 shaders; base s4; spill-pointer candidates 0", ok.stdout)
        bad, _ = self.run_tool(SPILLED, "--expect-no-spill")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("SPILL POINTER CANDIDATES", bad.stderr)
        plain, _ = self.run_tool(SPILLED)  # without the flag a spill is reported, not a failure
        self.assertEqual(plain.returncode, 0, plain.stderr)

    def test_partial_base_keeps_the_other_defaults(self):
        # The review's case: a PS address read at s7 (below the PS base s8). Overriding only cs must not drop ps.
        index = {"p.ps": {"stage": "ps", "user_sgpr_reads": reads((7, "ptr32_lo"))}}
        default, _ = self.run_tool(index, "--expect-no-spill")
        self.assertEqual(default.returncode, 1, default.stderr)
        partial, summary = self.run_tool(index, "--expect-no-spill", "--base", "cs=4")
        self.assertEqual(partial.returncode, 1, partial.stderr)
        self.assertEqual(summary["stages"]["ps"]["base"], 8)
        moved, summary = self.run_tool(index, "--expect-no-spill", "--base", "ps=7")
        self.assertEqual(moved.returncode, 0, moved.stderr)  # a named stage does take the override
        self.assertEqual(summary["stages"]["ps"]["base"], 7)

    def test_bad_base_is_a_usage_error(self):
        for assignment in ("ps", "ps=bad", "=8", "ps=-1"):
            result, summary = self.run_tool(NO_SPILL, "--base", assignment)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("expected STAGE=SGPR", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIsNone(summary)

    def test_list_index_is_refused(self):
        result, summary = self.run_tool([{"stage": "ps"}])
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a translator index", result.stderr)
        self.assertIsNone(summary)

    def test_json_creates_missing_parent_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            out = os.path.join(tmp, "reports", "sub", "report.json")
            r = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isfile(out))
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertIn("stages", rep)

    def test_json_refuses_a_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            sub = os.path.join(tmp, "other_dir")
            os.makedirs(sub)
            for target in (tmp, sub):
                with self.subTest(target=target):
                    r = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", target],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 2)
                    self.assertIn("is a folder, not a file", r.stderr)
                    self.assertNotIn("Traceback", r.stderr)

    def test_json_refuses_to_overwrite_an_input_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            with open(idx, "rb") as f:
                before = f.read()
            # Direct match
            r1 = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", idx],
                                capture_output=True, text=True)
            self.assertEqual(r1.returncode, 2)
            self.assertIn("must not overwrite an input index", r1.stderr)
            self.assertNotIn("Traceback", r1.stderr)
            with open(idx, "rb") as f:
                self.assertEqual(f.read(), before)

            # Dot-aliased spelling
            alias = os.path.join(tmp, ".", "index.json")
            r2 = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", alias],
                                capture_output=True, text=True)
            self.assertEqual(r2.returncode, 2)
            self.assertIn("must not overwrite an input index", r2.stderr)

            # With --carve
            carve = os.path.join(tmp, "carve.json")
            with open(carve, "w", encoding="utf-8") as f:
                json.dump([{"name": "x"}], f)
            r3 = subprocess.run([sys.executable, "-I", TOOL, idx, "--carve", carve, "--json", carve],
                                capture_output=True, text=True)
            self.assertEqual(r3.returncode, 2)
            self.assertIn("must not overwrite an input index", r3.stderr)

    def test_json_in_input_folder_can_be_written_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            out = os.path.join(tmp, "report.json")
            r1 = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", out],
                                capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertTrue(os.path.isfile(out))
            # Run again: overwriting earlier report must succeed
            r2 = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", out],
                                capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stderr)

    def test_json_failed_write_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            blocker = os.path.join(tmp, "not_a_dir")
            with open(blocker, "wb") as f:
                f.write(b"file")
            out = os.path.join(blocker, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, idx, "--json", out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("error: cannot write", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


    def test_index_saved_with_bom_is_read(self):
        # PowerShell 5.1 Set-Content and some editors save JSON with a UTF-8 BOM; the index is still JSON.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbf" + json.dumps(NO_SPILL).encode("utf-8"))
            result = subprocess.run([sys.executable, "-I", TOOL, path], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("cs: 2 shaders; base s4", result.stdout)


    def test_carve_index_saved_with_bom_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = os.path.join(tmp, "index.json")
            with open(index, "w", encoding="utf-8") as f:
                json.dump(NO_SPILL, f)
            carve = os.path.join(tmp, "carve.json")
            with open(carve, "wb") as f:
                f.write(b"\xef\xbb\xbf" + json.dumps([{"name": "x", "parts": {"RTS0": 1}}]).encode("utf-8"))
            result = subprocess.run([sys.executable, "-I", TOOL, index, "--carve", carve],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("containers: 1, with an RTS0 part: 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
