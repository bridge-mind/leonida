"""CPU-only unit tests for tools/f16_widen_probe.py: the models, the sweep through a fake oracle, and the CLI contracts."""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "f16_widen_probe.py")

sys.dont_write_bytecode = True
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import f16_widen_probe as probe  # noqa: E402
from hw_oracle import asm as hw_asm, runner as hw_runner  # noqa: E402


class FakeOracle:
    """Plays the GPU: returns the payload-preserving widening for every staged half, lane count from the load."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, rule=None):
        self.rule = rule or probe.h2f_payload

    def run(self, kernels, words=4):
        out = []
        for k in kernels:
            half = int(
                [ln for ln in k.lines if ln.startswith("v_mov_b32 v4, 0x")][-1].split(
                    "0x"
                )[1],
                16,
            )
            lanes = [self.rule(half)] * words
            out.append(lanes)
        return out

    def close(self):
        pass


def run_main(oracle, *args):
    buf = io.StringIO()
    with (
        unittest.mock.patch.object(hw_runner, "open_oracle", return_value=oracle),
        redirect_stdout(buf),
    ):
        rc = probe.main(list(args))
    return rc, buf.getvalue()


class Models(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run(
            [sys.executable, "-I", PROBE, "--self-test"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_the_three_rules_are_distinct(self):
        self.assertEqual(probe.h2f_payload(0x7DDB), 0x7FBB6000)
        self.assertEqual(probe.h2f_canonical(0x7DDB), 0x7FC00000)
        self.assertEqual(probe.h2f_quiet(0x7DDB), 0x7FFB6000)
        for h in (0x3C00, 0xC000):
            self.assertEqual(probe.h2f_payload(h), probe.h2f_canonical(h))
            self.assertEqual(probe.h2f_payload(h), probe.h2f_quiet(h))


class Sweep(unittest.TestCase):
    def test_every_case_is_payload_on_the_measured_rule(self):
        rc, out = run_main(FakeOracle(), "--waves", "32")
        self.assertEqual(rc, 0, out)
        oracle = FakeOracle()
        report = probe.sweep(oracle, hw_asm, 32)
        self.assertEqual(
            len(report), 19
        )  # three formats x two forms x three widths, plus the ALU case
        self.assertIn("alu/v_cvt_f32_f16", report)
        for case, halves in report.items():
            for label, entry in halves.items():
                with self.subTest(case=case, half=label):
                    self.assertTrue(
                        all(c == "payload" for c in entry["classes"]), entry
                    )

    def test_a_canonicalising_gpu_is_reported_as_such(self):
        report = probe.sweep(FakeOracle(probe.h2f_canonical), hw_asm, 32)
        entry = report["fmt13/16_FLOAT/buffer_x"]
        self.assertEqual(entry["snan_7ddb"]["classes"], ["canonical"])
        self.assertEqual(entry["qnan_7fe6"]["classes"], ["canonical"])
        self.assertEqual(
            entry["one_3c00"]["classes"], ["payload"]
        )  # a finite half is the same either way

    def test_json_report_has_the_measured_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "report.json")
            rc, out = run_main(FakeOracle(), "--waves", "32", "--json", path)
            self.assertEqual(rc, 0, out)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["arch"], "gfx1036")
            entry = data["waves"]["32"]["fmt13/16_FLOAT/buffer_x"]["snan_7ddb"]
            self.assertEqual(entry["half"], "0x7ddb")
            self.assertEqual(entry["words"], ["0x7fbb6000"])
            self.assertEqual(entry["classes"], ["payload"])


class Contracts(unittest.TestCase):
    def test_non_gfx10_or_missing_oracle_exits_77(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"
        }
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run(
            [sys.executable, "-I", PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)

    def test_invalid_waves_rejected_before_open_oracle(self):
        for bad in ("", "   ", "abc", "32,foo", "16", "32,128"):
            with self.subTest(waves=bad):
                err = io.StringIO()
                with (
                    unittest.mock.patch.object(
                        hw_runner,
                        "open_oracle",
                        side_effect=AssertionError("open_oracle called"),
                    ),
                    redirect_stderr(err),
                ):
                    rc = probe.main(["--waves", bad])
                self.assertEqual(rc, 2)
                self.assertIn("error:", err.getvalue())

    def test_invalid_json_rejected_before_open_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_ancestor = os.path.join(tmp, "not_a_dir")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")
            for bad in (
                "",
                "   ",
                "bad\x00.json",
                tmp,
                os.path.join(file_ancestor, "out.json"),
            ):
                with self.subTest(json_path=bad):
                    err = io.StringIO()
                    with (
                        unittest.mock.patch.object(
                            hw_runner,
                            "open_oracle",
                            side_effect=AssertionError("open_oracle called"),
                        ),
                        redirect_stderr(err),
                    ):
                        rc = probe.main(["--json", bad])
                    self.assertEqual(rc, 2)
                    self.assertIn("error:", err.getvalue())

    def test_write_json_report_uses_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "sub", "dir", "report.json")
            err = probe.write_json_report(
                nested, {"arch": "gfx1036", "waves": {"32": {}}}
            )
            self.assertIsNone(err)
            with open(nested, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw.decode("utf-8"))["arch"], "gfx1036")

    def test_write_json_oserror_returns_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.json")
            out, err = io.StringIO(), io.StringIO()
            with (
                unittest.mock.patch.object(
                    hw_runner, "open_oracle", return_value=FakeOracle()
                ),
                unittest.mock.patch.object(hw_asm, "family", return_value="gfx10"),
                unittest.mock.patch("builtins.open", side_effect=OSError("disk full")),
                redirect_stdout(out),
                redirect_stderr(err),
            ):
                rc = probe.main(["--waves", "32", "--json", out_path])
            self.assertEqual(rc, 1)
            self.assertIn("disk full", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_kernels_stage_then_load_the_same_half(self):
        k = probe._kernel(hw_asm, "buffer_load_format_xy", 29, 0x7FE6, 64)
        self.assertEqual(k.wave, 64)
        self.assertIn("buffer_store_dword v4, off, s[8:11], 0 offset:32 glc", k.lines)
        self.assertIn("s_waitcnt_vscnt null, 0x0", k.lines)
        self.assertIn(
            "buffer_load_format_xy v[4:5], off, s[8:11], 0 offset:32", k.lines
        )
        self.assertIn("buffer_store_dword v5, off, s[8:11], 0 offset:4", k.lines)
        self.assertEqual(k.lines[-1], "s_endpgm")
        t = probe._kernel(hw_asm, "tbuffer_load_format_x", 13, 0x7DDB, 32)
        self.assertTrue(
            any(
                ln.startswith("tbuffer_load_format_x v4,") and "format:13" in ln
                for ln in t.lines
            )
        )
        self.assertIn("v_mov_b32 v4, 0x7ddb", t.lines)


if __name__ == "__main__":
    unittest.main()
