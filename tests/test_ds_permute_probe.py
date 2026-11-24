"""CPU-only unit tests for tools/ds_permute_probe.py (SHADER-133)."""
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "ds_permute_probe.py")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import ds_permute_probe  # noqa: E402


class _FakeUnavailable(Exception):
    pass


def _fake_hw_oracle(open_side_effect=None):
    asm_mod = types.SimpleNamespace(family=lambda arch: "gfx10")
    open_oracle = mock.Mock(side_effect=open_side_effect or _FakeUnavailable("no gpu"))
    runner_mod = types.SimpleNamespace(Unavailable=_FakeUnavailable, open_oracle=open_oracle)
    pkg = types.SimpleNamespace(asm=asm_mod, runner=runner_mod)
    return pkg, asm_mod, runner_mod, open_oracle


class DsPermuteProbeTests(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--self-test"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_missing_oracle_exits_77(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run([sys.executable, "-I", PROBE], env=env,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)

    def test_invalid_waves_rejected_before_open_oracle(self):
        pkg, asm_mod, runner_mod, open_oracle = _fake_hw_oracle()
        with mock.patch.dict(sys.modules, {"hw_oracle": pkg, "hw_oracle.asm": asm_mod, "hw_oracle.runner": runner_mod}):
            for bad in ("", " , ", "abc", "32,abc", "16", "128"):
                with self.subTest(waves=bad):
                    stderr = io.StringIO()
                    with mock.patch.object(sys, "stderr", stderr):
                        rc = ds_permute_probe.main(["--waves", bad])
                    self.assertEqual(rc, 2)
                    self.assertIn("error:", stderr.getvalue())
                    open_oracle.assert_not_called()

    def test_invalid_json_paths_rejected_before_open_oracle(self):
        pkg, asm_mod, runner_mod, open_oracle = _fake_hw_oracle()
        with tempfile.TemporaryDirectory() as tmp:
            existing_dir = os.path.join(tmp, "dir")
            os.mkdir(existing_dir)
            file_ancestor = os.path.join(tmp, "not_a_dir")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")

            bad_paths = [
                "",
                "   ",
                "bad\x00out.json",
                existing_dir,
                os.path.join(tmp, "trailing") + os.sep,
                os.path.join(file_ancestor, "sub", "out.json"),
            ]
            with mock.patch.dict(sys.modules, {"hw_oracle": pkg, "hw_oracle.asm": asm_mod,
                                               "hw_oracle.runner": runner_mod}):
                for bad in bad_paths:
                    with self.subTest(json_out=bad):
                        stderr = io.StringIO()
                        with mock.patch.object(sys, "stderr", stderr):
                            rc = ds_permute_probe.main(["--json", bad])
                        self.assertEqual(rc, 2)
                        self.assertIn("error:", stderr.getvalue())
                        open_oracle.assert_not_called()

    def test_valid_nested_json_does_not_create_dirs_when_oracle_unavailable(self):
        pkg, asm_mod, runner_mod, open_oracle = _fake_hw_oracle()
        with tempfile.TemporaryDirectory() as tmp:
            nested_parent = os.path.join(tmp, "a", "b")
            json_out = os.path.join(nested_parent, "report.json")
            with mock.patch.dict(sys.modules, {"hw_oracle": pkg, "hw_oracle.asm": asm_mod,
                                               "hw_oracle.runner": runner_mod}):
                stderr = io.StringIO()
                with mock.patch.object(sys, "stderr", stderr):
                    rc = ds_permute_probe.main(["--json", json_out])
            self.assertEqual(rc, 77)
            open_oracle.assert_called_once()
            self.assertFalse(os.path.exists(os.path.join(tmp, "a")))

    def test_write_json_report_creates_parents_and_writes_lf_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "nested", "dir", "out.json")
            payload = {"arch": "gfx1030", "waves": {"32": {"cases_total": 1, "cases_ok": 1}}}
            err = ds_permute_probe.write_json_report(out_path, payload)
            self.assertIsNone(err)
            with open(out_path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw.decode("utf-8")), payload)

    def test_write_json_report_and_main_catch_oserror_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            with mock.patch("builtins.open", side_effect=PermissionError("denied")):
                err = ds_permute_probe.write_json_report(out_path, {"ok": True})
            self.assertIsNotNone(err)
            self.assertIn("cannot write --json", err)

            oracle = types.SimpleNamespace(arch="gfx1030")
            pkg, asm_mod, runner_mod, _ = _fake_hw_oracle(open_side_effect=lambda: oracle)
            fake_probe = {"cases_total": 1, "cases_ok": 1, "cases": {}}
            with mock.patch.dict(sys.modules, {"hw_oracle": pkg, "hw_oracle.asm": asm_mod,
                                               "hw_oracle.runner": runner_mod}), \
                 mock.patch.object(ds_permute_probe, "probe_wave", return_value=fake_probe), \
                 mock.patch("builtins.open", side_effect=OSError("disk full")):
                stdout, stderr = io.StringIO(), io.StringIO()
                with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
                    rc = ds_permute_probe.main(["--waves", "32", "--json", out_path])
            self.assertEqual(rc, 1)
            self.assertIn("error:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
