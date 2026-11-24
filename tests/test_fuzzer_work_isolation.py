"""The translator fuzzers must not remove or overwrite files in a caller's --work folder."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz
import valu_fuzz


class FuzzerWorkIsolation(unittest.TestCase):
    def test_existing_files_and_other_runs_survive(self):
        cases = (
            (lds_fuzz, [], lds_fuzz.programs(1, 1, (32,))[0]),
            (valu_fuzz, ["--family", "f16"], valu_fuzz.programs("f16", 1, 1, (32,))[0]),
        )
        for module, extra, program in cases:
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sentinels = {}
                for subdir in ("disasm", "fssm", "spirv"):
                    path = root / subdir / "keep.txt"
                    path.parent.mkdir()
                    path.write_bytes(b"caller data")
                    sentinels[path] = b"caller data"
                for suffix in (".asm", ".bin"):
                    path = root / (program.name + suffix)
                    path.write_bytes(b"caller artifact")
                    sentinels[path] = b"caller artifact"

                def assemble(cmd, **kwargs):
                    return subprocess.CompletedProcess(cmd, 0, "s_endpgm\n", "")

                def translate(cmd, **kwargs):
                    listing_dir = Path(cmd[3])
                    out = Path(cmd[cmd.index("--out") + 1])
                    out.mkdir(parents=True, exist_ok=True)
                    index = {}
                    for listing in listing_dir.glob("*.s"):
                        index[listing.stem] = {}
                        (out / (listing.stem + ".spv")).write_bytes(b"synthetic")
                    (out / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                outputs = []
                for _ in range(2):
                    output = io.StringIO()
                    argv = [module.__name__ + ".py", "--programs", "1", "--wave", "32", "--work", str(root), *extra]
                    with (patch.object(sys, "argv", argv),
                          patch.object(module.mc_command, "run", side_effect=assemble),
                          patch.object(module.subprocess, "run", side_effect=translate),
                          patch.object(module, "run_gpu", return_value=(77, "synthetic skip")),
                          contextlib.redirect_stdout(output)):
                        self.assertEqual(module.main(), 2)  # no GPU programs ran
                    outputs.append(output.getvalue())
                    for path, content in sentinels.items():
                        self.assertTrue(path.exists(), f"{module.__name__} removed {path}")
                        self.assertEqual(path.read_bytes(), content, f"{module.__name__} overwrote {path}")

                runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"disasm", "fssm", "spirv"})
                self.assertEqual(len(runs), 2, "each invocation needs its own retained output folder")
                for run in runs:
                    self.assertTrue((run / (program.name + ".asm")).is_file())
                    self.assertTrue((run / "spirv" / "index.json").is_file())
                self.assertNotEqual(runs[0], runs[1])
                self.assertTrue(any(str(runs[0]) in out for out in outputs))
                self.assertTrue(any(str(runs[1]) in out for out in outputs))

    def test_failed_translation_reports_retained_output_and_creates_parent(self):
        for module, extra in ((lds_fuzz, []), (valu_fuzz, ["--family", "f16"])):
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "new" / "work"
                output = io.StringIO()
                argv = [module.__name__ + ".py", "--programs", "1", "--wave", "32", "--work", str(parent), *extra]
                with (patch.object(sys, "argv", argv),
                      patch.object(module, "translate", side_effect=RuntimeError("synthetic failure")),
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(module.main(), 2)
                runs = list(parent.iterdir())
                self.assertEqual(len(runs), 1)
                self.assertTrue(runs[0].is_dir())
                self.assertIn(str(runs[0]), output.getvalue())
                self.assertIn("synthetic failure", output.getvalue())

    def test_existing_file_as_work_reports_one_error(self):
        for module, extra in ((lds_fuzz, []), (valu_fuzz, ["--family", "f16"])):
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "not-a-directory"
                parent.write_bytes(b"caller data")
                script = Path(__file__).resolve().parents[1] / "tools" / (module.__name__ + ".py")
                cmd = [sys.executable, "-I", str(script), "--programs", "1", "--wave", "32", "--work", str(parent), *extra]
                result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("ERROR: cannot create work directory", result.stderr)
                self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(parent.read_bytes(), b"caller data")

    def test_work_error_closes_an_open_hardware_oracle(self):
        from hw_oracle import runner

        class Oracle:
            device = "synthetic device"
            arch = "gfx1030"
            closed = False

            def close(self):
                self.closed = True

        for module, extra in ((lds_fuzz, []), (valu_fuzz, ["--family", "f16"])):
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "not-a-directory"
                parent.write_bytes(b"caller data")
                oracle = Oracle()
                argv = [module.__name__ + ".py", "--programs", "1", "--wave", "32", "--work", str(parent),
                        "--oracle", "translator,hw", *extra]
                stderr = io.StringIO()
                stats = ({32: {"match": 1, "mismatch": 0, "rejected": 0}}, [])
                with (patch.object(sys, "argv", argv), patch.object(runner, "open_oracle", return_value=oracle),
                      patch.object(module, "run_hw", return_value=stats), contextlib.redirect_stdout(io.StringIO()),
                      contextlib.redirect_stderr(stderr)):
                    self.assertEqual(module.main(), 2)
                self.assertTrue(oracle.closed)
                self.assertIn("ERROR: cannot create work directory", stderr.getvalue())
                self.assertEqual(parent.read_bytes(), b"caller data")

    def test_child_creation_error_preserves_existing_files(self):
        for module, extra in ((lds_fuzz, []), (valu_fuzz, ["--family", "f16"])):
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "work"
                parent.mkdir()
                sentinel = parent / "keep.txt"
                sentinel.write_bytes(b"caller data")
                argv = [module.__name__ + ".py", "--programs", "1", "--wave", "32", "--work", str(parent), *extra]
                stderr = io.StringIO()
                with (patch.object(sys, "argv", argv),
                      patch.object(module.tempfile, "mkdtemp", side_effect=PermissionError("synthetic deny")),
                      contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr)):
                    self.assertEqual(module.main(), 2)
                self.assertIn("ERROR: cannot create work directory", stderr.getvalue())
                self.assertIn("synthetic deny", stderr.getvalue())
                self.assertEqual(sentinel.read_bytes(), b"caller data")

    def test_model_only_does_not_create_a_work_folder(self):
        for module, extra in ((lds_fuzz, []), (valu_fuzz, ["--family", "f16"])):
            with self.subTest(fuzzer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "not-created"
                argv = [module.__name__ + ".py", "--programs", "1", "--wave", "32", "--model-only",
                        "--work", str(parent), *extra]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(), 0)
                self.assertFalse(parent.exists())


if __name__ == "__main__":
    unittest.main()
