"""image_fuzz must not overwrite a caller's --work files or reuse another run's outputs."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import image_fuzz as imf


class ImageFuzzWorkIsolation(unittest.TestCase):
    def test_existing_files_and_previous_runs_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "work"
            root.mkdir()
            build = Path(tmp) / "build"
            build.mkdir()
            (build / (imf.HARNESS + (".exe" if os.name == "nt" else ""))).touch()
            names = ("disasm/ps_image_load.s", "disasm/vs_ids.s", "ps_image_load.dxbc", "vs_ids.dxbc",
                     "texels.bin", "spec.txt", "out.raw")
            sentinels = {}
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"caller data")
                sentinels[path] = b"caller data"

            def assemble(cmd, **kwargs):
                return subprocess.CompletedProcess(cmd, 0, "s_endpgm\n", "")

            def translate(work, pc_tail=False):
                spirv = Path(work) / "spirv"
                spirv.mkdir()
                return str(spirv)

            def harness(cmd, **kwargs):
                return subprocess.CompletedProcess(cmd, 77, "unsupported: synthetic test\n", "")

            outputs = []
            for _ in range(2):
                out = io.StringIO()
                argv = ["image_fuzz.py", "--programs", "1", "--formats", "R8G8B8A8_UNORM",
                        "--work", str(root), "--build", str(build)]
                with (patch.object(sys, "argv", argv), patch.object(imf.mc_command, "run", side_effect=assemble),
                      patch.object(imf, "translate", side_effect=translate),
                      patch.object(imf.subprocess, "run", side_effect=harness), contextlib.redirect_stdout(out)):
                    self.assertEqual(imf.main(), 0)
                outputs.append(out.getvalue())
                self.assertIn("1 skipped", outputs[-1])
                for path, content in sentinels.items():
                    self.assertEqual(path.read_bytes(), content, f"image_fuzz overwrote {path}")

            runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "disasm")
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertTrue((run / "disasm" / "ps_image_load.s").is_file())
                self.assertTrue((run / "texels.bin").is_file())
                self.assertTrue((run / "spec.txt").is_file())
                self.assertTrue(any(str(run) in output for output in outputs))
            self.assertNotEqual(runs[0], runs[1])

    def test_concurrent_processes_get_distinct_folders(self):
        child = r'''
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import image_fuzz

def run_case(*args, **kwargs):
    (Path(args[4]) / "worker.bin").write_bytes(str(os.getpid()).encode("ascii"))
    return True

image_fuzz.run_case = run_case
sys.argv = ["image_fuzz.py", "--programs", "1", "--work", sys.argv[2]]
raise SystemExit(image_fuzz.main())
'''
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "work"
            parent.mkdir()
            sentinel = parent / "texels.bin"
            sentinel.write_bytes(b"caller data")
            tools = str(Path(__file__).resolve().parents[1] / "tools")
            command = [sys.executable, "-I", "-c", child, tools, str(parent)]
            workers = []
            try:
                for _ in range(2):
                    workers.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE, text=True))
                results = [worker.communicate(timeout=20) for worker in workers]
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.communicate()
            paths = []
            for worker, (stdout, stderr) in zip(workers, results):
                self.assertEqual(worker.returncode, 0, (stdout, stderr))
                lines = [line.removeprefix("work dir ") for line in stdout.splitlines() if line.startswith("work dir ")]
                self.assertEqual(len(lines), 1, stdout)
                paths.append(Path(lines[0]))
            self.assertNotEqual(paths[0], paths[1])
            self.assertTrue(all(path.parent == parent for path in paths))
            self.assertNotEqual((paths[0] / "worker.bin").read_bytes(), (paths[1] / "worker.bin").read_bytes())
            self.assertEqual(sentinel.read_bytes(), b"caller data")

    def test_existing_file_as_work_reports_one_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "not-a-directory"
            parent.write_bytes(b"caller data")
            script = Path(__file__).resolve().parents[1] / "tools" / "image_fuzz.py"
            result = subprocess.run([sys.executable, "-I", str(script), "--programs", "1", "--work", str(parent)],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("ERROR:", result.stderr)
            self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(parent.read_bytes(), b"caller data")

    def test_work_error_closes_an_open_hardware_oracle(self):
        from hw_oracle import runner

        class Oracle:
            arch = "gfx1030"
            device = "synthetic device"
            closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "not-a-directory"
            parent.write_bytes(b"caller data")
            oracle = Oracle()
            argv = ["image_fuzz.py", "--oracle", "hw", "--programs", "1", "--work", str(parent)]
            stderr = io.StringIO()
            with (patch.object(sys, "argv", argv), patch.object(runner, "open_oracle", return_value=oracle) as opened,
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr)):
                self.assertEqual(imf.main(), 2)
            # since #4087 a --work that is a file is refused before the oracle is opened; an opened one is closed
            self.assertTrue(oracle.closed or not opened.called)
            self.assertIn("ERROR:", stderr.getvalue())
            self.assertEqual(parent.read_bytes(), b"caller data")

    def test_setup_failure_reports_the_retained_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "new" / "work"
            out = io.StringIO()
            argv = ["image_fuzz.py", "--programs", "1", "--work", str(parent)]
            with (patch.object(sys, "argv", argv),
                  patch.object(imf, "assemble", side_effect=SystemExit("synthetic setup error")),
                  contextlib.redirect_stdout(out)):
                with self.assertRaises(SystemExit):
                    imf.main()
            runs = list(parent.iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].is_dir())
            self.assertIn(str(runs[0]), out.getvalue())

    def test_self_test_does_not_create_a_work_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "not-created"
            with (patch.object(sys, "argv", ["image_fuzz.py", "--self-test", "--work", str(parent)]),
                  patch.object(imf, "self_test", return_value=0)):
                self.assertEqual(imf.main(), 0)
            self.assertFalse(parent.exists())


if __name__ == "__main__":
    unittest.main()
