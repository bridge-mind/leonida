"""tex_headers.py report paths: never overwrite a carve manifest or a scanned .bin stream.

Run with ``python -I tests/test_tex_headers_report_paths.py``. Set ``LEONIDA_TEX_HEADERS`` to another
copy of the tool to use as a negative control (main's version writes the report over the .bin it just
scanned). All inputs are hand-built; no game data is used.
"""
import contextlib
import io
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_TEX_HEADERS", str(ROOT / "tools" / "tex_headers.py")))
for folder in (str(ROOT), str(ROOT / "tools")):
    if folder not in sys.path:
        sys.path.insert(0, folder)
import tex_headers


def make_carve(folder):
    """One synthetic BC1 texture: a CPU descriptor blob followed by its GPU data blob."""
    carve = pathlib.Path(folder) / "carve"
    arch = carve / "common"
    arch.mkdir(parents=True)
    cpu_off, gpu_off = 0x1000, 0x2000
    qwords = [0] * 32
    qwords[0] = 0
    qwords[1] = 0x123456789ABCDEF0
    k = 10
    qwords[k - 2] = 0xCAFEBABE00000001
    qwords[k - 1] = 0x980
    qwords[k] = 0x60000000 + 0x1000
    qwords[k + 4] = struct.unpack("<Q", struct.pack("<HHHBB", 512, 256, 1, 0x42, 71))[0]
    qwords[k + 5] = (9 << 16) | 1
    cpu = struct.pack(f"<{len(qwords)}Q", *qwords)
    gpu = b"\x00" * (0x1000 + 4096)
    cpu_path = arch / f"{cpu_off:012x}.bin"
    gpu_path = arch / f"{gpu_off:012x}.bin"
    cpu_path.write_bytes(cpu)
    gpu_path.write_bytes(gpu)
    index = {"files": [
        {"off": cpu_off, "head": cpu[:16].hex(), "size": len(cpu)},
        {"off": gpu_off, "head": gpu[:16].hex(), "size": len(gpu)},
    ]}
    (arch / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return carve, cpu_path, gpu_path


class ReportPathTest(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_out_cannot_overwrite_a_scanned_blob(self):
        for which in ("cpu", "gpu"):
            with self.subTest(blob=which), tempfile.TemporaryDirectory() as td:
                carve, cpu_path, gpu_path = make_carve(td)
                blob = cpu_path if which == "cpu" else gpu_path
                before = blob.read_bytes()
                result = self.run_tool(carve, "--out", blob)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("would overwrite an input blob", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(blob.read_bytes(), before)

    def test_out_cannot_be_a_hard_link_to_a_scanned_blob(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, _ = make_carve(td)
            alias = pathlib.Path(td) / "report.json"
            try:
                os.link(cpu_path, alias)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"hard links unavailable: {e}")
            before = cpu_path.read_bytes()
            self.assertTrue(os.path.samefile(cpu_path, alias))
            result = self.run_tool(carve, "--out", alias)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite an input blob", result.stderr)
            self.assertEqual(cpu_path.read_bytes(), before)

    def test_out_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            carve, _, _ = make_carve(td)
            outdir = pathlib.Path(td) / "outdir"
            outdir.mkdir()
            result = self.run_tool(carve, "--out", outdir)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("is a directory", result.stderr)

    def test_report_inside_the_carve_folder_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, gpu_path = make_carve(td)
            cpu, gpu = cpu_path.read_bytes(), gpu_path.read_bytes()
            out = carve / "common" / "textures_index.json"
            result = self.run_tool(carve, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["w"], rows[0]["h"], rows[0]["fmt_name"]),
                             (512, 256, "BC1_UNORM"))
            self.assertEqual(cpu_path.read_bytes(), cpu)
            self.assertEqual(gpu_path.read_bytes(), gpu)


class PrefilterTest(unittest.TestCase):
    def run_main(self, *args):
        calls = []
        real = tex_headers.same_output_path

        def counting(first, second):
            calls.append((first, second))
            return real(first, second)

        with mock.patch.object(tex_headers, "same_output_path", counting), \
                mock.patch.object(sys, "argv", ["tex_headers.py", *map(str, args)]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                tex_headers.main()
            except SystemExit as exc:
                return calls, exc.code
        return calls, None

    def test_existing_report_json_skips_the_per_file_guard(self):
        with tempfile.TemporaryDirectory() as td:
            carve, _, _ = make_carve(td)
            out = pathlib.Path(td) / "report.json"
            out.write_text("old report", encoding="utf-8")
            calls, code = self.run_main(carve, "--out", out)
            self.assertIsNone(code)
            self.assertEqual(calls, [])
            self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))), 1)

    def test_hard_link_report_still_calls_the_guard(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, _ = make_carve(td)
            alias = pathlib.Path(td) / "report.json"
            os.link(cpu_path, alias)
            before = cpu_path.read_bytes()
            calls, code = self.run_main(carve, "--out", alias)
            self.assertEqual(code, 2)
            self.assertGreater(len(calls), 0)
            self.assertEqual(cpu_path.read_bytes(), before)

    def test_input_named_report_still_calls_the_guard(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, _ = make_carve(td)
            before = cpu_path.read_bytes()
            calls, code = self.run_main(carve, "--out", cpu_path)
            self.assertEqual(code, 2)
            self.assertGreater(len(calls), 0)
            self.assertEqual(cpu_path.read_bytes(), before)

    def test_symlinked_blob_report_still_calls_the_guard(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, _ = make_carve(td)
            report = pathlib.Path(td) / "report.json"
            report.write_text("old report", encoding="utf-8")
            alias = cpu_path.parent / "ffffffffffff.bin"
            try:
                alias.symlink_to(report)
            except OSError:
                self.skipTest("symbolic links unavailable")
            calls, code = self.run_main(carve, "--out", report)
            self.assertEqual(code, 2)
            self.assertGreater(len(calls), 0)
            self.assertEqual(report.read_text(encoding="utf-8"), "old report")


if __name__ == "__main__":
    unittest.main()
