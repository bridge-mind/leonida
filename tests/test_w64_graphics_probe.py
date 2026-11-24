"""Synthetic failure-path checks for the VS/GS subgroup probes."""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "test" / "shaders" / "w64_graphics.py"
spec = importlib.util.spec_from_file_location("w64_graphics", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProbeFailureTests(unittest.TestCase):
    def run_fixture(self, name, failed_width, diagnostic):
        stage = name[:2]
        count = runner.VS_VERTICES if stage == "vs" else runner.VS_VERTICES // 3
        records = [runner.FIXTURES[name][1](i) for i in range(count)]

        def draw_heap(tools, spirv, fixture, size, tmp):
            if fixture == runner.PROBES[stage]:
                if size == failed_width:
                    return None, diagnostic
                return [(i % 64, 0, 0, 0) for i in range(count)], None
            return records, None

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, clear=True), \
                patch.object(runner, "_draw_heap", side_effect=draw_heap), \
                patch.dict(runner._applied, {}, clear=True):
            spirv = Path(folder, "spirv")
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = runner.main([folder, name])
        return status, output.getvalue()

    def test_probe_dispatch_failure_is_not_a_pass(self):
        for name, width in (("vs_w64_a_skip", 64), ("gs_w64_a_ring", 32)):
            with self.subTest(fixture=name, width=width):
                status, output = self.run_fixture(name, width, "exit 1: synthetic dispatch failure")
                self.assertEqual(status, 1, output)
                self.assertIn(f"{width}-wide {name[:2]} subgroup probe", output)
                self.assertIn("synthetic dispatch failure", output)

    def test_missing_probe_heap_is_not_a_pass(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "exit 0: no heap dump")
        self.assertEqual(status, 1, output)
        self.assertIn("no heap dump", output)

    def test_unsupported_probe_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "unsupported: no geometry stage")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)

    def test_successful_probe_without_width_change_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", None, "")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)


if __name__ == "__main__":
    unittest.main()
