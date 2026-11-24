"""The GPU suite must judge the full Mandelbrot readback, without needing a device."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt
from check import mandelbrot_counts


def reference_frame(width: int) -> bytes:
    return b"".join(struct.pack("<4I", 0, 0, 0, count) for count in mandelbrot_counts(width))


class MandelbrotReadbackTest(unittest.TestCase):
    def run_frame(self, data: bytes | None) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            args = SimpleNamespace(filter="ps_mandelbrot.asm", build=directory, llvm_mc="unused")

            def launch(cmd: list[str], **kwargs: object) -> tuple[int, str]:
                if cmd[0] == sys.executable:
                    (work / "spirv" / "ps_mandelbrot.spv").touch()
                elif Path(cmd[0]).stem == "vk_harness":
                    if data is not None:
                        Path(cmd[-1]).write_bytes(data)
                else:
                    self.fail(f"unexpected command: {cmd}")
                return 0, ""

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=launch),
                  redirect_stdout(output)):
                result = rgt.run_suite(args, directory)
            return result, output.getvalue()

    def test_missing_empty_short_or_partial_frames_fail_the_test(self) -> None:
        for label, data in (
            ("missing", None), ("empty", b""), ("one complete pixel", reference_frame(1)),
            ("partial pixel", bytes(8)),
        ):
            with self.subTest(label=label):
                result, output = self.run_frame(data)
                self.assertEqual(result, 1, output)
                self.assertIn("FAIL  ps_mandelbrot", output)
                # a frame that was never written is reported as a missing readback (TEST-837) before its size is read
                self.assertIn("missing readback" if data is None else "expected 4096 pixels", output)

    def test_complete_frame_still_compares_the_iteration_counts(self) -> None:
        correct = reference_frame(64)
        result, output = self.run_frame(correct)
        self.assertEqual(result, 0, output)
        self.assertIn("4096/4096 iteration counts match", output)
        wrong = bytearray(correct)
        struct.pack_into("<I", wrong, 12, 65)
        result, output = self.run_frame(bytes(wrong))
        self.assertEqual(result, 1, output)
        self.assertIn("1 of 4096 iteration counts differ", output)


if __name__ == "__main__":
    unittest.main()
