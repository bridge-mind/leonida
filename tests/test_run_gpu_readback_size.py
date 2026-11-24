"""A missing or partial GPU readback fails that test only; it doesn't stop the suite.

    python -I tests/test_run_gpu_readback_size.py

scripts/run_gpu_tests.py unpacked the readbacks of ps_interp_f16_high (also drawn by ms_f16_pack) and ps_ij_computed
with a bare struct.iter_unpack("<4I", ...): a file whose size isn't a multiple of 16 raised struct.error, a missing
one FileNotFoundError, and nothing caught either, so the run ended there and every other test's result was lost.
The suite is run here with the harness stood in for (as tests/test_run_gpu_mandelbrot.py of #3442 does): no device,
no llvm-mc, no game files.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt  # noqa: E402


class ReadbackSizeTest(unittest.TestCase):
    def run_test(self, name: str, data: bytes | None) -> tuple[int, str]:
        """run_suite on one pixel test whose harness writes `data` as its readback (None: writes nothing)."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            args = SimpleNamespace(filter=f"{name}.asm", build=directory, llvm_mc="unused")

            def launch(cmd: list[str], **kwargs: object) -> tuple[int, str]:
                if cmd[0] == sys.executable:
                    (work / "spirv" / f"{name}.spv").touch()
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
                  # ps_ij_computed is skipped under either PerVertexKHR fallback, which is on by default on a device
                  # where PerVertexKHR does not compile (or with no device): this test is about its readback
                  patch.object(rgt, "bary_fallback", return_value=False),
                  patch.object(rgt, "f16_interp_fallback", return_value=False),
                  redirect_stdout(output)):
                result = rgt.run_suite(args, directory)
            return result, output.getvalue()

    def test_bad_readbacks_fail_the_test_without_stopping_the_suite(self) -> None:
        for name in ("ps_interp_f16_high", "ps_ij_computed"):
            for label, data in (("missing", None), ("empty", b""), ("partial pixel", bytes(8)),
                                ("one byte short of a frame", bytes(64 * 64 * 16 - 1))):
                with self.subTest(test=name, readback=label):
                    result, output = self.run_test(name, data)
                    self.assertEqual(result, 1, output)
                    self.assertIn(f"FAIL  {name}", output)
                    self.assertIn("0/1 passed", output)  # the summary line: the suite ran to its end


if __name__ == "__main__":
    unittest.main()
