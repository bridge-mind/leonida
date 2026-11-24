"""Tests for pure-Python SPIR-V assembly and --disasm-dir / --no-validate in run_gpu_tests.py and rdna2spirv.py."""
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

import rdna2spirv
import run_gpu_tests


class PythonAssemblerTests(unittest.TestCase):
    def test_resolve_assembler_auto_and_explicit(self) -> None:
        self.assertEqual(run_gpu_tests.resolve_assembler("python"), "python")
        self.assertEqual(run_gpu_tests.resolve_assembler("spirv-as"), "spirv-as")
        with patch.object(run_gpu_tests.shutil, "which", return_value="/usr/bin/spirv-as"):
            self.assertEqual(run_gpu_tests.resolve_assembler("auto"), "spirv-as")
        with patch.object(run_gpu_tests.shutil, "which", return_value=None):
            self.assertEqual(run_gpu_tests.resolve_assembler("auto"), "python")
        with self.assertRaises(ValueError):
            run_gpu_tests.resolve_assembler("bogus")

    def test_spirv_as_python_assembles_helper_shaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, src in (
                ("vs_tri.spv", run_gpu_tests.VS_TRI_SPVASM),
                ("fs_passthru.spv", run_gpu_tests.make_fs_passthru(0)),
            ):
                out = Path(tmp) / name
                run_gpu_tests.spirv_as(src, str(out), assembler="python")
                data = out.read_bytes()
                self.assertGreaterEqual(len(data), 20)
                self.assertEqual(struct.unpack("<I", data[:4])[0], 0x07230203)

    def test_rdna2spirv_assemble_spirv_no_validate_skips_spirv_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spvasm = Path(tmp) / "fs_passthru.spvasm"
            spv = Path(tmp) / "fs_passthru.spv"
            asm = run_gpu_tests.make_fs_passthru(0)
            with patch.object(rdna2spirv, "_run_spirv_tool",
                              side_effect=AssertionError("external spirv tool called")) as run_tool:
                res = rdna2spirv.assemble_spirv(str(spvasm), str(spv), asm_text=asm,
                                                assembler="python", validate=False)
            run_tool.assert_not_called()
            self.assertEqual(res.returncode, 0)
            self.assertTrue(spv.is_file())
            self.assertEqual(struct.unpack("<I", spv.read_bytes()[:4])[0], 0x07230203)

    def test_rdna2spirv_cli_python_assembler_and_no_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disasm = root / "disasm"
            fssm = root / "fssm"
            out = root / "out"
            disasm.mkdir()
            fssm.mkdir()
            # Minimal valid compute listing (s_endpgm) and synthetic 0x148-byte FSSM header.
            (disasm / "cs_noop.s").write_text("; synthetic\n\ts_endpgm\n", encoding="utf-8", newline="\n")
            hdr = bytearray(0x148)
            hdr[0:4] = b"FSSM"
            struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
            (fssm / "cs_noop.fssm").write_bytes(bytes(hdr))
            stdout = io.StringIO()
            with patch("sys.stdout", stdout), \
                    patch.object(sys, "argv", [
                        "rdna2spirv.py", str(disasm), "--out", str(out), "--assembler", "python", "--no-validate",
                    ]), \
                    patch.object(rdna2spirv, "_run_spirv_tool",
                                 side_effect=AssertionError("external spirv tool called")):
                rc = rdna2spirv.main()
            self.assertIn(rc, (0, None))
            self.assertIn("1 shaders: 1 translated, 1 assembled (spirv-val skipped)", stdout.getvalue())
            spv = out / "cs_noop.spv"
            self.assertTrue(spv.is_file())
            self.assertEqual(struct.unpack("<I", spv.read_bytes()[:4])[0], 0x07230203)
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertIn("cs_noop", index)

    def test_missing_disasm_dir_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            ext = ".exe" if os.name == "nt" else ""
            for prog in run_gpu_tests.TEST_PROGRAMS:
                (build / (prog + ext)).write_bytes(b"")
            missing = Path(tmp) / "no_such_disasm"
            err = io.StringIO()
            with patch("sys.stderr", err), \
                    patch.object(sys, "argv", ["run_gpu_tests.py", "--build", str(build), "--disasm-dir", str(missing)]):
                rc = run_gpu_tests.main()
            self.assertEqual(rc, 2)
            self.assertIn("--disasm-dir is not a directory", err.getvalue())

    def test_run_suite_reuses_existing_work_listings(self) -> None:
        listing = b"; synthetic\n\ts_endpgm\n"
        run_command = run_gpu_tests.run_cmd
        for placement in ("in-place", "hard-link", "separate"):
            with self.subTest(placement=placement), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "work"
                working = work / "disasm" / "cs_lds.s"
                working.parent.mkdir(parents=True)
                source = working if placement == "in-place" else Path(tmp) / "source" / working.name
                source.parent.mkdir(exist_ok=True)
                source.write_bytes(listing)
                if placement == "hard-link":
                    try:
                        os.link(source, working)
                    except OSError as exc:
                        self.skipTest(f"hard links unavailable: {exc}")

                args = SimpleNamespace(assembler="python", no_validate=True, disasm_dir=str(source.parent),
                                       filter="cs_lds", build=str(work), require_pass=True)

                def run(command, env=None, gpu=True):
                    if not gpu:
                        selected = Path(command[3])
                        self.assertNotEqual(selected, working.parent)
                        self.assertEqual((selected / working.name).read_bytes(), listing)
                        return run_command(command, env=env, gpu=False)
                    self.assertTrue(Path(command[1]).is_file())
                    return 0, "synthetic GPU result"

                with (patch.object(run_gpu_tests, "run_cmd", side_effect=run) as launches,
                      patch.object(sys, "stdout", io.StringIO())):
                    self.assertEqual(run_gpu_tests.run_suite(args, str(work)), 0)
                self.assertEqual(launches.call_count, 2)
                self.assertEqual(source.read_bytes(), listing)
                self.assertEqual(working.read_bytes(), listing)


if __name__ == "__main__":
    unittest.main()
