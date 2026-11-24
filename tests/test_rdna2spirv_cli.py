"""rdna2spirv.py command-line validation and SPIR-V output tests.

    python -I tests/test_rdna2spirv_cli.py

Synthetic inputs only. The end-to-end retry test also needs SPIRV-Tools on PATH.
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
TOOL = os.path.join(ROOT, "tools", "rdna2spirv.py")

import rdna2spirv  # noqa: E402


class ArgumentTypes(unittest.TestCase):
    def test_limit(self):
        self.assertEqual(rdna2spirv._limit_arg("200"), 200)
        self.assertEqual(rdna2spirv._limit_arg("1"), 1)
        for text in ("0", "-1", "-200", "x", ""):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._limit_arg(text)

    def test_shard(self):
        self.assertEqual(rdna2spirv._shard_arg("0/8"), (0, 8))
        self.assertEqual(rdna2spirv._shard_arg("7/8"), (7, 8))
        self.assertEqual(rdna2spirv._shard_arg("0/1"), (0, 1))
        for text in ("0/0", "8/8", "9/8", "-1/8", "1/-8", "a/b", "0", "0/8/1", "", "/8"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._shard_arg(text)

    def test_max_structured_depth(self):
        self.assertEqual(rdna2spirv._depth_arg("0"), 0)
        self.assertEqual(rdna2spirv._depth_arg("64"), 64)
        for text in ("-1", "x", "", "1.5"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._depth_arg(text)

    def test_max_structured_depth_sets_the_bound(self):
        # SHADER-427: the option replaces MAX_STRUCTURED_DEPTH for the run; without it the default is 64, written out
        # here rather than read from the module so that changing the default fails this test.
        self.assertEqual(rdna2spirv.MAX_STRUCTURED_DEPTH, 64)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rdna2spirv, "MAX_STRUCTURED_DEPTH", rdna2spirv.MAX_STRUCTURED_DEPTH):
            for extra, expected in (([], 64), (["--max-structured-depth", "0"], 0),
                                    (["--max-structured-depth", "40"], 40)):
                with self.subTest(extra=extra), \
                        mock.patch.object(sys, "argv", [TOOL, tmp, "--out", os.path.join(tmp, "o")] + extra), \
                        redirect_stdout(io.StringIO()):
                    rdna2spirv.MAX_STRUCTURED_DEPTH = 64
                    rdna2spirv.main()
                    self.assertEqual(rdna2spirv.MAX_STRUCTURED_DEPTH, expected)


class SourceDiscovery(unittest.TestCase):
    """The src argument is a .s file or a folder of .s files; other names are not inputs."""

    def run_main(self, src, out):
        def spirv_tool(cmd, **kwargs):
            tool = cmd[0]
            if tool == "spirv-as":
                Path(kwargs.get("cwd", ""), cmd[3]).write_bytes(b"new module")
            elif tool == "spirv-val":
                self.assertEqual(Path(kwargs.get("cwd", ""), cmd[-1]).read_bytes(), b"new module")
            else:
                self.fail(f"unexpected tool: {tool}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        output = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(src), "--out", str(out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), \
                redirect_stdout(output):
            rdna2spirv.main()
        return output.getvalue()

    def test_missing_src_is_refused_with_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "spirv")
            # The .s suffix matters: without isfile() a missing shader path would be trusted as a file.
            p = subprocess.run([sys.executable, "-I", TOOL, os.path.join(tmp, "absent.s"), "--out", out],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("not a .s file or a folder of .s files", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_ending_in_s_as_src_is_a_directory(self):
        # A .s suffix alone is not a shader: an existing directory is still scanned as a folder.
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shader.s")
            os.mkdir(src)
            p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", os.path.join(tmp, "spirv")],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertIn("0 shaders:", p.stdout)

    def test_non_file_non_dir_entry_is_skipped(self):
        # Covers entries that are neither files nor directories (broken symlinks, FIFOs, devices).
        # is_file() rejects them; `not is_dir()` would feed one to translate() and crash.
        class OddEntry:
            name = "foo.s"
            path = os.path.join("not-a-file", "foo.s")

            def is_file(self):
                return False

            def is_dir(self):
                return False

        class FakeScandir:
            def __enter__(self):
                return iter([OddEntry()])

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            output = io.StringIO()
            with mock.patch.object(sys, "argv", [TOOL, src, "--out", os.path.join(tmp, "spirv")]), \
                    mock.patch.object(rdna2spirv.os, "scandir", lambda path: FakeScandir()), \
                    redirect_stdout(output):
                rdna2spirv.main()
            self.assertIn("0 shaders:", output.getvalue())

    def test_directory_ending_in_s_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "disasm"
            src.mkdir()
            (src / "ps_ok.s").write_text("\ts_endpgm\n", encoding="utf-8")
            (root / "ps_ok.dxbc").write_bytes(b"DXBC" + bytes(28))
            (src / "nested.s").mkdir()  # a directory that looks like a shader
            out = root / "spirv"
            out.mkdir()
            stdout = self.run_main(src, out)
            self.assertIn("1 pass spirv-val", stdout)
            self.assertTrue((out / "ps_ok.spv").is_file())
            self.assertFalse((out / "nested.spv").exists())

    def test_a_new_output_folder_with_missing_parents_still_works(self):
        # The ancestor walk in the --out guard must not reject folders that simply don't exist yet.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "disasm"
            src.mkdir()
            (src / "ps_ok.s").write_text("\ts_endpgm\n", encoding="utf-8")
            (root / "ps_ok.dxbc").write_bytes(b"DXBC" + bytes(28))
            out = root / "new" / "deep" / "spirv"
            stdout = self.run_main(src, out)
            self.assertIn("1 pass spirv-val", stdout)
            self.assertTrue((out / "ps_ok.spv").is_file())

    def test_non_s_file_as_src_is_refused_with_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "notes.txt")
            Path(src).write_text("not a shader", encoding="utf-8")
            p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", os.path.join(tmp, "spirv")],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)


class CommandLine(unittest.TestCase):
    def test_bad_limit_or_shard_is_refused_before_out_is_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            with open(os.path.join(src, "ps_x.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm\n")
            for args, message in ((["--limit", "0"], "must be at least 1"),
                                  (["--limit", "-5"], "must be at least 1"),
                                  (["--shard", "0/0"], "need n >= 1"),
                                  (["--shard", "9/8"], "need n >= 1 and 0 <= i < n"),
                                  (["--shard", "a/b"], "expected i/n")):
                out = os.path.join(tmp, "spirv")
                with self.subTest(args=args):
                    p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", out, *args],
                                       capture_output=True, text=True, errors="replace")
                    self.assertEqual(p.returncode, 2)
                    self.assertIn(message, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
                    self.assertFalse(os.path.exists(out))     # 9/8 used to write an empty index.part9.json


class OutputFolderValidation(unittest.TestCase):
    """--out must be a folder path: a file there is a usage error, never a traceback."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = self.tmp / "disasm"
        self.src.mkdir()
        (self.src / "ps_x.s").write_text("\ts_endpgm\n", encoding="utf-8")

    def run_tool(self, out, *extra, timeout=None):
        return subprocess.run([sys.executable, "-I", TOOL, str(self.src), "--out", os.fspath(out), *extra],
                              capture_output=True, text=True, errors="replace", timeout=timeout)

    def assert_usage_error(self, p, message):
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        # argparse prints its usage and one error line; no exception text besides that line.
        self.assertEqual(len([line for line in p.stderr.splitlines() if "error:" in line]), 1, p.stderr)
        self.assertIn(message, p.stderr)

    def test_a_file_at_out_is_refused_and_left_alone(self):
        out = self.tmp / "out.txt"
        out.write_text("keep", encoding="utf-8")
        p = self.run_tool(out)
        self.assert_usage_error(p, "is a file, not a folder")
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")

    def test_an_out_under_a_file_is_refused_and_the_file_is_left_alone(self):
        blocker = self.tmp / "blocker.txt"
        blocker.write_text("keep", encoding="utf-8")
        p = self.run_tool(blocker / "sub")
        self.assert_usage_error(p, "cannot create")
        self.assertEqual(blocker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((blocker / "sub").exists())

    def test_a_file_at_out_is_refused_with_merge(self):
        out = self.tmp / "out.txt"
        out.write_text("keep", encoding="utf-8")
        p = self.run_tool(out, "--merge")
        self.assert_usage_error(p, "is a file, not a folder")
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")

    def test_a_file_at_out_with_a_trailing_separator_is_refused(self):
        out = self.tmp / "out.txt"
        out.write_text("keep", encoding="utf-8")
        p = self.run_tool(str(out) + os.sep)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn(str(out), p.stderr)
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")

    def test_an_unmapped_drive_root_is_refused_without_hanging(self):
        if os.name != "nt":
            self.skipTest("unmapped drive roots are a Windows path case")
        drive = next((f"{c}:\\" for c in "ZYXWVUTSRQPONMLKJIHGFEDCBA" if not os.path.exists(f"{c}:\\")), None)
        if drive is None:
            self.skipTest("every drive letter exists on this machine")
        out = os.path.join(drive, "shaders")
        p = self.run_tool(out, timeout=30)
        # A root that does not exist used to make the ancestor walk spin forever.
        self.assert_usage_error(p, "cannot create")

    def test_a_symlink_to_a_file_at_out_is_refused(self):
        target = self.tmp / "target.txt"
        target.write_text("keep", encoding="utf-8")
        link = self.tmp / "link-out"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks are not available here: {e}")
        p = self.run_tool(link)
        self.assert_usage_error(p, "is a file, not a folder")
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")


class OutputInsideInput(unittest.TestCase):
    """--out equal to the input folder, or inside it, is refused before anything is written (BUILD-1044)."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = self.tmp / "disasm"
        self.src.mkdir()
        (self.src / "ps_x.s").write_text("\ts_endpgm\n", encoding="utf-8")
        (self.tmp / "ps_x.dxbc").write_bytes(b"DXBC" + bytes(28))

    def run_tool(self, out):
        return subprocess.run([sys.executable, "-I", TOOL, str(self.src), "--out", os.fspath(out)],
                              capture_output=True, text=True, timeout=120)

    def assert_refused(self, out):
        p = self.run_tool(out)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(len(p.stderr.strip().splitlines()), 1, p.stderr)
        self.assertTrue(p.stderr.startswith("ERROR: --out "), p.stderr)
        self.assertIn("separate output folder", p.stderr)
        self.assertFalse((self.src / "ps_x.spvasm").exists())
        self.assertFalse((self.src / "ps_x.spv").exists())
        self.assertFalse((self.src / "index.json").exists())

    def test_out_equal_to_the_input_folder_is_refused(self):
        self.assert_refused(self.src)

    def test_out_inside_the_input_folder_is_refused(self):
        out = self.src / "spirv"
        self.assert_refused(out)
        self.assertFalse(out.exists())      # refused before mkdir, not left as an empty folder

    def test_a_distinct_out_folder_still_translates(self):
        out = self.tmp / "spirv"

        def spirv_tool(cmd, **kwargs):
            if cmd[0] == "spirv-as":
                Path(kwargs.get("cwd", ""), cmd[3]).write_bytes(b"new module")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool):
            self.assertIsNone(rdna2spirv.main())
        self.assertTrue((out / "ps_x.spv").is_file())
        self.assertTrue((out / "index.json").is_file())


class ValidatedOutput(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = root / "disasm" / "ps_fixture.s"
        self.src.parent.mkdir()
        self.src.write_text("\ts_endpgm\n", encoding="utf-8")
        (root / "ps_fixture.dxbc").write_bytes(b"DXBC" + bytes(28))
        self.out = root / "spirv"
        self.out.mkdir()
        self.module = self.out / "ps_fixture.spv"
        self.index = self.out / "index.json"

    def translate(self, *, reject_at=None):
        def spirv_tool(cmd, **kwargs):
            tool = cmd[0]
            if tool == "spirv-as":
                Path(kwargs.get("cwd", ""), cmd[3]).write_bytes(b"new module")
            elif tool == "spirv-val":
                self.assertEqual(Path(kwargs.get("cwd", ""), cmd[-1]).read_bytes(), b"new module")
                if reject_at is None:
                    self.assertFalse(self.module.exists(), "the binary was published before validation")
            else:
                self.fail(f"unexpected tool: {tool}")
            rejected = tool == reject_at
            return subprocess.CompletedProcess(cmd, int(rejected), "", "synthetic rejection" if rejected else "")

        output = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), redirect_stdout(output):
            self.exit_code = rdna2spirv.main()
        return output.getvalue()

    def test_existing_single_s_file_still_translates(self):
        self.assertIn("1 pass spirv-val", self.translate())

    def test_missing_required_inputs_retire_stale_outputs_and_continue(self):
        listing_dir = self.src.parent
        root = listing_dir.parent
        missing = ("cs_dispatch", "ms_reanchor")
        for name in missing:
            (root / (name + ".dxbc")).write_bytes((root / "ps_fixture.dxbc").read_bytes())
            (listing_dir / (name + ".s")).write_bytes(self.src.read_bytes())
            (self.out / (name + ".spv")).write_bytes(b"previous module")
        (listing_dir / "ms_reanchor.s").write_bytes(
            b"\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n; invalid instruction encoding\n")
        self.index.write_bytes(json.dumps(dict.fromkeys(missing, {})).encode())
        self.src = listing_dir
        err = io.StringIO()
        with redirect_stderr(err):
            output = self.translate()
        self.assertEqual(self.exit_code, 1)
        self.assertIn("3 shaders: 1 translated, 1 pass spirv-val", output)
        self.assertIn("2 of 3 selected shaders missing required inputs", err.getvalue())
        for name in missing:
            self.assertIn(name, err.getvalue())
            self.assertFalse((self.out / (name + ".spv")).exists())
        self.assertTrue(self.module.exists())
        self.assertEqual(set(json.loads(self.index.read_bytes())), {"ps_fixture"})

    def test_missing_dispatch_invalidates_completed_shard(self):
        src = self.src.with_name("cs_dispatch.s")
        src.write_bytes(self.src.read_bytes())
        container = self.src.parent.parent / "ps_fixture.dxbc"
        shutil.copyfile(container, container.with_name("cs_dispatch.dxbc"))
        part = self.out / "index.part0.json"
        stale = self.out / "cs_dispatch.spv"
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                selection = src
                if interrupted:
                    # The compute input retires its module, then the next listing hits a read error.
                    container.unlink()
                    container.mkdir()
                    selection = src.parent
                stale.write_bytes(b"old compute")
                self.module.write_bytes(b"previous pixel")
                part.write_bytes(b'{"cs_dispatch": {}, "ps_fixture": {}}')
                self.index.write_bytes(part.read_bytes())
                result = subprocess.run([sys.executable, "-I", TOOL, str(selection), "--out", str(self.out),
                                         "--shard", "0/1"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("missing translation input", result.stderr)
                if interrupted:
                    self.assertIn("ps_fixture.dxbc", result.stderr)
                self.assertFalse(part.exists())
                self.assertFalse(self.index.exists())
                self.assertFalse(stale.exists())
                self.assertEqual(self.module.read_bytes(), b"previous pixel")

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "requires SPIRV-Tools")
    def test_failed_shard_retry_cannot_be_merged(self):
        def run(out, *args, hide_tools=False):
            env = os.environ.copy()
            if hide_tools:
                env["PATH"] = str(self.out)
            return subprocess.run([sys.executable, "-I", TOOL, str(self.src.parent),
                                   "--out", str(out), *args], env=env, capture_output=True, text=True)

        def success(out, *args):
            result = run(out, *args)
            self.assertEqual(result.returncode, 0, result.stderr)

        def snapshot(out):
            return {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

        self.src.with_name("ps_other.s").write_bytes(self.src.read_bytes())
        shutil.copyfile(self.src.parent.parent / "ps_fixture.dxbc", self.src.parent.parent / "ps_other.dxbc")
        for shard in ("0/2", "1/2"):
            success(self.out, "--shard", shard)
        success(self.out, "--merge")
        baseline = snapshot(self.out)
        self.assertEqual(set(json.loads(self.index.read_bytes())), {"ps_fixture", "ps_other"})
        for assembler in ("spirv-as", "python"):
            with self.subTest(assembler=assembler):
                value = 1 if assembler == "spirv-as" else 2
                self.src.write_bytes(f"\tv_mov_b32_e32 v0, {value}\n\ts_endpgm\n".encode())
                result = run(self.out, "--shard", "0/2", "--assembler", assembler, hide_tools=True)
                self.assertFalse(self.index.exists())
                merge = run(self.out, "--merge")
                self.assertEqual(merge.returncode, 2, merge.stdout)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("SPIRV-Tools", result.stderr)
                self.assertFalse((self.out / "index.part0.json").exists())
                for name in ("ps_fixture.spv", "ps_other.spv", "shard1.json", "index.part1.json"):
                    self.assertEqual((self.out / name).read_bytes(), baseline[name], name)
                success(self.out, "--shard", "0/2")
                success(self.out, "--merge")
                self.assertNotEqual(self.module.read_bytes(), baseline[self.module.name])
                baseline = snapshot(self.out)
        fresh = self.out.with_name("fresh")
        for shard in ("0/2", "1/2"):
            success(fresh, "--shard", shard)
        success(fresh, "--merge")
        self.assertEqual(snapshot(self.out), snapshot(fresh))

    def test_missing_assembler_is_one_setup_error(self):
        err = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run",
                                  side_effect=FileNotFoundError(2, "No such file or directory", "spirv-as")), \
                redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = rdna2spirv.main()
        message = err.getvalue()
        self.assertEqual(code, 2, message)
        self.assertIn("spirv-as", message)
        self.assertIn("SPIRV-Tools", message)
        self.assertNotIn("Traceback", message)
        self.assertEqual(len(message.strip().splitlines()), 1, message)
        self.assertFalse(self.module.exists())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_missing_validator_does_not_publish_the_module(self):
        def spirv_tool(cmd, **kwargs):
            if cmd[0] == "spirv-as":
                Path(kwargs.get("cwd", ""), cmd[3]).write_bytes(b"new module")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise FileNotFoundError(2, "No such file or directory", "spirv-val")

        err = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), \
                redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = rdna2spirv.main()
        message = err.getvalue()
        self.assertEqual(code, 2, message)
        self.assertIn("spirv-val", message)
        self.assertIn("SPIRV-Tools", message)
        self.assertNotIn("Traceback", message)
        self.assertEqual(len(message.strip().splitlines()), 1, message)
        self.assertFalse(self.module.exists())
        self.assertFalse(self.index.exists())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_validator_rejection_does_not_publish_binary(self):
        self.assertIn("INVALID: synthetic rejection", self.translate(reject_at="spirv-val"))
        self.assertFalse(self.module.exists())
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), {})
        self.assertTrue((self.out / "ps_fixture.spvasm").is_file())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_rejection_clears_retried_shader_but_preserves_unrelated_output(self):
        other = self.out / "cs_other.spv"
        other.write_bytes(b"unrelated module")
        previous = {"cs_other": {"stage": "cs"}}
        for tool in ("spirv-as", "spirv-val"):
            with self.subTest(tool=tool):
                self.module.write_bytes(b"old module")
                self.index.write_text(json.dumps({**previous, "ps_fixture": {"stage": "ps"}}), encoding="utf-8")
                self.assertIn("INVALID: synthetic rejection", self.translate(reject_at=tool))
                self.assertFalse(self.module.exists())
                self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), previous)
                self.assertEqual(other.read_bytes(), b"unrelated module")

    def test_unsupported_clears_retried_shader_but_preserves_unrelated_output(self):
        # s_barrier is compute-only, so this PS is Unsupported before any SPIR-V tool runs. A module an
        # earlier run published for it must not survive: the index is make_game.py's proof of completion.
        self.src.write_text("\ts_barrier\n\ts_endpgm\n", encoding="utf-8")
        other = self.out / "cs_other.spv"
        other.write_bytes(b"unrelated module")
        previous = {"cs_other": {"stage": "cs"}}
        self.module.write_bytes(b"old module")
        self.index.write_text(json.dumps({**previous, "ps_fixture": {"stage": "ps"}}), encoding="utf-8")
        output = self.translate()
        self.assertIn("s_barrier outside compute", output)
        self.assertFalse(self.module.exists())
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), previous)
        self.assertEqual(other.read_bytes(), b"unrelated module")
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_translator_error_clears_retried_shader(self):
        self.module.write_bytes(b"old module")
        self.index.write_text(json.dumps({"ps_fixture": {"stage": "ps"}}), encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(rdna2spirv, "translate", side_effect=ValueError("synthetic translator error")), \
                mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                redirect_stdout(output):
            rdna2spirv.main()
        self.assertIn("translator error ValueError", output.getvalue())
        self.assertFalse(self.module.exists())
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), {})

    def test_validation_success_publishes_binary_and_metadata(self):
        self.assertIn("1 pass spirv-val", self.translate())
        self.assertEqual(self.module.read_bytes(), b"new module")
        self.assertIn("ps_fixture", json.loads(self.index.read_text(encoding="utf-8")))


class EmptyShader(unittest.TestCase):
    """A .s with no tab-indented instruction parses to an empty shader.

    It used to translate to a module that passes spirv-val, lands in index.json and does no work at all, which is
    what a hand-written or re-indented .s looks like: found while writing a fixture with spaces instead of tabs.
    """

    def test_no_instructions_is_reported_and_no_module_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            with open(os.path.join(src, "ps_empty.s"), "w", encoding="utf-8") as f:
                f.write("; ps_empty  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words\n    s_endpgm\n")
            with open(os.path.join(tmp, "ps_empty.dxbc"), "wb") as f:
                f.write(b"DXBC" + bytes(28))
            out = os.path.join(tmp, "spirv")
            p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", out], capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("0 translated", p.stdout, p.stdout)
            self.assertIn("no instructions (nothing tab-indented?)", p.stdout, p.stdout)
            self.assertFalse(os.path.exists(os.path.join(out, "ps_empty.spv")))


class PopsInvocationInterlock(unittest.TestCase):
    """SHADER-350: POPS pixel interlock shaders must emit OpBeginInvocationInterlockEXT and OpEndInvocationInterlockEXT."""

    def test_straight_line_pops_pixel_shader_emits_begin_and_end_interlock(self):
        gen = rdna2spirv.Gen("ps")
        gen.uses_pops = True
        gen.body = ["OpNoOp"]
        spv = rdna2spirv.assemble(gen)
        self.assertEqual(spv.count("OpBeginInvocationInterlockEXT"), 1)
        self.assertEqual(spv.count("OpEndInvocationInterlockEXT"), 1)
        self.assertLess(spv.index("OpBeginInvocationInterlockEXT"), spv.index("OpEndInvocationInterlockEXT"))

    def test_branching_pops_pixel_shader_emits_begin_and_end_interlock(self):
        gen = rdna2spirv.Gen("ps")
        gen.uses_pops = True
        gen.blocks = [["OpNoOp"]]
        gen.succ = [[]]
        spv = rdna2spirv.assemble(gen)
        self.assertEqual(spv.count("OpBeginInvocationInterlockEXT"), 1)
        self.assertEqual(spv.count("OpEndInvocationInterlockEXT"), 1)
        self.assertLess(spv.index("OpBeginInvocationInterlockEXT"), spv.index("OpEndInvocationInterlockEXT"))

    def test_non_pops_pixel_shader_omits_interlock_ops(self):
        gen = rdna2spirv.Gen("ps")
        gen.uses_pops = False
        gen.body = ["OpNoOp"]
        spv = rdna2spirv.assemble(gen)
        self.assertEqual(spv.count("OpBeginInvocationInterlockEXT"), 0)
        self.assertEqual(spv.count("OpEndInvocationInterlockEXT"), 0)


if __name__ == "__main__":
    unittest.main()
