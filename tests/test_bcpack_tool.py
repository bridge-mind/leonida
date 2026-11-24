"""Unit tests for tools/bcpack_tool.py BCPack inspection, decode and profiling tool.

Usage:
    python -I tests/test_bcpack_tool.py
"""
from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TESTS = os.path.join(ROOT, "tests")
for p in (TOOLS, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import bcpack_tool
import test_bcpack


def _synthetic_bc1_blocks(n_blocks: int = 4) -> bytes:
    """Create n_blocks of valid synthetic BC1 blocks."""
    blocks = bytearray()
    for i in range(n_blocks):
        # c0 = 0xF800 (red), c1 = 0x07E0 (green), 4 bytes indices
        blocks.extend(struct.pack("<HHBBBB", 0xF800, 0x07E0, i & 0xFF, (i + 1) & 0xFF, 0, 0))
    return bytes(blocks)


class TestBcpackTool(unittest.TestCase):
    def setUp(self) -> None:
        self.original_blocks = _synthetic_bc1_blocks(4)
        self.compressed_stream = test_bcpack.encode_bc1_blocks(self.original_blocks)

    def test_inspect_stream_synthetic(self) -> None:
        info = bcpack_tool.inspect_stream(
            self.compressed_stream, codec=1, out_size=len(self.original_blocks)
        )
        self.assertEqual(info["codec"], "BC1")
        self.assertEqual(info["codec_id"], 1)
        self.assertEqual(info["blocks"], 4)
        self.assertEqual(info["uncompressed_bytes"], 32)
        self.assertGreater(info["compressed_bytes"], 0)
        self.assertEqual(len(info["lanes"]), 12)
        for lane in info["lanes"]:
            self.assertEqual(lane["symbols"], 4)
            self.assertGreater(lane["bit_width"], 0)

    def test_profile_stream_synthetic(self) -> None:
        prof = bcpack_tool.profile_stream(
            self.compressed_stream, codec=1, out_size=len(self.original_blocks)
        )
        self.assertIn("histograms", prof)
        self.assertEqual(len(prof["histograms"]), 12)
        for h in prof["histograms"]:
            self.assertGreaterEqual(h["unique_symbols"], 1)
            self.assertGreater(len(h["top_symbols"]), 0)

    def test_decode_file_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "test.bcpack")
            out_file = os.path.join(tmp_dir, "test.raw")

            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)

            written = bcpack_tool.decode_file(
                in_file, out_file, codec=1, out_size=len(self.original_blocks)
            )
            self.assertEqual(written, len(self.original_blocks))

            with open(out_file, "rb") as f:
                decoded = f.read()

            self.assertEqual(decoded, self.original_blocks)

    def test_decode_refuses_same_input_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "same.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            with self.assertRaises(ValueError) as ctx:
                bcpack_tool.decode_file(
                    in_file, in_file, codec=1, out_size=len(self.original_blocks)
                )
            self.assertIn(in_file, str(ctx.exception))
            with open(in_file, "rb") as f:
                self.assertEqual(f.read(), self.compressed_stream)

    def test_decode_refuses_symlink_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "same.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            link_file = os.path.join(tmp_dir, "link.bcpack")
            try:
                os.symlink(in_file, link_file)
            except (OSError, NotImplementedError, AttributeError) as e:
                self.skipTest(f"symlinks unavailable: {e}")
            with self.assertRaises(ValueError) as ctx:
                bcpack_tool.decode_file(
                    in_file, link_file, codec=1, out_size=len(self.original_blocks)
                )
            self.assertIn(link_file, str(ctx.exception))
            with open(in_file, "rb") as f:
                self.assertEqual(f.read(), self.compressed_stream)

    def test_decode_refuses_hardlink_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "same.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            link_file = os.path.join(tmp_dir, "link.bcpack")
            try:
                os.link(in_file, link_file)
            except (OSError, NotImplementedError, AttributeError) as e:
                self.skipTest(f"hard links unavailable: {e}")
            with self.assertRaises(ValueError) as ctx:
                bcpack_tool.decode_file(
                    in_file, link_file, codec=1, out_size=len(self.original_blocks)
                )
            self.assertIn(link_file, str(ctx.exception))
            with open(in_file, "rb") as f:
                self.assertEqual(f.read(), self.compressed_stream)

    def test_cli_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "sample.bcpack")
            out_file = os.path.join(tmp_dir, "sample.raw")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)

            # Test inspect CLI
            ret = bcpack_tool.main([
                "inspect", in_file,
                "--codec", "1",
                "--size", str(len(self.original_blocks)),
                "--json",
            ])
            self.assertEqual(ret, 0)

            # Test profile CLI
            ret = bcpack_tool.main([
                "profile", in_file,
                "--codec", "1",
                "--size", str(len(self.original_blocks)),
            ])
            self.assertEqual(ret, 0)

            # Test decode CLI
            ret = bcpack_tool.main([
                "decode", in_file, out_file,
                "--codec", "1",
                "--size", str(len(self.original_blocks)),
            ])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.isfile(out_file))

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            bcpack_tool.inspect_stream(self.compressed_stream, codec=99, out_size=32)

        with self.assertRaises(ValueError):
            bcpack_tool.inspect_stream(self.compressed_stream, codec=1, out_size=33)


class TestCliErrorPaths(unittest.TestCase):
    """The CLI must report a failed input/output in one ERROR line with exit status 2,
    never a raw traceback (bcpack.py library behavior is unchanged)."""

    def setUp(self) -> None:
        self.original_blocks = _synthetic_bc1_blocks(4)
        self.compressed_stream = test_bcpack.encode_bc1_blocks(self.original_blocks)

    def _cli(self, argv, tmp_dir, name):
        from contextlib import redirect_stderr, redirect_stdout
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = bcpack_tool.main(argv)
        with open(os.path.join(tmp_dir, name + ".stderr"), "w", encoding="utf-8", newline="\n") as f:
            f.write(err.getvalue())
        return rc, err.getvalue(), out.getvalue()

    def _assert_one_error_line(self, rc, err, src):
        self.assertEqual(rc, 2)
        lines = err.splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("ERROR: "), err)
        self.assertIn(src, lines[0])
        self.assertNotIn("Traceback", err)

    def test_missing_input_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = os.path.join(tmp_dir, "absent.bcpack")
            rc, err, _ = self._cli(
                ["inspect", missing, "--codec", "1", "--size", "32"], tmp_dir, "inspect_missing"
            )
            self._assert_one_error_line(rc, err, "absent.bcpack")

    def test_missing_input_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = os.path.join(tmp_dir, "absent.bcpack")
            rc, err, _ = self._cli(
                ["profile", missing, "--codec", "1", "--size", "32"], tmp_dir, "profile_missing"
            )
            self._assert_one_error_line(rc, err, "absent.bcpack")

    def test_missing_input_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = os.path.join(tmp_dir, "absent.bcpack")
            out_file = os.path.join(tmp_dir, "out.raw")
            rc, err, _ = self._cli(
                ["decode", missing, out_file, "--codec", "1", "--size", "32"],
                tmp_dir, "decode_missing",
            )
            self._assert_one_error_line(rc, err, "absent.bcpack")
            self.assertFalse(os.path.exists(out_file))

    def test_empty_stream_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "empty.bcpack")
            with open(in_file, "wb") as f:
                f.write(b"")
            rc, err, _ = self._cli(
                ["inspect", in_file, "--codec", "1", "--size", "32"], tmp_dir, "inspect_empty"
            )
            self._assert_one_error_line(rc, err, "empty.bcpack")

    def test_malformed_stream_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "junk.bcpack")
            with open(in_file, "wb") as f:
                f.write(bytes(range(256)))
            rc, err, _ = self._cli(
                ["inspect", in_file, "--codec", "1", "--size", "4096"], tmp_dir, "inspect_junk"
            )
            self._assert_one_error_line(rc, err, "junk.bcpack")

    def test_malformed_stream_decode_writes_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "junk.bcpack")
            out_file = os.path.join(tmp_dir, "out.raw")
            with open(in_file, "wb") as f:
                f.write(b"\x00" * 64)
            rc, err, _ = self._cli(
                ["decode", in_file, out_file, "--codec", "1", "--size", "32"],
                tmp_dir, "decode_junk",
            )
            self._assert_one_error_line(rc, err, "junk.bcpack")
            self.assertFalse(os.path.exists(out_file))

    def test_decode_output_is_a_directory(self) -> None:
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "sample.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            out_dir = os.path.join(tmp_dir, "out_dir")
            os.makedirs(out_dir)
            with mock.patch.object(bcpack_tool.bcpack, "decode_with_info",
                                   return_value=(self.original_blocks, frozenset())):
                rc, err, _ = self._cli(
                    ["decode", in_file, out_dir, "--codec", "1", "--size", "32"],
                    tmp_dir, "decode_outdir",
                )
            self._assert_one_error_line(rc, err, "out_dir")

    def test_malformed_history_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "sample.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            history = os.path.join(tmp_dir, "hist.json")
            with open(history, "w", encoding="utf-8", newline="\n") as f:
                f.write("{not json")
            out_file = os.path.join(tmp_dir, "out.raw")
            rc, err, _ = self._cli(
                ["decode", in_file, out_file, "--codec", "1", "--size", "32",
                 "--history", history],
                tmp_dir, "decode_hist",
            )
            self._assert_one_error_line(rc, err, "hist.json")
            self.assertFalse(os.path.exists(out_file))

    def test_decode_same_input_output_cli_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "same.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            rc, err, _ = self._cli(
                ["decode", in_file, in_file, "--codec", "1", "--size", "32"],
                tmp_dir, "decode_same_io",
            )
            self._assert_one_error_line(rc, err, "same.bcpack")
            with open(in_file, "rb") as f:
                self.assertEqual(f.read(), self.compressed_stream)

    def test_success_path_still_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "sample.bcpack")
            with open(in_file, "wb") as f:
                f.write(self.compressed_stream)
            out_file = os.path.join(tmp_dir, "out.raw")
            rc, err, _ = self._cli(
                ["decode", in_file, out_file, "--codec", "1", "--size", "32"],
                tmp_dir, "decode_ok",
            )
            self.assertEqual(rc, 0, err)
            self.assertEqual(err, "")
            with open(out_file, "rb") as f:
                self.assertEqual(f.read(), self.original_blocks)


if __name__ == "__main__":
    unittest.main()
