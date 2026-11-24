"""amd_swizzle.load: parses addrlib's pattern tables without evaluating them as Python.

    python -m pytest tests/test_amd_swizzle.py
"""
import io
import os
import pathlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import pytest  # noqa: E402 (keeps the CI bash loop from running this file)
except ImportError:
    pytest = None

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    import amd_swizzle as A  # noqa: E402
    HAVE_NUMPY = True
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    A = None
    HAVE_NUMPY = False

# tools/amd_swizzle.py needs numpy (tests/requirements.txt). Without it the module-level tests below are skipped,
# as the __main__ block skips them; the CLI guard tests fake numpy and still run.
if pytest is not None:
    needs_numpy = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy is not installed")
else:
    def needs_numpy(fn):
        return fn


def table(name, rows):
    body = "\n".join("    {" + ", ".join(r) + ", }, // " + str(i) for i, r in enumerate(rows))
    return f"const UINT_64 GFX10_SW_PATTERN_{name}[][{len(rows[0])}] =\n{{\n{body}\n}};\n"


def header(nibble01_row):
    return (table("NIBBLE01", [nibble01_row]) + table("NIBBLE2", [["0"] * 4]) +
            table("NIBBLE3", [["0"] * 4]) + table("NIBBLE4", [["0"] * 4]) +
            "const ADDR_SW_PATINFO GFX10_SW_64K_D_X_PATINFO[] =\n{\n"
            "    {   0,    0,    0,    0,    0, } , // 1 pipes 1 bpe\n};\n")


def load(text):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gfx10SwizzlePattern.h")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return A.load(path)


@needs_numpy
def test_symbols_and_xor():
    nibbles, tables = load(header(["X0", "0", "Y1", "X3 ^ Y3", "Z0", "S2 ^ Y4 ^ X5", "7", "Y0"]))
    assert nibbles["NIBBLE01"] == [[1, 0, 1 << 17, (1 << 3) ^ (1 << 19), 1 << 32,
                                    (1 << 50) ^ (1 << 20) ^ (1 << 5), 7, 1 << 16]]
    assert nibbles["NIBBLE2"] == [[0, 0, 0, 0]]
    assert tables["GFX10_SW_64K_D_X_PATINFO"] == [(0, 0, 0, 0, 0, "1 pipes 1 bpe")]


@needs_numpy
def test_rejects_code():
    for cell in ("__import__('os').getcwd()", "X0 | Y0", "X16", "-1"):
        try:
            load(header([cell] + ["0"] * 7))
        except ValueError:
            continue
        raise AssertionError(f"accepted {cell!r}")


def _load_amd_swizzle_for_cli():
    if A is not None:
        return A
    fake_np = types.ModuleType("numpy")
    fake_np.uint8 = int
    fake_np.float32 = float
    fake_np.int64 = int
    fake_np.ndarray = object
    with mock.patch.dict(sys.modules, {"numpy": fake_np}):
        import amd_swizzle
        return amd_swizzle


class AmdSwizzleCliGuardTests(unittest.TestCase):
    def setUp(self):
        if not HAVE_NUMPY:
            fake_np = types.ModuleType("numpy")
            fake_np.uint8 = int
            fake_np.float32 = float
            fake_np.int64 = int
            fake_np.ndarray = object
            p = mock.patch.dict(sys.modules, {"numpy": fake_np})
            p.start()
            self.addCleanup(p.stop)
        self.mod = _load_amd_swizzle_for_cli()

    def test_missing_input_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist.bin")
            stderr = io.StringIO()
            with mock.patch.object(sys, "stderr", stderr):
                rc = self.mod.main([missing, "--width", "64", "--height", "64"])
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_bad_out_paths_rejected_before_reading_or_searching(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = os.path.join(tmp, "blob.bin")
            with open(blob, "wb") as f:
                f.write(b"\x01\x02\x03\x04" * 64)
            existing_dir = os.path.join(tmp, "out_dir")
            os.mkdir(existing_dir)
            file_ancestor = os.path.join(tmp, "file_ancestor")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")

            bad_outs = [
                "",
                "   ",
                "bad\x00out.png",
                existing_dir,
                os.path.join(tmp, "trailing") + os.sep,
                os.path.join(file_ancestor, "sub", "out.png"),
                blob,
                os.path.join(tmp, ".", "blob.bin"),
            ]
            for bad in bad_outs:
                with self.subTest(out=bad):
                    stderr = io.StringIO()
                    with mock.patch.object(self.mod, "load") as load_mock, \
                         mock.patch.object(sys, "stderr", stderr):
                        rc = self.mod.main([blob, "--width", "64", "--height", "64", "--out", bad])
                    self.assertEqual(rc, 2)
                    self.assertIn("ERROR:", stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
                    load_mock.assert_not_called()

    def _run_with_save_side_effect(self, tmp, out_name, side_effect):
        blob = os.path.join(tmp, "blob.bin")
        with open(blob, "wb") as f:
            f.write(b"\x00" * 64)
        out_path = os.path.join(tmp, "nested", out_name)

        fake_lin = mock.MagicMock()
        fake_lin.size = 64
        fake_lin.tobytes.return_value = b"\x00" * 64
        fake_img = mock.MagicMock()
        fake_img.reshape.return_value = fake_img

        fake_pil = types.ModuleType("PIL")
        fake_pil_image = types.ModuleType("PIL.Image")
        fake_opened = mock.MagicMock()
        fake_opened.convert.return_value.save.side_effect = side_effect
        fake_pil_image.open = mock.Mock(return_value=fake_opened)
        fake_pil.Image = fake_pil_image

        fake_tables = {"GFX10_SW_64K_S_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")]}
        with mock.patch.dict(sys.modules, {"PIL": fake_pil, "PIL.Image": fake_pil_image}), \
             mock.patch.object(self.mod, "load", return_value=({}, fake_tables)), \
             mock.patch.object(self.mod, "equation", return_value=[]), \
             mock.patch.object(self.mod, "detile", return_value=fake_lin), \
             mock.patch.object(self.mod, "feature", return_value=fake_img), \
             mock.patch("tile_probe.score", return_value=1.0):
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
                rc = self.mod.main([blob, "--width", "8", "--height", "8", "--out", out_path])
        return rc, stderr.getvalue()

    def test_write_oserror_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self._run_with_save_side_effect(tmp, "best.png", PermissionError("denied"))
        self.assertEqual(rc, 2)
        self.assertIn("ERROR:", err)
        self.assertNotIn("Traceback", err)

    def test_write_value_error_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, err = self._run_with_save_side_effect(
                tmp, "best.unknown", ValueError("unknown file extension: .unknown")
            )
        self.assertEqual(rc, 2)
        self.assertIn("ERROR:", err)
        self.assertIn("unknown file extension: .unknown", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    if HAVE_NUMPY:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    unittest.main()


@needs_numpy
def test_64k_block_extent_comes_from_the_element_count():
    # XOR modes have no x/y bit that appears alone, so the old solo-bit guess returned 1x1 or too small.
    for bpe, bw, bh in ((1, 256, 256), (2, 256, 128), (4, 128, 128), (8, 128, 64), (16, 64, 64)):
        _, meta = A.offsets([(1, 0), (0, 1)], bpe, 1, 1)
        assert meta[:2] == (bw, bh), (bpe, meta)
        assert bw * bh * bpe == 65536


def cli(tmp_path, *args):
    import subprocess
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "amd_swizzle.py")
    return subprocess.run([sys.executable, "-I", tool, str(tmp_path / "blob.bin"), *args],
                          capture_output=True, text=True, timeout=120, check=False)


@needs_numpy
def test_out_cannot_overwrite_the_input(tmp_path):
    # Before: --out naming the blob rendered the preview over the sample it had just detiled.
    blob = tmp_path / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 64)
    before = blob.read_bytes()
    p = cli(tmp_path, "--width", "256", "--height", "256", "--out", str(blob))
    assert p.returncode == 2, p.stderr
    assert "overwrite" in p.stderr
    assert "Traceback" not in p.stderr
    assert blob.read_bytes() == before


@needs_numpy
def test_out_directory_is_a_message_not_a_traceback(tmp_path):
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 64)
    p = cli(tmp_path, "--width", "256", "--height", "256", "--out", str(tmp_path))
    assert p.returncode == 2, p.stderr
    assert "is a directory" in p.stderr
    assert "Traceback" not in p.stderr


@needs_numpy
def test_uncompressed_feature_uses_raw_rgb_bytes():
    import numpy as np
    raw = np.arange(8, dtype=np.uint8).reshape(-1, 4)
    got = A.feature(raw, 4)
    assert got.shape == (2, 3)
    assert got.tolist() == [[0, 1, 2], [4, 5, 6]]


@pytest.mark.parametrize(("bpe", "texels", "pixels"), [
    (1, bytes((17, 63, 129, 245)), [(17, 17, 17), (63, 63, 63), (129, 129, 129), (245, 245, 245)]),
    (2, bytes((17, 33, 63, 94, 129, 159, 245, 200)), [(17, 33, 0), (63, 94, 0), (129, 159, 0), (245, 200, 0)]),
], ids=["r8", "rg8"])
@needs_numpy
def test_cli_previews_uncompressed_r8_and_rg8(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, bpe: int, texels: bytes, pixels: list[tuple[int, int, int]],
) -> None:
    image = pytest.importorskip("PIL.Image")
    # A synthetic 4 KiB row-major tile, 64 elements per row in both formats.
    bits = ["0"] * (bpe - 1) + [f"X{i}" for i in range(6)] + [f"Y{i}" for i in range(7 - bpe)]
    rows = {"NIBBLE01": bits[:8], "NIBBLE2": bits[8:], "NIBBLE3": ["0"] * 4, "NIBBLE4": ["0"] * 4}
    header_path = tmp_path / "swizzle.h"
    header_path.write_text(
        "".join(table(name, [row]) for name, row in rows.items())
        + "const ADDR_SW_PATINFO GFX10_SW_4K_S_PATINFO[] =\n{\n"
        + f"    {{   0,    0,    0,    0,    0, }} , // synthetic {bpe} bpe\n}};\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("AMD_SWIZZLE_HEADER", str(header_path))
    tile = bytearray(4096)
    tile[:2 * bpe] = texels[:2 * bpe]
    tile[64 * bpe:66 * bpe] = texels[2 * bpe:]
    (tmp_path / "blob.bin").write_bytes(tile)
    output = tmp_path / "preview.png"

    result = cli(tmp_path, "--bpe", str(bpe), "--width", "2", "--height", "2", "--modes", "4K", "--out", str(output))

    assert result.returncode == 0, result.stderr
    with image.open(output) as png:
        assert png.size == (2, 2)
        assert png.tobytes() == bytes(channel for pixel in pixels for channel in pixel)


@needs_numpy
def test_cli_detiles_4k_tables_with_4k_blocks():
    # A 128x128 RGBA gradient tiled in 4 KB blocks (32x32 elements, row-major). The same pattern entry under
    # a 64K table name must not be picked by --modes 4K, and the 4K table must detile with 12-bit, 4 KB blocks.
    import subprocess
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from tile_probe import score
    rows = {"NIBBLE01": ["0", "0", "X0", "Y0", "X1", "Y1", "X2", "Y2"], "NIBBLE2": ["X3", "Y3", "X4", "Y4"],
            "NIBBLE3": ["X5", "Y5", "X6", "Y6"], "NIBBLE4": ["0", "0", "0", "0"]}
    text = "".join(table(name, [row]) for name, row in rows.items())
    for name in ("GFX10_SW_4K_S_PATINFO", "GFX10_SW_64K_S_PATINFO"):
        text += (f"const ADDR_SW_PATINFO {name}[] =\n{{\n"
                 "    {   0,    0,    0,    0,    0, } , // 1 pipes 4 bpe\n};\n")
    y, x = np.mgrid[0:128, 0:128]
    linear = np.stack([x * 2, y * 2, (x + y) // 2, np.full_like(x, 255)], axis=-1).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "gfx10SwizzlePattern.h")
        with open(hdr, "w", encoding="utf-8") as f:
            f.write(text)
        nibbles, tables = A.load(hdr)
        eq = A.equation(nibbles, tables["GFX10_SW_4K_S_PATINFO"][0], 12)
        off, meta = A.offsets(eq, 4, 128, 128, 4096)
        assert meta[:2] == (32, 32)
        tiled = np.zeros(128 * 128 * 4, np.uint8)
        for c in range(4):
            tiled[off.reshape(-1) + c] = linear[..., c].reshape(-1)
        blob = os.path.join(d, "blob.bin")
        tiled.tofile(blob)
        p = subprocess.run([sys.executable, "-I", A.__file__, blob, "--bpe", "4", "--width", "128",
                            "--height", "128", "--modes", "4K"], capture_output=True, text=True,
                           env=dict(os.environ, AMD_SWIZZLE_HEADER=hdr))
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert [ln.split()[1] for ln in lines] == ["GFX10_SW_4K_S_PATINFO"], p.stdout
    assert float(lines[0].split()[0]) == round(score(linear[..., :3].astype(np.float32)), 2), p.stdout


@needs_numpy
def test_preview_keeps_partial_bc_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import struct

    image = pytest.importorskip("PIL.Image")
    preview = tmp_path / "preview.png"
    tile = bytearray(4096)
    colors = (
        (0xF800, 0xF800, 0x07E0),
        (0xF800, 0xF800, 0x07E0),
        (0x001F, 0x001F, 0xFFFF),
    )
    for y, row in enumerate(colors):
        for x, color in enumerate(row):
            struct.pack_into("<HHI", tile, 8 * (32 * y + x), color, color, 0)
    (tmp_path / "blob.bin").write_bytes(tile)

    # A 4 KiB tile with 32 blocks per row and 16 rows.
    rows = {
        "NIBBLE01": ["0", "0", "0", "X0", "X1", "X2", "X3", "X4"],
        "NIBBLE2": ["Y0", "Y1", "Y2", "Y3"],
        "NIBBLE3": ["0"] * 4,
        "NIBBLE4": ["0"] * 4,
    }
    header_path = tmp_path / "swizzle.h"
    header_path.write_text(
        "".join(table(name, [row]) for name, row in rows.items())
        + "const ADDR_SW_PATINFO GFX10_SW_4K_S_PATINFO[] =\n{\n"
        "    {   0,    0,    0,    0,    0, } , // synthetic 8 bpe\n};\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AMD_SWIZZLE_HEADER", str(header_path))

    result = cli(tmp_path, "--width", "9", "--height", "9", "--modes", "4K", "--out", str(preview))

    assert result.returncode == 0, result.stderr
    with image.open(preview) as png:
        assert png.size == (9, 9)
        assert png.getpixel((7, 7)) == (255, 0, 0)
        assert png.getpixel((8, 7)) == (0, 255, 0)
        assert png.getpixel((7, 8)) == (0, 0, 255)
        assert png.getpixel((8, 8)) == (255, 255, 255)
