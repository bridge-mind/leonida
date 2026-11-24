"""Unit tests for tools/swizzle_search.py covering permutation matching and coordinate inversion.

    python -I tests/test_swizzle_search.py
"""
from __future__ import annotations

import importlib.util
import io
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    import numpy as np
    import swizzle_search
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

TOOL = TOOLS_DIR / "swizzle_search.py"


def invert_coordinates(
    img_x: int | np.ndarray,
    img_y: int | np.ndarray,
    seq: str | list[str],
    tw: int,
    th: int,
    wblocks: int,
) -> int | np.ndarray:
    """Invert global block coordinates (img_x, img_y) to raw stored block index."""
    tile_x = img_x // tw
    tile_y = img_y // th
    x = img_x % tw
    y = img_y % th
    tiles_x = wblocks // tw
    tile = tile_y * tiles_x + tile_x

    is_scalar = not isinstance(img_x, np.ndarray)
    inner = 0 if is_scalar else np.zeros_like(img_x, dtype=np.int64)
    xi = yi = 0
    for k, c in enumerate(seq):
        if c == "x":
            bit = (x >> xi) & 1
            xi += 1
        else:
            bit = (y >> yi) & 1
            yi += 1
        inner |= bit << k

    per = tw * th
    return tile * per + inner


def forward_coordinates(
    stored_idx: int | np.ndarray,
    seq: str | list[str],
    tw: int,
    th: int,
    wblocks: int,
) -> tuple[int | np.ndarray, int | np.ndarray]:
    """Forward mapping from stored block index to global block coordinates (img_x, img_y)."""
    per = tw * th
    tile = stored_idx // per
    inner = stored_idx % per
    is_scalar = not isinstance(stored_idx, np.ndarray)
    x = 0 if is_scalar else np.zeros_like(stored_idx, dtype=np.int64)
    y = 0 if is_scalar else np.zeros_like(stored_idx, dtype=np.int64)
    xi = yi = 0
    for k, c in enumerate(seq):
        bit = (inner >> k) & 1
        if c == "x":
            x |= bit << xi
            xi += 1
        else:
            y |= bit << yi
            yi += 1
    tiles_x = wblocks // tw
    img_x = (tile % tiles_x) * tw + x
    img_y = (tile // tiles_x) * th + y
    return img_x, img_y


def make_synthetic_gradient_bc1(wb: int, hb: int) -> np.ndarray:
    """Create a 2D grid of BC1 blocks with smooth horizontal (R) and vertical (G) gradient."""
    blocks = np.zeros((hb * wb, 8), dtype=np.uint8)
    for by in range(hb):
        for bx in range(wb):
            disp_idx = by * wb + bx
            r5 = (bx * 31) // max(1, wb - 1)
            g6 = (by * 63) // max(1, hb - 1)
            rgb565 = (r5 << 11) | (g6 << 5)
            blocks[disp_idx, 0] = rgb565 & 0xFF
            blocks[disp_idx, 1] = (rgb565 >> 8) & 0xFF
    return blocks


def make_synthetic_gradient_bc3(wb: int, hb: int) -> np.ndarray:
    """Create a 2D grid of BC3 blocks with smooth gradient in endpoints and alpha."""
    blocks = np.zeros((hb * wb, 16), dtype=np.uint8)
    for by in range(hb):
        for bx in range(wb):
            disp_idx = by * wb + bx
            r5 = (bx * 31) // max(1, wb - 1)
            g6 = (by * 63) // max(1, hb - 1)
            rgb565 = (r5 << 11) | (g6 << 5)
            blocks[disp_idx, 8] = rgb565 & 0xFF
            blocks[disp_idx, 9] = (rgb565 >> 8) & 0xFF
            blocks[disp_idx, 0] = (bx * 255) // max(1, wb - 1)
    return blocks


def swizzle_blocks(disp_blocks: np.ndarray, seq: str | list[str], tw: int, th: int, wb: int) -> bytes:
    """Swizzle row-major display blocks into stored order using the specified bit sequence."""
    nblocks = len(disp_blocks)
    pos = swizzle_search.positions(seq, nblocks, tw, th, wb)
    stored = np.zeros_like(disp_blocks)
    stored[np.arange(nblocks)] = disp_blocks[pos]
    return stored.tobytes()


@unittest.skipUnless(HAVE_NUMPY, "numpy required")
class TestSwizzleCoordinateInversionAndBitmask(unittest.TestCase):
    def test_positions_permutation_bijectivity(self):
        """positions must map raw blocks to a complete permutation of range(nblocks)."""
        patterns = [
            ("xyxy", 4, 4, 16, 2),        # Morton order, square
            ("yxyx", 4, 4, 16, 2),        # Transposed Morton
            ("xxyy", 4, 4, 16, 2),        # Clustered X then Y
            ("yyxx", 4, 4, 16, 2),        # Clustered Y then X
            ("xyxyxy", 8, 8, 32, 2),      # 64 blocks/tile, 6-bit
            ("xyxyx", 8, 4, 32, 2),       # Asymmetric: 32 blocks/tile (tw=8, th=4)
            ("yxyxy", 4, 8, 16, 4),       # Asymmetric: 32 blocks/tile (tw=4, th=8)
        ]
        for seq, tw, th, wb, rows in patterns:
            with self.subTest(seq=seq, tw=tw, th=th, wb=wb):
                tiles_x = wb // tw
                nblocks = tiles_x * rows * tw * th
                pos = swizzle_search.positions(seq, nblocks, tw, th, wb)
                self.assertEqual(len(pos), nblocks)
                self.assertEqual(pos.min(), 0)
                self.assertEqual(pos.max(), nblocks - 1)
                np.testing.assert_array_equal(np.sort(pos), np.arange(nblocks))

    def test_coordinate_inversion_roundtrip_scalar(self):
        """Scalar forward_coordinates and invert_coordinates must invert each other exactly."""
        seq = "xyxxyy"
        tw, th = 8, 8
        wb = 32
        hb = 16
        for by in range(hb):
            for bx in range(wb):
                stored = invert_coordinates(bx, by, seq, tw, th, wb)
                rx, ry = forward_coordinates(stored, seq, tw, th, wb)
                self.assertEqual((rx, ry), (bx, by), f"Mismatch at ({bx}, {by}) -> stored={stored}")

    def test_coordinate_inversion_roundtrip_vector(self):
        """Vectorized forward and invert coordinates must round-trip across all blocks."""
        test_cases = [
            ("xyxy", 4, 4, 16, 8),
            ("xxyy", 4, 4, 16, 8),
            ("xyxyxy", 8, 8, 32, 16),
            ("xyxyx", 8, 4, 32, 8),
            ("yxyxy", 4, 8, 16, 16),
        ]
        for seq, tw, th, wb, hb in test_cases:
            with self.subTest(seq=seq, tw=tw, th=th):
                nblocks = wb * hb
                stored_indices = np.arange(nblocks, dtype=np.int64)
                ix, iy = forward_coordinates(stored_indices, seq, tw, th, wb)
                recovered = invert_coordinates(ix, iy, seq, tw, th, wb)
                np.testing.assert_array_equal(recovered, stored_indices)

    def test_positions_matches_forward_coordinates(self):
        """Linear display positions from swizzle_search.positions must match forward coordinates."""
        seq = "xyxyxy"
        tw, th = 8, 8
        wb = 32
        nblocks = 32 * 16
        pos = swizzle_search.positions(seq, nblocks, tw, th, wb)
        stored_indices = np.arange(nblocks, dtype=np.int64)
        ix, iy = forward_coordinates(stored_indices, seq, tw, th, wb)
        expected_pos = iy * wb + ix
        np.testing.assert_array_equal(pos, expected_pos)

    def test_synthetic_bitmask_isolation(self):
        """Each bit position k in seq must route strictly to either x or y coordinate."""
        for seq in ("xyxyxy", "xxyyxx", "yyxxyy", "xyxyx", "yxyxy"):
            bits = len(seq)
            xbits = seq.count("x")
            ybits = seq.count("y")
            tw = 1 << xbits
            th = 1 << ybits
            wb = tw * 2
            with self.subTest(seq=seq):
                for k in range(bits):
                    inner_mask = 1 << k
                    ix, iy = forward_coordinates(inner_mask, seq, tw, th, wb)
                    expected_xi = seq[:k].count("x")
                    expected_yi = seq[:k].count("y")
                    if seq[k] == "x":
                        self.assertEqual(ix, 1 << expected_xi, f"Bit {k} must shift into x at bit {expected_xi}")
                        self.assertEqual(iy, 0, f"Bit {k} in seq marked 'x' must leave y as 0")
                    else:
                        self.assertEqual(iy, 1 << expected_yi, f"Bit {k} must shift into y at bit {expected_yi}")
                        self.assertEqual(ix, 0, f"Bit {k} in seq marked 'y' must leave x as 0")

    def test_synthetic_bitmask_composite_patterns(self):
        """Composite bitmask patterns must separate correctly across X and Y coordinate bits."""
        seq = "xyxyxy"
        tw, th, wb = 8, 8, 16
        ix, iy = forward_coordinates(63, seq, tw, th, wb)
        self.assertEqual((ix, iy), (7, 7))

        ix, iy = forward_coordinates(21, seq, tw, th, wb)
        self.assertEqual((ix, iy), (7, 0))

        ix, iy = forward_coordinates(42, seq, tw, th, wb)
        self.assertEqual((ix, iy), (0, 7))

    def test_multi_tile_striding(self):
        """Cross-tile indexing must preserve row-major tile ordering across columns and rows."""
        seq = "xyxy"
        tw, th = 4, 4
        wb = 16
        per = tw * th

        self.assertEqual(forward_coordinates(0 * per, seq, tw, th, wb), (0, 0))
        self.assertEqual(forward_coordinates(1 * per, seq, tw, th, wb), (4, 0))
        self.assertEqual(forward_coordinates(4 * per, seq, tw, th, wb), (0, 4))
        self.assertEqual(forward_coordinates(5 * per, seq, tw, th, wb), (4, 4))


@unittest.skipUnless(HAVE_NUMPY, "numpy required")
class TestSwizzleSearchPermutationMatching(unittest.TestCase):
    def test_bc1_square_tile_search_matching(self):
        """search() must identify ground truth BC1 swizzle sequence as the minimum score winner."""
        bpe = 8
        tile_bytes = 512
        tw, th = 8, 8
        wpx = 64
        wb = wpx // 4
        hb = 8
        seq_true = "xyxyxy"

        disp = make_synthetic_gradient_bc1(wb, hb)
        data = swizzle_blocks(disp, seq_true, tw, th, wb)

        results = swizzle_search.search(data, bpe, tile_bytes, [wpx], ntiles=2)
        self.assertGreater(len(results), 0, "search() must return candidate results")

        best_score, best_wpx, best_seq = results[0]
        self.assertEqual(best_seq, seq_true, "Best candidate must match ground truth swizzle")
        self.assertEqual(best_wpx, wpx, "Best candidate must match ground truth width")

        second_score = results[1][0]
        self.assertLess(best_score, second_score, "Ground truth score must be strictly lower than runner up")

    def test_bc3_square_tile_search_matching(self):
        """search() must identify ground truth BC3 (bpe=16) swizzle sequence as winner."""
        bpe = 16
        tile_bytes = 1024
        tw, th = 8, 8
        wpx = 64
        wb = wpx // 4
        hb = 8
        seq_true = "yxyxxy"

        disp = make_synthetic_gradient_bc3(wb, hb)
        data = swizzle_blocks(disp, seq_true, tw, th, wb)

        results = swizzle_search.search(data, bpe, tile_bytes, [wpx], ntiles=2)
        self.assertGreater(len(results), 0)

        best_score, best_wpx, best_seq = results[0]
        self.assertEqual(best_seq, seq_true)
        self.assertEqual(best_wpx, wpx)
        self.assertLess(best_score, results[1][0])

    def test_asymmetric_tile_search_matching(self):
        """search() must match ground truth for non-square tiles where bits is odd."""
        bpe = 8
        tile_bytes = 256
        tw, th = 8, 4
        wpx = 64
        wb = wpx // 4
        hb = 8
        seq_true = "xyxyx"

        disp = make_synthetic_gradient_bc1(wb, hb)
        data = swizzle_blocks(disp, seq_true, tw, th, wb)

        results = swizzle_search.search(data, bpe, tile_bytes, [wpx], ntiles=4)
        self.assertGreater(len(results), 0)

        best_score, best_wpx, best_seq = results[0]
        self.assertEqual(best_seq, seq_true)
        self.assertEqual(best_wpx, wpx)
        self.assertLess(best_score, results[1][0])

    def test_width_candidate_discrimination(self):
        """search() must discriminate between correct width and wrong candidate widths."""
        bpe = 8
        tile_bytes = 512
        tw, th = 8, 8
        true_wpx = 128
        wb = true_wpx // 4  # 32 blocks (4 tiles wide)
        hb = 8              # 1 tile high
        seq_true = "xyxyxy"

        disp = make_synthetic_gradient_bc1(wb, hb)
        data = swizzle_blocks(disp, seq_true, tw, th, wb)

        # Candidate widths including true width (128) and wrong width (64)
        results = swizzle_search.search(data, bpe, tile_bytes, [64, 128], ntiles=4)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][1], true_wpx)
        self.assertEqual(results[0][2], seq_true)
        # Verify score of correct width is strictly better than wrong width
        score_true_width = results[0][0]
        wrong_width_scores = [r[0] for r in results if r[1] != true_wpx]
        self.assertLess(score_true_width, min(wrong_width_scores))

    def test_search_scores_are_sorted(self):
        """search() results must be sorted in ascending order of roughness score."""
        bpe = 8
        tile_bytes = 512
        tw, th = 8, 8
        wb, hb = 16, 8
        disp = make_synthetic_gradient_bc1(wb, hb)
        data = swizzle_blocks(disp, "xyxyxy", tw, th, wb)

        results = swizzle_search.search(data, bpe, tile_bytes, [64], ntiles=2)
        scores = [r[0] for r in results]
        self.assertEqual(scores, sorted(scores))

    def test_search_skips_incompatible_tile_widths(self):
        """search() must cleanly skip widths where wblocks is not divisible by tw."""
        bpe = 8
        tile_bytes = 512
        data = bytes(512 * 4)
        results = swizzle_search.search(data, bpe, tile_bytes, [48], ntiles=2)
        self.assertEqual(results, [])

    def test_search_empty_when_buffer_too_small(self):
        """search() must return empty list if raw data is smaller than one row of tiles."""
        bpe = 8
        tile_bytes = 65536
        data = bytes(128)
        results = swizzle_search.search(data, bpe, tile_bytes, [1024], ntiles=48)
        self.assertEqual(results, [])


class TestSwizzleSearchCliAndRender(unittest.TestCase):
    @unittest.skipUnless(HAVE_NUMPY and HAVE_PIL, "numpy and PIL required")
    def test_default_tile_target_previews_a_short_complete_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "two_tiles.bin"
            preview = Path(td) / "preview.png"
            red_block = bytes.fromhex("00 f8 00 00 00 00 00 00")
            green_block = bytes.fromhex("e0 07 00 00 00 00 00 00")
            blob.write_bytes(red_block * 64 + green_block * 64)

            result = subprocess.run(
                [sys.executable, "-I", str(TOOL), str(blob), "--tile", "512", "--widths", "64",
                 "--out", str(preview)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with Image.open(preview) as image:
                self.assertEqual(image.size, (64, 32))
                expected_row = bytes((255, 0, 0)) * 32 + bytes((0, 255, 0)) * 32
                self.assertEqual(image.tobytes(), expected_row * 32)

    @unittest.skipUnless(HAVE_NUMPY, "numpy required")   # the tool imports numpy before it parses arguments
    def test_cli_bad_numeric_arguments_are_usage_errors(self):
        """Bad --bpe/--tile/--widths/--tiles must be usage errors, not arithmetic tracebacks."""
        cases = [
            (["--bpe", "0"], "--bpe must be at least 1"),
            (["--bpe", "4"], "--bpe must be 8 or 16"),
            (["--tile", "0"], "--tile must be a multiple"),
            (["--tile", "1"], "--tile must be a multiple"),
            (["--tile", "3"], "--tile must be a multiple"),
            (["--tile", "24"], "--tile / --bpe must be a power of two"),
            (["--widths", "0"], "--widths must be positive multiples of 4"),
            (["--widths", "abc"], "--widths must be a comma-separated list"),
            (["--tiles", "0"], "--tiles must be at least 1"),
        ]
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "valid.bin"
            blob.write_bytes(bytes(8 * 16 * 8))
            for args, expected in cases:
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-I", str(TOOL), str(blob)] + args,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(expected, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(HAVE_NUMPY, "numpy required")
    def test_cli_missing_input_is_an_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "definitely_missing_path")
            result = subprocess.run(
                [sys.executable, "-I", str(TOOL), missing], capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("ERROR: cannot read", result.stderr)
        self.assertIn("definitely_missing_path", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(HAVE_NUMPY, "numpy required")   # the tool imports numpy before it parses arguments
    def test_preview_with_no_fitting_layout_reports_the_problem(self):
        """CLI with tiny blob and --out must exit 1, report 'no layout', and not create file."""
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "tiny.bin"
            blob.write_bytes(bytes(32))
            preview = Path(td) / "preview.png"
            result = subprocess.run(
                [sys.executable, "-I", str(TOOL), str(blob), "--out", str(preview)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("no layout", result.stderr.lower())
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(preview.exists())

    @unittest.skipUnless(HAVE_NUMPY, "numpy required")
    def test_cli_successful_search_output(self):
        """CLI on valid synthetic buffer must exit 0 and print formatted scores and bit sequences."""
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "valid.bin"
            blob.write_bytes(bytes(8 * 16 * 8))
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(TOOL),
                    str(blob),
                    "--bpe",
                    "8",
                    "--tile",
                    "512",
                    "--widths",
                    "64",
                    "--tiles",
                    "2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("score", result.stdout)
            self.assertIn("bits(low->high)", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(HAVE_NUMPY and HAVE_PIL, "numpy and PIL required")
    def test_preview_rejects_input_and_hardlink_alias(self):
        """A preview must not overwrite the input or a hard link to it."""
        for hardlink in (False, True):
            with self.subTest(hardlink=hardlink):
                with tempfile.TemporaryDirectory() as td:
                    blob = Path(td) / "input.png"
                    original = bytes(8 * 16 * 8)
                    blob.write_bytes(original)

                    output = Path(td) / "alias.png" if hardlink else blob
                    if hardlink:
                        try:
                            os.link(blob, output)
                        except (OSError, NotImplementedError) as e:
                            self.skipTest(f"hard links unavailable: {e}")

                    result = subprocess.run(
                        [
                            sys.executable, "-I", str(TOOL), str(blob),
                            "--bpe", "8",
                            "--tile", "512",
                            "--widths", "64",
                            "--tiles", "2",
                            "--out", str(output),
                        ],
                        capture_output=True,
                        text=True,
                    )

                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("overwrite", result.stderr.lower())
                    self.assertEqual(blob.read_bytes(), original)
                    self.assertEqual(output.read_bytes(), original)

    @unittest.skipUnless(HAVE_NUMPY and HAVE_PIL, "numpy and PIL required")
    def test_preview_ranks_thin_candidates_by_their_existing_edges(self) -> None:
        from PIL import Image

        green = bytes.fromhex("e007 0000 00000000")
        white = bytes.fromhex("ffff 0000 00000000")
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "input.bin"
            preview = Path(td) / "preview.png"
            # Two complete green tiles for an 8x8 candidate. The narrower
            # candidate also samples a third tile and sees a green/white edge.
            blob.write_bytes(green * 4 + white * 2)
            result = subprocess.run(
                [sys.executable, "-I", str(TOOL), str(blob), "--tile", "16",
                 "--widths", "4,8", "--tiles", "3", "--out", str(preview)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with Image.open(preview) as image:
                self.assertEqual(image.size, (8, 8))
                self.assertEqual(image.getcolors(), [(64, (0, 255, 0))])
            self.assertNotIn("nan", result.stdout.lower())

    @unittest.skipUnless(HAVE_NUMPY and HAVE_PIL, "numpy and PIL required")
    def test_render_bc1_synthetic_output(self):
        """render() must decode synthetic BC1 data and save a valid PNG preview."""
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "preview_bc1.png"
            data = bytes(8 * 16 * 8)
            swizzle_search.render(data, 8, 512, 64, "xyxyxy", str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
            with open(out_path, "rb") as f:
                header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

    @unittest.skipUnless(HAVE_NUMPY and HAVE_PIL, "numpy and PIL required")
    def test_render_bc3_synthetic_output(self):
        """render() must decode synthetic BC3/BC10 data and save a valid PNG preview."""
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "preview_bc3.png"
            data = bytes(16 * 16 * 8)
            swizzle_search.render(data, 16, 1024, 64, "xyxyxy", str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
            with open(out_path, "rb") as f:
                header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")


def _load_swizzle_search_no_numpy():
    if HAVE_NUMPY:
        return swizzle_search
    spec = importlib.util.spec_from_file_location("swizzle_search_out_guard_test", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": mock.MagicMock()}):
        spec.loader.exec_module(mod)
    return mod


class TestSwizzleSearchOutGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_swizzle_search_no_numpy()

    def test_bad_out_paths_fail_before_search(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            blob = root / "blob.bin"
            blob.write_bytes(bytes(64))
            existing_dir = root / "outdir"
            existing_dir.mkdir()
            regular_file = root / "notadir.txt"
            regular_file.write_text("x", encoding="utf-8")

            bad_outs = (
                "",
                "   ",
                "bad\x00out.png",
                str(existing_dir),
                str(root / "trailing") + os.sep,
                str(regular_file / "sub" / "preview.png"),
            )
            for bad in bad_outs:
                with self.subTest(out=bad):
                    err = io.StringIO()
                    with mock.patch.object(self.mod, "search") as mock_search:
                        with mock.patch.object(sys, "stderr", err):
                            rc = self.mod.main([str(blob), "--out", bad])
                    self.assertEqual(rc, 2)
                    mock_search.assert_not_called()
                    self.assertIn("ERROR:", err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())

    def test_valid_nested_out_creates_parent_dirs_and_handles_render_oserror(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            blob = root / "blob.bin"
            blob.write_bytes(bytes(512))
            nested_out = root / "nested" / "sub" / "preview.png"

            with mock.patch.object(self.mod, "search", return_value=[(1.0, 64, "xyxy")]), \
                 mock.patch.object(self.mod, "render") as mock_render:
                rc = self.mod.main([str(blob), "--bpe", "8", "--tile", "128", "--widths", "64", "--out", str(nested_out)])
            self.assertIn(rc, (0, None))
            self.assertTrue((root / "nested" / "sub").is_dir())
            mock_render.assert_called_once()

            err = io.StringIO()
            with mock.patch.object(self.mod, "search", return_value=[(1.0, 64, "xyxy")]), \
                 mock.patch.object(self.mod, "render", side_effect=PermissionError("denied")), \
                 mock.patch.object(sys, "stderr", err):
                rc_err = self.mod.main([str(blob), "--bpe", "8", "--tile", "128", "--widths", "64", "--out", str(nested_out)])
            self.assertEqual(rc_err, 2)
            self.assertIn("ERROR:", err.getvalue())
            self.assertIn("denied", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_nested_out_does_not_create_dirs_when_blob_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing_blob = root / "missing.bin"
            nested_out = root / "never_created" / "sub" / "preview.png"
            err = io.StringIO()
            with mock.patch.object(sys, "stderr", err):
                rc = self.mod.main([str(missing_blob), "--out", str(nested_out)])
            self.assertEqual(rc, 2)
            self.assertFalse((root / "never_created").exists())
            self.assertIn("ERROR:", err.getvalue())

    def test_render_valueerror_unknown_extension_returns_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            blob = root / "blob.bin"
            blob.write_bytes(bytes(512))
            bad_ext_out = root / "preview.unrecognized"
            err = io.StringIO()
            with mock.patch.object(self.mod, "search", return_value=[(1.0, 64, "xyxy")]), \
                 mock.patch.object(self.mod, "render", side_effect=ValueError("unknown file extension: .unrecognized")), \
                 mock.patch.object(sys, "stderr", err):
                rc = self.mod.main([str(blob), "--bpe", "8", "--tile", "128", "--widths", "64", "--out", str(bad_ext_out)])
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", err.getvalue())
            self.assertIn("unknown file extension: .unrecognized", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


def test_preview_with_no_fitting_layout_reports_the_problem(tmp_path=None):
    """Module-level compatibility wrapper for pytest with tmp_path fixture."""
    if not HAVE_NUMPY:
        raise unittest.SkipTest("numpy required")
    td = None
    if tmp_path is None:
        td = tempfile.TemporaryDirectory()
        tmp_path = Path(td.name)
    try:
        blob = tmp_path / "tiny.bin"
        blob.write_bytes(bytes(32))
        preview = tmp_path / "preview.png"
        result = subprocess.run(
            [sys.executable, "-I", str(TOOL), str(blob), "--out", str(preview)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "no layout" in result.stderr.lower()
        assert "Traceback" not in result.stderr
        assert not preview.exists()
    finally:
        if td is not None:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()
