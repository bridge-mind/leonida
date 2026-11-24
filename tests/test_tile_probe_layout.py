"""Synthetic block layout, scoring and preview tests for tools/tile_probe.py."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tile_probe


@pytest.mark.parametrize("mode,bpe,tile_width,tile_height", [
    ("4k", 8, 32, 16),
    ("4k", 16, 16, 16),
    ("64k", 8, 128, 64),
    ("64k", 16, 64, 64),
])
def test_tiled_layout_maps_stored_blocks_to_image_positions(mode, bpe, tile_width, tile_height):
    width = 2 * tile_width
    height = 2 * tile_height
    blocks_per_tile = tile_width * tile_height
    positions = tile_probe.layout(width * height, width, mode, bpe)

    for stored, x, y in (
        (1, 1, 0), (2, 0, 1), (4, 2, 0), (8, 0, 2),
        (blocks_per_tile - 1, tile_width - 1, tile_height - 1),
        (blocks_per_tile, tile_width, 0),
        (2 * blocks_per_tile, 0, tile_height),
    ):
        assert positions[stored] == y * width + x
    np.testing.assert_array_equal(np.sort(positions), np.arange(width * height))


@pytest.mark.parametrize("block_count,width", [(768, 48), (1023, 64)])
def test_4k_layout_rejects_incomplete_tile_rows(block_count, width):
    assert tile_probe.layout(block_count, width, "4k", 8) is None


@pytest.mark.parametrize("block,rgb", [
    ("a3 8c ff ff 00 00 00 00", [139, 149, 24]),
    ("fe f8 aa aa aa aa aa aa a3 8c ff ff 00 00 00 00", [196, 149, 24]),
])
def test_block_feature_reads_the_format_specific_endpoint(block, rgb):
    raw = np.frombuffer(bytes.fromhex(block), dtype=np.uint8)
    feature = tile_probe.block_feature(raw.reshape(1, -1), raw.size)
    np.testing.assert_array_equal(feature, [rgb])


@pytest.fixture
def black_and_white_bc1_tiles():
    black = bytes(8)
    white = b"\xff\xff" + bytes(6)
    return b"hdr" + black * 512 + white * 512 + b"tail"


def test_probe_ranks_adjacent_tiles_above_linear_rows(black_and_white_bc1_tiles):
    results = tile_probe.probe(black_and_white_bc1_tiles, offset=3)
    scores = {mode: value for value, bpe, width, mode in results if bpe == 8 and width == 256}
    assert results[0][1:] == (8, 256, "4k")
    assert scores["4k"] == pytest.approx(255 / 63)
    assert scores["linear"] == pytest.approx(255 / 15)


def test_preview_preserves_the_boundary_between_tiles(black_and_white_bc1_tiles, tmp_path):
    image = pytest.importorskip("PIL.Image")
    output = tmp_path / "preview.png"
    assert tile_probe.render(black_and_white_bc1_tiles, 3, 8, 256, "4k", output) == (256, 64)
    with image.open(output) as preview:
        assert preview.size == (256, 64)
        assert preview.getpixel((127, 0)) == (0, 0, 0)
        assert preview.getpixel((128, 0)) == (255, 255, 255)
