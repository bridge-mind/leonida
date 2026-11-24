"""Tests for tools/tex_export.py, on synthetic textures and swizzle equations (no game data, no addrlib header).

    python -m pytest tests/test_tex_export.py -q
"""
import pytest  # noqa: E402 (keeps the CI bash loop from running this file)

import json
import os
import struct
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

np = pytest.importorskip("numpy")
import tex_export as T  # noqa: E402


def linear_eq(bpe, xor=False):
    """A 4 KB equation in the shape eq4k returns: byte bits, then x bits, then y bits (row-major in the tile).
    With xor=True each x bit also takes a y bit, which keeps it a bijection but no longer row-major."""
    lb = (4096 // bpe).bit_length() - 1
    xb, yb = (lb + 1) // 2, lb - (lb + 1) // 2
    low = bpe.bit_length() - 1
    xs = [(1 << k, (1 << (k % yb)) if xor else 0) for k in range(xb)]
    return [(0, 0)] * low + xs + [(0, 1 << k) for k in range(yb)]


@pytest.fixture
def eq(monkeypatch):
    """Install synthetic equations in eq4k's cache, so nothing reads the addrlib header."""
    def install(bpe, xor=False):
        monkeypatch.setitem(T._EQ, bpe, linear_eq(bpe, xor))
    return install


def tile(elements, offs, nbytes):
    """Inverse of detiling: put each row-major element at the address detile_offsets gives it."""
    bpe = elements.shape[1]
    buf = np.zeros(nbytes, np.uint8)
    buf.reshape(-1, bpe)[offs] = elements
    return buf.tobytes()


def gradient(we, he, bpe):
    y, x = np.mgrid[0:he, 0:we]
    e = np.zeros((he * we, bpe), np.uint8)
    e[:, 0] = (x * 255 // max(1, we - 1)).reshape(-1)
    e[:, 1] = (y * 255 // max(1, he - 1)).reshape(-1)
    e[:, 2] = 128
    e[:, 3] = 255
    return e


def noise(nbytes, seed=1):
    return np.random.default_rng(seed).integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def write_blob(carve, archive, off, data):
    d = os.path.join(carve, archive)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{off:012x}.bin"), "wb") as f:
        f.write(data)


def texture(fmt, fmt_name, w, h, mips, data_off=0, gpu=0x1000, **extra):
    t = {"archive": "x.rpf", "cpu": 0xABC, "q_index": 3, "gpu": gpu, "gpu_size": 0, "fmt": fmt,
         "fmt_name": fmt_name, "w": w, "h": h, "mips": mips, "data_off": data_off}
    t.update(extra)
    return t


# ---- eq4k ----

def test_import_does_not_read_the_addrlib_header(monkeypatch):
    import importlib.util
    import amd_swizzle

    def no_header(*a, **k):
        raise AssertionError("load() called at import")

    monkeypatch.setattr(amd_swizzle, "load", no_header)
    spec = importlib.util.spec_from_file_location("tex_export_fresh", T.__file__)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_eq4k_reads_the_patterns_once_and_picks_the_entry_for_bpe(monkeypatch):
    calls = []
    bits8 = [0, 0, 0] + [1 << k for k in range(5)] + [(1 << k) << 16 for k in range(4)]
    bits4 = [0, 0] + [1 << k for k in range(5)] + [(1 << k) << 16 for k in range(5)]
    nibbles = {"NIBBLE01": [bits8[:8], bits4[:8]], "NIBBLE2": [bits8[8:12], bits4[8:12]],
               "NIBBLE3": [[0] * 4], "NIBBLE4": [[0] * 4]}
    tables = {"GFX10_SW_4K_S_PATINFO": [(1, 1, 1, 0, 0, "SW_4K_S 4 bpe"), (1, 0, 0, 0, 0, "SW_4K_S 8 bpe")]}

    def fake_load():
        calls.append(1)
        return nibbles, tables

    monkeypatch.setattr(T, "load", fake_load)
    monkeypatch.setattr(T, "_PATTERNS", [])
    monkeypatch.setattr(T, "_EQ", {})
    assert T.eq4k(8) == linear_eq(8)
    assert T.eq4k(4) == linear_eq(4)
    assert T.eq4k(8) == linear_eq(8)
    assert len(calls) == 1


# ---- detile_offsets / level_bytes ----

@pytest.mark.parametrize("bpe", [1, 2, 4, 8, 16])
def test_linear_equation_detiles_row_major_tiles(eq, bpe):
    eq(bpe)
    offs, bw, bh = T.detile_offsets(bpe, 2 * 64, 2 * 64)
    assert bw * bh * bpe == 4096
    we = 2 * 64
    y, x = np.divmod(np.arange(we * we), we)
    tile_index = (y // bh) * (we // bw) + x // bw
    assert np.array_equal(offs, tile_index * (4096 // bpe) + (y % bh) * bw + x % bw)


@pytest.mark.parametrize("bpe", [4, 8, 16])
def test_offsets_are_a_permutation_of_whole_tiles(eq, bpe):
    eq(bpe, xor=True)
    _, bw, bh = T.detile_offsets(bpe, 1, 1)
    we, he = 3 * bw, 2 * bh
    offs, _, _ = T.detile_offsets(bpe, we, he)
    assert sorted(offs.tolist()) == list(range(we * he))
    assert T.level_bytes(we, he, bpe, bw, bh) == 6 * 4096


def test_partial_tiles_are_padded_to_whole_tiles(eq):
    eq(4)
    _, bw, bh = T.detile_offsets(4, 1, 1)
    assert T.level_bytes(1, 1, 4, bw, bh) == 4096
    assert T.level_bytes(bw + 1, bh, 4, bw, bh) == 2 * 4096
    offs, _, _ = T.detile_offsets(4, bw + 1, 1)
    assert int(offs.max()) == 4096 // 4  # first element of the second tile
    assert len(set(offs.tolist())) == bw + 1


# ---- dds_bytes ----

def test_dds_header_is_a_dx10_texture2d():
    payload = bytes(range(64))
    b = T.dds_bytes(16, 8, 71, payload)
    assert len(b) == 128 + 20 + len(payload)
    assert b[:4] == b"DDS "
    size, flags, h, w, linear = struct.unpack_from("<IIIII", b, 4)
    assert (size, h, w, linear) == (124, 8, 16, len(payload))
    assert flags & 0x1007 == 0x1007 and flags & 0x80000  # caps|height|width|pixelformat, linearsize
    assert struct.unpack_from("<II4s", b, 76) == (32, 4, b"DX10")
    assert struct.unpack_from("<I", b, 108)[0] == 0x1000
    fmt, dim, misc, array, misc2 = struct.unpack_from("<IIIII", b, 128)
    assert (fmt, dim, misc, array, misc2) == (71, 3, 0, 1, 0)
    assert b[148:] == payload


# ---- export_one ----

def test_r16_export_scores_words_across_a_byte_boundary(eq: Callable[[int], None], tmp_path: Path) -> None:
    carve = tmp_path / "carve"
    out = tmp_path / "out"
    eq(2)
    offsets, _, _ = T.detile_offsets(2, 4, 4)
    top = np.tile(np.arange(254, 258, dtype="<u2"), 4)
    # The smaller mip's tile also contains padding. As a 4x4 candidate, its
    # alternating word values look smoother when the high byte is scored separately.
    lower_and_padding = np.tile(np.array([0xFF00, 0xFF05], dtype="<u2"), 8)
    data = b"".join(tile(values.view(np.uint8).reshape(-1, 2), offsets, 4096)
                    for values in (lower_and_padding, top))
    write_blob(carve, "x.rpf", 0x1000, data)
    record = texture(56, "R16_UNORM", 4, 4, 2)

    assert T.export_one(record, carve, out, {}) == "ok:last"
    dds = (out / "x.rpf" / record["export"]["file"]).read_bytes()
    assert dds[148:] == top.tobytes()


def build_case(eq, tmp_path, fmt, top_first):
    """A 64x64 texture with two mips: a smooth top mip and a noise mip, in either order in the GPU blob."""
    bpe, blk = T.FMT[fmt]
    eq(bpe)
    w = h = 64
    we, he = w // blk, h // blk
    offs, bw, bh = T.detile_offsets(bpe, we, he)
    top_bytes = T.level_bytes(we, he, bpe, bw, bh)
    small = T.level_bytes(max(1, (w // 2) // blk), max(1, (h // 2) // blk), bpe, bw, bh)
    top = gradient(we, he, bpe)
    tiled = tile(top, offs, top_bytes)
    blob = tiled + noise(small) if top_first else noise(small) + tiled
    carve = str(tmp_path / "carve")
    write_blob(carve, "x.rpf", 0x1000, blob)
    return carve, top.tobytes()


@pytest.mark.parametrize("fmt,name", [(10, "R16G16B16A16_FLOAT"), (28, "R8G8B8A8_UNORM"),
                                      # Pillow 13.0.0 cannot decode these three: scored on raw elements instead
                                      (81, "BC4_SNORM"), (87, "B8G8R8A8_UNORM"), (91, "B8G8R8A8_UNORM_SRGB")])
@pytest.mark.parametrize("top_first", [True, False])
def test_export_picks_the_top_mip_placement_and_writes_dds(eq, tmp_path, fmt, name, top_first):
    if fmt in T.PILLOW_OK:
        pytest.importorskip("PIL")
    carve, expected = build_case(eq, tmp_path, fmt, top_first)
    out = str(tmp_path / "out")
    t = texture(fmt, name, 64, 64, mips=2)
    r = T.export_one(t, carve, out, {})
    place = "first" if top_first else "last"
    assert r == "ok:" + place
    assert t["export"]["place"] == place
    assert t["export"]["heuristic"] is True
    assert t["export"]["file"] == f"{0xABC:012x}_3_64x64_{name}.dds"
    with open(os.path.join(out, "x.rpf", t["export"]["file"]), "rb") as f:
        dds = f.read()
    assert struct.unpack_from("<I", dds, 128)[0] == fmt
    assert dds[148:] == expected


def test_fmt_reuses_the_indexer_table_without_12_byte_rgb32():
    from tex_headers import FMT_INFO
    assert T.FMT == {f: v for f, v in FMT_INFO.items() if f not in (6, 7, 8)}
    assert len(T.FMT) == 56


@pytest.mark.parametrize("fmt", sorted(T.FMT))
def test_every_tileable_format_exports_exact_top_mip_and_header(eq, tmp_path, fmt):
    bpe, blk = T.FMT[fmt]
    eq(bpe)
    w, h = 43, 37  # not a whole number of tiles or BC blocks
    we, he = (w + blk - 1) // blk, (h + blk - 1) // blk
    offs, bw, bh = T.detile_offsets(bpe, we, he)
    top = np.frombuffer(noise(we * he * bpe, seed=fmt), np.uint8).reshape(-1, bpe)
    carve = str(tmp_path / "carve")
    write_blob(carve, "x.rpf", 0x1000, tile(top, offs, T.level_bytes(we, he, bpe, bw, bh)))
    t = texture(fmt, "F", w, h, mips=1)
    out = str(tmp_path / "out")
    assert T.export_one(t, carve, out, {}) == "ok:first"
    assert t["export"]["heuristic"] is False  # one mip: both placements are the same offset
    with open(os.path.join(out, "x.rpf", t["export"]["file"]), "rb") as f:
        dds = f.read()
    flags, pitch = struct.unpack_from("<I", dds, 8)[0], struct.unpack_from("<I", dds, 20)[0]
    if blk == 1:
        assert (flags, pitch) == (0x1007 | 0x8, w * bpe)
    else:
        assert (flags, pitch) == (0x1007 | 0x80000, we * he * bpe)
    assert struct.unpack_from("<I", dds, 128)[0] == fmt
    assert dds[148:] == top.tobytes()


@pytest.mark.parametrize("pillow_failure", ["missing", "unreadable DDS"])
def test_mip_scoring_uses_raw_elements_when_pillow_cannot_decode(eq, tmp_path, monkeypatch, pillow_failure) -> None:
    if pillow_failure == "missing":
        monkeypatch.setitem(sys.modules, "PIL", None)
    else:
        image = pytest.importorskip("PIL.Image")

        def cannot_open(*args: object, **kwargs: object) -> None:
            raise OSError("synthetic Pillow DDS decode failure")

        monkeypatch.setattr(image, "open", cannot_open)

    carve, expected = build_case(eq, tmp_path, 28, top_first=False)
    out = tmp_path / "out"
    row = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)

    assert T.export_one(row, carve, str(out), {}) == "ok:last"
    assert (out / "x.rpf" / row["export"]["file"]).read_bytes()[148:] == expected


def test_gpu_chain_parts_are_read_in_order(eq, tmp_path):
    carve, expected = build_case(eq, tmp_path, 10, top_first=True)
    with open(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"), "rb") as f:
        blob = f.read()
    os.remove(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"))
    cut = 12345
    write_blob(carve, "x.rpf", 0x1000, blob[:cut])
    write_blob(carve, "x.rpf", 0x9000, blob[cut:])
    t = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=2,
                gpu_chain=[[0x1000, cut], [0x9000, len(blob) - cut]])
    out = str(tmp_path / "out")
    assert T.export_one(t, carve, out, {}) == "ok:first"
    with open(os.path.join(out, "x.rpf", t["export"]["file"]), "rb") as f:
        assert f.read()[148:] == expected


def test_blob_cache_is_reused_for_the_same_blob(eq, tmp_path):
    carve, _ = build_case(eq, tmp_path, 10, top_first=True)
    cache = {}
    t = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=2)
    assert T.export_one(t, carve, str(tmp_path / "out"), cache) == "ok:first"
    os.remove(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"))  # a second read would fail
    t2 = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=2)
    assert T.export_one(t2, carve, str(tmp_path / "out"), cache) == "ok:first"


def test_blob_cache_distinguishes_different_gpu_chains_at_same_offset(eq, tmp_path):
    carve, expected = build_case(eq, tmp_path, 10, top_first=True)
    with open(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"), "rb") as f:
        blob = f.read()
    os.remove(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"))
    cut = 12345
    write_blob(carve, "x.rpf", 0x1000, blob[:cut])
    write_blob(carve, "x.rpf", 0x9000, blob[cut:])
    cache = {}
    out = str(tmp_path / "out")
    short = texture(10, "R16G16B16A16_FLOAT", 16, 16, mips=1, gpu=0x1000, gpu_size=cut)
    assert T.export_one(short, carve, out, cache) == "ok:first"
    chained = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=2, q_index=4,
                      gpu=0x1000, gpu_size=len(blob),
                      gpu_chain=[[0x1000, cut], [0x9000, len(blob) - cut]])
    assert T.export_one(chained, carve, out, cache) == "ok:first"
    with open(os.path.join(out, "x.rpf", chained["export"]["file"]), "rb") as f:
        assert f.read()[148:] == expected


@pytest.mark.parametrize("fmt", [9999, 6, 7, 8])  # 6-8: 12-byte R32G32B32 has no SW_4K_S equation
def test_unsupported_format_is_reported(fmt):
    assert T.export_one(texture(fmt, "X", 4, 4, 1), "unused", "unused", {}) == "unsupported format"


@pytest.mark.parametrize("fmt,bpe", [(61, 1), (56, 2), (10, 8), (28, 4)])
@pytest.mark.parametrize("w,h", [(5, 3), (5, 1), (1, 5)])
def test_score_prefers_a_smooth_gradient(fmt, bpe, w, h):
    if fmt in T.PILLOW_OK:
        pytest.importorskip("PIL")
    pixels = max(w, h)
    values = [i * 240 // (pixels - 1) for i in range(pixels)]
    zigzag = values[::2] + values[1::2][::-1]
    copies = min(w, h)
    ramp = b"".join(bytes([value]) * bpe for value in values * copies)
    rough = b"".join(bytes([value]) * bpe for value in zigzag * copies)
    assert T.score_image(w, h, fmt, ramp, bpe) < T.score_image(w, h, fmt, rough, bpe)


def test_bc2_srgb_is_scored_as_bc2():
    pytest.importorskip("PIL")
    payload = noise(4 * 4 * 16)
    assert T.score_image(16, 16, 75, payload, 16) == T.score_image(16, 16, 74, payload, 16)


@pytest.mark.parametrize("fmt", [81, 87, 91])
def test_score_falls_back_to_raw_elements_when_pillow_cannot_decode(monkeypatch, fmt):
    PIL_Image = pytest.importorskip("PIL.Image")
    bpe, blk = T.FMT[fmt]

    def unimplemented(*args, **kwargs):
        raise NotImplementedError(f"Unimplemented DXGI format {fmt}")
    monkeypatch.setattr(PIL_Image, "open", unimplemented)
    payload = noise(4 * 4 * bpe, seed=fmt)
    score = T.score_image(4 * blk, 4 * blk, fmt, payload, bpe)
    assert 0 < score < 1e6
    monkeypatch.setattr(T, "PILLOW_OK", set())  # the branch formats outside PILLOW_OK take
    assert score == T.score_image(4 * blk, 4 * blk, fmt, payload, bpe)


def test_single_element_has_no_edge_transitions():
    assert T.score_image(1, 1, 61, b"\x80", 1) == 0


def test_data_outside_the_blob_is_reported(eq, tmp_path):
    eq(8)
    carve = str(tmp_path / "carve")
    write_blob(carve, "x.rpf", 0x1000, bytes(100))
    t = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=1)
    assert T.export_one(t, carve, str(tmp_path / "out"), {}) == "data outside GPU blob"
    assert "export" not in t


# ---- main ----

def test_main_writes_the_log_and_honours_limit(eq, tmp_path, monkeypatch, capsys):
    carve, _ = build_case(eq, tmp_path, 10, top_first=True)
    index = tmp_path / "textures_index.json"
    good = texture(10, "R16G16B16A16_FLOAT", 64, 64, mips=2)
    bad = texture(9999, "X", 4, 4, 1, gpu=0x2000)
    index.write_text(json.dumps([bad, good]), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), carve, str(out)])
    T.main()
    log = json.loads((out / "export_log.json").read_text(encoding="utf-8"))
    assert [t["gpu"] for t in log] == [0x1000, 0x2000]  # sorted by (archive, gpu)
    assert log[0]["export"]["place"] == "first" and "export" not in log[1]
    assert "2 textures:" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), carve, str(out), "--limit", "1"])
    T.main()
    assert len(json.loads((out / "export_log.json").read_text(encoding="utf-8"))) == 1


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_main_rejects_a_limit_below_one(tmp_path, monkeypatch, capsys, value):
    # --limit 0 exported every texture (0 read as "no limit") and -1 silently dropped the last one.
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), str(tmp_path), str(out), "--limit", value])
    with pytest.raises(SystemExit) as e:
        T.main()
    assert e.value.code == 2
    assert "--limit" in capsys.readouterr().err
    assert not out.exists()


# ---- --png ----

def read_png(path):
    """(width, height, RGBA bytes) of a PNG as encode_png writes it: 8-bit RGBA, filter 0 on every row."""
    import zlib
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(b):
        n, kind = struct.unpack_from(">I4s", b, pos)
        body = b[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    w, h, depth, ctype, _, _, interlace = ihdr
    assert (depth, ctype, interlace) == (8, 6, 0)
    raw = zlib.decompress(idat)
    stride = 1 + w * 4
    assert len(raw) == h * stride
    assert all(raw[y * stride] == 0 for y in range(h))
    return w, h, b"".join(raw[y * stride + 1:(y + 1) * stride] for y in range(h))


def png_main(tmp_path, monkeypatch, capsys, textures, carve, *flags):
    index = tmp_path / "textures_index.json"
    index.write_text(json.dumps(textures), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), carve, str(out), *flags])
    code = T.main()
    return out, code, capsys.readouterr().out


def tree(out):
    return sorted(os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs)


def test_main_accepts_png(tmp_path, monkeypatch):
    # the usage line documented --png, but argparse stopped with "unrecognized arguments"
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), str(tmp_path), str(tmp_path / "out"), "--png"])
    assert T.main() is None


def test_png_is_written_beside_the_dds_with_the_texels(eq, tmp_path, monkeypatch, capsys):
    carve, expected = build_case(eq, tmp_path, 28, top_first=True)
    t = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [t], carve, "--png")
    assert code is None
    stem = f"{0xABC:012x}_3_64x64_R8G8B8A8_UNORM"
    assert tree(out) == ["export_log.json", f"x.rpf{os.sep}{stem}.dds", f"x.rpf{os.sep}{stem}.png"]
    w, h, rgba = read_png(out / "x.rpf" / (stem + ".png"))
    assert (w, h) == (64, 64)
    assert rgba == expected  # RGBA8 texels pass through unchanged (the gradient sets alpha to 255)
    log = json.loads((out / "export_log.json").read_text(encoding="utf-8"))
    assert log[0]["export"]["png"] == "written"
    assert text.splitlines() == ["1 textures: {'ok:first': 1} -> " + str(out), "png: {'written': 1}"]


def test_packed_r10g10b10a2_writes_a_png(eq, tmp_path, monkeypatch, capsys):
    eq(4)
    packed = struct.pack("<II", 1023 | (3 << 30), (1023 << 20) | (1 << 30))
    carve = str(tmp_path / "carve")
    write_blob(carve, "x.rpf", 0x1000, packed + bytes(4096 - len(packed)))
    t = texture(24, "R10G10B10A2_UNORM", 2, 1, mips=1)
    out, code, _ = png_main(tmp_path, monkeypatch, capsys, [t], carve, "--png")
    assert code is None
    stem = out / "x.rpf" / f"{0xABC:012x}_3_2x1_R10G10B10A2_UNORM"
    assert stem.with_suffix(".dds").read_bytes()[148:] == packed
    log = json.loads((out / "export_log.json").read_text(encoding="utf-8"))
    assert log[0]["export"]["png"] == "written"
    assert read_png(stem.with_suffix(".png")) == (2, 1, bytes((255, 0, 0, 255, 0, 0, 255, 85)))


def test_undecodable_format_keeps_the_dds_and_counts_unsupported(eq, tmp_path, monkeypatch, capsys):
    carve, expected = build_case(eq, tmp_path, 41, top_first=True)  # R32_FLOAT: dds_png has no decoder
    t = texture(41, "R32_FLOAT", 64, 64, mips=2)
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [t], carve, "--png")
    assert code is None  # main() returns None, so the process exits 0
    assert tree(out) == ["export_log.json", f"x.rpf{os.sep}{0xABC:012x}_3_64x64_R32_FLOAT.dds"]
    assert (out / "x.rpf" / f"{0xABC:012x}_3_64x64_R32_FLOAT.dds").read_bytes()[148:] == expected
    assert json.loads((out / "export_log.json").read_text(encoding="utf-8"))[0]["export"]["png"] == "unsupported"
    assert text.splitlines()[1:] == ["png: {'unsupported': 1}"]


def test_mixed_run_counts_each_kind(eq, tmp_path, monkeypatch, capsys):
    carve, _ = build_case(eq, tmp_path, 28, top_first=True)
    with open(os.path.join(carve, "x.rpf", f"{0x1000:012x}.bin"), "rb") as f:
        write_blob(carve, "x.rpf", 0x2000, f.read())  # same 4 bpe size, so the equation is already in place
    textures = [texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2),
                texture(41, "R32_FLOAT", 64, 64, mips=2, gpu=0x2000),
                texture(9999, "X", 4, 4, 1, gpu=0x3000)]
    out, code, text = png_main(tmp_path, monkeypatch, capsys, textures, carve, "--png")
    assert text.splitlines()[1] == "png: {'written': 1, 'unsupported': 1}"
    assert len([f for f in tree(out) if f.endswith(".png")]) == 1



@pytest.mark.parametrize("fmt,name,codes,rgba", [
    # 1, 0.5 -> red 255, 128 (green and blue 0, alpha 255)
    (54, "R16_FLOAT", [0x3C00, 0x3800], [255, 0, 0, 255, 128, 0, 0, 255]),
    # (+inf, NaN, -1, 0.5) and (0.25, 2, 0, 1): +inf and 2 saturate, NaN and negatives give 0
    (10, "R16G16B16A16_FLOAT", [0x7C00, 0x7E00, 0xBC00, 0x3800, 0x3400, 0x4000, 0x0000, 0x3C00],
     [255, 0, 0, 128, 64, 255, 0, 255]),
])
def test_half_float_png_export(eq, tmp_path, monkeypatch, capsys, fmt, name, codes, rgba):
    eq(T.FMT[fmt][0])
    texels = struct.pack(f"<{len(codes)}H", *codes)
    carve = str(tmp_path / "carve")
    write_blob(carve, "x.rpf", 0x1000, texels.ljust(4096, b"\0"))
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [texture(fmt, name, 2, 1, 1)], carve, "--png")
    assert code is None
    assert text.splitlines()[1] == "png: {'written': 1}"
    assert next(out.rglob("*.dds")).read_bytes()[148:] == texels
    assert read_png(next(out.rglob("*.png"))) == (2, 1, bytes(rgba))

def test_png_module_is_not_imported_without_the_flag(eq, tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "dds_png", raising=False)
    carve, _ = build_case(eq, tmp_path, 28, top_first=True)
    png_main(tmp_path, monkeypatch, capsys, [texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)], carve)
    assert "dds_png" not in sys.modules
    with open(T.__file__, encoding="utf-8") as f:
        assert not [line for line in f if line.startswith(("import dds_png", "from dds_png"))]


def test_without_png_nothing_changes(eq, tmp_path, monkeypatch, capsys):
    carve, expected = build_case(eq, tmp_path, 28, top_first=True)
    t = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [t], carve)
    stem = f"{0xABC:012x}_3_64x64_R8G8B8A8_UNORM"
    assert tree(out) == ["export_log.json", f"x.rpf{os.sep}{stem}.dds"]
    assert text == f"1 textures: {{'ok:first': 1}} -> {out}\n"
    log = json.loads((out / "export_log.json").read_text(encoding="utf-8"))
    assert set(log[0]["export"]) == {"file", "place", "score", "heuristic"}


def test_failed_png_write_keeps_the_texture_ok_and_is_counted(eq, tmp_path, monkeypatch, capsys):
    carve, _ = build_case(eq, tmp_path, 28, top_first=True)
    stem = f"{0xABC:012x}_3_64x64_R8G8B8A8_UNORM"
    blocked = tmp_path / "out" / "x.rpf" / (stem + ".png")
    blocked.mkdir(parents=True)  # a directory at the PNG's path makes the write fail
    t = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [t], carve, "--png")
    assert code is None
    assert (out / "x.rpf" / (stem + ".dds")).is_file() and blocked.is_dir()
    entry = json.loads((out / "export_log.json").read_text(encoding="utf-8"))[0]["export"]
    assert entry["png"].startswith("error:")
    lines = text.splitlines()
    assert lines[0].startswith("1 textures: {'ok:first': 1}")
    assert lines[1] == "png: {'" + entry["png"] + "': 1}"


def test_stale_export_entry_is_not_counted(eq, tmp_path, monkeypatch, capsys):
    carve, _ = build_case(eq, tmp_path, 28, top_first=True)
    stale = {"file": "old.dds", "place": "first", "score": 1.0, "png": "written"}
    t = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2, gpu=0x5000, export=stale)  # blob 0x5000 does not exist
    out, code, text = png_main(tmp_path, monkeypatch, capsys, [t], carve, "--png")
    assert text.splitlines()[0].startswith("1 textures: {'error:FileNotFoundError': 1}")
    assert text.splitlines()[1] == "png: {}"

def test_texture_io_error_fails_the_run_but_preserves_successes(eq, tmp_path, monkeypatch, capsys):
    carve, _ = build_case(eq, tmp_path, 28, top_first=True)
    good = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2)
    missing = texture(28, "R8G8B8A8_UNORM", 64, 64, mips=2, gpu=0x5000)
    index = tmp_path / "index.json"
    index.write_text(json.dumps([good, missing]), encoding="utf-8")
    out = tmp_path / "out"
    _argv(monkeypatch, index, carve, out)
    assert T.main() == 1
    captured = capsys.readouterr()
    assert "1 texture(s) could not be read or written" in captured.err
    assert "000000005000.bin" in captured.err
    assert "error:FileNotFoundError" in captured.out
    exported, failed = json.loads((out / "export_log.json").read_text(encoding="utf-8"))
    assert (out / "x.rpf" / exported["export"]["file"]).is_file()
    assert "export" not in failed


def test_main_reports_export_log_write_failure(tmp_path, monkeypatch, capsys):
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"
    (out / "export_log.json").mkdir(parents=True)
    _argv(monkeypatch, index, tmp_path, out)
    assert T.main() == 1
    captured = capsys.readouterr()
    assert "ERROR: cannot write" in captured.err
    assert "export_log.json" in captured.err
    assert not captured.out


# ---- bad index / --out guards (#1607's class in main) ----

def _argv(monkeypatch, index, carve, out, *extra):
    monkeypatch.setattr(sys, "argv", ["tex_export.py", str(index), str(carve), str(out), *extra])


def test_main_reports_a_non_list_index_as_one_error_line(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text('{"files": []}', encoding="utf-8")
    out = tmp_path / "out"
    _argv(monkeypatch, index, tmp_path, out)
    assert T.main() == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: cannot read "), err
    assert len(err.strip().splitlines()) == 1, err
    assert not out.exists()


def test_main_reports_an_index_row_without_keys(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text('[{"gpu": 0}]', encoding="utf-8")
    _argv(monkeypatch, index, tmp_path, tmp_path / "out")
    assert T.main() == 2
    err = capsys.readouterr().err
    assert "ERROR: cannot read " in err, err
    assert "row has no 'archive'" in err, err
    assert "Traceback" not in err


def test_main_reads_an_index_saved_with_a_utf8_bom(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text("\ufeff[]", encoding="utf-8")   # Notepad and PowerShell's ">" write a BOM
    out = tmp_path / "out"
    _argv(monkeypatch, index, tmp_path, out)
    assert T.main() is None       # read as an empty texture list; nothing to export
    assert (out / "export_log.json").read_text(encoding="utf-8") == "[]"
    assert capsys.readouterr().err == ""


def test_main_rejects_an_out_that_is_a_file(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    out = tmp_path / "existing.txt"
    out.write_text("x", encoding="utf-8")
    _argv(monkeypatch, index, tmp_path, out)
    assert T.main() == 2
    err = capsys.readouterr().err
    assert "is a file, not a folder" in err, err
    assert out.read_text(encoding="utf-8") == "x"


def test_main_reports_an_out_under_an_existing_file(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    _argv(monkeypatch, index, tmp_path, blocker / "sub")
    assert T.main() == 2
    err = capsys.readouterr().err
    assert "ERROR: cannot create " in err, err
    assert "Traceback" not in err


def test_main_reports_a_dangling_symlink_out(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    link = tmp_path / "dangling-out"
    try:
        os.symlink(str(tmp_path / "missing-target"), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    _argv(monkeypatch, index, tmp_path, link)
    assert T.main() == 2
    err = capsys.readouterr().err
    assert "ERROR: cannot create " in err, err
    assert "Traceback" not in err


def test_main_still_accepts_an_empty_list(tmp_path, monkeypatch, capsys):
    index = tmp_path / "textures_index.json"
    index.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"
    _argv(monkeypatch, index, tmp_path, out)
    assert T.main() in (None, 0)   # main() has no explicit return; sys.exit(None) is exit 0
    assert (out / "export_log.json").is_file()
