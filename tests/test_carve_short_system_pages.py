"""Declared short model pages survive the real carver and folder joiner. Synthetic bytes only."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
import test_restofworld_model_join as fixture  # noqa: E402
import restofworld_model_join as joiner  # noqa: E402


def model_pages(final_size=48, middle_pages=0, first_size=65536):
    system, graphics = fixture.make_standalone_drawable_1x1()
    first = bytearray(system[0].ljust(first_size, b"\0"))
    fixture._write_page_map(first, first_size - 64, middle_pages + 2, 1)
    if final_size >= 22:
        struct.pack_into("<Q", first, 168, fixture.SYS_BASE + first_size + 65536 * middle_pages + 8)
        final = b"\0" * 8 + b"synthetic_tail\0".ljust(final_size - 8, b"\0")
    else:
        final = b"Q" * final_size
    return [bytes(first), *([b"M" * 65536] * middle_pages), final, *graphics]


def compressed_pages(pages):
    """Keep every payload compressed, even when its bytes resemble a raw video."""
    data = bytearray(fixture.pack_synthetic_rpf([])[0][:0x1010])
    offsets = []
    for page in pages:
        data.extend(bytes(-len(data) % 16))
        offsets.append(len(data))
        data.extend(zlib.compress(page))
    return bytes(data), offsets


class ShortSystemPagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.number = 0

    def tearDown(self):
        self.tmp.cleanup()

    def carve(self, data, *options):
        self.number += 1
        source = self.root / f"source-{self.number}.rpf"
        folder = self.root / f"carved-{self.number}"
        source.write_bytes(data)
        run = subprocess.run([sys.executable, "-I", str(ROOT / "tools/rpf9_carve.py"),
                              str(source), str(folder), *map(str, options)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(source.read_bytes(), data)
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["summary"]["files"], len(index["files"]))
        self.assertEqual(index["summary"]["span"], index["summary"]["compressed_bytes"]
                         + index["summary"]["zero_padding"] + index["summary"]["unread_bytes"])
        return source, folder, index

    def joined_cli(self, source):
        output = self.root / f"report-{self.number}-{source.name}.json"
        run = subprocess.run([sys.executable, "-I", str(ROOT / "tools/restofworld_model_join.py"),
                              str(source), str(output), "--quiet"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_actual_carver_and_joiner_preserve_final_page_boundaries(self):
        for middle_pages, first_size in ((0, 65536), (1, 65536), (0, 131072)):
            for final_size in (1, 48, 63, 64):
                with self.subTest(middle_pages=middle_pages, first_size=first_size, final_size=final_size):
                    pages = model_pages(final_size, middle_pages, first_size)
                    data, recorded = fixture.pack_synthetic_rpf([pages])
                    source, folder, index = self.carve(data)
                    self.assertEqual([row["size"] for row in index["files"]], list(map(len, pages)))
                    self.assertEqual([row["off"] for row in index["files"]], recorded[0][0])
                    for row, page in zip(index["files"], pages):
                        self.assertEqual(row["stored"], len(zlib.compress(page)))
                        self.assertEqual((folder / f"{row['off']:012x}.bin").read_bytes(), page)
                    archive = self.joined_cli(source)
                    carved = self.joined_cli(folder)
                    self.assertEqual(archive["resources"], carved["resources"])
                    for report in (archive, carved):
                        self.assertEqual(report["summary"]["joined_resources"], 1)
                        self.assertEqual(report["summary"]["zlib_streams"], len(pages))
                        resource, = report["resources"]
                        self.assertEqual([row["size"] for row in resource["system_streams"]],
                                         list(map(len, pages[:-1])))
                    for field in ("by_kind", "page_shapes", "primary_vertices", "primary_triangles",
                                  "all_lod_vertices", "all_lod_triangles"):
                        self.assertEqual(carved["summary"][field], archive["summary"][field])

    def test_indexed_folder_rejects_omitted_stream_before_graphics(self):
        pages = model_pages()
        data, offsets = compressed_pages([*pages[:-1], b"x" * 47, pages[-1]])
        source, folder, index = self.carve(data)
        self.assertNotIn(offsets[-2], [row["off"] for row in index["files"]])
        for path in (source, folder):
            output = self.root / "interrupted.json"
            run = subprocess.run([sys.executable, "-I", str(ROOT / "tools/restofworld_model_join.py"),
                                  str(path), str(output), "--quiet"], capture_output=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            report = json.loads(output.read_bytes())
            self.assertEqual(report["summary"]["joined_resources"], 0)
            self.assertEqual(report["summary"]["unjoined_candidates"], 1)
        self.assertTrue(joiner._aliases_input(folder / "index.json", folder))
        (folder / "index.json").unlink()
        self.assertTrue(joiner._aliases_input(folder / "index.json", folder))
        self.assertEqual(self.joined_cli(folder)["summary"]["joined_resources"], 1)

    def test_indexed_folder_checks_padding_before_graphics(self):
        pages = model_pages()
        data, offsets = compressed_pages(pages)
        end = offsets[-2] + len(zlib.compress(pages[-2]))
        self.assertLess(end, offsets[-1])
        for replacement in (b"!", bytes(17)):
            with self.subTest(replacement=replacement):
                source, folder, _ = self.carve(data[:end] + replacement + data[end + 1:])
                for path in (source, folder):
                    scan = joiner.scan_dir if path == folder else joiner.scan_rpf
                    self.assertEqual(scan(str(path))[1]["joined_resources"], 0)

    def test_invalid_index_cannot_restore_file_order_join(self):
        _, folder, index = self.carve(compressed_pages(model_pages())[0])
        for change in ({"stored": 0}, {"off": 0}, {"pre": "00"}, {"size": 65}):
            with self.subTest(change=change):
                original = index["files"][1].copy()
                index["files"][1].update(change)
                (folder / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
                with self.assertRaises(ValueError):
                    joiner.scan_dir(folder)
                index["files"][1] = original

    def test_unrelated_short_streams_and_files_keep_the_cutoff(self):
        pages = model_pages()
        data, _ = fixture.pack_synthetic_rpf([[b"unrelated".ljust(48, b"\0")], pages, [b"tail" * 12]])
        _, folder, index = self.carve(data)
        self.assertEqual([row["size"] for row in index["files"]], [65536, 48, len(pages[-1])])
        (folder / "000000000001.bin").write_bytes(b"unrelated".ljust(48, b"\0"))
        (folder / "ffffffffffff.bin").write_bytes(b"tail" * 12)
        report = self.joined_cli(folder)
        self.assertEqual(report["summary"]["joined_resources"], 1)
        self.assertEqual(report["summary"]["zlib_streams"], 3)

    def test_compressed_bink_prefix_stays_a_model_page(self):
        for kind in ("ABC middle", "KB2 middle", "Bink header middle", "KB2 final 48", "KB2 final 64"):
            with self.subTest(kind=kind):
                final = kind.startswith("KB2 final")
                pages = model_pages(final_size=64 if kind.endswith("64") else 48, middle_pages=0 if final else 1)
                page = bytearray(pages[1])
                if kind == "Bink header middle":
                    video = fixture.make_bink2()
                    page[:len(video)] = video
                    struct.pack_into("<I", page, 4, len(page) - 8)
                else:
                    page[:3] = b"ABC" if kind == "ABC middle" else b"KB2"
                pages[1] = bytes(page)
                data, _ = compressed_pages(pages)
                source, folder, index = self.carve(data)
                self.assertEqual(index["summary"]["bink2_files"], 0)
                self.assertEqual([row["size"] for row in index["files"]], list(map(len, pages)))
                for row, expected in zip(index["files"], pages):
                    self.assertEqual((folder / f"{row['off']:012x}.bin").read_bytes(), expected)
                archive = self.joined_cli(source)
                for report in (archive, self.joined_cli(folder)):
                    self.assertEqual(report["summary"]["joined_resources"], 1)
                    self.assertEqual(report["summary"]["zlib_streams"], len(pages))
                    resource, = report["resources"]
                    self.assertEqual([row["size"] for row in resource["system_streams"]],
                                     list(map(len, pages[:-1])))
                    self.assertEqual(resource["primary_triangles"], 3)
                    self.assertEqual(resource["all_lod_triangles"], 4)

    def test_raw_bink_files_still_interrupt_short_page_runs(self):
        for video_size in (96, 65536):
            with self.subTest(video_size=video_size):
                pages = model_pages(middle_pages=1)
                video = bytearray(fixture.make_bink2().ljust(video_size, b"\0"))
                struct.pack_into("<I", video, 4, video_size - 8)
                pages[1] = bytes(video)
                # The fixture writes this middle page raw; the same bytes compressed above are a .bin.
                data, recorded = fixture.pack_synthetic_rpf([pages])
                source, folder, index = self.carve(data)
                self.assertEqual(index["summary"]["bink2_files"], 1)
                video_offset = recorded[0][0][1]
                self.assertEqual((folder / f"{video_offset:012x}.bk2").read_bytes(), video)
                self.assertEqual([row["size"] for row in index["files"]], [65536, video_size, 512])
                # Also check the folder's own stop rule when a bare short file follows the raw video.
                tail_offset = recorded[0][0][2]
                (folder / f"{tail_offset:012x}.bin").write_bytes(pages[2])
                for path in (source, folder):
                    output = self.root / "interrupted.json"
                    run = subprocess.run([sys.executable, "-I", str(ROOT / "tools/restofworld_model_join.py"),
                                          str(path), str(output), "--quiet"], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 1, run.stderr)
                    summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
                    self.assertEqual(summary["joined_resources"], 0)
                    self.assertEqual(summary["zlib_streams"], 2)

    def test_short_page_requires_valid_root_counts_and_full_nonfinal_pages(self):
        for kind in ("single page", "too many pages", "reserved map word", "short nonfinal", "empty final"):
            with self.subTest(kind=kind):
                pages = model_pages()
                first = bytearray(pages[0])
                if kind == "single page":
                    fixture._write_page_map(first, 65536 - 64, 1, 1)
                elif kind == "too many pages":
                    first[65536 - 64 + 8] = 33
                elif kind == "reserved map word":
                    first[65536 - 64] = 1
                elif kind == "short nonfinal":
                    fixture._write_page_map(first, 65536 - 64, 3, 1)
                    pages.insert(1, b"nonfinal".ljust(48, b"\0"))
                else:
                    pages[1] = b""
                pages[0] = bytes(first)
                data, _ = fixture.pack_synthetic_rpf([pages])
                source, folder, index = self.carve(data)
                self.assertTrue(all(row["size"] >= 64 for row in index["files"]))
                self.assertEqual(joiner.scan_rpf(str(source))[1]["joined_resources"], 0)
                self.assertEqual(joiner.scan_dir(str(folder))[1]["joined_resources"], 0)

    def test_archive_framing_breaks_do_not_admit_the_short_page(self):
        for kind in ("wide gap", "nonzero padding", "unaligned root", "nonzero root prefix", "bad checksum", "Bink"):
            with self.subTest(kind=kind):
                pages = model_pages()
                data, recorded = fixture.pack_synthetic_rpf([pages])
                offsets = recorded[0][0]
                data = bytearray(data)
                if kind == "wide gap":
                    data[offsets[1]:offsets[1]] = b"\0" * 16
                elif kind == "nonzero padding":
                    self.assertLess(offsets[0] + len(zlib.compress(pages[0])), offsets[1])
                    data[offsets[1] - 1] = 1
                elif kind == "unaligned root":
                    data[offsets[0]:offsets[0]] = b"\0" * 16
                elif kind == "nonzero root prefix":
                    data[offsets[0] - 1] = 1
                elif kind == "bad checksum":
                    data[offsets[1] + len(zlib.compress(pages[1])) - 1] ^= 1
                else:
                    data[offsets[1]:offsets[1]] = fixture.make_bink2()
                source, folder, index = self.carve(bytes(data))
                self.assertTrue(all(row["size"] >= 64 for row in index["files"]))
                self.assertEqual(joiner.scan_rpf(str(source))[1]["joined_resources"], 0)
                self.assertEqual(joiner.scan_dir(str(folder))[1]["joined_resources"], 0)

    def test_scan_windows_still_bound_the_complete_continuation(self):
        pages = model_pages()
        data, recorded = fixture.pack_synthetic_rpf([pages])
        offsets = recorded[0][0]
        end = offsets[1] + len(zlib.compress(pages[1]))
        for delta, expected in ((-1, [65536]), (0, [65536, 48])):
            with self.subTest(checksum_end_delta=delta):
                _, _, index = self.carve(data, "--limit", end + delta - offsets[0])
                self.assertEqual([row["size"] for row in index["files"]], expected)
        _, _, index = self.carve(data, "--start", offsets[1])
        self.assertEqual([row["size"] for row in index["files"]], [len(pages[-1])])

    def test_folder_mode_keeps_its_framing_policy_and_inflate_limit(self):
        folder = self.root / "bare"
        folder.mkdir()
        for i, page in enumerate(model_pages()):
            (folder / f"{i + 1:012x}.bin").write_bytes(page)
        resources, summary = joiner.scan_dir(str(folder))
        self.assertEqual(summary["joined_resources"], 1)
        self.assertEqual(resources[0]["system_bytes"], 65536 + 48)
        _, capped = joiner.scan_dir(str(folder), max_inflate=32768)
        self.assertEqual(capped["skipped_candidates"], 1)
        self.assertEqual(capped["zlib_streams"], 1)
        self.assertEqual(capped["joined_resources"], 0)
        # All read-only mappings must be closed before returning, including on Windows.
        os.replace(folder / "000000000001.bin", folder / "renamed.bin")


if __name__ == "__main__":
    unittest.main()
