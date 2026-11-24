"""Tests for tools/bink_names.py, built from synthetic Bink 2 headers and mount lists (no game data).

    python -I tests/test_bink_names.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "bink_names.py")
    spec = importlib.util.spec_from_file_location("bink_names_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


names_mod = load_module()


def make_stream(width, height, frames, body=64):
    """A minimal Bink 2 stream that bink_index.parse_header accepts."""
    extra = b"\x00\x05\xe9\x87"
    total = 0x2C + len(extra) + 4 * frames + body
    header = bytearray(0x2C)
    header[0:4] = b"KB2n"
    struct.pack_into("<10I", header, 4, total - 8, frames, body, frames,
                     width, height, 30, 1, 0x40013, 0)
    return bytes(header) + extra + bytes(4 * frames) + bytes(body)


def stream(offset, width=1280, height=720, frames=100, source="a.rpf"):
    return {"source": source, "offset": offset, "size": 0x100, "width": width, "height": height,
            "frames": frames}


def mount_xml(paths, ident="test_binks"):
    items = "".join(f"<Item><filename>$(common)/movies/{p}</filename></Item>" for p in paths)
    return (f"<?xml version=\"1.0\"?><CDataFileMgr__ContentChangeSetDef><id>{ident}</id>"
            f"<filesToEnable>{items}</filesToEnable></CDataFileMgr__ContentChangeSetDef>")


class TestListOrder(unittest.TestCase):
    def write(self, tmp, name, text):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_sorted_by_folder_then_name_without_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.write(tmp, "a.xml", mount_xml(["zz/b.bk2", "Aa/clip_x.bk2", "aa/clipe.bk2"]))
            b = self.write(tmp, "b.xml", mount_xml(["AA/CLIP_X.bk2", "mm/one.bk2"]))
            got = names_mod.read_mount_lists([a, b])
        # duplicates collapse case-insensitively; '_' (0x5F) sorts before 'e' (0x65)
        self.assertEqual(got, ["Aa/clip_x.bk2", "aa/clipe.bk2", "mm/one.bk2", "zz/b.bk2"])

    def test_name_without_extension_compared(self):
        # "ab" is a prefix of "ab-1", so it sorts first; with the extension '-' (0x2D) < '.' (0x2E) would flip them
        self.assertEqual(sorted(["f/ab-1.bk2", "f/ab.bk2"], key=names_mod.sort_key), ["f/ab.bk2", "f/ab-1.bk2"])

    def test_directory_is_searched(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "sub"))
            self.write(tmp, "sub/1.bin", mount_xml(["q/x.bk2"]))
            self.write(tmp, "2.bin", "<other>no movies here</other>")
            self.assertEqual(names_mod.read_mount_lists([tmp]), ["q/x.bk2"])

    def test_carve_list_split_across_streams(self):
        # rpf9_carve.py layout: index.json plus one inflated <offset>.bin per zlib stream
        doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               + mount_xml(["s/a.bk2", "s/split_here.bk2", "s/z.bk2"]).split("?>", 1)[1]).encode()
        cut = doc.index(b"split_") + 3
        parts = [doc[:cut], doc[cut:]]
        with tempfile.TemporaryDirectory() as tmp:
            files, off = [], 0x1000
            for part in parts:
                with open(os.path.join(tmp, f"{off:012x}.bin"), "wb") as fh:
                    fh.write(part)
                files.append({"off": off, "stored": len(part), "size": len(part)})
                off += 0x1000
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as fh:
                json.dump({"files": files, "summary": {}}, fh)
            self.assertEqual(names_mod.read_mount_lists([tmp]), ["s/a.bk2", "s/split_here.bk2", "s/z.bk2"])

    def test_other_filenames_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "c.xml", "<filename>$(common)/data/x.meta</filename>"
                                            "<filename>$(common)/movies/t/a.bk2</filename>")
            self.assertEqual(names_mod.read_mount_lists([path]), ["t/a.bk2"])


class TestAlign(unittest.TestCase):
    def run_names(self, names, streams, pins=None):
        rows, unmatched = names_mod.name_streams(names, streams, pins)
        return [r["path"] for r in rows], [r["confidence"] for r in rows], unmatched

    def test_equal_counts_zip(self):
        names = ["f/a.bk2", "f/b.bk2", "f/c.bk2"]
        paths, conf, unmatched = self.run_names(names, [stream(0), stream(1), stream(2)])
        self.assertEqual(paths, names)
        self.assertEqual(unmatched, [])
        self.assertNotIn("positional", conf)

    def test_height_hint_places_skip(self):
        names = ["f/a_720.bk2", "f/b_1080.bk2", "f/c_720.bk2"]
        streams = [stream(0, 1280, 720), stream(1, 1280, 720)]
        paths, conf, unmatched = self.run_names(names, streams)
        self.assertEqual(paths, ["f/a_720.bk2", "f/c_720.bk2"])
        self.assertEqual(unmatched, ["f/b_1080.bk2"])
        self.assertEqual(conf, ["anchored", "anchored"])

    def test_trailing_skip_limit_selects_best_permitted_match(self):
        names = ["f/a_720.bk2", "f/b_1080.bk2", "f/c_720.bk2"]
        rows, unmatched = names_mod.name_streams(names, [stream(0)], max_skip=1)
        self.assertEqual([row["path"] for row in rows], [names[1]])
        self.assertEqual(unmatched, [names[0], names[2]])

    def test_family_siblings_share_shape(self):
        # x_01/x_02 share a shape, y does not exist in the archive
        names = ["f/x_01.bk2", "f/x_02.bk2", "f/y.bk2", "f/z.bk2"]
        streams = [stream(0, 512, 512, 30), stream(1, 512, 512, 30), stream(2, 1920, 1080, 900)]
        paths, conf, _ = self.run_names(names, streams)
        self.assertEqual(paths[:2], ["f/x_01.bk2", "f/x_02.bk2"])
        self.assertEqual(conf[:2], ["anchored", "anchored"])

    def test_pins_verify_and_override(self):
        names = ["f/a.bk2", "f/b.bk2", "f/c.bk2"]
        streams = [stream(0), stream(1)]
        pins = {("", 1): {"f/b.bk2"}}
        paths, conf, unmatched = self.run_names(names, streams, pins)
        self.assertEqual(paths, ["f/a.bk2", "f/b.bk2"])
        self.assertEqual(conf[1], "verified")
        self.assertEqual(unmatched, ["f/c.bk2"])
        # an alternative-set pin anchors but does not verify
        paths, conf, _ = self.run_names(names, streams, {("", 1): {"f/b.bk2", "f/c.bk2"}})
        self.assertEqual(conf[1], "anchored")

    def test_bracketed_vs_positional(self):
        # two hinted anchors with a gap: equal counts between them -> bracketed
        names = ["f/a_1080.bk2", "f/m.bk2", "f/n.bk2", "f/z_1080.bk2"]
        streams = [stream(0, 1920, 1080, 1), stream(1, 640, 480, 2), stream(2, 800, 600, 3),
                   stream(3, 1920, 1080, 4)]
        _, conf, _ = self.run_names(names, streams)
        self.assertEqual(conf, ["anchored", "bracketed", "bracketed", "anchored"])
        # one extra unanchored name: the skip could be anywhere -> positional
        names.insert(2, "f/extra.bk2")
        _, conf, _ = self.run_names(names, streams)
        self.assertEqual(conf[1:3], ["positional", "positional"])

    def test_too_few_names(self):
        with self.assertRaises(ValueError):
            names_mod.name_streams(["f/a.bk2"], [stream(0), stream(1)])

    def test_no_streams(self):
        self.assertEqual(names_mod.name_streams(["f/a.bk2"], [], max_skip=0), ([], ["f/a.bk2"]))

    def test_family_strips_4k_suffix_for_sibling_anchoring(self):
        self.assertEqual(names_mod.family("f/clip_01_4k.bk2"), "f/clip")
        self.assertEqual(names_mod.family("f/clip_4k.bk2"), "f/clip")
        self.assertEqual(names_mod.family("f/clip_02_1080.bk2"), "f/clip")
        names = ["f/clip_01.bk2", "f/clip_02_4k.bk2", "f/extra.bk2"]
        streams = [stream(0, 3840, 2160, 30), stream(1, 3840, 2160, 30)]
        paths, conf, unmatched = self.run_names(names, streams)
        self.assertEqual(paths, ["f/clip_01.bk2", "f/clip_02_4k.bk2"])
        self.assertEqual(conf, ["anchored", "anchored"])
        self.assertEqual(unmatched, ["f/extra.bk2"])

    def test_contradicted_pairs_stay_positional(self):
        # A contradicted height hint must not anchor either itself or a neutral family sibling.
        paths, conf, _ = self.run_names(
            ["f/c_01_720.bk2", "f/c_02_1080.bk2"],
            [stream(0, 1280, 720, 30), stream(1, 1280, 720, 30)],
        )
        self.assertEqual(paths, ["f/c_01_720.bk2", "f/c_02_1080.bk2"])
        self.assertEqual(conf, ["anchored", "positional"])

        paths, conf, _ = self.run_names(
            ["f/c_01.bk2", "f/c_02_1080.bk2", "f/extra_1080.bk2"],
            [stream(0, 1280, 720, 30), stream(1, 1280, 720, 30)],
        )
        self.assertEqual(paths, ["f/c_01.bk2", "f/c_02_1080.bk2"])
        self.assertEqual(conf, ["positional", "positional"])

        # Contradicted height/orientation/tv hints must not become "bracketed" between valid anchors.
        for mid, h in (("f/m_1080.bk2", (1280, 720)),
                       ("snapmatic/s.bk2", (1280, 720)),
                       ("tv/sprt/s.bk2", (640, 480))):
            with self.subTest(mid=mid):
                paths, conf, _ = self.run_names(
                    ["f/a_1080.bk2", mid, "f/z_1080.bk2"],
                    [stream(0, 1920, 1080), stream(1, *h), stream(2, 1920, 1080)],
                )
                self.assertEqual(paths, ["f/a_1080.bk2", mid, "f/z_1080.bk2"])
                self.assertEqual(conf, ["anchored", "positional", "anchored"])

        # A contradicted pin must not become "anchored" (via family), anchor a neutral sibling, or become "bracketed".
        _, conf_fam, _ = self.run_names(
            ["f/x_1.bk2", "f/x_2.bk2"],
            [stream(0, 1280, 720, 10), stream(1, 1280, 720, 10)],
            pins={("", 1): {"f/other.bk2"}},
        )
        self.assertEqual(conf_fam, ["bracketed", "positional"])

        _, conf_fam_hinted, _ = self.run_names(
            ["f/x_1_720.bk2", "f/x_2.bk2"],
            [stream(0, 1280, 720, 10), stream(1, 1280, 720, 10)],
            pins={("", 1): {"f/other.bk2"}},
        )
        self.assertEqual(conf_fam_hinted, ["anchored", "positional"])

        _, conf_brk, _ = self.run_names(
            ["f/a_1080.bk2", "f/m.bk2", "f/z_1080.bk2"],
            [stream(0, 1920, 1080), stream(1, 640, 480), stream(2, 1920, 1080)],
            pins={("", 1): {"f/other.bk2"}},
        )
        self.assertEqual(conf_brk, ["anchored", "positional", "anchored"])

        # Positive pin still overrides a contradicting filename hint.
        _, conf_pin, _ = self.run_names(
            ["f/m_1080.bk2"],
            [stream(0, 1280, 720)],
            pins={("", 0): {"f/m_1080.bk2"}},
        )
        self.assertEqual(conf_pin, ["verified"])


class TestLoadPins(unittest.TestCase):
    def test_repeated_rows_union_and_sources_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pins.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["offset", "path", "source"])
                w.writeheader()
                # same offset across two rows, different sources: each source accumulates independently
                w.writerow({"offset": 0x10, "path": "f/a.bk2", "source": "a.rpf"})
                w.writerow({"offset": 0x10, "path": "F/B.BK2", "source": "a.rpf"})
                w.writerow({"offset": 0x10, "path": "f/c.bk2", "source": "b.rpf"})
                # empty-source fallback stays separate and accumulates too
                w.writerow({"offset": 0x20, "path": "g/x.bk2"})
                w.writerow({"offset": 0x20, "path": "g/y.bk2"})
            pins = names_mod.load_pins(csv_path)
        self.assertEqual(pins[("a.rpf", 0x10)], {"f/a.bk2", "f/b.bk2"})
        self.assertEqual(pins[("b.rpf", 0x10)], {"f/c.bk2"})
        self.assertEqual(pins[("", 0x20)], {"g/x.bk2", "g/y.bk2"})

    def test_single_pipe_row_and_split_rows_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "split.csv")
            pipe = os.path.join(tmp, "pipe.csv")
            for name, rows in (
                (split, [{"offset": 0x10, "path": "f/a.bk2"},
                         {"offset": 0x10, "path": "f/b.bk2"}]),
                (pipe, [{"offset": 0x10, "path": "f/a.bk2|f/b.bk2"}]),
            ):
                with open(name, "w", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=["offset", "path"])
                    w.writeheader()
                    w.writerows(rows)
            self.assertEqual(names_mod.load_pins(split), names_mod.load_pins(pipe))

    def test_normalizes_backslashes_and_movies_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pins.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write("offset,path\n0,$(common)/movies/F\\A.bk2|$(COMMON)\\MOVIES\\f\\b.bk2\n")
            pins = names_mod.load_pins(csv_path)
        self.assertEqual(pins[("", 0)], {"f/a.bk2", "f/b.bk2"})
        rows, unmatched = names_mod.name_streams(
            ["f/a.bk2", "f/b.bk2", "f/c.bk2"],
            [stream(0), stream(1)],
            pins={("", 0): pins[("", 0)] & {"f/a.bk2"}},
        )
        self.assertEqual(rows[0]["confidence"], "verified")
        self.assertEqual(unmatched, ["f/c.bk2"])

    def test_a_windows_source_from_before_slashes_still_matches(self):
        # bink_index wrote "movies\x.bk2" on Windows before it used "/"; older pins and index files keep working.
        with tempfile.TemporaryDirectory() as tmp:
            pins_csv = os.path.join(tmp, "pins.csv")
            index_csv = os.path.join(tmp, "index.csv")
            with open(pins_csv, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("offset,path,source\n0,f/a.bk2,movies\\x.bk2\n")
            with open(index_csv, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("source,offset,size,width,height,frames\nmovies\\x.bk2,0,64,16,16,1\n")
            pins = names_mod.load_pins(pins_csv)
            streams = names_mod.load_streams(index_csv)
        self.assertEqual(pins, {("movies/x.bk2", 0): {"f/a.bk2"}})
        self.assertEqual([s["source"] for s in streams], ["movies/x.bk2"])

    def test_rejects_empty_and_invalid_pin_rows(self):
        bad_rows = [
            "0,\n",
            "0,   \n",
            "0,| \n",
            "0,$(common)/movies/\n",
            "-1,f/a.bk2\n",
            "1.5,f/a.bk2\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pins.csv")
            for row in bad_rows:
                with self.subTest(row=row.strip()):
                    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                        fh.write("offset,path\n" + row)
                    with self.assertRaisesRegex(ValueError, "row 2"):
                        names_mod.load_pins(csv_path)


class TestCli(unittest.TestCase):
    def test_unreadable_mount_directory_preserves_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "empty.bk2")
            with open(source, "wb"):
                pass
            mount = os.path.join(tmp, "mount")
            os.mkdir(mount)
            blocked = os.path.join(mount, "blocked")
            report = os.path.join(tmp, "names.json")
            with open(report, "wb") as f:
                f.write(b"previous report")

            def walk(path, onerror=None):
                if onerror:
                    onerror(PermissionError(13, "permission denied", blocked))
                return iter(())

            with mock.patch.object(names_mod.os, "walk", walk), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(names_mod.main([source, "--mount", mount, "--out", report]), 1)
            self.assertIn(blocked, err.getvalue())
            with open(report, "rb") as f:
                self.assertEqual(f.read(), b"previous report")

    def test_rejects_negative_max_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "list.xml")
            with open(xml, "w", encoding="utf-8") as f:
                f.write(mount_xml([]))
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    names_mod.main([tmp, "--mount", xml, "--max-skip", "-1"])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--max-skip must be 0 or more", err.getvalue())

    def test_zero_max_skip_disallows_a_trailing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "test.rpf")
            with open(archive, "wb") as fh:
                fh.write(make_stream(1280, 720, 5))
            xml = os.path.join(tmp, "list.xml")
            with open(xml, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(mount_xml(["f/a.bk2", "f/b.bk2"]))
            report = os.path.join(tmp, "names.json")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = names_mod.main([archive, "--mount", xml, "--max-skip", "0", "--out", report])
            self.assertEqual(code, 1)
            self.assertIn("no alignment within --max-skip", err.getvalue())
            self.assertFalse(os.path.exists(report))

    def test_archive_end_to_end(self):
        shapes = [(1280, 720, 40), (1920, 1080, 50), (512, 512, 15)]
        with tempfile.TemporaryDirectory() as tmp:
            blob = bytearray()
            offsets = []
            for shape in shapes:
                blob += bytes((-len(blob)) % 0x1000)
                offsets.append(len(blob))
                blob += make_stream(*shape)
            archive = os.path.join(tmp, "test.rpf")
            with open(archive, "wb") as fh:
                fh.write(blob)
            xml = os.path.join(tmp, "list.xml")
            with open(xml, "w", encoding="utf-8") as fh:
                fh.write(mount_xml(["b/d.bk2", "b/c_1080.bk2", "b/b_720.bk2", "b/a_720.bk2"]))
            out_csv = os.path.join(tmp, "names.csv")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(names_mod.main([archive, "--mount", xml, "--out", out_csv]), 0)
            self.assertIn("4 list entries, 3 streams", log.getvalue())
            with open(out_csv, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([int(r["offset"]) for r in rows], offsets)
            # sorted list: a_720, b_720, c_1080, d; b_720 cannot sit on the 1080 stream, so it is the skip
            self.assertEqual([r["path"] for r in rows], ["b/a_720.bk2", "b/c_1080.bk2", "b/d.bk2"])
            self.assertEqual(rows[0]["source"], "test.rpf")

            # a bink_index JSON as the source gives the same names
            index = os.path.join(tmp, "index.json")
            rows_idx, summary = names_mod.bink_index.scan_target(archive)
            names_mod.bink_index.write_json(index, archive, rows_idx, summary, False)
            out_json = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(names_mod.main([index, "--mount", xml, "--out", out_json]), 0)
            with open(out_json, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual([r["path"] for r in doc["streams"]], [r["path"] for r in rows])
            self.assertEqual(doc["summary"]["streams"], 3)
            self.assertEqual(len(doc["unmatched_list_entries"]), 1)

    def test_missing_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(names_mod.main(["/nonexistent.rpf", "--mount", "/nonexistent.xml"]), 2)

    def test_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "list.xml")
            with open(xml, "w", encoding="utf-8") as fh:
                fh.write(mount_xml(["a/b.bk2"]))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                names_mod.main([tmp, "--mount", xml, "--out", xml])

    def test_refuses_to_overwrite_pins_input(self):
        # --out targeting the same --pins CSV must be rejected before the file is opened for writing
        with tempfile.TemporaryDirectory() as tmp:
            pins = os.path.join(tmp, "pins.csv")
            with open(pins, "w", encoding="utf-8", newline="") as fh:
                fh.write("offset,path\n0,f/a.bk2\n")
            xml = os.path.join(tmp, "list.xml")
            with open(xml, "w", encoding="utf-8") as fh:
                fh.write(mount_xml(["f/a.bk2", "f/b.bk2"]))
            archive = os.path.join(tmp, "test.rpf")
            with open(archive, "wb") as fh:
                fh.write(make_stream(1280, 720, 40))
            with open(pins, "rb") as fh:
                before = fh.read()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                names_mod.main([archive, "--mount", xml, "--pins", pins, "--out", pins])
            with open(pins, "rb") as fh:
                self.assertEqual(fh.read(), before, "the pins CSV must not be modified")

    def test_refuses_to_overwrite_input_inside_source_or_mount_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = os.path.join(tmp, "streams")
            mount_dir = os.path.join(tmp, "mounts")
            os.mkdir(src_dir)
            os.mkdir(mount_dir)
            stream_file = os.path.join(src_dir, "one.bk2")
            stream_bytes = make_stream(1280, 720, 40)
            with open(stream_file, "wb") as fh:
                fh.write(stream_bytes)
            xml_text = mount_xml(["f/a.bk2"])
            xml_file = os.path.join(mount_dir, "list.xml")
            with open(xml_file, "w", encoding="utf-8") as fh:
                fh.write(xml_text)
            for folder in (src_dir, mount_dir):
                reserved = os.path.join(folder, "index.json")
                with self.subTest(folder=folder), contextlib.redirect_stderr(io.StringIO()) as err:
                    with self.assertRaises(SystemExit) as cm:
                        names_mod.main([src_dir, "--mount", mount_dir, "--out", reserved])
                    self.assertEqual(cm.exception.code, 2)
                    self.assertIn("carve manifest", err.getvalue())
                    self.assertFalse(os.path.exists(reserved))
            for target in (stream_file, xml_file, src_dir):
                with self.subTest(target=target):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                        names_mod.main([src_dir, "--mount", mount_dir, "--out", target])
                    self.assertEqual(cm.exception.code, 2)
            with open(stream_file, "rb") as fh:
                self.assertEqual(fh.read(), stream_bytes)
            with open(xml_file, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), xml_text)


class TestReportErrors(unittest.TestCase):
    """Bad bink_index.csv/.json sources fail with a one-line, path-specific error and no report."""

    def write(self, tmp, name, text):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def run_cli(self, source, tmp):
        xml = self.write(tmp, "list.xml", mount_xml(["a/b.bk2"]))
        out = os.path.join(tmp, "names.csv")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = names_mod.main([source, "--mount", xml, "--out", out])
        return code, err.getvalue(), out

    def test_bad_sources_are_rejected(self):
        # (extension, contents, expected error fragment or None to only require the path)
        cases = [
            ("index.json", '{"summary": {}, "target": "x.rpf"}', "streams"),
            ("index.json", '[]', "streams"),
            ("index.json", '{"streams": [null]}', "stream 1"),
            ("index.json", '{"streams": [{"source": "a", "offset": 0}]}', "stream 1"),
            ("index.json", '{"streams": [', None),
            ("index.csv", "source,offset,size,width,height\n", "frames"),
            ("index.csv", "source,offset,size,width,height,frames\n"
                          "a.rpf,0,0x100,1280,720,100\n"
                          "a.rpf,bogus,0x100,1280,720,100\n", "row 3"),
            ("index.csv", "", "frames"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, text, needle in cases:
                with self.subTest(name=name, text=text):
                    source = self.write(tmp, name, text)
                    code, err, out = self.run_cli(source, tmp)
                    self.assertEqual(code, 1)
                    self.assertIn(source, err)
                    if needle is not None:
                        self.assertIn(needle, err)
                    self.assertFalse(os.path.exists(out))

    def test_bad_pins_are_rejected(self):
        cases = [
            ("pins.csv", "", "offset, path"),
            ("pins.csv", "offset,source\n0,a.rpf\n", "path"),
            ("pins.csv", "offset,path\nbogus,f/a.bk2\n", "row 2"),
            ("pins.csv", "offset,path\n0\n", "row 2"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write(tmp, "index.json", '{"streams": []}')
            xml = self.write(tmp, "list.xml", mount_xml(["a/b.bk2"]))
            for name, text, needle in cases:
                with self.subTest(text=text):
                    pins = self.write(tmp, name, text)
                    out = os.path.join(tmp, "names.csv")
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        code = names_mod.main([source, "--mount", xml, "--pins", pins, "--out", out])
                    self.assertEqual(code, 1)
                    self.assertIn(pins, err.getvalue())
                    self.assertIn(needle, err.getvalue())
                    self.assertFalse(os.path.exists(out))
            # A directory passed as --pins must also report a clean error rather than an uncaught OSError.
            pins_dir = os.path.join(tmp, "pins_dir")
            os.mkdir(pins_dir)
            out = os.path.join(tmp, "names.csv")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = names_mod.main([source, "--mount", xml, "--pins", pins_dir, "--out", out])
            self.assertEqual(code, 1)
            self.assertIn(pins_dir, err.getvalue())
            self.assertFalse(os.path.exists(out))

    def test_empty_sources_are_accepted(self):
        cases = [
            ("index.csv", "source,offset,size,width,height,frames\n"),
            ("index.json", '{"streams": []}'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in cases:
                with self.subTest(name=name):
                    source = self.write(tmp, name, text)
                    xml = self.write(tmp, "list.xml", mount_xml(["a/b.bk2"]))
                    out = os.path.join(tmp, "names.csv")
                    log = io.StringIO()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(io.StringIO()):
                        code = names_mod.main([source, "--mount", xml, "--out", out])
                    self.assertEqual(code, 0)
                    self.assertIn("1 list entries, 0 streams", log.getvalue())

    def test_empty_pin_path_fails_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write(tmp, "index.json", '{"streams": []}')
            xml = self.write(tmp, "list.xml", mount_xml(["a/b.bk2"]))
            pins = self.write(tmp, "pins.csv", "offset,path\n0, | \n")
            out = os.path.join(tmp, "names.csv")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = names_mod.main([source, "--mount", xml, "--pins", pins, "--out", out])
            self.assertEqual(code, 1)
            self.assertIn(pins, err.getvalue())
            self.assertIn("row 2", err.getvalue())
            self.assertFalse(os.path.exists(out))

    def test_rejects_invalid_stream_numbers_and_keeps_zero_offset(self):
        base = {"source": "a.rpf", "offset": 0, "size": 1, "width": 1, "height": 1, "frames": 1}
        with tempfile.TemporaryDirectory() as tmp:
            good_json = self.write(tmp, "good.json", json.dumps({"streams": [base]}))
            self.assertEqual(names_mod.load_streams(good_json), [base])

            bad_mutations = [
                ("offset", True),
                ("offset", False),
                ("offset", -1),
                ("size", 0),
                ("width", 0),
                ("height", -1),
                ("frames", 1.5),
                ("frames", 0),
            ]
            for field, val in bad_mutations:
                row = dict(base)
                row[field] = val
                with self.subTest(field=field, val=val):
                    bad_json = self.write(tmp, "index.json", json.dumps({"streams": [row]}))
                    code, err, out = self.run_cli(bad_json, tmp)
                    self.assertEqual(code, 1)
                    self.assertIn(bad_json, err)
                    self.assertIn("stream 1", err)
                    self.assertFalse(os.path.exists(out))

            for csv_row in (
                "a.rpf,-1,256,1280,720,100\n",
                "a.rpf,0,0,1280,720,100\n",
                "a.rpf,0,256,0,720,100\n",
                "a.rpf,0,256,1280,720,1.5\n",
            ):
                with self.subTest(csv_row=csv_row.strip()):
                    bad_csv = self.write(
                        tmp,
                        "index.csv",
                        "source,offset,size,width,height,frames\n" + csv_row,
                    )
                    code, err, out = self.run_cli(bad_csv, tmp)
                    self.assertEqual(code, 1)
                    self.assertIn(bad_csv, err)
                    self.assertIn("row 2", err)
                    self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
