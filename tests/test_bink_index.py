"""Tests for tools/bink_index.py, built from synthetic Bink 2 headers (no game data).

    python -I tests/test_bink_index.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "bink_index.py")
    spec = importlib.util.spec_from_file_location("bink_index_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bink = load_module()


def make_stream(revision="n", frames=4, width=320, height=240, audio=0, body=256,
                fps_num=30, fps_den=1):
    """A plausible Bink 2 stream: fixed header, audio metadata, frame table, payload."""
    extra = b"\x00\x05\xe9\x87" if revision in "ijkn" else b""
    audio_blob = b""
    if audio:
        audio_blob += struct.pack("<I", 0x23A00) * audio       # max decoded sizes
        for _ in range(audio):
            audio_blob += struct.pack("<HH", 48000, 0x7000)   # rate, flags
        for track in range(audio):
            audio_blob += struct.pack("<I", track)            # track IDs
    total = 0x2C + len(extra) + len(audio_blob) + 4 * frames + body
    header = bytearray(0x2C)
    header[0:4] = b"KB2" + revision.encode()
    struct.pack_into("<10I", header, 4, total - 8, frames, body, frames,
                     width, height, fps_num, fps_den, 0x40013, audio)
    return bytes(header) + extra + audio_blob + bytes(4 * frames) + bytes(body)


class TestHeader(unittest.TestCase):
    def test_fields(self):
        row = bink.parse_header(make_stream(), 0)
        self.assertEqual(row["revision"], "n")
        self.assertEqual((row["width"], row["height"]), (320, 240))
        self.assertEqual(row["frames"], 4)
        self.assertEqual((row["fps_num"], row["fps_den"], row["fps"]), (30, 1, 30.0))
        self.assertEqual(row["size"], len(make_stream()))
        self.assertEqual(row["audio_tracks"], 0)

    def test_revision_j_and_audio_descriptors(self):
        row = bink.parse_header(make_stream("j", audio=2), 0)
        self.assertEqual(row["audio_tracks"], 2)
        self.assertEqual(row["audio_rates"], [48000, 48000])
        self.assertEqual(row["audio_flags"], ["0x7000", "0x7000"])
        self.assertEqual(row["audio_codecs"], ["binkaudio_dct", "binkaudio_dct"])
        self.assertEqual(row["audio_channels"], [2, 2])

    def test_audio_descriptors_stay_within_declared_size(self):
        stream = struct.pack("<4s10I", b"KB2a", 40, 1, 0, 1, 16, 16, 30, 1, 0, 1) + bytes(4)
        prefix = b"archive prefix"
        data = prefix + stream
        row = bink.parse_header(data, len(prefix))
        self.assertEqual(row["size"], 48)
        self.assertEqual(row["audio_rates"], [])
        self.assertEqual(row["audio_channels"], [])
        self.assertEqual(bink.parse_header(data + struct.pack("<HHI", 44100, 0x2000, 17),
                                           len(prefix)), row)

    def test_truncated_file_is_skipped(self):
        data = make_stream()[:-100]
        self.assertIsNone(bink.parse_header(data, 0))

    def test_bad_fields_are_rejected(self):
        bad_fps = bytearray(make_stream())
        struct.pack_into("<I", bad_fps, 0x1C, 0)
        self.assertIsNone(bink.parse_header(bad_fps, 0))
        bad_width = bytearray(make_stream())
        struct.pack_into("<I", bad_width, 0x14, 0)
        self.assertIsNone(bink.parse_header(bad_width, 0))
        bad_audio = bytearray(make_stream())
        struct.pack_into("<I", bad_audio, 0x28, 999)
        self.assertIsNone(bink.parse_header(bad_audio, 0))

    def test_frame_rate_limits(self):
        for num, den in [(240, 1), (0xFFFFFFFF, 0xFFFFFFFF)]:
            with self.subTest(num=num, den=den):
                self.assertIsNotNone(bink.parse_header(make_stream(fps_num=num, fps_den=den), 0))
        for num, den in [(0, 1), (1, 0), (0, 0), (1, 2), (240001, 1000)]:
            with self.subTest(num=num, den=den):
                self.assertIsNone(bink.parse_header(make_stream(fps_num=num, fps_den=den), 0))


class TestScanAndWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, "one.bk2"), "wb") as fh:
            fh.write(make_stream(frames=7))
        with open(os.path.join(self.dir, "truncated.bk2"), "wb") as fh:
            fh.write(make_stream()[:-64])
        with open(os.path.join(self.dir, "junk.bin"), "wb") as fh:
            fh.write(b"not a movie" * 64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_scan_skips_bad_files(self):
        rows, summary = bink.scan_target(self.dir)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "one.bk2")
        self.assertEqual(rows[0]["frames"], 7)
        self.assertEqual(summary["streams"], 1)

    def test_nested_sources_use_slashes_and_sort_alike_everywhere(self):
        # "movies/x.bk2" sorts before "movies0.bk2" ("/" < "0"); with "\" it would sort after.
        os.mkdir(os.path.join(self.dir, "movies"))
        for name, frames in (("movies0.bk2", 3), (os.path.join("movies", "x.bk2"), 5)):
            with open(os.path.join(self.dir, name), "wb") as fh:
                fh.write(make_stream(frames=frames))
        rows, _summary = bink.scan_target(self.dir)
        self.assertEqual([(r["source"], r["frames"]) for r in rows],
                         [("movies/x.bk2", 5), ("movies0.bk2", 3), ("one.bk2", 7)])

    def test_short_file_bytes_are_accounted_for(self):
        path = os.path.join(self.dir, "tiny.bk2")
        with open(path, "wb") as fh:
            fh.write(b"KB2a" + bytes(16))
        rows, summary = bink.scan_target(path)
        self.assertEqual(rows, [])
        self.assertEqual(summary["span"], 20)
        self.assertEqual(summary["unread_bytes"], 20)
        self.assertEqual(summary["zero_padding"], 0)

    def rpf(self, trailer_size, trailer, magic=b"9FPR"):
        """A synthetic RPF9: header (data start 0x40, +12 = trailer_size), one stream, 100 zeros, the trailer."""
        header = magic + struct.pack("<III", 0x40, 0x30, trailer_size) + bytes(0x30)
        path = os.path.join(self.dir, "synthetic.rpf")
        with open(path, "wb") as fh:
            fh.write(header + make_stream(frames=3) + bytes(100) + trailer)
        return path

    def test_rpf_trailer_after_the_last_stream(self):
        rows, summary = bink.scan_target(self.rpf(48, bytes(range(1, 49))))
        self.assertEqual(len(rows), 1)
        self.assertEqual((summary["zero_padding"], summary["rpf_trailer"], summary["unread_bytes"]), (100, 48, 0))

    def test_rpf_trailer_longer_than_the_gap_is_not_split_off(self):
        rows, summary = bink.scan_target(self.rpf(4096, bytes(range(1, 49))))
        self.assertEqual(len(rows), 1)
        self.assertEqual((summary["zero_padding"], summary["rpf_trailer"], summary["unread_bytes"]), (0, 0, 148))

    def test_only_an_rpf9_header_declares_a_trailer(self):
        # An RPF8-style or other "?FPR" header still gives the data start, but its +12 is not a measured trailer.
        rows, summary = bink.scan_target(self.rpf(48, bytes(range(1, 49)), magic=b"8FPR"))
        self.assertEqual(len(rows), 1)
        self.assertEqual((summary["zero_padding"], summary["rpf_trailer"], summary["unread_bytes"]), (0, 0, 148))

    def test_a_trailer_size_that_is_not_a_multiple_of_16_is_not_split_off(self):
        rows, summary = bink.scan_target(self.rpf(47, bytes(range(1, 48))))
        self.assertEqual(len(rows), 1)
        self.assertEqual((summary["zero_padding"], summary["rpf_trailer"], summary["unread_bytes"]), (0, 0, 147))

    def test_fractional_rates_are_indexed(self):
        path = os.path.join(self.dir, "fractional.bk2")
        with open(path, "wb") as fh:
            for num in (24000, 30000, 60000):
                fh.write(make_stream(frames=120, fps_num=num, fps_den=1001))
        rows, summary = bink.scan_target(path)
        self.assertEqual(summary["streams"], 3)
        self.assertEqual(
            [(r["fps_num"], r["fps_den"], r["fps"], r["duration_s"]) for r in rows],
            [(24000, 1001, 23.976, 5.005), (30000, 1001, 29.97, 4.004),
             (60000, 1001, 59.94, 2.002)],
        )

    def test_probe_handles_failed_or_malformed_results(self):
        rows, _ = bink.scan_target(self.dir)
        valid_info = {"streams": [{"codec_type": "video", "codec_name": "bink2"}],
                      "format": {"duration": "0.233"}}
        for status, info, expected in (
            (0, valid_info, ("yes", "bink2", "0.233")),
            (1, valid_info, ("no", "", "")),
            (0, [], ("no", "", "")),
            (0, {"streams": [None]}, ("no", "", "")),
            (0, {"streams": valid_info["streams"], "format": None}, ("no", "", "")),
        ):
            with self.subTest(status=status, info=info):
                proc = subprocess.CompletedProcess(["ffprobe"], status, json.dumps(info))
                with mock.patch.object(bink.subprocess, "run", return_value=proc):
                    row = bink.probe_row(dict(rows[0]), "ffprobe")
                self.assertEqual((row["decodes"], row["codec"], row["probe_duration_s"]), expected)

    def test_csv_and_json(self):
        rows, summary = bink.scan_target(self.dir)
        csv_path = os.path.join(self.dir, "index.csv")
        json_path = os.path.join(self.dir, "index.json")
        bink.write_csv(csv_path, rows, probe=False)
        bink.write_json(json_path, self.dir, rows, summary, probe=False)
        with open(csv_path, encoding="utf-8", newline="") as fh:
            got = list(csv.DictReader(fh))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["revision"], "n")
        self.assertNotIn("decodes", got[0])
        with open(json_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["summary"]["streams"], 1)
        self.assertEqual(doc["streams"][0]["width"], 320)
        self.assertNotIn("decodes", doc["streams"][0])


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "one.bk2")
        with open(self.path, "wb") as fh:
            fh.write(make_stream(frames=7))

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_on_stdout_has_one_line_end_per_row(self):
        # A text stream that writes "\n" as "\r\n", as Windows' stdout does, on any platform.
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding="utf-8", newline="\r\n")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bink.main([self.path]), 0)
        out.flush()
        data = raw.getvalue()
        self.assertNotIn(b"\r\r\n", data)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
        self.assertEqual(len(rows), 2)   # header and the one stream, no blank rows
        self.assertEqual(rows[0][:len(bink.BASE_FIELDS)], bink.BASE_FIELDS)

    def test_missing_target_is_an_error_not_a_traceback(self):
        missing = os.path.join(self.tmp.name, "nonexistent.bk2")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([missing]), 2)
        self.assertIn("no such file or directory", err.getvalue())

    def test_a_named_file_that_is_not_a_stream_is_an_error(self):
        # The scan only looks for KB2 headers, so a wrong path used to be a successful scan of 0 streams.
        junk = os.path.join(self.tmp.name, "wrong-path.bin")
        with open(junk, "wb") as fh:
            fh.write(b"not a movie" * 64)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([junk]), 2)
        self.assertEqual(err.getvalue().splitlines(),
                         [f"bink_index: error: {junk}: not a Bink 2 stream (KB2) or an RPF archive"])

    def test_an_empty_named_file_is_an_error(self):
        empty = os.path.join(self.tmp.name, "empty.bk2")
        with open(empty, "wb"):
            pass
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([empty]), 2)
        self.assertEqual(err.getvalue().splitlines(), [f"bink_index: error: {empty}: empty file"])

    def test_a_named_rpf_without_a_stream_still_scans(self):
        # An archive that holds no movie is a real answer, not a wrong path.
        archive = os.path.join(self.tmp.name, "empty-movies.rpf")
        with open(archive, "wb") as fh:
            fh.write(b"9FPR" + struct.pack("<III", 0x40, 0, 0) + bytes(0x34))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([archive]), 0)
        self.assertIn("0 Bink 2 streams", err.getvalue())

    def test_unreadable_directory_entry_reports_the_file_without_writing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "inputs")
            os.mkdir(target)
            source = os.path.join(target, "movie.bk2")
            with open(source, "wb") as fh:
                fh.write(make_stream())
            out = os.path.join(tmp, "report.json")
            stderr = io.StringIO()
            error = PermissionError(13, "input is locked", source)
            with (mock.patch.object(bink, "open", create=True, side_effect=error),
                  contextlib.redirect_stderr(stderr)):
                self.assertEqual(bink.main([target, "--out", out]), 2)
            self.assertEqual(stderr.getvalue().splitlines(),
                             [f"bink_index: error: cannot read {source}: input is locked"])
            self.assertFalse(os.path.exists(out))

    def test_incomplete_directory_walk_does_not_replace_the_report(self):
        target = self.tmp.name
        blocked = os.path.join(target, "blocked")
        out = os.path.join(target, "report.json")
        with open(out, "wb") as f:
            f.write(b"old report")

        def walk(root, onerror=None):
            yield root, ["blocked"], ["one.bk2"]
            if onerror:
                onerror(PermissionError(13, "directory is locked", blocked))

        with mock.patch.object(bink.os, "walk", walk), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(bink.main([target, "--out", out]), 2)
        self.assertIn(f"cannot read {blocked}: directory is locked", err.getvalue())
        with open(out, "rb") as f:
            self.assertEqual(f.read(), b"old report")

    def test_output_write_failure_is_an_error(self):
        for extension in ("csv", "json"):
            with self.subTest(extension=extension):
                path = os.path.join(self.tmp.name, "missing", f"index.{extension}")
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(bink.main([self.path, "--out", path]), 2)
                self.assertEqual(out.getvalue(), "")
                self.assertIn(f"cannot write {path}", err.getvalue())


    def test_output_cannot_create_a_report_in_place_of_the_carve_manifest(self):
        # <target>/index.json is the carve manifest's name even before a carve has written it.
        carve = os.path.join(self.tmp.name, "carve_no_manifest")
        os.makedirs(carve, exist_ok=True)
        manifest = os.path.join(carve, "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bink.main([carve, "--out", manifest])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("would overwrite the carve manifest", err.getvalue())
        self.assertFalse(os.path.exists(manifest))

    def test_output_cannot_overwrite_carve_manifest(self):
        carve = os.path.join(self.tmp.name, "carve")
        os.makedirs(carve, exist_ok=True)
        manifest = os.path.join(carve, "index.json")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write('{"carve_manifest": true}')
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bink.main([carve, "--out", manifest])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("would overwrite the carve manifest", err.getvalue())
        with open(manifest, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"carve_manifest": true}')

    def test_output_cannot_overwrite_a_carved_input(self):
        carve = os.path.join(self.tmp.name, "carve2")
        os.makedirs(carve, exist_ok=True)
        payload = os.path.join(carve, "00000001.bin")
        with open(payload, "wb") as f:
            f.write(make_stream())
        linked = os.path.join(carve, "linked.json")   # a .json alias of the .bin input
        try:
            os.link(payload, linked)
        except OSError:
            self.skipTest("hard links are not available")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bink.main([carve, "--out", linked])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("would overwrite an input file", err.getvalue())
        with open(payload, "rb") as f:
            self.assertEqual(f.read(), make_stream())

    def test_output_folder_is_rejected(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bink.main([self.path, "--out", self.tmp.name])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("is a folder, not a report file", err.getvalue())

    def test_repeat_run_produces_identical_summary(self):
        # A directory with one stream and a chosen report.json: the second run
        # must not inflate files_scanned/span/unread bytes by scanning the report.
        work = os.path.join(self.tmp.name, "movies")
        os.makedirs(work)
        with open(os.path.join(work, "one.bk2"), "wb") as fh:
            fh.write(make_stream(frames=11))
        report = os.path.join(work, "report.json")
        summaries = []
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(bink.main([work, "--out", report, "--format", "json"]), 0)
            with open(report, encoding="utf-8") as fh:
                summaries.append(json.load(fh)["summary"])
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["files_scanned"], 1)
        self.assertEqual(summaries[0]["streams"], 1)

    def test_a_single_named_file_that_is_not_a_stream_or_rpf_is_refused(self):
        path = os.path.join(self.tmp.name, "junk.bin")
        with open(path, "wb") as fh:
            fh.write(b"not a movie" * 64)
        out = os.path.join(self.tmp.name, "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([path, "--out", out]), 2)
        self.assertIn("not a Bink 2 stream or an RPF archive", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_a_scanned_directory_with_no_stream_still_reports_empty(self):
        # The refusal is for one named file: an empty folder scan is a valid "0 streams" answer.
        folder = os.path.join(self.tmp.name, "movies")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "notes.bin"), "wb") as fh:
            fh.write(b"not a movie" * 64)
        err = io.StringIO()
        out = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            self.assertEqual(bink.main([folder]), 0)
        self.assertIn("0 Bink 2 streams", err.getvalue())
        self.assertTrue(out.getvalue().splitlines())

    def test_an_rpf_archive_with_no_stream_is_still_indexed(self):
        # A named file with an RPF header is an archive, even if it holds no Bink 2 stream.
        path = os.path.join(self.tmp.name, "empty.rpf")
        with open(path, "wb") as fh:
            fh.write(b"9FPR" + struct.pack("<III", 0x40, 0x30, 0) + bytes(0x30))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bink.main([path]), 0)
        self.assertIn("0 Bink 2 streams", err.getvalue())
        self.assertNotIn("not a Bink 2 stream", err.getvalue())


if __name__ == "__main__":
    unittest.main()
