"""Synthetic tests for tools/opus_tags.py (no game data: invented banks, names and subtitles)."""
import codecs
import contextlib
import io
import csv
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "opus_tags.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import awc_opus  # noqa: E402
import opus_tags as T  # noqa: E402

PACKET = bytes([0xF8]) + b"\x55" * 40          # CELT-only 20 ms mono frame code 0: 960 samples


def make_ogg(path, packets=200, stream_id=0x0ABCDEF1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    awc_opus.write_ogg_opus(path, [PACKET] * packets, 1, stream_id, packets * 960 - 100)


def read_tags(data):
    """(vendor, [comments]) of the OpusTags packet on page 1."""
    pages = list(T.pages(data))
    at, _ = pages[1]
    body = at + 27 + data[at + 26]
    n = struct.unpack_from("<I", data, body + 8)[0]
    vendor = data[body + 12:body + 12 + n]
    at = body + 12 + n
    count = struct.unpack_from("<I", data, at)[0]
    at += 4
    out = []
    for _ in range(count):
        k = struct.unpack_from("<I", data, at)[0]
        out.append(data[at + 4:at + 4 + k].decode("utf-8"))
        at += 4 + k
    return vendor, out


def crc_ok(data):
    for a, e in T.pages(data):
        page = bytearray(data[a:e])
        want = struct.unpack_from("<I", page, 22)[0]
        page[22:26] = b"\0\0\0\0"
        if awc_opus.ogg_crc(bytes(page)) != want:
            return False
    return True


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


NAMES = ["source", "bank", "stream_id", "name", "rule"]
DIALOGUE = ["wave", "key", "label", "speaker", "text_en", "text_de"]


class TestPacket(unittest.TestCase):
    def test_tags_packet_layout_and_empty_values(self):
        p = T.tags_packet([("TITLE", "wave_a"), ("SPEAKER", ""), ("SUBTITLE", "Grüß dich")])
        self.assertTrue(p.startswith(b"OpusTags" + struct.pack("<I", 8) + T.VENDOR))
        self.assertEqual(struct.unpack_from("<I", p, 20)[0], 2)
        self.assertIn("SUBTITLE=Grüß dich".encode(), p)

    def test_markup_is_removed(self):
        self.assertEqual(T.clean("~sl:0.0:1.5~Hold ~COLOR_RED~still~s~,~n~friend.  "), "Hold still , friend.")


class TestRetag(unittest.TestCase):
    def test_only_the_tags_page_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b", "0ABCDEF1.ogg")
            make_ogg(path)
            with open(path, "rb") as f:
                old = f.read()
        new = T.retag(old, T.tags_packet([("TITLE", "x" * 3000)]))
        old_pages, new_pages = list(T.pages(old)), list(T.pages(new))
        self.assertEqual(len(old_pages), len(new_pages))
        self.assertEqual(old[:old_pages[1][0]], new[:new_pages[1][0]])
        self.assertEqual(old[old_pages[2][0]:], new[new_pages[2][0]:])
        self.assertTrue(crc_ok(new))
        self.assertEqual(read_tags(new), (T.VENDOR, ["TITLE=" + "x" * 3000]))
        self.assertEqual(T.retag(new, T.tags_packet([("TITLE", "x" * 3000)])), new)

    def test_retag_rejects_bad_crc_on_header_tags_and_audio_pages(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bank", "00000001.ogg")
            make_ogg(path, packets=3)
            original = Path(path).read_bytes()
        for page_index, (start, end) in enumerate(T.pages(original)):
            for corrupt_at in (start + 22, end - 1):
                with self.subTest(page=page_index, corrupt_at=corrupt_at):
                    damaged = bytearray(original)
                    damaged[corrupt_at] ^= 1
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        T.retag(bytes(damaged), T.tags_packet([("TITLE", "new")]))

    def test_retag_rejects_truncated_audio_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bank", "00000001.ogg")
            make_ogg(path, packets=3)
            original = Path(path).read_bytes()
        with self.assertRaisesRegex(ValueError, "end of the file"):
            T.retag(original[:-1], T.tags_packet([]))

    def test_rejects_what_it_cannot_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b", "0ABCDEF1.ogg")
            make_ogg(path)
            with open(path, "rb") as f:
                data = f.read()
        with self.assertRaisesRegex(ValueError, "one page"):
            T.retag(data, b"OpusTags" + b"\0" * T.MAX_PACKET)
        with self.assertRaisesRegex(ValueError, "Ogg page"):
            T.retag(b"RIFF" + data[4:], T.tags_packet([]))
        first = next(T.pages(data))[1]
        with self.assertRaisesRegex(ValueError, "two Ogg pages"):
            T.retag(data[:first], T.tags_packet([]))


class TestFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.opus = os.path.join(d, "opus")
        make_ogg(os.path.join(self.opus, "story", "bank_01", "0ABCDEF1.ogg"))
        make_ogg(os.path.join(self.opus, "story", "bank_01", "00000123.ogg"), stream_id=0x123)
        make_ogg(os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg"), stream_id=0x1FFFFFFF)
        self.names = os.path.join(d, "names.csv")
        write_csv(self.names, NAMES, [
            {"source": "story", "bank": "bank_01", "stream_id": "0abcdef1", "name": "heistau_abcd_01", "rule": "dialogue"},
            {"source": "story", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
            {"source": "other", "bank": "bank_02", "stream_id": "1fffffff", "name": "wrong_name", "rule": "family"},
        ])
        self.dialogue = os.path.join(d, "dialogue.csv")
        write_csv(self.dialogue, DIALOGUE, [{"wave": "heistau_abcd_01", "key": "HEISTAU_ABCD", "label": "HEIST_ZED_1",
                                             "speaker": "ZED", "text_en": "~z~Move.~n~Now.", "text_de": "Los."}])

    def tearDown(self):
        self.tmp.cleanup()

    def tags(self, *parts):
        with open(os.path.join(self.opus, *parts), "rb") as f:
            return read_tags(f.read())[1]

    def test_preserves_file_with_an_extra_packet_on_the_tags_page(self) -> None:
        path = os.path.join(self.opus, "story", "bank_01", "0ABCDEF1.ogg")
        with open(path, "rb") as f:
            data = f.read()
        tags_at, tags_end = list(T.pages(data))[1]
        serial, = struct.unpack_from("<I", data, tags_at + 14)
        mixed_page = awc_opus.ogg_page(serial, 1, 960, [T.tags_packet([]), PACKET], 0)
        original = data[:tags_at] + mixed_page + data[tags_end:]
        with open(path, "wb") as f:
            f.write(original)

        names, ambiguous = T.load_names(self.names, "story")
        stats = T.tag_folder(self.opus, names, ambiguous=ambiguous)

        with open(path, "rb") as f:
            self.assertTrue(f.read() == original, "tagging changed a file whose tags page includes audio")
        self.assertEqual(stats["skipped: the OpusTags page contains another packet"], 1)

    def test_corrupt_stream_is_skipped_unchanged_in_normal_and_dry_run(self):
        path = Path(self.opus) / "story" / "bank_01" / "0ABCDEF1.ogg"
        damaged = bytearray(path.read_bytes())
        start, _ = list(T.pages(damaged))[1]
        damaged[start + 22] ^= 1
        path.write_bytes(damaged)
        names, ambiguous = T.load_names(self.names, "story")
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                stats = T.tag_folder(self.opus, names, dry_run=dry_run, ambiguous=ambiguous)
                self.assertEqual(stats.get("skipped: Ogg checksum mismatch"), 1)
                self.assertEqual(stats.get("tagged"), 1)  # the other named stream still works
                self.assertEqual(path.read_bytes(), damaged)

    def test_names_and_dialogue(self):
        names, ambiguous = T.load_names(self.names, "story")
        dialogue, langs = T.load_dialogue(self.dialogue)
        stats = T.tag_folder(self.opus, names, dialogue, langs, ambiguous=ambiguous)
        self.assertEqual((stats["ogg files"], stats["tagged"], stats["without a name"]), (3, 2, 1))
        self.assertEqual((stats["with a dialogue line"], stats["with a subtitle"]), (1, 1))
        self.assertEqual(self.tags("story", "bank_01", "0ABCDEF1.ogg"),
                         ["TITLE=heistau_abcd_01", "LABEL=HEIST_ZED_1", "SPEAKER=ZED", "SUBTITLE=Move. Now.",
                          "SUBTITLE_EN=Move. Now.", "SUBTITLE_DE=Los."])
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=door_slam_03"])
        self.assertEqual(self.tags("story", "bank_01", "00000123.ogg"), [])

    def test_cli_accepts_a_missing_trailing_subtitle(self):
        # A CSV row can omit its final optional field rather than end with a comma.
        with open(self.dialogue, "w", encoding="utf-8", newline="\n") as f:
            csv.writer(f, lineterminator="\n").writerows([
                ["wave", "key", "label", "speaker", "text_de", "text_en"],
                ["heistau_abcd_01", "HEISTAU_ABCD", "HEIST_ZED_1", "ZED", "~z~Los."],
            ])
        parts = ("story", "bank_01", "0ABCDEF1.ogg")
        path = os.path.join(self.opus, *parts)
        with open(path, "rb") as f:
            before = f.read()
        result = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names,
                                 "--source", "story", "--dialogue", self.dialogue], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.tags(*parts), ["TITLE=heistau_abcd_01", "LABEL=HEIST_ZED_1", "SPEAKER=ZED",
                                             "SUBTITLE=Los.", "SUBTITLE_DE=Los."])
        with open(path, "rb") as f:
            after = f.read()
        self.assertEqual(before[list(T.pages(before))[2][0]:], after[list(T.pages(after))[2][0]:])
        self.assertTrue(crc_ok(after))

    def test_duplicate_id_stream_gets_the_same_tags(self):
        # awc_opus.py names a bank's second stream with an id already used <id>~2 (export_stems); it has the same
        # id, so it is the same name and dialogue line.
        stem = awc_opus.export_stems([0x0ABCDEF1, 0x0ABCDEF1])[1]
        make_ogg(os.path.join(self.opus, "story", "bank_01", stem + ".ogg"))
        dialogue, langs = T.load_dialogue(self.dialogue)
        names, ambiguous = T.load_names(self.names, "story")
        stats = T.tag_folder(self.opus, names, dialogue, langs, ambiguous=ambiguous)
        self.assertEqual((stats["ogg files"], stats["tagged"], stats["without a name"]), (4, 3, 1))
        self.assertEqual(self.tags("story", "bank_01", stem + ".ogg"), self.tags("story", "bank_01", "0ABCDEF1.ogg"))
        self.assertIn("TITLE=heistau_abcd_01", self.tags("story", "bank_01", stem + ".ogg"))

    def test_second_run_replaces_its_own_tags(self):
        names, ambiguous = T.load_names(self.names, "story")
        T.tag_folder(self.opus, names, ambiguous=ambiguous)
        path = os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg")
        with open(path, "rb") as f:
            once = f.read()
        T.tag_folder(self.opus, names, ambiguous=ambiguous)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), once)
        self.assertTrue(crc_ok(once))

    def test_dry_run_writes_nothing(self):
        path = os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg")
        with open(path, "rb") as f:
            before = f.read()
        names, ambiguous = T.load_names(self.names, "story")
        stats = T.tag_folder(self.opus, names, dry_run=True, ambiguous=ambiguous)
        self.assertEqual(stats["tagged"], 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_manifest_bank_label_with_extension(self):
        # awc_names.py labels a bank from an audio_manifest.json with its archive Path, ".awc" included; awc_opus.py
        # writes that bank under story/bank_02/ (no extension).
        names = os.path.join(self.tmp.name, "manifest_names.csv")
        write_csv(names, NAMES, [{"source": "audio_manifest.json", "bank": "story/bank_02.awc",
                                  "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"}])
        loaded, ambiguous = T.load_names(names)
        stats = T.tag_folder(self.opus, loaded, ambiguous=ambiguous)
        # 3 files; this CSV names only bank_02's, so bank_01's two stay without a name
        self.assertEqual((stats["ogg files"], stats.get("tagged", 0), stats.get("without a name", 0)), (3, 1, 2))
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=door_slam_03"])

    def test_bank_label_as_is_wins_over_its_awc_spelling(self):
        # "story/bank_02" and "story/bank_02.awc" for one id (a directory scan can give both, from bank_02.awc and
        # bank_02.awc.awc): each folder keeps its own name, whatever the row order.
        make_ogg(os.path.join(self.opus, "story", "bank_02.awc", "1FFFFFFF.ogg"), stream_id=0x1FFFFFFF)

        def row(bank, name):
            return {"source": "s", "bank": bank, "stream_id": "1fffffff", "name": name, "rule": "family"}

        rows = [row("story/bank_02", "plain_label"), row("story/bank_02.awc", "awc_label")]
        names = os.path.join(self.tmp.name, "both.csv")
        for order in (rows, rows[::-1]):
            write_csv(names, NAMES, order)
            loaded, ambiguous = T.load_names(names)
            self.assertEqual(ambiguous, {})
            T.tag_folder(self.opus, loaded, ambiguous=ambiguous)
            self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=plain_label"])
            self.assertEqual(self.tags("story", "bank_02.awc", "1FFFFFFF.ogg"), ["TITLE=awc_label"])

    def test_an_id_a_bank_names_twice_is_reported_and_left_untagged(self):
        # Only the low 29 bits of a name hash are stored, so two names can claim one id: two sources over one archive
        # (--source is optional) or a genuine collision. The last CSV row used to win silently, which also put that
        # name's dialogue subtitle in the file.
        clash = os.path.join(self.tmp.name, "clash.csv")
        write_csv(clash, NAMES, [
            {"source": "story", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
            {"source": "other", "bank": "bank_02", "stream_id": "1fffffff", "name": "wrong_name", "rule": "family"},
        ])
        names, ambiguous = T.load_names(clash)
        self.assertEqual(ambiguous, {"bank_02": {0x1FFFFFFF}})
        self.assertEqual(names["bank_02"], {})
        path = os.path.join(self.opus, "story", "bank_02", "1FFFFFFF.ogg")
        with open(path, "rb") as f:
            before = f.read()
        stats = T.tag_folder(self.opus, names, ambiguous=ambiguous)
        self.assertEqual((stats["ogg files"], stats["ambiguous name"], stats.get("tagged", 0)), (3, 1, 0))
        self.assertEqual(stats["without a name"], 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_one_name_from_two_sources_is_not_ambiguous(self):
        # awc_names.py can list the same name twice (a manifest and a name list); that is one name, not a clash.
        twice = os.path.join(self.tmp.name, "twice.csv")
        write_csv(twice, NAMES, [
            {"source": "story", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
            {"source": "other", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
        ])
        names, ambiguous = T.load_names(twice)
        self.assertEqual(ambiguous, {})
        self.assertEqual(names["bank_02"], {0x1FFFFFFF: "door_slam_03"})
        stats = T.tag_folder(self.opus, names, ambiguous=ambiguous)
        self.assertEqual(stats["tagged"], 1)
        self.assertNotIn("ambiguous name", stats)
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=door_slam_03"])

    def test_cli_reports_incomplete_directory_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "opus")
            os.mkdir(root)
            blocked = os.path.join(root, "blocked")
            names = os.path.join(directory, "names.csv")
            with open(names, "w", encoding="utf-8", newline="\n") as f:
                f.write("source,bank,stream_id,name\n")

            def walk(path, onerror=None):
                if onerror:
                    onerror(PermissionError(13, "Permission denied", blocked))
                return iter(())

            with mock.patch.object(T.os, "walk", walk), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(T.main([root, "--names", names, "--dry-run"]), 2)
            self.assertIn(blocked, err.getvalue())

    def test_cli(self):
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names, "--source", "story",
                            "--dialogue", self.dialogue], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("with a subtitle", r.stdout)
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names + ".missing"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)

    def test_cli_reads_utf8_bom_in_names_and_dialogue(self):
        for path in map(Path, (self.names, self.dialogue)):
            path.write_bytes(codecs.BOM_UTF8 + path.read_bytes())
        result = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names,
                                 "--source", "story", "--dialogue", self.dialogue], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), ["TITLE=door_slam_03"])
        self.assertIn("SUBTITLE=Move. Now.", self.tags("story", "bank_01", "0ABCDEF1.ogg"))

    def test_cli_reports_a_clashing_id(self):
        clash = os.path.join(self.tmp.name, "clash_cli.csv")
        write_csv(clash, NAMES, [
            {"source": "story", "bank": "bank_02", "stream_id": "1fffffff", "name": "door_slam_03", "rule": "family"},
            {"source": "other", "bank": "bank_02", "stream_id": "1fffffff", "name": "wrong_name", "rule": "family"},
        ])
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", clash, "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ambiguous name", r.stdout)
        self.assertIn("1", r.stdout.split("ambiguous name")[1].split()[0])

    def test_cli_names_row_cut_short_is_one_error_line(self):
        with open(self.names, "a", encoding="utf-8", newline="") as fh:
            fh.write("story,bank_01\n")          # an interrupted write: stream_id and later columns missing
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)
        self.assertIn("line 5", r.stderr)
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), [])     # nothing was tagged

    def test_cli_names_row_cut_after_stream_id_is_one_error_line(self):
        with open(self.names, "a", encoding="utf-8", newline="") as fh:
            fh.write("story,bank_01,1FFFFFFF\n")   # cut after stream_id: the name column is missing
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)
        self.assertIn("line 5", r.stderr)
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), [])     # nothing was tagged

    def test_cli_dialogue_row_missing_wave_is_one_error_line(self):
        with open(self.dialogue, "w", encoding="utf-8", newline="") as fh:
            fh.write("label,wave,text_en\n")
            fh.write("ONLY_LABEL\n")
        r = subprocess.run([sys.executable, "-I", TOOL, self.opus, "--names", self.names,
                            "--dialogue", self.dialogue], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)
        self.assertIn("line 2", r.stderr)
        self.assertEqual(self.tags("story", "bank_02", "1FFFFFFF.ogg"), [])


if __name__ == "__main__":
    unittest.main()
