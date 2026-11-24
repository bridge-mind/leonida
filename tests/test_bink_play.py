"""tools/bink_play.py: name matching and the name-table cache. Synthetic names only, no game files."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bink_play  # noqa: E402

NAMES = [
    ("Stinger/INTRO_A.bk2", 4096, "3840", "2160", "450"),
    ("Stinger/INTRO_B.bk2", 8192, "3840", "2160", "300"),
    ("TV/News_01.bk2", 12288, "1280", "720", "40"),
]


class ListTest(unittest.TestCase):
    def test_filter_accepts_both_path_separators_and_ignores_case(self):
        from io import StringIO

        names = [("Clips/Demo.bk2", 4096, "640", "480", "24"), *NAMES]
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(bink_play, "load_names", return_value=names):
            for query in ("Clips/Demo", r"Clips\Demo", "cLiPs/dEmO", r"cLiPs\dEmO"):
                with self.subTest(query=query), mock.patch.object(sys, "stdout", new_callable=StringIO) as stdout:
                    rc = bink_play.main(["--game-dir", "synthetic", "--list", query])
                    self.assertEqual(rc, 0)
                    self.assertEqual(stdout.getvalue(), "Clips/Demo.bk2  640x480, 24 frames\n")


class PickTest(unittest.TestCase):
    def test_exact_path_wins_case_insensitively_and_with_backslashes(self):
        hit, _ = bink_play.pick(NAMES, "stinger\\intro_a.BK2")
        self.assertEqual(hit[1], 4096)

    def test_a_unique_part_of_the_name_is_enough(self):
        hit, _ = bink_play.pick(NAMES, "news")
        self.assertEqual(hit[0], "TV/News_01.bk2")

    def test_an_ambiguous_part_returns_the_candidates(self):
        hit, candidates = bink_play.pick(NAMES, "intro")
        self.assertIsNone(hit)
        self.assertEqual([c[0] for c in candidates], ["Stinger/INTRO_A.bk2", "Stinger/INTRO_B.bk2"])

    def test_no_match(self):
        self.assertEqual(bink_play.pick(NAMES, "credits"), (None, []))


class CacheTest(unittest.TestCase):
    def test_matching_but_damaged_cache_is_rebuilt_from_the_sources(self):
        with tempfile.TemporaryDirectory() as d:
            archive, mount, cache = (Path(d) / n for n in ("streams.csv", "mount.xml", "cache.csv"))
            archive.write_text("source,offset,size,width,height,frames\nsample,32,64,16,16,2\n",
                               encoding="utf-8", newline="\n")
            mount.write_text("<filename>$(common)/movies/synthetic.bk2</filename>",
                             encoding="utf-8", newline="\n")
            stamp = ("# " + bink_play.stamp(archive) + "|" + bink_play.stamp(mount) + "\n").encode()
            header = b"path,offset,width,height,frames\n"
            expected = [("synthetic.bk2", 32, "16", "16", "2")]
            with mock.patch.object(bink_play, "cache_path", return_value=cache):
                for damaged in (b"", b"wrong,columns\n", header + b"prefix,1,1,1,1\nbad,row\n",
                                header + b"x,garbled,1,1,1\n", header + b'x,1,1,1,"2\n',
                                header + b"\xff,1,1,1,2\n"):
                    with self.subTest(damaged=damaged):
                        cache.write_bytes(stamp + damaged)
                        self.assertEqual(bink_play.load_names(archive, mount), expected)
                        with mock.patch.object(bink_play.bink_names, "main", side_effect=AssertionError):
                            self.assertEqual(bink_play.load_names(archive, mount), expected)
                cache.write_bytes(stamp + header)
                with mock.patch.object(bink_play.bink_names, "main", side_effect=AssertionError):
                    self.assertEqual(bink_play.load_names(archive, mount), [])

    def test_the_table_is_named_once_and_rebuilt_when_the_archive_changes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, mount = d / "movies.rpf", d / "common.rpf"
            archive.write_bytes(b"x" * 16)
            mount.write_bytes(b"y")
            calls = []

            def fake_names(argv):
                calls.append(argv)
                out = Path(argv[argv.index("--out") + 1])
                out.write_text("source,offset,size,path,confidence,width,height,frames\n"
                               "movies.rpf,4096,10,Stinger/INTRO_A.bk2,anchored,3840,2160,450\n", encoding="utf-8")
                return 0

            with mock.patch.object(bink_play, "cache_path", return_value=d / "cache" / "bink_names.csv"), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                first = bink_play.load_names(archive, mount)
                second = bink_play.load_names(archive, mount)
                self.assertEqual(len(calls), 1, "the second call reads the cache")
                self.assertEqual(first, second)
                self.assertEqual(first[0][:2], ("Stinger/INTRO_A.bk2", 4096))
                archive.write_bytes(b"x" * 32)  # a different archive size: the cache is stale
                bink_play.load_names(archive, mount)
                self.assertEqual(len(calls), 2)
                bink_play.load_names(archive, mount, use_cache=False)
                self.assertEqual(len(calls), 3, "--no-cache names again")

    def test_a_different_mount_source_names_the_streams_again(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, old, new = d / "movies.rpf", d / "old.xml", d / "new.xml"
            archive.write_bytes(b"x" * 16)
            old.write_bytes(b"<old/>")
            new.write_bytes(b"<new/>")
            cache = d / "cache" / "bink_names.csv"

            def fake_names(argv):
                name = "OLD.bk2" if Path(argv[argv.index("--mount") + 1]).name == "old.xml" else "NEW.bk2"
                Path(argv[argv.index("--out") + 1]).write_text(
                    "source,offset,size,path,confidence,width,height,frames\n"
                    f"movies.rpf,4096,10,Stinger/{name},anchored,3840,2160,450\n", encoding="utf-8", newline="\n")
                return 0

            with mock.patch.object(bink_play, "cache_path", return_value=cache), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                self.assertEqual(bink_play.load_names(archive, old)[0][0], "Stinger/OLD.bk2")
                self.assertEqual(bink_play.load_names(archive, new)[0][0], "Stinger/NEW.bk2",
                                 "names from another mount source are not served from the cache")
                # a table written before the mount source was part of the key is a miss, not a crash
                cache.write_text("# " + bink_play.stamp(archive) + "\npath,offset,width,height,frames\n"
                                 "Stinger/STALE.bk2,4096,3840,2160,450\n", encoding="utf-8", newline="\n")
                self.assertEqual(bink_play.load_names(archive, new)[0][0], "Stinger/NEW.bk2")

    def test_two_lookups_at_once_each_get_their_own_names(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive, first, second = d / "movies.rpf", d / "first.xml", d / "second.xml"
            archive.write_bytes(b"x" * 16)
            first.write_bytes(b"<a/>")
            second.write_bytes(b"<b/>")
            ready, done = threading.Event(), threading.Event()
            got = {}

            def fake_names(argv):
                mount = Path(argv[argv.index("--mount") + 1])
                Path(argv[argv.index("--out") + 1]).write_text(
                    "source,offset,size,path,confidence,width,height,frames\n"
                    f"movies.rpf,4096,10,Stinger/{mount.stem}.bk2,anchored,3840,2160,450\n",
                    encoding="utf-8", newline="\n")
                if mount == first:  # hold the first lookup between writing its report and reading it
                    ready.set()
                    done.wait(10)
                return 0

            def look_up(mount):
                try:
                    got[mount.stem] = bink_play.load_names(archive, mount)[0][0]
                except SystemExit as e:
                    got[mount.stem] = str(e)

            with mock.patch.object(bink_play, "cache_path", return_value=d / "cache" / "bink_names.csv"), \
                    mock.patch.object(bink_play.bink_names, "main", side_effect=fake_names):
                worker = threading.Thread(target=look_up, args=(first,))
                worker.start()
                self.assertTrue(ready.wait(10))
                try:
                    look_up(second)
                finally:
                    done.set()
                    worker.join()
            self.assertEqual(got, {"first": "Stinger/first.bk2", "second": "Stinger/second.bk2"})
            self.assertEqual(sorted(p.name for p in (d / "cache").iterdir()), ["bink_names.csv"])


if __name__ == "__main__":
    unittest.main()
