"""Tests for tools/save_keys.py: the hash families, key collection from JSAV/ASAV/JSON, and dictionary resolution
(plain, zlib-wrapped and composed names). All names and containers are synthetic; no game data is used.

    python -I tests/test_save_keys.py
"""
import collections
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import save_keys as sk  # noqa: E402
import saves  # noqa: E402


def key(name):
    return "$0x%08x" % sk.joaat(name.encode())


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sk.main(argv)
    return buf.getvalue()


class HashFamilies(unittest.TestCase):
    def test_reference_vectors(self):
        self.assertEqual(sk.joaat(b"a"), 0xCA2E9442)
        self.assertEqual(sk.joaat(b"adder"), 0xB779A091)
        self.assertNotEqual(sk.joaat(b"Adder"), sk.joaat(b"adder"))   # case-sensitive
        self.assertEqual(sk.murmur3_32(b"hello"), 0x248BFA47)
        self.assertEqual(sk.murmur3_x64_64(b"hello"), 0xCBD8A7B341BD9B02)
        self.assertEqual(sk.murmur3_x64_64(b"The quick brown fox jumps over the lazy dog"), 0xE34BBC7BBC071B6C)
        self.assertEqual(sk.fnv1a(b"a"), 0xE40C292C)
        self.assertEqual(sk.fnv1(b"a"), 0x050C5D7E)

    def test_incremental_matches_whole(self):
        self.assertEqual(sk.joaat_final(sk.joaat_step(sk.joaat_step(0, b"Left"), b"Right")), sk.joaat(b"LeftRight"))


class Resolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        names = ["WidgetBlock", "GadgetBlock", "Colour", "spinRate", "SprocketCount", "MysteryField"]
        self.names = names
        blocks, hashes = [], []
        for i, n in enumerate(["WidgetBlock", "GadgetBlock"]):
            h = sk.joaat(n.encode())
            hashes.append(h)
            body = {key("Colour"): i, key("spinRate"): 1.5, key("SprocketCount"): {"$x": 1, "$y": 2},
                    key("MysteryField"): 0}
            blocks.append(json.dumps({key("CodeGroup"): {"$0x%08x" % h: body}}, separators=(",", ":")).encode())
        self.jsav = os.path.join(self.dir, "SLOT")
        with open(self.jsav, "wb") as f:
            f.write(saves.build_jsav(3, 2, hashes, [1, 1], blocks))
        # dictionary: some names in clear, one inside a zlib stream, the parts of another as separate words
        self.dict = os.path.join(self.dir, "strings.bin")
        with open(self.dict, "wb") as f:
            f.write(b"\0\0WidgetBlock\0Colour\0spinRate\0\x01CodeGroup\0")
            f.write(zlib.compress(b"<Item>GadgetBlock</Item>"))
            f.write(b"\0Sprocket\0Count\0Sprocket\0Count\0")

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_counts_index_and_json_keys(self):
        keys = sk.collect_keys([self.jsav])
        # 2 block ids + group + 4 fields
        self.assertEqual(len(keys), 7)
        self.assertEqual(keys[sk.joaat(b"WidgetBlock")], 2)    # index entry + inner key
        self.assertIn("7 distinct keys", run(["collect", self.jsav]))

    def test_collect_from_extracted_folder(self):
        out = os.path.join(self.dir, "x")
        with contextlib.redirect_stdout(io.StringIO()):
            saves.extract(self.jsav, out)
        self.assertEqual(set(sk.collect_keys([out])), set(sk.collect_keys([self.jsav])))

    def test_directory_scan_reads_jsav_files(self):
        # #844: a folder of .jsav saves counts the same as naming each file
        with open(self.jsav, "rb") as f:
            data = f.read()
        folder = os.path.join(self.dir, "saves")
        os.makedirs(folder)
        for name in ("a.jsav", "b.ASAV", "c.bin", "d"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data)
        one = sk.collect_keys([self.jsav])
        self.assertEqual(sk.collect_keys([folder]), collections.Counter({k: 4 * n for k, n in one.items()}))

    def test_unlistable_subfolder_is_a_clean_error_not_a_silent_skip(self):
        # os.walk skips a directory it cannot list unless onerror raises; the collection used to report
        # the remaining saves and exit 0, hiding the incomplete directory.
        with open(self.jsav, "rb") as f:
            data = f.read()
        folder = os.path.join(self.dir, "saves_sub")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.jsav"), "wb") as f:
            f.write(data)
        sub = os.path.join(folder, "sub")
        os.makedirs(sub)
        with open(os.path.join(sub, "b.jsav"), "wb") as f:
            f.write(data)
        os.chmod(sub, 0o000)
        try:
            try:
                os.listdir(sub)
                self.skipTest("the sub-folder is still listable (running as root, or Windows)")
            except OSError:
                pass
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = sk.main(["collect", folder])
            self.assertEqual(rc, 1)
            self.assertIn("error:", err.getvalue())
            self.assertIn(sub, err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
        finally:
            os.chmod(sub, 0o755)

    def test_unlistable_subfolder_is_a_clean_error_when_chmod_is_not_enough(self):
        # The chmod case above self-skips on Windows and as root, and CI only runs this file on
        # windows-latest, so the guard went unexercised there. Forcing os.scandir to raise for one
        # folder gives the same PermissionError os.walk hits for real, on every platform.
        with open(self.jsav, "rb") as f:
            data = f.read()
        folder = os.path.join(self.dir, "saves_scandir")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.jsav"), "wb") as f:
            f.write(data)
        sub = os.path.join(folder, "sub")
        os.makedirs(sub)
        with open(os.path.join(sub, "b.jsav"), "wb") as f:
            f.write(data)
        real_scandir = os.scandir
        locked = os.path.normcase(os.path.abspath(sub))

        def scandir(path, *args, **kwargs):
            if os.path.normcase(os.path.abspath(str(path))) == locked:
                raise PermissionError(13, "Access is denied", str(path))
            return real_scandir(path, *args, **kwargs)

        err = io.StringIO()
        with mock.patch.object(sk.os, "scandir", scandir):
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = sk.main(["collect", folder])
        self.assertEqual(rc, 1, err.getvalue())
        self.assertIn("error:", err.getvalue())
        self.assertIn(sub, err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_resolve_counts_per_family(self):
        res = os.path.join(self.dir, "res.json")
        text = run(["resolve", self.jsav, "--dict", self.dict, "--out", res])
        line = {ln.split()[0]: int(ln.split()[1]) for ln in text.splitlines()[1:] if ln.startswith("  ")
                and ln.split()[1].isdigit()}
        self.assertEqual(line["joaat"], 5)          # WidgetBlock, GadgetBlock (zlib), Colour, spinRate, CodeGroup
        self.assertEqual(line["joaat_lower"], 0)
        self.assertEqual(line["fnv1a"], 0)
        with open(res, encoding="utf-8") as f:
            got = json.load(f)
        self.assertEqual(got["%08x" % sk.joaat(b"GadgetBlock")], ["GadgetBlock"])
        self.assertNotIn("%08x" % sk.joaat(b"MysteryField"), got)

    def resolve_cli(self, *args):
        tool = os.path.join(ROOT, "tools", "save_keys.py")
        return subprocess.run([sys.executable, "-I", tool, "resolve", *args], cwd=ROOT, capture_output=True, text=True)

    def test_out_naming_an_input_is_refused(self):
        # --out must not replace the user's dictionary, a save, a file found in an input folder, or a hard link to one
        folder = os.path.join(self.dir, "saves")
        os.makedirs(folder)
        inside = os.path.join(folder, "more.json")
        with open(inside, "w", encoding="utf-8") as f:
            f.write("{}")
        cases = [(self.dict, [self.jsav]), (self.jsav, [self.jsav]), (inside, [folder])]
        link = os.path.join(self.dir, "link.bin")
        try:
            os.link(self.dict, link)
        except (OSError, NotImplementedError) as e:
            with self.subTest(out=os.path.basename(link)):
                self.skipTest(f"hard links unavailable: {e}")
        else:
            cases.append((link, [self.jsav]))
        for out, inputs in cases:
            with self.subTest(out=os.path.basename(out)):
                with open(out, "rb") as f:
                    before = f.read()
                r = self.resolve_cli(*inputs, "--dict", self.dict, "--out", out)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertIn("refusing to overwrite", r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                with open(out, "rb") as f:
                    self.assertEqual(f.read(), before)

    def test_out_in_a_missing_folder_is_one_error_line(self):
        r = self.resolve_cli(self.jsav, "--dict", self.dict, "--out", os.path.join(self.dir, "nope", "res.json"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("error: the folder for --out", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(r.stdout, "")      # refused before the resolve ran

    def test_rejects_negative_compose_count(self):
        result = self.resolve_cli(self.dir, "--dict", self.dict, "--compose", "-1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--compose must be 0 or more", result.stderr)

    def test_compose_finds_two_word_names(self):
        text = run(["resolve", self.jsav, "--dict", self.dict, "--compose", "50"])
        self.assertIn("joaat, composed       1 more keys", text)       # SprocketCount from Sprocket + Count
        self.assertIn("total (joaat)         6 of 7 keys", text)


class MalformedJsavTest(unittest.TestCase):
    """A lying JSAV entry count must warn and skip the table, not abort a whole batch."""

    def collect(self, data, name="save.jsav"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            keys = sk.collect_keys([path])
        return keys, err.getvalue(), path

    def jsav(self, n_code, n_script=0, entries=(), tail=b""):
        data = bytearray(b"JSAV" + struct.pack("<III", 0, n_code, n_script))
        for key in entries:
            data += struct.pack("<I", key) + bytes(16)
        return bytes(data) + tail

    def test_header_declaring_more_entries_than_present_is_skipped(self):
        keys, err, _ = self.collect(self.jsav(4))
        self.assertEqual(keys, {})
        self.assertIn("JSAV header declares 4 entries", err)
        self.assertIn("skipping its entry table", err)

    def test_partial_entry_table_is_skipped(self):
        keys, err, _ = self.collect(self.jsav(1) + bytes(10))
        self.assertEqual(keys, {})
        self.assertIn("declares 1 entries", err)

    def test_header_shorter_than_16_bytes_is_skipped(self):
        keys, err, _ = self.collect(b"JSAV" + bytes(4))
        self.assertEqual(keys, {})
        self.assertIn("JSAV header is only 8 bytes", err)

    def test_declared_zero_entries_is_not_a_warning(self):
        keys, err, _ = self.collect(self.jsav(0))
        self.assertEqual(keys, {})
        self.assertEqual(err, "")

    def test_complete_table_is_still_collected(self):
        keys, err, _ = self.collect(self.jsav(2, entries=(0x12345678, 0xABCDEF01)))
        self.assertEqual(set(keys), {0x12345678, 0xABCDEF01})
        self.assertEqual(err, "")

    def test_text_keys_from_a_truncated_file_are_kept(self):
        keys, err, _ = self.collect(self.jsav(4) + b'\n"$0x0000abcd"\n')
        self.assertEqual(set(keys), {0xABCD})
        self.assertIn("skipping its entry table", err)

    def test_cli_reports_a_warning_and_exits_zero(self):
        _keys, _err, path = self.collect(self.jsav(4))
        tool = os.path.join(ROOT, "tools", "save_keys.py")
        r = subprocess.run([sys.executable, "-I", tool, "collect", path],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("JSAV header declares 4 entries", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("0 distinct keys", r.stdout)


class MissingFilesTest(unittest.TestCase):
    """A path that can't be read is one error line naming it and exit code 1, for both commands."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.save = os.path.join(self.tmp.name, "save.json")
        with open(self.save, "w", encoding="utf-8") as f:
            json.dump({key("health"): 1}, f)
        self.words = os.path.join(self.tmp.name, "words.txt")
        with open(self.words, "wb") as f:
            f.write(b"health\0")
        self.missing = os.path.join(self.tmp.name, "nope.jsav")

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args):
        tool = os.path.join(ROOT, "tools", "save_keys.py")
        return subprocess.run([sys.executable, "-I", tool, *args], cwd=ROOT, capture_output=True, text=True)

    def assert_one_error_line(self, r, path):
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
        self.assertTrue(r.stderr.startswith("error: "), r.stderr)
        self.assertIn(path, r.stderr)

    def test_collect_names_a_missing_input(self):
        self.assert_one_error_line(self.cli("collect", self.missing), self.missing)

    def test_collect_names_the_missing_one_of_several(self):
        self.assert_one_error_line(self.cli("collect", self.save, self.missing), self.missing)

    def test_resolve_names_a_missing_input(self):
        self.assert_one_error_line(self.cli("resolve", self.missing, "--dict", self.words), self.missing)

    def test_resolve_names_a_missing_dictionary(self):
        r = self.cli("resolve", self.save, "--dict", self.missing)
        self.assert_one_error_line(r, self.missing)

    def test_resolve_with_a_missing_dictionary_writes_no_report(self):
        out = os.path.join(self.tmp.name, "names.json")
        r = self.cli("resolve", self.save, "--dict", self.missing, "--out", out)
        self.assert_one_error_line(r, self.missing)
        self.assertFalse(os.path.exists(out))

    def test_readable_files_still_resolve(self):
        r = self.cli("resolve", self.save, "--dict", self.words)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 distinct keys", r.stdout)


class DirectoryScanTest(unittest.TestCase):
    """A directory of saves must include .jsav files, not just .bin/.json/extensionless names."""

    def jsav(self, key):
        data = bytearray(b"JSAV" + struct.pack("<III", 0, 1, 0))
        data += struct.pack("<I", key) + bytes(16)
        return bytes(data)

    def test_jsav_files_in_a_directory_are_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.jsav", "b.JSAV", "c.bin", "plain"):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(self.jsav(0x12345678))
            keys = sk.collect_keys([tmp])
            self.assertEqual(keys[0x12345678], 4)

    def test_warning_names_the_offending_file_in_a_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.bin")
            with open(path, "wb") as f:
                f.write(b"JSAV" + struct.pack("<III", 0, 4, 0))  # claims 4 entries, has no table
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                keys = sk.collect_keys([tmp])
            self.assertEqual(keys, {})
            self.assertIn("broken.bin", err.getvalue())
            self.assertIn("JSAV header declares 4 entries", err.getvalue())
            self.assertEqual(err.getvalue().count("warning:"), 1)

    def test_direct_jsav_file_still_matches_directory_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.jsav")
            with open(path, "wb") as f:
                f.write(self.jsav(0xABCDEF01))
            self.assertEqual(sk.collect_keys([path]), sk.collect_keys([tmp]))



class LongIdentifiers(unittest.TestCase):
    """harvest() takes a run of identifier characters whole: it used to cut after 81 and go on in the rest."""

    def harvest(self, *chunks):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "strings.bin")
            with open(path, "wb") as f:
                for chunk in chunks:
                    f.write(b"\0" + chunk + b"\0")
            return sk.harvest([path])

    def test_a_name_longer_than_81_characters_is_one_name(self):
        name = b"A" * 81 + b"TailPart"
        got = self.harvest(name, b"Short")
        self.assertIn(name, got)
        self.assertIn(b"Short", got)
        self.assertNotIn(b"A" * 81, got)       # the old cut
        self.assertNotIn(b"TailPart", got)     # and the name it invented from the rest

    def test_the_same_inside_a_zlib_stream(self):
        name = b"Group_" + b"x" * 120
        got = self.harvest(zlib.compress(b"<Item>" + name + b"</Item>"))
        self.assertIn(name, got)
        self.assertNotIn(name[81:], got)

    def test_a_run_past_the_limit_is_left_out_whole(self):
        blob = b"Q" * (sk.IDENT_MAX + 1)
        got = self.harvest(blob, b"Q" * sk.IDENT_MAX)
        self.assertIn(b"Q" * sk.IDENT_MAX, got)
        self.assertFalse([name for name in got if len(name) > sk.IDENT_MAX])
        self.assertNotIn(b"Q", got)
        self.assertEqual({name for name in got if set(name) == {ord("Q")}}, {b"Q" * sk.IDENT_MAX})

    def test_a_127_character_name_is_harvested(self):
        name = b"Vehicle" + b"Part" * 30
        self.assertEqual(self.harvest(name), {name})


class ComposeDeterminism(unittest.TestCase):
    """`--compose` must report one answer for one input: tied word counts used to pick words in hash order."""

    WORDS = ["Widget", "Gadget", "Sprocket", "Colour", "Spin", "Rate", "Mystery", "Field", "Alpha",
             "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo",
             "Lima"]

    def keys(self):
        return {sk.joaat(b"Alpha"), sk.joaat(b"Gadget")}

    def test_the_vocabulary_does_not_depend_on_the_input_order(self):
        # compose() builds its Counter by iterating a set, so the words it chose followed hash order: the same
        # twenty words in reverse picked a different six. The six are now the alphabetically first, so Alpha is
        # one of them and Gadget, far down the list, is not.
        forward = sk.compose([w.encode() for w in self.WORDS], self.keys(), 6)
        reverse = sk.compose([w.encode() for w in reversed(self.WORDS)], self.keys(), 6)
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward[0]), {sk.joaat(b"Alpha")})
        self.assertNotIn(sk.joaat(b"Gadget"), forward[0])

    def test_a_tie_is_broken_by_the_word(self):
        keys = {sk.joaat(b"Alpha"), sk.joaat(b"Zulu")}
        hits, tried = sk.compose({b"Alpha", b"Bravo", b"Zulu"}, keys, 2)
        self.assertEqual(tried, 18)                        # 6 one-word and 3 x 4 two-word candidates
        self.assertEqual(set(hits), {sk.joaat(b"Alpha")})  # Alpha and Bravo win; Zulu is out of the six

    def test_the_cli_report_is_identical_across_runs(self):
        # Hash randomisation already varies per process, so the same command disagreed with itself.
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "k.json")
            with open(save, "w", encoding="utf-8") as f:
                f.write('{"%s": 1}' % key("AlphaBravo"))
            dic = os.path.join(tmp, "words.bin")
            with open(dic, "wb") as f:
                for w in self.WORDS:
                    f.write(b"\0" + w.encode() + b"\0")
            tool = os.path.join(ROOT, "tools", "save_keys.py")
            runs = [subprocess.run([sys.executable, "-I", tool, "resolve", save, "--dict", dic,
                                    "--compose", "6"], cwd=ROOT, capture_output=True, text=True)
                    for _ in range(5)]
            for r in runs:
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len({r.stdout for r in runs}), 1, [r.stdout for r in runs])
        composed = [ln for ln in runs[0].stdout.splitlines() if "composed" in ln][0]
        # AlphaBravo is not a harvested string, so only the two-word pass can resolve it; Alpha and Bravo are
        # both inside the six ties that used to be chosen in hash order.
        self.assertEqual(composed.split()[2], "1")


if __name__ == "__main__":
    unittest.main()
