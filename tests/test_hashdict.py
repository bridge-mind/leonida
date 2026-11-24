"""Unit tests for tools/hashdict.py: Jenkins one-at-a-time (joaat) hashing, string harvesting,
lookup CLI, case-insensitivity, and collision handling.

    python -I tests/test_hashdict.py
"""
import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "hashdict.py")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

import hashdict


class JenkinsHashTest(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(hashdict.joaat(""), 0x00000000)

    def test_known_reference_vectors(self):
        # Known Jenkins one-at-a-time reference vectors
        self.assertEqual(hashdict.joaat("a"), 0xCA2E9442)
        self.assertEqual(hashdict.joaat("apple"), 0x88F08EF3)
        self.assertEqual(hashdict.joaat("the quick brown fox jumps over the lazy dog"), 0x2D1643CE)
        # Canonical vehicle name hash documented in docs/design.md
        self.assertEqual(hashdict.joaat("adder"), 0xB779A091)

    def test_case_insensitivity(self):
        # joaat normalizes strings by lowercasing prior to hashing
        self.assertEqual(hashdict.joaat("ADDER"), 0xB779A091)
        self.assertEqual(hashdict.joaat("Adder"), 0xB779A091)
        self.assertEqual(hashdict.joaat("aDdEr"), 0xB779A091)
        self.assertEqual(
            hashdict.joaat("vehicle_model/car01.wft"),
            hashdict.joaat("VEHICLE_MODEL/CAR01.WFT"),
        )
        self.assertEqual(
            hashdict.joaat("Shader_Disasm:CodeBlock"),
            hashdict.joaat("shader_disasm:codeblock"),
        )

    def test_unsigned_32bit_range(self):
        test_strings = [
            "",
            "x",
            "test_token_123",
            "very_long_string_identifier_" * 10,
            "path/to/asset.bin",
        ]
        for s in test_strings:
            h = hashdict.joaat(s)
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 0xFFFFFFFF)
            self.assertEqual(h & 0xFFFFFFFF, h)

    def test_latin1_encoding_and_replacement(self):
        # Latin-1 character in range 0-255: é hashes as the one byte 0xE9. "café" and "caf\xe9" are the same
        # Python string, so the value is pinned (Latin-1 joaat, computed independently) instead of compared.
        self.assertEqual(hashdict.joaat("caf\xe9"), 0xEEFBF723)
        # Character outside Latin-1 range is replaced with '?'
        self.assertEqual(hashdict.joaat("caf\u20ac"), hashdict.joaat("caf?"))


class StringHarvestTest(unittest.TestCase):
    def test_harvest_basic_identifiers(self):
        data = b"\x00\x01first_identifier\x00second_identifier\xff"
        found = set()
        hashdict.harvest(data, found, minlen=3)
        self.assertIn("first_identifier", found)
        self.assertIn("second_identifier", found)

    def test_harvest_delimiters_split(self):
        # Strings are split by [./:\\] to extract nested token pieces
        data = b"\x00folder/sub_dir\\file_item.ext:asset_name\x00"
        found = set()
        hashdict.harvest(data, found, minlen=3)
        self.assertIn("folder/sub_dir\\file_item.ext:asset_name", found)
        self.assertIn("folder", found)
        self.assertIn("sub_dir", found)
        self.assertIn("file_item", found)
        self.assertIn("ext", found)
        self.assertIn("asset_name", found)

    def test_harvest_minlen_threshold(self):
        # Subparts shorter than minlen are filtered out
        data = b"\x00ab.cde.f.ghij\x00"
        found3 = set()
        hashdict.harvest(data, found3, minlen=3)
        self.assertNotIn("ab", found3)
        self.assertNotIn("f", found3)
        self.assertIn("cde", found3)
        self.assertIn("ghij", found3)

        found4 = set()
        hashdict.harvest(data, found4, minlen=4)
        self.assertNotIn("cde", found4)
        self.assertIn("ghij", found4)

    def test_harvest_short_standalone_identifiers(self):
        # The default run pattern needs three bytes, so --min 1/2 must use a shorter matcher to
        # keep one- and two-letter identifiers that stand alone (and still keep the longer ones).
        data = b"\x00a\x00\x00ab\x00\x00abc\x00"

        found1 = set()
        hashdict.harvest(data, found1, minlen=1)
        self.assertIn("a", found1)
        self.assertIn("ab", found1)
        self.assertIn("abc", found1)

        found2 = set()
        hashdict.harvest(data, found2, minlen=2)
        self.assertNotIn("a", found2)
        self.assertIn("ab", found2)
        self.assertIn("abc", found2)

        found3 = set()
        hashdict.harvest(data, found3, minlen=3)
        self.assertNotIn("a", found3)
        self.assertNotIn("ab", found3)
        self.assertIn("abc", found3)


class LongRunHarvestTest(unittest.TestCase):
    """A printable run longer than hashdict.MAX_NAME_LEN used to be cut at 201 characters, and the next match
    started mid-run, so the component straddling the cut was lost and two fragments were added instead."""

    def harvest(self, text):
        found = set()
        hashdict.harvest(b"\x00" + text.encode("ascii") + b"\x00", found, minlen=3)
        return found

    def test_long_path_keeps_the_component_that_straddles_the_old_cut(self):
        dirs = [f"dir_{i:02d}_abc" for i in range(15)]
        dirs[-1] += "x" * (181 - len("/".join(dirs)))
        path = "platform:/levels/" + "/".join(dirs) + "/props_lev_des_building_name.ytd"
        self.assertEqual(len(path), 230)
        self.assertTrue(path[:201].endswith("/pr"))   # the old pattern cut inside "props_..."
        found = self.harvest(path)
        for part in ["platform", "levels", "props_lev_des_building_name", "ytd"] + dirs:
            self.assertIn(part, found)
        # no truncated prefix, no fragment of the straddling component, and the 230-character run is not a name
        self.assertNotIn(path, found)
        self.assertNotIn(path[:201], found)
        self.assertNotIn("ops_lev_des_building_name", found)
        self.assertNotIn("ops_lev_des_building_name.ytd", found)

    def test_delimiter_free_token_over_the_cap_adds_nothing(self):
        token = "a" * (hashdict.MAX_NAME_LEN + 1)
        self.assertEqual(self.harvest(token), set())

    def test_run_of_exactly_the_cap_is_added_whole(self):
        token = "n" * hashdict.MAX_NAME_LEN
        self.assertEqual(self.harvest(token), {token})
        # with a delimiter: the whole run and its one piece of at least minlen ("d" is too short)
        path = "d/" + "n" * (hashdict.MAX_NAME_LEN - 2)
        self.assertEqual(len(path), hashdict.MAX_NAME_LEN)
        self.assertEqual(self.harvest(path), {path, "n" * (hashdict.MAX_NAME_LEN - 2)})


class LookupCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hashes_file = os.path.join(self.tmp.name, "hashes.json")
        self.mapping = {
            "b779a091": "adder",
            "88f08ef3": "apple",
            "18c8836d": "str_85_83667",
        }
        with open(self.hashes_file, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            capture_output=True,
            text=True,
        )

    def test_lookup_rejects_a_dictionary_that_is_not_an_object(self):
        # #1034: a JSON array is one error line naming the file, not an AttributeError traceback
        bad = os.path.join(self.tmp.name, "list.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("[]")
        p = self.run_tool("--lookup", bad, "0x1")
        self.assertEqual(p.returncode, 2)
        self.assertIn("list.json", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_build_rejects_a_malformed_archive_index(self):
        # #1043: a bad <carve>/<archive>/index.json is one ERROR line and exit 2, not a JSONDecodeError traceback
        for body in ("not json", "[]", '{"files": 3}', '{"other": []}'):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as carve:
                os.makedirs(os.path.join(carve, "arch"))
                with open(os.path.join(carve, "arch", "index.json"), "w", encoding="utf-8") as f:
                    f.write(body)
                p = self.run_tool(carve, "--out", os.path.join(carve, "out.json"))
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("ERROR: cannot read", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_lookup_known_hashes(self):
        p = self.run_tool("--lookup", self.hashes_file, "0xb779a091", "88f08ef3")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [line.strip() for line in p.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(lines, [
            "0xb779a091 adder",
            "88f08ef3 apple",
        ])

    def test_lookup_unknown_hash_prints_question_mark(self):
        p = self.run_tool("--lookup", self.hashes_file, "0xdeadbeef")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0xdeadbeef ?", p.stdout)

    def test_lookup_requires_at_least_one_hash(self):
        result = self.run_tool("--lookup", self.hashes_file)
        self.assertEqual(result.returncode, 2)
        self.assertIn("hash", result.stderr.lower())

    def test_lookup_hex_formatting_variants(self):
        # Supports 0x prefix, uppercase, lowercase, and raw hex
        p = self.run_tool("--lookup", self.hashes_file, "0XB779A091", "0x18c8836d", "88F08EF3")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [line.strip() for line in p.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(lines, [
            "0XB779A091 adder",
            "0x18c8836d str_85_83667",
            "88F08EF3 apple",
        ])


class LookupBadInputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hashes_file = os.path.join(self.tmp.name, "hashes.json")
        with open(self.hashes_file, "w", encoding="utf-8") as f:
            json.dump({"88f08ef3": "apple"}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_non_hex_value_is_a_usage_error_before_any_output(self):
        p = self.run_tool("--lookup", self.hashes_file, "88f08ef3", "apple")
        self.assertEqual(p.returncode, 2)
        self.assertIn("'apple' is not a hexadecimal hash", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stdout, "")          # nothing printed for the valid value before the bad one

    def test_unreadable_dictionary_is_one_line(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        for path in (os.path.join(self.tmp.name, "missing.json"), bad):
            with self.subTest(path=os.path.basename(path)):
                p = self.run_tool("--lookup", path, "88f08ef3")
                self.assertEqual(p.returncode, 2)
                self.assertIn(path, p.stderr)
                self.assertNotIn("Traceback", p.stderr)


class CarveAndExtraCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.carve_dir = os.path.join(self.dir, "carve")
        os.makedirs(self.carve_dir)

        # 1. Archive in default --archives list ('common')
        common_dir = os.path.join(self.carve_dir, "common")
        os.makedirs(common_dir)
        common_index = {
            "files": [
                {"off": 0, "head": "00000000", "type": "bin"},
                {"off": 16, "head": "00000000", "type": "bink2"},
            ]
        }
        with open(os.path.join(common_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(common_index, f)
        with open(os.path.join(common_dir, f"{0:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00common_archive_asset\x00\x00")
        with open(os.path.join(common_dir, f"{16:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00bink2_skipped_asset\x00\x00")

        # 2. Archive not in default list, but with metadata head (PSIN: 5053494e)
        meta_dir = os.path.join(self.carve_dir, "custom_pack")
        os.makedirs(meta_dir)
        meta_index = {
            "files": [
                {"off": 32, "head": "5053494e1234", "type": "bin"},
            ]
        }
        with open(os.path.join(meta_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(meta_index, f)
        with open(os.path.join(meta_dir, f"{32:012x}.bin"), "wb") as f:
            f.write(b"\x00\x00pso_metadata_token\x00\x00")

        # 3. Extra directory with filenames
        self.extra_dir = os.path.join(self.dir, "extra_files")
        os.makedirs(self.extra_dir)
        with open(os.path.join(self.extra_dir, "model_prop.mesh"), "w") as f:
            f.write("")

        # 4. Extra standalone file
        self.extra_file = os.path.join(self.dir, "standalone.txt")
        with open(self.extra_file, "wb") as f:
            f.write(b"\x00standalone_symbol_name\x00")

        self.out_json = os.path.join(self.dir, "out_hashes.json")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            capture_output=True,
            text=True,
        )

    def test_missing_extra_input_leaves_existing_dictionary_untouched(self):
        missing = os.path.join(self.dir, "missing.txt")
        previous = {"00000001": "previous_name"}
        with open(self.out_json, "w", encoding="utf-8", newline="\n") as output:
            json.dump(previous, output)

        result = self.run_tool(self.carve_dir, "--extra", self.extra_file, missing, "--out", self.out_json)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--extra", result.stderr)
        self.assertIn(missing, result.stderr)
        with open(self.out_json, encoding="utf-8") as output:
            self.assertEqual(json.load(output), previous)

    def test_build_dictionary_and_roundtrip_lookup(self):
        p = self.run_tool(
            self.carve_dir,
            "--out", self.out_json,
            "--extra", self.extra_dir, self.extra_file,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(os.path.isfile(self.out_json))

        with open(self.out_json, "r", encoding="utf-8") as f:
            d = json.load(f)

        # Confirm harvested strings are in dictionary under their joaat hex
        self.assertEqual(d.get(f"{hashdict.joaat('common_archive_asset'):08x}"), "common_archive_asset")
        self.assertEqual(d.get(f"{hashdict.joaat('pso_metadata_token'):08x}"), "pso_metadata_token")
        self.assertEqual(d.get(f"{hashdict.joaat('model_prop'):08x}"), "model_prop")
        self.assertEqual(d.get(f"{hashdict.joaat('standalone_symbol_name'):08x}"), "standalone_symbol_name")

        # Bink2 entries should be skipped
        self.assertNotIn(f"{hashdict.joaat('bink2_skipped_asset'):08x}", d)

        # Round-trip lookup via CLI
        lookup_hash = f"0x{hashdict.joaat('model_prop'):08x}"
        p_lookup = self.run_tool("--lookup", self.out_json, lookup_hash)
        self.assertEqual(p_lookup.returncode, 0, p_lookup.stderr)
        self.assertIn(f"{lookup_hash} model_prop", p_lookup.stdout)

    def test_unreadable_extra_subdirectory_preserves_existing_dictionary(self):
        blocked = os.path.join(self.extra_dir, "blocked")
        os.mkdir(blocked)
        previous = {"12345678": "keep"}
        with open(self.out_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(previous, f)
        scandir = os.scandir

        def scan(path):
            if path == blocked:
                raise PermissionError(errno.EACCES, "synthetic access denied", path)
            return scandir(path)

        command = [TOOL, self.carve_dir, "--extra", self.extra_dir, "--out", self.out_json]
        with (mock.patch.object(os, "scandir", side_effect=scan),
              mock.patch.object(sys, "argv", command),
              mock.patch.object(sys, "stdout", new_callable=io.StringIO),
              mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr):
            result = hashdict.main()

        self.assertEqual(result, 2, stderr.getvalue())
        self.assertIn("--extra", stderr.getvalue())
        self.assertIn(blocked, stderr.getvalue())
        with open(self.out_json, encoding="utf-8") as f:
            self.assertEqual(json.load(f), previous)

    def test_one_archive_folder_is_harvested_as_that_archive(self):
        # <work>/carve/common (its own index.json, as the gamedata_* tools take it) used to give an empty
        # dictionary with exit 0: only <carve>/<archive>/index.json was looked for.
        p = self.run_tool(os.path.join(self.carve_dir, "common"), "--out", self.out_json)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("common: ", p.stdout)
        with open(self.out_json, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d.get(f"{hashdict.joaat('common_archive_asset'):08x}"), "common_archive_asset")
        self.assertNotIn(f"{hashdict.joaat('bink2_skipped_asset'):08x}", d)
        # An archive outside --archives still contributes only its PSIN/RBF0 blobs.
        p = self.run_tool(os.path.join(self.carve_dir, "custom_pack"), "--out", self.out_json)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(self.out_json, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {f"{hashdict.joaat('pso_metadata_token'):08x}": "pso_metadata_token"})

    def test_out_naming_a_one_archive_index_is_refused(self):
        index = os.path.join(self.carve_dir, "common", "index.json")
        with open(index, "rb") as f:
            before = f.read()
        p = self.run_tool(os.path.join(self.carve_dir, "common"), "--out", index)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("is one of the inputs", p.stderr)
        with open(index, "rb") as f:
            self.assertEqual(f.read(), before)

    def assert_carved_blob_preserved(self, output, blob):
        with open(blob, "rb") as f:
            original = f.read()
        p = self.run_tool(self.carve_dir, "--out", output)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("is one of the inputs", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        with open(blob, "rb") as f:
            self.assertEqual(f.read(), original)

    def test_out_cannot_replace_a_harvested_blob(self):
        # The existing --out check protects index.json and --extra, but the
        # <offset>.bin files read during harvesting are inputs as well.
        for archive, offset in (("common", 0), ("custom_pack", 32)):
            with self.subTest(archive=archive):
                blob = os.path.join(self.carve_dir, archive, f"{offset:012x}.bin")
                self.assert_carved_blob_preserved(blob, blob)

    def test_out_cannot_replace_a_hard_link_to_a_harvested_blob(self):
        blob = os.path.join(self.carve_dir, "common", f"{0:012x}.bin")
        alias = os.path.join(self.dir, "harvest_alias.json")
        try:
            os.link(blob, alias)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"hard links unavailable: {error}")
        self.assert_carved_blob_preserved(alias, blob)
        self.assertTrue(os.path.samefile(blob, alias))

    def test_out_cannot_replace_a_symlink_to_a_harvested_blob(self):
        blob = os.path.join(self.carve_dir, "common", f"{0:012x}.bin")
        alias = os.path.join(self.dir, "harvest_link.json")
        try:
            os.symlink(blob, alias)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.assert_carved_blob_preserved(alias, blob)
        self.assertTrue(os.path.islink(alias))

    def test_build_requires_an_explicit_carve_directory(self):
        empty = os.path.join(self.dir, "empty")
        os.mkdir(empty)
        with open(self.out_json, "w", encoding="utf-8") as f:
            f.write('{"12345678": "keep"}')
        result = subprocess.run([sys.executable, "-I", TOOL, "--out", self.out_json],
                                cwd=empty, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("carve", result.stderr.lower())
        with open(self.out_json, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"12345678": "keep"})

    def test_missing_or_file_carve_is_one_line(self):
        with open(self.out_json, "w", encoding="utf-8") as f:
            f.write('{"12345678": "keep"}')
        for carve, what in ((os.path.join(self.dir, "no_such_carve"), "does not exist"),
                            (self.extra_file, "is not a folder")):
            with self.subTest(what=what):
                p = self.run_tool(carve, "--out", self.out_json)
                self.assertEqual(p.returncode, 2)
                self.assertIn(carve, p.stderr)
                self.assertIn(what, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                with open(self.out_json, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), {"12345678": "keep"})

    def test_non_bink_row_head_and_offset_are_validated_before_writing(self):
        # Missing/nonstring head and invalid (missing, negative, float, bool) offsets must be one ERROR
        # line naming the file and row, exit 2, and never overwrite --out.
        cases = [
            {"off": 0, "type": "bin"},                       # missing head
            {"off": 0, "head": 123, "type": "bin"},          # non-string head
            {"head": "00000000", "type": "bin"},             # missing offset
            {"off": -1, "head": "00000000", "type": "bin"},  # negative offset
            {"off": 1.5, "head": "00000000", "type": "bin"}, # non-integer offset
            {"off": True, "head": "00000000", "type": "bin"},# bool offset (True == 1)
        ]
        for row in cases:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as carve:
                os.makedirs(os.path.join(carve, "arch"))
                with open(os.path.join(carve, "arch", "index.json"), "w", encoding="utf-8") as f:
                    json.dump({"files": [row]}, f)
                out = os.path.join(carve, "out.json")
                p = self.run_tool(carve, "--out", out)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("index.json", p.stderr)
                self.assertIn("row 0", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertFalse(os.path.exists(out), "no report may be written on bad input")

    def test_listed_blob_missing_from_disk_is_skipped_and_reported(self):
        # The PSIN blob is listed in custom_pack/index.json but gone: the run keeps every other string.
        os.remove(os.path.join(self.carve_dir, "custom_pack", f"{32:012x}.bin"))
        p = self.run_tool(self.carve_dir, "--out", self.out_json)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("1 carved blobs", p.stderr)
        self.assertIn(f"{32:012x}.bin", p.stderr)
        with open(self.out_json, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d.get(f"{hashdict.joaat('common_archive_asset'):08x}"), "common_archive_asset")
        self.assertNotIn(f"{hashdict.joaat('pso_metadata_token'):08x}", d)

    def test_archive_index_saved_with_a_utf8_bom_is_read(self):
        # Windows PowerShell 5.1 (Set-Content, Out-File) and old Notepad write JSON with a UTF-8 BOM; the
        # names in that index must still be harvested, not end the build with "Expecting value".
        index = os.path.join(self.carve_dir, "common", "index.json")
        with open(index, "rb") as f:
            raw = f.read()
        with open(index, "wb") as f:
            f.write(b"\xef\xbb\xbf" + raw)

        p = self.run_tool(self.carve_dir, "--out", self.out_json)

        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertNotIn("Expecting value", p.stderr)
        with open(self.out_json, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d.get(f"{hashdict.joaat('common_archive_asset'):08x}"), "common_archive_asset")

    def test_build_is_reproducible_with_collisions(self):
        with open(self.extra_file, "wb") as f:
            f.write(b"widget Widget str_85_83667 str_63346_847129")

        flags = ["-P"] if sys.version_info >= (3, 11) else []
        results = []
        for seed in ("1", "2"):
            # Unlike -I, -P keeps PYTHONHASHSEED while leaving the script directory off sys.path.
            result = subprocess.run(
                [sys.executable, *flags, TOOL, self.carve_dir, "--out", self.out_json, "--extra", self.extra_file],
                env=dict(os.environ, PYTHONHASHSEED=seed), capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(self.out_json, "rb") as f:
                results.append(f.read())

        self.assertEqual(results[0], results[1])
        mapping = json.loads(results[0])
        self.assertEqual(mapping[f"{hashdict.joaat('widget'):08x}"], "Widget")
        # The two synthetic identifiers share a 32-bit hash.
        self.assertEqual(mapping["18c8836d"], "str_63346_847129")


class ContinuationStreamTest(unittest.TestCase):
    """A carved stream that fills whole 64 KiB blocks continues in the next one (docs/page_directory.md): the
    blob is harvested with its continuation, so a name across the boundary is kept whole."""

    NAME = b"synthetic_rig_name"
    CUT = 8                       # the boundary falls after "syntheti"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carve = os.path.join(self.tmp.name, "carve")
        self.out = os.path.join(self.tmp.name, "hashes.json")

    def write_archive(self, arch, streams):
        """One archive folder of carved streams laid out back to back, as rpf9_carve.py writes them."""
        folder = os.path.join(self.carve, arch)
        os.makedirs(folder)
        rows, off = [], 0
        for data in streams:
            with open(os.path.join(folder, f"{off:012x}.bin"), "wb") as f:
                f.write(data)
            rows.append({"off": off, "stored": len(data), "size": len(data), "head": data[:16].hex()})
            off += 0x10000 * ((len(data) + 0xFFFF) // 0x10000)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": rows}, f)

    def first_stream(self, magic, size=65536):
        """`size` bytes starting with `magic` and ending in the first CUT bytes of NAME."""
        tail = b"\x00" + self.NAME[:self.CUT]
        return magic + b"\x01" * (size - len(magic) - len(tail)) + tail

    def names(self, *args):
        p = subprocess.run([sys.executable, "-I", TOOL, self.carve, "--out", self.out, *args],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(self.out, encoding="utf-8") as f:
            return set(json.load(f).values())

    def test_pso_blob_is_read_past_its_first_stream(self):
        # levels is not in the default --archives: only PSIN/RBF0 blobs are read there, now with their continuation
        self.write_archive("levels", [self.first_stream(b"PSIN" + bytes(12)),
                                      self.NAME[self.CUT:] + b"\x00continuation_only_name\x00"])
        names = self.names()
        self.assertIn(self.NAME.decode(), names)
        self.assertIn("continuation_only_name", names)
        self.assertNotIn(self.NAME[:self.CUT].decode(), names)

    def test_name_across_the_boundary_of_a_fully_harvested_archive(self):
        # a first stream of 128 KiB continues as well (docs/page_directory.md)
        self.write_archive("common", [self.first_stream(b"\x00" * 16, 2 * 65536), self.NAME[self.CUT:] + b"\x00"])
        names = self.names()
        self.assertIn(self.NAME.decode(), names)
        self.assertNotIn(self.NAME[:self.CUT].decode(), names)
        self.assertNotIn(self.NAME[self.CUT:].decode(), names)

    def test_streams_that_do_not_continue_are_harvested_apart(self):
        # a stream short of a whole block ends its blob, and a new PSIN stream never continues the one before
        self.write_archive("common", [self.first_stream(b"\x00" * 16, 65535), self.NAME[self.CUT:] + b"\x00",
                                      self.first_stream(b"\x00" * 16), b"PSIN" + self.NAME[self.CUT:] + b"\x00"])
        names = self.names()
        self.assertNotIn(self.NAME.decode(), names)
        self.assertIn(self.NAME[:self.CUT].decode(), names)
        self.assertIn(self.NAME[self.CUT:].decode(), names)


class OutputPathTest(unittest.TestCase):
    def test_out_creates_its_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            out = os.path.join(tmp, "nested", "deeper", "hashes.json")
            p = subprocess.run([sys.executable, "-I", TOOL, carve, "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertTrue(os.path.isfile(out))
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {})

    def test_out_naming_an_input_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(os.path.join(carve, "common"))
            index = os.path.join(carve, "common", "index.json")
            with open(index, "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            extra = os.path.join(tmp, "names.txt")
            with open(extra, "wb") as f:
                f.write(b"adder\n")
            for out in (extra, index):
                with open(out, "rb") as f:
                    before = f.read()
                p = subprocess.run([sys.executable, "-I", TOOL, carve, "--extra", extra, "--out", out],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("is one of the inputs", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                with open(out, "rb") as f:
                    self.assertEqual(f.read(), before)


if __name__ == "__main__":
    unittest.main()
