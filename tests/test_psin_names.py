"""Synthetic tests for tools/psin_names.py (no game data: every name and hash is invented)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psin_names.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psin_names as N  # noqa: E402
from test_psin_data import make_container, ptr  # noqa: E402
from test_psin_index import joaat  # noqa: E402


def mm64(name):
    return N.murmur3_x64_128(name.encode())[0]


ROOT_T = joaat("PropSet")
MODEL, KEY, MODELS, OTHER = joaat("model"), joaat("key"), joaat("models"), joaat("other")
MEMBERS = [
    (0x100, 0, 0x4A, 0, 0, 0),           # 0: element descriptor, u32 hash
    (MODEL, 0x00, 0x4A, 0, 0, 0),        # u32 name hash
    (OTHER, 0x04, 0x4D, 0, 0, 0),        # u32 hash that is not a listed name
    (KEY, 0x08, 0x4F, 0, 0, 0),          # u64 name key
    (MODELS, 0x10, 0x5C, 0, 0, 0),       # {u32 count, u32 capacity, ptr} -> u32 hash[]
]
SCHEMA = [(ROOT_T, 0x3, 0x20, 1, MEMBERS)]
NAMES = ["prop_crate_01", "prop_barrel_02", "Prop_Lamp_03"]


def sample(model="prop_crate_01", key="prop_crate_01", models=("prop_barrel_02", "prop_lamp_03"), other=0x12345678):
    root = bytearray(0x20)
    struct.pack_into("<IIQ", root, 0, joaat(model) if model else 0, other, mm64(key))
    struct.pack_into("<IIQ", root, 0x10, len(models), len(models), ptr(2))
    items = b"".join(struct.pack("<I", joaat(m)) for m in models)
    return make_container([(ROOT_T, bytes(root)), (0x4A, items)], SCHEMA)


class TestHashValues(unittest.TestCase):
    def test_every_hash_member_is_found_with_its_field(self):
        found = N.hash_values(sample())
        fields = [(s, f, t) for s, f, t, _ in found]
        self.assertEqual(fields, [(ROOT_T, MODEL, 0x4A), (ROOT_T, OTHER, 0x4D), (ROOT_T, KEY, 0x4F),
                                  (ROOT_T, MODELS, 0x4A), (ROOT_T, MODELS, 0x4A)])
        self.assertEqual(found[0][3], joaat("prop_crate_01"))
        self.assertEqual(found[2][3], mm64("prop_crate_01"))

    def test_not_a_container(self):
        with self.assertRaises(ValueError):
            N.hash_values(b"PSIN" + b"\0" * 12)


class TestDictionaries(unittest.TestCase):
    def test_u32_is_lower_cased_joaat_and_u64_covers_both_cases(self):
        d32, d64 = N.dictionaries({"Prop_Lamp_03"})
        self.assertIn(joaat("prop_lamp_03"), d32)
        self.assertIn(mm64("Prop_Lamp_03"), d64)
        self.assertIn(mm64("prop_lamp_03"), d64)

    def test_control_is_salted(self):
        d32, d64 = N.dictionaries({"prop_crate_01"}, N.CONTROL_SALT)
        self.assertNotIn(joaat("prop_crate_01"), d32)
        self.assertIn(joaat("prop_crate_01" + N.CONTROL_SALT), d32)
        self.assertEqual(set(d64.values()), {"prop_crate_01"})

    def test_names_from_rpf9_names_json_and_text(self):
        with tempfile.TemporaryDirectory() as d:
            j, t = os.path.join(d, "n.json"), os.path.join(d, "n.txt")
            with open(j, "w") as f:
                json.dump({"files": [{"names": ["prop_crate_01.#dr", "veh_frame.#ft"]}]}, f)
            with open(t, "w") as f:
                f.write("prop_lamp_03\n\n")
            self.assertEqual(N.load_names([j, t]), {"prop_crate_01", "veh_frame", "prop_lamp_03"})


class TestScan(unittest.TestCase):
    def folder(self, d, blobs):
        for i, b in enumerate(blobs):
            with open(os.path.join(d, f"{i:04x}.bin"), "wb") as f:
                f.write(b)
        return d

    def test_counts_and_fields(self):
        with tempfile.TemporaryDirectory() as d:
            self.folder(d, [sample(), sample(model=None, models=()), b"PSIN" + b"\0" * 12, b"not psin"])
            summary, table = N.scan(d, set(NAMES))
        self.assertEqual(summary["PSIN files"], 3)
        self.assertEqual(summary["containers"], 2)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["u32 values"], 6)
        self.assertEqual(summary["u32 resolved"], 3)
        self.assertEqual(summary["u64 resolved"], 2)
        self.assertEqual(summary.get("u32 control", 0) + summary.get("u64 control", 0), 0)
        self.assertEqual(summary["containers with a resolved name"], 2)
        rows = {r["field"]: r for r in table}
        self.assertEqual(rows[f"{MODELS:08x}"]["resolved"], 2)
        self.assertEqual(rows[f"{MODEL:08x}"]["zero"], 1)
        self.assertEqual(rows[f"{OTHER:08x}"]["resolved"], 0)
        self.assertEqual(table[0]["field"], f"{KEY:08x}")       # most resolved first, then by hash

    def test_control_counts_a_planted_salted_value(self):
        with tempfile.TemporaryDirectory() as d:
            self.folder(d, [sample(model="prop_crate_01" + N.CONTROL_SALT)])
            summary, _ = N.scan(d, set(NAMES))
        self.assertEqual(summary["u32 control"], 1)


class TestCli(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_text_json_and_refs(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c")
            os.mkdir(src)
            with open(os.path.join(src, "0001.bin"), "wb") as f:
                f.write(sample())
            names = os.path.join(d, "names.txt")
            with open(names, "w") as f:
                f.write("\n".join(NAMES))
            refs = os.path.join(d, "refs.tsv")
            r = self.run_tool(src, "--names", names, "--refs", refs)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("u32 resolved", r.stdout)
            with open(refs) as f:
                rows = [line.split("\t") for line in f.read().splitlines()]
            self.assertEqual(rows[0][0], "file")
            self.assertEqual(sorted(row[5] for row in rows[1:]),
                             ["Prop_Lamp_03", "prop_barrel_02", "prop_crate_01", "prop_crate_01"])
            r = self.run_tool(src, "--names", names, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["summary"]["u64 resolved"], 1)

    def test_rejects_refs_aliasing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "one.bin")
            names = os.path.join(tmp, "names.txt")
            for path, data in ((source, sample()), (names, b"prop_crate_01\n")):
                with open(path, "wb") as f:
                    f.write(data)
            for destination in (source, names):
                with self.subTest(destination=destination):
                    with open(destination, "rb") as f:
                        original = f.read()
                    alias = os.path.join(tmp, ".", os.path.basename(destination))
                    result = self.run_tool(source, "--names", names, "--refs", alias)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--refs must not overwrite an input", result.stderr)
                    with open(destination, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_rejects_refs_directory_and_files_inside_src_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "carved")
            os.mkdir(src)
            bin_file = os.path.join(src, "0001.bin")
            original = sample()
            with open(bin_file, "wb") as f:
                f.write(original)
            names = os.path.join(tmp, "names.txt")
            with open(names, "w", encoding="utf-8") as f:
                f.write("prop_crate_01\n")
            other_dir = os.path.join(tmp, "out_dir")
            os.mkdir(other_dir)

            for bad_refs, expected_msg in (
                (other_dir, "--refs must be a file path, not a directory"),
                (src, "--refs must be a file path, not a directory"),
                (bin_file, "--refs must not overwrite an input"),
                (os.path.join(src, "..", "carved", "0001.bin"), "--refs must not overwrite an input"),
                (os.path.join(src, "new_refs.tsv"), "--refs must not overwrite an input"),
            ):
                with self.subTest(bad_refs=bad_refs):
                    r = self.run_tool(src, "--names", names, "--refs", bad_refs)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertIn(expected_msg, r.stderr)
                    with open(bin_file, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_rejects_external_hard_link_to_a_carved_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "carved")
            os.mkdir(src)
            payload = os.path.join(src, "0001.bin")
            names = os.path.join(tmp, "names.txt")
            original = sample()
            with open(payload, "wb") as f:
                f.write(original)
            with open(names, "w", encoding="utf-8", newline="\n") as f:
                f.write("prop_crate_01\n")
            refs = os.path.join(tmp, "refs.tsv")
            try:
                os.link(payload, refs)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"hard links unavailable: {e}")

            result = self.run_tool(src, "--names", names, "--refs", refs)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--refs must not overwrite an input", result.stderr)
            with open(payload, "rb") as f:
                self.assertEqual(f.read(), original)

    def test_uses_shared_same_report_path_for_refs_and_carve_index(self):
        import report_paths
        from unittest import mock
        self.assertIs(N.same_report_path, report_paths.same_report_path)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "carved")
            os.mkdir(src)
            payload = os.path.join(src, "0001.bin")
            index_path = os.path.join(src, "index.json")
            names = os.path.join(tmp, "names.txt")
            with open(payload, "wb") as f:
                f.write(sample())
            with open(index_path, "w", encoding="utf-8", newline="\n") as f:
                f.write('{"files": []}\n')
            with open(names, "w", encoding="utf-8", newline="\n") as f:
                f.write("prop_crate_01\n")

            folded_names = os.path.join(tmp, "NAMES.TXT")
            with (mock.patch.object(report_paths, "path_key", side_effect=str),
                  mock.patch.object(report_paths.os.path, "samefile", side_effect=OSError("not same")),
                  mock.patch.object(report_paths, "same_name_on_folding_volume", return_value=True) as probe,
                  mock.patch.object(sys, "argv", ["psin_names.py", src, "--names", names, "--refs", folded_names])):
                with self.assertRaises(SystemExit) as cm:
                    N.main()
                self.assertEqual(cm.exception.code, 2)
            self.assertGreaterEqual(probe.call_count, 1)

            index_link = os.path.join(tmp, "index_alias.tsv")
            try:
                os.link(index_path, index_link)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"hard links unavailable: {e}")
            result = self.run_tool(src, "--names", names, "--refs", index_link)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--refs must not overwrite an input", result.stderr)
            with open(index_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"files": []}\n')

    def test_refs_unwritable_parent_dir_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "one.bin")
            names = os.path.join(tmp, "names.txt")
            with open(src, "wb") as f:
                f.write(sample())
            with open(names, "w", encoding="utf-8") as f:
                f.write("prop_crate_01\n")
            missing_parent = os.path.join(tmp, "no_such_dir", "refs.tsv")
            r = self.run_tool(src, "--names", names, "--refs", missing_parent)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertTrue(r.stderr.startswith("ERROR:"), r.stderr)

    def test_missing_names_file_is_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_tool(d, "--names", os.path.join(d, "none.txt"))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)

    def test_names_json_of_another_shape_is_one_line(self):
        """--names x.json that is not what rpf9_names.py writes used to end in AttributeError or TypeError."""
        top_message = "expected rpf9_names.py JSON, an object with a 'files' array"
        row_message = "each 'files' row must be an object whose 'names' is an array of strings"
        cases = [
            ([], top_message),
            ("names", top_message),
            ({"files": None}, top_message),
            ({"files": {"names": ["prop_crate_01"]}}, top_message),
            ({"files": [3]}, row_message),
            ({"files": [{"names": ["prop_crate_01"]}, ["prop_barrel_02"]]}, row_message),
            ({"files": [{"names": "prop_crate_01"}]}, row_message),
            ({"files": [{"names": None}]}, row_message),
            ({"files": [{"names": ["prop_crate_01", 5]}]}, row_message),
        ]
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c")
            os.mkdir(src)
            with open(os.path.join(src, "0001.bin"), "wb") as f:
                f.write(sample())
            good = os.path.join(d, "good.txt")
            with open(good, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(NAMES))
            refs = os.path.join(d, "refs.tsv")
            with open(refs, "w", encoding="utf-8", newline="\n") as f:
                f.write("an earlier report\n")
            for i, (document, message) in enumerate(cases):
                with self.subTest(document=document):
                    bad = os.path.join(d, f"bad{i}.json")
                    with open(bad, "w", encoding="utf-8", newline="\n") as f:
                        json.dump(document, f)
                    # A good list first: every --names file is checked, and the message names the bad one.
                    r = self.run_tool(src, "--names", good, "--names", bad, "--refs", refs)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertEqual(r.stderr, f"ERROR: {bad}: {message}\n")
                    self.assertEqual(r.stdout, "")
                    with open(refs, encoding="utf-8") as f:
                        self.assertEqual(f.read(), "an earlier report\n")

    def test_names_file_that_is_not_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            for i, blob in enumerate((b"", b"{not json", b"\xff\xfe{}")):
                with self.subTest(blob=blob):
                    bad = os.path.join(d, f"bad{i}.json")
                    with open(bad, "wb") as f:
                        f.write(blob)
                    r = self.run_tool(d, "--names", bad)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    lines = r.stderr.splitlines()
                    self.assertEqual(len(lines), 1, r.stderr)
                    self.assertTrue(lines[0].startswith(f"ERROR: {bad}: not JSON: "), r.stderr)

    def test_names_json_without_files_or_names_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, document in enumerate(({}, {"summary": {}}, {"files": []}, {"files": [{}, {"names": []}]},
                                          {"files": [{"names": ["prop_crate_01.#dr"]}]})):
                paths.append(os.path.join(d, f"n{i}.json"))
                with open(paths[-1], "w", encoding="utf-8", newline="\n") as f:
                    json.dump(document, f)
            self.assertEqual(N.load_names(paths), {"prop_crate_01"})


if __name__ == "__main__":
    unittest.main()
