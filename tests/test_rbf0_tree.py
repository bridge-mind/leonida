"""Synthetic-tree tests for tools/rbf0_tree.py.

    python -I tests/test_rbf0_tree.py

Every byte here is built by the test. No game file, game asset path or game name is used.
"""
import contextlib
import csv
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_tree  # noqa: E402
import rbf0_index  # noqa: E402

WIDE = b"RBF0" + rbf0_tree.PROLOGUE
CLOSE = b"\xff\xff"


def _name(name):
    return struct.pack("<H", len(name)) + name


def _open(index, name=None, attrs=0):
    out = bytes([index, 0x00])
    if name is not None:
        out += _name(name)
    return out + struct.pack("<HHH", 0, 0, attrs)


def _value(index, kind, payload=b"", name=None):
    out = bytes([index, kind])
    if name is not None:
        out += _name(name)
    return out + payload


def _text(text, nul=True):
    raw = text + (b"\x00" if nul else b"")
    return b"\xfd\xff" + struct.pack("<I", len(raw)) + raw


def wide_blob():
    """Root -> Item(attrs=2) -> Child(text), then a value after the child."""
    b = bytearray(WIDE)
    b += _open(0, b"TestRoot")
    b += _open(1, b"Item", attrs=2)
    b += _value(2, 0x10, struct.pack("<Q", 7), b"count")
    b += _value(3, 0x60, struct.pack("<H", 5) + b"alpha", b"label")
    b += _open(4, b"Child")
    b += _text(b"body")
    b += CLOSE
    b += _value(5, 0x40, struct.pack("<d", 0.5), b"ratio")
    b += CLOSE
    b += _open(6, b"Empty")
    b += CLOSE
    b += CLOSE
    return bytes(b)


def narrow_blob():
    b = bytearray(b"RBF0")
    b += _open(0, b"TestRoot")
    b += _value(1, 0x10, struct.pack("<I", 7), b"count")
    b += _value(2, 0x40, struct.pack("<f", 0.5), b"ratio")
    b += CLOSE
    return bytes(b)


def other_blob():
    b = bytearray(WIDE)
    b += _open(0, b"OtherRoot")
    b += _value(1, 0x20, b"", b"flag")
    b += CLOSE
    return bytes(b)


class ParseTest(unittest.TestCase):
    def test_tree_offsets_and_nesting(self):
        blob = wide_blob()
        doc = rbf0_tree.parse_resource(blob)
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["root_name"], "TestRoot")
        self.assertEqual(doc["layout"], "wide-64")
        self.assertEqual(doc["size"], len(blob))
        self.assertEqual(doc["end"], len(blob))
        self.assertEqual((doc["counts"]["elements"], doc["counts"]["values"], doc["counts"]["texts"]),
                         (4, 3, 1))
        root = doc["tree"]
        self.assertEqual(root["offset"], 8)
        self.assertEqual(root["header_offset"], 20)
        self.assertEqual(root["end"], len(blob))
        self.assertEqual((root["children"], root["content_count"], root["attrs_actual"]), (2, 2, 0))
        item = root["content"][0]
        self.assertEqual(item["name"], "Item")
        self.assertEqual(item["offset"], 26)
        self.assertEqual(item["header_offset"], 34)
        self.assertEqual(item["depth"], 2)
        self.assertEqual(item["attrs_declared"], 2)
        self.assertEqual((item["children"], item["content_count"], item["attrs_actual"]), (1, 2, 2))
        self.assertEqual([a["name"] for a in item["attributes"]], ["count", "label"])
        self.assertEqual([a["value"] for a in item["attributes"]], [7, "alpha"])
        self.assertEqual([n["kind"] for n in item["content"]], ["element", "value"])
        child = item["content"][0]
        self.assertEqual(child["name"], "Child")
        self.assertEqual(child["depth"], 3)
        self.assertEqual((child["children"], child["content_count"], child["attrs_actual"]), (0, 1, 0))
        self.assertEqual(child["content"][0]["kind"], "text")
        self.assertEqual(child["content"][0]["depth"], 4)
        self.assertEqual(child["content"][0]["text"], "body")
        self.assertTrue(child["content"][0]["nul_terminated"])
        self.assertEqual(item["content"][1]["name"], "ratio")
        self.assertEqual(item["content"][1]["value"], 0.5)
        self.assertTrue(item["value_after_child"])
        empty = root["content"][1]
        self.assertEqual(empty["name"], "Empty")
        self.assertEqual((empty["children"], empty["content_count"], empty["attrs_actual"]), (0, 0, 0))
        self.assertEqual(empty["content"], [])
        self.assertEqual(empty["end"], len(blob) - 2)
        self.assertEqual(doc["attribute_mismatches"], 0)
        self.assertEqual(doc["attribute_order_violations"], 0)
        self.assertEqual(doc["max_depth"], 3)  # elements only; the text child is at element depth 4

    def test_siblings_and_empty_element(self):
        doc = rbf0_tree.parse_resource(wide_blob())
        root = doc["tree"]
        siblings = [c for c in root["content"] if c["kind"] == "element"]
        self.assertEqual([c["name"] for c in siblings], ["Item", "Empty"])
        self.assertEqual([c["offset"] for c in siblings],
                         sorted(c["offset"] for c in siblings))
        self.assertEqual(root["children"], 2)
        self.assertEqual(siblings[1]["children"], 0)
        self.assertEqual(siblings[1]["content"], [])
        self.assertIsNotNone(siblings[1]["end"])

    def test_index_output_stays_flat(self):
        parsed = rbf0_index.parse_rbf0(wide_blob())
        self.assertEqual(parsed["status"], "ok")
        self.assertNotIn("tree", parsed)

    def test_malformed_close_matches_index_parser(self):
        cases = [
            ("bad close", WIDE + _open(0, b"TestRoot") + b"\xff\x00", "bad", "close tag is not FF FF"),
            ("close without open", WIDE + b"\xff\xff", "bad", "close without an open element"),
            ("bad text tag", WIDE + _open(0, b"TestRoot") + b"\xfd\x00" + struct.pack("<I", 1) + b"\x00",
             "bad", "text tag is not FD FF"),
            ("truncated", WIDE + _open(0, b"TestRoot"), "truncated", "unclosed element(s)"),
        ]
        for label, blob, status, needle in cases:
            with self.subTest(label):
                tree_doc = rbf0_tree.parse_resource(blob)
                index_doc = rbf0_index.parse_rbf0(blob)
                self.assertEqual(tree_doc["status"], status)
                self.assertEqual(index_doc["status"], status)
                self.assertIn(needle, tree_doc["error"])
                self.assertIn(needle, index_doc["error"])

    def test_descriptor_table(self):
        doc = rbf0_tree.parse_resource(wide_blob())
        self.assertEqual(doc["descriptor_count"], 7)
        table = {d["index"]: d for d in doc["descriptors"]}
        self.assertEqual(table[0]["name"], "TestRoot")
        self.assertEqual(table[0]["first_type"], "open")
        self.assertEqual(table[2]["name"], "count")
        self.assertEqual(table[2]["first_type"], "int")
        self.assertEqual(table[3]["name"], "label")
        self.assertEqual(table[3]["first_type"], "string")
        self.assertEqual(table[5]["types"], {"float": 1})
        self.assertEqual(table[3]["uses"], 1)
        self.assertEqual(doc["distinct_descriptor_names"], 7)

    def test_descriptor_is_a_name_index_not_a_type(self):
        b = bytearray(WIDE)
        b += _open(0, b"TestRoot")
        b += _open(1, b"Item", attrs=2)
        b += _value(2, 0x10, struct.pack("<Q", 5), b"value")
        b += _value(2, 0x40, struct.pack("<d", 1.5))
        b += CLOSE
        b += CLOSE
        doc = rbf0_tree.parse_resource(bytes(b))
        self.assertEqual(doc["status"], "ok")
        desc = {d["index"]: d for d in doc["descriptors"]}[2]
        self.assertEqual(desc["name"], "value")
        self.assertEqual(desc["first_type"], "int")
        self.assertEqual(desc["types"], {"int": 1, "float": 1})
        item = doc["tree"]["content"][0]
        self.assertEqual([a["type"] for a in item["attributes"]], ["int", "float"])

    def test_narrow_widths(self):
        doc = rbf0_tree.parse_resource(narrow_blob())
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["layout"], "narrow-32")
        item = doc["tree"]
        self.assertEqual(item["content"][0]["value"], 7)
        self.assertEqual(item["content"][0]["size"], 4)
        self.assertEqual(item["content"][1]["value"], 0.5)
        self.assertEqual(item["content"][1]["size"], 4)

    def test_text_and_string_framing(self):
        doc = rbf0_tree.parse_resource(wide_blob())
        item = doc["tree"]["content"][0]
        label = item["attributes"][1]
        self.assertEqual(label["type"], "string")
        self.assertEqual(label["byte_length"], 5)
        self.assertFalse(label["nul_terminated"])
        self.assertEqual(label["payload_offset"] + label["size"], label["payload_offset"] + 5)
        text = item["content"][0]["content"][0]
        self.assertEqual(text["length"], 5)
        self.assertTrue(text["nul_terminated"])
        self.assertEqual(text["payload_offset"] + text["length"], text["offset"] + 2 + 4 + 5)

    def test_trailing_bytes(self):
        blob = wide_blob() + b"\x00\x00"
        doc = rbf0_tree.parse_resource(blob)
        self.assertEqual(doc["status"], "trailing")
        self.assertEqual(doc["tree"]["end"], len(blob) - 2)
        self.assertIn("after the root closed", doc["error"])

    def test_truncated(self):
        doc = rbf0_tree.parse_resource(wide_blob()[:30])
        self.assertEqual(doc["status"], "truncated")
        self.assertIn("truncated", doc["error"])

    def test_unknown_descriptor_fails_closed(self):
        b = WIDE + _open(0, b"TestRoot") + bytes([9, 0x10]) + struct.pack("<Q", 1) + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("descriptor index 9 of 1", doc["error"])
        self.assertEqual(doc["descriptor_count"], 1)
        self.assertEqual(doc["tree"]["name"], "TestRoot")
        self.assertEqual(doc["tree"]["content"], [])
        self.assertEqual(doc["counts"].get("values", 0), 0)

    def test_unknown_record_type_fails_closed(self):
        b = WIDE + _open(0, b"TestRoot") + bytes([1, 0x99]) + _name(b"weird") + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("unknown record type 0x99", doc["error"])
        self.assertEqual([d["name"] for d in doc["descriptors"]], ["TestRoot", "weird"])
        self.assertEqual(doc["tree"]["content"], [])

    def test_bad_element_word_fails_closed(self):
        b = WIDE + bytes([0, 0x00]) + _name(b"TestRoot") + struct.pack("<HHH", 1, 0, 0) + CLOSE
        doc = rbf0_tree.parse_resource(b)
        self.assertEqual(doc["status"], "bad")
        self.assertIn("word is not zero", doc["error"])
        self.assertIsNone(doc["tree"])

    def test_string_keeps_a_trailing_nul(self):
        b = WIDE + _open(0, b"TestRoot") + _value(1, 0x60, struct.pack("<H", 2) + b"a\x00", b"label") + CLOSE
        doc = rbf0_tree.parse_resource(b)
        value = doc["tree"]["content"][0]
        self.assertEqual(value["value"], "a\x00")
        self.assertTrue(value["nul_terminated"])
        self.assertEqual(value["byte_length"], 2)

    def test_non_finite_float_is_json_safe(self):
        b = WIDE + _open(0, b"TestRoot") + _value(1, 0x40, struct.pack("<d", float("nan")), b"ratio") + CLOSE
        doc = rbf0_tree.parse_resource(b)
        value = doc["tree"]["content"][0]
        self.assertIsNone(value["value"])
        self.assertEqual(value["float_special"], "nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tree.json")
            rbf0_tree.write_json(path, [doc], {"resources": 1})
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
        self.assertIsNone(loaded["resources"][0]["tree"]["content"][0]["value"])

    def test_not_rbf0(self):
        with self.assertRaises(rbf0_tree.TreeError):
            rbf0_tree.parse_resource(b"PSIN" + bytes(16))


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_join_at_64k(self):
        b = bytearray(WIDE)
        b += _open(0, b"BigRoot")
        filler = b"x" * 4000
        b += _value(1, 0x60, struct.pack("<H", len(filler)) + filler, b"filler")
        while len(b) < 70000:
            b += _value(1, 0x60, struct.pack("<H", len(filler)) + filler)
        b += CLOSE
        blob = bytes(b)
        self.assertGreater(len(blob), 65536)
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(blob[:65536])
        with open(os.path.join(self.root, "00010000.bin"), "wb") as f:
            f.write(blob[65536:])
        docs, summary = rbf0_tree.decode_source(self.root)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["chunks"], 2)
        self.assertEqual(docs[0]["size"], len(blob))
        self.assertEqual(docs[0]["status"], "ok")
        self.assertEqual(docs[0]["root_name"], "BigRoot")
        self.assertEqual(summary["resources"], 1)
        self.assertEqual(summary["ok"], 1)

    def test_single_resource_file(self):
        path = os.path.join(self.root, "resource.bin")
        with open(path, "wb") as f:
            f.write(wide_blob())
        docs, summary = rbf0_tree.decode_source(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["root_name"], "TestRoot")
        self.assertEqual(summary["ok"], 1)

    def test_source_without_rbf0(self):
        path = os.path.join(self.root, "not-rbf0.bin")
        with open(path, "wb") as f:
            f.write(b"PSIN" + bytes(16))
        with self.assertRaises(SystemExit):
            rbf0_tree.decode_source(path)


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_xml_preserves_carriage_returns_in_text(self) -> None:
        source = os.path.join(self.root, "text.bin")
        expected = "first\rsecond\r\n<last & final>"
        with open(source, "wb") as f:
            f.write(WIDE + _open(0, b"TestRoot") + _text(expected.encode("utf-8")) + CLOSE)
        out = os.path.join(self.root, "tree.json")
        xml_dir = os.path.join(self.root, "xml")

        self.assertEqual(rbf0_tree.main([source, out, "--xml", xml_dir, "--quiet"]), 0)

        tree = ET.parse(os.path.join(xml_dir, "00000000.xml"))
        self.assertEqual(tree.findtext(".//text"), expected)

    def test_json_and_xml(self):
        blob = bytearray(WIDE)
        blob += _open(0, b"TestRoot", attrs=1)
        blob += _value(1, 0x60, struct.pack("<H", 3) + b"a &", b"xmlns:xi")
        blob += _open(2, b"Text")
        blob += _text(b"a < b & c")
        blob += CLOSE
        blob += _open(3, b"Ctrl")
        blob += _text(b"a\x01b")
        blob += CLOSE
        blob += CLOSE
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(blob)
        out = os.path.join(self.root, "tree.json")
        xml_dir = os.path.join(self.root, "xml")
        self.assertEqual(rbf0_tree.main([self.root, out, "--xml", xml_dir, "--quiet"]), 0)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["resources"], 1)
        self.assertEqual(doc["summary"]["elements"], 3)
        self.assertEqual(doc["resources"][0]["tree"]["attributes"][0]["name"], "xmlns:xi")
        xml_path = os.path.join(xml_dir, "00000000.xml")
        self.assertTrue(os.path.exists(xml_path))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.assertEqual(root.tag, "rbf0")
        attr = root.find(".//attr")
        self.assertEqual(attr.get("name"), "xmlns:xi")
        self.assertEqual(attr.text, "a &")
        texts = root.findall(".//text")
        self.assertEqual(texts[0].text, "a < b & c")
        self.assertEqual(texts[1].text, "a\ufffdb")  # XML 1.0 control bytes are replaced, not emitted
        rbf0_tree.write_xml(xml_dir, doc["resources"])  # a rerun overwrites, it does not add -0000 duplicates
        self.assertEqual(len(os.listdir(xml_dir)), 1)

    def test_csv_distinguishes_trees_with_the_same_offset_and_root(self):
        names = ["00000100-left.bin", "00000100-right.bin"]
        for name in names:
            with open(os.path.join(self.root, name), "wb") as f:
                f.write(wide_blob())
        out = os.path.join(self.root, "tree.json")
        table = os.path.join(self.root, "nodes.csv")
        self.assertEqual(rbf0_tree.main([self.root, out, "--csv", table, "--quiet"]), 0)
        with open(table, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({row["resource"] for row in rows}, {"0x100"})
        self.assertEqual({row["source"] for row in rows}, set(names))
        for name in names:
            nodes = [row for row in rows if row["source"] == name]
            self.assertEqual(len(nodes), 8)
            self.assertEqual({node["kind"] for node in nodes}, {"element", "value", "text"})

    def test_duplicate_offsets_keep_each_source(self):
        names = ["000000000100-left.bin", "000000000100-right.bin"]
        roots = ["RootA", "RootB"]
        for name, root in zip(names, roots):
            with open(os.path.join(self.root, name), "wb") as f:
                f.write(b"RBF0" + _open(0, root.encode("ascii")) + CLOSE)
        out = os.path.join(self.root, "tree.json")
        self.assertEqual(rbf0_tree.main([self.root, out, "--quiet"]), 0)
        with open(out, encoding="utf-8") as f:
            resources = json.load(f)["resources"]
        self.assertEqual([(doc["source"], doc["root_name"]) for doc in resources], list(zip(names, roots)))

    def test_deep_tree_preserves_existing_json(self):
        depth = 600
        blob = b"RBF0" + _open(0, b"TestRoot") + _open(0) * (depth - 1) + CLOSE * depth
        doc = rbf0_tree.parse_resource(blob)
        self.assertEqual((doc["status"], doc["max_depth"]), ("ok", depth))
        source = os.path.join(self.root, "deep.bin")
        with open(source, "wb") as f:
            f.write(blob)
        out = os.path.join(self.root, "tree.json")
        previous = b'{"previous": true}\n'
        with open(out, "wb") as f:
            f.write(previous)
        tool = os.path.join(ROOT, "tools", "rbf0_tree.py")
        result = subprocess.run([sys.executable, "-I", tool, source, out, "--quiet"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        with open(out, "rb") as f:
            self.assertTrue(f.read() == previous, "existing JSON report changed")
        self.assertEqual(len(result.stderr.splitlines()), 1)
        self.assertIn(os.path.basename(out), result.stderr)
        self.assertIn("depth", result.stderr)
        self.assertCountEqual(os.listdir(self.root), ["deep.bin", "tree.json"])

    def test_error_lines_name_the_path_as_typed(self):
        """repr() doubled every Windows backslash in the one line an error printed; a POSIX path stays the same."""
        source = os.path.join(self.root, "deep.bin")
        with open(source, "wb") as f:
            f.write(WIDE + _open(0, b"TestRoot") + CLOSE)
        out = os.path.join(self.root, "tree.json")
        cases = (
            ("decode_source", RecursionError(), source, "tree exceeds the export depth limit"),
            ("write_json", OSError(28, "No space left on device"), out, "No space left on device"),
        )
        for failure_in, failure, path, text in cases:
            with self.subTest(failure_in):
                stderr = io.StringIO()
                with patch.object(rbf0_tree, failure_in, side_effect=failure), \
                        contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(rbf0_tree.main([source, out, "--quiet"]), 1)
                self.assertEqual(stderr.getvalue(), f"{path}: {text}\n")
        self.assertEqual(os.listdir(self.root), ["deep.bin"])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_report_keeps_the_usual_permission_bits(self):
        out = os.path.join(self.root, "tree.json")
        old = os.umask(0o022)
        try:
            rbf0_tree.write_json(out, [], {"resources": 0})
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o644)
            os.chmod(out, 0o640)
            rbf0_tree.write_json(out, [], {"resources": 0})
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)
        finally:
            os.umask(old)

    def test_failed_replace_of_read_only_report_removes_writable_temp(self):
        out = os.path.join(self.root, "tree.json")
        previous = b'{"previous": true}\n'
        with open(out, "wb") as f:
            f.write(previous)
        os.chmod(out, 0o444)
        self.addCleanup(lambda: os.path.exists(out) and os.chmod(out, 0o600))
        made_writable = []
        real_chmod = os.chmod

        def chmod(path, mode, *args, **kwargs):
            if isinstance(path, str) and os.path.basename(path).startswith(".rbf0_tree-") and mode == 0o600:
                made_writable.append(path)
            return real_chmod(path, mode, *args, **kwargs)

        with patch.object(rbf0_tree.os, "replace", side_effect=OSError("replace denied")), \
                patch.object(rbf0_tree.os, "chmod", chmod):
            with self.assertRaises(OSError):
                rbf0_tree.write_json(out, [], {"resources": 0})
        self.assertEqual(len(made_writable), 1)
        with open(out, "rb") as f:
            self.assertEqual(f.read(), previous)
        self.assertEqual([n for n in os.listdir(self.root) if n.startswith(".rbf0_tree-")], [])

    def _run_cli(self, *argv):
        tool = os.path.join(ROOT, "tools", "rbf0_tree.py")
        return subprocess.run([sys.executable, "-I", tool, *argv, "--quiet"], capture_output=True, text=True)

    def test_out_or_csv_over_the_source_file_is_refused(self):
        source = os.path.join(self.root, "resource.bin")
        blob = wide_blob()
        with open(source, "wb") as f:
            f.write(blob)
        other = os.path.join(self.root, "tree.json")
        for argv in ((source, os.path.join(self.root, ".", "resource.bin")),
                     (source, other, "--csv", source)):
            with self.subTest(argv=argv):
                result = self._run_cli(*argv)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("would overwrite the source", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                with open(source, "rb") as f:
                    self.assertEqual(f.read(), blob)
        self.assertFalse(os.path.exists(other))

    def test_out_csv_or_xml_over_a_stream_of_the_source_folder_is_refused(self):
        stream = os.path.join(self.root, "00000000.bin")
        blob = wide_blob()
        with open(stream, "wb") as f:
            f.write(blob)
        other = os.path.join(self.root, "reports", "tree.json")
        cases = (((self.root, os.path.join(self.root, "00000000.BIN" if os.name == "nt" else "00000000.bin")),
                  "would overwrite the stream"),
                 ((self.root, other, "--csv", stream), "would overwrite the stream"),
                 ((self.root, other, "--xml", os.path.join(self.root, ".")), "must not be the source folder"))
        for argv, message in cases:
            with self.subTest(argv=argv):
                result = self._run_cli(*argv)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                with open(stream, "rb") as f:
                    self.assertEqual(f.read(), blob)
                self.assertEqual(sorted(os.listdir(self.root)), ["00000000.bin"])

    def test_reports_beside_the_streams_are_still_allowed(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        out = os.path.join(self.root, "tree.json")
        result = self._run_cli(self.root, out, "--csv", os.path.join(self.root, "nodes.csv"),
                               "--xml", os.path.join(self.root, "xml"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.isfile(out))

    def test_out_and_csv_on_the_same_file_are_refused(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        out = os.path.join(self.root, "reports", "tree.json")
        result = self._run_cli(self.root, out, "--csv", out)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("must be different files", result.stderr)
        self.assertFalse(os.path.exists(out))

    def test_root_filter_and_limit(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        with open(os.path.join(self.root, "00001000.bin"), "wb") as f:
            f.write(other_blob())
        docs, summary = rbf0_tree.decode_source(self.root, roots=["OtherRoot"])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["root_name"], "OtherRoot")
        self.assertEqual(summary["roots"], {"OtherRoot": 1})
        docs, _ = rbf0_tree.decode_source(self.root, limit=1)
        self.assertEqual(len(docs), 1)

    def test_decode_source_validates_selection_before_reading(self):
        for settings, message in (({"limit": 0}, "--limit must be at least 1"),
                                  ({"limit": -1}, "--limit must be at least 1"),
                                  ({"offset": -1}, "--offset must be non-negative")):
            with self.subTest(settings=settings), patch.object(
                    rbf0_tree, "_rows", side_effect=AssertionError("invalid selection reached the source reader")):
                with self.assertRaisesRegex(ValueError, message):
                    rbf0_tree.decode_source(self.root, **settings)

    def test_bad_resource_exit_code(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(WIDE + _open(0, b"TestRoot") + bytes([9, 0x10]) + struct.pack("<Q", 1))
        out = os.path.join(self.root, "tree.json")
        self.assertEqual(rbf0_tree.main([self.root, out, "--quiet"]), 1)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["bad"], 1)

    def test_cli_rejects_invalid_selection_before_writing(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        tool = os.path.join(ROOT, "tools", "rbf0_tree.py")
        out = os.path.join(self.root, "tree.json")
        for option, value in (("--limit", "0"), ("--limit", "-1"), ("--offset", "-1")):
            with self.subTest(option=option, value=value):
                result = subprocess.run([sys.executable, "-I", tool, self.root, out, option, value],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(option, result.stderr)
                self.assertFalse(os.path.exists(out))

    def test_summary_tree_shape(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        _, summary = rbf0_tree.decode_source(self.root)
        self.assertEqual(summary["nodes"], 8)
        self.assertEqual(summary["elements_by_depth"], {"1": 1, "2": 2, "3": 1})
        self.assertEqual(summary["children_histogram"], {"0": 2, "1": 1, "2": 1})
        self.assertEqual(summary["content_length_histogram"], {"0": 1, "1": 1, "2": 2})
        self.assertEqual(summary["attribute_count_histogram"], {"0": 3, "2": 1})
        self.assertEqual(summary["leaf_elements"], 2)
        self.assertEqual(summary["empty_elements"], 1)
        self.assertEqual(summary["text_only_elements"], 1)
        self.assertEqual(summary["parent_child_edges"], 3)
        self.assertEqual(summary["max_children"], 2)
        self.assertEqual(summary["mean_children"], 0.75)

    def test_csv_node_table(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        out = os.path.join(self.root, "tree.json")
        csv_path = os.path.join(self.root, "nodes.csv")
        self.assertEqual(rbf0_tree.main([self.root, out, "--csv", csv_path, "--quiet"]), 0)
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        root = rows[0]
        self.assertEqual((root["kind"], root["name"], root["parent"], root["children"],
                          root["content_count"]), ("element", "TestRoot", "", "2", "2"))
        item = next(r for r in rows if r["name"] == "Item")
        self.assertEqual((item["slot"], item["sibling"], item["child_index"]), ("content", "0", "0"))
        self.assertEqual(item["parent"], "0x8")
        self.assertEqual((item["attrs"], item["children"]), ("2", "1"))
        count = next(r for r in rows if r["slot"] == "attribute" and r["name"] == "count")
        self.assertEqual(count["parent"], item["offset"])
        self.assertEqual((count["type"], count["value"]), ("int", "7"))
        text = next(r for r in rows if r["kind"] == "text")
        self.assertEqual((text["value"], text["nul"], text["utf8"]), ("body", "1", "1"))
        empty = next(r for r in rows if r["name"] == "Empty")
        self.assertEqual((empty["children"], empty["content_count"]), ("0", "0"))

    def test_directory_skips_json_csv_reports_and_oversized_continuation(self):
        b = bytearray(b"RBF0" + rbf0_tree.PROLOGUE)
        b += bytes([0, 0x00]) + struct.pack("<H", 4) + b"Root" + struct.pack("<HHH", 0, 0, 0)
        filler = b"x" * 4000
        first = True
        while len(b) < 70000:
            b += bytes([1, 0x60]) + ((struct.pack("<H", 6) + b"filler") if first else b"")
            b += struct.pack("<H", len(filler)) + filler
            first = False
        first_chunk = bytes(b[:rbf0_tree.MAX_CHUNK])
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(first_chunk)
        for report in ("index.json", "nodes.CSV"):
            with open(os.path.join(self.root, report), "w", encoding="utf-8", newline="\n") as f:
                f.write("{\"files\": []}\n")
        docs, _ = rbf0_tree.decode_source(self.root)
        self.assertEqual((docs[0]["chunks"], docs[0]["size"], docs[0]["status"]),
                         (1, rbf0_tree.MAX_CHUNK, "truncated"))
        # A stream larger than 64 KiB is not a split continuation chunk, matching rbf0_index.join_resource.
        with open(os.path.join(self.root, "00010000.bin"), "wb") as f:
            f.write(b"\x00" * (rbf0_tree.MAX_CHUNK + 1))
        docs, _ = rbf0_tree.decode_source(self.root)
        self.assertEqual((docs[0]["chunks"], docs[0]["size"]), (1, rbf0_tree.MAX_CHUNK))


if __name__ == "__main__":
    unittest.main()
