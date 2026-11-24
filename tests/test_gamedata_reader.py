"""gamedata_reader: the generic conversion of RAGE parser XML, root/type dispatch, streaming and the CLI (no game files).

    python -I tests/test_gamedata_reader.py

Every document is written here by hand, in the conventions docs/gamedata.md describes for common.rpf.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import gamedata_reader as R  # noqa: E402

DOC = b"""\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>
<CSyntheticRoot version="3">
  <enabled value="true"/>
  <count value="12"/>
  <mask value="0x1F"/>
  <scale value="1.5"/>
  <bias value="-2.5f"/>
  <limit value="inf"/>
  <label>synthetic_name</label>
  <empty/>
  <offset x="1.0" y="2.5" z="-3"/>
  <tint x="0.5" y="0.25" z="0" w="1"/>
  <ids content="short_array">
    1 2 3
    4 5
  </ids>
  <weights content="float_array">0.5 1.0 2.0</weights>
  <uvs content="vector2_array">0 1 2 3</uvs>
  <basis content="matrix34">
    1 0 0
    0 1 0
    0 0 1
    4 5 6
  </basis>
  <items>
    <Item type="CThing"><size value="2"/></Item>
    <Item type="NULL"/>
    <Item><size value="3"/></Item>
  </items>
  <lookup>
    <Item key="alpha"><v value="1"/></Item>
    <Item key="beta"><v value="2"/></Item>
  </lookup>
  <target ref="other_entry"/>
  <entry name="first" userFlags="3"><v value="1"/></entry>
  <entry name="second" userFlags="0"><v value="2"/></entry>
</CSyntheticRoot>
"""

WANT = {
    "@version": 3, "enabled": True, "count": 12, "mask": 31, "scale": 1.5, "bias": -2.5, "limit": float("inf"),
    "label": "synthetic_name", "empty": "", "offset": (1.0, 2.5, -3.0), "tint": (0.5, 0.25, 0.0, 1.0),
    "ids": [1, 2, 3, 4, 5], "weights": [0.5, 1, 2.0], "uvs": [(0, 1), (2, 3)],
    "basis": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (4, 5, 6)],
    "items": [{"$type": "CThing", "size": 2}, None, {"size": 3}],
    "lookup": {"alpha": {"v": 1}, "beta": {"v": 2}},
    "target": {"$ref": "other_entry"},
    "entry": [{"v": 1, "@name": "first", "@userFlags": 3}, {"v": 2, "@name": "second", "@userFlags": 0}],
}


class Conventions(unittest.TestCase):
    def test_every_convention(self):
        reader = R.Reader(root_handlers={}, type_handlers={})
        tag, typ, value = reader.read(DOC)
        self.assertEqual((tag, typ), ("CSyntheticRoot", None))
        self.assertEqual(value, WANT)
        self.assertEqual(reader.types, {"CThing": 1})
        for kind in ("scalar", "vector", "array", "list", "map", "null", "object", "reference", "struct", "text",
                     "extra attributes"):
            self.assertGreater(reader.stats[kind], 0, kind)

    def test_streaming_equals_whole_tree(self):
        # read() converts each top-level field as it ends and clears it; the result is the whole-tree conversion
        reader = R.Reader(root_handlers={}, type_handlers={})
        whole = reader.convert(ET.fromstring(DOC[3:]))
        self.assertEqual(reader.read(DOC)[2], whole)

    def test_scalars(self):
        for text, want in (("true", True), ("false", False), ("7", 7), ("-7", -7), ("0xff", 255), ("1e3", 1000.0),
                           ("2.0f", 2.0), ("-inf", float("-inf")), ("abc", "abc"), ("1.2.3", "1.2.3")):
            self.assertEqual(R.parse_scalar(text), want, text)
            self.assertIs(type(R.parse_scalar(text)), type(want), text)

    def test_vector_array_must_be_whole(self):
        with self.assertRaises(ValueError):
            R.parse_content("vec3v_array", "1 2 3 4")

    def test_polymorphic_root_and_list_root(self):
        reader = R.Reader(root_handlers={}, type_handlers={})
        self.assertEqual(reader.read(b'<R type="CRootType"><a value="1"/></R>'), ("R", "CRootType",
                                                                                  {"$type": "CRootType", "a": 1}))
        self.assertEqual(reader.read(b'<R><Item value="1"/><Item value="2"/></R>')[2], [1, 2])
        self.assertEqual(reader.read(b'<R>leaf</R>')[2], "leaf")

    def test_vector_with_extra_attributes_and_non_vector_xyzw(self):
        reader = R.Reader(root_handlers={}, type_handlers={})
        self.assertEqual(
            reader.read(b'<R><offset x="1.0" y="2.5" z="-3" name="head" userFlags="3"/></R>')[2],
            {"offset": {"$value": (1.0, 2.5, -3.0), "@name": "head", "@userFlags": 3}},
        )
        self.assertEqual(reader.stats["vector"], 1)
        self.assertEqual(reader.stats["extra attributes"], 1)

        self.assertEqual(
            reader.read(b'<R><node x="label" y="1.0"/></R>')[2],
            {"node": {"@x": "label", "@y": 1.0}},
        )
        self.assertEqual(
            reader.read(b'<R><node y="2.0" z="3.0"/></R>')[2],
            {"node": {"@y": 2.0, "@z": 3.0}},
        )
        self.assertEqual(
            reader.read(b'<R><s value="10" x="2"/></R>')[2],
            {"s": {"$value": 10, "@x": 2}},
        )

    def test_streaming_drops_top_level_children_from_root(self):
        orig_iterparse = ET.iterparse
        roots_seen = []
        max_children = 0

        def spy_iterparse(*args, **kwargs):
            nonlocal max_children
            for event, el in orig_iterparse(*args, **kwargs):
                if event == "start" and not roots_seen:
                    roots_seen.append(el)
                elif roots_seen:
                    max_children = max(max_children, len(roots_seen[0]))
                yield event, el

        with mock.patch.object(ET, "iterparse", side_effect=spy_iterparse):
            R.Reader(root_handlers={}, type_handlers={}).read(DOC)
        self.assertEqual(len(roots_seen), 1)
        self.assertEqual(len(roots_seen[0]), 0)

        roots_seen.clear()
        max_children = 0
        count = 1000
        large_xml = (
            b"<CLargeRoot>"
            + b"".join(f'<Item><idx value="{i}"/><pos x="{i}.0" y="1.0" z="2.0"/></Item>'.encode("ascii")
                       for i in range(count))
            + b"</CLargeRoot>"
        )
        reader = R.Reader(root_handlers={}, type_handlers={})
        with mock.patch.object(ET, "iterparse", side_effect=spy_iterparse):
            tag, _, items = reader.read(large_xml)
        self.assertEqual(tag, "CLargeRoot")
        self.assertEqual(len(items), count)
        self.assertEqual([item["idx"] for item in items], list(range(count)))
        self.assertEqual(reader.stats["list"], 1)
        self.assertEqual(len(roots_seen[0]), 0)
        self.assertLess(max_children, count)


class Dispatch(unittest.TestCase):
    def test_type_handler(self):
        calls = []

        def thing(el, reader):
            calls.append(el.tag)
            generic = reader.convert(el, use_type_handler=False)
            return ("thing", generic["size"])

        reader = R.Reader(root_handlers={}, type_handlers={"CThing": thing})
        value = reader.read(DOC)[2]
        self.assertEqual(value["items"], [("thing", 2), None, {"size": 3}])
        self.assertEqual(calls, ["Item"])
        self.assertEqual(reader.stats["type handler"], 1)

    def test_type_handler_does_not_double_count_types_or_extra_attributes(self):
        def thing(el, reader):
            generic = reader.convert(el, use_type_handler=False)
            return ("thing", generic["size"])

        reader = R.Reader(root_handlers={}, type_handlers={"CThing": thing})
        _, _, value = reader.read(b'<Root><Item type="CThing" name="first"><size value="2"/></Item></Root>')
        self.assertEqual(value, [{"$value": ("thing", 2), "@name": "first"}])
        self.assertEqual(reader.types, {"CThing": 1})
        self.assertEqual(reader.stats["type handler"], 1)
        self.assertEqual(reader.stats["extra attributes"], 1)
        self.assertEqual(reader.stats["object"], 1)

        direct_reader = R.Reader(root_handlers={}, type_handlers={"CThing": thing})
        direct_val = direct_reader.convert(
            ET.fromstring(b'<Item type="CThing" name="direct"><size value="5"/></Item>'),
            use_type_handler=False,
        )
        self.assertEqual(direct_val, {"$type": "CThing", "size": 5, "@name": "direct"})
        self.assertEqual(direct_reader.types, {"CThing": 1})
        self.assertEqual(direct_reader.stats["type handler"], 0)
        self.assertEqual(direct_reader.stats["extra attributes"], 1)
        self.assertEqual(direct_reader.stats["object"], 1)

    def test_streaming_parity_on_typed_null_and_attribute_roots(self):
        def thing(el, reader):
            generic = reader.convert(el, use_type_handler=False)
            return ("thing", generic["size"])

        cases = (
            (b'<R type="CRootType" version="1"><a value="1"/></R>', {}),
            (b'<R type="CListRoot"><Item value="1"/><Item value="2"/></R>', {}),
            (b'<R type="NULL" name="empty"><a value="1"/></R>', {}),
            (b'<R type="CThing" id="7"><size value="2"/></R>', {"CThing": thing}),
            (b'<R value="10" name="root"><a value="1"/></R>', {}),
            (b'<R content="short_array" name="arr">1 2<a value="1"/></R>', {}),
            (b'<R x="1.0" y="2.0" name="vec"><a value="1"/></R>', {}),
            (b'<R ref="target" name="ref"><a value="1"/></R>', {}),
            (b'<Item type="CThing" x="1.0" y="2.0"/>', {"CThing": lambda el, r: ("vec", r.convert(el, use_type_handler=False))}),
        )
        for raw, handlers in cases:
            with self.subTest(raw=raw):
                whole_reader = R.Reader(root_handlers={}, type_handlers=handlers)
                stream_reader = R.Reader(root_handlers={}, type_handlers=handlers)
                whole_val = whole_reader.convert(ET.fromstring(raw))
                _, _, stream_val = stream_reader.read(raw)
                self.assertEqual(stream_val, whole_val)
                self.assertEqual(stream_reader.stats, whole_reader.stats)
                self.assertEqual(stream_reader.types, whole_reader.types)

    def test_root_handler_gets_the_whole_root(self):
        def root(el, reader):
            return sorted({child.tag for child in el})

        reader = R.Reader(root_handlers={"CSyntheticRoot": root}, type_handlers={})
        tag, _, value = reader.read(DOC)
        self.assertEqual(tag, "CSyntheticRoot")
        self.assertIn("lookup", value)
        self.assertEqual(reader.stats["root handler"], 1)

    def test_decorators_register(self):
        saved_r, saved_t = dict(R.ROOT_HANDLERS), dict(R.TYPE_HANDLERS)
        try:
            @R.register_root("CSyntheticRoot")
            def _root(el, reader):
                return "root"

            @R.register_type("CThing")
            def _thing(el, reader):
                return "thing"

            self.assertEqual(R.Reader().read(DOC)[2], "root")
            self.assertEqual(R.Reader().read(b'<A><Item type="CThing"/></A>')[2], ["thing"])
        finally:
            R.ROOT_HANDLERS.clear(); R.ROOT_HANDLERS.update(saved_r)
            R.TYPE_HANDLERS.clear(); R.TYPE_HANDLERS.update(saved_t)


class Run(unittest.TestCase):
    BAD_DOCUMENTS = (
        ("truncated array", b'<CBroken><v content="vec3v_array">1 2 3 4</v></CBroken>'),
        ("unknown encoding", b'<?xml version="1.0" encoding="x-no-such-encoding"?><CBroken/>'),
        ("unsupported encoding", b'<?xml version="1.0" encoding="UTF-32"?><CBroken/>'),
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "xml")
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "a.xml"), "wb") as f:
            f.write(DOC)
        with open(os.path.join(self.dir, "b.xml"), "wb") as f:
            f.write(b'<COther><x value="1"/>')                  # truncated, as a tail-cut carve can be
        with open(os.path.join(self.dir, "c.xml"), "wb") as f:
            f.write(b'<COther><x value="2"/></COther>')

    def test_summary_counts(self):
        s = R.run([self.dir], reader=R.Reader(root_handlers={}, type_handlers={}))
        self.assertEqual((s["documents"], s["roots"], s["not_well_formed"]), (2, 2, 1))
        self.assertEqual(s["per_root_documents"], {"CSyntheticRoot": 1, "COther": 1})
        self.assertEqual(s["type_classes"], 1)

    def test_root_filter(self):
        s = R.run([self.dir], root_filter="COther", reader=R.Reader(root_handlers={}, type_handlers={}))
        self.assertEqual(s["documents"], 1)

    def test_conversion_and_encoding_errors_do_not_abort_batch(self):
        bad = os.path.join(self.tmp.name, "bad.xml")
        good = os.path.join(self.dir, "c.xml")
        for name, data in self.BAD_DOCUMENTS:
            with self.subTest(name=name):
                with open(bad, "wb") as f:
                    f.write(data)
                s = R.run([bad, good], reader=R.Reader(root_handlers={}, type_handlers={}))
                self.assertEqual((s["documents"], s["roots"], s["not_well_formed"]), (1, 1, 1))
                self.assertEqual(s["per_root_documents"], {"COther": 1})
                self.assertEqual(sum(s["parse_errors"].values()), 1)

    def test_cli_continues_after_conversion_and_encoding_errors(self):
        bad = os.path.join(self.tmp.name, "bad.xml")
        good = os.path.join(self.dir, "c.xml")
        for name, data in self.BAD_DOCUMENTS:
            with self.subTest(name=name):
                with open(bad, "wb") as f:
                    f.write(data)
                out = os.path.join(self.tmp.name, name)
                r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"),
                                    bad, good, "--out", out], capture_output=True, text=True, timeout=10)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("1 documents read, 1 roots, 1 not well-formed", r.stdout)
                self.assertNotIn("Traceback", r.stderr)
                files = os.listdir(out)
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].startswith("COther_"))
                with open(os.path.join(out, files[0]), encoding="utf-8") as f:
                    self.assertEqual(json.load(f), {"x": 2})

    def test_cli_prints_counts_and_writes_json(self):
        out = os.path.join(self.tmp.name, "json", "nested")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"), self.dir, "--out", out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 documents read, 2 roots, 1 not well-formed", r.stdout)
        self.assertNotIn("synthetic_name", r.stdout)                # counts only, no values
        files = sorted(os.listdir(out))
        self.assertEqual(len(files), 2)
        with open(os.path.join(out, next(f for f in files if f.startswith("CSyntheticRoot"))), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["label"], "synthetic_name")

    def test_exports_all_documents_with_the_same_basename(self):
        targets = []
        for index in range(1, 4):
            folder = os.path.join(self.tmp.name, str(index))
            os.mkdir(folder)
            path = os.path.join(folder, "same.xml")
            with open(path, "wb") as f:
                f.write(f'<COther><x value="{index}"/></COther>'.encode("ascii"))
            targets.append(path)
        out = os.path.join(self.tmp.name, "json")
        for _ in range(2):
            summary = R.run(targets, out_dir=out)
            self.assertEqual(summary["documents"], 3)
            files = sorted(os.listdir(out))
            self.assertEqual(len(files), 3)
            values = []
            for name in files:
                with open(os.path.join(out, name), encoding="utf-8") as f:
                    values.append(json.load(f)["x"])
            self.assertEqual(sorted(values), [1, 2, 3])
    def test_cli_fails_when_no_document_can_be_read(self):
        # A corpus with a truncated tail still reads its other documents and exits 0 (the test above); a run that
        # read nothing because every document was broken must not look like an empty read.
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.xml")
            with open(bad, "wb") as f:
                f.write(b'<COther><x value="1"/>')              # truncated, and the only document
            r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"), bad],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn("no document could be read", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_cli_missing_target(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"),
                            os.path.join(self.tmp.name, "nope")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not exist", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def _cli(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "gamedata_reader.py"), *args],
                              capture_output=True, text=True)

    def test_cli_bad_index_json_is_one_error_line(self):
        for name, text in (("notjson", "{bad"), ("wrongshape", "[]"), ("nofiles", "{}")):
            with self.subTest(name):
                d = os.path.join(self.tmp.name, name)
                os.makedirs(d)
                with open(os.path.join(d, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                r = self._cli(d)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                self.assertIn("ERROR: cannot read", r.stderr)
                self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)

    def test_cli_out_that_is_a_file_is_refused(self):
        xml = os.path.join(self.tmp.name, "a.xml")
        out = os.path.join(self.tmp.name, "afile")
        for path, data in ((xml, b"<a/>"), (out, b"x")):
            with open(path, "wb") as f:
                f.write(data)
        r = self._cli(xml, "--out", out)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("not a directory", r.stderr)


if __name__ == "__main__":
    unittest.main()
