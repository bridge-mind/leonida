"""gamedata_index.py must group RPF9 zlib blocks into payloads without game data.

    python -I tests/test_gamedata_index.py

Every fixture here is synthetic: small XML documents, a text list, a binary blob and two archive
magics. The archive test builds a minimal RPF9-shaped file with a zlib stream and zero padding.
"""
import csv
import importlib.util
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
TOOL = os.path.join(ROOT, "tools", "gamedata_index.py")


def load_tool():
    spec = importlib.util.spec_from_file_location("gamedata_index_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunked(data: bytes, first: int = 65536):
    """Split a payload into a 64 KiB block and the remainder, the way RPF9 stores large files."""
    return [data[:first], data[first:]] if len(data) > first else [data]


T = load_tool()


class GamedataIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload, chunks=None):
        parts = chunks if chunks is not None else [payload]
        for part in parts:
            off = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({
                "off": off,
                "stored": len(part),
                "size": len(part),
                "head": part[:16].hex(),
                "pre": "",
            })

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}, f)

    def run_tool(self, target, *args):
        return subprocess.run([sys.executable, "-I", TOOL, target, *args], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_a_resource_continues_after_any_whole_number_of_64k_blocks(self):
        # A 0DRP resource whose first stream is 128 KiB continues in the next stream (docs/page_directory.md);
        # a binary run of the same shape has no header proving that, so it keeps the exact 64 KiB rule.
        page = bytes(T.PAGE_MARKER_AT) + T.PAGE_MARKER
        resource = page + b"\x11" * (2 * 65536 - len(page))
        self.add(resource, [resource, b"\x22" * 1000])
        binary = bytes(range(256)) * 512                        # 128 KiB with no header
        self.add(binary, [binary, bytes(range(255, -1, -1)) * 4])
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([(r["type"], r["chunks"]) for r in data["rows"]],
                         [("other", 2), ("binary", 1), ("binary", 1)])

    def test_rejects_a_shared_csv_and_json_destination(self):
        self.finish()
        out = os.path.join(self.root, "index.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write("existing report")
        alias = os.path.join(self.root, ".", "index.json")

        result = self.run_tool(self.carve, "--csv", out, "--json", alias)

        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), "existing report")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--csv and --json must use different paths", result.stderr)

    def test_a_report_cannot_land_on_a_carved_stream(self):
        # A carve folder holds one <offset>.bin/.bk2 per stream; a report written over one would eat the input.
        self.add(b"<a/>")
        self.finish()
        stream = os.path.join(self.carve, f"{self.rows[0]['off']:012x}.bin")
        with open(stream, "rb") as f:
            before = f.read()
        for flag in ("--json", "--csv"):
            with self.subTest(flag=flag):
                p = self.run_tool(self.carve, flag, stream)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("would overwrite the carved stream", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
        with open(stream, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_bad_targets_are_one_line(self):
        empty = os.path.join(self.root, "empty.rpf")
        open(empty, "wb").close()
        no_index = os.path.join(self.root, "no_index")
        os.mkdir(no_index)
        bad_index = os.path.join(self.root, "bad_index")
        os.mkdir(bad_index)
        with open(os.path.join(bad_index, "index.json"), "w", encoding="utf-8") as f:
            f.write('{"summary": {}}')                     # no "files" list
        cases = ((os.path.join(self.root, "missing.rpf"), "does not exist"), (empty, "empty.rpf"),
                 (no_index, "index.json is missing"), (bad_index, "cannot read"))
        for target, message in cases:
            with self.subTest(target=os.path.basename(target)):
                p = self.run_tool(target, "--json", os.path.join(self.root, "out.json"))
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn(message, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertEqual(p.stdout, "")
        self.assertFalse(os.path.exists(os.path.join(self.root, "out.json")))

    def test_directory_groups_chunks_and_roots(self):
        self.add(b'<?xml version="1.0"?>\n<RootA><!-- synthetic fixture, longer than 64 bytes --><Item value="1"/></RootA>')
        second = b'<?xml version="1.0"?>\n<RootB>' + b"x" * (65536 - len(b'<?xml version="1.0"?>\n<RootB>'))
        self.add(second, chunked(second + b"</RootB>"))
        text = b"line of text\n" * 6000
        self.add(text, chunked(text))
        self.add(b"RBF0" + bytes(60))
        self.add(b"PSIN" + bytes(60))
        self.add(bytes(range(256)) * 2)
        self.add(b"MHIU" + b"\x17" * 100, [b"MHIU" + b"\x17" * 65532, b"\x17" * 100])
        self.add(b"<Item/>\n</TailRoot>")       # no declaration, root inferred from the close tag
        self.add(b"<Item>mid")                  # continuation whose start block was not carved
        self.finish()

        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--csv", os.path.join(self.root, "index.csv"), "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["types"]["xml"]["files"], 4)      # RootA, RootB, TailRoot, fragment
        self.assertEqual(s["types"]["text"]["files"], 1)
        self.assertEqual(s["types"]["rbf0"]["files"], 1)
        self.assertEqual(s["types"]["psin"]["files"], 1)
        self.assertEqual(s["types"]["binary"]["files"], 1)
        self.assertEqual(s["types"]["other"]["files"], 1)
        self.assertEqual(s["xml_documents"], 3)
        self.assertEqual(s["xml_with_header"], 2)
        self.assertEqual(s["xml_tail_only"], 1)
        self.assertEqual(s["xml_incomplete"], 1)
        self.assertEqual(set(s["roots"]), {"RootA", "RootB", "TailRoot"})
        self.assertEqual(len(data["rows"]), 9)
        self.assertEqual([r["root"] for r in data["rows"] if r["type"] == "xml" and r["complete"]],
                         ["RootA", "RootB", "TailRoot"])
        self.assertEqual(data["rows"][1]["chunks"], 2)
        self.assertEqual(data["rows"][2]["chunks"], 2)
        self.assertEqual(data["rows"][6]["detail"], "MHIU")
        self.assertEqual(data["rows"][7]["type"], "xml")
        self.assertTrue(data["rows"][7]["complete"])
        self.assertFalse(data["rows"][7]["header"])
        self.assertFalse(data["rows"][8]["complete"])

        with open(os.path.join(self.root, "index.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["root"], "RootA")
        self.assertEqual(rows[0]["type"], "xml")

    def test_sps1_magic_is_classified_as_other_sps1(self):
        # +0x04 is u32 entry_count; 90 (0x5a = 'Z') reproduces the 5th byte that used to read "SPS1Z".
        sps = bytearray(b"SPS1" + struct.pack("<I", 90) + b"\x05a.sps\x00\x02a\x00\x00" * 90)
        self.add(bytes(sps))
        self.finish()
        out = os.path.join(self.root, "sps1_index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            row = json.load(f)["rows"][0]
        self.assertEqual((row["type"], row["detail"], row["chunks"]), ("other", "SPS1", 1))

    def test_printable_page_directory_stream_is_not_xml(self):
        # A resource stream with the "0DRP" page marker at +16 whose first 4 KiB are over 95% printable and hold a
        # "<": data.rpf's 337 such streams were reported as incomplete XML with no root.
        head = bytes(8) + bytes.fromhex("80ca005000000000") + b"0DRP" + bytes(12)
        page = head + (b"<abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ>" * 80)[:4096 - len(head)]
        page += bytes(512)
        tool = load_tool()
        self.assertGreater(tool.printable(page), 0.95)      # the heuristic alone would call it text
        self.add(page)
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertNotIn("xml", data["summary"]["types"])
        self.assertEqual(data["summary"]["xml_incomplete"], 0)
        self.assertEqual([(r["type"], r["detail"]) for r in data["rows"]], [("other", "0DRP")])

    def test_archive_runs_the_carver_and_reads_roots(self):
        xml = b'<?xml version="1.0"?>\n<RootA><!-- synthetic fixture, longer than 64 bytes --><Item value="1"/></RootA>'
        blob = b"\x00\x01\x02binary\xff" * 32
        start = 16
        body = struct.pack("<I", start)  # RPF9 header: magic + data start
        archive = bytearray(b"9FPR" + body + bytes(8))
        first = zlib.compress(xml)
        archive += first
        archive += bytes(16)
        second = zlib.compress(blob)
        archive += second
        archive += bytes(2 * 1024 * 1024)  # keeps the last stream away from EOF for the carver
        rpf = os.path.join(self.root, "synthetic.rpf")
        with open(rpf, "wb") as f:
            f.write(archive)

        out = os.path.join(self.root, "archive.json")
        p = self.run_tool(rpf, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["streams"], 2)
        self.assertEqual(s["types"]["xml"]["files"], 1)
        self.assertEqual(s["types"]["xml"]["inflated"], len(xml))
        self.assertEqual(s["types"]["binary"]["files"], 1)
        self.assertEqual(s["roots"], {"RootA": {"files": 1, "inflated": len(xml)}})

    def test_unwritable_csv_or_json_is_one_error_line(self):
        self.add(b'<?xml version="1.0"?><RootA/>')
        self.finish()
        blocker = os.path.join(self.root, "blocker")
        with open(blocker, "wb") as f:
            f.write(b"x")
        for flag, name in (("--csv", "out.csv"), ("--json", "out.json")):
            with self.subTest(flag=flag):
                out = os.path.join(blocker, name)
                p = self.run_tool(self.carve, flag, out)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("ERROR: cannot write", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_reads_carve_index_with_utf8_bom(self):
        self.add(b'<?xml version="1.0"?><RootA/>')
        index_path = os.path.join(self.carve, "index.json")
        doc = {"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}
        with open(index_path, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps(doc).encode("utf-8"))
        out = os.path.join(self.root, "bom_out.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["summary"]["xml_documents"], 1)


class XmlClassificationTest(unittest.TestCase):
    """Synthetic XML classification cases; no game data, no files written."""

    def setUp(self):
        self.tool = load_tool()
        self.bom = b"\xef\xbb\xbf"
        self.decl = b'<?xml version="1.0"?>'

    def test_declaration_after_bom_and_whitespace_is_detected(self):
        for space in (b"", b"\n", b"   ", b"\r\n\t"):
            with self.subTest(space=space):
                data = self.bom + space + self.decl + b"\n<Root></Root>"
                self.assertTrue(self.tool.has_xml_decl(data))
                self.assertEqual(self.tool.stream_kind(data, {}), "decl")
                self.assertEqual(self.tool.xml_root(data), "Root")
                self.assertTrue(self.tool.standalone_xml(data))

    def test_bom_only_empty_and_mid_document_declaration_have_no_leading_declaration(self):
        for data in (b"", self.bom, self.bom + b"   ", b"<Root>" + self.decl + b"</Root>"):
            with self.subTest(data=data):
                self.assertFalse(self.tool.has_xml_decl(data))
                self.assertEqual(self.tool.xml_root(data), "Root" if b"<Root>" in data else "")

    def test_nested_same_name_root_is_not_a_closed_document(self):
        for data in (b"<Foo><Foo></Foo>", b"<Foo><Foo></Foo>\n", b"<Foo><Foo>x</Foo>"):
            with self.subTest(data=data):
                self.assertFalse(self.tool.xml_complete("Foo", data))
                self.assertFalse(self.tool.standalone_xml(self.decl + data))

    def test_balanced_and_self_closing_roots_remain_complete(self):
        for data in (b"<Foo></Foo>", b"<Foo>text<Foo>inner</Foo></Foo>",
                     b"<Foo/>", b"  <Foo />\n", b"<Foo>\n</Foo>"):
            with self.subTest(data=data):
                self.assertTrue(self.tool.xml_complete("Foo", data))
        self.assertFalse(self.tool.xml_complete("", b"<Foo></Foo>"))

    def test_quoted_greater_than_in_attributes_does_not_end_a_tag(self):
        for data in (b'<Foo><Foo attr="x>y"/></Foo>', b"<Foo><Foo attr='x>y'/></Foo>",
                     b'<Foo attr="x>y">text</Foo>', b'<Foo attr="x>y"/>'):
            with self.subTest(data=data):
                self.assertTrue(self.tool.xml_complete("Foo", data))
        # A real second open element still leaves the document incomplete.
        self.assertFalse(self.tool.xml_complete("Foo", b'<Foo><Foo attr="x>y"></Foo>'))

    def test_tags_inside_comments_and_cdata_do_not_count(self):
        self.assertTrue(self.tool.xml_complete("Foo", b"<Foo><!-- <Foo> --><![CDATA[</Foo>]]></Foo>"))
        for data in (b"<Foo><Foo><!-- </Foo> -->", b"<Foo><Foo><![CDATA[</Foo>]]>"):
            with self.subTest(data=data):
                self.assertFalse(self.tool.xml_complete("Foo", data))

    def test_self_closing_root_is_standalone_xml(self):
        for data in (b"<Empty/>", b'  <Empty attr="a>b" />\n',
                     self.bom + self.decl + b"\n<!-- c --><Empty/>"):
            with self.subTest(data=data):
                self.assertTrue(self.tool.standalone_xml(data))
        for bad in (b"<Empty><Child/>", b"<Empty/><Other/>", b"<Empty/> trailing"):
            with self.subTest(bad=bad):
                self.assertFalse(self.tool.standalone_xml(bad))


class XmlGroupingRegressionTest(GamedataIndexTest):
    """The classification bugs must also show up in the index the tool writes."""

    def test_bom_with_whitespace_is_grouped_as_an_xml_document(self):
        doc = b"\xef\xbb\xbf\n" + b'<?xml version="1.0"?>' + b"\n<RootBom><Item/></RootBom>"
        self.add(doc)
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["type"], "xml")
        self.assertEqual(rows[0]["root"], "RootBom")
        self.assertTrue(rows[0]["header"])
        self.assertTrue(rows[0]["complete"])

    def test_nested_open_root_stays_incomplete_after_continuation(self):
        head = b"\xef\xbb\xbf" + b'<?xml version="1.0"?>' + b"\n<Foo><Foo>"
        head += b"x" * (65536 - len(head))       # a full 64 KiB stream, so the next one is absorbed
        tail = b"</Foo>"                         # closes only the nested Foo
        self.add(head + tail, chunked(head + tail))
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["types"]["xml"]["files"], 1)
        row = doc["rows"][0]
        self.assertEqual(row["chunks"], 2)
        self.assertTrue(row["header"])
        self.assertFalse(row["complete"])

    def test_self_closing_and_exact_64k_declarationless_xml_boundaries(self):
        # 1) A 64 KiB binary stream followed by a self-closing XML document must not swallow the XML.
        bin64 = bytes(range(256)) * 256
        self.add(bin64)
        self.add(b'<EmptyDoc attr="1"/>\n')
        # 2) An exact 64 KiB complete declarationless XML document must not swallow the next text stream.
        prefix, suffix = b"<Exact64>", b"</Exact64>"
        exact64 = prefix + b"a" * (65536 - len(prefix) - len(suffix)) + suffix
        self.add(exact64)
        self.add(b"plain text after xml\n")
        # 3) An incomplete 64 KiB declarationless XML stream must not swallow a following binary stream.
        unclosed64 = b"<OpenDoc>" + b"b" * (65536 - len(b"<OpenDoc>"))
        self.add(unclosed64)
        self.add(bytes(range(256)) * 2)
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        self.assertEqual(
            [(r["type"], r["root"], r["complete"], r["chunks"]) for r in rows],
            [
                ("binary", "", True, 1),
                ("xml", "EmptyDoc", True, 1),
                ("xml", "Exact64", True, 1),
                ("text", "", True, 1),
                ("xml", "", False, 1),
                ("binary", "", True, 1),
            ],
        )

    def test_declarationless_xml_keeps_utf8_continuation_stream(self):
        # A 64 KiB continuation of non-ASCII UTF-8 text falls below printable()'s ASCII ratio
        # and classifies as "binary", yet must stay part of the open declarationless XML document.
        accent = "é".encode("utf-8")
        # Split an "é" (0xC3 0xA9) across the 64 KiB chunk boundary as well.
        head = b"<Doc>" + b" " * (65536 - len(b"<Doc>") - 1) + accent[:1]
        middle = accent[1:] + b"<t>" + accent * 32766
        self.assertEqual(len(head), 65536)
        self.assertEqual(len(middle), 65536)
        tail = b"</t></Doc>"
        self.add(head)
        self.add(middle)
        self.add(tail)
        self.finish()
        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        self.assertEqual(
            [(r["type"], r["root"], r["complete"], r["chunks"]) for r in rows],
            [("xml", "Doc", True, 3)],
        )


class BuildRowsReadCountTest(unittest.TestCase):
    """The XML-shaped text branch must join the chunks it already read, not reinflate the group twice."""

    def test_text_group_with_late_close_reads_each_member_once(self):
        tool = load_tool()
        head = b"<SyntheticRoot>" + b" " * (65536 - len(b"<SyntheticRoot>"))
        tail = b"</SyntheticRoot>"
        rows = [
            {"off": 0x1000, "stored": len(head), "size": 65536, "head": head[:16].hex(), "type": ""},
            {"off": 0x2000, "stored": len(tail), "size": len(tail), "head": tail[:16].hex(), "type": ""},
        ]
        chunks_by_offset = {0x1000: head, 0x2000: tail}
        source = mock.Mock()
        source.read.side_effect = lambda row: chunks_by_offset[row["off"]]

        out = tool.build_rows(rows, source)

        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["type"], "xml")
        self.assertEqual(row["root"], "SyntheticRoot")
        self.assertTrue(row["complete"])
        self.assertFalse(row["header"])
        self.assertEqual(row["chunks"], 2)
        self.assertEqual(row["size"], 65536 + len(tail))
        self.assertEqual(row["stored"], len(head) + len(tail))
        self.assertEqual(source.read.call_count, 2)
        self.assertEqual([call.args[0]["off"] for call in source.read.call_args_list],
                         [0x1000, 0x2000])


if __name__ == "__main__":
    unittest.main()
