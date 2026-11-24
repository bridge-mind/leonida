"""gamedata_xml.py must catalog synthetic XML without reading any game file.

    python -I tests/test_gamedata_xml.py

Fixtures are hand-written documents: nested elements, attributes, macro and platform references,
an include-like manifest, a tail-only fragment and a malformed document. No game data is used.
"""
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_xml.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gamedata_xml as g  # noqa: E402

ALPHA = (
    b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>\n'
    b'<CDataFileMgr__ContentChangeSetDef version="1">\n'
    b'  <id>fixture</id>\n'
    b'  <fileList>\n'
    b'    <Item type="file"><filename>$(common)/movies/fixture.bk2</filename>'
    b'<fileType>bk2</fileType></Item>\n'
    b'    <Item type="file"><filename>platform:/levels/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'    <Item type="file"><filename>$(target)/data/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'    <Item type="file"><filename>platform:/data/fixture.xml</filename>'
    b'<fileType>xml</fileType></Item>\n'
    b'  </fileList>\n'
    b'  <includedSets><Item ref="synthetic_set"><loadPath>$(root)/fixture</loadPath></Item>'
    b'</includedSets>\n'
    b'  <!-- synthetic note -->\n'
    b'</CDataFileMgr__ContentChangeSetDef>'
)
BETA = (
    b'<timecycle_modifier_data version="1"><modifier name="fixture">'
    b'<postfx_exposure>0 1</postfx_exposure></modifier></timecycle_modifier_data>'
)
TAIL = b'<Item type="x"/>\n</TailRoot>'
BAD = b'<?xml version="1.0"?><Root><Item></Root>'
ROOTLESS = b'plain text, not an XML document\n'


class GamedataXmlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = os.path.join(self.root, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_files_catalog_roots_elements_references_and_values(self):
        alpha = self.write("alpha.xml", ALPHA)
        beta = self.write("beta.xml", BETA)
        out = os.path.join(self.root, "catalog.json")
        csv_path = os.path.join(self.root, "catalog.csv")
        p = self.run_tool(alpha, beta, "--json", out, "--csv", csv_path, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["well_formed"], 2)
        self.assertEqual(s["fragments"], 0)
        self.assertEqual(s["truncated"], 0)
        self.assertEqual(s["malformed"], 0)
        self.assertEqual(s["root_count"], 2)
        self.assertEqual(s["elements"], 21)
        self.assertEqual(s["max_depth"], 4)
        self.assertEqual(s["comments"], 1)
        self.assertEqual(s["comments_header"], 1)
        self.assertEqual(s["references"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(s["platform"], {"levels": 1, "other": 1})
        self.assertEqual(s["platform_total"], 2)
        self.assertEqual(s["filename"], 4)
        self.assertEqual(s["include_like"]["fileList"], 1)
        self.assertEqual(s["include_like"]["includedSets"], 1)
        self.assertEqual(s["include_like"]["loadPath"], 1)
        self.assertEqual(s["xi_include"], 0)

        rows = {row["source"]: row for row in data["rows"]}
        first = rows["alpha.xml"]
        self.assertEqual(first["root"], "CDataFileMgr__ContentChangeSetDef")
        self.assertEqual(first["category"], "mount / content manifests")
        self.assertEqual(first["element_count"], 18)
        self.assertEqual(first["max_depth"], 4)
        self.assertEqual(first["attributes"], ["ref", "type", "version"])
        self.assertEqual(first["refs"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(first["platform"], {"levels": 1, "other": 1})
        self.assertEqual(first["filename"], 4)
        self.assertTrue(first["header"])
        self.assertTrue(first["well_formed"])
        self.assertFalse(first["fragment"])

        second = rows["beta.xml"]
        self.assertEqual(second["root"], "timecycle_modifier_data")
        self.assertEqual(second["category"], "time, weather, sky, water")
        self.assertEqual(second["element_count"], 3)
        self.assertEqual(second["max_depth"], 3)
        self.assertEqual(second["attributes"], ["name", "version"])
        self.assertFalse(second["header"])

        root = data["roots"]["CDataFileMgr__ContentChangeSetDef"]
        self.assertEqual(root["files"], 1)
        self.assertEqual(root["category"], "mount / content manifests")
        self.assertEqual(root["elements"], 18)
        self.assertEqual(root["element_names"]["filename"], 4)
        self.assertEqual(root["element_names"]["Item"], 5)
        self.assertEqual(root["attributes"]["type"], 4)
        self.assertEqual(root["attributes"]["ref"], 1)
        self.assertEqual(root["refs"], {"$(common)": 1, "$(root)": 1, "$(target)": 1})
        self.assertEqual(root["platform"], {"levels": 1, "other": 1})
        self.assertEqual(root["filename"], 4)

        with open(csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["root"], "CDataFileMgr__ContentChangeSetDef")
        self.assertEqual(csv_rows[0]["refs"], "$(common)=1;$(root)=1;$(target)=1")
        self.assertEqual(csv_rows[0]["well_formed"], "true")

        # Values, text and comments must not leak into the catalog; names and counts are fine.
        with open(out, encoding="utf-8") as f:
            blob = f.read()
        for value in ("fixture.bk2", "synthetic note", "synthetic_set", "0 1", "fixture.xml"):
            self.assertNotIn(value, blob)

    def test_well_formed_roots_group_by_namespace_not_prefix(self):
        bare = self.write("bare.xml", b"<CTuningFile/>")
        default = self.write("default.xml", b'<CTuningFile xmlns="urn:example"/>')
        prefix = self.write("prefix.xml", b'<t:CTuningFile xmlns:t="urn:example"/>')
        out = os.path.join(self.root, "catalog.json")
        p = self.run_tool(bare, default, prefix, "--json", out, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        qualified = "{urn:example}CTuningFile"
        self.assertEqual(set(report["roots"]), {"CTuningFile", qualified})
        self.assertEqual(report["roots"][qualified]["files"], 2)
        self.assertEqual(report["roots"][qualified]["element_names"], {qualified: 2})
        self.assertEqual(report["roots"][qualified]["category"], "uncatalogued")
        self.assertEqual(report["roots"]["CTuningFile"]["files"], 1)
        self.assertEqual(report["summary"]["catalog_roots"], 1)

    def test_counts_tail_references_across_parser_buffer_boundary(self):
        start = b"<root><child>"
        close = b"</child>"
        tail = b"$(common)/x</root>"
        short = self.write("short.xml", start + b"x" + close + tail)
        padding = b"x" * (16384 - len(start) - len(close))
        long = self.write("long.xml", start + padding + close + tail)
        out = os.path.join(self.root, "catalog.json")
        p = self.run_tool(short, long, "--check", "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        rows = {row["source"]: row for row in data["rows"]}
        self.assertEqual(rows["short.xml"]["refs"], {"$(common)": 1})
        self.assertEqual(rows["long.xml"]["refs"], {"$(common)": 1})

    def test_check_reports_tail_only_fragment_and_malformed(self):
        tail = self.write("tail.xml", TAIL)
        p = self.run_tool(tail, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("1 truncated at the start", p.stdout)
        self.assertIn("0 rootless fragments", p.stdout)
        self.assertIn("0 malformed", p.stdout)
        self.assertIn("TailRoot", p.stdout)

        rootless = self.write("rootless.xml", ROOTLESS)
        p = self.run_tool(rootless, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("1 rootless fragment", p.stdout)

        bad = self.write("bad.xml", BAD)
        p = self.run_tool(bad, "--check")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("1 malformed", p.stdout)
        self.assertIn("line 1", p.stdout)

        # An end-truncated file that starts at <COther> and cuts off right after a closed child </x>
        # must keep root "COther" and report malformed (not tail_only under bogus root "x").
        cut = self.write("cut.xml", b"<COther><x>1</x>")
        out = os.path.join(self.root, "cut.json")
        p = self.run_tool(cut, "--check", "--json", out)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            row = json.load(f)["rows"][0]
        self.assertEqual(row["root"], "COther")
        self.assertFalse(row["tail_only"])
        self.assertFalse(row["truncated"])
        self.assertTrue(row["malformed"])

    def test_check_crossed_tags(self):
        cases = (
            (b"<R><x></R></x>", "R"),
            (b'<?xml version="1.0"?><R><x></R></x>', "R"),
            (b'<n:R xmlns:n="urn:test"><!-- </n:R> --><x><![CDATA[</x>]]></n:R></x>', "n:R"),
        )
        for payload, root in cases:
            with self.subTest(payload=payload):
                doc = self.write("crossed.xml", payload)
                p = self.run_tool(doc, "--check")
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertIn("malformed " + root + " @", p.stdout)
                self.assertIn("0 truncated at the start", p.stdout)

    def test_check_crossed_text_in_missing_comment_or_cdata_prefix(self):
        for opener, suffix in ((b"<!-- ", b" -->"), (b"<![CDATA[", b"]]>")):
            payload = b"<A><B></A></B>" + suffix + b"</R>"
            with self.subTest(opener=opener):
                # Establish that these are tails of a well-formed document.
                import xml.etree.ElementTree as ET
                ET.fromstring(b"<R>" + opener + payload)
                doc = self.write("lexical_tail.xml", payload)
                p = self.run_tool(doc, "--check")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertIn("1 truncated at the start", p.stdout)
                self.assertIn("truncated R @", p.stdout)
                # A stray terminator must not hide malformed text after it.
                broken = self.write("lexical_broken.xml", payload[:-4] + b"<C><D></C></D></R>")
                p = self.run_tool(broken, "--check")
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)

    def test_check_reports_unknown_encoding_as_malformed(self):
        unknown = self.write("unknown.xml", b'<?xml version="1.0" encoding="not-a-charset"?><root/>')
        valid = self.write("valid.xml", b"<root/>")
        p = self.run_tool(unknown, valid, "--check")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("1 well-formed", p.stdout)
        self.assertIn("1 malformed", p.stdout)
        self.assertIn("not-a-charset", p.stdout)

    def test_comments_in_a_document_that_does_not_parse_are_counted(self):
        # The fallback must respect the context of each delimiter after the parser stops.
        cases = {"open.xml": (b"<a><!--x--><!--y-->", 2),
                 "tail.xml": (b"</b><!--x--></a><!--y-->", 2),
                 "cdata.xml": (b"<a><!--x--><![CDATA[<!--no-->]]><b><![CDATA[<!--cut", 1),
                 "cdata-in-comment.xml": (b"<a></b><!--literal <![CDATA[--><!--later-->", 2),
                 "comment-in-pi.xml": (b"<a></b><?step literal <!-- ?> <!--later-->", 1),
                 "unfinished-comment.xml": (b"<a><!--unfinished <![CDATA[", 1)}
        for name, (body, want) in cases.items():
            with self.subTest(name=name):
                doc = self.write(name, body)
                out = os.path.join(self.root, name + ".json")
                p = self.run_tool(doc, "--json", out)
                self.assertEqual(p.returncode, 0, p.stderr)
                with open(out, encoding="utf-8") as f:
                    row = json.load(f)["rows"][0]
                self.assertFalse(row["well_formed"])
                self.assertEqual(row["comments"], want)

    def test_comments_inside_cdata_are_not_counted(self):
        doc = self.write("comments.xml", b"<!-- real --><Root><![CDATA[<!-- text -->]]></Root><!-- real -->")
        out = os.path.join(self.root, "catalog.json")
        p = self.run_tool(doc, "--json", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        self.assertTrue(report["rows"][0]["well_formed"])
        self.assertEqual(report["rows"][0]["comments"], 2)
        self.assertEqual(report["summary"]["comments"], 2)

    def test_carve_directory_joins_chunks_and_marks_fragment(self):
        carve = os.path.join(self.root, "carve")
        os.makedirs(carve)
        doc = b'<?xml version="1.0"?>\n<RootA>' + b"<Item value=\"1\"/>" * 20 + b"</RootA>"
        split = doc.index(b"<Item")
        parts = [doc[:split], doc[split:]]
        rows = []
        off = 0x1000
        for part in parts:
            with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            rows.append({"off": off, "stored": len(part), "size": len(part),
                         "head": part[:16].hex(), "pre": ""})
            off += 0x1000
        with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
            f.write(TAIL)
        rows.append({"off": off, "stored": len(TAIL), "size": len(TAIL),
                     "head": TAIL[:16].hex(), "pre": ""})
        with open(os.path.join(carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

        out = os.path.join(self.root, "carve.json")
        p = self.run_tool(carve, "--json", out, "--check")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["mode"], "carve")
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["well_formed"], 1)
        self.assertEqual(s["tail_only"], 1)
        self.assertEqual(s["truncated"], 1)
        self.assertEqual(s["fragments"], 0)
        self.assertEqual(s["root_count"], 2)
        first = data["rows"][0]
        self.assertEqual(first["root"], "RootA")
        self.assertEqual(first["chunks"], 2)
        self.assertEqual(first["bytes"], len(doc))
        self.assertEqual(first["offset_hex"], "1000")
        self.assertEqual(first["element_count"], 21)
        self.assertEqual(first["max_depth"], 2)
        tail = data["rows"][1]
        self.assertEqual(tail["root"], "TailRoot")
        self.assertTrue(tail["tail_only"])
        self.assertTrue(tail["truncated"])
        self.assertFalse(tail["malformed"])

    def test_carve_catalog_marks_parsed_headerless_documents_complete(self) -> None:
        carve = os.path.join(self.root, "headerless")
        os.makedirs(carve)
        rows = []
        for off, data in ((0x1000, b"<Plain><Child/></Plain>"), (0x2000, b"<Empty/>")):
            with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
                f.write(data)
            rows.append({"off": off, "stored": len(data), "size": len(data),
                         "head": data[:16].hex(), "pre": ""})
        with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": rows}, f)

        out = os.path.join(self.root, "headerless.json")
        result = self.run_tool(carve, "--json", out, "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual([row["root"] for row in report["rows"]], ["Plain", "Empty"])
        for row in report["rows"]:
            with self.subTest(root=row["root"]):
                self.assertTrue(row["well_formed"])
                self.assertTrue(row["complete"])
                self.assertFalse(row["header"])
                self.assertFalse(row["tail_only"])
                self.assertFalse(row["fragment"])
        self.assertEqual(report["summary"]["tail_only"], 0)
        self.assertEqual(report["summary"]["fragments"], 0)

    def test_a_folder_that_cannot_be_listed_fails_the_scan(self):
        def fake_walk(root, onerror=None):
            self.assertIsNotNone(onerror, "os.walk needs onerror, or an unlistable folder is skipped")
            onerror(PermissionError(13, "permission denied", os.path.join(root, "blocked")))
            return iter(())

        err = io.StringIO()
        with mock.patch.object(g.os, "walk", side_effect=fake_walk), \
                mock.patch.object(sys, "argv", ["gamedata_xml.py", self.root]), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = g.main()
        self.assertEqual(rc, 2)
        self.assertIn("permission denied", err.getvalue())

    @unittest.skipUnless(os.name == "posix" and getattr(os, "geteuid", lambda: 0)() != 0,
                         "needs a POSIX filesystem and a non-root user to make a folder unlistable")
    def test_a_real_unlistable_folder_fails_the_scan(self):
        blocked = os.path.join(self.root, "blocked")
        os.makedirs(blocked)
        os.chmod(blocked, 0)
        err = io.StringIO()
        try:
            with mock.patch.object(sys, "argv", ["gamedata_xml.py", self.root]), \
                    contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = g.main()
        finally:
            os.chmod(blocked, 0o755)
        self.assertEqual(rc, 2)
        self.assertIn("blocked", err.getvalue())

    def test_unwritable_csv_or_json_is_one_error_line(self):
        doc = self.write("alpha.xml", ALPHA)
        blocker = self.write("blocker", b"x")
        for flag, name in (("--csv", "out.csv"), ("--json", "out.json")):
            with self.subTest(flag=flag):
                out = os.path.join(blocker, name)
                p = self.run_tool(doc, flag, out)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("ERROR: cannot write", p.stderr)
                self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
