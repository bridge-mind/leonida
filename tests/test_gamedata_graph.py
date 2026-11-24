"""gamedata_graph.py must measure a synthetic content-set graph without game data.

    python -I tests/test_gamedata_graph.py

The fixtures are hand-written manifests: two change-set documents that include each other, one
content definition that includes the first, a missing set id, all four documented prefix rules, an
unknown ``common:/`` scheme, two parent locators, a tail-only document and a malformed document.
No game data is used.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_graph.py")
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import gamedata_graph as gg  # noqa: E402

ALPHA = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<CDataFileMgr__ContentChangeSetDef>\n'
    b'  <id>set.alpha</id>\n'
    b'  <fileList>\n'
    b'    <Item><filename>$(common)/fixture/a.bk2</filename></Item>\n'
    b'    <Item><filename>platform:/levels/fixture/a.rpf</filename></Item>\n'
    b'    <Item><filename>$(target)/fixture/a</filename></Item>\n'
    b'    <Item><filename>common:/FIXTURE/A.XML</filename></Item>\n'
    b'  </fileList>\n'
    b'  <includedSets><Item>set.beta</Item><Item>missing.set</Item></includedSets>\n'
    b'</CDataFileMgr__ContentChangeSetDef>'
)
BETA = (
    b'<CDataFileMgr__ContentChangeSetDef>\n'
    b'  <id>set.beta</id>\n'
    b'  <fileList><Item><filename>platform:/data/fixture</filename></Item></fileList>\n'
    b'  <includedSets><Item>set.alpha</Item></includedSets>\n'
    b'</CDataFileMgr__ContentChangeSetDef>'
)
GAMMA = (
    b'<?xml version="1.0"?>\n'
    b'<CDataFileMgr__ContentDef>\n'
    b'  <contentSets>\n'
    b'    <Item><id>set.gamma</id><fileList>'
    b'<Item><filename>$(common)/fixture/g.xml</filename></Item>'
    b'</fileList></Item>\n'
    b'  </contentSets>\n'
    b'  <includedSets><Item>set.alpha</Item></includedSets>\n'
    b'  <includedChangesets><Item><loadPath>$(root)/fixture/g.meta</loadPath></Item>'
    b'</includedChangesets>\n'
    b'</CDataFileMgr__ContentDef>'
)
DELTA = (
    b'<PlainRoot><Item parentPath="../Item[@key=\'X\']" parentFile="parent.dat"/></PlainRoot>'
)
PLAIN = b'<?xml version="1.0"?><PlainRoot2/>'
TAIL = b'<Item type="x"/>\n</TailRoot>'
BAD = b'<?xml version="1.0"?><Broken><Item>'


class GamedataGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload, parts=None):
        parts = parts if parts is not None else [payload]
        for part in parts:
            offset = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{offset:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({
                "off": offset,
                "stored": len(part),
                "size": len(part),
                "head": part[:16].hex(),
                "pre": "",
            })

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}, f)

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root,
                              capture_output=True, text=True, errors="replace")

    def test_unknown_encoding_counts_as_unparsed_beside_valid_document(self):
        unsupported = os.path.join(self.root, "unsupported.xml")
        valid = os.path.join(self.root, "valid.xml")
        out = os.path.join(self.root, "graph.json")
        with open(unsupported, "wb") as f:
            f.write(b'<?xml version="1.0" encoding="x-fixture-unknown"?><Unparsed/>')
        with open(valid, "wb") as f:
            f.write(ALPHA)
        result = self.run_tool(unsupported, valid, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(out, encoding="utf-8") as f:
            summary = json.load(f)["summary"]
        self.assertEqual(summary["payloads"], 2)
        self.assertEqual(summary["parsed"], 1)
        self.assertEqual(summary["unparsed"], 1)
        self.assertEqual(summary["rootless"], 0)
        self.assertEqual(summary["include_sites"], 2)

    def test_cli_reports_unwritable_graph_report(self):
        self.finish()
        out = os.path.join(self.root, "missing", "graph.json")

        result = self.run_tool(self.carve, "--json", out)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"cannot write {out}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_json_that_is_a_directory_is_refused(self):
        self.finish()
        r = self.run_tool(self.carve, "--json", self.root)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("is a directory", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_json_over_input_xml_file_is_refused(self):
        xml_path = os.path.join(self.root, "input.xml")
        with open(xml_path, "wb") as f:
            f.write(ALPHA)
        alias = os.path.join(self.root, ".", "input.xml")
        r = self.run_tool(xml_path, "--json", alias)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite the input", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(xml_path, "rb") as f:
            self.assertEqual(f.read(), ALPHA)

    def test_json_over_carve_dir_manifest_is_refused(self):
        self.add(ALPHA)
        self.finish()
        manifest = os.path.join(self.carve, "index.json")
        with open(manifest, "rb") as f:
            content = f.read()
        r = self.run_tool(self.carve, "--json", manifest)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite the input", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(manifest, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_json_over_carve_dir_bin_payload_is_refused(self):
        self.add(ALPHA)
        self.finish()
        bin_path = os.path.join(self.carve, f"{self.rows[0]['off']:012x}.bin")
        with open(bin_path, "rb") as f:
            content = f.read()
        r = self.run_tool(self.carve, "--json", bin_path)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite an input file", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(bin_path, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_json_cannot_replace_a_carved_payload_via_external_hard_link(self):
        self.add(ALPHA)
        self.finish()
        payload = os.path.join(self.carve, f"{self.rows[0]['off']:012x}.bin")
        report = os.path.join(self.root, "alias.json")
        try:
            os.link(payload, report)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        r = self.run_tool(self.carve, "--json", report)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite the input", r.stderr)
        with open(payload, "rb") as f:
            self.assertEqual(f.read(), ALPHA)

    def test_fresh_json_inside_carve_dir_is_allowed(self):
        self.add(ALPHA)
        self.finish()
        out = os.path.join(self.carve, "graph.json")
        r = self.run_tool(self.carve, "--json", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.isfile(out))
        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["summary"]["mode"], "carve")

    def test_json_over_directory_mode_nested_xml_is_refused(self):
        # Directory mode (no index.json) reads every nested *.xml; --json into one of them must not
        # truncate it.
        directory = os.path.join(self.root, "dir")
        nested_dir = os.path.join(directory, "nested")
        os.makedirs(nested_dir)
        xml_path = os.path.join(nested_dir, "a.xml")
        with open(xml_path, "wb") as f:
            f.write(ALPHA)

        r = self.run_tool(directory, "--json", xml_path)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite the input", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(xml_path, "rb") as f:
            self.assertEqual(f.read(), ALPHA)

    def test_json_over_directory_mode_xml_hardlink_is_refused(self):
        directory = os.path.join(self.root, "dir")
        os.makedirs(directory)
        xml_path = os.path.join(directory, "a.xml")
        with open(xml_path, "wb") as f:
            f.write(ALPHA)
        alias = os.path.join(self.root, "alias.xml")
        try:
            os.link(xml_path, alias)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")

        r = self.run_tool(directory, "--json", alias)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("would overwrite the input", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(xml_path, "rb") as f:
            self.assertEqual(f.read(), ALPHA)

    def test_cycle_at_end_of_a_deep_include_chain(self):
        size = sys.getrecursionlimit() + 50
        adjacency = {node: [node + 1] for node in range(size - 1)}
        adjacency[size - 1] = [size - 2]
        components = gg._strongly_connected(list(range(size)), adjacency)
        self.assertEqual(len(components), size - 1)
        self.assertEqual([set(c) for c in components if len(c) > 1], [{size - 2, size - 1}])

    def test_carve_graph_counts_cycles_reachability_and_no_leak(self):
        self.add(ALPHA)
        self.add(BETA)
        split = GAMMA.index(b"<includedSets>")
        self.add(GAMMA, [GAMMA[:split], GAMMA[split:]])
        self.add(DELTA)
        self.add(PLAIN)
        self.add(TAIL)
        self.add(BAD)
        self.finish()

        out = os.path.join(self.root, "graph.json")
        p = self.run_tool(self.carve, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]

        self.assertEqual(s["mode"], "carve")
        self.assertEqual(s["payloads"], 7)
        self.assertEqual(s["documents"], 6)
        self.assertEqual(s["parsed"], 5)
        self.assertEqual(s["unparsed"], 2)
        self.assertEqual(s["root_count"], 6)
        self.assertEqual(s["rootless"], 0)
        self.assertEqual(s["manifest_documents"], 3)
        self.assertEqual(s["content_sets"], 4)
        self.assertEqual(s["defined_content_sets"], 3)
        self.assertEqual(s["dangling_content_sets"], 1)
        self.assertEqual(s["duplicate_set_ids"], 0)
        self.assertEqual(s["reference_sites"], 13)
        self.assertEqual(s["target_sites"], 9)
        self.assertEqual(s["include_sites"], 4)
        self.assertEqual(s["references"], {
            "filename": 6, "includedSets": 4, "loadPath": 1, "parentFile": 1, "parentPath": 1,
        })
        self.assertEqual(s["prefixes"], {
            "$(common)": 2, "$(root)": 1, "$(target)": 1, "common:/": 1,
            "platform:/levels": 1, "platform:/other": 1, "relative/locator": 2,
        })
        self.assertEqual(s["resolution"], {
            "document": 3, "document_unmapped": 6, "include_id_missing": 1,
            "relative_locator": 2, "unrecognized_prefix": 1,
        })
        self.assertEqual(s["unique"], {
            "targets": 9,
            "targets_by_namespace": {
                "$(common)": 2, "$(root)": 1, "$(target)": 1, "common:/": 1,
                "platform:/levels": 1, "platform:/other": 1, "relative/locator": 2,
            },
            "document_unmapped": 6,
            "unknown_macro": 0,
            "unrecognized_prefix": 1,
            "relative_locator": 2,
            "content_set_ids": 4,
            "include_ids": 3,
        })
        self.assertEqual(s["nodes"], {"documents": 7, "content_sets": 4, "targets": 9, "total": 20})
        self.assertEqual(s["edges"], {
            "total": 16, "defines": 3, "includes": 4, "contains_file": 6,
            "loads_changeset": 1, "parent_path": 1, "parent_file": 1,
            "document_edges": 7, "target_edges": 9,
        })
        self.assertEqual(s["content_graph"], {
            "include_edges": 3,
            "documents_with_incoming_includes": 2,
            "documents_with_outgoing_includes": 3,
            "root_documents": 1,
            "root_fallback": False,
            "reachable_documents": 3,
            "max_depth": 2,
            "cycles": 1,
            "cycle_nodes": 2,
        })
        self.assertEqual(s["reachable"], {
            "documents": 3, "content_sets": 4, "targets": 7, "target_sites": 7,
        })

        mount = s["categories"]["mount / content manifests"]
        self.assertEqual(mount["documents"], 3)
        self.assertEqual(mount["manifest_documents"], 3)
        self.assertEqual(mount["content_sets"], 3)
        self.assertEqual(mount["include_references"], 4)
        self.assertEqual(mount["target_references"], 7)
        self.assertEqual(mount["references"], 11)
        self.assertEqual(mount["roots"], 2)
        unknown = s["categories"]["uncatalogued"]
        self.assertEqual(unknown["documents"], 4)
        self.assertEqual(unknown["target_references"], 2)
        self.assertEqual(unknown["references"], 2)
        self.assertEqual(unknown["roots"], 4)

        change_set = s["roots"]["CDataFileMgr__ContentChangeSetDef"]
        self.assertEqual(change_set["documents"], 2)
        self.assertEqual(change_set["content_sets"], 2)
        self.assertEqual(change_set["include_references"], 3)
        self.assertEqual(change_set["target_references"], 5)
        self.assertEqual(change_set["references"], 8)
        self.assertEqual(change_set["manifest_documents"], 2)
        content_def = s["roots"]["CDataFileMgr__ContentDef"]
        self.assertEqual(content_def["documents"], 1)
        self.assertEqual(content_def["content_sets"], 1)
        self.assertEqual(content_def["include_references"], 1)
        self.assertEqual(content_def["target_references"], 2)
        self.assertEqual(content_def["references"], 3)

        self.assertEqual(data["unresolved"], {
            "document_unmapped": {
                "$(common)": 2, "$(root)": 1, "$(target)": 1,
                "platform:/levels": 1, "platform:/other": 1,
            },
            "unknown_macro": {},
            "unrecognized_prefix": {"common:/": 1},
            "relative_locator": 2,
            "include_id_missing": 1,
            "total_sites": 10,
        })

        # Values, ids, locators and per-file names must not leak into the aggregate output.
        with open(out, encoding="utf-8") as f:
            blob = f.read()
        for value in ("a.bk2", "fixture/a.rpf", "FIXTURE/A.XML", "set.alpha", "set.beta",
                      "missing.set", "../Item", "parent.dat", "fixture/g.meta",
                      "alpha.xml", "g.xml"):
            self.assertNotIn(value, blob)

    def test_file_mode_keeps_manifest_documents_and_missing_ids(self):
        alpha = os.path.join(self.root, "alpha.xml")
        gamma = os.path.join(self.root, "gamma.xml")
        delta = os.path.join(self.root, "delta.xml")
        for path, payload in ((alpha, ALPHA), (gamma, GAMMA), (delta, DELTA)):
            with open(path, "wb") as f:
                f.write(payload)

        out = os.path.join(self.root, "files.json")
        p = self.run_tool(alpha, gamma, delta, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["mode"], "files")
        self.assertEqual(s["payloads"], 3)
        self.assertEqual(s["parsed"], 3)
        self.assertEqual(s["manifest_documents"], 2)
        self.assertEqual(s["content_sets"], 4)       # alpha, beta, gamma and the second missing id
        self.assertEqual(s["defined_content_sets"], 2)
        self.assertEqual(s["dangling_content_sets"], 2)
        self.assertEqual(s["resolution"]["document"], 1)
        self.assertEqual(s["resolution"]["include_id_missing"], 2)
        self.assertEqual(s["content_graph"]["include_edges"], 1)
        self.assertEqual(s["content_graph"]["reachable_documents"], 2)
        self.assertEqual(s["content_graph"]["cycles"], 0)

    def test_file_mode_counts_filenames_outside_file_lists(self):
        with open(os.path.join(self.root, "sample.xml"), "wb") as f:
            f.write(b"<root><group><filename>platform:/levels/a</filename>"
                    b"<loadPath>platform:/levels/b</loadPath></group></root>")

        p = self.run_tool("sample.xml", "--json", "graph.json")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(os.path.join(self.root, "graph.json"), encoding="utf-8") as f:
            s = json.load(f)["summary"]
        self.assertEqual(s["references"], {"filename": 1})
        self.assertEqual(s["nodes"]["targets"], 1)
        self.assertEqual(s["edges"]["contains_file"], 1)

    def test_prefix_classification_keeps_unknown_schemes_unresolved(self):
        self.assertEqual(gg.classify_target("$(common)/fixture/a.bk2"),
                         ("$(common)", "fixture/a.bk2", "document_unmapped"))
        self.assertEqual(gg.classify_target("$(root)/fixture/x.meta"),
                         ("$(root)", "fixture/x.meta", "document_unmapped"))
        self.assertEqual(gg.classify_target("platform:/levels/fixture/a.rpf"),
                         ("platform:/levels", "levels/fixture/a.rpf", "document_unmapped"))
        self.assertEqual(gg.classify_target("platform:/data/fixture"),
                         ("platform:/other", "data/fixture", "document_unmapped"))
        self.assertEqual(gg.classify_target("common:/FIXTURE/A.XML"),
                         ("common:/", "FIXTURE/A.XML", "unrecognized_prefix"))
        self.assertEqual(gg.classify_target("$(future)/a"),
                         ("$(future)", "a", "unknown_macro"))
        self.assertEqual(gg.classify_target("../Item[@key='X']"),
                         ("relative/locator", "../Item[@key='X']", "relative_locator"))


INHERIT = (
    b'<Root><Sets>'
    b'<Item key="base"/>'
    b'<Item key="mid" parentPath="../Item[@key=\'base\']"/>'
    b'<Item key="leaf" parentPath="../Item[@key=\'mid\']"/>'
    b'<Item key="dq" parentPath=\'../Item[@key="base"]\'/>'
    b'<Item key="case" parentPath="../Item[@key=\'BASE\']"/>'
    b'<Item key="nonform" parentPath="Item/base"/>'
    b'<Item key="twin"/><Item key="twin"/>'
    b'<Item key="amb" parentPath="../Item[@key=\'twin\']"/>'
    b'<Item key="loopa" parentPath="../Item[@key=\'loopb\']"/>'
    b'<Item key="loopb" parentPath="../Item[@key=\'loopa\']"/>'
    b'</Sets>'
    b'<Other><Item key="far" parentPath="../Item[@key=\'base\']"/></Other>'
    b'</Root>'
)


class ParentPathInheritanceTest(unittest.TestCase):
    def graph(self, *payloads):
        return gg.build_graph([{"data": p, "complete": True} for p in payloads])

    def test_sibling_locators_resolve_inside_the_document(self):
        s = self.graph(INHERIT)["summary"]["parent_path_inheritance"]
        self.assertEqual(s["documents"], 1)
        self.assertEqual(s["sites"], 9)
        self.assertEqual(s["xpath_sibling_form"], 8)       # "Item/base" is not the ../Tag[@a='v'] form
        # mid, leaf, dq, loopa, loopb; "case" is exact-match only; "far" has no sibling "base" in <Other>
        self.assertEqual(s["resolved_in_document"], 5)
        self.assertEqual(s["ambiguous"], 1)                # two siblings share the key "twin"
        self.assertEqual(s["unresolved"], 3)
        self.assertEqual(s["chain_depths"], {"1": 2, "2": 1})  # mid and dq -> base, leaf -> mid -> base
        self.assertEqual(s["cycles"], 2)                   # loopa and loopb each walk back to themselves

    def test_existing_counts_are_unchanged(self):
        s = self.graph(INHERIT)["summary"]
        self.assertEqual(s["references"], {"parentPath": 9})
        self.assertEqual(s["edges"]["parent_path"], 9)
        self.assertEqual(s["resolution"], {"relative_locator": 9})

    def test_no_locator_documents(self):
        s = self.graph(PLAIN, BAD)["summary"]["parent_path_inheritance"]
        self.assertEqual((s["documents"], s["sites"], s["cycles"], s["chain_depths"]), (0, 0, 0, {}))

    def test_summary_keeps_no_key_strings(self):
        text = json.dumps(self.graph(INHERIT))
        for word in ("base", "leaf", "twin", "loopa", "far"):
            self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
