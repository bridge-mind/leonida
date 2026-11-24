"""ASSET-149: hand-written content XML and carve indexes only."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from content_definitions import ContentError, carve_documents, parse_content

CHANGE = (b'<?xml version="1.0"?><CDataFileMgr__ContentChangeSetDef><id>fixture</id>'
          b'<fileList><Item><filename>$(fixture)/alpha.rpf</filename><fileType>SYNTHETIC</fileType>'
          b'<contents><Unknown a="01">unchanged</Unknown></contents><flags>FLAG_A FLAG_B</flags>'
          b'</Item><Item><filename>platform:/beta.rpf</filename><fileType/></Item></fileList>'
          b'<executionConditions><genericConditions><Item ref="fixture"/></genericConditions>'
          b'</executionConditions><mapAddAtPhase value="001"/></CDataFileMgr__ContentChangeSetDef>')
CONTENT = (b'<?xml version="1.0"?><CDataFileMgr__ContentDef><includedChangesets>'
           b'<Item><loadPath>$(fixture)/change.xml</loadPath><changesetIncludeFlags>KEEP</changesetIncludeFlags>'
           b'</Item></includedChangesets><includedSets><Item ref="fixture"/></includedSets>'
           b'<contentSets/></CDataFileMgr__ContentDef>')
SETUP = (b'<?xml version="1.0"?><SSetupData><deviceName>fixture</deviceName><order value="0x10"/>'
         b'<contentChangeSetGroups><Item><NameHash>symbolic</NameHash><ContentChangeSets>'
         b'<Item>fixture_one</Item><Item>fixture_two</Item></ContentChangeSets></Item>'
         b'</contentChangeSetGroups><future><x/>tail</future></SSetupData>')


class ContentTests(unittest.TestCase):
    def test_change_set_fields_and_unknown_records(self):
        doc = parse_content([CHANGE])
        self.assertEqual(doc.kind, "change_set")
        self.assertEqual([e.filename for e in doc.files], ["$(fixture)/alpha.rpf", "platform:/beta.rpf"])
        self.assertEqual([e.file_type for e in doc.files], ["SYNTHETIC", ""])
        self.assertEqual(doc.root.scalar("mapAddAtPhase"), "001")
        unknown = doc.files[0].record.select("contents", "Unknown")[0]
        self.assertEqual((unknown.text, unknown.attributes), ("unchanged", (("a", "01"),)))
        self.assertEqual(len(doc.root.select("executionConditions", "genericConditions", "Item")), 1)

    def test_content_keeps_references_without_opening_them(self):
        doc = parse_content([CONTENT])
        self.assertEqual(doc.kind, "content")
        self.assertEqual(doc.included_changesets[0].scalar("loadPath"), "$(fixture)/change.xml")
        self.assertEqual(doc.included_sets[0].attributes, (("ref", "fixture"),))
        self.assertEqual(doc.files, ())

    def test_setup_groups_order_and_tail(self):
        doc = parse_content([SETUP])
        self.assertEqual(doc.root.scalar("order"), "0x10")
        self.assertEqual([x.text for x in doc.change_set_groups[0].select("ContentChangeSets", "Item")],
                         ["fixture_one", "fixture_two"])
        self.assertEqual(doc.root.select("future", "x")[0].tail, "tail")
        self.assertEqual(doc.root.scalar("absent"), None)

    def test_chunk_boundaries_and_utf16(self):
        expected = parse_content([CHANGE])
        for size in (1, 7, 63):
            self.assertEqual(parse_content(CHANGE[i:i+size] for i in range(0, len(CHANGE), size)), expected)
        raw = '<?xml version="1.0" encoding="utf-16"?><SSetupData><label>café</label></SSetupData>'.encode('utf-16')
        self.assertEqual(parse_content(raw[i:i+1] for i in range(len(raw))).root.scalar("label"), "café")

    def test_ambiguous_known_fields_are_rejected(self):
        for fields in (b'<filename>a</filename><filename>b</filename>',
                       b'<filename><Item/></filename>', b'<filename value="a">b</filename>'):
            with self.assertRaises(ContentError):
                parse_content([b'<CDataFileMgr__ContentChangeSetDef><fileList><Item>' + fields +
                               b'</Item></fileList></CDataFileMgr__ContentChangeSetDef>'])

    def test_missing_optional_file_fields_stay_missing(self):
        doc = parse_content([b'<CDataFileMgr__ContentChangeSetDef><fileList><Item/></fileList>'
                             b'</CDataFileMgr__ContentChangeSetDef>'])
        self.assertIsNone(doc.files[0].filename)
        self.assertIsNone(doc.files[0].file_type)

    def test_invalid_unsupported_and_dtd_inputs(self):
        for raw in (b'', b'<SSetupData>', b'<SSetupData/><SSetupData/>', b'<private/>',
                    b'<SSetupData xmlns="private"/>', b'<SSetupData>\xff</SSetupData>',
                    b'<!DOCTYPE SSetupData SYSTEM "file:///private"><SSetupData/>',
                    b'<!DOCTYPE SSetupData [<!ENTITY x "private">]><SSetupData>&x;</SSetupData>',
                    b'<?xml version="1.0" encoding="UTF-32"?><SSetupData/>'):
            with self.subTest(raw=raw), self.assertRaises(ContentError) as caught:
                parse_content([raw])
            self.assertNotIn("private", str(caught.exception))

    def test_byte_and_depth_limits(self):
        with self.assertRaisesRegex(ContentError, 'byte limit'):
            parse_content([SETUP], max_bytes=10)
        with self.assertRaisesRegex(ContentError, 'nesting'):
            parse_content([b'<SSetupData>' + b'<x>' * 128 + b'</x>' * 128 + b'</SSetupData>'])

    def make_carve(self, directory, blocks):
        rows = []
        for i, data in enumerate(blocks):
            offset = 4096 + i * 1000
            with open(os.path.join(directory, f'{offset:012x}.bin'), 'wb') as f:
                f.write(data)
            rows.append(dict(off=offset, size=len(data), stored=100, head=data[:16].hex()))
        with open(os.path.join(directory, 'index.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(dict(files=rows), f)
        return rows

    def test_carve_groups_split_xml_and_keeps_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_carve(directory, [CHANGE[:180], CHANGE[180:], CONTENT, SETUP,
                                       b'<?xml version="1.0"?><Unrelated/>'])
            docs = list(carve_documents(directory))
            self.assertEqual(len(docs), 4)
            self.assertEqual(docs[0][:2], (4096, 200))
            self.assertEqual([d.kind if d else None for _, _, d in docs],
                             ['change_set', 'content', 'setup', None])
            self.assertEqual(docs[0][2], parse_content([CHANGE]))

    def test_carve_missing_block_is_not_replaced_with_header(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.make_carve(directory, [CHANGE])
            os.remove(os.path.join(directory, f'{rows[0]["off"]:012x}.bin'))
            with self.assertRaises(OSError):
                list(carve_documents(directory))

    def test_cli_counts_unrelated_standalone_xml_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, name) for name in ("setup.xml", "other.xml", "broken.xml")]
            for path, payload in zip(paths, (SETUP, b"<Other/>", b"<Other>")):
                with open(path, "wb") as f:
                    f.write(payload)
            cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "content_definitions.py")]
            for inputs, expected in ((paths[:2], 0), (paths, 1)):
                with self.subTest(inputs=len(inputs)):
                    result = subprocess.run(cmd + inputs, capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    counts = json.loads(result.stdout)
                    self.assertEqual((counts["setups"], counts["other_xml"], counts["failed_inputs"]),
                                     (1, 1, expected))

    def test_cli_reports_counts_only_and_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_carve(directory, [CHANGE, CONTENT, SETUP])
            cmd = [sys.executable, '-I', os.path.join(ROOT, 'tools', 'content_definitions.py'), directory]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), dict(documents=3, change_sets=1, contents=1,
                             setups=1, files=2, included_changesets=1, included_sets=1, groups=1,
                             other_xml=0, failed_inputs=0))
            self.assertEqual(result.stderr, '')
            self.assertNotIn('fixture', result.stdout)
            self.assertNotIn(directory, result.stdout)

            missing = os.path.join(directory, 'private.xml')
            result = subprocess.run(cmd + [missing], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout), dict(documents=3, change_sets=1, contents=1,
                             setups=1, files=2, included_changesets=1, included_sets=1, groups=1,
                             other_xml=0, failed_inputs=1))
            self.assertEqual(result.stderr.strip().splitlines(), [f'ERROR: {missing}: does not exist'])


if __name__ == '__main__':
    unittest.main()
