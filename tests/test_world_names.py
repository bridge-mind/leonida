"""Synthetic tests for tools/world_names.py (no game data).

    python -I tests/test_world_names.py
"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_names.py")
    spec = importlib.util.spec_from_file_location("world_names_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WN = load_tool()


class ManifestEnumerationTest(unittest.TestCase):
    def test_an_unreadable_subtree_does_not_replace_the_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = os.path.join(directory, "manifests")
            os.mkdir(manifests)
            blocked = os.path.join(manifests, "blocked")
            output = os.path.join(directory, "dictionary.json")
            with open(output, "wb") as f:
                f.write(b"existing dictionary")
            names = os.path.join(directory, "names.txt")
            with open(names, "w", encoding="utf-8", newline="\n") as f:
                f.write("synthetic_model\n")

            def walk(root, onerror=None):
                if onerror:
                    onerror(PermissionError(13, "Permission denied", blocked))
                return iter(())

            with mock.patch.object(WN.os, "walk", walk), contextlib.redirect_stderr(io.StringIO()) as err, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["build", "--manifest", manifests, "--names", names, "--out", output]), 1)
            self.assertIn(blocked, err.getvalue())
            with open(output, "rb") as f:
                self.assertEqual(f.read(), b"existing dictionary")


class HashTest(unittest.TestCase):
    def test_joaat_case_sensitive_field_names(self):
        # Confirmed by the 0x7a schema member tables.
        self.assertEqual(WN.joaat("CMapData", case_sensitive=True), 0xD3593FA6)
        self.assertEqual(WN.joaat("CEntityDef", case_sensitive=True), 0xCE501483)
        self.assertEqual(WN.joaat("archetypeName", case_sensitive=True), 0xA023A02C)
        self.assertEqual(WN.joaat("name", case_sensitive=True), 0xE6E17F14)
        self.assertEqual(WN.joaat("parent", case_sensitive=True), 0xA99C4651)

    def test_joaat_is_case_insensitive_by_default(self):
        self.assertEqual(WN.joaat("PLAYER"), WN.joaat("player"))
        self.assertNotEqual(WN.joaat("PLAYER", case_sensitive=True),
                            WN.joaat("player", case_sensitive=True))

    def test_murmur3_known_vectors(self):
        self.assertEqual(WN.murmur3_x64_128(b""), (0, 0))
        self.assertEqual(WN.murmur3_x64_128(b"hello"),
                         (0xCBD8A7B341BD9B02, 0x5B1E906A48AE1D19))
        self.assertEqual(WN.murmur3_x64_128(b"The quick brown fox jumps over the lazy dog"),
                         (0xE34BBC7BBC071B6C, 0x7A433CA9C49A9347))

    def test_key64_is_low_murmur_word(self):
        name = "synthetic_key_64"
        self.assertEqual(WN.hash64(name), WN.murmur3_x64_128(name.encode())[0])

    def test_u64_parse_and_format(self):
        self.assertEqual(WN.parse_u64(0x1122334455667788), 0x1122334455667788)
        self.assertEqual(WN.parse_u64("0x1122334455667788"), 0x1122334455667788)
        self.assertEqual(WN.parse_u64("1122334455667788"), 0x1122334455667788)
        self.assertEqual(WN.parse_u64(None), None)
        self.assertEqual(WN.format_u64(0xAB), "00000000000000ab")


class DictionaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_build_from_all_sources(self):
        manifest = self.write("manifest.json", json.dumps({
            "Items": [
                {"Name": "synthetic_prop.#dr"},
                {"Name": "synthetic_prop_lod.#ft"},
            ]
        }))
        scan = self.write("scan.json", json.dumps({
            "entries": [{"Path": "recovered/models/synthetic_cell_a._dr_at_0000000000000010.r9modeldata"}]
        }))
        names = self.write("names.txt", "synthetic_name_a\n# comment\nsynthetic_name_b\n")
        xml = self.write("common.xml",
                         "<root><filename>platform:/levels/synthetic/synthetic_cell_b.rpf</filename>"
                         "<name>SyntheticCellC</name></root>")
        dictionary = WN.build_dictionary([manifest], [scan], [names], [xml])
        counts = dictionary.counts()
        self.assertEqual(counts["exact_names"], 8)
        self.assertEqual(counts["exact_hashes"], 8)
        self.assertEqual(counts["exact_collisions"], 0)
        for name in ("synthetic_prop", "synthetic_prop_lod", "synthetic_cell_a",
                     "synthetic_name_a", "synthetic_name_b", "synthetic_cell_b",
                     "synthetic_cell_b.rpf", "SyntheticCellC"):
            self.assertIn(name, WN._unique_names(dictionary.names))

    def test_rpf9_names_index(self):
        index = self.write("rpf9_names.json", json.dumps({
            "summary": {"types": ["dr", "ft"]},
            "files": [
                {"offset": 16, "names": ["synthetic_tile_01_lod.#dr", "synthetic_tile_02_lod.#dr"]},
                {"offset": 64, "names": ["synthetic_frag.#ft"]},
                {"offset": 96, "names": "not a list"},
            ],
        }))
        dictionary = WN.build_dictionary(rpf9_names=[index])
        self.assertEqual(WN._unique_names(dictionary.names),
                         {"synthetic_tile_01_lod", "synthetic_tile_02_lod", "synthetic_frag"})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(WN.main(["build", "--rpf9-names", index,
                                      "--out", os.path.join(self.dir, "names.json")]), 0)

    def test_directory_manifest_discovery(self):
        models = os.path.join(self.dir, "assets", "archive", "models")
        os.makedirs(models)
        path = os.path.join(models, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"Items": [{"Name": "synthetic_dir_model.#dr"}]}, handle)
        dictionary = WN.build_dictionary([os.path.join(self.dir, "assets")])
        self.assertIn("synthetic_dir_model", WN._unique_names(dictionary.names))

    def test_collision_is_ambiguous_without_key64_and_resolved_with_it(self):
        dictionary = WN.NameDictionary()
        dictionary.add("PLAYER")
        dictionary.add("player")
        self.assertEqual(dictionary.counts()["exact_collisions"], 1)
        h = WN.joaat("player")
        ambiguous = dictionary.match(h)
        self.assertEqual(ambiguous["status"], "ambiguous")
        resolved = dictionary.match(h, WN.hash64("player"))
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["name"], "player")
        self.assertIn("PLAYER", resolved["rejected"])
        resolved_upper = dictionary.match(h, WN.hash64("PLAYER"))
        self.assertEqual(resolved_upper["name"], "PLAYER")

    def test_derived_lod_names_are_separate(self):
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_lod_base")
        self.assertEqual(dictionary.derive_lod(), 2)
        h = WN.joaat("synthetic_lod_base_lod")
        without = dictionary.match(h, WN.hash64("synthetic_lod_base_lod"))
        self.assertEqual(without["status"], "unresolved")
        with_derived = dictionary.match(h, WN.hash64("synthetic_lod_base_lod"),
                                        include_derived=True)
        self.assertEqual(with_derived["status"], "resolved")
        self.assertTrue(with_derived["derived"])
        self.assertEqual(with_derived["name"], "synthetic_lod_base_lod")

    def test_json_roundtrip(self):
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_roundtrip")
        dictionary.derive_lod()
        restored = WN.NameDictionary.from_json(json.loads(json.dumps(dictionary.to_json())))
        self.assertEqual(restored.counts()["exact_names"], 1)
        self.assertEqual(restored.counts()["derived_names"], 2)
        self.assertEqual(restored.match(WN.joaat("synthetic_roundtrip"))["name"],
                         "synthetic_roundtrip")


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.dictionary = WN.NameDictionary()
        for name in ("synthetic_map_cell", "synthetic_parent_cell", "synthetic_prop",
                     "synthetic_lod_base"):
            self.dictionary.add(name)
        self.dictionary.derive_lod()

    def test_map_rows(self):
        rows = [
            {"offset": 1, "map_name": WN.joaat("synthetic_map_cell"),
             "parent": WN.joaat("synthetic_parent_cell"),
             "map_name64": WN.format_u64(WN.hash64("synthetic_map_cell"))},
            {"offset": 2, "map_name": WN.joaat("synthetic_parent_cell"), "parent": 0},
            {"offset": 3, "kind": "meta"},
        ]
        stats = WN.resolve_maps(rows, self.dictionary)
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["name_present"], 2)
        self.assertEqual(stats["name_resolved"], 2)
        self.assertEqual(stats["name_verified"], 1)
        self.assertEqual(stats["parent_present"], 1)
        self.assertEqual(stats["parent_resolved"], 1)
        self.assertEqual(stats["parent_links"], 1)
        self.assertEqual(stats["parent_links_resolved"], 1)

    def test_map_64_mismatch_is_unconfirmed(self):
        rows = [{"map_name": WN.joaat("synthetic_map_cell"),
                 "map_name64": WN.format_u64(WN.hash64("synthetic_map_cell") + 1)}]
        stats = WN.resolve_maps(rows, self.dictionary)
        self.assertEqual(stats["name_resolved"], 0)
        self.assertEqual(stats["name_unconfirmed"], 1)

    def test_archetype_counts(self):
        exact = "synthetic_prop"
        derived = "synthetic_lod_base_lod"
        data = {
            "archetypes": {
                f"{WN.joaat(exact):08x}": 7,
                f"{WN.joaat(derived):08x}": 3,
                f"{WN.joaat('synthetic_missing'):08x}": 2,
            },
            "keys": {
                f"{WN.joaat(exact):08x}": WN.format_u64(WN.hash64(exact)),
                f"{WN.joaat(derived):08x}": WN.format_u64(WN.hash64(derived)),
            },
        }
        stats = WN.resolve_archetypes(data, self.dictionary, include_derived=True)
        self.assertEqual(stats["entities"], 12)
        self.assertEqual(stats["unique_hashes"], 3)
        self.assertEqual(stats["resolved_exact_entities"], 7)
        self.assertEqual(stats["resolved_derived_entities"], 3)
        self.assertEqual(stats["verified_entities"], 10)
        self.assertEqual(stats["unresolved_entities"], 2)

    def test_archetype_counts_skip_overflowing_values(self):
        name = WN.joaat_hex("synthetic_prop")
        data = json.loads(f'{{"archetypes":{{"00000100":1e309,"00000200":-1e309,"{name}":7}}}}')

        stats = WN.resolve_archetypes(data, self.dictionary)

        self.assertEqual(stats["entities"], 7)
        self.assertEqual(stats["unique_hashes"], 1)
        self.assertEqual(stats["resolved_exact_entities"], 7)

    def test_archetype_32_only_is_rejected_by_key64(self):
        dictionary = WN.NameDictionary()
        dictionary.add("PLAYER")
        dictionary.add("player")
        h = WN.joaat("player")
        stats = WN.resolve_archetypes(
            {"archetypes": {f"{h:08x}": 4}, "keys": {f"{h:08x}": WN.format_u64(WN.hash64("player"))}},
            dictionary)
        self.assertEqual(stats["resolved_exact_entities"], 4)
        self.assertEqual(stats["unconfirmed_entities"], 0)
        wrong = WN.resolve_archetypes(
            {"archetypes": {f"{h:08x}": 4}, "keys": {f"{h:08x}": WN.format_u64(WN.hash64("PLAYER") + 1)}},
            dictionary)
        self.assertEqual(wrong["unconfirmed_entities"], 4)


class CliTest(unittest.TestCase):
    def test_write_json_preserves_existing_temp_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "names.json")
            sibling = out + ".tmp"
            with open(sibling, "w", encoding="utf-8") as f:
                f.write("keep me")

            WN._write_json(out, {"names": {}})

            with open(sibling, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep me")
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"names": {}})
            self.assertCountEqual(os.listdir(tmp), ["names.json", "names.json.tmp"])

    def test_xml_unterminated_comment_keeps_later_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "concat.xml")
            with open(xml, "w", encoding="utf-8") as handle:
                handle.write("<root><name>First</name><!-- cut off here\n"
                             '<?xml version="1.0"?><root><name>RealName</name>'
                             "<filename>x/Real.rpf</filename></root>")
            d = WN.NameDictionary()
            d.add_xml(xml)
            self.assertEqual(sorted(n for names in d.to_json()["names"].values() for n in names),
                             ["First", "Real", "Real.rpf", "RealName"])

    def test_a_lone_surrogate_name_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "m.json")
            xml = os.path.join(tmp, "n.xml")
            out = os.path.join(tmp, "out.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write('{"Items": [{"Name": "bad\\ud800x"}, {"Name": "fine"}]}')
            with open(xml, "w", encoding="utf-8") as handle:
                handle.write("<root><name>one</name><name>bad&#xD800;x</name><name>two</name></root>")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["build", "--manifest", manifest, "--xml", xml, "--out", out]), 0)
            with open(out, encoding="utf-8") as handle:
                names = json.load(handle)["names"]
            self.assertEqual(sorted(n for group in names.values() for n in group), ["fine", "one", "two"])

    def test_xml_unescape_is_strict(self):
        self.assertEqual(WN._xml_unescape("A&amp;B&#x26;C&#38;D&lt;"), "A&B&C&D<")
        for text in ("Tom&notes", "R&reg_x", "A&nbsp;B", "A&B"):
            self.assertEqual(WN._xml_unescape(text), text)

    def test_build_xml_decodes_names_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "names.xml")
            out = os.path.join(tmp, "out.json")
            with open(xml, "w", encoding="utf-8") as handle:
                handle.write("<root><!-- <name>ghost</name><filename>ghost.rpf</filename> -->"
                             "<name>A&amp;B</name></root>\n"
                             '<?xml version="1.0"?><root>'
                             "<filename>levels/Cell&#x26;Prop.rpf</filename></root>")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["build", "--xml", xml, "--out", out]), 0)
            with open(out, encoding="utf-8") as handle:
                names = json.load(handle)["names"]
            self.assertEqual(names, {WN.joaat_hex(name): [name]
                                     for name in ("A&B", "Cell&Prop", "Cell&Prop.rpf")})

    def test_resolve_rejects_a_malformed_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = os.path.join(tmp, "names.json")
            out = os.path.join(tmp, "report.json")
            for field, data in (("names", [1]), ("derived", {"00000000": "name"}),
                                ("names", {"00000000": [42]})):
                with self.subTest(field=field, data=data):
                    with open(dictionary, "w", encoding="utf-8") as handle:
                        json.dump({"format": 1, field: data}, handle)
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        self.assertNotEqual(WN.main(["resolve", "--dict", dictionary, "--out", out]), 0)
                    self.assertIn(dictionary, err.getvalue())
                    self.assertIn(field, err.getvalue())
                    self.assertFalse(os.path.exists(out))

    def test_resolve_rejects_unrecognized_map_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = os.path.join(tmp, "names.json")
            maps = os.path.join(tmp, "wrong.json")
            out = os.path.join(tmp, "report.json")
            with open(dictionary, "w", encoding="utf-8") as handle:
                json.dump(WN.NameDictionary().to_json(), handle)
            with open(maps, "w", encoding="utf-8") as handle:
                json.dump({"summary": {}}, handle)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertNotEqual(WN.main(["resolve", "--dict", dictionary, "--maps", maps, "--out", out]), 0)
            self.assertIn(maps, err.getvalue())
            self.assertFalse(os.path.exists(out))

    def test_resolve_requires_a_recognized_archetype_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = os.path.join(tmp, "names.json")
            archetypes = os.path.join(tmp, "archetypes.json")
            out = os.path.join(tmp, "report.json")
            with open(dictionary, "w", encoding="utf-8", newline="\n") as f:
                json.dump(WN.NameDictionary().to_json(), f)

            args = ["resolve", "--dict", dictionary, "--archetypes", archetypes, "--out", out]
            for wrong in ([], {"summary": {}}, {"archetypes": []}):
                with self.subTest(report=wrong):
                    with open(archetypes, "w", encoding="utf-8", newline="\n") as f:
                        json.dump(wrong, f)
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        self.assertEqual(WN.main(args), 1)
                    self.assertIn(archetypes, stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertFalse(os.path.exists(out))

            for field in ("archetypes", "hashes"):
                with self.subTest(field=field):
                    with open(archetypes, "w", encoding="utf-8", newline="\n") as f:
                        json.dump({field: {}}, f)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(WN.main(args), 0)
                    with open(out, encoding="utf-8") as f:
                        self.assertEqual(json.load(f)["archetypes"]["entities"], 0)

    def test_resolve_out_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = os.path.join(tmp, "names.json")
            maps = os.path.join(tmp, "maps.json")
            archetypes = os.path.join(tmp, "archetypes.json")
            with open(dictionary, "w", encoding="utf-8") as handle:
                json.dump(WN.NameDictionary().to_json(), handle)
            with open(maps, "w", encoding="utf-8") as handle:
                json.dump({"maps": [{"map_name": WN.joaat("synthetic_cli_prop")}]}, handle)
            with open(archetypes, "w", encoding="utf-8") as handle:
                json.dump({"archetypes": {f"{WN.joaat('synthetic_cli_prop'):08x}": 1}}, handle)
            with open(dictionary, "rb") as handle:
                dict_bytes = handle.read()

            cases = [
                ("dict", ["--dict", dictionary]),
                ("map", ["--dict", dictionary, "--maps", maps]),
                ("archetype", ["--dict", dictionary, "--archetypes", archetypes]),
            ]
            for name, prefix in cases:
                with self.subTest(input=name):
                    source = {"dict": dictionary, "map": maps, "archetype": archetypes}[name]
                    with open(source, "rb") as handle:
                        original = handle.read()
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        self.assertNotEqual(WN.main(["resolve", *prefix, "--out", source]), 0)
                    self.assertIn(f"would overwrite input {source}", err.getvalue())
                    with open(source, "rb") as handle:
                        self.assertEqual(handle.read(), original)

            # A hard link is another spelling of the same inode.
            hardlink = os.path.join(tmp, "dict_hardlink.json")
            try:
                os.link(dictionary, hardlink)
            except OSError:
                self.skipTest("hard links unsupported on this filesystem")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(WN.main(["resolve", "--dict", dictionary, "--out", hardlink]), 0)
            with open(dictionary, "rb") as handle:
                self.assertEqual(handle.read(), dict_bytes)

    def test_parent_link_counts_only_map_rows(self) -> None:
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_parent")
        dictionary.add("synthetic_metadata")
        parent_hash = WN.joaat("synthetic_parent")
        meta_hash = WN.joaat("synthetic_metadata")
        rows = [
            {"kind": "map", "map_name": WN.joaat("synthetic_child"), "parent": parent_hash},
            {"kind": "map", "map_name": WN.joaat("synthetic_orphan"), "parent": meta_hash},
            {"kind": "map", "map_name": parent_hash, "parent": 0},
            {"kind": "meta", "name": meta_hash},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            names = os.path.join(tmp, "names.json")
            maps = os.path.join(tmp, "maps.json")
            out = os.path.join(tmp, "report.json")
            with open(names, "w", encoding="utf-8", newline="\n") as f:
                json.dump(dictionary.to_json(), f)
            with open(maps, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"resources": rows}, f)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["resolve", "--dict", names, "--maps", maps, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                report = json.load(f)["maps"]

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["parent_resolved"], 2)
        self.assertEqual(report["parent_links"], 1)
        self.assertEqual(report["parent_links_resolved"], 1)

    def test_resolve_links_parents_across_map_reports_in_either_order(self):
        dictionary = WN.NameDictionary()
        dictionary.add("synthetic_parent")
        parent = WN.joaat("synthetic_parent")
        with tempfile.TemporaryDirectory() as tmp:
            names, children, parents = (os.path.join(tmp, f"{p}.json")
                                        for p in ("names", "children", "parents"))
            for path, data in ((names, dictionary.to_json()),
                               (children, [{"name": 1, "parent": parent}, {"name": 2, "parent": parent}]),
                               (parents, {"maps": [{"name": parent}]})):
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(data, f)
            for first, second in ((children, parents), (parents, children)):
                with self.subTest(first=first), contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(WN.main(["resolve", "--dict", names, "--maps", first, "--maps", second]), 0)
                    stats = json.loads(out.getvalue())["maps"]
                    self.assertEqual(stats["rows"], 3)
                    self.assertEqual(stats["parent_links"], 2)
                    self.assertEqual(stats["parent_links_resolved"], 2)

    def test_build_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"Items": [{"Name": "synthetic_cli_prop.#dr"}]}, handle)
            dictionary = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["build", "--manifest", manifest, "--out", dictionary]), 0)
            with open(dictionary, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["counts"]["exact_names"], 1)

            maps = os.path.join(tmp, "maps.json")
            with open(maps, "w", encoding="utf-8") as handle:
                json.dump({"maps": [{"map_name": WN.joaat("synthetic_cli_prop")}]}, handle)
            report = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WN.main(["resolve", "--dict", dictionary, "--maps", maps,
                                          "--out", report]), 0)
            with open(report, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["maps"]["name_resolved"], 1)

    def test_build_out_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            scan = os.path.join(tmp, "scan.json")
            names = os.path.join(tmp, "names.txt")
            rpf9 = os.path.join(tmp, "rpf9_names.json")
            xml = os.path.join(tmp, "common.xml")
            models_dir = os.path.join(tmp, "assets", "archive", "models")
            os.makedirs(models_dir)
            dir_manifest = os.path.join(models_dir, "manifest.json")

            with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
                json.dump({"Items": [{"Name": "synthetic_m.#dr"}]}, handle)
            with open(dir_manifest, "w", encoding="utf-8", newline="\n") as handle:
                json.dump({"Items": [{"Name": "synthetic_dir_m.#dr"}]}, handle)
            with open(scan, "w", encoding="utf-8", newline="\n") as handle:
                json.dump({"entries": [{"Path": "models/synthetic_s._dr_at_0000000000000010.r9modeldata"}]}, handle)
            with open(names, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("synthetic_n\n")
            with open(rpf9, "w", encoding="utf-8", newline="\n") as handle:
                json.dump({"files": [{"names": ["synthetic_r.#dr"]}]}, handle)
            with open(xml, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("<root><name>synthetic_x</name></root>\n")

            cases = [
                ("manifest", ["--manifest", manifest], manifest),
                ("manifest_dir", ["--manifest", os.path.join(tmp, "assets")], dir_manifest),
                ("scan", ["--scan", scan], scan),
                ("names", ["--names", names], names),
                ("rpf9_names", ["--rpf9-names", rpf9], rpf9),
                ("xml", ["--xml", xml], xml),
            ]
            for label, prefix, source in cases:
                with self.subTest(input=label):
                    with open(source, "rb") as handle:
                        original = handle.read()
                    out_buf, err_buf = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                        self.assertEqual(WN.main(["build", *prefix, "--out", source]), 1)
                    self.assertIn(f"would overwrite input {source}", err_buf.getvalue())
                    self.assertEqual(out_buf.getvalue(), "")
                    with open(source, "rb") as handle:
                        self.assertEqual(handle.read(), original)

            # A hard link to either a direct input or a discovered directory manifest is refused too.
            for label, prefix, source, link_name in (
                ("manifest_hardlink", ["--manifest", manifest], manifest, "manifest_link.json"),
                ("dir_manifest_hardlink", ["--manifest", os.path.join(tmp, "assets")], dir_manifest, "dir_link.json"),
            ):
                hardlink = os.path.join(tmp, link_name)
                try:
                    os.link(source, hardlink)
                except OSError:
                    self.skipTest("hard links unsupported on this filesystem")
                with self.subTest(input=label):
                    with open(source, "rb") as handle:
                        original = handle.read()
                    err_buf = io.StringIO()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_buf):
                        self.assertEqual(WN.main(["build", *prefix, "--out", hardlink]), 1)
                    self.assertIn(f"would overwrite input {source}", err_buf.getvalue())
                    with open(source, "rb") as handle:
                        self.assertEqual(handle.read(), original)


if __name__ == "__main__":
    unittest.main()
