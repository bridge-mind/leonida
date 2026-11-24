"""character_census: synthetic decoded models, manifest rows and the report-path guard.

    python -I tests/test_character_census.py

All model.json files, manifests, shaders, textures and counts here are built by the test. No game
data is read or written.
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "character_census.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("character_census_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CC = load_tool()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def skinned_model():
    return {
        "name": "synthetic_skinned",
        "type": "dr",
        "vertex_count": 12,
        "triangle_count": 4,
        "drawable": {
            "name": "synthetic_skinned",
            "skeleton": [
                {"name": "SKEL_ROOT"},
                {"name": "SKEL_Pelvis"},
                {"name": "SKEL_Head"},
            ],
            "models": [{"bone_palette": [1, 2, 3]}],
            "shaders": [{"name_hash": "210FBC6A",
                         "textures": [{"hash64": "AABBCCDDEEFF0011"}]}],
        },
        "materials": [
            {"shader_name_hash": "210FBC6A",
             "textures": [{"param_hash": "C8E8C282", "hash64": "AABBCCDDEEFF0011"}]},
            {"shader_name_hash": "DEADBEEF",
             "textures": [{"hash64": "0011223344556677"}]},
        ],
        "meshes": [
            {"name": "mesh0", "vertex_count": 8, "triangle_count": 2, "skinned": True,
             "skin_influences": 8, "bones": ["SKEL_ROOT", "SKEL_Pelvis", "SKEL_Head"]},
            {"name": "mesh1", "vertex_count": 4, "triangle_count": 2, "skinned": True,
             "skin_influences": 4},
        ],
    }


def static_model():
    return {
        "name": "synthetic_static",
        "type": "dr",
        "vertex_count": 6,
        "triangle_count": 2,
        "materials": [
            {"shader_name_hash": "12345678", "textures": [{"hash32": "89ABCDEF"}]},
        ],
        "meshes": [
            {"name": "static", "vertex_count": 6, "triangle_count": 2, "skinned": False},
        ],
    }


def fragment_model():
    return {
        "name": "synthetic_fragment",
        "type": "ft",
        "vertex_count": 25,
        "triangle_count": 32,
        "drawable": {
            "skeleton": [{"name": "SKEL_ROOT"}, {"name": "SKEL_Pelvis"}],
        },
        "materials": [{"shader_name_hash": "B5D47C70", "textures": []}],
        "meshes": [
            {"name": "bounds", "vertex_count": 25, "triangle_count": 32, "skinned": True,
             "skin_influences": 1},
        ],
    }


def partial_model():
    return {
        "name": "synthetic_partial",
        "type": "ft",
        "vertex_count": "many",
        "triangle_count": -1,
        "meshes": "not-a-list",
        "materials": {},
    }


class CharacterCensusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name) / "models"
        self.root.mkdir()
        write_json(self.root / "skinned" / "model.json", skinned_model())
        write_json(self.root / "static" / "model.json", static_model())
        write_json(self.root / "fragment" / "model.json", fragment_model())
        write_json(self.root / "partial" / "model.json", partial_model())
        broken = self.root / "broken" / "model.json"
        broken.parent.mkdir()
        broken.write_text('{"name": "synthetic_broken",', encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def report(self, *args):
        path = pathlib.Path(self.temp.name) / "report.json"
        result = self.run_tool(self.root, "--json", path, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(path.read_text(encoding="utf-8"))

    def by_name(self, data):
        return {row["name"]: row for row in data["models"]}

    def test_table_prints_counts_kinds_and_skins(self):
        result = self.run_tool(self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("character census: 5 models", result.stdout)
        self.assertIn("43 vertices (2 without)", result.stdout)
        self.assertIn("38 triangles (2 without)", result.stdout)
        self.assertIn("? indices", result.stdout)
        self.assertIn("8-influence skin: 1 yes, 3 no, 1 unknown", result.stdout)
        self.assertIn("by kind:", result.stdout)
        self.assertIn("by shader (a model with several materials counts under each):", result.stdout)
        self.assertIn("0x210FBC6A", result.stdout)
        self.assertIn("0xB5D47C70", result.stdout)
        self.assertIn("synthetic_skinned", result.stdout)
        self.assertIn("synthetic_fragment", result.stdout)
        self.assertIn("problems:", result.stdout)
        self.assertIn("cannot read model.json", result.stdout)

    def test_json_summary_and_rows_are_reproducible(self):
        data = self.report()
        summary = data["summary"]
        self.assertEqual(summary["models"], 5)
        self.assertEqual(summary["vertices"], 43)
        self.assertEqual(summary["triangles"], 38)
        self.assertEqual(summary["indices"], 0)
        self.assertEqual(summary["unknown_vertices"], 2)
        self.assertEqual(summary["unknown_triangles"], 2)
        self.assertEqual(summary["unknown_indices"], 5)
        self.assertEqual(summary["skin_8"], 1)
        self.assertEqual(summary["skin_unknown"], 1)
        self.assertEqual(summary["malformed"], 2)
        self.assertEqual(summary["shader_hashes"], 4)
        self.assertEqual(summary["texture_keys"], 3)
        self.assertEqual(list(summary["by_kind"]), ["drawable", "fragment", "unknown"])
        self.assertEqual(summary["by_kind"]["drawable"]["models"], 2)
        self.assertEqual(summary["by_kind"]["drawable"]["vertices"], 18)
        self.assertEqual(summary["by_kind"]["drawable"]["triangles"], 6)
        self.assertEqual(summary["by_kind"]["fragment"]["models"], 2)
        self.assertEqual(summary["by_kind"]["fragment"]["vertices"], 25)
        self.assertEqual(summary["by_kind"]["fragment"]["triangles"], 32)
        self.assertEqual(summary["by_shader"]["0x210FBC6A"],
                         {"models": 1, "vertices": 12, "triangles": 4})
        self.assertEqual(summary["by_shader"]["0xDEADBEEF"],
                         {"models": 1, "vertices": 12, "triangles": 4})
        self.assertEqual(summary["by_shader"]["0x12345678"],
                         {"models": 1, "vertices": 6, "triangles": 2})
        self.assertEqual(summary["by_shader"]["0xB5D47C70"],
                         {"models": 1, "vertices": 25, "triangles": 32})

        rows = self.by_name(data)
        self.assertEqual(sorted(rows), ["broken", "synthetic_fragment", "synthetic_partial",
                                        "synthetic_skinned", "synthetic_static"])
        skinned = rows["synthetic_skinned"]
        self.assertEqual(skinned["kind"], "drawable")
        self.assertEqual(skinned["vertices"], 12)
        self.assertEqual(skinned["triangles"], 4)
        self.assertIsNone(skinned["indices"])
        self.assertEqual(skinned["bones"], 3)
        self.assertIs(skinned["skin_8"], True)
        self.assertEqual(skinned["shaders"], ["0x210FBC6A", "0xDEADBEEF"])
        self.assertEqual(skinned["textures"], ["0x0011223344556677", "0xAABBCCDDEEFF0011"])
        self.assertEqual(skinned["issues"], [])

        static = rows["synthetic_static"]
        self.assertEqual(static["bones"], 0)
        self.assertIs(static["skin_8"], False)
        self.assertEqual(static["textures"], ["0x89ABCDEF"])

        fragment = rows["synthetic_fragment"]
        self.assertEqual(fragment["kind"], "fragment")
        self.assertEqual(fragment["bones"], 2)
        self.assertIs(fragment["skin_8"], False)

        broken = rows["broken"]
        self.assertEqual(broken["kind"], "unknown")
        self.assertIsNone(broken["vertices"])
        self.assertTrue(any("cannot read model.json" in issue for issue in broken["issues"]))

        partial = rows["synthetic_partial"]
        self.assertEqual(partial["kind"], "fragment")
        self.assertEqual(partial["bones"], 0)
        self.assertIs(partial["skin_8"], False)
        for text in ("meshes is not a list", "materials is not a list",
                     "vertex count is not a non-negative integer", "missing vertex count",
                     "triangle count is not a non-negative integer", "missing triangle count"):
            self.assertTrue(any(text in issue for issue in partial["issues"]), text)

    def test_mesh_totals_require_all_contributions_and_derive_triangles_per_mesh(self):
        root = pathlib.Path(self.temp.name) / "mesh_counts"
        write_json(root / "model.json", {"name": "mixed", "meshes": [
            {"vertex_count": 5, "triangle_count": 2}, {"indices": [0, 1, 2]},
        ]})
        out = pathlib.Path(self.temp.name) / "counts.json"
        result = self.run_tool(root, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text(encoding="utf-8"))
        row, = report["models"]
        self.assertIsNone(row["vertices"])
        self.assertIsNone(row["indices"])
        self.assertEqual(row["triangles"], 3)
        self.assertEqual(report["summary"]["unknown_vertices"], 1)
        self.assertEqual(report["summary"]["unknown_indices"], 1)
        self.assertEqual(report["summary"]["by_kind"]["component"]["triangles"], 3)

    def test_indices_from_separate_meshes_do_not_form_a_triangle(self):
        row = CC.model_row({"meshes": [{"index_count": 2}, {"index_count": 2}]}, "model.json")
        self.assertEqual(row["indices"], 4)
        self.assertEqual(row["triangles"], 0)

    def test_optional_index_count_is_counted_when_an_input_supplies_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "one"
            write_json(root / "model.json", {
                "name": "synthetic_indexed",
                "type": "dr",
                "vertex_count": 4,
                "triangle_count": 1,
                "index_count": 12,
                "materials": [],
                "meshes": [{"name": "m", "vertex_count": 4, "triangle_count": 1}],
            })
            out = pathlib.Path(td) / "report.json"
            result = self.run_tool(root, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["indices"], 12)
            self.assertEqual(data["summary"]["unknown_indices"], 0)
            self.assertEqual(data["models"][0]["indices"], 12)

    def test_csv_has_one_row_per_model(self):
        path = pathlib.Path(self.temp.name) / "report.csv"
        result = self.run_tool(self.root, "--csv", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = {row["name"]: row for row in csv.DictReader(handle)}
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows["synthetic_skinned"]["skin_8"], "true")
        self.assertEqual(rows["synthetic_static"]["skin_8"], "false")
        self.assertEqual(rows["broken"]["skin_8"], "")
        self.assertEqual(rows["synthetic_skinned"]["shaders"], "0x210FBC6A;0xDEADBEEF")
        self.assertEqual(rows["synthetic_skinned"]["textures"],
                         "0x0011223344556677;0xAABBCCDDEEFF0011")
        self.assertEqual(rows["synthetic_static"]["vertices"], "6")
        self.assertEqual(rows["broken"]["vertices"], "")

    def test_json_and_csv_must_differ(self):
        out = pathlib.Path(self.temp.name) / "report.bin"
        result = self.run_tool(self.root, "--json", out, "--csv", out)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("must use different paths", result.stderr)

    def test_report_path_cannot_overwrite_a_nested_model(self):
        target = self.root / "skinned" / "model.json"
        before = target.read_text(encoding="utf-8")
        result = self.run_tool(self.root, "--json", target)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("would overwrite the input", result.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), before)

    def test_report_path_alias_is_refused(self):
        target = self.root / "skinned" / "model.json"
        before = target.read_text(encoding="utf-8")
        alias = self.root / "sub" / ".." / "skinned" / "model.json"
        result = self.run_tool(self.root, "--csv", alias)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("would overwrite the input", result.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), before)

    def test_report_path_cannot_be_a_directory(self):
        result = self.run_tool(self.root, "--json", self.root)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("is a directory", result.stderr)

    def test_report_write_failure_names_the_destination(self):
        blocker = pathlib.Path(self.temp.name) / "file"
        blocker.write_bytes(b"keep")
        for flag in ("--json", "--csv"):
            with self.subTest(flag=flag):
                target = blocker / ("report." + flag[2:])
                result = self.run_tool(self.root, flag, target)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(f"cannot write {target}", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(blocker.read_bytes(), b"keep")

    def test_report_inside_the_tree_but_not_an_input_is_allowed(self):
        out = self.root / "report.json"
        result = self.run_tool(self.root, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(out.is_file())
        self.assertTrue((self.root / "skinned" / "model.json").is_file())

    def test_manifest_items_are_counted_without_guessing_bones(self):
        manifest = pathlib.Path(self.temp.name) / "manifest.json"
        write_json(manifest, {
            "src": "synthetic",
            "Count": 2,
            "Items": [
                {"Name": "manifest_fragment.#ft", "File": "fragment.obj", "Verts": 25,
                 "Tris": 32, "IsFragment": True, "HasBoneBindings": True,
                 "Materials": [{"Shader": "B5D47C70", "Textures": []}]},
                {"Name": "manifest_drawable.#dr", "File": "drawable.obj", "Verts": 10,
                 "Tris": 4, "IsFragment": False, "HasBoneBindings": False,
                 "Materials": [{"Shader": "210FBC6A",
                                "Textures": [{"Id64": "AABBCCDDEEFF0011"}]}]},
            ],
        })
        out = pathlib.Path(self.temp.name) / "manifest-report.json"
        result = self.run_tool(manifest, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text(encoding="utf-8"))
        summary = data["summary"]
        self.assertEqual(summary["models"], 2)
        self.assertEqual(summary["vertices"], 35)
        self.assertEqual(summary["triangles"], 36)
        self.assertEqual(summary["malformed"], 0)
        self.assertEqual(summary["skin_8"], 0)
        self.assertEqual(summary["skin_unknown"], 1)
        self.assertEqual(summary["by_kind"]["fragment"]["models"], 1)
        self.assertEqual(summary["by_kind"]["drawable"]["models"], 1)
        self.assertEqual(summary["by_shader"]["0xB5D47C70"]["vertices"], 25)
        self.assertEqual(summary["by_shader"]["0x210FBC6A"]["vertices"], 10)
        rows = self.by_name(data)
        fragment = rows["manifest_fragment.#ft"]
        self.assertEqual(fragment["kind"], "fragment")
        self.assertIsNone(fragment["bones"])
        self.assertIsNone(fragment["skin_8"])
        drawable = rows["manifest_drawable.#dr"]
        self.assertEqual(drawable["kind"], "drawable")
        self.assertEqual(drawable["bones"], 0)
        self.assertIs(drawable["skin_8"], False)
        self.assertEqual(drawable["textures"], ["0xAABBCCDDEEFF0011"])

    def test_manifest_models_list_resolves_paths(self):
        manifest = self.root / "models.json"
        write_json(manifest, {"models": ["skinned/model.json", {"path": "static/model.json"}]})
        out = pathlib.Path(self.temp.name) / "models-report.json"
        result = self.run_tool(manifest, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["models"], 2)
        self.assertEqual(data["summary"]["vertices"], 18)
        self.assertEqual(data["summary"]["triangles"], 6)
        self.assertEqual({row["name"] for row in data["models"]},
                         {"synthetic_skinned", "synthetic_static"})
        self.assertTrue(all("models.json#" in row["source"] for row in data["models"]))

    def test_manifest_report_path_cannot_overwrite_a_referenced_model(self):
        manifest = self.root / "models.json"
        write_json(manifest, {"models": ["skinned/model.json", "static/model.json"]})
        target = self.root / "skinned" / "model.json"
        before = target.read_text(encoding="utf-8")
        result = self.run_tool(manifest, "--json", target)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("would overwrite the input", result.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), before)

    def test_manifest_without_bone_binding_flag_reports_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = pathlib.Path(td) / "manifest.json"
            write_json(manifest, {"Items": [
                {"Name": "old_row.#dr", "Verts": 4, "Tris": 2, "IsFragment": False,
                 "Materials": [{"Shader": "12345678", "Textures": []}]},
            ]})
            out = pathlib.Path(td) / "report.json"
            result = self.run_tool(manifest, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            row = data["models"][0]
            self.assertIsNone(row["bones"])
            self.assertIsNone(row["skin_8"])
            self.assertIn("HasBoneBindings is missing", row["issues"])
            self.assertEqual(data["summary"]["skin_unknown"], 1)

    def test_section_table_marks_eight_influence_without_mesh_field(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "one"
            write_json(root / "model.json", {
                "name": "synthetic_sections",
                "type": "dr",
                "vertex_count": 3,
                "triangle_count": 1,
                "drawable": {
                    "models": [{"bone_palette": [1, 2]}],
                    "buffers": [{"mesh_blob": {"sections": [
                        [6, 0x100, 0x26], [7, 0x110, 0x26],
                        [8, 0x120, 0x25], [9, 0x130, 0x25],
                    ]}}],
                },
                "meshes": [{"name": "m", "vertex_count": 3, "triangle_count": 1,
                            "skinned": True}],
            })
            out = pathlib.Path(td) / "report.json"
            result = self.run_tool(root, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(out.read_text(encoding="utf-8"))["models"][0]
            self.assertIs(row["skin_8"], True)
            self.assertEqual(row["bones"], 2)

    def test_single_model_file_is_a_census(self):
        target = self.root / "static" / "model.json"
        out = pathlib.Path(self.temp.name) / "single.json"
        result = self.run_tool(target, "--json", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["models"], 1)
        self.assertEqual(data["models"][0]["name"], "synthetic_static")

    def test_model_json_saved_with_a_utf8_bom_is_read(self):
        # A model.json saved by a Windows editor starts with a UTF-8 BOM. Read as plain utf-8, json.load refuses it
        # ("Unexpected UTF-8 BOM"), and the census counted the model as unknown with an issue instead of reading it.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "models"
            model = root / "bom" / "model.json"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"\xef\xbb\xbf" + json.dumps(static_model()).encode("utf-8"))
            out = pathlib.Path(tmp) / "report.json"
            result = self.run_tool(root, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["models"], 1)
            row = data["models"][0]
            self.assertEqual(row["name"], "synthetic_static")
            self.assertEqual(row["issues"], [])

    def test_signed_and_underscore_hashes_are_not_hashes(self):
        # Python's int(text, 16) accepts a sign and underscores; the documented grammar is hex digits only.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "models"
            model = {
                "name": "synthetic_bad_hashes",
                "type": "dr",
                "vertex_count": 0,
                "triangle_count": 0,
                "materials": [
                    {"shader_name_hash": "-1", "textures": [{"hash32": "+1"}]},
                    {"shader_name_hash": "+1", "textures": [{"hash32": "12_34"}]},
                    {"shader_name_hash": "12_34", "textures": [{"hash32": "-1"}]},
                ],
            }
            write_json(root / "bad" / "model.json", model)
            out = pathlib.Path(tmp) / "report.json"
            result = self.run_tool(root, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            row = self.by_name(data)["synthetic_bad_hashes"]
            self.assertEqual(row["shaders"], [])
            self.assertEqual(row["textures"], [])
            shader_issues = [i for i in row["issues"] if "is not an 8-digit shader hash" in i]
            texture_issues = [i for i in row["issues"] if "is not a hexadecimal texture key" in i]
            self.assertEqual(len(shader_issues), 3, row["issues"])
            self.assertEqual(len(texture_issues), 3, row["issues"])

    def test_missing_target_is_a_clean_error(self):
        result = self.run_tool(pathlib.Path(self.temp.name) / "absent")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_unlistable_subfolder_is_a_clean_error_not_a_silent_undercount(self):
        # os.walk skips a directory it cannot list unless onerror raises; the census used to report
        # the remaining models and exit 0, hiding the incomplete scan.
        sub = self.root / "static"
        os.chmod(sub, 0o000)
        try:
            try:
                os.listdir(sub)
                self.skipTest("the sub-folder is still listable (running as root, or Windows)")
            except OSError:
                pass
            result = self.run_tool(self.root)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("ERROR", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        finally:
            os.chmod(sub, 0o755)

    def test_unlistable_subfolder_is_a_clean_error_when_chmod_is_not_enough(self):
        # The chmod case above self-skips on Windows and as root, and CI only runs this file on
        # windows-latest, so the guard went unexercised there. Forcing os.scandir to raise for one
        # folder gives the same PermissionError os.walk hits for real, on every platform.
        sub = self.root / "static"
        real_scandir = os.scandir
        locked = os.path.normcase(os.path.abspath(str(sub)))

        def scandir(path, *args, **kwargs):
            if os.path.normcase(os.path.abspath(str(path))) == locked:
                raise PermissionError(13, "Access is denied", str(path))
            return real_scandir(path, *args, **kwargs)

        err = io.StringIO()
        out = io.StringIO()
        with mock.patch.object(CC.os, "scandir", scandir):
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = CC.main([str(self.root)])
        self.assertEqual(code, 2, err.getvalue())
        self.assertIn("ERROR", err.getvalue())
        self.assertIn("cannot read", err.getvalue())
        self.assertIn(str(sub), err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
