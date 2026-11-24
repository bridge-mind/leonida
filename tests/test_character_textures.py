"""character_textures.py on synthetic decoded model JSON.

Run with ``python -I tests/test_character_textures.py``. Covers a head material (diffusetex +
bumptex), a garment material (speculartex only, must flag missing albedo), an unknown slot, the
JSON/CSV reports and the report-path guard. Every fixture is hand-written; no game data is used.
"""
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "character_textures.py"


def model(materials, name="synthetic_model"):
    return {"Name": name, "Materials": materials}


def write_model(folder, document, name="model.json"):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def run_tool(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=60)


HEADER = "model\tmaterial\tslot\trole\ttexture\tasset_key"


def table(output):
    """The data rows of the tab-separated table (header and FLAG lines excluded)."""
    return [line.split("\t") for line in output.splitlines()
            if line.count("\t") == 5 and line != HEADER]


class CharacterTextureTests(unittest.TestCase):
    def test_manifest_file_names_are_independent_of_the_export_host(self):
        entries = [
            {"File": r"C:\exports\models\cap.v2.mesh", "Materials": [{}]},
            {"file": "models/shoes.mesh", "materials": [{}]},
            {"File": r"models\unused.mesh", "Name": "named", "Materials": [{}]},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, {"Items": entries})
            report_path = pathlib.Path(td) / "report.json"
            result = run_tool(path, "--json", report_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = ["cap.v2", "shoes", "named"]
            self.assertEqual([row[0] for row in table(result.stdout)], expected)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual([m["name"] for m in report["models"]], expected)

    def test_head_material_reports_diffusetex_and_bumptex(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{
                "Name": "head",
                "Shader": "a1b2c3d4",
                "Textures": [
                    {"Param": "0xC8E8C282", "Name": "face_diffusetex"},
                    {"Param": "0x1E77BF4C", "Name": "face_bumptex"},
                ],
            }]))
            result = run_tool(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = {row[3]: row for row in table(result.stdout)}
            self.assertEqual(rows["diffusetex"], [
                "synthetic_model", "head", "0xC8E8C282", "diffusetex", "face_diffusetex", "-"])
            self.assertEqual(rows["bumptex"], [
                "synthetic_model", "head", "0x1E77BF4C", "bumptex", "face_bumptex", "-"])
            self.assertNotIn("FLAG", result.stdout)

    def test_garment_without_albedo_is_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{
                "Name": "jacket",
                "Shader": "D5ED50F4",
                "Textures": [{"Param": "0xC6F46E79", "Name": "jacket_speculartex"}],
            }]))
            result = run_tool(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = table(result.stdout)
            self.assertEqual(len(rows), 1, result.stdout)
            self.assertEqual(rows[0][2:], ["0xC6F46E79", "speculartex", "jacket_speculartex", "-"])
            self.assertIn("FLAG garment-no-albedo", result.stdout)
            self.assertIn("d5ed50f4", result.stdout)
            self.assertIn("speculartex is not colour", result.stdout)
            self.assertNotIn("diffusetex", result.stdout)

    def test_unknown_slot_is_not_guessed_from_its_name(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{
                "Name": "mystery",
                "Shader": "a1b2c3d4",
                "Textures": [{"Param": 0xDEADBEEF, "Name": "diffusetex"}],
            }]))
            result = run_tool(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = table(result.stdout)
            self.assertEqual(rows[0][2], "0xDEADBEEF")
            self.assertEqual(rows[0][3], "unknown 0xDEADBEEF")
            self.assertEqual(rows[0][4], "diffusetex")
            self.assertIn("unknown slots", result.stdout)

    def test_every_verified_slot_maps_to_its_name(self):
        slots = {
            "0xC8E8C282": "diffusetex",
            "0xC6F46E79": "speculartex",
            "0x1E77BF4C": "bumptex",
            "0x739E3D0D": "speculartex2",
            "0x8ADB8480": "materialid",
        }
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{
                "Name": "slots",
                "Shader": "a1b2c3d4",
                "Textures": [{"Param": slot, "Name": role} for slot, role in slots.items()],
            }]))
            result = run_tool(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({row[2]: row[3] for row in table(result.stdout)}, slots)
            self.assertFalse([row for row in table(result.stdout) if row[3].startswith("unknown")])

    def test_missing_albedo_outside_a_garment_shader_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{
                "Name": "trim",
                "Shader": "a1b2c3d4",
                "Textures": [{"Param": "0x1E77BF4C", "Name": "trim_normal"}],
            }]), name="model.json")
            result = run_tool(path, "--json", pathlib.Path(td) / "report.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("garment-no-albedo", result.stdout)
            report = json.loads((pathlib.Path(td) / "report.json").read_text(encoding="utf-8"))
            material = report["models"][0]["materials"][0]
            self.assertTrue(material["missing_albedo"])
            self.assertEqual(material["flags"], [])

    def test_json_and_csv_reports(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([
                {"Name": "head", "Shader": "a1b2c3d4", "Textures": [
                    {"Param": "C8E8C282", "Name": "face_diffusetex"}]},
                {"Name": "jacket", "Shader": "ea8d09d0", "Textures": [
                    {"Param": "C6F46E79", "Name": "jacket_speculartex"}]},
            ]))
            report_path = pathlib.Path(td) / "out" / "report.json"
            csv_path = pathlib.Path(td) / "out" / "report.csv"
            result = run_tool(path, "--json", report_path, "--csv", csv_path)
            self.assertEqual(result.returncode, 0, result.stderr)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["materials"], 2)
            self.assertEqual(report["summary"]["bindings"], 2)
            self.assertEqual(report["summary"]["garment_without_albedo"], 1)
            head, jacket = report["models"][0]["materials"]
            self.assertEqual(head["bindings"][0]["role"], "diffusetex")
            self.assertTrue(jacket["garment"])
            self.assertTrue(jacket["missing_albedo"])
            self.assertEqual(jacket["flags"], ["garment-no-albedo"])
            self.assertTrue(report["warnings"][0].startswith("garment-no-albedo"))

            data = csv_path.read_bytes()
            self.assertNotIn(b"\r", data)
            lines = data.decode("utf-8").splitlines()
            self.assertEqual(lines[0], "source,model,material,shader,slot,role,texture,flags,asset_key")
            self.assertIn(",head,a1b2c3d4,0xC8E8C282,diffusetex,face_diffusetex,", lines[1])
            self.assertIn(",jacket,ea8d09d0,0xC6F46E79,speculartex,jacket_speculartex,garment-no-albedo",
                          lines[2])

    def test_shared_name_hash_preserves_distinct_asset_keys_in_each_report(self):
        name_hash = "12345678"
        keys = ["0000000112345678", "FEDCBA9812345678"]
        refs = [{"Param": "C8E8C282", "TextureIndex": i, "Name": name_hash, "Id64": key}
                for i, key in enumerate(keys)]
        refs.append({"Param": "C8E8C282", "Name": name_hash})
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, {"Items": [model([{"Textures": refs}, {"Textures": []}])]})
            report_path, csv_path = pathlib.Path(td) / "report.json", pathlib.Path(td) / "report.csv"
            result = run_tool(path, "--json", report_path, "--csv", csv_path)
            self.assertEqual(result.returncode, 0, result.stderr)

            self.assertEqual(result.stdout.splitlines()[0], HEADER)
            self.assertEqual([row[4:] for row in table(result.stdout)],
                             [[name_hash, key] for key in [*keys, "-"]] + [["-", "-"]])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            bindings = report["models"][0]["materials"][0]["bindings"]
            expected = [(name_hash, key) for key in [*keys, None]]
            self.assertEqual([(b["texture"], b["asset_key"]) for b in bindings], expected)
            self.assertEqual([(r["texture"], r["asset_key"]) for r in report["rows"]], expected + [("-", None)])
            with csv_path.open(encoding="utf-8", newline="") as f:
                self.assertEqual([(r["texture"], r["asset_key"]) for r in csv.DictReader(f)],
                                 [(name_hash, key) for key in [*keys, ""]] + [("-", "")])

    def test_asset_key_requires_an_explicit_64_bit_identity(self):
        refs = [{"id64": "0xfedcba9876543210"}, {"AssetKey": 0}, {"asset_key": 2**64 - 1},
                {"Id64": "12345678"}, {"Id64": 2**64}, {"Id64": True},
                {"Key": "1234567800000001"}, "1234567800000002"]
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{"Textures": refs}]))
            report_path = pathlib.Path(td) / "report.json"
            result = run_tool(path, "--json", report_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(report_path.read_text(encoding="utf-8"))["rows"]
            self.assertEqual([r["asset_key"] for r in rows],
                             ["FEDCBA9876543210", "0000000000000000", "FFFFFFFFFFFFFFFF"] + [None] * 5)

    def test_directory_is_searched_recursively(self):
        with tempfile.TemporaryDirectory() as td:
            write_model(pathlib.Path(td) / "a" / "b", model([{
                "Name": "head", "Shader": "a1b2c3d4",
                "Textures": [{"Param": "0xC8E8C282", "Name": "face"}]}], name="first"))
            write_model(pathlib.Path(td) / "c", model([{
                "Name": "jacket", "Shader": "39ddb9ca",
                "Textures": [{"Param": "0xC6F46E79", "Name": "spec"}]}], name="second"))
            result = run_tool(td)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("first", result.stdout)
            self.assertIn("second", result.stdout)
            self.assertEqual(result.stdout.count("FLAG garment-no-albedo"), 1)

    def test_json_cannot_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{"Name": "head", "Shader": "a1b2c3d4", "Textures": []}]))
            before = path.read_bytes()
            result = run_tool(path, "--json", path)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(path.read_bytes(), before)

    def test_csv_cannot_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{"Name": "head", "Shader": "a1b2c3d4", "Textures": []}]))
            before = path.read_bytes()
            result = run_tool(path, "--csv", path)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(path.read_bytes(), before)

    def test_json_and_csv_must_use_different_paths(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_model(td, model([{"Name": "head", "Shader": "a1b2c3d4", "Textures": []}]))
            out = pathlib.Path(td) / "report.bin"
            result = run_tool(path, "--json", out, "--csv", out)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("must use different paths", result.stderr)

    def test_unlistable_subfolder_is_an_error_not_a_silent_skip(self):
        import io
        from unittest import mock
        import importlib.util
        spec = importlib.util.spec_from_file_location("ct_under_test", TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            write_model(pathlib.Path(td) / "a", model([{"Name": "m", "Materials": []}]))
            orig_walk = mod.os.walk
            def fail_walk(top, *args, **kwargs):
                if "onerror" in kwargs and kwargs["onerror"]:
                    kwargs["onerror"](PermissionError(13, "Permission denied"))
                return orig_walk(top, *args, **kwargs)
            with mock.patch.object(mod.os, "walk", side_effect=fail_walk):
                with mock.patch.object(sys, "argv", ["character_textures.py", td]):
                    with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                        code = mod.main()
                        self.assertEqual(code, 2)
                        self.assertIn("ERROR:", err.getvalue())
                        self.assertIn("Permission denied", err.getvalue())

    def test_unwritable_report_path_exits_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            write_model(pathlib.Path(td) / "a", model([{"Name": "m", "Materials": []}]))
            blocker = pathlib.Path(td) / "file_parent"
            blocker.write_text("x", encoding="utf-8")
            bad_json = blocker / "out.json"
            proc = subprocess.run([sys.executable, "-I", TOOL, td, "--json", str(bad_json)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ERROR: cannot write", proc.stderr)


if __name__ == "__main__":
    unittest.main()
