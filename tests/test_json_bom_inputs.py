import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

__author__ = "it_komarov"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bcpack_history_calib as calibration
import bcpack_tables_bptc as history
import character_textures
import feature_scan
import gdk_runtime_calls
import ped_assemble
import shader_profile
import skinned_glb
import wave64_halves
import wave64_pipeline_audit as audit


class JsonBomInputTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def variants(self, document, name="input.json"):
        for encoding in ("utf-8", "utf-8-sig"):
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(document, ensure_ascii=False).encode(encoding))
            yield path

    def malformed(self, name="invalid.json"):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\xef\xbb\xbf{"value":"\xff"}')
        return path

    def call_main(self, main, arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = main(arguments)
        return result, out.getvalue(), err.getvalue()

    def test_character_textures_material_bindings(self):
        document = {"Name": "synthetic_model", "Materials": [{
            "Name": "head", "Shader": "a1b2c3d4",
            "Textures": [{"Param": "0xC8E8C282", "Name": "synthetic_diffuse"}],
        }]}
        with self.assertRaises(ValueError):
            character_textures.classify_file(str(self.malformed()))
        for path in self.variants(document):
            result = character_textures.classify_file(str(path))["models"]
            self.assertEqual(result[0]["name"], "synthetic_model")
            binding = result[0]["materials"][0]["bindings"][0]
            self.assertEqual(binding["role"], "diffusetex")
            self.assertEqual(binding["texture"], "synthetic_diffuse")

    def test_skinned_glb_model_json(self):
        document = {"Name": "synthetic_\u00e9", "Materials": [{"Name": "material"}]}
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb._load_json(str(self.malformed()))
        for path in self.variants(document, "model.json"):
            self.assertEqual(skinned_glb._load_json(str(path)), document)

    def test_gdk_runtime_class_and_interface_lookup(self):
        class_id = "00000000-0000-0000-0000-000000000001"
        interface_id = "00000000-0000-0000-0000-000000000002"
        document = {
            "source": "synthetic_\u00e9",
            "classes": {"Example": {"id": class_id, "implements": ["IExample"]}},
            "interfaces": {"IExample": {"id": interface_id, "class": "Example"}},
        }
        with self.assertRaises(ValueError):
            gdk_runtime_calls.RuntimeTable.load(str(self.malformed()))
        for path in self.variants(document):
            table = gdk_runtime_calls.RuntimeTable.load(str(path))
            self.assertEqual(table.source, "synthetic_\u00e9")
            self.assertEqual(table.ids(), {class_id, interface_id})
            self.assertEqual(table.label(interface_id), "IExample")
            self.assertEqual(table.class_of(interface_id), "Example")

    def test_shader_profile_preserves_stage_map(self):
        disasm = self.folder / "dxbc" / "disasm"
        disasm.mkdir(parents=True)
        document = [{"file": "opaque.code.dxbc", "stage": "ps"}]
        self.malformed("dxbc/index.json")
        self.assertEqual(shader_profile.stage_map(disasm), {})
        for _ in self.variants(document, "dxbc/index.json"):
            self.assertEqual(shader_profile.stage_map(disasm), {"opaque.code": "ps"})

    def test_wave64_pipeline_audit_index_and_saved_caps(self):
        document = {"synthetic": {"stage": "ps", "wave64": True, "wave64_graphics": "A"}}
        caps = {
            "subgroup_size": 32, "min_subgroup_size": 32, "max_subgroup_size": 32,
            "supported_subgroup_stages": 63, "required_subgroup_size_stages": 32,
            "subgroup_size_control": True, "max_compute_workgroup_subgroups": 32,
        }
        with self.assertRaises(ValueError):
            audit.load_json(self.malformed())
        for index in self.variants(document, "index.json"):
            for device in self.variants(caps, "caps.json"):
                result, out, err = self.call_main(audit.main, [
                    str(index), "--device-info", str(device), "--strict",
                ])
                self.assertEqual(result, 0, err)
                self.assertIn("shaders (1): partition-invariant=1", out)

    def test_bcpack_history_tables(self):
        bc6h = {"endpointHistory": {"7": [[1, 2, 3, 4]] * 3}, "endpointClass": {"7": 3}}
        bc7 = {"historyEndpoints": {"9": list(range(24))}}
        bc4 = {"codec": "bc4", "confirmed": {"11": [17, 23]}, "replace": True}
        for load in (history.load_bc6h, history.load_bc7,
                     lambda path: history.load_bc_history(4, path)):
            with self.assertRaises(ValueError):
                load(self.malformed())
        for path in self.variants({"bc6h": bc6h}, "bc6h.json"):
            loaded = history.load_bc6h(path)
            self.assertEqual(loaded[0][7], ((1, 2, 3, 4),) * 3)
            self.assertEqual(loaded[2][7], 3)
        for _ in self.variants(bc7, "bc7/bc7.json"):
            loaded = history.load_bc7(self.folder / "bc7")
            self.assertEqual(loaded["history_endpoints"][9], tuple(range(24)))
        for _ in self.variants(bc4, "bc4/bc4.json"):
            loaded = history.load_bc_history(4, self.folder / "bc4")
            self.assertEqual(loaded, ({11: (17, 23)}, {}, True))

    def test_wave64_halves_census_counts(self):
        result, _, err = self.call_main(wave64_halves.main, ["--census", str(self.malformed())])
        self.assertEqual(result, 2)
        self.assertIn("ERROR:", err)
        document = {"synthetic": {"stage": "vs", "wave64": True, "wave64_graphics": "B"}}
        for path in self.variants(document):
            result, out, err = self.call_main(wave64_halves.main, ["--census", str(path)])
            self.assertEqual(result, 0, err)
            self.assertIn("vs: B=1", out)
            self.assertNotIn("synthetic", out)

    def test_ped_assemble_recursive_mesh_manifest(self):
        mesh = {"name": "triangle", "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "triangles": [[0, 1, 2]]}
        with self.assertRaises(ValueError):
            ped_assemble.load_parts_from_path(str(self.malformed()))
        for path in self.variants(mesh, "mesh.json"):
            for manifest in self.variants({"parts": [path.name]}, "manifest.json"):
                parts, skeleton = ped_assemble.load_parts_from_path(str(manifest))
                self.assertIsNone(skeleton)
                self.assertEqual(len(parts), 1)
                self.assertEqual(parts[0].name, "triangle")
                self.assertEqual(parts[0].positions[1], (1.0, 0.0, 0.0))
                self.assertEqual(parts[0].triangles, [(0, 1, 2)])

    def test_feature_scan_carve_and_translation_indices(self):
        with self.assertRaises(ValueError):
            feature_scan.scan_spirv(str(self.malformed()))
        translated = {"synthetic": {"stage": "ps", "uses_code_base": True}}
        for path in self.variants(translated, "shaders/spirv/index.json"):
            self.assertEqual(feature_scan.scan_spirv(str(path)), {
                "translated": {"ps": 1}, "uses_code_base": {"ps": 1},
            })
            for carve in self.variants([], "shaders/dxbc/index.json"):
                result = subprocess.run([
                    sys.executable, "-B", "-I", str(ROOT / "tools" / "feature_scan.py"),
                    str(carve.parent.parent),
                ], capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("uses_code_base by stage: {'ps': 1} of {'ps': 1}", result.stdout)

    def test_bcpack_reference_pairs_dispatch(self):
        with self.assertRaises(ValueError):
            calibration.apply_ref_pairs_file(str(self.malformed()), {})
        document = {"pairs": [{"pool": "pool.ttpf", "dds": "reference.dds", "index": 2}]}
        calibrators = {4: object()}
        for path in self.variants(document):
            with mock.patch.object(calibration, "apply_ref_dds") as apply:
                calibration.apply_ref_pairs_file(str(path), calibrators)
                apply.assert_called_once_with(
                    str(self.folder / "pool.ttpf"), 2,
                    str(self.folder / "reference.dds"), calibrators,
                )


if __name__ == "__main__":
    unittest.main()
