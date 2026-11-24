"""Exercise the GPU atlas with synthetic vulkaninfo reports."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import turing_atlas

MATRIX_KEYS = ("vulkaninfo", "device", "subgroup", "subgroup_size_control", "mesh",
               "barycentric", "maintenance8", "descriptor_heap", "storageInputOutput16")


def report():
    return {"capabilities": {"device": {
        "extensions": {"VK_EXT_mesh_shader": 1},
        "features": {},
        "properties": {
            "VkPhysicalDeviceProperties": {"deviceName": "Example GPU", "driverVersion": 7, "apiVersion": 42},
        },
    }}}


class VulkanMatrixTests(unittest.TestCase):
    def probe(self, document, returncode=0):
        def run(cmd, cwd):
            Path(cwd, "VP_VULKANINFO_Example.json").write_text(json.dumps(document), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode)

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                return turing_atlas.vulkan_matrix()

    def assert_unknown(self, matrix):
        self.assertEqual(matrix, dict.fromkeys(MATRIX_KEYS))

    def test_recognized_profile(self):
        doc = report()
        device = doc["capabilities"]["device"]
        device["properties"]["VkPhysicalDeviceVulkan11Properties"] = {"subgroupSize": 32}
        device["properties"]["VkPhysicalDeviceVulkan13Properties"] = {"minSubgroupSize": 16, "maxSubgroupSize": 64}
        device["features"]["VkPhysicalDeviceVulkan11Features"] = {"storageInputOutput16": False}
        device["features"]["VkPhysicalDeviceVulkan13Features"] = {"subgroupSizeControl": True}
        matrix = self.probe(doc)
        self.assertEqual(tuple(matrix), MATRIX_KEYS)
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"], {"name": "Example GPU", "driver": 7, "api": 42})
        self.assertEqual(matrix["subgroup"]["size"], 32)
        self.assertEqual(matrix["subgroup"]["min"], 16)
        self.assertEqual(matrix["subgroup"]["max"], 64)
        self.assertIs(matrix["subgroup_size_control"]["subgroupSizeControl"], True)
        self.assertIs(matrix["mesh"], True)
        self.assertIs(matrix["barycentric"], False)
        self.assertIs(matrix["storageInputOutput16"], False)

    def test_older_device_without_versioned_structs(self):
        matrix = self.probe(report())
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertIsNone(matrix["subgroup"]["size"])
        self.assertIsNone(matrix["subgroup_size_control"]["subgroupSizeControl"])
        self.assertIsNone(matrix["storageInputOutput16"])

    def test_failed_subprocess_discards_even_a_valid_report(self):
        self.assert_unknown(self.probe(report(), returncode=1))

    def test_unusable_profiles_leave_the_matrix_unknown(self):
        no_name = report()
        del no_name["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"]["deviceName"]
        missing_extensions = report()
        del missing_extensions["capabilities"]["device"]["extensions"]
        invalid_extensions = report()
        invalid_extensions["capabilities"]["device"]["extensions"] = []
        invalid_feature_block = report()
        invalid_feature_block["capabilities"]["device"]["features"]["VkPhysicalDeviceVulkan11Features"] = []
        for doc in ([], {}, {"capabilities": []}, no_name, missing_extensions, invalid_extensions, invalid_feature_block):
            with self.subTest(document=doc):
                self.assert_unknown(self.probe(doc))

    def test_multiple_profiles_pick_native_discrete_over_igpu_dozen_and_corrupt(self):
        dozen = report()
        dozen["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Microsoft Direct3D12 (Discrete)",
            "vendorID": 0x1414,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })
        igpu = report()
        igpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Integrated Graphics",
            "vendorID": 0x1002,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        })
        dgpu = report()
        dgpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Native Discrete GPU",
            "vendorID": 0x1002,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })

        def run(cmd, cwd):
            Path(cwd, "VP_VULKANINFO_0_corrupt.json").write_text("{not valid json", encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_1_dozen.json").write_text(json.dumps(dozen), encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_2_igpu.json").write_text(json.dumps(igpu), encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_3_dgpu.json").write_text(json.dumps(dgpu), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0)

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                matrix = turing_atlas.vulkan_matrix()
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"]["name"], "Native Discrete GPU")

    def test_enumerates_gpus_from_summary_when_plain_j_only_writes_gpu0(self):
        llvmpipe = report()
        llvmpipe["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "llvmpipe (LLVM 19.1.0, 256 bits)",
            "vendorID": 0x10005,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_CPU",
        })
        dgpu = report()
        dgpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Discrete GPU 1",
            "vendorID": 0x10DE,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })
        calls = []

        def run(cmd, cwd):
            calls.append(tuple(cmd))
            if cmd == ["vulkaninfo", "-j"]:
                Path(cwd, "VP_VULKANINFO_llvmpipe.json").write_text(json.dumps(llvmpipe), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd == ["vulkaninfo", "--summary"]:
                out = "Devices:\n========\nGPU0:\n\tdeviceName = llvmpipe\nGPU1:\n\tdeviceName = Discrete GPU 1\n"
                return subprocess.CompletedProcess(cmd, 0, out, "")
            if cmd == ["vulkaninfo", "-j=1"]:
                Path(cwd, "VP_VULKANINFO_dgpu1.json").write_text(json.dumps(dgpu), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 1, "", "")

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                matrix = turing_atlas.vulkan_matrix()
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"]["name"], "Discrete GPU 1")
        self.assertEqual(calls, [("vulkaninfo", "-j"), ("vulkaninfo", "--summary"), ("vulkaninfo", "-j=1")])




class ValidationGateTests(unittest.TestCase):
    """The malformed control must fail, or the atlas results prove nothing. Tools are mocked."""

    def fake_run(self, accept_control=False, fail_assembly=False, fail_validation=False):
        def run(cmd, **kw):
            broken = any("broken_negative" in str(arg) for arg in cmd)
            if cmd[0] == "spirv-as":
                ok = 1 if fail_assembly and broken else 0
                return subprocess.CompletedProcess(cmd, ok, "", "syntax error" if ok else "")
            ok = 1 if fail_validation or (broken and not accept_control) else 0
            return subprocess.CompletedProcess(cmd, ok, "", "" if ok else "invalid module")
        return run

    def probe(self, accept_control=False, fail_assembly=False):
        with patch.object(turing_atlas.shutil, "which", return_value="tool"), \
                patch.object(turing_atlas, "run", self.fake_run(accept_control, fail_assembly)):
            return turing_atlas.validate_cases()

    def main_with(self, accept_control=False, fail_assembly=False, tools=True, argv=None, fail_validation=False):
        err = io.StringIO()
        with patch.object(sys, "argv", argv or ["turing_atlas.py"]), \
                patch.object(turing_atlas.shutil, "which", return_value="tool" if tools else None), \
                patch.object(turing_atlas, "run", self.fake_run(accept_control, fail_assembly, fail_validation)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = turing_atlas.main()
        return code, err.getvalue()

    def test_valid_cases_are_reported_normally_with_a_rejected_control(self):
        cases = self.probe()
        self.assertEqual(len(cases), len(turing_atlas.CASES))
        self.assertTrue(all(cases[n]["assemble"] and cases[n]["valid"] for n in cases
                            if n != "broken_negative"))
        self.assertTrue(cases["compute32"]["control_ok"])
        control = cases["broken_negative"]
        self.assertTrue(control["assemble"])
        self.assertFalse(control["valid"])
        self.assertTrue(control["control_ok"])
        self.assertNotIn("error", control)

    def test_assembly_rejection_of_the_control_also_counts(self):
        control = self.probe(fail_assembly=True)["broken_negative"]
        self.assertFalse(control["assemble"])
        self.assertTrue(control["control_ok"])
        self.assertNotIn("error", control)

    def test_an_accepted_control_is_flagged(self):
        control = self.probe(accept_control=True)["broken_negative"]
        self.assertFalse(control["control_ok"])
        self.assertIn("not rejecting malformed modules", control["error"])

    def test_main_exits_nonzero_when_the_control_was_accepted(self):
        code, err = self.main_with(accept_control=True)
        self.assertEqual(code, 2)
        self.assertIn("turing_atlas: error:", err)
        self.assertIn("broken_negative", err)

    def test_main_fails_when_the_validator_rejects_the_known_good_control(self) -> None:
        code, err = self.main_with(fail_validation=True)
        self.assertEqual(code, 2)
        self.assertIn("compute32", err)

    def test_main_succeeds_when_the_control_was_rejected(self):
        for kwargs in ({}, {"fail_assembly": True}):
            with self.subTest(**kwargs):
                code, err = self.main_with(**kwargs)
                self.assertEqual(code, 0)
                self.assertEqual(err, "")

    def test_out_in_a_missing_folder_creates_it(self):
        # #934's class: a --out whose parent does not exist used to lose the whole run to a traceback.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "dir", "atlas.json")
            code, err = self.main_with(argv=["turing_atlas.py", "--out", out])
            self.assertEqual((code, err), (0, ""))
            self.assertTrue(os.path.isfile(out))

    def test_reports_atlas_output_write_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "existing-file")
            with open(parent, "w", encoding="utf-8") as f:
                f.write("keep me")
            out = os.path.join(parent, "atlas.json")
            code, err = self.main_with(argv=["turing_atlas.py", "--out", out])

            self.assertEqual(code, 2)
            self.assertIn(f"cannot write {out}", err)
            with open(parent, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep me")

    def test_out_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self.main_with(argv=["turing_atlas.py", "--out", tmp])
            self.assertEqual(code, 2)
            self.assertIn("--out is a folder", err)

            trailing = os.path.join(tmp, "new_folder") + os.sep
            code, err = self.main_with(argv=["turing_atlas.py", "--out", trailing])
            self.assertEqual(code, 2)
            self.assertIn("--out is a folder", err)

            for empty_out in ("", "   "):
                with self.subTest(empty_out=repr(empty_out)):
                    code, err = self.main_with(argv=["turing_atlas.py", "--out", empty_out])
                    self.assertEqual(code, 2)
                    self.assertIn("turing_atlas: error:", err)


    def test_missing_tools_skip_without_treating_it_as_a_control_failure(self):
        with patch.object(turing_atlas.shutil, "which", return_value=None):
            self.assertEqual(turing_atlas.validate_cases(),
                             {"skipped": "spirv-as/spirv-val not on PATH (Vulkan SDK)"})
        code, err = self.main_with(tools=False)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
