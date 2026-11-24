"""Test graphics-wave width decisions and a counts-only report without game data or a GPU."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "semantic"))

import rdna2spirv  # noqa: E402
from support import translate_lines  # noqa: E402
from wave64_pipeline_audit import Assessment, Device, STAGE_BITS, assess  # noqa: E402


def shader(stage="ps", width=64, tier="C", **fields):
    return {"stage": stage, "wave64": width == 64, "subgroup_size": width, "wave64_graphics": tier, **fields}


CAPS32 = Device(32, 32, 32, sum(STAGE_BITS.values()), 0x20, True, 16)
CAPS64 = replace(CAPS32, default=64, minimum=64, maximum=64)
HYBRID = replace(CAPS32, maximum=64, required=0x30)
DEVICE_JSON = {
    "subgroup_size": 32,
    "min_subgroup_size": 32,
    "max_subgroup_size": 32,
    "supported_subgroup_stages": sum(STAGE_BITS.values()),
    "required_subgroup_size_stages": 0x20,
    "subgroup_size_control": True,
    "max_compute_workgroup_subgroups": 16,
}


class DecisionTest(unittest.TestCase):
    def test_actual_translator_marks_a_cross_half_pixel_shader(self):
        asm = translate_lines("ps_audit_readlane", [
            "s_mov_b64 s[10:11], exec",
            "v_readlane_b32 s6, v0, 40",
            "v_mov_b32_e32 v6, s6",
            "v_mov_b32_e32 v7, 0",
            "exp mrt0 v6, v7, v7, v7 done vm",
            "s_endpgm",
        ])
        entry = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
        self.assertEqual(entry["wave64_graphics"], "C")
        self.assertEqual(assess(entry, CAPS32).outcome, "approximate")
        self.assertEqual(assess(entry, HYBRID).outcome, "exact")

    def test_native_fragment_wave64_does_not_imply_native_vertex_or_geometry(self):
        for stage, expected in (("vs", "approximate"), ("gs", "approximate"), ("ps", "exact")):
            with self.subTest(stage=stage):
                self.assertEqual(assess(shader(stage), HYBRID).outcome, expected)

    def test_tier_and_metadata_distinguish_cross_half_dependence(self):
        for stage in ("vs", "gs", "ps"):
            for tier, want in (
                ("A", Assessment("partition-invariant", "tier-A")),
                ("B", Assessment("partition-invariant", "tier-B-slot-order-can-change")),
                ("C", Assessment("approximate", "tier-C-cross-half-dependence")),
                (None, Assessment("unknown", "missing-partition-analysis")),
            ):
                with self.subTest(stage=stage, tier=tier):
                    self.assertEqual(assess(shader(stage, tier=tier, wave64_emulated=True), CAPS32), want)
                    self.assertEqual(assess(shader(stage, tier=tier), CAPS64).outcome, "exact")

    def test_supported_stage_and_enabled_feature_are_both_required(self):
        for caps, expected in (
            (replace(HYBRID, control=False), "approximate"),
            (replace(HYBRID, supported=0x20), "unsupported"),
            (replace(HYBRID, default=64, control=False), "default-width"),
        ):
            with self.subTest(caps=caps):
                self.assertEqual(assess(shader(), caps).outcome, expected)

    def test_wave32_on_fixed_wave64_and_other_default_sizes(self):
        self.assertEqual(assess(shader(width=32), CAPS64).outcome, "unsupported")
        self.assertEqual(assess(shader(width=32), CAPS32).outcome, "exact")
        self.assertEqual(assess(shader(tier="A"), replace(CAPS32, default=8, minimum=8, maximum=8)).outcome,
                         "unsupported")

    def test_mesh_workgroup_limit_and_mesh_compute(self):
        native = replace(HYBRID, required=0xA0, max_workgroup_subgroups=2)
        self.assertEqual(assess(shader("ms", local_size=[128, 1, 1]), native).outcome, "exact")
        self.assertEqual(assess(shader("ms", local_size=[128, 2, 1]), native).outcome, "unsupported")
        self.assertEqual(assess(shader("ms"), native).reason, "missing-workgroup-size")
        entry = shader("ms", wave64_exchange="workgroup")
        self.assertEqual(assess(entry, CAPS32), Assessment("exact", "workgroup-exchange"))
        lowered = dict(entry, mesh_compute={"stride": 32})
        self.assertEqual(assess(lowered, CAPS32).outcome, "exact")
        self.assertEqual(assess(lowered, CAPS32, in_chain=True).reason, "mesh-compute-is-not-a-graphics-stage")

    def test_legacy_and_inconsistent_metadata(self):
        self.assertEqual(assess({"stage": "vs", "wave64": True}, CAPS64).outcome, "exact")
        self.assertEqual(assess({"stage": "vs"}, CAPS64).reason, "missing-wave-width")
        self.assertEqual(assess({"stage": "vs", "wave64": True, "subgroup_size": 32}, CAPS32).reason,
                         "conflicting-wave-width")

    def test_capability_input_requires_actual_stage_masks(self):
        self.assertEqual(Device.read(DEVICE_JSON), CAPS32)
        for bad in ({}, dict(DEVICE_JSON, subgroup_size=48),
                    dict(DEVICE_JSON, min_subgroup_size=64),
                    dict(DEVICE_JSON, supported_subgroup_stages=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Device.read(bad)


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.index = self.dir / "index.json"
        self.manifest = self.dir / "pairs.json"
        self.caps = self.dir / "caps.json"
        self.report = self.dir / "summary.json"
        self.dump(self.caps, DEVICE_JSON)
        self.dump(self.index, {"private.vertex": shader("vs", tier="A"),
                               "private.pixel": shader(tier="C"),
                               "private.fallback": shader(tier=None),
                               "known32": shader(width=32),
                               "unused_compute": shader("cs")})
        self.dump(self.manifest, {"version": 1, "pipelines": [
            {"name": "first private pair", "vs": "private.vertex", "ps": "private.pixel"},
            {"name": "second private pair", "vs": "private.vertex", "ps": "known32"},
            {"name": "third private pair", "vs": "private.vertex", "ps": "private.pixel"},
        ]})

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")

    def run_audit(self, *args):
        return subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "wave64_pipeline_audit.py"),
                               self.index, "--device-info", self.caps, *args],
                              capture_output=True, text=True, timeout=20)

    def test_whole_index_and_strict_gate(self):
        res = self.run_audit("--strict", "--report", self.report)
        self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
        self.assertIn("shaders (4)", res.stdout)
        doc = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertIsNone(doc["pipelines"])
        self.assertEqual(doc["shaders"]["approximate"], 1)
        self.assertEqual(doc["shaders"]["unknown"], 1)
        self.assertNotIn("cs", doc["stages"])

    def test_shared_manifest_counts_shaders_once_and_counts_each_chain(self):
        res = self.run_audit("--manifest", self.manifest, "--report", self.report)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        text = self.report.read_text(encoding="utf-8")
        doc = json.loads(text)
        self.assertIn("shaders (3)", res.stdout)
        self.assertEqual(doc["pipelines"]["approximate"], 2)
        self.assertEqual(doc["pipelines"]["partition-invariant"], 1)
        self.assertEqual(doc["shaders"]["approximate"], 1)
        self.assertNotIn("private", text + res.stdout + res.stderr)

    def test_strict_passes_a_chain_with_partition_invariance(self):
        self.dump(self.manifest, {"pipelines": [
            {"name": "partitioned", "vs": "private.vertex", "ps": "known32"}
        ]})
        proc = self.run_audit("--manifest", self.manifest, "--strict")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_wrong_stage_in_manifest_fails_before_report_creation(self):
        self.dump(self.manifest, {"pipelines": [
            {"name": "pair", "vs": "private.pixel", "ps": "known32"}
        ]})
        proc = self.run_audit("--manifest", self.manifest, "--report", self.report)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("stage mismatch", proc.stderr)
        self.assertFalse(self.report.exists())

    def test_report_cannot_overwrite_inputs(self):
        spv = self.dir / "private.pixel.spv"
        spv.write_bytes(b"unread")
        for path in (self.index, self.caps, self.manifest, spv):
            with self.subTest(path=path):
                proc = self.run_audit("--manifest", self.manifest, "--report", path)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("would overwrite", proc.stderr)
        self.assertEqual(spv.read_bytes(), b"unread")

    def test_empty_selection_is_not_a_strict_pass(self):
        self.dump(self.index, {"only_compute": shader("cs")})
        proc = self.run_audit("--strict")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("no graphics shaders", proc.stderr)


if __name__ == "__main__":
    unittest.main()
