"""Synthetic unit tests for tools/bary_fallback_gs.py.

Standard library only; no GPU or game data. Index files are constructed in temporary
directories. The rebuild regression uses SPIRV-Tools when available.

    python -I tests/test_bary_fallback_gs.py
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
TOOL_PATH = os.path.join(TOOLS_DIR, "bary_fallback_gs.py")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import bary_fallback_gs  # noqa: E402


def _successful_spirv_tools(cmd: list[str], *, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
    if cmd[0] == "spirv-as":
        Path(cmd[-1]).write_bytes(b"synthetic binary")
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _parse_spvasm(text: str) -> dict:
    """Extract capabilities, entry point, decorations, types, vars, and emissions from GS text."""
    lines = text.splitlines()
    capabilities = []
    entry_points = []
    execution_modes = []
    decorations = []
    defined_ids = {}
    variables = {}
    constants = {}

    for line in lines:
        if line.startswith("OpCapability "):
            capabilities.append(line.split()[1])
        elif line.startswith("OpEntryPoint "):
            entry_points.append(line)
        elif line.startswith("OpExecutionMode "):
            execution_modes.append(line[len("OpExecutionMode "):])
        elif line.startswith("OpDecorate "):
            decorations.append(line)
        m = re.match(r"^(%\w+)\s*=\s*(Op\S+)(.*)$", line)
        if m:
            ident, op, rest = m.group(1), m.group(2), m.group(3).strip()
            if ident in defined_ids:
                raise AssertionError(f"duplicate SSA definition: {ident}")
            defined_ids[ident] = (op, rest)
            if op == "OpVariable":
                ptr_type, storage = rest.split()
                variables[ident] = (ptr_type, storage)
            elif op == "OpConstant":
                constants[ident] = rest

    body_start = lines.index("%entry = OpLabel") + 1
    body_end = lines.index("OpEndPrimitive")
    body_lines = lines[body_start:body_end]
    emissions = []
    current = []
    for line in body_lines:
        if line == "OpEmitVertex":
            emissions.append(current)
            current = []
        else:
            current.append(line)

    return {
        "lines": lines,
        "capabilities": capabilities,
        "entry_points": entry_points,
        "execution_modes": execution_modes,
        "decorations": decorations,
        "defined_ids": defined_ids,
        "variables": variables,
        "constants": constants,
        "emissions": emissions,
        "trailing_lines": lines[body_end:],
        "dangling_after_emit": current,
    }


class TestGsSpvasm(unittest.TestCase):
    def _assert_valid_gs_structure(self, text: str, expected_locs: list[int], bary_loc: int,
                                   clip_count: int = 0, cull_count: int = 0):
        self.assertTrue(text.endswith("\n"))
        parsed = _parse_spvasm(text)
        self.assertEqual(parsed["dangling_after_emit"], [])
        self.assertEqual(parsed["trailing_lines"], ["OpEndPrimitive", "OpReturn", "OpFunctionEnd"])
        self.assertIn("OpMemoryModel Logical GLSL450", parsed["lines"])

        expected_caps = ["Geometry"]
        if clip_count:
            expected_caps.append("ClipDistance")
        if cull_count:
            expected_caps.append("CullDistance")
        self.assertEqual(parsed["capabilities"], expected_caps)

        self.assertEqual(
            parsed["execution_modes"],
            ["%main Triangles", "%main Invocations 1", "%main OutputTriangleStrip", "%main OutputVertices 3"],
        )

        # Check OpEntryPoint interface order and that every interface variable exists with the right storage.
        self.assertEqual(len(parsed["entry_points"]), 1)
        ep_tokens = parsed["entry_points"][0].split()
        self.assertEqual(ep_tokens[:4], ["OpEntryPoint", "Geometry", "%main", '"main"'])
        iface = ep_tokens[4:]

        in_vars = [f"%in{n}" for n in expected_locs]
        out_vars = [f"%out{n}" for n in expected_locs]
        dist_vars = []
        if clip_count:
            dist_vars = ["%in_clip", "%out_clip"]
        elif cull_count:
            dist_vars = ["%in_cull", "%out_cull"]
        expected_iface = ["%in_pos", "%out_pos", *in_vars, *out_vars, "%out_bary", *dist_vars]
        self.assertEqual(iface, expected_iface)
        self.assertEqual(set(parsed["variables"]), set(expected_iface))

        for var in ["%in_pos", *in_vars]:
            self.assertEqual(parsed["variables"][var], ("%ptr_in_arr", "Input"))
        for var in ["%out_pos", *out_vars]:
            self.assertEqual(parsed["variables"][var], ("%ptr_out_v4", "Output"))
        self.assertEqual(parsed["variables"]["%out_bary"], ("%ptr_out_v3", "Output"))

        # Check decorations match the declared variables and locations.
        expected_decs = [
            "OpDecorate %in_pos BuiltIn Position",
            "OpDecorate %out_pos BuiltIn Position",
        ]
        if clip_count:
            expected_decs.extend([
                "OpDecorate %in_clip BuiltIn ClipDistance",
                "OpDecorate %out_clip BuiltIn ClipDistance",
            ])
        elif cull_count:
            expected_decs.extend([
                "OpDecorate %in_cull BuiltIn CullDistance",
                "OpDecorate %out_cull BuiltIn CullDistance",
            ])
        expected_decs.extend(f"OpDecorate %in{n} Location {n}" for n in expected_locs)
        expected_decs.extend(f"OpDecorate %out{n} Location {n}" for n in expected_locs)
        expected_decs.append(f"OpDecorate %out_bary Location {bary_loc}")
        self.assertEqual(parsed["decorations"], expected_decs)

        # Check one-hot barycentric constants.
        self.assertEqual(parsed["constants"]["%f0"], "%float 0")
        self.assertEqual(parsed["constants"]["%f1"], "%float 1")
        self.assertIn("%bary0 = OpConstantComposite %v3float %f1 %f0 %f0", parsed["lines"])
        self.assertIn("%bary1 = OpConstantComposite %v3float %f0 %f1 %f0", parsed["lines"])
        self.assertIn("%bary2 = OpConstantComposite %v3float %f0 %f0 %f1", parsed["lines"])

        # Check distance type declarations and storage classes when active.
        if clip_count or cull_count:
            kind = "cull" if cull_count else "clip"
            count = clip_count or cull_count
            self.assertEqual(parsed["constants"][f"%{kind}_count"], f"%uint {count}")
            self.assertIn(f"%{kind} = OpTypeArray %float %{kind}_count", parsed["lines"])
            self.assertIn(f"%triangle_{kind} = OpTypeArray %{kind} %uint_3", parsed["lines"])
            self.assertEqual(parsed["variables"][f"%in_{kind}"], (f"%ptr_in_triangle_{kind}", "Input"))
            self.assertEqual(parsed["variables"][f"%out_{kind}"], (f"%ptr_out_{kind}", "Output"))

        # Check exact pass-through order per emitted triangle vertex.
        self.assertEqual(len(parsed["emissions"]), 3)
        pairs = (("%in_pos", "%out_pos"), *zip(in_vars, out_vars))
        for v, emission in enumerate(parsed["emissions"]):
            expected_emission = []
            for src, dst in pairs:
                stem = src[1:]
                expected_emission.extend([
                    f"%p{v}_{stem} = OpAccessChain %ptr_in_v4 {src} %int_{v}",
                    f"%x{v}_{stem} = OpLoad %v4float %p{v}_{stem}",
                    f"OpStore {dst} %x{v}_{stem}",
                ])
            if clip_count or cull_count:
                kind = "cull" if cull_count else "clip"
                expected_emission.extend([
                    f"%{kind}_ptr{v} = OpAccessChain %ptr_in_{kind} %in_{kind} %int_{v}",
                    f"%distances{v} = OpLoad %{kind} %{kind}_ptr{v}",
                    f"OpStore %out_{kind} %distances{v}",
                ])
            expected_emission.append(f"OpStore %out_bary %bary{v}")
            self.assertEqual(emission, expected_emission)

        # Verify every %id referenced anywhere in the module is defined.
        for line in parsed["lines"]:
            if re.match(r"^%\w+\s*=", line):
                rhs = line.split("=", 1)[1]
                refs = re.findall(r"%\w+", rhs)
            else:
                refs = re.findall(r"%\w+", line)
            for ref in refs:
                self.assertIn(ref, parsed["defined_ids"], f"undefined SSA ID {ref} in line: {line}")

    def test_empty_locations(self):
        asm = bary_fallback_gs.gs_spvasm([], 0)
        self._assert_valid_gs_structure(asm, [], 0)

        asm_loc5 = bary_fallback_gs.gs_spvasm((), 5)
        self._assert_valid_gs_structure(asm_loc5, [], 5)

    def test_locations_sorted_deduplicated_and_generator_input(self):
        canonical = bary_fallback_gs.gs_spvasm([1, 3, 5], 2)
        self._assert_valid_gs_structure(canonical, [1, 3, 5], 2)
        self.assertEqual(bary_fallback_gs.gs_spvasm([5, 1, 5, 3, 1], 2), canonical)
        self.assertEqual(bary_fallback_gs.gs_spvasm((x for x in (5, 3, 1, 3)), 2), canonical)
        self.assertEqual(bary_fallback_gs.gs_spvasm(["5", "1", "3"], 2), canonical)

    def test_sparse_and_gap_bary_locations(self):
        self._assert_valid_gs_structure(bary_fallback_gs.gs_spvasm([30], 0), [30], 0)
        self._assert_valid_gs_structure(bary_fallback_gs.gs_spvasm([0, 2], 1), [0, 2], 1)

    def test_bary_location_collision_raises(self):
        for locs, bary in (([0], 0), ([2, 0, 5], 2), ([4, 1, 4], 1), ((n for n in [3, 7]), 7)):
            with self.subTest(locs=locs, bary=bary):
                with self.assertRaisesRegex(ValueError, "collides"):
                    bary_fallback_gs.gs_spvasm(locs, bary)

    def test_clip_distance_counts_1_to_8(self):
        for count in range(1, 9):
            with self.subTest(clip_count=count):
                asm = bary_fallback_gs.gs_spvasm([1, 4], 0, clip_count=count)
                self._assert_valid_gs_structure(asm, [1, 4], 0, clip_count=count)
                self.assertNotIn("CullDistance", asm)

    def test_cull_distance_counts_1_to_8(self):
        for count in range(1, 9):
            with self.subTest(cull_count=count):
                asm = bary_fallback_gs.gs_spvasm([0, 2], 3, cull_count=count)
                self._assert_valid_gs_structure(asm, [0, 2], 3, cull_count=count)
                self.assertNotIn("ClipDistance", asm)

    def test_invalid_distance_counts_and_mixed_rejected(self):
        bad_values = (-10, -1, 9, 100, True, False, 1.0, "1", None)
        for bad in bad_values:
            with self.subTest(clip_count=bad):
                with self.assertRaisesRegex(ValueError, "clip distance count"):
                    bary_fallback_gs.gs_spvasm([0], 1, clip_count=bad)
            with self.subTest(cull_count=bad):
                with self.assertRaisesRegex(ValueError, "cull distance count"):
                    bary_fallback_gs.gs_spvasm([0], 1, cull_count=bad)

        for clip_c, cull_c in ((1, 1), (2, 5), (8, 8)):
            with self.subTest(clip_count=clip_c, cull_count=cull_c):
                with self.assertRaisesRegex(ValueError, "mixed ClipDistance and CullDistance"):
                    bary_fallback_gs.gs_spvasm([0], 1, clip_count=clip_c, cull_count=cull_c)


class TestWriteForIndex(unittest.TestCase):
    def _write_index(self, directory: str, data: dict | str) -> str:
        path = os.path.join(directory, "index.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f)
        return path

    def test_missing_index_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bary_fallback_gs.write_for_index(tmp), [])
            self.assertEqual(os.listdir(tmp), [])
            with self.assertRaises(FileNotFoundError):
                bary_fallback_gs.write_for_index(tmp, pair=("vs_a", "ps_a"))

    def test_filters_entries_sorts_outputs_and_writes_lf(self):
        index = {
            "_meta": "v1",
            "_null": None,
            "cs_compute": {"stage": "cs"},
            "vs_main": {"stage": "vs", "outputs": [0, 1]},
            "ps_no_fb": {"stage": "ps", "inputs": [0]},
            "ps_none_fb": {"stage": "ps", "inputs": [0], "bary_fallback_input": None},
            "ps_empty_fb": {"stage": "ps", "inputs": [0], "bary_fallback_input": {}},
            "ps_z": {"stage": "ps", "inputs": [2, 0, 2], "bary_fallback_input": {"location": 1}},
            "ps_a": {"stage": "ps", "bary_fallback_input": {"location": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            written = bary_fallback_gs.write_for_index(tmp)
            expected = [
                os.path.join(tmp, "ps_a" + bary_fallback_gs.SUFFIX + ".spvasm"),
                os.path.join(tmp, "ps_z" + bary_fallback_gs.SUFFIX + ".spvasm"),
            ]
            self.assertEqual(written, expected)

            with open(written[0], "rb") as f:
                raw_a = f.read()
            self.assertNotIn(b"\r\n", raw_a)
            self.assertEqual(raw_a.decode("utf-8"), bary_fallback_gs.gs_spvasm([], 0))

            with open(written[1], "rb") as f:
                raw_z = f.read()
            self.assertNotIn(b"\r\n", raw_z)
            self.assertEqual(raw_z.decode("utf-8"), bary_fallback_gs.gs_spvasm([0, 2], 1))

            self.assertEqual(
                sorted(os.listdir(tmp)),
                ["index.json", "ps_a.bary_gs.spvasm", "ps_z.bary_gs.spvasm"],
            )

    def test_index_without_bary_fallback_entries_returns_empty(self):
        index = {
            "vs_a": {"stage": "vs"},
            "ps_a": {"stage": "ps", "inputs": [0]},
            "cs_a": {"stage": "cs"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            self.assertEqual(bary_fallback_gs.write_for_index(tmp), [])
            self.assertEqual(os.listdir(tmp), ["index.json"])

    def test_pair_no_distances_writes_clip0_and_only_requested_ps(self):
        index = {
            "vs_plain": {"stage": "vs", "capabilities": ["Shader"], "clip_distance_count": 4},
            "ps_target": {"stage": "ps", "inputs": [1], "bary_fallback_input": {"location": 0}},
            "ps_other": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            written = bary_fallback_gs.write_for_index(tmp, pair=("vs_plain", "ps_target"))
            expected = os.path.join(tmp, "ps_target.bary_gs.clip0.spvasm")
            self.assertEqual(written, [expected])
            self.assertEqual(sorted(os.listdir(tmp)), ["index.json", "ps_target.bary_gs.clip0.spvasm"])
            with open(expected, encoding="utf-8") as f:
                self.assertEqual(f.read(), bary_fallback_gs.gs_spvasm([1], 0, 0, 0))

    def test_pair_clip_and_cull_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = {
                "vs_clip": {"stage": "vs", "capabilities": ["ClipDistance"], "clip_distance_count": 4},
                "vs_cull": {"stage": "vs", "capabilities": ["CullDistance"], "cull_distance_count": 7},
                "ps_fb": {"stage": "ps", "inputs": [0, 3], "bary_fallback_input": {"location": 1}},
            }
            self._write_index(tmp, index)
            clip_path, = bary_fallback_gs.write_for_index(tmp, pair=("vs_clip", "ps_fb"))
            self.assertEqual(os.path.basename(clip_path), "ps_fb.bary_gs.clip4.spvasm")
            with open(clip_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), bary_fallback_gs.gs_spvasm([0, 3], 1, clip_count=4))

            cull_path, = bary_fallback_gs.write_for_index(tmp, pair=("vs_cull", "ps_fb"))
            self.assertEqual(os.path.basename(cull_path), "ps_fb.bary_gs.cull7.spvasm")
            with open(cull_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), bary_fallback_gs.gs_spvasm([0, 3], 1, cull_count=7))

    def test_pair_rejects_invalid_stages_missing_entries_and_capabilities(self):
        base_index = {
            "vs_ok": {"stage": "vs"},
            "ps_ok": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}},
            "ps_no_fb": {"stage": "ps", "inputs": [0]},
            "cs_other": {"stage": "cs"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, base_index)
            bad_pairs = [
                ("missing_vs", "ps_ok"),
                ("vs_ok", "missing_ps"),
                ("cs_other", "ps_ok"),
                ("ps_ok", "ps_ok"),
                ("vs_ok", "vs_ok"),
                ("vs_ok", "ps_no_fb"),
            ]
            for pair in bad_pairs:
                with self.subTest(pair=pair):
                    with self.assertRaisesRegex(ValueError, "requires a translated VS and a barycentric-fallback PS"):
                        bary_fallback_gs.write_for_index(tmp, pair=pair)

            cap_cases = [
                (["ShaderLayer"], "ShaderLayer"),
                (["ShaderViewportIndex"], "ShaderViewportIndex"),
                (["ShaderViewportIndex", "ShaderLayer"], "ShaderLayer, ShaderViewportIndex"),
            ]
            for caps, expected_msg in cap_cases:
                with self.subTest(caps=caps):
                    idx = dict(base_index, vs_bad={"stage": "vs", "capabilities": caps})
                    self._write_index(tmp, idx)
                    with self.assertRaisesRegex(ValueError, f"paired helper cannot forward {expected_msg}"):
                        bary_fallback_gs.write_for_index(tmp, pair=("vs_bad", "ps_ok"))

    def test_pair_rejects_invalid_distance_counts_and_mixed_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cap, field in (("ClipDistance", "clip_distance_count"), ("CullDistance", "cull_distance_count")):
                for bad_count in (None, 0, -1, 9, True, False, 2.0, "2"):
                    with self.subTest(cap=cap, bad_count=bad_count):
                        vs_meta = {"stage": "vs", "capabilities": [cap]}
                        if bad_count is not None:
                            vs_meta[field] = bad_count
                        idx = {
                            "vs": vs_meta,
                            "ps": {"stage": "ps", "bary_fallback_input": {"location": 0}},
                        }
                        self._write_index(tmp, idx)
                        with self.assertRaisesRegex(ValueError, f"missing or invalid VS {field}"):
                            bary_fallback_gs.write_for_index(tmp, pair=("vs", "ps"))

            mixed_idx = {
                "vs": {
                    "stage": "vs",
                    "capabilities": ["ClipDistance", "CullDistance"],
                    "clip_distance_count": 1,
                    "cull_distance_count": 2,
                },
                "ps": {"stage": "ps", "bary_fallback_input": {"location": 0}},
            }
            self._write_index(tmp, mixed_idx)
            with self.assertRaisesRegex(ValueError, "mixed ClipDistance and CullDistance"):
                bary_fallback_gs.write_for_index(tmp, pair=("vs", "ps"))

    def test_malformed_index_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, "{not valid json")
            with self.assertRaises(json.JSONDecodeError):
                bary_fallback_gs.write_for_index(tmp)


class TestBuildForIndex(unittest.TestCase):
    def _write_index(self, directory: str, data: dict):
        with open(os.path.join(directory, "index.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_no_fallback_entries_does_not_invoke_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(bary_fallback_gs.subprocess, "run") as run_mock:
                self.assertEqual(bary_fallback_gs.build_for_index(tmp), [])
                run_mock.assert_not_called()

                self._write_index(tmp, {"ps": {"stage": "ps", "inputs": [0]}})
                self.assertEqual(bary_fallback_gs.build_for_index(tmp), [])
                run_mock.assert_not_called()

    def test_build_multiple_entries_invokes_spirv_as_and_spirv_val(self):
        index = {
            "ps_b": {"stage": "ps", "inputs": [2], "bary_fallback_input": {"location": 0}},
            "ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            seen_asm_exists = []

            def fake_run(cmd, capture_output=False, text=False):
                self.assertTrue(capture_output)
                self.assertTrue(text)
                if cmd[0] == "spirv-as":
                    seen_asm_exists.append(os.path.isfile(cmd[3]))
                return _successful_spirv_tools(cmd, capture_output=capture_output, text=text)

            with mock.patch.object(bary_fallback_gs.subprocess, "run", side_effect=fake_run) as run_mock:
                built = bary_fallback_gs.build_for_index(tmp)

            asm_a = os.path.join(tmp, "ps_a.bary_gs.spvasm")
            spv_a = os.path.join(tmp, "ps_a.bary_gs.spv")
            asm_b = os.path.join(tmp, "ps_b.bary_gs.spvasm")
            spv_b = os.path.join(tmp, "ps_b.bary_gs.spv")
            self.assertEqual(built, [spv_a, spv_b])
            self.assertEqual(seen_asm_exists, [True, True])
            commands = [call.args[0] for call in run_mock.call_args_list]
            self.assertEqual(len(commands), 4)
            for asm, spv, assemble, validate in zip((asm_a, asm_b), built, commands[::2], commands[1::2], strict=True):
                staged = Path(assemble[-1])
                staged_asm = staged.with_suffix(".spvasm")
                self.assertEqual(assemble, ["spirv-as", "--target-env", "vulkan1.2", str(staged_asm), "-o", str(staged)])
                self.assertEqual(validate, ["spirv-val", "--target-env", "vulkan1.2", str(staged)])
                self.assertTrue(Path(asm).is_file())
                self.assertEqual(staged.name, Path(spv).name)
                self.assertEqual(staged.parent.parent, Path(tmp))
                self.assertFalse(staged.parent.exists())
                self.assertEqual(Path(spv).read_bytes(), b"synthetic binary")

    def test_build_pair_forwards_suffix(self):
        index = {
            "vs_clip": {"stage": "vs", "capabilities": ["ClipDistance"], "clip_distance_count": 3},
            "ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            with mock.patch.object(
                bary_fallback_gs.subprocess,
                "run",
                side_effect=_successful_spirv_tools,
            ) as run_mock:
                built = bary_fallback_gs.build_for_index(tmp, pair=("vs_clip", "ps_a"))

            asm = os.path.join(tmp, "ps_a.bary_gs.clip3.spvasm")
            spv = os.path.join(tmp, "ps_a.bary_gs.clip3.spv")
            self.assertEqual(built, [spv])
            assemble, validate = [call.args[0] for call in run_mock.call_args_list]
            staged = Path(assemble[-1])
            staged_asm = staged.with_suffix(".spvasm")
            self.assertEqual(assemble, ["spirv-as", "--target-env", "vulkan1.2", str(staged_asm), "-o", str(staged)])
            self.assertEqual(validate, ["spirv-val", "--target-env", "vulkan1.2", str(staged)])
            self.assertTrue(Path(asm).is_file())
            self.assertEqual(staged.name, Path(spv).name)
            self.assertEqual(staged.parent.parent, Path(tmp))
            self.assertEqual(Path(spv).read_bytes(), b"synthetic binary")

    def test_spirv_as_failure_raises_runtime_error_and_stops(self):
        index = {"ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            asm = os.path.join(tmp, "ps_a.bary_gs.spvasm")
            with mock.patch.object(
                bary_fallback_gs.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1, "out msg\n", "bad op\n"),
            ) as run_mock:
                with self.assertRaises(RuntimeError) as ctx:
                    bary_fallback_gs.build_for_index(tmp)

            msg = str(ctx.exception)
            self.assertIn(f"spirv-as failed for {asm}", msg)
            self.assertIn("out msg", msg)
            self.assertIn("bad op", msg)
            self.assertEqual(run_mock.call_count, 1)

    def test_spirv_val_failure_raises_runtime_error(self):
        index = {"ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            asm = os.path.join(tmp, "ps_a.bary_gs.spvasm")
            results = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 2, "val out\n", "invalid spirv\n"),
            ]
            with mock.patch.object(bary_fallback_gs.subprocess, "run", side_effect=results) as run_mock:
                with self.assertRaises(RuntimeError) as ctx:
                    bary_fallback_gs.build_for_index(tmp)

            msg = str(ctx.exception)
            self.assertIn(f"spirv-val failed for {asm}", msg)
            self.assertIn("val out", msg)
            self.assertIn("invalid spirv", msg)
            self.assertEqual(run_mock.call_count, 2)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs SPIRV-Tools")
    def test_invalid_rebuild_preserves_the_validated_pair(self) -> None:
        index = {"ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            built, = bary_fallback_gs.build_for_index(tmp)
            published = Path(built)
            published_asm = published.with_suffix(".spvasm")
            original = published.read_bytes()
            original_asm = published_asm.read_bytes()
            valid = bary_fallback_gs.gs_spvasm([0], 1)
            invalid = valid.replace("OpStore %out_bary %bary0", "OpStore %out_bary %f1")
            self.assertNotEqual(invalid, valid)
            with mock.patch.object(bary_fallback_gs, "gs_spvasm", return_value=invalid):
                with self.assertRaisesRegex(RuntimeError, "spirv-val failed"):
                    bary_fallback_gs.build_for_index(tmp)
            self.assertEqual(published.read_bytes(), original)
            self.assertEqual(published_asm.read_bytes(), original_asm)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", published], check=True)
            with tempfile.TemporaryDirectory() as verification:
                reassembled = Path(verification) / published.name
                subprocess.run(["spirv-as", "--target-env", "vulkan1.2", published_asm, "-o", reassembled], check=True)
                self.assertEqual(reassembled.read_bytes(), original)
            self.assertEqual(sorted(os.listdir(tmp)), ["index.json", "ps_a.bary_gs.spv", "ps_a.bary_gs.spvasm"])

    def test_missing_spirv_binaries_raise_file_not_found(self):
        index = {"ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, index)
            with mock.patch.object(
                bary_fallback_gs.subprocess,
                "run",
                side_effect=FileNotFoundError(2, "No such file", "spirv-as"),
            ) as run_mock:
                with self.assertRaises(FileNotFoundError):
                    bary_fallback_gs.build_for_index(tmp)
            self.assertEqual(run_mock.call_count, 1)

            side_effects = [
                subprocess.CompletedProcess([], 0, "", ""),
                FileNotFoundError(2, "No such file", "spirv-val"),
            ]
            with mock.patch.object(bary_fallback_gs.subprocess, "run", side_effect=side_effects) as run_mock:
                with self.assertRaises(FileNotFoundError):
                    bary_fallback_gs.build_for_index(tmp)
            self.assertEqual(run_mock.call_count, 2)


class TestMainAndCLI(unittest.TestCase):
    def _call_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                bary_fallback_gs.main(argv)
            except SystemExit as exc:
                code = exc.code if exc.code is not None else 0
        return code, stdout.getvalue(), stderr.getvalue()

    def test_main_rejects_missing_dir_and_regular_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope")
            code, out, err = self._call_main(["bary_fallback_gs.py", missing])
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("does not exist", err)

            file_path = os.path.join(tmp, "not_a_dir.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("x")
            code, out, err = self._call_main(["bary_fallback_gs.py", file_path])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("not a directory", err)

    def test_main_prints_spv_paths_on_success(self):
        index = {
            "vs_cull": {"stage": "vs", "capabilities": ["CullDistance"], "cull_distance_count": 2},
            "ps_a": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}},
            "ps_b": {"stage": "ps", "inputs": [1], "bary_fallback_input": {"location": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)

            with mock.patch.object(
                bary_fallback_gs.subprocess,
                "run",
                side_effect=_successful_spirv_tools,
            ):
                code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
                self.assertEqual(code, 0)
                self.assertEqual(err, "")
                self.assertEqual(
                    out.splitlines(),
                    [
                        os.path.join(tmp, "ps_a.bary_gs.spv"),
                        os.path.join(tmp, "ps_b.bary_gs.spv"),
                    ],
                )

                code, out, err = self._call_main(["bary_fallback_gs.py", tmp, "--pair", "vs_cull", "ps_a"])
                self.assertEqual(code, 0)
                self.assertEqual(err, "")
                self.assertEqual(out.splitlines(), [os.path.join(tmp, "ps_a.bary_gs.cull2.spv")])

    def test_main_turns_oserror_valueerror_runtimeerror_into_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            # OSError: --pair with missing index.json
            code, out, err = self._call_main(["bary_fallback_gs.py", tmp, "--pair", "vs", "ps"])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("index.json", err)

            # ValueError: malformed JSON
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                f.write("{bad")
            code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotEqual(err.strip(), "")

            # ValueError: colliding location in index
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"ps": {"stage": "ps", "inputs": [1], "bary_fallback_input": {"location": 1}}}, f)
            code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("collides", err)

            # OSError: missing spirv-as
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"ps": {"stage": "ps", "inputs": [0], "bary_fallback_input": {"location": 1}}}, f)
            with mock.patch.object(
                bary_fallback_gs.subprocess,
                "run",
                side_effect=FileNotFoundError("spirv-as not found"),
            ):
                code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("spirv-as not found", err)

            # RuntimeError: spirv-val failed
            results = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 1, "", "validation error"),
            ]
            with mock.patch.object(bary_fallback_gs.subprocess, "run", side_effect=results):
                code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("spirv-val failed", err)
            self.assertIn("validation error", err)

    def test_main_rejects_non_object_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index in ([], ["ps"], None):
                with self.subTest(index=index):
                    with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                        json.dump(index, f)

                    code, out, err = self._call_main(["bary_fallback_gs.py", tmp])
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertIn("translation index must be a JSON object", err)

    def test_main_pair_rejects_non_object_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.json")
            good = {"vs": {"stage": "vs"}, "ps": {"stage": "ps", "bary_fallback_input": {"location": 1}}}
            for bad in ([], 1, None, "vs"):
                for name in ("vs", "ps"):
                    with self.subTest(entry=name, value=bad):
                        with open(index_path, "w", encoding="utf-8", newline="\n") as f:
                            json.dump({**good, name: bad}, f)
                        code, out, err = self._call_main(["bary_fallback_gs.py", tmp, "--pair", "vs", "ps"])
                        self.assertEqual(code, 1)
                        self.assertEqual(out, "")
                        self.assertEqual(err, f"{index_path}: index entry {name!r} must be a JSON object\n")

    def test_cli_subprocess_isolated_mode(self):
        def run_cli(*args):
            return subprocess.run(
                [sys.executable, "-I", TOOL_PATH, *args],
                capture_output=True,
                text=True,
            )

        p = run_cli("--help")
        self.assertEqual(p.returncode, 0)
        self.assertIn("--pair VS PS", p.stdout)
        self.assertEqual(p.stderr, "")

        p = run_cli()
        self.assertEqual(p.returncode, 2)
        self.assertNotIn("Traceback", p.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            p = run_cli(tmp, "--pair", "only_vs")
            self.assertEqual(p.returncode, 2)
            self.assertNotIn("Traceback", p.stderr)

            missing = os.path.join(tmp, "missing")
            p = run_cli(missing)
            self.assertEqual(p.returncode, 2)
            self.assertEqual(p.stdout, "")
            self.assertIn("does not exist", p.stderr)
            self.assertNotIn("Traceback", p.stderr)

            # Empty directory (no index.json) succeeds silently without spirv-as/spirv-val.
            p = run_cli(tmp)
            self.assertEqual(p.returncode, 0)
            self.assertEqual(p.stdout, "")
            self.assertEqual(p.stderr, "")

            # --pair without index.json exits 1 cleanly.
            p = run_cli(tmp, "--pair", "vs", "ps")
            self.assertEqual(p.returncode, 1)
            self.assertEqual(p.stdout, "")
            self.assertIn("index.json", p.stderr)
            self.assertNotIn("Traceback", p.stderr)

            # Unicode directory path under python -I with non-fallback index succeeds cleanly.
            uni_dir = os.path.join(tmp, "spirv-\u00e9-\u2603")
            os.makedirs(uni_dir)
            with open(os.path.join(uni_dir, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"ps": {"stage": "ps", "inputs": [0]}}, f)
            p = run_cli(uni_dir)
            self.assertEqual(p.returncode, 0)
            self.assertEqual(p.stdout, "")
            self.assertEqual(p.stderr, "")

            # Malformed index.json exits 1 without a traceback.
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                f.write("{invalid")
            p = run_cli(tmp)
            self.assertEqual(p.returncode, 1)
            self.assertEqual(p.stdout, "")
            self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
