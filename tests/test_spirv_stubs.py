"""Tests for tools/spirv_stubs.py and graphics pipeline metadata in tools/rdna2spirv.py.

Runs with the standard library only (no GPU required):
    python -I tests/test_spirv_stubs.py
When spirv-val is on PATH, every generated stub is also validated with spirv-val --target-env vulkan1.2.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import spirv_stubs  # noqa: E402


def decode_spirv_words(data):
    assert len(data) >= 20 and len(data) % 4 == 0
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    magic, version, _gen, bound, schema = words[:5]
    assert magic == spirv_stubs.SPIRV_MAGIC
    assert version == spirv_stubs.SPIRV_VERSION_1_0
    assert bound > 1
    assert schema == 0
    insts = []
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        assert wc >= 1 and i + wc <= len(words)
        insts.append((op, words[i + 1 : i + wc]))
        i += wc
    return bound, insts


class SpirvStubsTests(unittest.TestCase):
    def _maybe_spirv_val(self, data):
        val = shutil.which("spirv-val")
        if not val:
            return
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "stub.spv")
            with open(p, "wb") as f:
                f.write(data)
            r = subprocess.run([val, "--target-env", "vulkan1.2", p], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"spirv-val failed:\n{r.stderr}")

    def test_stub_name_canonical(self):
        self.assertEqual(spirv_stubs.stub_name("vs", []), "vs_empty.spv")
        self.assertEqual(spirv_stubs.stub_name("vs", [3, 0, 1, 0]), "vs_0_1_3.spv")
        self.assertEqual(spirv_stubs.stub_name("ps", []), "ps_empty.spv")
        self.assertEqual(spirv_stubs.stub_name("ps", [2, 1]), "ps_1_2.spv")

    def test_vs_stubs_decode_and_validate(self):
        for locs in ([], [0], [0, 2, 5]):
            data = spirv_stubs.build_vs_stub(locs)
            _, insts = decode_spirv_words(data)
            decs = [args for op, args in insts if op == spirv_stubs.OP_DECORATE]
            # BuiltIn Position + one Location decoration per unique output location.
            self.assertTrue(any(d[1:] == [spirv_stubs.DEC_BUILTIN, spirv_stubs.BUILTIN_POSITION] for d in decs))
            got_locs = sorted(d[2] for d in decs if len(d) == 3 and d[1] == spirv_stubs.DEC_LOCATION)
            self.assertEqual(got_locs, sorted(locs))
            self._maybe_spirv_val(data)

    def test_ps_stubs_decode_and_validate(self):
        for locs, flat in (([], []), ([0], []), ([0, 1, 4], [1])):
            data = spirv_stubs.build_ps_stub(locs, flat_locations=flat)
            _, insts = decode_spirv_words(data)
            # Must declare OriginUpperLeft.
            modes = [args for op, args in insts if op == spirv_stubs.OP_EXECUTION_MODE]
            self.assertTrue(any(m[1] == spirv_stubs.MODE_ORIGIN_UPPER_LEFT for m in modes))
            decs = [args for op, args in insts if op == spirv_stubs.OP_DECORATE]
            # Location 0 output + each input location.
            got_locs = sorted(d[2] for d in decs if len(d) == 3 and d[1] == spirv_stubs.DEC_LOCATION)
            self.assertEqual(got_locs, sorted([0] + list(locs)))
            flat_decs = [d for d in decs if len(d) == 2 and d[1] == spirv_stubs.DEC_FLAT]
            self.assertEqual(len(flat_decs), len(flat))
            self._maybe_spirv_val(data)

    def test_failed_rebuild_keeps_existing_stub(self):
        index = {"ps": {"stage": "ps", "inputs": [0, 2]}}

        @contextmanager
        def interrupted_open(path, mode):
            self.assertEqual(Path(path).parent, target.parent)
            with open(path, mode) as f:
                write = f.write

                def interrupted_write(data):
                    write(data[:7])
                    raise OSError("injected write failure")

                with patch.object(f, "write", interrupted_write):
                    yield f

        failures = (
            ("write", patch.object(spirv_stubs, "open", interrupted_open, create=True)),
            ("replace", patch.object(spirv_stubs.os, "replace", side_effect=PermissionError("injected replace failure"))),
        )
        for step, failure in failures:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                written = spirv_stubs.generate_for_index(tmp, index)
                target = Path(tmp) / "stubs" / written[0]
                original = target.read_bytes()
                self._maybe_spirv_val(original)
                with failure, self.assertRaisesRegex(OSError, f"injected {step} failure"):
                    spirv_stubs.generate_for_index(tmp, index)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(target.parent.iterdir()), [target])

    @unittest.skipUnless(os.name == "nt", "Windows replacement sharing")
    def test_rebuild_retries_transient_windows_replace_denial(self):
        index = {"ps": {"stage": "ps", "inputs": [0, 2]}}
        with tempfile.TemporaryDirectory() as tmp:
            written = spirv_stubs.generate_for_index(tmp, index)
            replace = os.replace
            error = OSError(0, "injected sharing denial", None, 5)

            def retry_replace(src, dst):
                if attempt.call_count == 1:
                    raise error
                replace(src, dst)

            with patch.object(spirv_stubs.os, "replace", side_effect=retry_replace) as attempt:
                self.assertEqual(spirv_stubs.generate_for_index(tmp, index), written)
            self.assertEqual(attempt.call_count, 2)
            self.assertEqual(os.listdir(os.path.join(tmp, "stubs")), written)

    @unittest.skipUnless(os.name == "posix", "POSIX file mode")
    def test_rebuild_preserves_existing_mode(self):
        index = {"ps": {"stage": "ps", "inputs": [0]}}
        with tempfile.TemporaryDirectory() as tmp:
            written = spirv_stubs.generate_for_index(tmp, index)
            target = Path(tmp) / "stubs" / written[0]
            target.chmod(0o640)
            self.assertEqual(spirv_stubs.generate_for_index(tmp, index), written)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_required_stubs_and_deduplication(self):
        index = {
            "cs_a": {"stage": "cs"},
            "ps_a": {"stage": "ps", "inputs": [0, 1], "outputs": [0]},
            "ps_b": {"stage": "ps", "inputs": [1, 0], "outputs": [0, 1]},  # same VS stub as ps_a
            "ps_c": {"stage": "ps", "outputs": [0]},
            "vs_a": {"stage": "vs", "outputs": [0, 1]},
            "ms_a": {"stage": "ms", "params": [0, 1]},  # same PS stub as vs_a
            "gs_a": {"stage": "gs", "gs_inputs": [0, 1], "gs_outputs": [0, 1],
                     "gs_input_locations": [1], "gs_output_locations": [1]},
            "gs_legacy": {"stage": "gs", "gs_inputs": [1, 0, 1], "gs_outputs": [1, 0],
                          "gs_output_locations": None},
        }
        with tempfile.TemporaryDirectory() as tmp:
            written = spirv_stubs.generate_for_index(tmp, index)
            self.assertEqual(written, ["ps_0_1.spv", "ps_1.spv", "vs_empty.spv", "vs_0_1.spv", "vs_1.spv"])
            for fname in written:
                path = os.path.join(tmp, "stubs", fname)
                self.assertTrue(os.path.isfile(path))
                with open(path, "rb") as f:
                    self._maybe_spirv_val(f.read())

            # CLI smoke test on index.json.
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            self.assertEqual(spirv_stubs.main([tmp]), 0)


class ShaderMetaTests(unittest.TestCase):
    def test_shader_meta_graphics_fields(self):
        # PS with flat + interpolated inputs and depth write.
        g = rdna2spirv.Gen("ps")
        g.inputs = {"%in_0": 0, "%in_1": 1}
        g.interpolated = {"%in_1"}
        g.outputs = {"%out_0": "mrt0", "%out_2": "mrt2"}
        g.writes_depth = True
        m = rdna2spirv.shader_meta(g, "OpCapability Shader\n")
        self.assertEqual(m["stage"], "ps")
        self.assertEqual(m["inputs"], [0, 1])
        self.assertEqual(m["outputs"], [0, 2])
        self.assertTrue(m["writes_depth"])

        # GS separating BuiltIn Position (sv == 1) from Location registers.
        gg = rdna2spirv.Gen("gs")
        gg.gs = {"in": [(0, 1), (1, 0), (3, 0)], "out": [(0, 1), (2, 0)], "nreg": 2, "maxv": 4}
        mg = rdna2spirv.shader_meta(gg, "OpCapability Geometry\n")
        self.assertEqual(mg["gs_inputs"], [0, 1, 3])
        self.assertEqual(mg["gs_outputs"], [0, 2])
        self.assertEqual(mg["gs_input_locations"], [1, 3])
        self.assertEqual(mg["gs_output_locations"], [2])
        self.assertIn("geometryShader", mg["features"])


class MalformedIndexTest(unittest.TestCase):
    def run_tool(self, contents):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                f.write(contents)
            tool = os.path.join(ROOT, "tools", "spirv_stubs.py")
            return subprocess.run([sys.executable, "-I", tool, tmp], capture_output=True, text=True)

    def test_bad_index_json_is_one_error_not_a_traceback(self):
        p = self.run_tool("not json")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("cannot read", p.stderr)
        self.assertIn("index.json", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_help_prints_usage_and_exits_zero(self):
        tool = os.path.join(ROOT, "tools", "spirv_stubs.py")
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                p = subprocess.run([sys.executable, "-I", tool, flag], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0)
                self.assertIn("spirv_stubs.py <spirv dir>", p.stdout)
                self.assertEqual(p.stderr, "")
                self.assertEqual(spirv_stubs.main([flag]), 0)

    def test_valid_json_of_the_wrong_shape_is_one_error(self):
        for contents in ("5", "[]"):
            with self.subTest(contents=contents):
                p = self.run_tool(contents)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("cannot read", p.stderr)
                self.assertIn("expected a JSON object", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_invalid_graphics_locations_are_one_error_not_a_traceback(self):
        cases = (
            ("ps_bad", {"stage": "ps", "inputs": ["not-a-location"]}),
            ("vs_bad", {"stage": "vs", "outputs": [None]}),
            ("ms_bad", {"stage": "ms", "params": [-1]}),
            ("gs_bad", {"stage": "gs", "gs_input_locations": [0x100000000]}),
        )
        for name, metadata in cases:
            with self.subTest(shader=name):
                p = self.run_tool(json.dumps({name: metadata}))
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("cannot generate stubs", p.stderr)
                self.assertIn("index.json", p.stderr)
                if name in ("ms_bad", "gs_bad"):
                    self.assertIn("invalid Location", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_malformed_location_types_stop_the_entire_preflight(self):
        cases = (
            {"stage": "ps", "inputs": [1.9]},
            {"stage": "vs", "outputs": [True]},
            {"stage": "ms", "params": ["3"]},
            {"stage": "gs", "gs_input_locations": "23"},
            {"stage": "gs", "gs_output_locations": {"2": 0}},
            {"stage": "gs", "gs_input_locations": None, "gs_inputs": [0.5]},
        )
        for meta in cases:
            with self.subTest(meta=meta), tempfile.TemporaryDirectory() as tmp:
                index = {"good": {"stage": "ps", "inputs": [0]}, "bad": meta}
                with self.assertRaisesRegex(ValueError, "bad: invalid graphics interface locations"):
                    spirv_stubs.generate_for_index(tmp, index)
                self.assertFalse(os.path.exists(os.path.join(tmp, "stubs")))

    def test_non_finite_location_is_one_error_not_a_traceback(self):
        p = self.run_tool('{"ps_bad": {"stage": "ps", "inputs": [1e999]}}')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ps_bad: invalid graphics interface locations", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


class IndexEncodingTest(unittest.TestCase):
    def test_index_saved_with_a_utf8_bom_is_read(self):
        # A Windows editor or PowerShell `>` writes the BOM; the JSON behind it is unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8-sig") as f:
                json.dump({"ps_a": {"stage": "ps", "inputs": [0]}}, f)
            tool = os.path.join(ROOT, "tools", "spirv_stubs.py")
            p = subprocess.run([sys.executable, "-I", tool, tmp], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("wrote 1 stubs", p.stdout)
            self.assertEqual(os.listdir(os.path.join(tmp, "stubs")), ["vs_0.spv"])


if __name__ == "__main__":
    unittest.main()
