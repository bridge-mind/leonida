"""rdna2spirv.py --bary-fallback: raw barycentric reads without VK_KHR_fragment_shader_barycentric.

The input is the GPU test runtime/test/shaders/ps_bary (hand-written, no game data; fixture from regen_fixtures.py).
It reads the raw barycentric pair (v0 = i, v1 = j) next to v_interp. By default the translator gives the pair as
BaryCoordKHR.y / .z. With --bary-fallback the same values come from an ordinary vec3 input at the lowest location no
attribute uses; tools/bary_fallback_gs.py writes the pass-through geometry shader that fills it. On the GPU,
run_gpu_tests.py / check.py with LEONIDA_BARY_FALLBACK=1 check ps_bary through that GS. Here:
* the module declares neither the capability nor the extension nor the builtin, and the metadata names the input
  (the lowest free location, also for sparse attributes such as attr30 alone);
* the shader reads i and j from the new input exactly as it reads them from BaryCoordKHR (one-lane interpreter);
* a shader that doesn't read the pair translates to the same module with and without the flag;
* the GS text has the vertex count, the one-hot weights and the locations the PS expects, and refuses a collision.
"""
import gc
import json
from pathlib import Path
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

import support
import rdna2spirv
import bary_fallback_gs

I, J = 0.3125, 0.4375          # exact in binary32, i + j < 1
ATTR0 = (0.75, 0.5, 0.0, 1.0)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def translate(name, fallback):
    old = rdna2spirv.BARY_FALLBACK
    rdna2spirv.BARY_FALLBACK = fallback
    try:
        spv = support.translate_fixture(name)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.BARY_FALLBACK = old
    return spv, meta


class BaryFallbackModule(unittest.TestCase):
    def test_default_uses_the_builtin(self):
        spv, meta = translate("ps_bary", False)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        self.assertIn("OpCapability FragmentBarycentricKHR", spv)
        self.assertIn('OpExtension "SPV_KHR_fragment_shader_barycentric"', spv)
        self.assertNotIn("bary_fallback_input", meta)
        self.assertIn("fragmentShaderBarycentric", meta["features"])

    def test_fallback_is_a_plain_input_past_the_attributes(self):
        spv, meta = translate("ps_bary", True)
        self.assertNotIn("BaryCoordKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("SPV_KHR_fragment_shader_barycentric", spv)
        self.assertNotIn("fragmentShaderBarycentric", meta["features"])
        loc = meta["bary_fallback_input"]["location"]
        self.assertEqual(loc, min(set(range(len(meta["inputs"]) + 1)) - set(meta["inputs"])))   # lowest free
        self.assertIn(f"OpDecorate %bary Location {loc}", spv)
        # Perspective-correct like BaryCoordKHR: no Flat / NoPerspective on it.
        self.assertIsNone(re.search(r"OpDecorate %bary (Flat|NoPerspective)", spv))
        self.assertRegex(spv, r"%bary = OpVariable %_ptr_Input_v3f Input")

    def test_sparse_attributes_take_the_lowest_free_location(self):
        # A PS whose only attribute is attr30 (#804 review): one past the highest would be 31, beyond what drivers
        # with 128 fragment input components accept; the lowest free location is 0.
        with open(os.path.join(support.FIXTURE_DIR, "ps_bary.s"), encoding="utf-8") as f:
            text = f.read().replace("attr0.", "attr30.")
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            src = os.path.join(tmp, "disasm", "ps_bary_sparse.s")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            with open(os.path.join(tmp, "ps_bary_sparse.dxbc"), "wb") as f:
                f.write(support.STUB_DXBC)
            old = rdna2spirv.BARY_FALLBACK
            rdna2spirv.BARY_FALLBACK = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    spv = rdna2spirv.translate(src)
                    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
                    gc.collect()
            finally:
                rdna2spirv.BARY_FALLBACK = old
        self.assertEqual(meta["inputs"], [30])
        self.assertEqual(meta["bary_fallback_input"], {"location": 0})
        self.assertIn("OpDecorate %bary Location 0", spv)
        self.assertIn("OpDecorate %in_attr30 Location 30", spv)
        gs = bary_fallback_gs.gs_spvasm(meta["inputs"], 0)
        self.assertIn("OpDecorate %out_bary Location 0", gs)
        self.assertIn("OpDecorate %out30 Location 30", gs)

    def test_same_values_from_either_source(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                spv, _ = translate("ps_bary", fallback)
                words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0, "%bary": (1.0 - I - J, I, J)})
                # mrt0 = (attr0.x, attr0.y, raw i, raw j): the single-lane interpreter models v_interp as the
                # attribute value itself, and the raw pair is BaryCoord .y / .z.
                self.assertEqual(words, [f32_bits(ATTR0[0]), f32_bits(ATTR0[1]), f32_bits(I), f32_bits(J)])

    def test_shaders_without_the_pair_are_unchanged(self):
        for name in ("ps_ops", "ps_flat", "cs_bounds"):
            with self.subTest(name=name):
                fssm = None
                if name == "cs_bounds":
                    fssm = bytearray(0x148)
                    struct.pack_into("<4I", fssm, 0x138, 0x8088, 64, 1, 1)
                    fssm = bytes(fssm)
                old = rdna2spirv.BARY_FALLBACK
                try:
                    rdna2spirv.BARY_FALLBACK = False
                    off = support.translate_fixture(name, fssm=fssm)
                    rdna2spirv.BARY_FALLBACK = True
                    on = support.translate_fixture(name, fssm=fssm)
                finally:
                    rdna2spirv.BARY_FALLBACK = old
                self.assertEqual(off, on)


class BaryFallbackGeometryShader(unittest.TestCase):
    def test_text(self):
        text = bary_fallback_gs.gs_spvasm([2, 0], 3)
        self.assertIn("OpExecutionMode %main Triangles", text)
        self.assertIn("OpExecutionMode %main OutputVertices 3", text)
        self.assertEqual(text.count("OpEmitVertex"), 3)
        self.assertEqual(text.count("OpEndPrimitive"), 1)
        for n in (0, 2):
            self.assertIn(f"OpDecorate %in{n} Location {n}", text)
            self.assertIn(f"OpDecorate %out{n} Location {n}", text)
        self.assertIn("OpDecorate %out_bary Location 3", text)
        # Vertex k gets weight 1 in component k, in emission order.
        stores = re.findall(r"OpStore %out_bary %bary(\d)", text)
        self.assertEqual(stores, ["0", "1", "2"])
        self.assertIn("%bary0 = OpConstantComposite %v3float %f1 %f0 %f0", text)
        self.assertIn("%bary1 = OpConstantComposite %v3float %f0 %f1 %f0", text)
        self.assertIn("%bary2 = OpConstantComposite %v3float %f0 %f0 %f1", text)

    def test_pair_forwards_the_producers_clip_array_for_each_vertex(self):
        index = {"vs_clip": {"stage": "vs", "capabilities": ["ClipDistance"], "clip_distance_count": 2},
                 "ps_bary": {"stage": "ps", "inputs": [1], "bary_fallback_input": {"location": 0}}}
        with tempfile.TemporaryDirectory() as tmp:
            index_file = Path(tmp, "index.json")
            index_file.write_text(json.dumps(index), encoding="utf-8")
            ordinary, = bary_fallback_gs.write_for_index(tmp)
            paired, = bary_fallback_gs.write_for_index(tmp, pair=("vs_clip", "ps_bary"))
            self.assertNotEqual(ordinary, paired)
            self.assertNotIn("BuiltIn ClipDistance", Path(ordinary).read_text())
            asm = Path(paired).read_text()
            self.assertIn("%clip_count = OpConstant %uint 2", asm)
            self.assertEqual(asm.count("BuiltIn ClipDistance"), 2)
            for vertex, emission in enumerate(asm.split("OpEmitVertex")[:-1]):
                self.assertIn(f"OpAccessChain %ptr_in_clip %in_clip %int_{vertex}", emission)
                self.assertIn(f"OpStore %out_clip %distances{vertex}", emission)
            del index["vs_clip"]["clip_distance_count"]
            index_file.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "regenerate"):
                bary_fallback_gs.write_for_index(tmp, pair=("vs_clip", "ps_bary"))

    def test_pair_forwards_cull_distances_and_checks_the_layout(self):
        vs = {"stage": "vs", "capabilities": ["CullDistance"], "cull_distance_count": 2}
        index = {"vs": vs, "ps": {"stage": "ps", "bary_fallback_input": {"location": 0}}}
        with tempfile.TemporaryDirectory() as tmp:
            index_file = Path(tmp, "index.json")
            index_file.write_text(json.dumps(index), encoding="utf-8")
            path, = bary_fallback_gs.write_for_index(tmp, pair=("vs", "ps"))
            self.assertEqual(Path(path).name, "ps.bary_gs.cull2.spvasm")
            asm = Path(path).read_text()
            self.assertEqual(asm.count("BuiltIn CullDistance"), 2)
            self.assertNotIn("ClipDistance", asm)
            self.assertIn("%cull_count = OpConstant %uint 2", asm)
            for vertex, emission in enumerate(asm.split("OpEmitVertex")[:-1]):
                self.assertIn(f"OpAccessChain %ptr_in_cull %in_cull %int_{vertex}", emission)
                self.assertIn(f"OpLoad %cull %cull_ptr{vertex}", emission)
                self.assertIn(f"OpStore %out_cull %distances{vertex}", emission)
            del vs["cull_distance_count"]
            index_file.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cull_distance_count"):
                bary_fallback_gs.write_for_index(tmp, pair=("vs", "ps"))
            vs.update(capabilities=["ClipDistance", "CullDistance"], clip_distance_count=1, cull_distance_count=1)
            index_file.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed"):
                bary_fallback_gs.write_for_index(tmp, pair=("vs", "ps"))
            self.assertEqual(list(Path(tmp).glob("*.spvasm")), [Path(path)])

    def test_a_missing_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing")
            p = subprocess.run([sys.executable, "-I", bary_fallback_gs.__file__, missing],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)                              # a missing input exits 2 (BUILD-1046)
            self.assertEqual(p.stdout, "")
            self.assertIn("does not exist", p.stderr)
            self.assertEqual(os.listdir(tmp), [])                          # nothing written

    def test_collision_is_refused(self):
        with self.assertRaises(ValueError):
            bary_fallback_gs.gs_spvasm([0, 1], 1)

    def test_index_drives_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = {"ps_a": {"stage": "ps", "inputs": [0, 1], "bary_fallback_input": {"location": 2}},
                     "ps_b": {"stage": "ps", "inputs": [0]},
                     "cs_c": {"stage": "cs"}}
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            written = bary_fallback_gs.write_for_index(tmp)
            self.assertEqual([os.path.basename(p) for p in written], ["ps_a.bary_gs.spvasm"])
            with open(written[0], encoding="utf-8") as f:
                self.assertEqual(f.read(), bary_fallback_gs.gs_spvasm([0, 1], 2))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bary_fallback_gs.write_for_index(tmp), [])     # no index: nothing to do

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_assembles_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            asm = os.path.join(tmp, "gs.spvasm")
            for clip, cull in ((0, 0), (1, 0), (2, 0), (3, 0), (8, 0), (0, 1), (0, 2), (0, 8)):
                with open(asm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(bary_fallback_gs.gs_spvasm([0, 1, 5], 6, clip, cull))
                spv = os.path.join(tmp, "gs.spv")
                for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", spv],
                            ["spirv-val", "--target-env", "vulkan1.2", spv]):
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
