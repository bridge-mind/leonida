"""Each declaration and Vulkan device feature follows its own dynamic descriptor accesses."""
from pathlib import Path
import re
import unittest

import support
import rdna2spirv


CAPABILITY = "SampledImageArrayDynamicIndexing"
FEATURE = "shaderSampledImageArrayDynamicIndexing"
STORAGE_CAPABILITY = "StorageImageArrayDynamicIndexing"
STORAGE_FEATURE = "shaderStorageImageArrayDynamicIndexing"
EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]


def translate(instruction):
    return support.translate_lines("ps_dynamic_index", [instruction, *EPILOGUE])


class SampledDynamicIndex(unittest.TestCase):
    def check_accesses(self, asm, sampled_bindings, storage_bindings=()):
        bindings = {var: int(binding) for var, binding in re.findall(r"OpDecorate (%\w+) Binding (\d+)", asm)}
        lookups = set(re.findall(r"(%\w+) = OpFunctionCall %uint %desc_lookup_fn ", asm))
        broadcasts = re.findall(r"(%\w+) = OpGroupNonUniformBroadcastFirst %uint %\w+ (%\w+)", asm)
        # a wave32 pixel shader in a 64-lane subgroup takes its own half's first lane with a shuffle (design.md,
        # "Execution model": shuffle sources keep the current half)
        broadcasts += re.findall(r"(%\w+) = OpGroupNonUniformShuffle %uint %\w+ (%\w+) %\w+", asm)
        lookup_indices = lookups | {result for result, source in broadcasts if source in lookups}
        sampled, storage = set(), set()
        for var, index in re.findall(r"OpAccessChain %_ptr_UC_\w+ (%\w+) (%\w+)$", asm, re.M):
            self.assertIn(index, lookup_indices)
            (storage if var.startswith(("%simages", "%stv")) else sampled).add(bindings[var])
        self.assertEqual(sampled, set(sampled_bindings))
        self.assertEqual(storage, set(storage_bindings))
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
        self.assertEqual(sum(line.strip() == f"OpCapability {CAPABILITY}" for line in asm.splitlines()), bool(sampled))
        self.assertEqual(FEATURE in meta["features"], bool(sampled))
        self.assertEqual(sum(line.strip() == f"OpCapability {STORAGE_CAPABILITY}" for line in asm.splitlines()),
                         bool(storage))
        self.assertEqual(STORAGE_FEATURE in meta["features"], bool(storage))
        self.assertNotIn("SampledImageArrayNonUniformIndexing", meta["capabilities"])
        self.assertNotIn("StorageImageArrayNonUniformIndexing", meta["capabilities"])

    def test_sample_with_literal_sampler_still_indexes_both_arrays_dynamically(self):
        fixture = Path(__file__).with_name("fixtures") / "sampled_dynamic_index.asm"
        asm = support.translate_lines("ps_dynamic_index", fixture.read_text(encoding="utf-8").splitlines())
        self.check_accesses(asm, (0, 1))
        self.assertIn("OpSampledImage", asm)
        self.assertIn("OpImageSampleExplicitLod", asm)
        for signedness in (1, 2):  # UINT/SINT alias the same sampled binding as the float view.
            self.assertRegex(asm, rf"OpAccessChain %_ptr_UC_itx{signedness}_0 %itx{signedness}_0 ")

    def test_fetch_and_query_paths(self):
        for op, dim, coords, sampler, expected_op, bindings in (
            ("image_load", "1D", "v4", "", "OpImageFetch", (5,)),
            ("image_load", "2D_ARRAY", "v[4:6]", "", "OpImageFetch", (3,)),
            ("image_load_mip", "2D", "v[4:6]", "", "OpImageFetch", (0,)),
            ("image_load_mip", "3D", "v[4:7]", "", "OpImageFetch", (6,)),
            ("image_get_resinfo", "2D", "v4", "", "OpImageQuerySizeLod", (0,)),
            ("image_get_resinfo", "2D_ARRAY", "v4", "", "OpImageQuerySizeLod", (3,)),
            ("image_get_lod", "2D_ARRAY", "v[4:6]", ", s[8:11]", "OpImageQueryLod", (1, 3)),
            ("image_gather4_lz", "2D", "v[4:5]", ", s[8:11]", "OpImageGather", (0, 1)),
            ("image_gather4_l", "2D", "v[4:6]", ", s[8:11]", "OpImageFetch", (0, 1)),
        ):
            with self.subTest(op=op, dim=dim):
                asm = translate(f"{op} v[0:3], {coords}, s[0:7]{sampler} dmask:0x1 dim:SQ_RSRC_IMG_{dim}")
                self.check_accesses(asm, bindings)
                self.assertIn(expected_op, asm)
                if op == "image_gather4_l":
                    self.assertNotIn("OpSampledImage", asm)  # manual taps use the S# words directly

    def test_storage_only_and_constant_register_indices(self):
        for instruction, bindings in (
            ("image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D glc", (8, 9, 10)),
            ("image_store v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D", (8, 9, 10)),
            ("image_atomic_add v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D", (4,)),
            ("image_atomic_add v0, v[4:6], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D", (7,)),
            ("v_mov_b32_e32 v0, 1", ()),
        ):
            with self.subTest(instruction=instruction):
                asm = translate(instruction)
                self.check_accesses(asm, (), bindings)
                if bindings:
                    self.assertIn("OpCapability RuntimeDescriptorArray", asm)

    def test_literal_storage_descriptor_and_mixed_sampled_load(self):
        fixture = Path(__file__).with_name("fixtures") / "storage_dynamic_index.asm"
        lines = fixture.read_text(encoding="utf-8").splitlines()
        self.check_accesses(support.translate_lines("ps_storage_index", lines), (), (8, 9, 10))
        load = "image_load v0, v[4:5], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"
        asm = support.translate_lines("ps_mixed_index", lines[:-2] + [load, "s_waitcnt vmcnt(0)"] + lines[-2:])
        self.check_accesses(asm, (0,), (8, 9, 10))
        self.assertIn("OpImageFetch", asm)
        self.assertIn("OpImageWrite", asm)

    def test_storage_aliases_in_every_dimension(self):
        for dim, coords, first_binding in (("2D", "v[4:5]", 8), ("2D_ARRAY", "v[4:6]", 11),
                                           ("3D", "v[4:6]", 14), ("1D", "v4", 17)):
            for op, mods, expected in (("image_load", "glc dlc d16", "OpImageRead"),
                                        ("image_store", "", "OpImageWrite")):
                with self.subTest(dim=dim, op=op):
                    asm = translate(f"{op} v0, {coords}, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_{dim} unorm {mods}")
                    self.check_accesses(asm, (), range(first_binding, first_binding + 3))
                    self.assertIn(expected, asm)

    def test_atomic_texel_pointers_use_storage_indices(self):
        for dim, coords, binding in (("2D", "v[4:5]", 4), ("3D", "v[4:6]", 7)):
            for op, data, expected in (("image_atomic_add", "v0", "OpAtomicIAdd"),
                                        ("image_atomic_cmpswap", "v[0:1]", "OpAtomicCompareExchange")):
                with self.subTest(dim=dim, op=op):
                    asm = translate(f"{op} {data}, {coords}, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_{dim} glc")
                    self.check_accesses(asm, (), (binding,))
                    (texel, image), = re.findall(r"(%\w+) = OpImageTexelPointer %_ptr_Image_uint (%\w+)", asm)
                    self.assertIn(f"{expected} %uint {texel} ", asm)
                    self.assertIn(f"{image} = OpAccessChain %_ptr_UC_simg", asm)
