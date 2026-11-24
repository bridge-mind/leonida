"""Regression tests for tools/spirv_metal_set0.py file hygiene and aliased set-0 bindings.

Standard library only; uses synthetic SPIR-V disassembly, no game data and no external SPIRV-Tools binaries.

    python -I -W error::ResourceWarning tests/test_spirv_metal_set0_aliases.py -v
"""
import ast
import contextlib
import gc
import io
import os
import re
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
TOOL_PATH = os.path.join(TOOLS_DIR, "spirv_metal_set0.py")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_metal_set0 as m  # noqa: E402


def _parse_defs_and_order(text):
    """Parse SSA definitions, check for duplicates and declaration-before-use ordering."""
    defs = {}
    def_line = {}
    decs = []
    lines = text.splitlines()
    for idx, ln in enumerate(lines):
        m_dec = re.match(r"^\s*OpDecorate (%\S+) (\S+) (\d+)$", ln)
        if m_dec:
            decs.append((m_dec.group(1), m_dec.group(2), int(m_dec.group(3))))
            continue
        m_def = re.match(r"^\s*(%\S+) = (.*)$", ln)
        if not m_def:
            continue
        rid, rhs = m_def.group(1), m_def.group(2)
        if rid in defs:
            raise AssertionError(f"duplicate SSA definition for {rid} at line {idx + 1}: {ln}")
        defs[rid] = rhs
        def_line[rid] = idx

    # Verify declaration-before-use for constants, array types, pointers, and variables.
    for rid, rhs in defs.items():
        cur = def_line[rid]
        parts = rhs.split()
        op = parts[0]
        if op == "OpConstant":
            ty = parts[1]
            assert def_line[ty] < cur, f"{rid} uses {ty} before definition"
        elif op == "OpTypeRuntimeArray":
            elem = parts[1]
            assert def_line[elem] < cur, f"{rid} uses {elem} before definition"
        elif op == "OpTypeArray":
            elem, length = parts[1], parts[2]
            assert def_line[elem] < cur, f"{rid} uses {elem} before definition"
            assert def_line[length] < cur, f"{rid} uses {length} before definition"
        elif op == "OpTypePointer":
            pointee = parts[2]
            assert def_line[pointee] < cur, f"{rid} uses {pointee} before definition"
        elif op == "OpVariable":
            ptr = parts[1]
            assert def_line[ptr] < cur, f"{rid} uses {ptr} before definition"

    return defs, def_line, decs


class TestFileHygiene(unittest.TestCase):
    def test_capacities_and_print_layout_emit_no_resource_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            good_hdr = os.path.join(tmp, "good.h")
            bad_hdr = os.path.join(tmp, "bad.h")
            with open(good_hdr, "w", encoding="utf-8") as f:
                f.write("std::array<Binding, 2> bindings = {{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16}, "
                        "{VK_DESCRIPTOR_TYPE_SAMPLER, 8}}};\n")
            with open(bad_hdr, "w", encoding="utf-8") as f:
                f.write("// missing table\n")

            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)
                counts = m.capacities()
                self.assertEqual(len(counts), 26)
                self.assertEqual(m.capacities(good_hdr), [16, 8])
                with self.assertRaises(SystemExit):
                    m.capacities(bad_hdr)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(m.main(["spirv_metal_set0.py", "--print-layout"]), 0)
                gc.collect()

    def test_spirv_metal_set0_has_no_unmanaged_open_calls(self):
        with open(TOOL_PATH, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=TOOL_PATH)
        managed = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        managed.add(id(item.context_expr))
        bare = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and id(node) not in managed
        ]
        self.assertEqual(bare, [])


class TestSingleAndAliasedBindings(unittest.TestCase):
    def test_single_variable_bindings_keep_canonical_msl_ids(self):
        module = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpDecorate %desc_map DescriptorSet 0
               OpDecorate %desc_map Binding 2
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
        %smp = OpTypeSampler
     %rt_img = OpTypeRuntimeArray %img2d
     %rt_smp = OpTypeRuntimeArray %smp
    %ptr_img = OpTypePointer UniformConstant %rt_img
    %ptr_smp = OpTypePointer UniformConstant %rt_smp
     %rt_map = OpTypeRuntimeArray %uint
        %map = OpTypeStruct %rt_map
    %ptr_map = OpTypePointer StorageBuffer %map
     %images = OpVariable %ptr_img UniformConstant
   %samplers = OpVariable %ptr_smp UniformConstant
   %desc_map = OpVariable %ptr_map StorageBuffer
"""
        counts = m.capacities()
        out = m.rewrite(module, counts)
        defs, _, decs = _parse_defs_and_order(out)

        self.assertEqual(defs["%msl_cap_b0"], f"OpConstant %uint {counts[0]}")
        self.assertEqual(defs["%msl_arr_b0"], "OpTypeArray %img2d %msl_cap_b0")
        self.assertEqual(defs["%msl_ptr_b0"], "OpTypePointer UniformConstant %msl_arr_b0")
        self.assertEqual(defs["%images"], "OpVariable %msl_ptr_b0 UniformConstant")

        self.assertEqual(defs["%msl_cap_b1"], f"OpConstant %uint {counts[1]}")
        self.assertEqual(defs["%msl_arr_b1"], "OpTypeArray %smp %msl_cap_b1")
        self.assertEqual(defs["%msl_ptr_b1"], "OpTypePointer UniformConstant %msl_arr_b1")
        self.assertEqual(defs["%samplers"], "OpVariable %msl_ptr_b1 UniformConstant")
        self.assertEqual(defs["%desc_map"], "OpVariable %ptr_map StorageBuffer")

        bindings = {vid: val for vid, kind, val in decs if kind == "Binding"}
        self.assertEqual(bindings, {"%images": 0, "%samplers": 4096, "%desc_map": 5120})

    def test_distinct_element_type_aliases_at_same_binding(self):
        # Mirrors tools/rdna2spirv.py emitting %images + typed %itx* aliases at binding 0
        # and %images_arr + %itx1_1 at binding 3, plus a 3D alias at binding 0.
        module = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpDecorate %images_arr DescriptorSet 0
               OpDecorate %images_arr Binding 3
               OpDecorate %itx1_0 DescriptorSet 0
               OpDecorate %itx1_0 Binding 0
               OpDecorate %itx2_0 DescriptorSet 0
               OpDecorate %itx2_0 Binding 0
               OpDecorate %itx3d_0 DescriptorSet 0
               OpDecorate %itx3d_0 Binding 0
               OpDecorate %itx1_1 DescriptorSet 0
               OpDecorate %itx1_1 Binding 3
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
        %smp = OpTypeSampler
     %img2da = OpTypeImage %float 2D 0 1 0 1 Unknown
  %itximg1_0 = OpTypeImage %uint 2D 0 0 0 1 Unknown
  %itximg2_0 = OpTypeImage %int 2D 0 0 0 1 Unknown
     %img_3d = OpTypeImage %float 3D 0 0 0 1 Unknown
  %itximg1_1 = OpTypeImage %uint 2D 0 1 0 1 Unknown
     %rt_img = OpTypeRuntimeArray %img2d
    %ptr_img = OpTypePointer UniformConstant %rt_img
     %rt_smp = OpTypeRuntimeArray %smp
    %ptr_smp = OpTypePointer UniformConstant %rt_smp
    %rt_imga = OpTypeRuntimeArray %img2da
   %ptr_imga = OpTypePointer UniformConstant %rt_imga
   %rt_itx10 = OpTypeRuntimeArray %itximg1_0
  %ptr_itx10 = OpTypePointer UniformConstant %rt_itx10
   %rt_itx20 = OpTypeRuntimeArray %itximg2_0
  %ptr_itx20 = OpTypePointer UniformConstant %rt_itx20
     %rt_3d0 = OpTypeRuntimeArray %img_3d
    %ptr_3d0 = OpTypePointer UniformConstant %rt_3d0
   %rt_itx11 = OpTypeRuntimeArray %itximg1_1
  %ptr_itx11 = OpTypePointer UniformConstant %rt_itx11
     %images = OpVariable %ptr_img UniformConstant
   %samplers = OpVariable %ptr_smp UniformConstant
 %images_arr = OpVariable %ptr_imga UniformConstant
     %itx1_0 = OpVariable %ptr_itx10 UniformConstant
     %itx2_0 = OpVariable %ptr_itx20 UniformConstant
    %itx3d_0 = OpVariable %ptr_3d0 UniformConstant
     %itx1_1 = OpVariable %ptr_itx11 UniformConstant
"""
        counts = m.capacities()
        ids = m.metal_ids(counts)
        out = m.rewrite(module, counts)
        defs, _, decs = _parse_defs_and_order(out)

        # OpConstant is emitted once per binding that has UniformConstant runtime arrays.
        b0_consts = [rid for rid, rhs in defs.items() if rhs == f"OpConstant %uint {counts[0]}"]
        b1_b3_consts = [rid for rid, rhs in defs.items() if rhs == f"OpConstant %uint {counts[3]}"]
        self.assertEqual(b0_consts, ["%msl_cap_b0"])
        self.assertEqual(counts[1], counts[3])
        self.assertEqual(b1_b3_consts, ["%msl_cap_b1", "%msl_cap_b3"])
        for b in (0, 1, 3):
            self.assertEqual(out.count(f"%msl_cap_b{b} = "), 1)

        # Every variable keeps the exact element type of its original runtime array and binding capacity.
        expected_vars = {
            "%images": ("%img2d", "%msl_cap_b0", ids[0]),
            "%samplers": ("%smp", "%msl_cap_b1", ids[1]),
            "%images_arr": ("%img2da", "%msl_cap_b3", ids[3]),
            "%itx1_0": ("%itximg1_0", "%msl_cap_b0", ids[0]),
            "%itx2_0": ("%itximg2_0", "%msl_cap_b0", ids[0]),
            "%itx3d_0": ("%img_3d", "%msl_cap_b0", ids[0]),
            "%itx1_1": ("%itximg1_1", "%msl_cap_b3", ids[3]),
        }
        bindings = {vid: val for vid, kind, val in decs if kind == "Binding"}
        seen_ptrs = set()
        for var_id, (elem, cap, expected_bind) in expected_vars.items():
            self.assertEqual(bindings[var_id], expected_bind)
            m_var = re.match(r"^OpVariable (%\S+) UniformConstant$", defs[var_id])
            self.assertIsNotNone(m_var, defs[var_id])
            ptr_id = m_var.group(1)
            seen_ptrs.add(ptr_id)
            m_ptr = re.match(r"^OpTypePointer UniformConstant (%\S+)$", defs[ptr_id])
            self.assertIsNotNone(m_ptr, defs[ptr_id])
            arr_id = m_ptr.group(1)
            self.assertEqual(defs[arr_id], f"OpTypeArray {elem} {cap}")

        # Distinct element types get distinct pointer types.
        self.assertEqual(len(seen_ptrs), len(expected_vars))

    def test_reversed_decoration_order_still_declares_cap_before_first_array(self):
        # %itx1_0 is decorated before %images, yet %ptr_img is declared before %ptr_itx.
        module = """\
               OpDecorate %itx1_0 DescriptorSet 0
               OpDecorate %itx1_0 Binding 0
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
  %itximg1_0 = OpTypeImage %int 2D 0 0 0 1 Unknown
     %rt_img = OpTypeRuntimeArray %img2d
    %ptr_img = OpTypePointer UniformConstant %rt_img
     %rt_itx = OpTypeRuntimeArray %itximg1_0
    %ptr_itx = OpTypePointer UniformConstant %rt_itx
     %images = OpVariable %ptr_img UniformConstant
     %itx1_0 = OpVariable %ptr_itx UniformConstant
"""
        out = m.rewrite(module, [4096])
        defs, def_line, decs = _parse_defs_and_order(out)

        self.assertLess(def_line["%msl_cap_b0"], def_line["%msl_arr_b0"])
        self.assertLess(def_line["%msl_cap_b0"], def_line["%msl_arr_b0_1"])
        self.assertEqual(defs["%images"], "OpVariable %msl_ptr_b0 UniformConstant")
        self.assertEqual(defs["%itx1_0"], "OpVariable %msl_ptr_b0_1 UniformConstant")
        self.assertEqual(defs["%msl_arr_b0"], "OpTypeArray %img2d %msl_cap_b0")
        self.assertEqual(defs["%msl_arr_b0_1"], "OpTypeArray %itximg1_0 %msl_cap_b0")
        self.assertEqual([(v, val) for v, k, val in decs if k == "Binding"], [("%itx1_0", 0), ("%images", 0)])

    def test_same_binding_and_same_optypepointer_shares_sized_type_without_duplicates(self):
        module = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %alias DescriptorSet 0
               OpDecorate %alias Binding 0
               OpDecorate %single DescriptorSet 0
               OpDecorate %single Binding 0
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
     %rt_img = OpTypeRuntimeArray %img2d
    %ptr_img = OpTypePointer UniformConstant %rt_img
    %ptr_one = OpTypePointer UniformConstant %img2d
     %images = OpVariable %ptr_img UniformConstant
      %alias = OpVariable %ptr_img UniformConstant
     %single = OpVariable %ptr_one UniformConstant
"""
        out = m.rewrite(module, [128])
        defs, _, decs = _parse_defs_and_order(out)

        self.assertEqual(defs["%msl_cap_b0"], "OpConstant %uint 128")
        self.assertEqual(defs["%msl_arr_b0"], "OpTypeArray %img2d %msl_cap_b0")
        self.assertEqual(defs["%msl_ptr_b0"], "OpTypePointer UniformConstant %msl_arr_b0")
        self.assertEqual(defs["%images"], "OpVariable %msl_ptr_b0 UniformConstant")
        self.assertEqual(defs["%alias"], "OpVariable %msl_ptr_b0 UniformConstant")
        self.assertEqual(defs["%single"], "OpVariable %ptr_one UniformConstant")
        self.assertEqual(out.count("%msl_cap_b0 = "), 1)
        self.assertEqual(out.count("%msl_arr_b0 = "), 1)
        self.assertEqual(out.count("%msl_ptr_b0 = "), 1)
        self.assertEqual({v: val for v, k, val in decs if k == "Binding"},
                         {"%images": 0, "%alias": 0, "%single": 0})

    def test_same_binding_and_same_element_via_distinct_pointers(self):
        module = """\
               OpDecorate %v1 DescriptorSet 0
               OpDecorate %v1 Binding 0
               OpDecorate %v2 DescriptorSet 0
               OpDecorate %v2 Binding 0
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
     %rt_a = OpTypeRuntimeArray %img2d
     %ptr_a = OpTypePointer UniformConstant %rt_a
     %rt_b = OpTypeRuntimeArray %img2d
     %ptr_b = OpTypePointer UniformConstant %rt_b
        %v1 = OpVariable %ptr_a UniformConstant
        %v2 = OpVariable %ptr_b UniformConstant
"""
        out = m.rewrite(module, [64])
        defs, def_line, _ = _parse_defs_and_order(out)

        for var_id in ("%v1", "%v2"):
            m_var = re.match(r"^OpVariable (%\S+) UniformConstant$", defs[var_id])
            self.assertIsNotNone(m_var)
            ptr_id = m_var.group(1)
            self.assertLess(def_line[ptr_id], def_line[var_id])
            m_ptr = re.match(r"^OpTypePointer UniformConstant (%\S+)$", defs[ptr_id])
            self.assertIsNotNone(m_ptr)
            self.assertEqual(defs[m_ptr.group(1)], "OpTypeArray %img2d %msl_cap_b0")

    def test_late_uint_is_defined_before_the_synthesized_array_length(self) -> None:
        module = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
     %rt_img = OpTypeRuntimeArray %img2d
    %ptr_img = OpTypePointer UniformConstant %rt_img
     %images = OpVariable %ptr_img UniformConstant
       %uint = OpTypeInt 32 0
  %old_count = OpConstant %uint 4
"""
        _parse_defs_and_order(module)
        defs, _, _ = _parse_defs_and_order(m.rewrite(module, [64]))

        self.assertEqual(defs["%msl_cap_b0"], "OpConstant %uint 64")
        self.assertEqual(defs["%images"], "OpVariable %msl_ptr_b0 UniformConstant")
        self.assertEqual(defs["%old_count"], "OpConstant %uint 4")

    def test_preexisting_msl_ids_in_input_never_collide(self):
        module = """\
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %itx1_0 DescriptorSet 0
               OpDecorate %itx1_0 Binding 0
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
%msl_cap_b0 = OpConstant %uint 7
%msl_arr_b0 = OpTypeFloat 16
%msl_ptr_b0 = OpTypeStruct %float
%msl_arr_b0_1 = OpConstant %uint 9
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
  %itximg1_0 = OpTypeImage %int 2D 0 0 0 1 Unknown
     %rt_img = OpTypeRuntimeArray %img2d
    %ptr_img = OpTypePointer UniformConstant %rt_img
     %rt_itx = OpTypeRuntimeArray %itximg1_0
    %ptr_itx = OpTypePointer UniformConstant %rt_itx
     %images = OpVariable %ptr_img UniformConstant
     %itx1_0 = OpVariable %ptr_itx UniformConstant
"""
        out = m.rewrite(module, [4096])
        defs, _, _ = _parse_defs_and_order(out)

        # Pre-existing IDs keep their original definitions untouched.
        self.assertEqual(defs["%msl_cap_b0"], "OpConstant %uint 7")
        self.assertEqual(defs["%msl_arr_b0"], "OpTypeFloat 16")
        self.assertEqual(defs["%msl_ptr_b0"], "OpTypeStruct %float")
        self.assertEqual(defs["%msl_arr_b0_1"], "OpConstant %uint 9")

        # Newly synthesized IDs avoid every existing definition and match each variable's element type.
        m_img = re.match(r"^OpVariable (%\S+) UniformConstant$", defs["%images"])
        m_itx = re.match(r"^OpVariable (%\S+) UniformConstant$", defs["%itx1_0"])
        self.assertIsNotNone(m_img)
        self.assertIsNotNone(m_itx)
        ptr_img, ptr_itx = m_img.group(1), m_itx.group(1)
        self.assertNotEqual(ptr_img, ptr_itx)

        arr_img = re.match(r"^OpTypePointer UniformConstant (%\S+)$", defs[ptr_img]).group(1)
        arr_itx = re.match(r"^OpTypePointer UniformConstant (%\S+)$", defs[ptr_itx]).group(1)
        cap_img = re.match(r"^OpTypeArray %img2d (%\S+)$", defs[arr_img]).group(1)
        cap_itx = re.match(r"^OpTypeArray %itximg1_0 (%\S+)$", defs[arr_itx]).group(1)
        self.assertEqual(cap_img, cap_itx)
        self.assertEqual(defs[cap_img], "OpConstant %uint 4096")


if __name__ == "__main__":
    unittest.main()
