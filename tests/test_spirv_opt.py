"""Tests for tools/spirv_opt.py (sound basic-block SPIR-V peephole optimizer, #884)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_opt  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


class SpirvOptTests(unittest.TestCase):
    def test_alu_chain_bit_exact_and_shrinks_significantly(self):
        lines = [
            "v_mov_b32_e32 v1, 0x01020304",
            "v_mov_b32_e32 v2, 0x05060708",
            "v_mov_b32_e32 v3, 0x000003e8",
            "v_dot4_i32_i8 v8, v1, v2, v3",
            "v_dot4c_i32_i8 v8, v1, v2",
            "v_add_f32_e32 v9, v8, 1.0",
            "v_mul_f32_e32 v10, v9, 2.0",
            "exp mrt0 v8, v9, v10, v8 done vm",
            "s_endpgm",
        ]
        raw = support.translate_lines("ps_opt_alu", lines)
        opt = spirv_opt.optimize(raw)
        _validate(opt)
        self.assertEqual(support.mrt0_words(opt), support.mrt0_words(raw))
        # Straight-line ALU chain should shed >50% of its lines (register traffic, %active loads, round-trip bitcasts,
        # unused push-constant prologue loads, and duplicate pure expressions).
        self.assertLess(len(opt.splitlines()), len(raw.splitlines()) * 0.5)
        self.assertNotIn("OpLoad %bool %active", opt)

    def test_dynamic_register_indexing_preserved_soundly(self):
        # v_movrels_b32 / s_movrels_b32 use dynamic OpAccessChain into %vgpr / %sgpr; optimizer must NOT
        # drop stores or forward stale values across dynamic accesses.
        lines = [
            "v_mov_b32_e32 v1, 0x11111111",
            "v_mov_b32_e32 v2, 0x22222222",
            "v_mov_b32_e32 v3, 0x33333333",
            "s_mov_b32 m0, 1",
            "v_movrels_b32_e32 v8, v1",   # reads v[1 + 1] = v2 = 0x22222222
            "s_mov_b32 m0, 2",
            "v_movrels_b32_e32 v9, v1",   # reads v[1 + 2] = v3 = 0x33333333
            "v_movreld_b32_e32 v1, v8",   # writes v[1 + 2] = v3 = 0x22222222
            "v_mov_b32_e32 v10, v3",      # must observe the dynamic write -> 0x22222222, NOT stale 0x33333333!
            "exp mrt0 v8, v9, v10, v2 done vm",
            "s_endpgm",
        ]
        raw = support.translate_lines("ps_opt_dyn_reg", lines)
        opt = spirv_opt.optimize(raw)
        _validate(opt)
        expected = [0x22222222, 0x33333333, 0x22222222, 0x22222222]
        self.assertEqual(support.mrt0_words(raw), expected)
        self.assertEqual(support.mrt0_words(opt), expected)

    def test_control_flow_and_exec_masking_preserved(self):
        # Branches + exec masking across blocks: intra-block forwarding must reset at OpLabel boundaries.
        lines = [
            "v_mov_b32_e32 v1, 10 ; encoding: [0x8a,0x02,0x02,0x7e]",
            "v_mov_b32_e32 v2, 20 ; encoding: [0x94,0x02,0x04,0x7e]",
            "v_mov_b32_e32 v8, 0 ; encoding: [0x80,0x02,0x10,0x7e]",
            "s_mov_b32 s4, 0 ; encoding: [0x80,0x03,0x84,0xbe]",
            "s_cmp_eq_u32 s4, 0 ; encoding: [0x04,0x80,0x06,0xbf]",
            "s_cbranch_scc0 2 ; encoding: [0x02,0x00,0x84,0xbf]",
            "v_add_nc_u32_e32 v8, v1, v2 ; encoding: [0x01,0x05,0x10,0x4a]",
            "s_branch 1 ; encoding: [0x01,0x00,0x82,0xbf]",
            "v_sub_nc_u32_e32 v8, v2, v1 ; encoding: [0x02,0x03,0x10,0x4c]",
            "v_add_nc_u32_e32 v9, v8, 5 ; encoding: [0x85,0x10,0x12,0x4a]",
            "exp mrt0 v8, v9, v1, v2 done vm ; encoding: [0x0f,0x18,0x00,0xf8,0x08,0x09,0x01,0x02]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        raw = support.translate_lines("ps_opt_cf", lines)
        opt = spirv_opt.optimize(raw)
        _validate(opt)
        self.assertEqual(support.mrt0_words(opt), support.mrt0_words(raw))
        self.assertEqual(support.mrt0_words(opt), [30, 35, 10, 20])
        # Non-divergent scalar-branch shader never stores %active: cross-block read-only propagation eliminates
        # every OpLoad %bool %active and the dead %active OpVariable across all blocks.
        self.assertNotIn("OpLoad %bool %active", opt)
        self.assertNotIn("%active = OpVariable", opt)

    def test_cross_block_constant_folding_and_dead_func_vars(self):
        # Constant integer/bit folding + algebraic identities across straight-line ALU.
        lines = [
            "v_mov_b32_e32 v1, 2",
            "v_lshlrev_b32_e32 v2, 2, v1",      # 2 << (2 & 0x1f) -> folds (2 & 0x1f) and shift
            "v_add_nc_u32_e32 v3, v2, 0",       # + 0 identity
            "v_xor_b32_e32 v4, v3, v3",         # x ^ x -> 0
            "v_or_b32_e32 v5, v3, 0",           # | 0 identity
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ]
        raw = support.translate_lines("ps_opt_fold", lines)
        opt = spirv_opt.optimize(raw)
        _validate(opt)
        self.assertEqual(support.mrt0_words(opt), [8, 8, 0, 8])
        self.assertNotIn("OpBitwiseXor", opt)
        self.assertNotIn("OpIAdd", opt)
        # Unused Function-scope variables (%m0, %scc, %active, etc.) should be removed.
        self.assertNotIn("%m0 = OpVariable", opt)
        self.assertNotIn("%active = OpVariable", opt)

    def test_trivial_phi_and_dominator_cse_in_loop(self):
        # Hand-crafted SPIR-V module exercising trivial OpPhi (identical incoming + self-loop) and
        # dominator-tree OpAccessChain + Input OpLoad CSE across a structured loop.
        spvasm = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in0 Location 0
               OpDecorate %in0 Flat
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_In = OpTypePointer Input %uint
   %_ptr_Out = OpTypePointer Output %v4uint
   %_ptr_Fun = OpTypePointer Function %uint
        %c0  = OpConstant %uint 0
        %c1  = OpConstant %uint 1
        %c4  = OpConstant %uint 4
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %in0 = OpVariable %_ptr_In Input
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %local = OpVariable %_ptr_Fun Function %c0
         %v0 = OpLoad %uint %in0
               OpStore %local %v0
               OpBranch %head
       %head = OpLabel
        %phi = OpPhi %uint %v0 %entry %phi %cont
          %i = OpPhi %uint %c0 %entry %inext %cont
               OpLoopMerge %exit %cont None
               OpBranch %body
       %body = OpLabel
         %v1 = OpLoad %uint %in0
        %sum = OpIAdd %uint %phi %v1
       %cond = OpULessThan %bool %i %c4
               OpSelectionMerge %merge None
               OpBranchConditional %cond %cont %exit
      %merge = OpLabel
               OpUnreachable
       %cont = OpLabel
      %inext = OpIAdd %uint %i %c1
               OpBranch %head
       %exit = OpLabel
        %res = OpCompositeConstruct %v4uint %phi %v0 %c0 %c1
               OpStore %out0 %res
               OpReturn
               OpFunctionEnd
"""
        _validate(spvasm)
        opt = spirv_opt.optimize(spvasm)
        _validate(opt)
        # Trivial self-referential %phi must collapse to %v0, duplicate %v1 = OpLoad %in0 CSEs to %v0,
        # dead %sum and dead %local OpVariable are eliminated.
        self.assertNotIn("%phi = OpPhi", opt)
        self.assertEqual(opt.count("OpLoad %uint %in0"), 1)
        self.assertNotIn("%local = OpVariable", opt)
        self.assertNotIn("%sum = OpIAdd", opt)

    def test_alias_signed_unsigned_index_volatile_and_aggregate_overlap(self):
        # Regression tests for standalone memory aliasing cases (PR #2298 review):
        # 1) Store 1 via %uint 0, store 2 via %int 0, load via first pointer -> must observe 2, not 1.
        # 2) Store 1, Volatile store 2, ordinary load -> must NOT forward stale 1.
        # 3) Store 1, Volatile load, store 2 -> must NOT kill store 1 before volatile load.
        # 4) Store element 0 = 1, store whole array = {2, 2}, load element 0 -> must NOT forward stale 1.
        spv_idx = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
         %c2 = OpConstant %uint 2
        %arr = OpTypeArray %uint %c2
     %v4uint = OpTypeVector %uint 4
    %_ptr_Fa = OpTypePointer Function %arr
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
        %u0  = OpConstant %uint 0
        %s0  = OpConstant %int 0
        %u1  = OpConstant %uint 1
        %u2  = OpConstant %uint 2
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpVariable %_ptr_Fa Function
         %p0 = OpAccessChain %_ptr_Fu %a %u0
         %p1 = OpAccessChain %_ptr_Fu %a %s0
               OpStore %p0 %u1
               OpStore %p1 %u2
          %r = OpLoad %uint %p0
        %vec = OpCompositeConstruct %v4uint %r %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _validate(spv_idx)
        opt_idx = spirv_opt.optimize(spv_idx)
        _validate(opt_idx)
        self.assertIn("OpCompositeConstruct %v4uint %u2 %u0 %u0 %u0", opt_idx)

        spv_vol_store = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
        %u0  = OpConstant %uint 0
        %u1  = OpConstant %uint 1
        %u2  = OpConstant %uint 2
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %x = OpVariable %_ptr_Fu Function
               OpStore %x %u1
               OpStore %x %u2 Volatile
          %r = OpLoad %uint %x
        %vec = OpCompositeConstruct %v4uint %r %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _validate(spv_vol_store)
        opt_vs = spirv_opt.optimize(spv_vol_store)
        _validate(opt_vs)
        self.assertIn("OpStore %x %u2 Volatile", opt_vs)
        self.assertIn("%r = OpLoad %uint %x", opt_vs)
        self.assertIn("OpCompositeConstruct %v4uint %r %u0 %u0 %u0", opt_vs)

        spv_vol_load = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v4uint = OpTypeVector %uint 4
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
        %u0  = OpConstant %uint 0
        %u1  = OpConstant %uint 1
        %u2  = OpConstant %uint 2
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %x = OpVariable %_ptr_Fu Function
               OpStore %x %u1
          %r = OpLoad %uint %x Volatile
               OpStore %x %u2
        %vec = OpCompositeConstruct %v4uint %r %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _validate(spv_vol_load)
        opt_vl = spirv_opt.optimize(spv_vol_load)
        _validate(opt_vl)
        self.assertIn("OpStore %x %u1", opt_vl)
        self.assertIn("%r = OpLoad %uint %x Volatile", opt_vl)
        self.assertIn("OpStore %x %u2", opt_vl)

        spv_agg = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out0
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
         %c2 = OpConstant %uint 2
        %arr = OpTypeArray %uint %c2
     %v4uint = OpTypeVector %uint 4
    %_ptr_Fa = OpTypePointer Function %arr
    %_ptr_Fu = OpTypePointer Function %uint
   %_ptr_Out = OpTypePointer Output %v4uint
        %u0  = OpConstant %uint 0
        %u1  = OpConstant %uint 1
        %u2  = OpConstant %uint 2
       %a22  = OpConstantComposite %arr %u2 %u2
       %out0 = OpVariable %_ptr_Out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpVariable %_ptr_Fa Function
         %p0 = OpAccessChain %_ptr_Fu %a %u0
               OpStore %p0 %u1
               OpStore %a %a22
          %r = OpLoad %uint %p0
        %vec = OpCompositeConstruct %v4uint %r %u0 %u0 %u0
               OpStore %out0 %vec
               OpReturn
               OpFunctionEnd
"""
        _validate(spv_agg)
        opt_agg = spirv_opt.optimize(spv_agg)
        _validate(opt_agg)
        self.assertIn("OpStore %a %a22", opt_agg)
        self.assertIn("%r = OpLoad %uint %p0", opt_agg)
        self.assertIn("OpCompositeConstruct %v4uint %r %u0 %u0 %u0", opt_agg)

    def test_wave64_emulated_compute_validates_and_shrinks(self):
        cs_lines = [
            "s_load_dwordx2 s[4:5], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, 0x3f800000",
            "v_add_f32_e32 v2, v1, 2.0",
            "v_mul_f32_e32 v3, v2, 3.0",
            "v_lshlrev_b32_e32 v4, 2, v0",
            "ds_write_b32 v4, v3",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v5, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_cmp_lt_u32_e64 vcc, v0, 32",
            "v_cndmask_b32_e64 v6, v5, v3, vcc",
            "global_store_dword v4, v6, s[4:5]",
            "s_endpgm",
        ]
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            s_path = os.path.join(d, "disasm", "cs_opt_w64.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_opt_w64  code 0x0..0x100, gfx1013, 0 invalid words\n")
                for ln in cs_lines:
                    f.write(f"\t{ln} ; encoding: [0x00,0x00,0x00,0x00]\n")
            with open(os.path.join(d, "fspx", "cs_opt_w64.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (8 << 1) | (16 << 15), "threads": [64, 1, 1]}, f)
            old_w64, old_opt = rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV
            try:
                rdna2spirv.WAVE64_EMULATE = True
                rdna2spirv.OPT_SPIRV = False
                raw = rdna2spirv.translate(s_path)
                rdna2spirv.OPT_SPIRV = True
                opt = rdna2spirv.translate(s_path)
            finally:
                rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV = old_w64, old_opt
        _validate(opt)
        self.assertLess(len(opt.splitlines()), len(raw.splitlines()))

    def test_decorations_on_deleted_ids_removed_and_surviving_kept(self):
        # Deleting a dead NoContraction float op must also drop its OpDecorate line so spirv-val stays clean,
        # while a live NoContraction float op keeps its OpDecorate.
        lines = [
            "v_mov_b32_e32 v1, 1.0",
            "v_add_f32_e32 v2, v1, 2.0",   # dead: overwritten below before any read
            "v_mul_f32_e32 v2, v1, 3.0",   # live: exported
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        raw = support.translate_lines("ps_opt_dec", lines)
        opt = spirv_opt.optimize(raw)
        _validate(opt)
        self.assertEqual(support.mrt0_words(opt), support.mrt0_words(raw))
        self.assertIn("NoContraction", opt)
        # Dead OpFAdd must be gone along with its decoration.
        self.assertNotIn("OpFAdd", opt)

    def test_cli_round_trip(self):
        raw = support.translate_lines("ps_opt_cli", [
            "v_mov_b32_e32 v1, 42",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ])
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.spvasm"), os.path.join(d, "out.spvasm")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(raw)
            rc = spirv_opt.main([src, "-o", dst])
            self.assertEqual(rc, 0)
            with open(dst, encoding="utf-8") as f:
                opt = f.read()
        _validate(opt)
        self.assertEqual(support.mrt0_words(opt), [42] * 4)

    def test_trivial_phi_requires_strict_dominance(self):
        # 1. Reachable loop header %hdr has:
        #    - %p_dead whose sole non-self operand %u comes from unreachable %dead,
        #    - %p_body whose sole non-self operand %x comes from non-dominating %latch, and
        #    - %p_self whose sole non-self operand %s is defined later in %hdr itself
        #      (cand_blk == b, exercising strict vs. non-strict dominance).
        #    None of %u, %x, or %s strictly dominates %hdr, so trivial OpPhi simplification
        #    must leave all three phis intact instead of creating a use-not-dominated-by-def
        #    or self-referential '%s = OpIAdd %uint %s %c1' violation.
        # 2. Control: %p_ok = OpPhi %uint %v0 %entry %p_ok %latch IS strictly dominated by
        #    %entry's %v0 and must still fold to %v0.
        spv = "\n".join([
            "               OpCapability Shader",
            "               OpMemoryModel Logical GLSL450",
            '               OpEntryPoint Fragment %main "main" %in0 %out0',
            "               OpExecutionMode %main OriginUpperLeft",
            "               OpDecorate %in0 Location 0",
            "               OpDecorate %in0 Flat",
            "               OpDecorate %out0 Location 0",
            "       %void = OpTypeVoid",
            "       %fn_t = OpTypeFunction %void",
            "       %uint = OpTypeInt 32 0",
            "     %v4uint = OpTypeVector %uint 4",
            "       %bool = OpTypeBool",
            "  %ptr_in_u  = OpTypePointer Input %uint",
            " %ptr_out_v4 = OpTypePointer Output %v4uint",
            "        %in0 = OpVariable %ptr_in_u Input",
            "       %out0 = OpVariable %ptr_out_v4 Output",
            "         %c0 = OpConstant %uint 0",
            "         %c1 = OpConstant %uint 1",
            "      %false = OpConstantFalse %bool",
            "       %main = OpFunction %void None %fn_t",
            "      %entry = OpLabel",
            "         %v0 = OpLoad %uint %in0",
            "               OpBranch %hdr",
            "       %dead = OpLabel",
            "         %u0 = OpLoad %uint %in0",
            "          %u = OpIAdd %uint %u0 %c1",
            "               OpBranch %hdr",
            "        %hdr = OpLabel",
            "     %p_dead = OpPhi %uint %p_dead %entry %u %dead",
            "     %p_body = OpPhi %uint %p_body %entry %x %latch",
            "     %p_self = OpPhi %uint %p_self %entry %s %latch",
            "       %p_ok = OpPhi %uint %v0 %entry %p_ok %latch",
            "          %s = OpIAdd %uint %p_self %c1",
            "               OpLoopMerge %exit %latch None",
            "               OpBranchConditional %false %latch %exit",
            "      %latch = OpLabel",
            "          %x = OpIAdd %uint %p_body %c1",
            "               OpBranch %hdr",
            "       %exit = OpLabel",
            "        %vec = OpCompositeConstruct %v4uint %p_dead %p_body %p_ok %p_self",
            "               OpStore %out0 %vec",
            "               OpReturn",
            "               OpFunctionEnd",
            "",
        ])
        opt = spirv_opt.optimize(spv)
        # Non-strictly-dominating phis (including %p_self where cand_blk == b) must survive.
        self.assertIn("%p_dead = OpPhi", opt)
        self.assertIn("%p_body = OpPhi", opt)
        self.assertIn("%p_self = OpPhi", opt)
        self.assertIn("%s = OpIAdd %uint %p_self %c1", opt)
        # Strictly dominated trivial phi %p_ok must still fold away.
        self.assertNotIn("%p_ok = OpPhi", opt)
        self.assertIn("OpCompositeConstruct %v4uint %p_dead %p_body %v0 %p_self", opt)

    def test_decorate_id_string_and_group_preserved_or_cleaned(self):
        # Live ops decorated via OpDecorateString, OpDecorateStringGOOGLE, OpDecorateId, or
        # OpGroupDecorate (including tab-separated lines) must not be folded/CSE'd away; dead
        # ones removed by DCE must have their decoration lines dropped (or target removed from
        # OpGroupDecorate) with no dangling SSA IDs left in the module header.
        spv = "\n".join([
            "               OpCapability Shader",
            '               OpExtension "SPV_GOOGLE_decorate_string"',
            "               OpMemoryModel Logical GLSL450",
            '               OpEntryPoint Fragment %main "main" %out0',
            "               OpExecutionMode %main OriginUpperLeft",
            "               OpDecorate %out0 Location 0",
            "        %grp = OpDecorationGroup",
            "               OpDecorate %grp NoContraction",
            "               OpGroupDecorate %grp %live_grp %dead_grp",
            "               OpGroupDecorate\t%grp %dead_only",
            '               OpDecorateString\t%live_str UserSemantic "keep"',
            '               OpDecorateString\t%dead_str UserSemantic "drop"',
            '               OpDecorateStringGOOGLE %live_g HlslSemanticGOOGLE "KEEP"',
            '               OpDecorateStringGOOGLE\t%dead_g HlslSemanticGOOGLE "DROP"',
            "               OpDecorateId %live_id CounterBuffer %out0",
            "               OpDecorateId\t%dead_id CounterBuffer %out0",
            "       %void = OpTypeVoid",
            "       %fn_t = OpTypeFunction %void",
            "       %uint = OpTypeInt 32 0",
            "     %v4uint = OpTypeVector %uint 4",
            " %ptr_out_v4 = OpTypePointer Output %v4uint",
            "       %out0 = OpVariable %ptr_out_v4 Output",
            "         %c0 = OpConstant %uint 0",
            "         %c1 = OpConstant %uint 1",
            "       %main = OpFunction %void None %fn_t",
            "      %entry = OpLabel",
            "   %live_grp = OpIAdd %uint %c1 %c0",
            "   %dead_grp = OpIAdd %uint %c1 %c0",
            "  %dead_only = OpIAdd %uint %c1 %c0",
            "   %live_str = OpIAdd %uint %c1 %c0",
            "   %dead_str = OpIAdd %uint %c1 %c0",
            "     %live_g = OpIAdd %uint %c1 %c0",
            "     %dead_g = OpIAdd %uint %c1 %c0",
            "    %live_id = OpIAdd %uint %c1 %c0",
            "    %dead_id = OpIAdd %uint %c1 %c0",
            "        %vec = OpCompositeConstruct %v4uint %live_grp %live_str %live_g %live_id",
            "               OpStore %out0 %vec",
            "               OpReturn",
            "               OpFunctionEnd",
            "",
        ])
        opt = spirv_opt.optimize(spv)
        # All four live decorated ops must survive (not folded to %c1) along with their decorations.
        for live in ("%live_grp", "%live_str", "%live_g", "%live_id"):
            self.assertIn(f"{live} = OpIAdd", opt)
        self.assertIn('OpDecorateString\t%live_str UserSemantic "keep"', opt)
        self.assertIn('OpDecorateStringGOOGLE %live_g HlslSemanticGOOGLE "KEEP"', opt)
        self.assertIn("OpDecorateId %live_id CounterBuffer %out0", opt)
        self.assertIn("OpGroupDecorate %grp %live_grp", opt)
        # All dead decorated ops and their dangling decorations must be gone.
        for dead in ("%dead_grp", "%dead_only", "%dead_str", "%dead_g", "%dead_id"):
            self.assertNotIn(dead, opt)


if __name__ == "__main__":
    unittest.main()
