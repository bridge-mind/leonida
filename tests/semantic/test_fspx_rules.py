"""Semantic tests for FSPX / FidelityFX translator rules (#1754 follow-ups, docs/fspx.md):

1. LLVM VCCTuple scalar register lists ending in vcc_lo,vcc_hi (`[s100,...,s105,vcc_lo,vcc_hi]` and
   `[s92,...,s105,vcc_lo,vcc_hi]`) parsed in tools/shader_ir.py, clearing `vcc` in tools/lane_masks.py,
   and handled in s_buffer_load_dwordx*/s_load_dword*, buffer/global addressing, and T#/S# lookups.
2. `src_scc` special source operand (RDNA2 encoding 253 = SCC, 0 or 1) in scalar and vector ALU ops.
3. `image_atomic_*` / `image_atomic_cmpswap` with `a16` packed 16-bit unsigned coordinates (2D and 3D).

Standard library only; no GPU, no LLVM, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE = 0x1000
OUT = 0x3000


def translate(body, *, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_fspx.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_fspx.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_fspx.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_fspx\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(body, memory):
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


class VccTupleSmemTests(unittest.TestCase):
    def test_shader_ir_parses_vcc_tuple_lists(self):
        op8 = shader_ir.operand("[s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]")
        self.assertEqual(op8, {"k": "slist", "r": [100, 101, 102, 103, 104, 105, "vcc_lo", "vcc_hi"], "n": 8})
        op4 = shader_ir.operand("[s104, s105, vcc_lo, vcc_hi]")
        self.assertEqual(op4, {"k": "slist", "r": [104, 105, "vcc_lo", "vcc_hi"], "n": 4})
        ins = shader_ir.parse_line(
            "\ts_buffer_load_dwordx8 [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[0:3], 0x0"
        )
        self.assertEqual(ins["args"][0]["k"], "slist")
        self.assertEqual(ins["args"][0]["n"], 8)

    def test_lane_masks_and_shader_meta_handle_slist(self):
        # A load into [s104,s105,vcc_lo,vcc_hi] must clear vcc from lane-mask tracking.
        state = {104: lane_masks.MASK, "vcc": lane_masks.MASK}
        lane_masks.put(state, {"k": "slist", "r": [104, 105, "vcc_lo", "vcc_hi"], "n": 4}, None)
        self.assertNotIn(104, state)
        self.assertNotIn("vcc", state)
        self.assertEqual(
            shader_meta._sgpr_regs({"k": "slist", "r": [104, 105, "vcc_lo", "vcc_hi"], "n": 4}),
            [104, 105],
        )

    def test_s_buffer_load_dwordx8_ending_in_vcc(self):
        words = [0x11110000 + i for i in range(8)]
        mem = {BASE + 4 * i: w for i, w in enumerate(words)}
        body = [
            f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 32", "s_mov_b32 s7, 0x30000000",
            "s_buffer_load_dwordx8 [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, s100",
            "v_mov_b32_e32 v11, s105",
            "v_mov_b32_e32 v12, vcc_lo",
            "v_mov_b32_e32 v13, vcc_hi",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dwordx4 v6, v[10:13], s[0:1]",
        ]
        res = run_cs(body, mem)
        self.assertEqual([res.get(OUT + 4 * i) for i in range(4)], [words[0], words[5], words[6], words[7]])

    def test_s_buffer_load_dwordx16_ending_in_vcc(self):
        words = [0x22220000 + i for i in range(16)]
        mem = {BASE + 4 * i: w for i, w in enumerate(words)}
        regs = ",".join(f"s{r}" for r in range(92, 106)) + ",vcc_lo,vcc_hi"
        body = [
            f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 64", "s_mov_b32 s7, 0x30000000",
            f"s_buffer_load_dwordx16 [{regs}], s[4:7], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, s92",
            "v_mov_b32_e32 v11, s105",
            "v_mov_b32_e32 v12, vcc_lo",
            "v_mov_b32_e32 v13, vcc_hi",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dwordx4 v6, v[10:13], s[0:1]",
        ]
        res = run_cs(body, mem)
        self.assertEqual([res.get(OUT + 4 * i) for i in range(4)], [words[0], words[13], words[14], words[15]])

    def test_slist_as_vsharp_and_tsharp(self):
        # Load a V# into [s104,s105,vcc_lo,vcc_hi], then use it as a V# for both s_buffer_load and buffer_load.
        vsharp = [BASE + 0x100, 0, 16, 0x30000000]
        mem = {BASE + 4 * i: w for i, w in enumerate(vsharp)}
        mem[BASE + 0x100] = 0xCAFE1234
        mem[BASE + 0x104] = 0xBEEF5678
        body = [
            f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 16", "s_mov_b32 s7, 0x30000000",
            "s_buffer_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_buffer_load_dword s10, [s104,s105,vcc_lo,vcc_hi], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, 4",
            "buffer_load_dword v11, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32_e32 v10, s10",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dwordx2 v6, v[10:11], s[0:1]",
        ]
        res = run_cs(body, mem)
        self.assertEqual([res.get(OUT), res.get(OUT + 4)], [0xCAFE1234, 0xBEEF5678])

        # Also verify translation of an image_load using an 8-dword VCCTuple T#.
        spv = translate([
            "s_buffer_load_dwordx8 [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0",
            "image_load v[4:7], v[1:2], [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dword v6, v4, s[0:1]",
        ])
        self.assertIn("OpFunctionCall %uint %desc_lookup_fn", spv)

    def test_slist_length_mismatch_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate([
                "s_buffer_load_dwordx8 [s104,s105,vcc_lo,vcc_hi], s[0:3], 0x0",
            ])


class SrcSccTests(unittest.TestCase):
    def test_src_scc_in_scalar_and_vector_alu(self):
        body = [
            "s_mov_b32 s4, 5", "s_mov_b32 s5, 5",
            "s_cmp_eq_u32 s4, s5",                 # SCC = 1
            "s_add_u32 s6, 10, src_scc",           # 11
            "v_mov_b32_e32 v1, 20",
            "v_add_nc_u32_e32 v2, src_scc, v1",    # 21 (still 1, s_add_u32 above set SCC=0? check!)
        ]
        # Note: s_add_u32 writes SCC = carry (0 here!), so test SCC=1 and SCC=0 explicitly.
        body = [
            "s_cmp_eq_u32 5, 5",                   # SCC = 1
            "v_mov_b32_e32 v1, 20",
            "v_add_nc_u32_e32 v2, src_scc, v1",    # 21
            "s_mov_b32 s6, src_scc",               # 1 (s_mov_b32 does not touch SCC)
            "s_add_u32 s7, 10, src_scc",           # 11, carry=0 -> SCC = 0
            "v_add_nc_u32_e32 v3, src_scc, v1",    # 20
            "v_mov_b32_e32 v4, s6",
            "v_mov_b32_e32 v5, s7",
            f"v_mov_b32_e32 v8, {OUT:#x}",
            "global_store_dwordx4 v8, v[2:5], s[0:1]",
        ]
        res = run_cs(body, {})
        self.assertEqual([res.get(OUT + 4 * i) for i in range(4)], [21, 20, 1, 11])


class ImageAtomicA16Tests(unittest.TestCase):
    def test_image_atomic_add_2d_and_3d_a16_translate(self):
        # 2D a16: one VGPR packs (x: u16 low, y: u16 high); 3D a16: two VGPRs pack (x, y) and (z, _).
        spv_2d = translate([
            "v_mov_b32_e32 v1, 0x00050003",        # x=3, y=5
            "v_mov_b32_e32 v2, 1",
            "image_atomic_add v2, v1, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 glc",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dword v6, v2, s[0:1]",
        ])
        self.assertIn("OpImageTexelPointer", spv_2d)
        self.assertIn("OpAtomicIAdd", spv_2d)
        self.assertIn("OpShiftRightLogical %uint", spv_2d)

        spv_3d = translate([
            "v_mov_b32_e32 v1, 0x00050003",        # x=3, y=5
            "v_mov_b32_e32 v2, 0x00000007",        # z=7
            "v_mov_b32_e32 v3, 1",
            "image_atomic_add v3, v[1:2], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm a16",
            "s_waitcnt vmcnt(0)",
        ])
        self.assertIn("OpCompositeConstruct %v3int", spv_3d)
        self.assertIn("OpAtomicIAdd", spv_3d)

    def test_image_atomic_2d_too_few_coords_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate([
                "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 1",
                "image_atomic_add v2, v1, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            ])


if __name__ == "__main__":
    unittest.main()
