"""Semantic tests for multi-dword LDS instructions: ds_write_b96, ds_read_b96, ds_write_b128, ds_read_b128.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA ("Data Share Instructions"):
  * ds_write_b96  vaddr, vdata[0:2] [offset:imm]:
        MEM[ADDR + 0] = vdata[0], MEM[ADDR + 4] = vdata[1], MEM[ADDR + 8] = vdata[2]
  * ds_read_b96   vdst[0:2], vaddr  [offset:imm]:
        vdst[0] = MEM[ADDR + 0], vdst[1] = MEM[ADDR + 4], vdst[2] = MEM[ADDR + 8]
  * ds_write_b128 vaddr, vdata[0:3] [offset:imm]:
        MEM[ADDR + 0..12] = vdata[0..3]
  * ds_read_b128  vdst[0:3], vaddr  [offset:imm]:
        vdst[0..3] = MEM[ADDR + 0..12]
where ADDR = (vaddr + offset) & 0xFFFFFFFF. Inactive lanes (EXEC = 0) must neither
modify LDS nor overwrite destination VGPRs.
"""
import os
import re
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import spirv_mini_interp  # noqa: E402
import support  # noqa: E402


def fssm(rsrc2, group):
    """Build the minimal compute-shader FSSM header read by rdna2spirv (RSRC2 + threadgroup size)."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# 4 user SGPRs (the output V#), workgroup id in s4, 64x1x1 threads.
CS_FSSM = fssm(0x8088, (64, 1, 1))
# Output V# passed in s[0:3]: base 0, num_records 0xffffffff, 32_FLOAT, OOB_SELECT 3.
OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

W3 = (0x11223344, 0x55667788, 0x99AABBCC)
W4 = (0xA1B2C3D4, 0xE5F60718, 0x293A4B5C, 0x6D7E8F90)


def run_spv(spv):
    """Run a translated compute shader on lane 0 and return the output buffer memory dict."""
    memory = {}
    outputs = spirv_mini_interp.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=OUT_VSHARP + [0] * 28,
        memory=memory,
    )
    assert outputs == {}, f"unexpected fragment outputs: {outputs}"
    return memory


def translate_cs(name, lines):
    """Translate a compute instruction list (ending with s_endpgm) to SPIR-V assembly."""
    return support.translate_lines(name, list(lines) + ["s_endpgm"], fssm=CS_FSSM)


def load_vgprs(base_reg, values):
    """Emit v_mov_b32_e32 instructions to populate consecutive VGPRs."""
    return [f"v_mov_b32_e32 v{base_reg + i}, {val:#x}" for i, val in enumerate(values)]


def store_vgprs(regs, byte_offset=0):
    """Emit buffer_store_dword instructions to dump VGPRs into the output buffer."""
    out = []
    for i, reg in enumerate(regs):
        out.append(f"v_mov_b32_e32 v31, {byte_offset + 4 * i}")
        out.append(f"buffer_store_dword v{reg}, v31, s[0:3], 0 offen")
    return out


def seed_lds_b32(base_addr, values):
    """Emit ds_write_b32 instructions to seed consecutive dwords in LDS."""
    out = []
    for i, val in enumerate(values):
        out.append(f"v_mov_b32_e32 v30, {base_addr + 4 * i:#x}")
        out.append(f"v_mov_b32_e32 v29, {val:#x}")
        out.append("ds_write_b32 v30, v29")
    return out


def read_lds_b32(base_addr, count, dst_base_reg):
    """Emit ds_read_b32 instructions to read consecutive dwords from LDS."""
    out = []
    for i in range(count):
        out.append(f"v_mov_b32_e32 v30, {base_addr + 4 * i:#x}")
        out.append(f"ds_read_b32 v{dst_base_reg + i}, v30")
    return out


class LdsB96B128Semantics(unittest.TestCase):
    def test_roundtrip_b96(self):
        lines = [
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W3),
            "ds_write_b96 v2, v[4:6]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b96 v[10:12], v2",
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12]),
        ]
        mem = run_spv(translate_cs("cs_lds_rt_b96", lines))
        self.assertEqual([mem.get(4 * i) for i in range(3)], list(W3))

    def test_roundtrip_b128(self):
        lines = [
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W4),
            "ds_write_b128 v2, v[4:7]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b128 v[10:13], v2",
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 13]),
        ]
        mem = run_spv(translate_cs("cs_lds_rt_b128", lines))
        self.assertEqual([mem.get(4 * i) for i in range(4)], list(W4))

    def test_offset_imm_and_b32_cross_check(self):
        # ds_write_b96 with vaddr=4, offset:16 -> effective byte address 20.
        lines_w96 = [
            "v_mov_b32_e32 v2, 4",
            *load_vgprs(4, W3),
            "ds_write_b96 v2, v[4:6] offset:16",
            "s_waitcnt lgkmcnt(0)",
            *read_lds_b32(20, 3, 10),
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12]),
        ]
        mem_w96 = run_spv(translate_cs("cs_lds_off_w96", lines_w96))
        self.assertEqual([mem_w96.get(4 * i) for i in range(3)], list(W3))

        # ds_write_b128 with vaddr=8, offset:20 -> effective byte address 28.
        lines_w128 = [
            "v_mov_b32_e32 v2, 8",
            *load_vgprs(4, W4),
            "ds_write_b128 v2, v[4:7] offset:20",
            "s_waitcnt lgkmcnt(0)",
            *read_lds_b32(28, 4, 10),
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 13]),
        ]
        mem_w128 = run_spv(translate_cs("cs_lds_off_w128", lines_w128))
        self.assertEqual([mem_w128.get(4 * i) for i in range(4)], list(W4))

        # Seed with ds_write_b32 and read back with ds_read_b96 / ds_read_b128 + offset.
        lines_r = [
            *seed_lds_b32(20, W3),
            *seed_lds_b32(44, W4),
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 4",
            "ds_read_b96 v[10:12], v2 offset:16",
            "v_mov_b32_e32 v3, 24",
            "ds_read_b128 v[14:17], v3 offset:20",
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 14, 15, 16, 17]),
        ]
        mem_r = run_spv(translate_cs("cs_lds_off_r", lines_r))
        self.assertEqual([mem_r.get(4 * i) for i in range(7)], list(W3) + list(W4))

    def test_adjacent_lds_dwords_preserved(self):
        # ds_write_b96 at offset 0 must not clobber the 4th dword at offset 12 (or offset 16).
        sent_b96 = [0xDEAD0000 + i for i in range(5)]
        lines_96 = [
            *seed_lds_b32(0, sent_b96),
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W3),
            "ds_write_b96 v2, v[4:6]",
            "s_waitcnt lgkmcnt(0)",
            *read_lds_b32(0, 5, 10),
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 13, 14]),
        ]
        mem_96 = run_spv(translate_cs("cs_lds_adj_b96", lines_96))
        self.assertEqual(
            [mem_96.get(4 * i) for i in range(5)],
            list(W3) + sent_b96[3:],
        )

        # ds_write_b128 at offset 0 must not clobber the 5th dword at offset 16 (or offset 20).
        sent_b128 = [0xBEEF0000 + i for i in range(6)]
        lines_128 = [
            *seed_lds_b32(0, sent_b128),
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W4),
            "ds_write_b128 v2, v[4:7]",
            "s_waitcnt lgkmcnt(0)",
            *read_lds_b32(0, 6, 10),
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 13, 14, 15]),
        ]
        mem_128 = run_spv(translate_cs("cs_lds_adj_b128", lines_128))
        self.assertEqual(
            [mem_128.get(4 * i) for i in range(6)],
            list(W4) + sent_b128[4:],
        )

    def test_inactive_lane_preserves_lds_and_vgprs(self):
        lds_sent = [0xCAFE0001 + i for i in range(4)]
        vgpr_sent = [0xFACE0001 + i for i in range(4)]

        for op_w, op_r, count in (
            ("ds_write_b128 v2, v[4:7]", "ds_read_b128 v[20:23], v2", 4),
            ("ds_write_b96 v2, v[4:6]", "ds_read_b96 v[20:22], v2", 3),
        ):
            with self.subTest(op_w=op_w, op_r=op_r):
                lines = [
                    *seed_lds_b32(0, lds_sent),
                    *load_vgprs(20, vgpr_sent),
                    "v_mov_b32_e32 v2, 0",
                    *load_vgprs(4, W4[:count]),
                    "s_waitcnt lgkmcnt(0)",
                    "s_mov_b32 exec_lo, 0",
                    op_w,
                    op_r,
                    "s_mov_b32 exec_lo, 1",
                    "s_waitcnt lgkmcnt(0)",
                    *read_lds_b32(0, 4, 24),
                    "s_waitcnt lgkmcnt(0)",
                    *store_vgprs([20, 21, 22, 23, 24, 25, 26, 27]),
                ]
                mem = run_spv(translate_cs("cs_lds_inactive", lines))
                self.assertEqual([mem.get(4 * i) for i in range(4)], vgpr_sent)
                self.assertEqual([mem.get(4 * i) for i in range(4, 8)], lds_sent)

    def test_negative_controls_spirv_mutations(self):
        # 1. Corrupting the +4/+8 word-offset OpIAdd in ds_write_b128 must break the round-trip.
        spv_rt = translate_cs("cs_lds_neg_off", [
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W4),
            "ds_write_b128 v2, v[4:7]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b128 v[10:13], v2",
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([10, 11, 12, 13]),
        ])
        mutated_off, n_off = re.subn(
            r"(= OpIAdd %uint %\w+ )%c4\b",
            r"\1%c8",
            spv_rt,
            count=1,
        )
        self.assertEqual(n_off, 1)
        mem_off = run_spv(mutated_off)
        self.assertNotEqual([mem_off.get(4 * i) for i in range(4)], list(W4))

        # 2. Removing the %active guard on OpBranchConditional in the inactive-lane shader
        #    must cause LDS or VGPRs to be overwritten when exec_lo = 0.
        lds_sent = [0xCAFE0001 + i for i in range(4)]
        vgpr_sent = [0xFACE0001 + i for i in range(4)]
        spv_inact = translate_cs("cs_lds_neg_act", [
            *seed_lds_b32(0, lds_sent),
            *load_vgprs(20, vgpr_sent),
            "v_mov_b32_e32 v2, 0",
            *load_vgprs(4, W4),
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0",
            "ds_write_b128 v2, v[4:7]",
            "ds_read_b128 v[20:23], v2",
            "s_mov_b32 exec_lo, 1",
            "s_waitcnt lgkmcnt(0)",
            *read_lds_b32(0, 4, 24),
            "s_waitcnt lgkmcnt(0)",
            *store_vgprs([20, 21, 22, 23, 24, 25, 26, 27]),
        ])
        # Flip every OpBranchConditional condition to %true so the ds_write_b128/ds_read_b128
        # blocks execute unconditionally even while exec_lo = 0.
        self.assertIn("%true = OpConstantTrue %bool", spv_inact)
        mutated_act, n_act = re.subn(
            r"OpBranchConditional %\w+ ",
            "OpBranchConditional %true ",
            spv_inact,
        )
        self.assertGreater(n_act, 0)
        mem_act = run_spv(mutated_act)
        self.assertNotEqual(
            [mem_act.get(4 * i) for i in range(8)],
            vgpr_sent + lds_sent,
        )


if __name__ == "__main__":
    unittest.main()
