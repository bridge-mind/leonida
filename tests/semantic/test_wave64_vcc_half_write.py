"""Wave64: a 32-bit write to vcc_lo or vcc_hi changes only that half of the VCC lane mask.

RDNA2 ISA: VCC is one 64-bit register; VCC_LO (SGPR 106) is VCC[31:0] and VCC_HI (SGPR 107) is VCC[63:32], each
a separately addressable 32-bit scalar operand. In wave64, v_cndmask_b32 reads VCC[lane] for lanes 0-63 and VCCZ
(s_cbranch_vccz) is set when all 64 bits are 0. So `s_mov_b32 vcc_lo, 0` must leave lanes 32-63 of VCC as they
were, and `s_mov_b32 vcc_hi, 0` must clear lanes 32-63 and keep lanes 0-31.

Each case builds VCC in a 64-thread compute shader from two push-constant words (bit i of s4 for lanes 0-31, of s5
for lanes 32-63), writes one half with a 0 / -1 constant (a mask value in every representation), then stores what
every lane sees: v_cndmask_b32 on VCC (its own bit) and the s_cbranch_vccz outcome (the wave's). Translated by
tools/rdna2spirv.py and run on all 64 lanes with tests/semantic/spirv_mini_interp.run_workgroup; no llvm-mc, GPU
or game data. The listing carries `; encoding:` bytes (zeros; only their length is read) so the branch target is
placed.
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # s[0:3]: raw buffer (OOB_SELECT 3), any offset in range

# (instruction, size in bytes). VCC lane i = bit (i & 31) of (s4 if i < 32 else s5).
BUILD_VCC = [
    ("v_mov_b32_e32 v11, s5", 4),
    ("v_cmp_lt_u32_e64 vcc, v0, 32", 8),
    ("v_cndmask_b32_e64 v1, v11, s4, vcc", 8),
    ("v_and_b32_e32 v2, 31, v0", 4),
    ("v_lshrrev_b32_e32 v3, v2, v1", 4),
    ("v_and_b32_e32 v3, 1, v3", 4),
    ("v_cmp_ne_u32_e64 vcc, 0, v3", 8),
    ("s_and_b64 s[12:13], vcc, exec", 4),     # also what marks the program wave64 (a 64-bit op on VCC/EXEC)
]
# out[lane] = VCC[lane] ? 5 : 3; out[64 + lane] = 7 if s_cbranch_vccz was taken, else 9.
OBSERVE = [
    ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
    ("v_lshlrev_b32_e32 v5, 2, v0", 4),
    ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
    ("s_mov_b32 s20, 7", 4),
    ("s_cbranch_vccz 1", 4),
    ("s_mov_b32 s20, 9", 4),
    ("v_mov_b32_e32 v6, s20", 4),
    ("buffer_store_dword v6, v5, s[0:3], 0 offen offset:256", 8),
    ("s_endpgm", 4),
]


def f32_u32(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


def translate_asm(name, lines, *, cs=True):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if cs:
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                # COMPUTE_PGM_RSRC2 0x8090: 16 user SGPRs (s[0:3] V#, s4..s15 push constants); 64x1x1 threads.
                f.write(bytes(0x138) + struct.pack("<4I", 0x8020, 64, 1, 1))
        saved_w64g = dict(rdna2spirv.W64G_COUNTS)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.W64G_COUNTS.clear()
            rdna2spirv.W64G_COUNTS.update(saved_w64g)
        return spv


def translate_cs(name, lines):
    return translate_asm(name, lines, cs=True)


def run(spv, lo, hi):
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                            "%sg_lane": i}, push_constants=pc, memory={})
    return [mem[4 * l] for l in range(64)], [mem[256 + 4 * l] for l in range(64)]


def expect(lo, hi):
    vcc = (lo & M32) | (hi & M32) << 32
    return [5 if vcc >> l & 1 else 3 for l in range(64)], [7 if vcc == 0 else 9] * 64


# The half write under test, and the new (lo, hi) halves of VCC as a function of the old ones.
CASES = {
    "mov_vcc_lo_0": ([("s_mov_b32 vcc_lo, 0", 4)], lambda lo, hi: (0, hi)),
    "mov_vcc_hi_0": ([("s_mov_b32 vcc_hi, 0", 4)], lambda lo, hi: (lo, 0)),
    "mov_vcc_lo_all": ([("s_mov_b32 vcc_lo, -1", 4)], lambda lo, hi: (M32, hi)),
    "mov_vcc_hi_all": ([("s_mov_b32 vcc_hi, -1", 4)], lambda lo, hi: (lo, M32)),
    "and_vcc_lo_0": ([("s_and_b32 vcc_lo, vcc_lo, 0", 4)], lambda lo, hi: (0, hi)),
    "or_vcc_hi_all": ([("s_or_b32 vcc_hi, vcc_hi, -1", 4)], lambda lo, hi: (lo, M32)),
    "mov_sgpr_0_vcc_lo": ([("s_mov_b32 s6, 0", 4), ("s_mov_b32 vcc_lo, s6", 4)], lambda lo, hi: (0, hi)),
    "mov_sgpr_all_vcc_hi": ([("s_mov_b32 s6, -1", 4), ("s_mov_b32 vcc_hi, s6", 4)], lambda lo, hi: (lo, M32)),
}
MASKS = [(0x0000FFFF, 0xFFFF0000), (0, 1), (1, 0), (0x80000006, 0x00010000), (0, 0), (M32, M32)]


class Wave64VccHalfWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = {k: translate_cs("cs_w64_vcc_" + k, BUILD_VCC + body + OBSERVE)
                   for k, (body, _) in CASES.items()}
        cls.seq_spv = translate_cs("cs_w64_vcc_seq", BUILD_VCC + [
            ("s_mov_b32 vcc_hi, 0", 4),
            ("s_bcnt1_i32_b64 s6, vcc", 4),
            ("s_mov_b32 vcc_hi, -1", 4),
            ("s_bcnt1_i32_b64 s7, vcc", 4),
            ("s_mov_b32 vcc_lo, 0", 4),
            ("s_bcnt1_i32_b64 s8, vcc", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("v_mov_b32_e32 v6, s6", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen offset:512", 8),
            ("v_mov_b32_e32 v6, s7", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen offset:768", 8),
            ("v_mov_b32_e32 v6, s8", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen offset:1024", 8),
        ] + OBSERVE)

    def test_reference(self):
        self.assertEqual(expect(0, 1)[1][0], 9)
        self.assertEqual(expect(0, 0)[1][0], 7)
        self.assertEqual(expect(0, 1)[0][32], 5)

    def test_half_write_keeps_other_half(self):
        for name, (body, after) in CASES.items():
            for lo, hi in MASKS:
                with self.subTest(case=name, lo=hex(lo), hi=hex(hi)):
                    got = run(self.spv[name], lo, hi)
                    self.assertEqual(got, expect(*after(lo, hi)))

    def test_back_to_back_half_mask_writes_preserve_whole_wave_vcc(self):
        for lo, hi in MASKS:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
                mem = si.run_workgroup(self.seq_spv, 64, 64,
                                       inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                                       push_constants=pc, memory={})
                lo_cnt = (lo & M32).bit_count()
                self.assertEqual([mem[512 + 4 * l] for l in range(64)], [lo_cnt] * 64)
                self.assertEqual([mem[768 + 4 * l] for l in range(64)], [lo_cnt + 32] * 64)
                self.assertEqual([mem[1024 + 4 * l] for l in range(64)], [32] * 64)
                got = ([mem[4 * l] for l in range(64)], [mem[256 + 4 * l] for l in range(64)])
                self.assertEqual(got, expect(0, M32))


class Wave64VccHalfDataWriteTests(unittest.TestCase):
    """Plain-data writes to vcc_lo / vcc_hi after v_cmp in wave64 must write all 64 lanes as scalar data rather than
    updating only one 32-lane mask half (which left 0xFFFFFFFF = -NaN in lanes 32..63 after v_cmp)."""

    LOAD_ADDR = 0x2000

    @classmethod
    def setUpClass(cls):
        cls.ps_spv = translate_asm("ps_w64_vcc_lo_smem_sub", [
            ("s_mov_b64 s[12:13], exec", 4),
            ("v_cmp_ne_u32_e64 vcc, 0, s4", 8),
            ("s_buffer_load_dword vcc_lo, s[0:3], 0x10", 8),
            ("v_mov_b32_e32 v1, s5", 4),
            ("v_sub_f32_e64 v2, vcc_lo, v1", 8),
            ("exp mrt0 v2, v2, v2, v2 done vm", 8),
            ("s_endpgm", 4),
        ], cs=False)

        tail = lambda expr: [
            ("v_cvt_f32_u32_e32 v4, v0", 4),
            (expr, 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        cls.cs_spv = {
            "sbuf_vcc_lo": translate_cs("cs_w64_sbuf_vcc_lo", BUILD_VCC + [
                ("s_buffer_load_dword vcc_lo, s[0:3], 0x2000", 8),
            ] + tail("v_sub_f32_e64 v6, vcc_lo, v4")),
            "sload_vcc_lo": translate_cs("cs_w64_sload_vcc_lo", BUILD_VCC + [
                ("s_load_dword vcc_lo, s[8:9], 0x8", 8),
            ] + tail("v_sub_f32_e64 v6, vcc_lo, v4")),
            "sbuf_vcc_hi": translate_cs("cs_w64_sbuf_vcc_hi", BUILD_VCC + [
                ("s_buffer_load_dword vcc_hi, s[0:3], 0x2004", 8),
            ] + tail("v_add_f32_e64 v6, vcc_hi, v4")),
            "salu_vcc_lo": translate_cs("cs_w64_salu_vcc_lo", BUILD_VCC + [
                ("s_add_u32 vcc_lo, s6, 1", 4),
                ("v_add_nc_u32_e64 v6, vcc_lo, v0", 8),
                ("v_lshlrev_b32_e32 v5, 2, v0", 4),
                ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]),
            "mov_imm_vcc_lo": translate_cs("cs_w64_mov_imm_vcc_lo", BUILD_VCC + [
                ("s_mov_b32 vcc_lo, 0x3f800000", 8),
            ] + tail("v_add_f32_e64 v6, vcc_lo, v4")),
            "mov_sgpr_vcc_lo": translate_cs("cs_w64_mov_sgpr_vcc_lo", BUILD_VCC + [
                ("s_mov_b32 vcc_lo, s6", 4),
            ] + tail("v_sub_f32_e64 v6, vcc_lo, v4")),
            "mov_imm_vcc_hi": translate_cs("cs_w64_mov_imm_vcc_hi", BUILD_VCC + [
                ("s_mov_b32 vcc_hi, 0x40800000", 8),
            ] + tail("v_add_f32_e64 v6, vcc_hi, v4")),
        }

    def run_cs(self, key, lo, hi, *, s6=0, ptr=0, init_mem=None):
        pc = VSHARP + [lo & M32, hi & M32, s6 & M32, 0, ptr & M32, (ptr >> 32) & M32] + [0] * 22
        mem = dict(init_mem or {})
        si.run_workgroup(self.cs_spv[key], 64, 64,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=pc, memory=mem)
        return [mem[4 * l] for l in range(64)]

    def test_ps_s_buffer_load_vcc_lo_sub_f32_all_64_lanes(self):
        vsharp = [0x1000, 0, 0x100, 0x31016FAC]
        pc = vsharp + [1, f32_u32(1.25)] + [0] * 26
        mem = {0x1010: f32_u32(3.5)}
        want = (f32_u32(3.5 - 1.25),) * 4
        for lane in range(64):
            out = si.run_fragment(self.ps_spv, inputs={"%sg_lane": lane}, push_constants=pc, memory=dict(mem))
            self.assertEqual(out["%out_mrt0"], want, f"lane {lane}")

    def test_cs_plain_data_writes_to_vcc_lo_and_vcc_hi_across_64_lanes(self):
        for lo, hi in [(M32, M32), (0xAAAAAAAA, 0x55555555)]:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                got = self.run_cs("sbuf_vcc_lo", lo, hi, init_mem={0x2000: f32_u32(100.5)})
                self.assertEqual(got, [f32_u32(100.5 - float(l)) for l in range(64)])

                got = self.run_cs("sload_vcc_lo", lo, hi, ptr=0x3000, init_mem={0x3008: f32_u32(72.25)})
                self.assertEqual(got, [f32_u32(72.25 - float(l)) for l in range(64)])

                got = self.run_cs("sbuf_vcc_hi", lo, hi, init_mem={0x2004: f32_u32(12.5)})
                self.assertEqual(got, [f32_u32(12.5 + float(l)) for l in range(64)])

                got = self.run_cs("salu_vcc_lo", lo, hi, s6=99)
                self.assertEqual(got, [(100 + l) & M32 for l in range(64)])

                got = self.run_cs("mov_imm_vcc_lo", lo, hi)
                self.assertEqual(got, [f32_u32(1.0 + float(l)) for l in range(64)])

                got = self.run_cs("mov_sgpr_vcc_lo", lo, hi, s6=f32_u32(2.5))
                self.assertEqual(got, [f32_u32(2.5 - float(l)) for l in range(64)])

                got = self.run_cs("mov_imm_vcc_hi", lo, hi)
                self.assertEqual(got, [f32_u32(4.0 + float(l)) for l in range(64)])


class LaneMasksVccHalfTests(unittest.TestCase):
    """Direct lane_masks.analyse checks for vcc_lo / vcc_hi mask vs plain-data writes."""

    @staticmethod
    def masks_after(*lines):
        blk = [parse_line(ln) for ln in (*lines, "s_endpgm")]
        out, _ = lane_masks.analyse([blk], [[]])
        return out[id(blk[-1])]

    def test_mask_and_const_half_writes_keep_vcc_mask(self):
        self.assertEqual(
            self.masks_after(
                "v_cmp_lt_u32_e64 vcc, v0, 32",
                "s_mov_b32 vcc_hi, 0",
                "s_mov_b32 vcc_hi, -1",
                "s_mov_b32 vcc_lo, 0",
            ),
            {"vcc": lane_masks.MASK},
        )
        self.assertEqual(
            self.masks_after(
                "v_cmp_lt_u32_e64 vcc, v0, 32",
                "s_mov_b32 s4, 0",
                "s_mov_b32 vcc_hi, s4",
            ),
            {"vcc": lane_masks.MASK},
        )
        self.assertEqual(
            self.masks_after(
                "s_mov_b32 vcc_lo, exec_lo",
                "s_mov_b32 vcc_hi, exec_hi",
            ),
            {"vcc": lane_masks.MASK},
        )

    def test_plain_data_half_writes_clear_vcc_mask(self):
        for write_line in (
            "s_buffer_load_dword vcc_lo, s[0:3], 0x0",
            "s_buffer_load_dword vcc_hi, s[0:3], 0x0",
            "s_load_dword vcc_lo, s[0:1], 0x0",
            "s_add_u32 vcc_lo, s4, 1",
            "s_mov_b32 vcc_lo, 0x3f800000",
            "s_mov_b32 vcc_hi, 0x3f800000",
            "s_mov_b32 vcc_lo, s4",
        ):
            with self.subTest(write=write_line):
                self.assertEqual(self.masks_after("v_cmp_lt_u32_e64 vcc, v0, 32", write_line), {})


if __name__ == "__main__":
    unittest.main()
