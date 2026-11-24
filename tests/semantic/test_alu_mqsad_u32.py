"""Semantic interpreter tests for RDNA2 v_mqsad_u32_u8 and v_mqsad_pk_u16_u8.

Four masked byte-SADs with full 32-bit accumulators (RDNA ISA pseudocode). S0 is 64 bits, S1 32 bits, S2 and D are
128 bits (four VGPRs):
  D[k] = (S2[k] + sum over the 4 bytes of |W_k.byte - S1.byte|, skipping bytes where S1's byte is 0) mod 2^32,
  W_k = S0 bits [8k+31 : 8k]
so the four windows slide over S0 one byte at a time, as for v_qsad_pk_u16_u8, but each result is a whole dword.
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares D with a plain-Python reference. D leaves through exp as eight 16-bit halves ORed into 0x40000000, so no
export channel is a NaN or denormal pattern that a float round trip could change.
"""
from __future__ import annotations

import os
import random
import struct
import sys
import tempfile
from typing import Sequence
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M16 = 0xFFFF
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_byte_window(s0: int, k: int) -> int:
    """Extract 32-bit window W_k at byte offset k from 64-bit s0: bits [8*k+31 : 8*k]."""
    return (s0 >> (8 * k)) & M32


def ref_quad_byte_sad(win: int, s1: int, masked: bool = True) -> int:
    """Compute sum of absolute differences across 4 bytes between win and s1."""
    total = 0
    for i in range(4):
        b_win = (win >> (8 * i)) & 0xFF
        b_s1 = (s1 >> (8 * i)) & 0xFF
        if masked and b_s1 == 0:
            continue
        total += abs(b_win - b_s1)
    return total


def ref_mqsad_u32_u8(s0: int, s1: int, accs: Sequence[int]) -> list[int]:
    """Reference implementation of RDNA2 v_mqsad_u32_u8."""
    out = []
    for k in range(4):
        win = ref_byte_window(s0, k)
        sad = ref_quad_byte_sad(win, s1, masked=True)
        out.append(((accs[k] & M32) + sad) & M32)
    return out


def ref_mqsad_pk_u16_u8(s0: int, s1: int, acc64: int) -> tuple[int, int]:
    """Reference implementation of RDNA2 v_mqsad_pk_u16_u8 returning two 32-bit dwords."""
    words = []
    for k in range(4):
        win = ref_byte_window(s0, k)
        sad = ref_quad_byte_sad(win, s1, masked=True)
        acc_k = (acc64 >> (16 * k)) & M16
        words.append((acc_k + sad) & M16)
    d0 = (words[0] & M16) | ((words[1] & M16) << 16)
    d1 = (words[2] & M16) | ((words[3] & M16) << 16)
    return d0, d1


def ref(x: int, y: int, accs: Sequence[int]) -> list[int]:
    """Compatibility alias for ref_mqsad_u32_u8."""
    return ref_mqsad_u32_u8(x, y, accs)


X64 = [0, M64, 0x0807060504030201, 0xFF00FF00FF00FF00, 0x0123456789ABCDEF]
Y32 = [0, 0xFFFFFFFF, 0x04030201, 0x00FF00FF, 0x12000034]
ACCS = [[0, 0, 0, 0], [M32, M32 - 1, 0x7FFFFFFF, 1], [1, 2, 3, 4]]

PK_ACCS_64 = [
    0,
    M64,
    0x0001000200030004,
    0xFFFF0001FFFF0002,
    0x123456789ABCDEF0,
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mqsad_u32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mqsad_u32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> list:
    """Run the lines, return v[8:11] (exported as eight 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = []
    for k in range(4):
        lo, hi = 20 + 2 * k, 21 + 2 * k
        tail += [f"v_and_b32 v{lo}, 0xffff, v{8 + k}", f"v_lshrrev_b32 v{hi}, 16, v{8 + k}",
                 f"v_or_b32 v{lo}, 0x40000000, v{lo}", f"v_or_b32 v{hi}, 0x40000000, v{hi}"]
    asm = ("; test_mqsad_u32\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v20, v21, v22, v23 vm\n\texp mrt1 v24, v25, v26, v27 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    h = list(out["%out_mrt0"]) + list(out["%out_mrt1"])
    assert all(v >> 16 == 0x4000 for v in h), [hex(v) for v in h]
    return [(h[2 * k] & 0xFFFF) | (h[2 * k + 1] & 0xFFFF) << 16 for k in range(4)]


def run_lines_pk_u16(lines) -> tuple[int, int]:
    """Run lines computing v[8:9] with v_mqsad_pk_u16_u8 and export via mrt0."""
    if _TMP_DIR is None:
        setUpModule()
    tail = [
        "v_and_b32 v20, 0xffff, v8",
        "v_lshrrev_b32 v21, 16, v8",
        "v_and_b32 v22, 0xffff, v9",
        "v_lshrrev_b32 v23, 16, v9",
        "v_or_b32 v20, 0x40000000, v20",
        "v_or_b32 v21, 0x40000000, v21",
        "v_or_b32 v22, 0x40000000, v22",
        "v_or_b32 v23, 0x40000000, v23",
    ]
    asm = ("; test_mqsad_pk_u16\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v20, v21, v22, v23 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    h = list(out["%out_mrt0"])
    assert all(v >> 16 == 0x4000 for v in h), [hex(v) for v in h]
    d0 = (h[0] & M16) | ((h[1] & M16) << 16)
    d1 = (h[2] & M16) | ((h[3] & M16) << 16)
    return d0, d1


def run_op(x: int, y: int, accs: Sequence[int]) -> list[int]:
    return run_lines([f"v_mov_b32 v2, {x & M32:#x}", f"v_mov_b32 v3, {x >> 32:#x}", f"v_mov_b32 v1, {y:#x}"]
                     + [f"v_mov_b32 v{4 + k}, {accs[k]:#x}" for k in range(4)]
                     + ["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7]"])


def run_op_pk_u16(x: int, y: int, acc64: int) -> tuple[int, int]:
    return run_lines_pk_u16([
        f"v_mov_b32 v2, {x & M32:#x}",
        f"v_mov_b32 v3, {x >> 32:#x}",
        f"v_mov_b32 v1, {y:#x}",
        f"v_mov_b32 v4, {acc64 & M32:#x}",
        f"v_mov_b32 v5, {(acc64 >> 32) & M32:#x}",
        "v_mqsad_pk_u16_u8 v[8:9], v[2:3], v1, v[4:5]",
    ])


class SimdALUInterpreter:
    """Software interpreter modeling SIMD lane execution under dynamic EXEC mask."""

    def __init__(self, wave_size: int = 64, num_vgprs: int = 64, num_sgprs: int = 106):
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.num_sgprs = num_sgprs
        self.vgprs = [[0] * wave_size for _ in range(num_vgprs)]
        self.sgprs = [0] * num_sgprs
        self.exec_mask = (1 << wave_size) - 1

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def set_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & M32

    def get_vgpr(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & M32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & M32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & M32

    def execute_v_mqsad_u32_u8(self, dst_lo: int, src0_lo: int, src1_reg: int, src2_lo: int,
                               s0_is_sgpr: bool = False) -> None:
        for lane in range(self.wave_size):
            if not ((self.exec_mask >> lane) & 1):
                continue
            if s0_is_sgpr:
                s0 = (self.sgprs[src0_lo] & M32) | ((self.sgprs[src0_lo + 1] & M32) << 32)
            else:
                s0 = (self.vgprs[src0_lo][lane] & M32) | ((self.vgprs[src0_lo + 1][lane] & M32) << 32)
            s1 = self.vgprs[src1_reg][lane] & M32
            accs = [self.vgprs[src2_lo + k][lane] & M32 for k in range(4)]
            res = ref_mqsad_u32_u8(s0, s1, accs)
            for k in range(4):
                self.vgprs[dst_lo + k][lane] = res[k]

    def execute_v_mqsad_pk_u16_u8(self, dst_lo: int, src0_lo: int, src1_reg: int, src2_lo: int,
                                  s0_is_sgpr: bool = False) -> None:
        for lane in range(self.wave_size):
            if not ((self.exec_mask >> lane) & 1):
                continue
            if s0_is_sgpr:
                s0 = (self.sgprs[src0_lo] & M32) | ((self.sgprs[src0_lo + 1] & M32) << 32)
            else:
                s0 = (self.vgprs[src0_lo][lane] & M32) | ((self.vgprs[src0_lo + 1][lane] & M32) << 32)
            s1 = self.vgprs[src1_reg][lane] & M32
            acc64 = (self.vgprs[src2_lo][lane] & M32) | ((self.vgprs[src2_lo + 1][lane] & M32) << 32)
            d0, d1 = ref_mqsad_pk_u16_u8(s0, s1, acc64)
            self.vgprs[dst_lo][lane] = d0
            self.vgprs[dst_lo + 1][lane] = d1


class MqsadU32OracleTests(unittest.TestCase):
    """Unit tests for reference oracle mathematical properties."""

    def test_window_sliding_offsets(self):
        s0 = 0x0807060504030201
        self.assertEqual(ref_byte_window(s0, 0), 0x04030201)
        self.assertEqual(ref_byte_window(s0, 1), 0x05040302)
        self.assertEqual(ref_byte_window(s0, 2), 0x06050403)
        self.assertEqual(ref_byte_window(s0, 3), 0x07060504)

    def test_masked_vs_unmasked_quad_sad(self):
        win = 0x10203040
        s1_zeros = 0x00000000
        self.assertEqual(ref_quad_byte_sad(win, s1_zeros, masked=True), 0)
        self.assertEqual(ref_quad_byte_sad(win, s1_zeros, masked=False), 0x10 + 0x20 + 0x30 + 0x40)

        s1_partial = 0x00200040
        self.assertEqual(ref_quad_byte_sad(win, s1_partial, masked=True), 0)
        self.assertEqual(ref_quad_byte_sad(win, s1_partial, masked=False), 0x10 + 0x30)

    def test_accumulator_32bit_wraparound(self):
        res = ref_mqsad_u32_u8(0x0000000000000000, 0xFFFFFFFF, [M32, M32 - 1, 0x80000000, 1])
        self.assertEqual(res[0], (M32 + 1020) & M32)
        self.assertEqual(res[1], (M32 - 1 + 1020) & M32)
        self.assertEqual(res[2], (0x80000000 + 1020) & M32)
        self.assertEqual(res[3], 1021)

    def test_accumulator_16bit_packed_wraparound(self):
        d0, d1 = ref_mqsad_pk_u16_u8(0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
        expected_u16 = (M16 + 1020) & M16
        self.assertEqual(d0 & M16, expected_u16)
        self.assertEqual((d0 >> 16) & M16, expected_u16)
        self.assertEqual(d1 & M16, expected_u16)
        self.assertEqual((d1 >> 16) & M16, expected_u16)


class MqsadU32Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(0x0807060504030201, 0x04030201, [0] * 4), [0, 4, 8, 12])
        self.assertEqual(ref(M64, 0, [5, 6, 7, 8]), [5, 6, 7, 8])
        self.assertEqual(ref(0, 0xFFFFFFFF, [M32] * 4), [1019] * 4)

    def test_edge_values(self):
        for x in X64:
            for y in Y32:
                accs = ACCS[(x ^ y) % len(ACCS)]
                with self.subTest(x=hex(x), y=hex(y), accs=[hex(a) for a in accs]):
                    self.assertEqual(run_op(x, y, accs), ref(x, y, accs))

    def test_seeded(self):
        rng = random.Random(0x3AD)
        for _ in range(25):
            x, y = rng.getrandbits(64), rng.getrandbits(32)
            if rng.random() < 0.5:
                y &= rng.choice([0xFF00FF00, 0x00FFFF00, 0xFFFFFF00])
            accs = [rng.getrandbits(32) for _ in range(4)]
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y, accs), ref(x, y, accs))

    def test_null_pixel_source(self) -> None:
        y, accs = 0x04000201, [0, 10, 20, M32]
        lines = [f"v_mov_b32 v1, {y:#x}"] + [f"v_mov_b32 v{4 + k}, {v:#x}" for k, v in enumerate(accs)]
        got = run_lines(lines + ["v_mqsad_u32_u8 v[8:11], null, v1, v[4:7]"])
        self.assertEqual(got, ref(0, y, accs))

    def test_sgpr_source_pair(self):
        got = run_lines(["s_mov_b32 s2, 0x04030201", "s_mov_b32 s3, 0x08070605", "v_mov_b32 v1, 0x04030201"]
                        + [f"v_mov_b32 v{4 + k}, 0" for k in range(4)]
                        + ["v_mqsad_u32_u8 v[8:11], s[2:3], v1, v[4:7]"])
        self.assertEqual(got, [0, 4, 8, 12])

    def test_exec_mask_inactive_lane_preserved(self):
        got = run_lines([
            "v_mov_b32 v8, 0x11111111",
            "v_mov_b32 v9, 0x22222222",
            "v_mov_b32 v10, 0x33333333",
            "v_mov_b32 v11, 0x44444444",
            "v_mov_b32 v2, 0x04030201",
            "v_mov_b32 v3, 0x08070605",
            "v_mov_b32 v1, 0x04030201",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0",
            "v_mov_b32 v6, 0",
            "v_mov_b32 v7, 0",
            "s_mov_b32 exec_lo, 0",
            "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7]",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, [0x11111111, 0x22222222, 0x33333333, 0x44444444])

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7] clamp"])


class MqsadPkU16Tests(unittest.TestCase):
    """End-to-end and reference tests for 64-bit multi-dword v_mqsad_pk_u16_u8."""

    def test_reference_known_halves(self):
        d0, d1 = run_op_pk_u16(0x0807060504030201, 0x04030201, 0)
        self.assertEqual((d0, d1), (0x00040000, 0x000c0008))

    def test_edge_values_pk_u16(self):
        for x in X64:
            for y in Y32:
                acc64 = PK_ACCS_64[(x ^ y) % len(PK_ACCS_64)]
                with self.subTest(x=hex(x), y=hex(y), acc=hex(acc64)):
                    self.assertEqual(run_op_pk_u16(x, y, acc64), ref_mqsad_pk_u16_u8(x, y, acc64))

    def test_seeded_pk_u16(self):
        rng = random.Random(0x3AD16)
        for _ in range(20):
            x = rng.getrandbits(64)
            y = rng.getrandbits(32)
            if rng.random() < 0.5:
                y &= rng.choice([0xFF00FF00, 0x00FFFF00, 0xFFFFFF00])
            acc64 = rng.getrandbits(64)
            with self.subTest(x=hex(x), y=hex(y), acc=hex(acc64)):
                self.assertEqual(run_op_pk_u16(x, y, acc64), ref_mqsad_pk_u16_u8(x, y, acc64))

    def test_sgpr_source_pair_pk_u16(self):
        got = run_lines_pk_u16([
            "s_mov_b32 s2, 0x04030201",
            "s_mov_b32 s3, 0x08070605",
            "v_mov_b32 v1, 0x04030201",
            "s_mov_b32 s4, 0x00010002",
            "s_mov_b32 s5, 0x00030004",
            "v_mqsad_pk_u16_u8 v[8:9], s[2:3], v1, s[4:5]",
        ])
        expected = ref_mqsad_pk_u16_u8(0x0807060504030201, 0x04030201, 0x0003000400010002)
        self.assertEqual(got, expected)

    def test_exec_mask_inactive_lane_preserved_pk_u16(self):
        got = run_lines_pk_u16([
            "v_mov_b32 v8, 0xDEADBEEF",
            "v_mov_b32 v9, 0xCAFEBABE",
            "v_mov_b32 v2, 0x04030201",
            "v_mov_b32 v3, 0x08070605",
            "v_mov_b32 v1, 0x04030201",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0",
            "s_mov_b32 exec_lo, 0",
            "v_mqsad_pk_u16_u8 v[8:9], v[2:3], v1, v[4:5]",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, (0xDEADBEEF, 0xCAFEBABE))


class MqsadSimdDivergenceTests(unittest.TestCase):
    """Multi-lane SIMD divergence and execution mask tests."""

    def test_divergence_alternating_lanes_u32(self):
        interp = SimdALUInterpreter(wave_size=64)
        for lane in range(64):
            for k in range(4):
                interp.set_vgpr(8 + k, lane, 0x11111111 * (k + 1))
            interp.set_vgpr(2, lane, 0x04030201)
            interp.set_vgpr(3, lane, 0x08070605)
            interp.set_vgpr(1, lane, 0x04030201)
            for k in range(4):
                interp.set_vgpr(4 + k, lane, 0)

        interp.set_exec(0x5555555555555555)
        interp.execute_v_mqsad_u32_u8(8, 2, 1, 4)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(
                    [interp.get_vgpr(8 + k, lane) for k in range(4)],
                    [0, 4, 8, 12],
                )
            else:
                self.assertEqual(
                    [interp.get_vgpr(8 + k, lane) for k in range(4)],
                    [0x11111111, 0x22222222, 0x33333333, 0x44444444],
                )

    def test_divergence_half_wave_pk_u16(self):
        interp = SimdALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(8, lane, 0xAAAAAAAA)
            interp.set_vgpr(9, lane, 0x55555555)
            interp.set_vgpr(2, lane, 0x04030201)
            interp.set_vgpr(3, lane, 0x08070605)
            interp.set_vgpr(1, lane, 0x04030201)
            interp.set_vgpr(4, lane, 0)
            interp.set_vgpr(5, lane, 0)

        interp.set_exec(0x00000000FFFFFFFF)
        interp.execute_v_mqsad_pk_u16_u8(8, 2, 1, 4)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(interp.get_vgpr(8, lane), 0x00040000)
                self.assertEqual(interp.get_vgpr(9, lane), 0x000c0008)
            else:
                self.assertEqual(interp.get_vgpr(8, lane), 0xAAAAAAAA)
                self.assertEqual(interp.get_vgpr(9, lane), 0x55555555)

    def test_data_dependent_motion_search_divergence(self):
        interp = SimdALUInterpreter(wave_size=32)
        target = 0x10203040
        for lane in range(32):
            s0 = ((lane * 3) & 0xFF) | (((lane * 7) & 0xFF) << 32)
            interp.set_vgpr(2, lane, s0 & M32)
            interp.set_vgpr(3, lane, (s0 >> 32) & M32)
            interp.set_vgpr(1, lane, target)
            val16 = (100 * lane) & M16
            for k in range(4):
                interp.set_vgpr(4 + k, lane, val16)
            interp.set_vgpr(14, lane, val16 | (val16 << 16))
            interp.set_vgpr(15, lane, val16 | (val16 << 16))

        interp.set_exec(0xFFFFFFFF)
        interp.execute_v_mqsad_u32_u8(8, 2, 1, 4)
        interp.execute_v_mqsad_pk_u16_u8(18, 2, 1, 14)

        for lane in range(32):
            u32_res = [interp.get_vgpr(8 + k, lane) for k in range(4)]
            d0 = interp.get_vgpr(18, lane)
            d1 = interp.get_vgpr(19, lane)
            pk_res = [d0 & M16, (d0 >> 16) & M16, d1 & M16, (d1 >> 16) & M16]
            for k in range(4):
                self.assertEqual(u32_res[k] & M16, pk_res[k])


class MqsadBoundaryMatricesTests(unittest.TestCase):
    """Boundary conditions, extreme byte patterns, and accumulator wrap matrices."""

    def test_extremes_all_zeros_and_all_ones(self):
        self.assertEqual(run_op(0, 0, [0, 0, 0, 0]), [0, 0, 0, 0])
        self.assertEqual(run_op(M64, 0xFFFFFFFF, [0, 0, 0, 0]), [0, 0, 0, 0])
        self.assertEqual(run_op(0, 0xFFFFFFFF, [0, 0, 0, 0]), [1020, 1020, 1020, 1020])
        self.assertEqual(run_op(M64, 0x01010101, [0, 0, 0, 0]), [1016, 1016, 1016, 1016])

    def test_sign_bit_boundary_differences(self):
        s0_msb = 0x8080808080808080
        s1_pre = 0x7F7F7F7F
        self.assertEqual(run_op(s0_msb, s1_pre, [0, 0, 0, 0]), [4, 4, 4, 4])
        self.assertEqual(run_op_pk_u16(s0_msb, s1_pre, 0), (0x00040004, 0x00040004))

    def test_sparse_mask_permutations(self):
        s0 = 0x1122334455667788
        single_byte_masks = [0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000]
        for mask in single_byte_masks:
            with self.subTest(mask=hex(mask)):
                self.assertEqual(run_op(s0, mask, [0, 0, 0, 0]), ref_mqsad_u32_u8(s0, mask, [0, 0, 0, 0]))

        multi_byte_masks = [0x00FF00FF, 0xFF00FF00, 0x00FFFFFF, 0xFFFFFF00]
        for mask in multi_byte_masks:
            with self.subTest(mask=hex(mask)):
                self.assertEqual(run_op(s0, mask, [0, 0, 0, 0]), ref_mqsad_u32_u8(s0, mask, [0, 0, 0, 0]))

    def test_accumulator_boundary_transitions(self):
        accs_wrapping = [M32, M32 - 1019, 0x7FFFFFFF, 0x80000000]
        res = run_op(0, 0xFFFFFFFF, accs_wrapping)
        self.assertEqual(res[0], 1019)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0x7FFFFFFF + 1020)
        self.assertEqual(res[3], 0x80000000 + 1020)


class MqsadByteShiftingExtremesTests(unittest.TestCase):
    """Selective matching, dword boundary crossings, and unused byte invariance."""

    def test_sliding_window_offsets_selective_match(self):
        target = 0x2468ACE0
        for target_offset in range(4):
            s0 = (target << (8 * target_offset)) & M64
            res = run_op(s0, target, [0, 0, 0, 0])
            self.assertEqual(res[target_offset], 0)
            for other in range(4):
                if other != target_offset:
                    self.assertGreater(res[other], 0)

    def test_dword_boundary_crossing_windows(self):
        s0 = 0x0807060504030201
        res = run_op(s0, 0x06050403, [0, 0, 0, 0])
        self.assertEqual(res[0], 2 + 2 + 2 + 2)
        self.assertEqual(res[1], 1 + 1 + 1 + 1)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 1 + 1 + 1 + 1)

    def test_unused_byte_7_invariance(self):
        base_s0 = 0x0007060504030201
        mutated_s0 = 0xFF07060504030201
        target = 0x04030201
        res_base_u32 = run_op(base_s0, target, [10, 20, 30, 40])
        res_mut_u32 = run_op(mutated_s0, target, [10, 20, 30, 40])
        self.assertEqual(res_base_u32, res_mut_u32)

        res_base_pk = run_op_pk_u16(base_s0, target, 0x0004000300020001)
        res_mut_pk = run_op_pk_u16(mutated_s0, target, 0x0004000300020001)
        self.assertEqual(res_base_pk, res_mut_pk)


class MqsadNegativeControlsTests(unittest.TestCase):
    """Negative controls, operand validations, and semantic sensitivity."""

    def test_clamp_modifier_rejection_all_forms(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7] clamp"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines_pk_u16(["v_mqsad_pk_u16_u8 v[8:9], v[2:3], v1, v[4:5] clamp"])

    def test_invalid_register_groupings_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v[8:9], v[2:3], v1, v[4:7]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v8, v[2:3], v1, v[4:7]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:5]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines_pk_u16(["v_mqsad_pk_u16_u8 v[8:11], v[2:3], v1, v[4:5]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines_pk_u16(["v_mqsad_pk_u16_u8 v8, v[2:3], v1, v[4:5]"])
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines_pk_u16(["v_mqsad_pk_u16_u8 v[8:9], v[2:3], v1, v[4:7]"])

    def test_semantic_sensitivity_controls(self):
        base_x = 0x0807060504030201
        base_y = 0x04030201
        base_acc = [0, 0, 0, 0]
        base_res = run_op(base_x, base_y, base_acc)

        mutated_y = base_y ^ 0x00000001
        mutated_res_y = run_op(base_x, mutated_y, base_acc)
        self.assertNotEqual(base_res, mutated_res_y)

        mutated_x = base_x ^ 0x0000000000000100
        mutated_res_x = run_op(mutated_x, base_y, base_acc)
        self.assertNotEqual(base_res, mutated_res_x)


if __name__ == "__main__":
    unittest.main()
