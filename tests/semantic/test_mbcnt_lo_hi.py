"""Semantic interpreter tests for AMD RDNA masked bit count instructions:
  - v_mbcnt_lo_u32_b32: D.u32 = BitCount(S0.u32 & ((1 << lane) - 1)) + S1.u32
  - v_mbcnt_hi_u32_b32:
      D.u32 = BitCount(S0.u32 & ((1 << (lane - 32)) - 1)) + S1.u32

Validates thread rank in wave (Wave32 and Wave64), stream compaction,
active-lane prefix scanning, accumulator addition wrapping, boundary
conditions at lane 0 / 31 / 32 / 63, and SPIR-V translation / execution.
Standard library only; zero game data.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
import tempfile
import unittest
import warnings

# Polyfill int.bit_count for Python 3.9 so spirv_mini_interp can run OpBitCount
if not hasattr(int, "bit_count"):
    try:
        _mp = int.__dict__
        _dict_addr = ctypes.c_void_p.from_address(id(_mp) + 16).value
        ctypes.pythonapi.PyDict_SetItemString(
            ctypes.c_void_p(_dict_addr),
            b"bit_count",
            ctypes.py_object(lambda self: bin(self).count("1")),
        )
        ctypes.pythonapi.PyType_Modified(ctypes.py_object(int))
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


# -----------------------------------------------------------------------------
# 1. Pure Reference Semantics (AMD RDNA2 ISA: Masked Bit Count)
# -----------------------------------------------------------------------------

def popcount(x: int) -> int:
    """Deterministic population count for unsigned integers."""
    return bin(x).count("1")


def ref_mbcnt_lo(lane: int, mask_lo: int, acc: int = 0) -> int:
    """Reference oracle for v_mbcnt_lo_u32_b32.

    Counts set bits in mask_lo for lanes strictly below this lane (0..31).
    For lane < 32: active bits are mask_lo & ((1 << lane) - 1).
    For lane >= 32: all 32 bits of the low half are below this lane, so active
    bits are mask_lo & 0xFFFFFFFF.
    The bit count is added to acc with 32-bit unsigned wrapping.
    """
    lane &= 0x3F  # bounded to 6-bit wave lane (0..63)
    mask_lo &= MASK32
    acc &= MASK32
    if lane < 32:
        bitmask = (1 << lane) - 1
    else:
        bitmask = MASK32
    cnt = popcount(mask_lo & bitmask)
    return (cnt + acc) & MASK32


def ref_mbcnt_hi(lane: int, mask_hi: int, acc: int = 0) -> int:
    """Reference oracle for v_mbcnt_hi_u32_b32.

    Counts set bits in mask_hi for high-half lanes strictly below (32..63).
    For lane < 32: there are no high-half lanes below this lane (count 0).
    For lane >= 32: rel = lane - 32; active bits are:
        mask_hi & ((1 << rel) - 1).
    The bit count is added to acc with 32-bit unsigned wrapping.
    """
    lane &= 0x3F  # bounded to 6-bit wave lane (0..63)
    mask_hi &= MASK32
    acc &= MASK32
    if lane < 32:
        bitmask = 0
    else:
        rel = lane - 32
        bitmask = (1 << rel) - 1 if rel < 32 else MASK32
    cnt = popcount(mask_hi & bitmask)
    return (cnt + acc) & MASK32


def ref_thread_rank_wave32(
    lane: int, mask_lo: int = MASK32, base: int = 0
) -> int:
    """Computes thread rank in a 32-lane wave using v_mbcnt_lo."""
    return ref_mbcnt_lo(lane, mask_lo, base)


def ref_thread_rank_wave64(
    lane: int,
    mask_lo: int = MASK32,
    mask_hi: int = MASK32,
    base: int = 0,
) -> int:
    """Computes thread rank in a 64-lane wave using paired lo/hi v_mbcnt."""
    lo_rank = ref_mbcnt_lo(lane, mask_lo, base)
    return ref_mbcnt_hi(lane, mask_hi, lo_rank)


def ref_active_compaction(
    active_lanes: set[int] | list[int], wave_size: int = 64
) -> list[int]:
    """Computes compacted rank for all lanes under a given active set."""
    active_set = set(active_lanes)
    mask64 = 0
    for lane_idx in active_set:
        if 0 <= lane_idx < wave_size:
            mask64 |= 1 << lane_idx
    mask_lo = mask64 & MASK32
    mask_hi = (mask64 >> 32) & MASK32

    ranks = []
    for lane in range(wave_size):
        if wave_size == 32:
            ranks.append(ref_thread_rank_wave32(lane, mask_lo))
        else:
            ranks.append(ref_thread_rank_wave64(lane, mask_lo, mask_hi))
    return ranks


# -----------------------------------------------------------------------------
# 2. Multi-Lane SIMD Wave Interpreter
# -----------------------------------------------------------------------------

class SimdWave:
    """Simulates an AMD RDNA SIMD wave executing vector mbcnt sequences."""

    def __init__(self, wave_size: int = 64, exec_mask: int = MASK64):
        if wave_size not in (32, 64):
            raise ValueError("wave_size must be 32 or 64")
        self.wave_size = wave_size
        self.exec_mask = exec_mask & ((1 << wave_size) - 1)
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 128

    def run_v_mbcnt_lo(
        self, dst: int, src0_mask: int, src1_acc: int | list[int]
    ):
        """Executes v_mbcnt_lo_u32_b32 across all active lanes."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                acc = (
                    src1_acc[lane]
                    if isinstance(src1_acc, list)
                    else src1_acc
                )
                self.vgprs[lane][dst] = ref_mbcnt_lo(lane, src0_mask, acc)

    def run_v_mbcnt_hi(
        self, dst: int, src0_mask: int, src1_acc: int | list[int]
    ):
        """Executes v_mbcnt_hi_u32_b32 across all active lanes."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                acc = (
                    src1_acc[lane]
                    if isinstance(src1_acc, list)
                    else src1_acc
                )
                self.vgprs[lane][dst] = ref_mbcnt_hi(lane, src0_mask, acc)

    def compute_thread_ranks(
        self, dst: int = 0, mask_lo: int = MASK32, mask_hi: int = MASK32
    ):
        """Executes the standard hardware thread-rank sequence into VGPR."""
        self.run_v_mbcnt_lo(dst, mask_lo, 0)
        if self.wave_size == 64:
            acc_list = [self.vgprs[ln][dst] for ln in range(self.wave_size)]
            self.run_v_mbcnt_hi(dst, mask_hi, acc_list)


# -----------------------------------------------------------------------------
# 3. Unit Tests: Pure Reference Semantics and Edge Cases
# -----------------------------------------------------------------------------

class TestMbcntReferenceSemantics(unittest.TestCase):
    """Exhaustive validation of reference mathematical semantics."""

    def test_wave32_all_lanes_identity(self):
        """When mask is -1, lanes 0..31 compute their own lane index."""
        for lane in range(32):
            with self.subTest(lane=lane):
                rank = ref_thread_rank_wave32(lane, MASK32, 0)
                self.assertEqual(rank, lane)

    def test_wave64_all_lanes_identity(self):
        """When mask is -1, lanes 0..63 compute their own lane index."""
        for lane in range(64):
            with self.subTest(lane=lane):
                rank = ref_thread_rank_wave64(lane, MASK32, MASK32, 0)
                self.assertEqual(rank, lane)

    def test_lane_zero_always_zero_count(self):
        """Lane 0 has no lower lanes; count is 0 regardless of mask."""
        masks = [0, 1, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x80000000]
        for m in masks:
            with self.subTest(mask=hex(m)):
                self.assertEqual(ref_mbcnt_lo(0, m, 0), 0)
                self.assertEqual(ref_mbcnt_hi(0, m, 0), 0)
                self.assertEqual(ref_mbcnt_lo(0, m, 42), 42)
                self.assertEqual(ref_mbcnt_hi(0, m, 42), 42)

    def test_cross_half_boundary_lane31_and_32(self):
        """Tests boundary semantics between halves (lanes 31 and 32)."""
        # Lane 31 in Wave64 with full mask
        lo_31 = ref_mbcnt_lo(31, MASK32, 0)
        hi_31 = ref_mbcnt_hi(31, MASK32, lo_31)
        self.assertEqual(lo_31, 31)
        self.assertEqual(hi_31, 31)

        # Lane 32: lo counts all 32 low bits; hi has rel=0, so hi adds 0
        lo_32 = ref_mbcnt_lo(32, MASK32, 0)
        hi_32 = ref_mbcnt_hi(32, MASK32, lo_32)
        self.assertEqual(lo_32, 32)
        self.assertEqual(hi_32, 32)

    def test_lane63_top_boundary(self):
        """Lane 63: lo counts 32 bits, hi counts 31 bits; total is 63."""
        lo_63 = ref_mbcnt_lo(63, MASK32, 0)
        hi_63 = ref_mbcnt_hi(63, MASK32, lo_63)
        self.assertEqual(lo_63, 32)
        self.assertEqual(hi_63, 63)

    def test_accumulator_addition_and_overflow_wrap(self):
        """Accumulator addition wraps modulo 2^32."""
        self.assertEqual(ref_mbcnt_lo(5, MASK32, 100), 105)
        # (1 + 0xFFFFFFFF) wraps to 0
        self.assertEqual(ref_mbcnt_lo(1, MASK32, MASK32), 0)
        self.assertEqual(ref_mbcnt_hi(33, MASK32, MASK32), 0)
        self.assertEqual(ref_mbcnt_lo(10, MASK32, 0xFFFFFFF8), 2)

    def test_zero_mask_returns_accumulator(self):
        """Zero mask has no set bits; result is strictly equal to acc."""
        for lane in range(64):
            self.assertEqual(ref_mbcnt_lo(lane, 0, 77), 77)
            self.assertEqual(ref_mbcnt_hi(lane, 0, 88), 88)

    def test_single_active_bit_boundary(self):
        """A mask with a single bit set at position k."""
        for k in (0, 1, 15, 30, 31):
            mask = 1 << k
            for lane in range(32):
                expected = 1 if lane > k else 0
                got = ref_mbcnt_lo(lane, mask, 0)
                self.assertEqual(
                    got, expected, f"Failed at k={k}, lane={lane}"
                )

    def test_sparse_active_compaction(self):
        """Stream compaction matches the index within active lanes."""
        active = [3, 8, 19, 32, 45, 60]
        ranks = ref_active_compaction(active, wave_size=64)
        for expected_rank, lane in enumerate(active):
            self.assertEqual(ranks[lane], expected_rank)

        # Inactive lanes strictly maintain prefix sum count
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[4], 1)
        self.assertEqual(ranks[20], 3)
        self.assertEqual(ranks[33], 4)
        self.assertEqual(ranks[63], 6)

    def test_even_and_odd_alternating_masks(self):
        """Tests even (0x55555555) and odd (0xAAAAAAAA) alternating masks."""
        even_mask = 0x55555555
        odd_mask = 0xAAAAAAAA

        for t in range(32):
            self.assertEqual(ref_mbcnt_lo(t, even_mask, 0), (t + 1) // 2)
            self.assertEqual(ref_mbcnt_lo(t, odd_mask, 0), t // 2)

    def test_duality_partition_property(self):
        """Duality: popcnt(mask & below) + popcnt(~mask & below) == lane."""
        for lane in range(64):
            for mask in (0x12345678, 0xDEADBEEF, 0x00FF00FF, 0x55555555):
                inv_mask = (~mask) & MASK32
                if lane < 32:
                    cnt_1 = ref_mbcnt_lo(lane, mask, 0)
                    cnt_0 = ref_mbcnt_lo(lane, inv_mask, 0)
                    self.assertEqual(cnt_1 + cnt_0, lane)
                else:
                    rel = lane - 32
                    cnt_1 = ref_mbcnt_hi(lane, mask, 0)
                    cnt_0 = ref_mbcnt_hi(lane, inv_mask, 0)
                    self.assertEqual(cnt_1 + cnt_0, rel)

    def test_monotonicity(self):
        """Thread rank is monotonically non-decreasing with lane index."""
        for mask_lo in (0xAAAAAAAA, 0x0F0F0F0F, 0x80000001):
            for mask_hi in (0x55555555, 0xF0F0F0F0, 0x00000001):
                prev = -1
                for lane in range(64):
                    cur = ref_thread_rank_wave64(lane, mask_lo, mask_hi, 0)
                    self.assertGreaterEqual(cur, prev)
                    if prev != -1:
                        diff = cur - prev
                        self.assertIn(diff, (0, 1))
                    prev = cur


# -----------------------------------------------------------------------------
# 4. Multi-Lane SIMD Wave Simulation Tests
# -----------------------------------------------------------------------------

class TestSimdWaveInterpreter(unittest.TestCase):
    """Tests executing sequence through multi-lane SIMD wave interpreter."""

    def test_wave32_full_exec_thread_rank(self):
        wave = SimdWave(wave_size=32, exec_mask=MASK32)
        wave.compute_thread_ranks(dst=5)
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][5], lane)

    def test_wave64_full_exec_thread_rank(self):
        wave = SimdWave(wave_size=64, exec_mask=MASK64)
        wave.compute_thread_ranks(dst=3)
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][3], lane)

    def test_inactive_lanes_preserve_vgpr(self):
        """Lanes masked off by EXEC do not write destination VGPR."""
        exec_mask = (1 << 16) - 1
        wave = SimdWave(wave_size=64, exec_mask=exec_mask)
        for lane in range(64):
            wave.vgprs[lane][2] = 0xDEADBEEF

        wave.compute_thread_ranks(dst=2)
        for lane in range(16):
            self.assertEqual(wave.vgprs[lane][2], lane)
        for lane in range(16, 64):
            self.assertEqual(wave.vgprs[lane][2], 0xDEADBEEF)

    def test_compaction_with_simulated_ballot(self):
        """Ballot condition followed by v_mbcnt compaction produces slots."""
        wave = SimdWave(wave_size=64, exec_mask=MASK64)
        matching_lanes = [lane for lane in range(64) if (lane & 7) == 3]
        ballot_mask = sum(1 << lane for lane in matching_lanes)
        mask_lo = ballot_mask & MASK32
        mask_hi = (ballot_mask >> 32) & MASK32

        wave.compute_thread_ranks(dst=0, mask_lo=mask_lo, mask_hi=mask_hi)
        for expected_slot, lane in enumerate(matching_lanes):
            self.assertEqual(wave.vgprs[lane][0], expected_slot)


# -----------------------------------------------------------------------------
# 5. Translator Emission & AST Verification via rdna2spirv.Gen
# -----------------------------------------------------------------------------

class TestMbcntTranslatorEmission(unittest.TestCase):
    """Verifies that rdna2spirv translates v_mbcnt into SPIR-V ops."""

    def test_v_mbcnt_lo_emission_structure(self):
        g = rdna2spirv.Gen("cs")
        g.cs = {"size": (64, 1, 1)}
        ins = {
            "op": "v_mbcnt_lo_u32_b32",
            "args": [
                {"k": "vgpr", "r": 0},
                {"k": "imm", "v": 0xFFFFFFFF},
                {"k": "imm", "v": 0},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpULessThan %bool", emitted)
        self.assertIn("OpShiftLeftLogical %uint", emitted)
        self.assertIn("OpISub %uint", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)

    def test_v_mbcnt_hi_emission_wave64(self):
        g = rdna2spirv.Gen("cs")
        g.wave64 = True
        g.cs = {"size": (64, 1, 1)}
        ins = {
            "op": "v_mbcnt_hi_u32_b32",
            "args": [
                {"k": "vgpr", "r": 0},
                {"k": "imm", "v": 0xFFFFFFFF},
                {"k": "vgpr", "r": 0},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpISub %uint", emitted)
        self.assertIn("OpULessThan %bool", emitted)
        self.assertIn("OpShiftLeftLogical %uint", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)


# -----------------------------------------------------------------------------
# 6. End-to-End SPIR-V Shader Execution via spirv_mini_interp
# -----------------------------------------------------------------------------

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_PS_PATH: str | None = None
_CS_PATH: str | None = None


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)

    _PS_PATH = os.path.join(disasm_dir, "ps_mbcnt.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_mbcnt.s")

    # Pixel shader stub DXBC
    with open(os.path.join(_TMP_DIR.name, "ps_mbcnt.dxbc"), "wb") as f:
        f.write(STUB_DXBC)

    # Compute shader stub FSSM (COMPUTE_PGM_RSRC2 0x8090: 8 user SGPRs, 64x1x1)
    cs_fssm_path = os.path.join(fssm_dir, "cs_mbcnt.fssm")
    cs_hdr = bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1)
    with open(cs_fssm_path, "wb") as f:
        f.write(cs_hdr)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_ps_mbcnt_lines(lines: list[str], lane: int) -> int:
    """Translates a synthetic pixel shader and evaluates mrt0 output word."""
    if _TMP_DIR is None:
        setUpModule()
    body = "".join(f"\t{ln}\n" for ln in lines)
    asm = (
        "; test_ps_mbcnt\n"
        f"{body}"
        "\texp mrt0 v0, v0, v0, v0 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
    out = si.run_fragment(spv, inputs={"%sg_lane": lane})["%out_mrt0"]
    return out[0]


class TestSpirvMiniInterpMbcnt(unittest.TestCase):
    """End-to-end SPIR-V interpreter validation of translated shaders."""

    def test_ps_wave64_thread_rank_all_lanes(self):
        """Lanes 0..63 compute exact lane index in translated SPIR-V shader."""
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
        ]
        results = [run_ps_mbcnt_lines(lines, lane) for lane in range(64)]
        self.assertEqual(results, list(range(64)))

    def test_ps_mbcnt_lo_wave32(self):
        """A wave32 pixel shader counts lanes independently in both halves of a 64-lane subgroup."""
        lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0"]
        for lane in range(64):
            with self.subTest(subgroup_lane=lane):
                self.assertEqual(run_ps_mbcnt_lines(lines, lane), lane & 31)

    def test_ps_mbcnt_hi_lanes(self):
        """v_mbcnt_hi on lanes < 32 adds 0; lanes >= 32 count relative bits."""
        lines = ["v_mbcnt_hi_u32_b32 v0, -1, 0"]
        for lane in range(32):
            self.assertEqual(run_ps_mbcnt_lines(lines, lane), 0)
        for lane in range(32, 64):
            self.assertEqual(run_ps_mbcnt_lines(lines, lane), lane - 32)

    def test_ps_accumulator_offset(self):
        """Verifies base offset passed in accumulator operand."""
        base = 0x1000
        lines = [
            f"v_mov_b32 v1, {base:#x}",
            "v_mbcnt_lo_u32_b32 v0, -1, v1",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
        ]
        for lane in (0, 7, 31, 32, 63):
            got = run_ps_mbcnt_lines(lines, lane)
            self.assertEqual(got, base + lane)

    def test_ps_accumulator_wrapping(self):
        """Verifies 32-bit wrapping in SPIR-V evaluation."""
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0xFFFFFFFF",
        ]
        # lane 0: count is 0 -> 0xFFFFFFFF
        self.assertEqual(run_ps_mbcnt_lines(lines, 0), 0xFFFFFFFF)
        # lane 1: count is 1 -> 0 (wrapped)
        self.assertEqual(run_ps_mbcnt_lines(lines, 1), 0)
        # lane 2: count is 2 -> 1
        self.assertEqual(run_ps_mbcnt_lines(lines, 2), 1)

    def test_ps_arbitrary_mask(self):
        """Verifies non-trivial mask through SGPR operand."""
        mask_lo = 0x55555555
        mask_hi = 0xAAAAAAAA
        lines = [
            f"s_mov_b32 s4, {mask_lo:#x}",
            f"s_mov_b32 s5, {mask_hi:#x}",
            "v_mbcnt_lo_u32_b32 v0, s4, 0",
            "v_mbcnt_hi_u32_b32 v0, s5, v0",
        ]
        for lane in (0, 1, 2, 3, 16, 31, 32, 33, 48, 63):
            got = run_ps_mbcnt_lines(lines, lane)
            want = ref_thread_rank_wave64(lane, mask_lo, mask_hi, 0)
            self.assertEqual(got, want)

    def test_cs_workgroup_whole_wave_memory_dump(self):
        """A 64-thread compute shader stores each thread's ID into memory."""
        if _TMP_DIR is None:
            setUpModule()
        asm = """; cs_mbcnt
\tv_mbcnt_lo_u32_b32 v0, -1, 0
\tv_mbcnt_hi_u32_b32 v0, -1, v0
\tv_lshlrev_b32_e32 v1, 2, v0
\tbuffer_store_dword v0, v1, s[0:3], 0 offen
\ts_endpgm
"""
        with open(_CS_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_CS_PATH)

        vsharp = [0, 0, MASK32, 0x31016FAC]
        mem = {}
        si.run_workgroup(
            spv,
            64,
            64,
            inputs_for=lambda i: {
                "%wg_id": (0, 0, 0),
                "%local_id": (i, 0, 0),
                "%sg_lane": i % 64,
            },
            push_constants=vsharp + [0] * 28,
            memory=mem,
        )
        stored = [mem.get(i * 4) for i in range(64)]
        self.assertEqual(stored, list(range(64)))


# -----------------------------------------------------------------------------
# 7. Negative Controls and Mutation Invariants
# -----------------------------------------------------------------------------

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Verifies that deliberate bugs or corruptions are detected."""

    def test_inclusive_scan_rejected(self):
        """Inclusive scan differs from exclusive scan for set lane 0 bit."""
        mask = 1
        inclusive_lane_0 = 1
        exclusive_lane_0 = ref_mbcnt_lo(0, mask, 0)
        self.assertEqual(exclusive_lane_0, 0)
        self.assertNotEqual(inclusive_lane_0, exclusive_lane_0)

    def test_swapped_lo_hi_masks_fail(self):
        """Swapping lo and hi masks produces wrong rank for high-half lanes."""
        mask_lo = 0xFFFFFFFF
        mask_hi = 0x00000000
        # Correct rank for lane 32 is 32
        correct = ref_thread_rank_wave64(32, mask_lo, mask_hi, 0)
        self.assertEqual(correct, 32)
        # Swapped: lo has 0, hi has -1 -> lo gives 0, hi gives 0 -> 0 != 32
        swapped = ref_thread_rank_wave64(32, mask_hi, mask_lo, 0)
        self.assertEqual(swapped, 0)
        self.assertNotEqual(correct, swapped)

    def test_unclamped_shift_mutation(self):
        """Mutated calculation omitting clamp produces incorrect count."""
        # Simulated buggy implementation without lane - 32
        def buggy_mbcnt_hi(lane: int, mask_hi: int, acc: int) -> int:
            bitmask = (1 << lane) - 1
            return (popcount(mask_hi & bitmask) + acc) & MASK32

        # Lane 33: rel should be 1 (bitmask 1), buggy produces (1 << 33) - 1
        mask_hi = 0xFFFFFFFF
        correct = ref_mbcnt_hi(33, mask_hi, 0)
        self.assertEqual(correct, 1)
        buggy = buggy_mbcnt_hi(33, mask_hi, 0)
        self.assertNotEqual(correct, buggy)


if __name__ == "__main__":
    unittest.main()
