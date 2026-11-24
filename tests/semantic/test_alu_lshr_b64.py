"""Semantic interpreter tests for AMD RDNA2 64-bit vector logical shift right (v_lshrrev_b64).

v_lshrrev_b64 shifts the 64-bit vector register pair S1 right by the shift amount in S0 (modulo 64),
shifting zeros into the most significant bits:
    D[63:0] = (S1[63:0] & 0xFFFFFFFFFFFFFFFF) >> (S0[31:0] & 63)

In "rev" form, the shift count is source 0 (S0), and the 64-bit input is source 1 (S1):
    - Destination: VGPR pair v[d:d+1]
    - S0: shift amount (VGPR, SGPR, or inline constant; bits [5:0] used, higher bits ignored)
    - S1: 64-bit unsigned source (VGPR pair v[s:s+1], SGPR pair s[s:s+1], or 64-bit literal)

Standard library only, deterministic, zero game data, Python 3.9+ compatible.
"""
from __future__ import annotations

import gc
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_v_lshrrev_b64(shift: int, val: int) -> int:
    """Reference oracle for AMD RDNA2 v_lshrrev_b64.

    Shifts 64-bit unsigned integer `val` right by `(shift & 63)` bits with zero-extension.
    """
    shift_count = shift & 63
    return (val & MASK64) >> shift_count


def ref_v_lshrrev_b64_dwords(shift: int, lo: int, hi: int) -> tuple[int, int]:
    """Decomposed reference oracle modeling 32-bit register pair execution.

    Validates bit movements across the 32-bit halves (lo: v[s], hi: v[s+1]):
      - shift == 0:
          out_lo = lo, out_hi = hi
      - 1 <= shift < 32:
          out_lo = (lo >> shift) | ((hi << (32 - shift)) & 0xFFFFFFFF)
          out_hi = hi >> shift
      - shift == 32:
          out_lo = hi
          out_hi = 0
      - 33 <= shift <= 63:
          out_lo = (hi >> (shift - 32)) & 0xFFFFFFFF
          out_hi = 0
    """
    sh = shift & 63
    lo &= MASK32
    hi &= MASK32

    if sh == 0:
        return lo, hi
    if sh < 32:
        out_lo = ((lo >> sh) | ((hi << (32 - sh)) & MASK32)) & MASK32
        out_hi = (hi >> sh) & MASK32
        return out_lo, out_hi
    if sh == 32:
        return hi, 0
    # 33 <= sh <= 63
    out_lo = (hi >> (sh - 32)) & MASK32
    out_hi = 0
    return out_lo, out_hi


def pack_u64(lo: int, hi: int) -> int:
    """Pack two 32-bit dwords into a 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def unpack_u64(val: int) -> tuple[int, int]:
    """Unpack a 64-bit integer into (lo, hi) 32-bit dwords."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


EXPORT_HALVES = [
    "v_and_b32 v5, 0xffff, v0",
    "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1",
    "v_lshrrev_b32 v8, 16, v1",
    "v_or_b32 v5, 0x40000000, v5",
    "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7",
    "v_or_b32 v8, 0x40000000, v8",
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_lshr_b64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_lshr_b64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run_ps_lines(lines: list[str]) -> int:
    """Run instructions in a synthetic fragment shader and return the 64-bit pair exported in v[0:1]."""
    if _TMP_DIR is None:
        setUpModule()
    body = list(lines) + EXPORT_HALVES
    asm = (
        "; test_alu_lshr_b64\n"
        + "".join(f"\t{ln}\n" for ln in body)
        + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(x >> 16 == 0x4000 for x in w), [hex(x) for x in w]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w))


def run_shader_lshr_b64(shift: int, val: int) -> int:
    """Execute v_lshrrev_b64 v[0:1], v4, v[2:3] with inputs shift and val."""
    lo, hi = unpack_u64(val)
    return run_ps_lines([
        f"v_mov_b32 v4, {shift & MASK32:#x}",
        f"v_mov_b32 v2, {lo:#x}",
        f"v_mov_b32 v3, {hi:#x}",
        "v_lshrrev_b64 v[0:1], v4, v[2:3]",
    ])


class TestLshrB64ReferenceOracle(unittest.TestCase):
    """Verifies reference oracle consistency, dword decomposition, and edge conditions."""

    PATTERNS_64 = [
        0,
        1,
        2,
        0x5555555555555555,
        0xAAAAAAAAAAAAAAAA,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0x00000000FFFFFFFF,
        0xFFFFFFFF00000000,
        0x0000FFFF0000FFFF,
        0xFFFF0000FFFF0000,
        0x7FFFFFFFFFFFFFFF,
        0x8000000000000000,
        0x8000000000000001,
        0xFFFFFFFFFFFFFFFE,
        MASK64,
    ]

    def test_shift_zero(self):
        """A shift count of 0 must preserve the 64-bit value exactly."""
        for val in self.PATTERNS_64:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_v_lshrrev_b64(0, val), val & MASK64)
                lo, hi = unpack_u64(val)
                d_lo, d_hi = ref_v_lshrrev_b64_dwords(0, lo, hi)
                self.assertEqual(pack_u64(d_lo, d_hi), val & MASK64)

    def test_oracle_dword_decomposition_equivalence(self):
        """ref_v_lshrrev_b64 and ref_v_lshrrev_b64_dwords must produce identical results for all shifts."""
        for shift in range(64):
            for val in self.PATTERNS_64:
                with self.subTest(shift=shift, val=hex(val)):
                    expected = ref_v_lshrrev_b64(shift, val)
                    lo, hi = unpack_u64(val)
                    d_lo, d_hi = ref_v_lshrrev_b64_dwords(shift, lo, hi)
                    self.assertEqual(pack_u64(d_lo, d_hi), expected)

    def test_dword_boundary_shift_32(self):
        """At shift 32, the high dword moves to the low dword and the high dword becomes 0."""
        lo = 0x12345678
        hi = 0x9ABCDEF0
        val = pack_u64(lo, hi)
        res = ref_v_lshrrev_b64(32, val)
        self.assertEqual(res, hi)
        r_lo, r_hi = unpack_u64(res)
        self.assertEqual(r_lo, hi)
        self.assertEqual(r_hi, 0)

    def test_sub_dword_and_cross_dword_shifts(self):
        """Verify bit movement across the 32-bit boundary for shifts 1..31 and 33..63."""
        val = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        for shift in (1, 15, 16, 31, 33, 47, 48, 63):
            with self.subTest(shift=shift):
                expected = (val & MASK64) >> shift
                self.assertEqual(ref_v_lshrrev_b64(shift, val), expected)
                lo, hi = unpack_u64(val)
                d_lo, d_hi = ref_v_lshrrev_b64_dwords(shift, lo, hi)
                self.assertEqual(pack_u64(d_lo, d_hi), expected)

    def test_single_bit_walking_patterns(self):
        """Walk a single 1-bit through all 64 positions and verify right shifts."""
        for bit_pos in range(64):
            val = 1 << bit_pos
            for shift in (0, 1, bit_pos, (bit_pos + 1) % 64, 63):
                with self.subTest(bit_pos=bit_pos, shift=shift):
                    expected = (val >> shift) if shift <= bit_pos else 0
                    self.assertEqual(ref_v_lshrrev_b64(shift, val), expected)


class TestLshrB64Execution(unittest.TestCase):
    """End-to-end execution of v_lshrrev_b64 against the reference oracle."""

    def test_signalling_nan_word_survives_export(self):
        """Ensure sNaN bit patterns in register pairs survive export through 16-bit halves."""
        snan = 0xFFFFFFFFFFA3A095
        lo, hi = unpack_u64(snan)
        got = run_ps_lines([f"v_mov_b32 v0, {lo:#x}", f"v_mov_b32 v1, {hi:#x}"])
        self.assertEqual(got, snan)

    def test_shader_execution_vs_oracle(self):
        """Execute v_lshrrev_b64 in translated SPIR-V shaders and compare with oracle."""
        test_shifts = [0, 1, 7, 16, 31, 32, 33, 48, 63]
        test_values = [
            0,
            1,
            MASK32,
            1 << 32,
            1 << 63,
            MASK64,
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
            0x8000000000000001,
        ]
        for shift in test_shifts:
            for val in test_values:
                with self.subTest(shift=shift, val=hex(val)):
                    got = run_shader_lshr_b64(shift, val)
                    expected = ref_v_lshrrev_b64(shift, val)
                    self.assertEqual(got, expected)

    def test_seeded_pseudorandom_samples(self):
        """Deterministic pseudorandom values and shifts tested against oracle."""
        rng = random.Random(0x64B64)
        for _ in range(25):
            shift = rng.randint(0, 63)
            val = rng.getrandbits(64)
            with self.subTest(shift=shift, val=hex(val)):
                got = run_shader_lshr_b64(shift, val)
                expected = ref_v_lshrrev_b64(shift, val)
                self.assertEqual(got, expected)


def run_wave_vs(
    lines: list[str],
    wave: int = 32,
    builtins: dict | None = None,
    out_regs: tuple[str, str] = ("v2", "v3"),
) -> list[tuple[int, int]]:
    """Execute instructions in a synthetic vertex shader across SIMD lanes and collect exported pairs."""
    r_lo, r_hi = out_regs
    body = [
        "v_mov_b32 v6, 0",
        "exp pos0 v6, v6, v6, v6 done",
    ] + list(lines) + [
        f"exp param0 {r_lo}, {r_hi}, v6, v6",
        "s_endpgm",
    ]
    spv = support.translate_lines("vs_lshr_b64", body)
    if builtins is None:
        builtins = {"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0}
    res = wi.run_wave(spv, wave=wave, builtins=builtins)
    results = []
    for lane_out in res["outputs"]:
        if lane_out is None or "%out_param0" not in lane_out:
            results.append((0, 0))
        else:
            w = lane_out["%out_param0"]
            results.append((w[0], w[1]))
    return results


class TestLshrB64SimdWave(unittest.TestCase):
    """Multi-lane SIMD wave execution and VGPR register pair semantics."""

    def test_wave32_divergent_shift_counts(self):
        """32 lanes concurrently execute shifts by per-lane amounts 0..31."""
        val = 0xFEDCBA9876543210
        lo, hi = unpack_u64(val)
        lines = [
            f"v_mov_b32 v4, {lo:#x}",
            f"v_mov_b32 v5, {hi:#x}",
            "v_lshrrev_b64 v[2:3], v0, v[4:5]",
        ]
        pairs = run_wave_vs(lines, wave=32)
        self.assertEqual(len(pairs), 32)
        for lane, (r_lo, r_hi) in enumerate(pairs):
            with self.subTest(lane=lane):
                got = pack_u64(r_lo, r_hi)
                expected = ref_v_lshrrev_b64(lane, val)
                self.assertEqual(got, expected)

    def test_wave64_full_shift_domain(self):
        """64-lane wave simultaneously tests all 64 possible shift counts [0..63]."""
        val = 0x8000000000000001
        lo, hi = unpack_u64(val)
        lines = [
            "s_mov_b64 s[10:11], exec",
            f"v_mov_b32 v4, {lo:#x}",
            f"v_mov_b32 v5, {hi:#x}",
            "v_lshrrev_b64 v[2:3], v0, v[4:5]",
        ]
        pairs = run_wave_vs(lines, wave=64)
        self.assertEqual(len(pairs), 64)
        for lane, (r_lo, r_hi) in enumerate(pairs):
            with self.subTest(lane=lane):
                got = pack_u64(r_lo, r_hi)
                expected = ref_v_lshrrev_b64(lane, val)
                self.assertEqual(got, expected)

    def test_wave32_divergent_payloads(self):
        """Each SIMD lane computes and shifts its own unique 64-bit payload."""
        lines = [
            "s_mov_b32 s4, 12",
            "v_lshlrev_b32 v4, 24, v0",
            "v_xor_b32 v4, 0x55555555, v4",
            "v_lshlrev_b32 v5, 25, v0",
            "v_or_b32 v5, 0x80000000, v5",
            "v_lshrrev_b64 v[2:3], s4, v[4:5]",
        ]
        pairs = run_wave_vs(lines, wave=32)
        self.assertEqual(len(pairs), 32)
        for lane, (r_lo, r_hi) in enumerate(pairs):
            with self.subTest(lane=lane):
                lane_lo = ((lane << 24) ^ 0x55555555) & MASK32
                lane_hi = ((lane << 25) | 0x80000000) & MASK32
                lane_val = pack_u64(lane_lo, lane_hi)
                expected = ref_v_lshrrev_b64(12, lane_val)
                got = pack_u64(r_lo, r_hi)
                self.assertEqual(got, expected)

    def test_vgpr_pairs_in_place(self):
        """In-place shift where destination and source VGPR pairs are identical: v[2:3], v0, v[2:3]."""
        val = 0xFEDCBA9876543210
        lo, hi = unpack_u64(val)
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_lshrrev_b64 v[2:3], v0, v[2:3]",
        ]
        pairs = run_wave_vs(lines, wave=32)
        for lane, (r_lo, r_hi) in enumerate(pairs):
            with self.subTest(lane=lane):
                got = pack_u64(r_lo, r_hi)
                expected = ref_v_lshrrev_b64(lane, val)
                self.assertEqual(got, expected)

    def test_vgpr_pairs_partially_overlapping(self):
        """Partially overlapping VGPR pairs must preserve unshifted source data across dword writes."""
        val = 0xA1B2C3D4E5F60718
        lo, hi = unpack_u64(val)

        # Overlap A: dest v[2:3], src v[3:4] (dest hi v3 is src lo v3)
        lines_a = [
            f"v_mov_b32 v3, {lo:#x}",
            f"v_mov_b32 v4, {hi:#x}",
            "v_lshrrev_b64 v[2:3], v0, v[3:4]",
        ]
        pairs_a = run_wave_vs(lines_a, wave=32, out_regs=("v2", "v3"))
        for lane, (r_lo, r_hi) in enumerate(pairs_a):
            with self.subTest(case="overlap_lo", lane=lane):
                self.assertEqual(pack_u64(r_lo, r_hi), ref_v_lshrrev_b64(lane, val))

        # Overlap B: dest v[3:4], src v[2:3] (dest lo v3 is src hi v3)
        lines_b = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_lshrrev_b64 v[3:4], v0, v[2:3]",
        ]
        pairs_b = run_wave_vs(lines_b, wave=32, out_regs=("v3", "v4"))
        for lane, (r_lo, r_hi) in enumerate(pairs_b):
            with self.subTest(case="overlap_hi", lane=lane):
                self.assertEqual(pack_u64(r_lo, r_hi), ref_v_lshrrev_b64(lane, val))

    def test_vgpr_pairs_non_overlapping_indices(self):
        """Non-overlapping VGPR register pairs across standard and higher register indices."""
        val = 0x1122334455667788
        lo, hi = unpack_u64(val)
        configs = [
            (("v0", "v1"), ("v2", "v3")),
            (("v6", "v7"), ("v8", "v9")),
            (("v20", "v21"), ("v22", "v23")),
            (("v40", "v41"), ("v42", "v43")),
        ]
        for (d_lo, d_hi), (s_lo, s_hi) in configs:
            d_base = d_lo[1:]
            d_next = str(int(d_base) + 1)
            s_base = s_lo[1:]
            s_next = str(int(s_base) + 1)
            lines = [
                f"v_mov_b32 {s_lo}, {lo:#x}",
                f"v_mov_b32 {s_hi}, {hi:#x}",
                f"v_lshrrev_b64 v[{d_base}:{d_next}], v0, v[{s_base}:{s_next}]",
            ]
            pairs = run_wave_vs(lines, wave=32, out_regs=(d_lo, d_hi))
            for lane, (r_lo, r_hi) in enumerate(pairs):
                with self.subTest(dst=d_lo, lane=lane):
                    self.assertEqual(pack_u64(r_lo, r_hi), ref_v_lshrrev_b64(lane, val))

    def test_sgpr_uniform_source_pair(self):
        """Source 1 from 64-bit SGPR pair s[4:5] shifted by divergent per-lane VGPR amounts."""
        val = 0x0123456789ABCDEF
        lo, hi = unpack_u64(val)
        lines = [
            f"s_mov_b32 s4, {lo:#x}",
            f"s_mov_b32 s5, {hi:#x}",
            "v_lshrrev_b64 v[2:3], v0, s[4:5]",
        ]
        pairs = run_wave_vs(lines, wave=32)
        for lane, (r_lo, r_hi) in enumerate(pairs):
            with self.subTest(lane=lane):
                self.assertEqual(pack_u64(r_lo, r_hi), ref_v_lshrrev_b64(lane, val))

    def test_sgpr_uniform_shift_amount(self):
        """Shift amount from scalar SGPR broadcast across all lanes in the wave."""
        val = 0xF0E1D2C3B4A59687
        lo, hi = unpack_u64(val)
        for shift in (0, 1, 13, 31, 32, 45, 63):
            lines = [
                f"s_mov_b32 s4, {shift}",
                f"v_mov_b32 v4, {lo:#x}",
                f"v_mov_b32 v5, {hi:#x}",
                "v_lshrrev_b64 v[2:3], s4, v[4:5]",
            ]
            pairs = run_wave_vs(lines, wave=32)
            expected = ref_v_lshrrev_b64(shift, val)
            for lane, (r_lo, r_hi) in enumerate(pairs):
                with self.subTest(shift=shift, lane=lane):
                    self.assertEqual(pack_u64(r_lo, r_hi), expected)

    def test_inline_constants_and_literals(self):
        """Inline constant shift counts and 64-bit literal operand evaluation."""
        val = 0xAABBCCDDEEFF0011
        lo, hi = unpack_u64(val)
        for imm_shift in (0, 1, 16, 32, 63):
            lines = [
                f"v_mov_b32 v4, {lo:#x}",
                f"v_mov_b32 v5, {hi:#x}",
                f"v_lshrrev_b64 v[2:3], {imm_shift}, v[4:5]",
            ]
            pairs = run_wave_vs(lines, wave=32)
            expected = ref_v_lshrrev_b64(imm_shift, val)
            for lane, (r_lo, r_hi) in enumerate(pairs):
                with self.subTest(imm_shift=imm_shift, lane=lane):
                    self.assertEqual(pack_u64(r_lo, r_hi), expected)

        lines_lit = [
            "v_lshrrev_b64 v[2:3], v0, 0x100",
        ]
        pairs_lit = run_wave_vs(lines_lit, wave=32)
        for lane, (r_lo, r_hi) in enumerate(pairs_lit):
            with self.subTest(case="literal_s1", lane=lane):
                self.assertEqual(pack_u64(r_lo, r_hi), ref_v_lshrrev_b64(lane, 0x100))


class TestLshrB64ExtremesAndControls(unittest.TestCase):
    """Zero-extension extremes, shift count modulo 64 behavior, and negative controls."""

    def test_zero_extension_extremes_and_overflow(self):
        """Zero-extension semantics when shifting out all bits or shifting by >= 64 (modulo 64 wrapping)."""
        val = MASK64
        for shift in (64, 65, 96, 127, 255):
            with self.subTest(shift=shift):
                mod_shift = shift & 63
                got = run_shader_lshr_b64(shift, val)
                expected = ref_v_lshrrev_b64(mod_shift, val)
                self.assertEqual(got, expected)

        val2 = 0x123456789ABCDEF0
        for shift in (64, 100, 1024):
            with self.subTest(shift=shift, val=hex(val2)):
                got = run_shader_lshr_b64(shift, val2)
                mod_shift = shift & 63
                self.assertEqual(got, ref_v_lshrrev_b64(mod_shift, val2))

    def test_negative_controls_invalid_mnemonics_or_syntax(self):
        """Negative controls verifying translation rejection or error handling for malformed instructions."""
        invalid_instructions = [
            "v_lshrrev_b64 v2, v0, v[4:5]",
            "v_lshrrev_b64 v[2:3], v0, v4",
            "v_lshrrev_b64 v[2:3], v[4:5], v[6:7]",
        ]
        for bad_line in invalid_instructions:
            with self.subTest(bad_line=bad_line):
                with self.assertRaises(Exception):
                    run_ps_lines([bad_line])


if __name__ == "__main__":
    unittest.main()
