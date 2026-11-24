"""Semantic interpreter tests for RDNA2 scalar bit-reversal operations and FFT index permutations.

Covers:
  - s_brev_b32 D, S0: 32-bit scalar bit reversal (SOP1 format).
      Reverses all 32 bits of scalar source S0 into 32-bit SGPR destination D.
      Bit i (0 <= i <= 31) moves to bit (31 - i).
      Scalar Condition Code (SCC) is preserved (NOT modified).
  - s_brev_b64 D, S0: 64-bit scalar bit reversal across SGPR pairs s[d:d+1] from s[s:s+1].
      Reverses all 64 bits of 64-bit scalar source S0 into 64-bit SGPR destination D.
      Bit i (0 <= i <= 63) moves to bit (63 - i).
      Equivalence with 32-bit dwords: D[31:0] = brev32(S0[63:32]), D[63:32] = brev32(S0[31:0]).
      Scalar Condition Code (SCC) is preserved (NOT modified).
  - FFT Index Permutations (Bit-Reversal Permutations):
      Radix-2 Cooley-Tukey decimation-in-time (DIT) FFT address generation.
      Index reversal: rev_k(n) = s_lshr_b32(s_brev_b32(n), 32 - k) for N = 2^k points.
      64-bit FFT index permutation: rev_k(n) = s_lshr_b64(s_brev_b64(n), 64 - k).
      Involution property: rev_k(rev_k(n)) == n (self-inverse permutation).
      Bijective mapping and in-place butterfly pair swapping (i < rev_k(i)).
      Dual packed FFT index reversals (packed 16-bit or 32-bit sub-indices).
  - Deterministic semantic interpreter ScalarBrevInterpreter:
      SGPR state table (106 registers), 32-bit and 64-bit reads/writes, in-place execution,
      overlapping SGPR pairs, immediate constant operands, and SCC preservation.
  - SPIR-V lowering and mini-interpreter validation:
      OpBitReverse %uint and OpBitReverse %ulong execution in spirv_mini_interp,
      rdna2spirv AST generation, and synthetic shader translation.

Deterministic, zero game data, Python 3.9+ compatible.
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


def u32(v: int) -> int:
    """Normalize integer to 32-bit unsigned."""
    return v & MASK32


def u64(v: int) -> int:
    """Normalize integer to 64-bit unsigned."""
    return v & MASK64


def split_dwords(v64: int) -> tuple[int, int]:
    """Split 64-bit unsigned integer into (lo32, hi32)."""
    v = v64 & MASK64
    return v & MASK32, (v >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (lo32, hi32) into 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def popcount(v: int) -> int:
    """Count number of set bits (Python 3.9 compatible without int.bit_count)."""
    return bin(v).count("1")


def ref_brev32(x: int) -> int:
    """Plain-Python 32-bit bit-reversal reference oracle."""
    return int(f"{x & MASK32:032b}"[::-1], 2)


def ref_brev64(x: int) -> int:
    """Plain-Python 64-bit bit-reversal reference oracle."""
    return int(f"{x & MASK64:064b}"[::-1], 2)


def fft_bitrev_index_32(index: int, log2_n: int) -> int:
    """Compute bit-reversed index for an N = 2^log2_n point FFT using 32-bit bit reversal.

    In RDNA2 shaders:
      s_brev_b32 s_rev, s_idx
      s_lshr_b32 s_rev, s_rev, (32 - log2_n)
    """
    assert 0 <= log2_n <= 32
    if log2_n == 0:
        return 0
    shift = 32 - log2_n
    return ref_brev32(index) >> shift


def fft_bitrev_index_64(index: int, log2_n: int) -> int:
    """Compute bit-reversed index for an N = 2^log2_n point FFT using 64-bit bit reversal.

    In RDNA2 shaders:
      s_brev_b64 s[rev_lo:rev_hi], s[idx_lo:idx_hi]
      s_lshr_b64 s[rev_lo:rev_hi], s[rev_lo:rev_hi], (64 - log2_n)
    """
    assert 0 <= log2_n <= 64
    if log2_n == 0:
        return 0
    shift = 64 - log2_n
    return ref_brev64(index) >> shift


def ref_fft_butterfly_pairs(n_points: int) -> list[tuple[int, int]]:
    """Return pairs of indices (i, j) with i < j to swap during in-place radix-2 DIT FFT reordering."""
    assert n_points > 0 and (n_points & (n_points - 1)) == 0
    log2_n = (n_points - 1).bit_length()
    pairs = []
    for i in range(n_points):
        j = fft_bitrev_index_32(i, log2_n)
        if i < j:
            pairs.append((i, j))
    return pairs


class ScalarBrevInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar bit-reversal operations.

    Tracks 32-bit SGPRs and SCC (Scalar Condition Code).
    In RDNA2 ISA:
      - s_brev_b32 D, S0 reverses all 32 bits of S0 and writes to 32-bit SGPR D.
        SCC is preserved (NOT modified).
      - s_brev_b64 D, S0 reverses all 64 bits of 64-bit operand S0 and writes
        to 64-bit SGPR pair s[d:d+1].
        Equivalently: D_lo = ref_brev32(S0_hi), D_hi = ref_brev32(S0_lo).
        SCC is preserved (NOT modified).
    """

    def __init__(self, num_sgprs: int = 106, initial_scc: int = 0) -> None:
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.scc = initial_scc & 1

    def reset(self, initial_scc: int = 0) -> None:
        self.sgpr = [0] * self.num_sgprs
        self.scc = initial_scc & 1

    def get_sgpr_u32(self, r: int) -> int:
        if not (0 <= r < self.num_sgprs):
            raise IndexError(f"SGPR index {r} out of range [0, {self.num_sgprs})")
        return self.sgpr[r] & MASK32

    def set_sgpr_u32(self, r: int, val: int) -> None:
        if not (0 <= r < self.num_sgprs):
            raise IndexError(f"SGPR index {r} out of range [0, {self.num_sgprs})")
        self.sgpr[r] = val & MASK32

    def get_sgpr_u64(self, r: int) -> int:
        if not (0 <= r < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {r} out of range [0, {self.num_sgprs - 1})")
        return join_dwords(self.get_sgpr_u32(r), self.get_sgpr_u32(r + 1))

    def set_sgpr_u64(self, r: int, val64: int) -> None:
        if not (0 <= r < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {r} out of range [0, {self.num_sgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(r, lo)
        self.set_sgpr_u32(r + 1, hi)

    def resolve_src32(self, opnd: int | dict) -> int:
        """Resolve a 32-bit source operand (integer, SGPR dict, or imm dict)."""
        if isinstance(opnd, int):
            return opnd & MASK32
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "sgpr":
                return self.get_sgpr_u32(opnd["r"])
            if k == "imm":
                return opnd["v"] & MASK32
            raise ValueError(f"Unsupported 32-bit operand kind: {k}")
        raise TypeError(f"Unsupported operand type: {type(opnd)}")

    def resolve_src64(self, opnd: int | dict) -> int:
        """Resolve a 64-bit source operand (integer, SGPR pair dict, or imm dict)."""
        if isinstance(opnd, int):
            return opnd & MASK64
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "sgpr":
                return self.get_sgpr_u64(opnd["r"])
            if k == "imm":
                return opnd["v"] & MASK64
            raise ValueError(f"Unsupported 64-bit operand kind: {k}")
        raise TypeError(f"Unsupported operand type: {type(opnd)}")

    def step_s_brev_b32(self, dst: int | dict, src: int | dict) -> int:
        """Execute s_brev_b32: D = ref_brev32(S0). SCC is preserved."""
        val = self.resolve_src32(src)
        res = ref_brev32(val)
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u32(dst_r, res)
        return res

    def step_s_brev_b64(self, dst: int | dict, src: int | dict) -> int:
        """Execute s_brev_b64: D = ref_brev64(S0). SCC is preserved."""
        val = self.resolve_src64(src)
        res = ref_brev64(val)
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u64(dst_r, res)
        return res

    def step_s_lshr_b32(self, dst: int | dict, src0: int | dict, src1: int | dict) -> int:
        """Execute s_lshr_b32 helper: D = S0 >> (S1 & 31), SCC = (D != 0)."""
        v0 = self.resolve_src32(src0)
        cnt = self.resolve_src32(src1) & 31
        res = (v0 >> cnt) & MASK32
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u32(dst_r, res)
        self.scc = 1 if res != 0 else 0
        return res

    def step_s_lshr_b64(self, dst: int | dict, src0: int | dict, src1: int | dict) -> int:
        """Execute s_lshr_b64 helper: D = S0 >> (S1 & 63), SCC = (D != 0)."""
        v0 = self.resolve_src64(src0)
        cnt = self.resolve_src32(src1) & 63
        res = (v0 >> cnt) & MASK64
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u64(dst_r, res)
        self.scc = 1 if res != 0 else 0
        return res

    def step_s_mov_b32(self, dst: int | dict, src: int | dict) -> int:
        """Execute s_mov_b32 helper."""
        val = self.resolve_src32(src)
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u32(dst_r, val)
        return val

    def step_s_mov_b64(self, dst: int | dict, src: int | dict) -> int:
        """Execute s_mov_b64 helper."""
        val = self.resolve_src64(src)
        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u64(dst_r, val)
        return val

    def execute_ir(self, insn: dict) -> int:
        """Execute an IR instruction dictionary matching tools/rdna2spirv.py AST format."""
        op = insn["op"]
        args = insn["args"]
        if op == "s_brev_b32":
            return self.step_s_brev_b32(args[0], args[1])
        elif op == "s_brev_b64":
            return self.step_s_brev_b64(args[0], args[1])
        elif op == "s_lshr_b32":
            return self.step_s_lshr_b32(args[0], args[1], args[2])
        elif op == "s_lshr_b64":
            return self.step_s_lshr_b64(args[0], args[1], args[2])
        elif op == "s_mov_b32":
            return self.step_s_mov_b32(args[0], args[1])
        elif op == "s_mov_b64":
            return self.step_s_mov_b64(args[0], args[1])
        raise KeyError(f"Unsupported instruction opcode: {op}")


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_brev_test.s")
    with open(os.path.join(_TMP_DIR.name, "ps_brev_test.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run_synthetic_s_brev_b32(x: int) -> int:
    """Translate and run a synthetic pixel shader with s_brev_b32 through rdna2spirv and mini_interp."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; test_s_brev_b32\n"
        f"\ts_mov_b32 s2, {x:#x}\n"
        "\ts_brev_b32 s0, s2\n"
        "\tv_mov_b32 v0, s0\n"
        "\tv_mov_b32 v1, 0\n"
        "\texp mrt0 v0, v1, v0, v0 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_synthetic_s_brev_b64(x64: int) -> int:
    """Translate and run a synthetic pixel shader with s_brev_b64 through rdna2spirv and mini_interp."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    lo, hi = split_dwords(x64)
    asm = (
        "; test_s_brev_b64\n"
        f"\ts_mov_b32 s2, {lo:#x}\n"
        f"\ts_mov_b32 s3, {hi:#x}\n"
        "\ts_brev_b64 s[0:1], s[2:3]\n"
        "\tv_mov_b32 v0, s0\n"
        "\tv_mov_b32 v1, s1\n"
        "\texp mrt0 v0, v1, v0, v0 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    r = si.run_fragment(spv)["%out_mrt0"]
    return join_dwords(r[0], r[1])


class TestReferenceBitReversal(unittest.TestCase):
    """Mathematical invariants, identities, and edge cases for 32-bit and 64-bit bit reversals."""

    VALUES32 = [
        0,
        1,
        2,
        0x80000000,
        0x7FFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFE,
        0x12345678,
        0x0000FFFF,
        0xFFFF0000,
        0x55555555,
        0xAAAAAAAA,
        0x33333333,
        0xCCCCCCCC,
        0x0F0F0F0F,
        0xF0F0F0F0,
        0x00FF00FF,
        0xFF00FF00,
        0xA5A5A5A5,
        0xDEADBEEF,
        0xCAFEBABE,
    ]

    VALUES64 = [
        0,
        1,
        2,
        0x8000000000000000,
        0x7FFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFE,
        0x00000000FFFFFFFF,
        0xFFFFFFFF00000000,
        0x123456789ABCDEF0,
        0x5555555555555555,
        0xAAAAAAAAAAAAAAAA,
        0x3333333333333333,
        0xCCCCCCCCCCCCCCCC,
        0x0F0F0F0F0F0F0F0F,
        0xF0F0F0F0F0F0F0F0,
        0x0000000100000000,
        0x0000000080000000,
        0xDEADBEEFCAFEBABE,
    ]

    def test_ref_brev32_known_values(self):
        self.assertEqual(ref_brev32(0), 0)
        self.assertEqual(ref_brev32(1), 0x80000000)
        self.assertEqual(ref_brev32(2), 0x40000000)
        self.assertEqual(ref_brev32(0x80000000), 1)
        self.assertEqual(ref_brev32(0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_brev32(0x12345678), 0x1E6A2C48)
        self.assertEqual(ref_brev32(0x0000FFFF), 0xFFFF0000)
        self.assertEqual(ref_brev32(0xFFFF0000), 0x0000FFFF)
        self.assertEqual(ref_brev32(0x55555555), 0xAAAAAAAA)
        self.assertEqual(ref_brev32(0xAAAAAAAA), 0x55555555)
        self.assertEqual(ref_brev32(0x33333333), 0xCCCCCCCC)
        self.assertEqual(ref_brev32(0x0F0F0F0F), 0xF0F0F0F0)

    def test_ref_brev32_single_bit_walk(self):
        for bit in range(32):
            val = 1 << bit
            expected = 1 << (31 - bit)
            self.assertEqual(ref_brev32(val), expected)

    def test_ref_brev32_involution(self):
        for val in self.VALUES32:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_brev32(ref_brev32(val)), val)

    def test_ref_brev32_popcount_invariant(self):
        for val in self.VALUES32:
            with self.subTest(val=hex(val)):
                self.assertEqual(popcount(ref_brev32(val)), popcount(val))

    def test_ref_brev32_bitwise_not_duality(self):
        for val in self.VALUES32:
            with self.subTest(val=hex(val)):
                not_val = (~val) & MASK32
                self.assertEqual(ref_brev32(not_val), (~ref_brev32(val)) & MASK32)

    def test_ref_brev32_palindromes(self):
        palindromes = [
            0,
            0xFFFFFFFF,
            0x80000001,
            0xC0000003,
            0x81818181,
            0x42424242,
            0x24242424,
            0x18181818,
            0x99999999,
        ]
        for p in palindromes:
            with self.subTest(palindrome=hex(p)):
                self.assertEqual(ref_brev32(p), p)

    def test_ref_brev64_known_values(self):
        self.assertEqual(ref_brev64(0), 0)
        self.assertEqual(ref_brev64(1), 0x8000000000000000)
        self.assertEqual(ref_brev64(2), 0x4000000000000000)
        self.assertEqual(ref_brev64(0x8000000000000000), 1)
        self.assertEqual(ref_brev64(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(ref_brev64(0x00000000FFFFFFFF), 0xFFFFFFFF00000000)
        self.assertEqual(ref_brev64(0xFFFFFFFF00000000), 0x00000000FFFFFFFF)
        self.assertEqual(ref_brev64(0x123456789ABCDEF0), 0x0F7B3D591E6A2C48)
        self.assertEqual(ref_brev64(0x5555555555555555), 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(ref_brev64(0xAAAAAAAAAAAAAAAA), 0x5555555555555555)

    def test_ref_brev64_single_bit_walk(self):
        for bit in range(64):
            val = 1 << bit
            expected = 1 << (63 - bit)
            self.assertEqual(ref_brev64(val), expected)

    def test_ref_brev64_involution(self):
        for val in self.VALUES64:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_brev64(ref_brev64(val)), val)

    def test_ref_brev64_popcount_invariant(self):
        for val in self.VALUES64:
            with self.subTest(val=hex(val)):
                self.assertEqual(popcount(ref_brev64(val)), popcount(val))

    def test_ref_brev64_dword_decomposition(self):
        for val in self.VALUES64:
            with self.subTest(val=hex(val)):
                lo = val & MASK32
                hi = (val >> 32) & MASK32
                expected_lo = ref_brev32(hi)
                expected_hi = ref_brev32(lo)
                expected_64 = (expected_hi << 32) | expected_lo
                self.assertEqual(ref_brev64(val), expected_64)


class TestScalarBrevInterpreter(unittest.TestCase):
    """Semantic interpreter execution of s_brev_b32 and s_brev_b64."""

    def setUp(self):
        self.interp = ScalarBrevInterpreter(num_sgprs=106, initial_scc=0)

    def test_sgpr_register_access_and_bounds(self):
        self.interp.set_sgpr_u32(0, 0x12345678)
        self.assertEqual(self.interp.get_sgpr_u32(0), 0x12345678)

        self.interp.set_sgpr_u64(2, 0x123456789ABCDEF0)
        self.assertEqual(self.interp.get_sgpr_u64(2), 0x123456789ABCDEF0)
        self.assertEqual(self.interp.get_sgpr_u32(2), 0x9ABCDEF0)
        self.assertEqual(self.interp.get_sgpr_u32(3), 0x12345678)

        with self.assertRaises(IndexError):
            self.interp.get_sgpr_u32(106)
        with self.assertRaises(IndexError):
            self.interp.set_sgpr_u32(-1, 0)
        with self.assertRaises(IndexError):
            self.interp.get_sgpr_u64(105)

    def test_s_brev_b32_basic(self):
        test_cases = [0, 1, 0x12345678, 0x80000000, 0x55555555, 0xFFFFFFFF]
        for x in test_cases:
            with self.subTest(x=hex(x)):
                self.interp.set_sgpr_u32(4, x)
                res = self.interp.step_s_brev_b32(0, {"k": "sgpr", "r": 4})
                self.assertEqual(res, ref_brev32(x))
                self.assertEqual(self.interp.get_sgpr_u32(0), ref_brev32(x))

    def test_s_brev_b32_in_place(self):
        self.interp.set_sgpr_u32(7, 0x12345678)
        self.interp.step_s_brev_b32(7, {"k": "sgpr", "r": 7})
        self.assertEqual(self.interp.get_sgpr_u32(7), 0x1E6A2C48)
        self.interp.step_s_brev_b32(7, {"k": "sgpr", "r": 7})
        self.assertEqual(self.interp.get_sgpr_u32(7), 0x12345678)

    def test_s_brev_b32_immediate_operand(self):
        self.interp.step_s_brev_b32(1, {"k": "imm", "v": 0x0000FFFF})
        self.assertEqual(self.interp.get_sgpr_u32(1), 0xFFFF0000)

    def test_s_brev_b32_preserves_scc(self):
        for initial_scc in (0, 1):
            for x in (0, 1, 0xFFFFFFFF, 0x80000000):
                with self.subTest(scc=initial_scc, x=hex(x)):
                    self.interp.reset(initial_scc=initial_scc)
                    self.interp.set_sgpr_u32(10, x)
                    self.interp.step_s_brev_b32(11, {"k": "sgpr", "r": 10})
                    self.assertEqual(self.interp.scc, initial_scc)

    def test_s_brev_b64_basic(self):
        test_cases = [0, 1, 0x8000000000000000, 0x123456789ABCDEF0, 0xFFFFFFFFFFFFFFFF]
        for val64 in test_cases:
            with self.subTest(val=hex(val64)):
                self.interp.set_sgpr_u64(2, val64)
                res = self.interp.step_s_brev_b64(0, {"k": "sgpr", "r": 2})
                self.assertEqual(res, ref_brev64(val64))
                self.assertEqual(self.interp.get_sgpr_u64(0), ref_brev64(val64))

    def test_s_brev_b64_in_place(self):
        val = 0x123456789ABCDEF0
        self.interp.set_sgpr_u64(4, val)
        self.interp.step_s_brev_b64(4, {"k": "sgpr", "r": 4})
        self.assertEqual(self.interp.get_sgpr_u64(4), ref_brev64(val))
        self.interp.step_s_brev_b64(4, {"k": "sgpr", "r": 4})
        self.assertEqual(self.interp.get_sgpr_u64(4), val)

    def test_s_brev_b64_overlapping_sgprs(self):
        val = 0x0123456789ABCDEF
        self.interp.set_sgpr_u64(2, val)
        self.interp.step_s_brev_b64(1, {"k": "sgpr", "r": 2})
        self.assertEqual(self.interp.get_sgpr_u64(1), ref_brev64(val))

    def test_s_brev_b64_immediate_operand(self):
        imm = 0x00000000FFFFFFFF
        self.interp.step_s_brev_b64(8, {"k": "imm", "v": imm})
        self.assertEqual(self.interp.get_sgpr_u64(8), 0xFFFFFFFF00000000)

    def test_s_brev_b64_preserves_scc(self):
        for initial_scc in (0, 1):
            for x in (0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000):
                with self.subTest(scc=initial_scc, x=hex(x)):
                    self.interp.reset(initial_scc=initial_scc)
                    self.interp.set_sgpr_u64(10, x)
                    self.interp.step_s_brev_b64(12, {"k": "sgpr", "r": 10})
                    self.assertEqual(self.interp.scc, initial_scc)

    def test_s_brev_b64_vs_lowered_dual_s_brev_b32(self):
        val64 = 0x123456789ABCDEF0
        self.interp.set_sgpr_u64(0, val64)
        self.interp.step_s_brev_b64(2, {"k": "sgpr", "r": 0})

        self.interp.step_s_brev_b32(4, {"k": "sgpr", "r": 1})
        self.interp.step_s_brev_b32(5, {"k": "sgpr", "r": 0})

        self.assertEqual(self.interp.get_sgpr_u64(2), self.interp.get_sgpr_u64(4))
        self.assertEqual(self.interp.get_sgpr_u64(2), ref_brev64(val64))

    def test_execute_ir(self):
        ins_b32 = {
            "op": "s_brev_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x12345678}],
        }
        res32 = self.interp.execute_ir(ins_b32)
        self.assertEqual(res32, 0x1E6A2C48)
        self.assertEqual(self.interp.get_sgpr_u32(0), 0x1E6A2C48)

        ins_b64 = {
            "op": "s_brev_b64",
            "args": [{"k": "sgpr", "r": 2}, {"k": "imm", "v": 1}],
        }
        res64 = self.interp.execute_ir(ins_b64)
        self.assertEqual(res64, 0x8000000000000000)
        self.assertEqual(self.interp.get_sgpr_u64(2), 0x8000000000000000)


class TestFFTIndexPermutations(unittest.TestCase):
    """Radix-2 Cooley-Tukey FFT bit-reversal address permutations and butterfly swaps."""

    def test_fft_known_permutations_n4(self):
        expected_n4 = [0, 2, 1, 3]
        for i in range(4):
            self.assertEqual(fft_bitrev_index_32(i, 2), expected_n4[i])

    def test_fft_known_permutations_n8(self):
        expected_n8 = [0, 4, 2, 6, 1, 5, 3, 7]
        for i in range(8):
            self.assertEqual(fft_bitrev_index_32(i, 3), expected_n8[i])

    def test_fft_known_permutations_n16(self):
        expected_n16 = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        for i in range(16):
            self.assertEqual(fft_bitrev_index_32(i, 4), expected_n16[i])

    def test_fft_permutation_bijection_and_fixed_points(self):
        for log2_n in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            n = 1 << log2_n
            with self.subTest(n=n, log2_n=log2_n):
                permuted = [fft_bitrev_index_32(i, log2_n) for i in range(n)]
                self.assertEqual(len(set(permuted)), n)
                self.assertEqual(min(permuted), 0)
                self.assertEqual(max(permuted), n - 1)
                self.assertEqual(permuted[0], 0)
                self.assertEqual(permuted[n - 1], n - 1)
                for i in range(n):
                    rev = permuted[i]
                    self.assertEqual(fft_bitrev_index_32(rev, log2_n), i)

    def test_fft_butterfly_pairs(self):
        pairs_n8 = ref_fft_butterfly_pairs(8)
        self.assertEqual(pairs_n8, [(1, 4), (3, 6)])

        pairs_n16 = ref_fft_butterfly_pairs(16)
        expected_n16 = [(1, 8), (2, 4), (3, 12), (5, 10), (7, 14), (11, 13)]
        self.assertEqual(pairs_n16, expected_n16)

    def test_fft_in_place_array_reordering(self):
        for log2_n in (3, 4, 6):
            n = 1 << log2_n
            data = [f"sample_{i}" for i in range(n)]
            pairs = ref_fft_butterfly_pairs(n)
            for i, j in pairs:
                data[i], data[j] = data[j], data[i]
            for idx in range(n):
                rev_idx = fft_bitrev_index_32(idx, log2_n)
                self.assertEqual(data[idx], f"sample_{rev_idx}")
            for i, j in pairs:
                data[i], data[j] = data[j], data[i]
            self.assertEqual(data, [f"sample_{i}" for i in range(n)])

    def test_fft_64bit_huge_index_permutations(self):
        for log2_n in (33, 40, 48, 60, 64):
            with self.subTest(log2_n=log2_n):
                self.assertEqual(fft_bitrev_index_64(0, log2_n), 0)
                self.assertEqual(fft_bitrev_index_64(1, log2_n), 1 << (log2_n - 1))
                self.assertEqual(fft_bitrev_index_64(1 << (log2_n - 1), log2_n), 1)

                max_idx = (1 << log2_n) - 1
                self.assertEqual(fft_bitrev_index_64(max_idx, log2_n), max_idx)

                test_indices = [
                    0x12345678,
                    (1 << (log2_n - 2)) | 0x55,
                    (1 << (log2_n - 1)) | 0x3,
                ]
                for idx in test_indices:
                    rev = fft_bitrev_index_64(idx, log2_n)
                    self.assertEqual(fft_bitrev_index_64(rev, log2_n), idx)

    def test_dual_fft_packed_indices(self):
        idx0 = 5
        idx1 = 9
        log2_sub = 4
        rev0 = fft_bitrev_index_32(idx0, log2_sub)
        rev1 = fft_bitrev_index_32(idx1, log2_sub)

        packed = (idx1 << 16) | idx0
        brev_packed = ref_brev32(packed)
        extracted_rev0 = (brev_packed >> (32 - log2_sub)) & 0xF
        extracted_rev1 = (brev_packed >> (16 - log2_sub)) & 0xF
        self.assertEqual(extracted_rev0, rev0)
        self.assertEqual(extracted_rev1, rev1)

    def test_scalar_interpreter_fft_shader_loop(self):
        interp = ScalarBrevInterpreter(num_sgprs=32)
        log2_n = 4
        n_points = 1 << log2_n
        shift = 32 - log2_n
        base_addr = 0x1000
        element_stride = 8

        generated_addrs = []
        for i in range(n_points):
            interp.step_s_mov_b32(0, i)
            interp.step_s_brev_b32(1, {"k": "sgpr", "r": 0})
            interp.step_s_lshr_b32(2, {"k": "sgpr", "r": 1}, shift)
            rev_idx = interp.get_sgpr_u32(2)
            addr = base_addr + rev_idx * element_stride
            generated_addrs.append(addr)

        for i in range(n_points):
            expected_rev = fft_bitrev_index_32(i, log2_n)
            expected_addr = base_addr + expected_rev * element_stride
            self.assertEqual(generated_addrs[i], expected_addr)


class TestSpirvTranslationAndExecution(unittest.TestCase):
    """SPIR-V translation and mini-interpreter execution for bit-reversal operations."""

    def test_rdna2spirv_gen_s_brev_b32(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_brev_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpBitReverse %uint", emitted)
        self.assertIn("OpStore", emitted)

    def test_rdna2spirv_gen_s_brev_b64(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_brev_b64",
            "args": [
                {"k": "sgpr", "r": 0, "n": 2},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertEqual(emitted.count("OpBitReverse %uint"), 2)
        self.assertEqual(emitted.count("OpStore"), 2)

    def test_spirv_mini_interp_op_bitreverse_u32(self):
        test_patterns = [0, 1, 0x12345678, 0x80000000, 0x0000FFFF, 0x55555555, 0xFFFFFFFF]
        for val in test_patterns:
            with self.subTest(val=hex(val)):
                mod = "\n".join([
                    "OpCapability Shader",
                    "%glsl = OpExtInstImport 'GLSL.std.450'",
                    "OpMemoryModel Logical GLSL450",
                    "OpEntryPoint Fragment %main 'main' %out",
                    "%void = OpTypeVoid",
                    "%fn = OpTypeFunction %void",
                    "%uint = OpTypeInt 32 0",
                    "%_ptr_Output_uint = OpTypePointer Output %uint",
                    "%out = OpVariable %_ptr_Output_uint Output",
                    f"%cin = OpConstant %uint {val}",
                    "%main = OpFunction %void None %fn",
                    "%entry = OpLabel",
                    "%res = OpBitReverse %uint %cin",
                    "OpStore %out %res",
                    "OpReturn",
                    "OpFunctionEnd",
                ])
                res = si.run_fragment(mod)["%out"][0]
                self.assertEqual(res, ref_brev32(val))

    def test_spirv_mini_interp_op_bitreverse_u64(self):
        test_patterns = [
            0,
            1,
            0x8000000000000000,
            0x123456789ABCDEF0,
            0x00000000FFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
        ]
        for val in test_patterns:
            with self.subTest(val=hex(val)):
                mod = "\n".join([
                    "OpCapability Shader",
                    "OpCapability Int64",
                    "%glsl = OpExtInstImport 'GLSL.std.450'",
                    "OpMemoryModel Logical GLSL450",
                    "OpEntryPoint Fragment %main 'main' %out_lo %out_hi",
                    "%void = OpTypeVoid",
                    "%fn = OpTypeFunction %void",
                    "%uint = OpTypeInt 32 0",
                    "%ulong = OpTypeInt 64 0",
                    "%_ptr_Output_uint = OpTypePointer Output %uint",
                    "%out_lo = OpVariable %_ptr_Output_uint Output",
                    "%out_hi = OpVariable %_ptr_Output_uint Output",
                    f"%c64 = OpConstant %ulong {val}",
                    "%c32 = OpConstant %ulong 32",
                    "%main = OpFunction %void None %fn",
                    "%entry = OpLabel",
                    "%rev = OpBitReverse %ulong %c64",
                    "%lo = OpUConvert %uint %rev",
                    "%sh = OpShiftRightLogical %ulong %rev %c32",
                    "%hi = OpUConvert %uint %sh",
                    "OpStore %out_lo %lo",
                    "OpStore %out_hi %hi",
                    "OpReturn",
                    "OpFunctionEnd",
                ])
                r = si.run_fragment(mod)
                res_lo = r["%out_lo"][0]
                res_hi = r["%out_hi"][0]
                res_64 = join_dwords(res_lo, res_hi)
                self.assertEqual(res_64, ref_brev64(val))

    def test_synthetic_shader_pipeline_32(self):
        test_values = [0, 1, 0x12345678, 0x80000000, 0x0000FFFF, 0xFFFFFFFF]
        for val in test_values:
            with self.subTest(val=hex(val)):
                res = run_synthetic_s_brev_b32(val)
                self.assertEqual(res, ref_brev32(val))

    def test_synthetic_shader_pipeline_64(self):
        test_values = [
            0,
            1,
            0x8000000000000000,
            0x123456789ABCDEF0,
            0x00000000FFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                res = run_synthetic_s_brev_b64(val)
                self.assertEqual(res, ref_brev64(val))


if __name__ == "__main__":
    unittest.main()
