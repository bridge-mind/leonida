"""Semantic interpreter tests for RDNA2 scalar 16-bit pack instructions.

This suite tests the following SOP2 scalar ALU instructions:
  - s_pack_ll_b32_b16 D, S0, S1: D = (S1[15:0] << 16) | S0[15:0]
  - s_pack_lh_b32_b16 D, S0, S1: D = (S1[31:16] << 16) | S0[15:0]
  - s_pack_hh_b32_b16 D, S0, S1: D = (S1[31:16] << 16) | S0[31:16]

Hardware semantics (RDNA2 ISA / SOP2 encoding):
  - D[15:0] receives a 16-bit halfword selected from source S0:
      - 'l' in first position: selects S0[15:0] (bits 15..0)
      - 'h' in first position: selects S0[31:16] (bits 31..16)
  - D[31:16] receives a 16-bit halfword selected from source S1:
      - 'l' in second position: selects S1[15:0] (bits 15..0)
      - 'h' in second position: selects S1[31:16] (bits 31..16)
  - The scalar condition code (SCC) is NOT written and remains unaltered.

Shaders are translated via tools/rdna2spirv.py and executed using
tests/semantic/spirv_mini_interp.py against a plain-Python reference oracle.

Standard library only; zero game data, zero proprietary assets.
"""
from __future__ import annotations

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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_pack_ll_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_ll_b32_b16: D = (S1[15:0] << 16) | S0[15:0]."""
    return (((s1 & MASK16) << 16) | (s0 & MASK16)) & MASK32


def ref_pack_lh_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_lh_b32_b16: D = (S1[31:16] << 16) | S0[15:0]."""
    return ((s1 & (MASK16 << 16)) | (s0 & MASK16)) & MASK32


def ref_pack_hh_b32_b16(s0: int, s1: int) -> int:
    """Reference oracle for s_pack_hh_b32_b16: D = (S1[31:16] << 16) | S0[31:16]."""
    return ((s1 & (MASK16 << 16)) | ((s0 >> 16) & MASK16)) & MASK32


PACK_OPS = {
    "s_pack_ll_b32_b16": ref_pack_ll_b32_b16,
    "s_pack_lh_b32_b16": ref_pack_lh_b32_b16,
    "s_pack_hh_b32_b16": ref_pack_hh_b32_b16,
}


def ref_pack(op: str, s0: int, s1: int) -> int:
    """Dispatch to the corresponding reference oracle function."""
    if op not in PACK_OPS:
        raise ValueError(f"Unknown pack opcode: {op}")
    return PACK_OPS[op](s0, s1)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_pack_b32.s")
    dxbc_path = os.path.join(_TMP_DIR.name, "ps_salu_pack_b32.dxbc")
    with open(dxbc_path, "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        try:
            _TMP_DIR.cleanup()
        except Exception:
            pass
        _TMP_DIR = None


def run_assembly_lines(lines: list[str]) -> tuple[int, int]:
    """Assemble synthetic shader lines, run via mini interpreter, return (v0, v1)."""
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; test_salu_pack_b32\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    mrt0 = out["%out_mrt0"]
    return (mrt0[0], mrt0[1])


def run_pack_op(op: str, s0: int, s1: int, scc_in: int = 0) -> tuple[int, int]:
    """Execute pack operation with SCC pre-condition, return (result, scc_after)."""
    lines = [
        f"s_mov_b32 s2, {s0 & MASK32:#x}",
        f"s_mov_b32 s3, {s1 & MASK32:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"{op} s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ]
    return run_assembly_lines(lines)


class TestSaluPackB32ReferenceOracle(unittest.TestCase):
    """Unit tests verifying the Python reference oracle against hand-computed values."""

    def test_oracle_hand_computed_bit_patterns(self):
        s0 = 0x12345678  # S0: hi=0x1234, lo=0x5678
        s1 = 0x9ABCDEF0  # S1: hi=0x9ABC, lo=0xDEF0

        # s_pack_ll_b32_b16: hi=S1[15:0]=0xDEF0, lo=S0[15:0]=0x5678
        self.assertEqual(ref_pack_ll_b32_b16(s0, s1), 0xDEF05678)
        self.assertEqual(ref_pack("s_pack_ll_b32_b16", s0, s1), 0xDEF05678)

        # s_pack_lh_b32_b16: hi=S1[31:16]=0x9ABC, lo=S0[15:0]=0x5678
        self.assertEqual(ref_pack_lh_b32_b16(s0, s1), 0x9ABC5678)
        self.assertEqual(ref_pack("s_pack_lh_b32_b16", s0, s1), 0x9ABC5678)

        # s_pack_hh_b32_b16: hi=S1[31:16]=0x9ABC, lo=S0[31:16]=0x1234
        self.assertEqual(ref_pack_hh_b32_b16(s0, s1), 0x9ABC1234)
        self.assertEqual(ref_pack("s_pack_hh_b32_b16", s0, s1), 0x9ABC1234)

    def test_oracle_halfword_extraction_and_masking(self):
        # High half of s0 must not affect s_pack_ll or s_pack_lh
        val_a = ref_pack_ll_b32_b16(0x0000BEEF, 0x0000CAFE)
        val_b = ref_pack_ll_b32_b16(0xFFFFBEEF, 0x0000CAFE)
        self.assertEqual(val_a, val_b)
        self.assertEqual(val_a, 0xCAFEBEEF)

        # Low half of s0 must not affect s_pack_hh
        val_h1 = ref_pack_hh_b32_b16(0x11110000, 0x22220000)
        val_h2 = ref_pack_hh_b32_b16(0x1111FFFF, 0x22220000)
        self.assertEqual(val_h1, val_h2)
        self.assertEqual(val_h1, 0x22221111)

        # Low half of s1 must not affect s_pack_lh or s_pack_hh
        val_lh1 = ref_pack_lh_b32_b16(0x00003333, 0x44440000)
        val_lh2 = ref_pack_lh_b32_b16(0x00003333, 0x4444FFFF)
        self.assertEqual(val_lh1, val_lh2)
        self.assertEqual(val_lh1, 0x44443333)

    def test_oracle_32bit_clamping(self):
        # Oracle handles operands exceeding 32-bit representations cleanly
        res = ref_pack_ll_b32_b16(0x1_0000_1234, 0x2_0000_5678)
        self.assertEqual(res, 0x56781234)

    def test_oracle_unknown_opcode_rejection(self):
        with self.assertRaises(ValueError):
            ref_pack("s_pack_hl_b32_b16", 0, 0)


class TestSaluPackB32ExecutionBasics(unittest.TestCase):
    """Basic semantic execution tests validating rdna2spirv and spirv_mini_interp."""

    def test_basic_execution_matches_oracle(self):
        s0, s1 = 0xA1B2C3D4, 0xE5F60718
        for op in PACK_OPS:
            expected = ref_pack(op, s0, s1)
            got, _ = run_pack_op(op, s0, s1, scc_in=0)
            self.assertEqual(got, expected, f"Failed on op={op}")


class TestSaluPackB32PermutationsAndRouting(unittest.TestCase):
    """Exhaustive testing of halfword permutations, register routing, and constants."""

    def test_native_halfword_permutation_matrix(self):
        # S0: hi=0x1111, lo=0x2222; S1: hi=0x3333, lo=0x4444
        s0 = 0x11112222
        s1 = 0x33334444

        # s_pack_ll_b32_b16: S1[15:0] as hi, S0[15:0] as lo -> 0x44442222
        res_ll, _ = run_pack_op("s_pack_ll_b32_b16", s0, s1)
        self.assertEqual(res_ll, 0x44442222)
        self.assertEqual(res_ll, ref_pack_ll_b32_b16(s0, s1))

        # s_pack_lh_b32_b16: S1[31:16] as hi, S0[15:0] as lo -> 0x33332222
        res_lh, _ = run_pack_op("s_pack_lh_b32_b16", s0, s1)
        self.assertEqual(res_lh, 0x33332222)
        self.assertEqual(res_lh, ref_pack_lh_b32_b16(s0, s1))

        # s_pack_hh_b32_b16: S1[31:16] as hi, S0[31:16] as lo -> 0x33331111
        res_hh, _ = run_pack_op("s_pack_hh_b32_b16", s0, s1)
        self.assertEqual(res_hh, 0x33331111)
        self.assertEqual(res_hh, ref_pack_hh_b32_b16(s0, s1))

    def test_halfword_permutation_matrix_second_vector(self):
        # S0: hi=0xAAAA, lo=0x5555; S1: hi=0xCCCC, lo=0x3333
        s0 = 0xAAAA5555
        s1 = 0xCCCC3333

        res_ll, _ = run_pack_op("s_pack_ll_b32_b16", s0, s1)
        self.assertEqual(res_ll, 0x33335555)
        self.assertEqual(res_ll, ref_pack_ll_b32_b16(s0, s1))

        res_lh, _ = run_pack_op("s_pack_lh_b32_b16", s0, s1)
        self.assertEqual(res_lh, 0xCCCC5555)
        self.assertEqual(res_lh, ref_pack_lh_b32_b16(s0, s1))

        res_hh, _ = run_pack_op("s_pack_hh_b32_b16", s0, s1)
        self.assertEqual(res_hh, 0xCCCCAAAA)
        self.assertEqual(res_hh, ref_pack_hh_b32_b16(s0, s1))

    def test_synthesized_hl_packing_via_operand_swap(self):
        # RDNA2 lacks an s_pack_hl instruction. HL packing (S0[31:16] as hi, S1[15:0] as lo)
        # is synthesized by executing s_pack_lh with swapped source registers:
        # s_pack_lh D, S1, S0 places S1[15:0] in D[15:0] and S0[31:16] in D[31:16].
        s0 = 0x11112222
        s1 = 0x33334444
        lines = [
            f"s_mov_b32 s2, {s0:#x}",
            f"s_mov_b32 s3, {s1:#x}",
            "s_pack_lh_b32_b16 s0, s3, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ]
        got, _ = run_assembly_lines(lines)
        self.assertEqual(got, 0x11114444)

    def test_register_routing_different_sgpr_banks(self):
        # Test routing through non-contiguous and higher-index SGPRs
        lines = [
            "s_mov_b32 s6, 0x01234567",
            "s_mov_b32 s8, 0x89ABCDEF",
            "s_pack_ll_b32_b16 s10, s6, s8",
            "s_pack_lh_b32_b16 s12, s6, s8",
            "s_pack_hh_b32_b16 s14, s6, s8",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s12",
        ]
        res0, res1 = run_assembly_lines(lines)
        self.assertEqual(res0, 0xCDEF4567)
        self.assertEqual(res1, 0x89AB4567)

    def test_scc_preservation_across_all_opcodes(self):
        # Test that both SCC=0 and SCC=1 are unmodified by all three pack opcodes
        for op in PACK_OPS:
            for scc_in in (0, 1):
                with self.subTest(op=op, scc_in=scc_in):
                    _, scc_out = run_pack_op(op, 0x12345678, 0x9ABCDEF0, scc_in=scc_in)
                    self.assertEqual(scc_out, scc_in, f"SCC corrupted for {op}: expected {scc_in}, got {scc_out}")

    def test_literal_and_inline_constant_operands(self):
        lines = [
            "s_mov_b32 s2, 0x11112222",
            "s_pack_ll_b32_b16 s0, 0x1234, -1",
            "s_pack_lh_b32_b16 s1, s2, 0x56780000",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        res0, res1 = run_assembly_lines(lines)
        self.assertEqual(res0, 0xFFFF1234)
        self.assertEqual(res1, 0x56782222)

        lines2 = [
            "s_pack_hh_b32_b16 s0, 0xABCD0000, 0x12340000",
            "s_pack_ll_b32_b16 s1, 0, 1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        res2_0, res2_1 = run_assembly_lines(lines2)
        self.assertEqual(res2_0, 0x1234ABCD)
        self.assertEqual(res2_1, 0x00010000)

    def test_seeded_pseudo_random_combinations(self):
        rng = random.Random(0x5A1C)
        for i in range(15):
            op = rng.choice(list(PACK_OPS.keys()))
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            scc_in = i % 2
            expected = ref_pack(op, s0, s1)
            with self.subTest(i=i, op=op, s0=hex(s0), s1=hex(s1), scc=scc_in):
                got, scc_out = run_pack_op(op, s0, s1, scc_in=scc_in)
                self.assertEqual(got, expected)
                self.assertEqual(scc_out, scc_in)


class TestSaluPackB32BoundariesAliasingAndNegativeControls(unittest.TestCase):
    """Tests boundary extremes (zeros, ones, sign bits), register aliasing, and negative controls."""

    def test_boundary_extremes_zero_and_max(self):
        extremes = [0x00000000, 0x0000FFFF, 0xFFFF0000, 0xFFFFFFFF, 0x80007FFF, 0x7FFF8000]
        for s0 in extremes:
            for s1 in extremes:
                for op in PACK_OPS:
                    expected = ref_pack(op, s0, s1)
                    got, _ = run_pack_op(op, s0, s1)
                    self.assertEqual(got, expected, f"Failed at extreme s0={s0:#x}, s1={s1:#x}, op={op}")

    def test_register_aliasing_same_source_registers(self):
        # S0 and S1 refer to the exact same scalar register
        s0 = 0x12345678
        for op in PACK_OPS:
            expected = ref_pack(op, s0, s0)
            lines = [
                f"s_mov_b32 s2, {s0:#x}",
                f"{op} s0, s2, s2",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, 0",
            ]
            got, _ = run_assembly_lines(lines)
            self.assertEqual(got, expected, f"Failed aliasing test on op={op}")

    def test_register_aliasing_destination_equals_source(self):
        # Destination register is the same as source register S0 or S1
        lines = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x9ABCDEF0",
            "s_pack_ll_b32_b16 s2, s2, s3",  # D is S0 (s2)
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
        ]
        got, _ = run_assembly_lines(lines)
        self.assertEqual(got, 0xDEF05678)

        lines2 = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x9ABCDEF0",
            "s_pack_ll_b32_b16 s3, s2, s3",  # D is S1 (s3)
            "v_mov_b32 v0, s3",
            "v_mov_b32 v1, 0",
        ]
        got2, _ = run_assembly_lines(lines2)
        self.assertEqual(got2, 0xDEF05678)

    def test_negative_controls_invalid_instruction_or_malformed_encoding(self):
        # Verify that malformed / unknown instruction mnemonics raise translation/assembly errors
        bad_lines = [
            "s_pack_hl_b32_b16 s0, s2, s3",
        ]
        with self.assertRaises(Exception):
            run_assembly_lines(bad_lines)


if __name__ == "__main__":
    unittest.main()
