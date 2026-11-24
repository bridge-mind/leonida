"""Semantic interpreter tests for RDNA2 scalar zero-bit count instructions.

Covers:
  - s_bcnt0_i32_b32: D = count_zero_bits(S0.u32), SCC = (D != 0)
  - s_bcnt0_i32_b64: D = count_zero_bits(S0.u64), SCC = (D != 0)
  - s_ff0_i32_b32:   D = find_first_zero_bit(S0.u32), -1 if none (SCC unchanged)
  - s_flbit_i32_b32: D = count_leading_zeros(S0.u32), -1 if zero (SCC unchanged)

RDNA2 ISA rules:
  - s_bcnt0_i32_b32 counts the number of zero bits in a 32-bit scalar operand.
    D receives a value in range [0, 32]. SCC is set to 1 if D != 0, else 0
    (i.e. SCC is 0 if and only if S0 is 0xFFFFFFFF).
  - s_bcnt0_i32_b64 counts the number of zero bits in a 64-bit scalar operand
    (SGPR pair, 64-bit mask, or constant). D receives a value in range [0, 64].
    SCC is set to 1 if D != 0, else 0 (SCC is 0 if and only if S0 is all 1s).
  - s_ff0_i32_b32 scans the 32-bit scalar operand starting from LSB (bit 0)
    towards MSB (bit 31) for the first 0 bit. If found, D is the bit index
    [0, 31]. If no zero bit is found (S0 == 0xFFFFFFFF), D receives -1 (0xFFFFFFFF).
    SCC is not modified.
  - s_flbit_i32_b32 scans the 32-bit scalar operand starting from MSB (bit 31)
    down towards LSB (bit 0) for the first set bit (1), counting leading zero bits.
    D receives 31 - msb_index (range [0, 31]). If S0 == 0, D receives -1 (0xFFFFFFFF).
    SCC is not modified.

Lowering & Translation (tools/rdna2spirv.py):
  - s_bcnt0_i32_b32: bitwise NOT of low word, OpBitCount, sets %scc via OpSelect.
  - s_bcnt0_i32_b64: bitwise NOT of both halves, two OpBitCount, OpIAdd, sets %scc.
  - s_ff0_i32_b32: bitwise NOT of source, FindILsb (GLSL.std.450), no SCC store.
  - s_flbit_i32_b32: FindUMsb (GLSL.std.450), 31 - FindUMsb, select -1 if zero, no SCC store.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_bcnt0_b32(s0: int) -> tuple[int, int]:
    """Reference oracle for s_bcnt0_i32_b32: returns (count, scc)."""
    val = (s0 & MASK32) ^ MASK32
    count = bin(val).count("1")
    scc = 1 if count != 0 else 0
    return count, scc


def ref_bcnt0_b64(s0_64: int) -> tuple[int, int]:
    """Reference oracle for s_bcnt0_i32_b64: returns (count, scc)."""
    val = (s0_64 & MASK64) ^ MASK64
    count = bin(val).count("1")
    scc = 1 if count != 0 else 0
    return count, scc


def ref_ff0_b32(s0: int) -> int:
    """Reference oracle for s_ff0_i32_b32: returns first zero bit from LSB, or -1."""
    val = (s0 & MASK32) ^ MASK32
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


def ref_flbit_b32(s0: int) -> int:
    """Reference oracle for s_flbit_i32_b32: returns leading zeros before first 1, or -1."""
    val = s0 & MASK32
    if val == 0:
        return -1
    return 32 - val.bit_length()


class ScalarBcntInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar zero-bit count instructions.

    Models SGPR register file (s0-s105), condition code SCC, and special registers
    (m0, vcc, exec).
    """

    def __init__(self, num_sgprs: int = 106) -> None:
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.scc = 0
        self.m0 = 0
        self.vcc = 0
        self.exec = MASK64

    def reset(self) -> None:
        self.sgpr = [0] * self.num_sgprs
        self.scc = 0
        self.m0 = 0
        self.vcc = 0
        self.exec = MASK64

    def get_sgpr_u32(self, r: int) -> int:
        return self.sgpr[r] & MASK32

    def set_sgpr_u32(self, r: int, val: int) -> None:
        self.sgpr[r] = val & MASK32

    def get_sgpr_u64(self, r: int) -> int:
        lo = self.get_sgpr_u32(r)
        hi = self.get_sgpr_u32(r + 1)
        return ((hi & MASK32) << 32) | lo

    def set_sgpr_u64(self, r: int, val64: int) -> None:
        val64 &= MASK64
        self.set_sgpr_u32(r, val64 & MASK32)
        self.set_sgpr_u32(r + 1, (val64 >> 32) & MASK32)

    def resolve_src_u32(self, opnd: int | dict | str) -> int:
        """Resolve a 32-bit scalar source operand."""
        if isinstance(opnd, int):
            return opnd & MASK32
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "imm":
                return opnd["v"] & MASK32
            if k == "fimm":
                packed = struct.pack("<f", float(opnd["v"]))
                return struct.unpack("<I", packed)[0]
            if k == "sgpr":
                return self.get_sgpr_u32(opnd["r"])
            if k == "special":
                t = opnd.get("t")
                if t == "m0":
                    return self.m0 & MASK32
                if t == "vcc_lo":
                    return self.vcc & MASK32
                if t == "vcc_hi":
                    return (self.vcc >> 32) & MASK32
                if t == "exec_lo":
                    return self.exec & MASK32
                if t == "exec_hi":
                    return (self.exec >> 32) & MASK32
                if t == "scc":
                    return self.scc & 1
            raise ValueError(f"Unknown 32-bit dict operand: {opnd}")
        if isinstance(opnd, str):
            if opnd.startswith("s") and opnd[1:].isdigit():
                return self.get_sgpr_u32(int(opnd[1:]))
            if opnd == "m0":
                return self.m0 & MASK32
            if opnd == "vcc_lo":
                return self.vcc & MASK32
            if opnd == "vcc_hi":
                return (self.vcc >> 32) & MASK32
            if opnd == "exec_lo":
                return self.exec & MASK32
            if opnd == "exec_hi":
                return (self.exec >> 32) & MASK32
            if opnd == "scc":
                return self.scc & 1
        raise TypeError(f"Unsupported 32-bit operand format: {opnd}")

    def resolve_src_u64(self, opnd: int | dict | str) -> int:
        """Resolve a 64-bit scalar source operand."""
        if isinstance(opnd, int):
            return opnd & MASK64
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "imm":
                return opnd["v"] & MASK64
            if k == "sgpr":
                return self.get_sgpr_u64(opnd["r"])
            if k == "special":
                t = opnd.get("t")
                if t == "vcc":
                    return self.vcc & MASK64
                if t == "exec":
                    return self.exec & MASK64
            raise ValueError(f"Unknown 64-bit dict operand: {opnd}")
        if isinstance(opnd, str):
            if opnd.startswith("s[") and ":" in opnd:
                idx = int(opnd[2:].split(":")[0])
                return self.get_sgpr_u64(idx)
            if opnd == "vcc":
                return self.vcc & MASK64
            if opnd == "exec":
                return self.exec & MASK64
        raise TypeError(f"Unsupported 64-bit operand format: {opnd}")

    def resolve_dst(self, opnd: int | dict | str) -> int:
        """Resolve destination SGPR index."""
        if isinstance(opnd, int):
            return opnd
        if isinstance(opnd, dict) and opnd.get("k") == "sgpr":
            return opnd["r"]
        if isinstance(opnd, str) and opnd.startswith("s") and opnd[1:].isdigit():
            return int(opnd[1:])
        raise ValueError(f"Cannot resolve destination operand: {opnd}")

    def step(self, insn: dict) -> tuple[int, int]:
        """Execute one instruction dictionary.

        Accepts either:
          {"op": op, "args": [dst, src]}
        or:
          {"op": op, "dst": dst, "s0": src}
        """
        op = insn["op"]
        if "args" in insn:
            dst_opnd = insn["args"][0]
            src_opnd = insn["args"][1]
        else:
            dst_opnd = insn["dst"]
            src_opnd = insn.get("src0", insn.get("s0"))

        dst_r = self.resolve_dst(dst_opnd)

        if op == "s_bcnt0_i32_b32":
            s0 = self.resolve_src_u32(src_opnd)
            res, scc = ref_bcnt0_b32(s0)
            self.set_sgpr_u32(dst_r, res)
            self.scc = scc
            return res, self.scc

        if op == "s_bcnt0_i32_b64":
            s0_64 = self.resolve_src_u64(src_opnd)
            res, scc = ref_bcnt0_b64(s0_64)
            self.set_sgpr_u32(dst_r, res)
            self.scc = scc
            return res, self.scc

        if op == "s_ff0_i32_b32":
            s0 = self.resolve_src_u32(src_opnd)
            res = ref_ff0_b32(s0)
            self.set_sgpr_u32(dst_r, res & MASK32)
            # SCC is not modified by s_ff0_i32_b32
            return res & MASK32, self.scc

        if op == "s_flbit_i32_b32":
            s0 = self.resolve_src_u32(src_opnd)
            res = ref_flbit_b32(s0)
            self.set_sgpr_u32(dst_r, res & MASK32)
            # SCC is not modified by s_flbit_i32_b32
            return res & MASK32, self.scc

        if op == "s_mov_b32":
            val = self.resolve_src_u32(src_opnd)
            self.set_sgpr_u32(dst_r, val)
            return val, self.scc

        if op == "s_mov_b64":
            val64 = self.resolve_src_u64(src_opnd)
            self.set_sgpr_u64(dst_r, val64)
            return val64 & MASK32, self.scc

        raise NotImplementedError(f"Unsupported instruction op: {op}")

    def run_sequence(self, instructions: list[dict]) -> tuple[int, int]:
        """Execute a sequence of instruction dictionaries sequentially."""
        last_res = 0
        for insn in instructions:
            last_res, _ = self.step(insn)
        return last_res, self.scc


class TestScalarBcntReferenceOracles(unittest.TestCase):
    """Unit tests for the standalone mathematical reference models."""

    def test_bcnt0_b32_all_single_zeros(self):
        for bit in range(32):
            pattern = (~(1 << bit)) & MASK32
            count, scc = ref_bcnt0_b32(pattern)
            self.assertEqual(count, 1, f"Expected 1 zero bit for pattern with bit {bit} cleared")
            self.assertEqual(scc, 1)

    def test_bcnt0_b32_all_single_ones(self):
        for bit in range(32):
            pattern = (1 << bit) & MASK32
            count, scc = ref_bcnt0_b32(pattern)
            self.assertEqual(count, 31, f"Expected 31 zero bits for single bit {bit} set")
            self.assertEqual(scc, 1)

    def test_bcnt0_b32_extrema_and_patterns(self):
        self.assertEqual(ref_bcnt0_b32(0), (32, 1))
        self.assertEqual(ref_bcnt0_b32(MASK32), (0, 0))
        self.assertEqual(ref_bcnt0_b32(0x55555555), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0xAAAAAAAA), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x0000FFFF), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0xFFFF0000), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x0F0F0F0F), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x000000FF), (24, 1))

    def test_bcnt0_b64_all_single_zeros(self):
        for bit in range(64):
            pattern = (~(1 << bit)) & MASK64
            count, scc = ref_bcnt0_b64(pattern)
            self.assertEqual(count, 1, f"Expected 1 zero bit for 64-bit pattern with bit {bit} cleared")
            self.assertEqual(scc, 1)

    def test_bcnt0_b64_all_single_ones(self):
        for bit in range(64):
            pattern = (1 << bit) & MASK64
            count, scc = ref_bcnt0_b64(pattern)
            self.assertEqual(count, 63, f"Expected 63 zero bits for 64-bit single bit {bit} set")
            self.assertEqual(scc, 1)

    def test_bcnt0_b64_extrema_and_patterns(self):
        self.assertEqual(ref_bcnt0_b64(0), (64, 1))
        self.assertEqual(ref_bcnt0_b64(MASK64), (0, 0))
        self.assertEqual(ref_bcnt0_b64(0x5555555555555555), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0xAAAAAAAAAAAAAAAA), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0x00000000FFFFFFFF), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0xFFFFFFFF00000000), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0x0000FFFF0000FFFF), (32, 1))

    def test_ff0_b32_exhaustive_single_zeros(self):
        for bit in range(32):
            pattern = (~(1 << bit)) & MASK32
            self.assertEqual(ref_ff0_b32(pattern), bit, f"First zero bit should be {bit}")

    def test_ff0_b32_dense_patterns(self):
        self.assertEqual(ref_ff0_b32(0), 0)
        self.assertEqual(ref_ff0_b32(1), 1)
        self.assertEqual(ref_ff0_b32(0b11), 2)
        self.assertEqual(ref_ff0_b32(0b111), 3)
        self.assertEqual(ref_ff0_b32(0b1111), 4)
        self.assertEqual(ref_ff0_b32(0x7FFFFFFF), 31)
        self.assertEqual(ref_ff0_b32(MASK32), -1)
        self.assertEqual(ref_ff0_b32(0xAAAAAAA0), 0)
        self.assertEqual(ref_ff0_b32(0xAAAAAAA1), 1)
        self.assertEqual(ref_ff0_b32(0xAAAAAAA3), 2)

    def test_flbit_b32_exhaustive_powers_of_two(self):
        for bit in range(32):
            pattern = 1 << bit
            expected_leading_zeros = 31 - bit
            self.assertEqual(
                ref_flbit_b32(pattern),
                expected_leading_zeros,
                f"Leading zeros for 1 << {bit} should be {expected_leading_zeros}",
            )

    def test_flbit_b32_dense_patterns(self):
        self.assertEqual(ref_flbit_b32(0), -1)
        self.assertEqual(ref_flbit_b32(0x80000000), 0)
        self.assertEqual(ref_flbit_b32(0xC0000000), 0)
        self.assertEqual(ref_flbit_b32(0xFFFFFFFF), 0)
        self.assertEqual(ref_flbit_b32(0x7FFFFFFF), 1)
        self.assertEqual(ref_flbit_b32(0x40000000), 1)
        self.assertEqual(ref_flbit_b32(0x3FFFFFFF), 2)
        self.assertEqual(ref_flbit_b32(0x20000000), 2)
        self.assertEqual(ref_flbit_b32(0x00000001), 31)
        self.assertEqual(ref_flbit_b32(0x00000003), 30)

    def test_mathematical_duality_and_invariants(self):
        for v in (0, 1, 42, 0x12345678, 0x55555555, 0x7FFFFFFF, MASK32):
            pop32 = bin(v & MASK32).count("1")
            zeros32, _ = ref_bcnt0_b32(v)
            self.assertEqual(pop32 + zeros32, 32)

            pop64 = bin(v & MASK64).count("1")
            zeros64, _ = ref_bcnt0_b64(v)
            self.assertEqual(pop64 + zeros64, 64)

            # ff0(x) finds least significant 0; for any pattern with at least one 0:
            ff0 = ref_ff0_b32(v)
            if v == MASK32:
                self.assertEqual(ff0, -1)
            else:
                self.assertTrue(0 <= ff0 < 32)
                self.assertEqual((v >> ff0) & 1, 0)
                # Lower bits before ff0 must all be 1
                lower_mask = (1 << ff0) - 1
                self.assertEqual(v & lower_mask, lower_mask)


class TestScalarBcntInterpreterExecution(unittest.TestCase):
    """Semantic tests executing instructions through ScalarBcntInterpreter."""

    def setUp(self):
        self.interp = ScalarBcntInterpreter()

    def test_s_bcnt0_i32_b32_registers(self):
        self.interp.set_sgpr_u32(1, 0x0000FFFF)
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": {"k": "sgpr", "r": 0},
            "src0": {"k": "sgpr", "r": 1},
        })
        self.assertEqual(res, 16)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(0), 16)
        self.assertEqual(self.interp.scc, 1)

    def test_s_bcnt0_i32_b32_in_place(self):
        self.interp.set_sgpr_u32(5, 0x0)
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s5",
            "src0": "s5",
        })
        self.assertEqual(res, 32)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(5), 32)

    def test_s_bcnt0_i32_b32_immediate_and_float_imm(self):
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s0",
            "src0": {"k": "imm", "v": 0xFFFFFFFF},
        })
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        # 1.0f in IEEE-754: 0x3F800000 -> 0011 1111 1000 ... (seven 1s, twenty-five 0s)
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s0",
            "src0": {"k": "fimm", "v": 1.0},
        })
        self.assertEqual(res, 25)
        self.assertEqual(scc, 1)

    def test_s_bcnt0_i32_b32_special_registers(self):
        self.interp.m0 = 0b11110000
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s0",
            "src0": {"k": "special", "t": "m0"},
        })
        self.assertEqual(res, 28)
        self.assertEqual(scc, 1)

        self.interp.exec = 0x00000000FFFFFFFF
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s0",
            "src0": {"k": "special", "t": "exec_lo"},
        })
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b32",
            "dst": "s0",
            "src0": {"k": "special", "t": "exec_hi"},
        })
        self.assertEqual(res, 32)
        self.assertEqual(scc, 1)

    def test_s_bcnt0_i32_b32_scc_transitions(self):
        # Start with SCC = 1
        self.interp.scc = 1
        # 0xFFFFFFFF has 0 zero bits -> sets SCC to 0
        self.interp.step({"op": "s_bcnt0_i32_b32", "dst": "s0", "src0": 0xFFFFFFFF})
        self.assertEqual(self.interp.scc, 0)

        # Non-0xFFFFFFFF has >0 zero bits -> sets SCC to 1
        self.interp.step({"op": "s_bcnt0_i32_b32", "dst": "s0", "src0": 0x7FFFFFFF})
        self.assertEqual(self.interp.scc, 1)

    def test_s_bcnt0_i32_b64_sgpr_pairs(self):
        # s[2:3] = 0x00000000_FFFFFFFF
        self.interp.set_sgpr_u32(2, 0xFFFFFFFF)
        self.interp.set_sgpr_u32(3, 0x00000000)
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b64",
            "dst": {"k": "sgpr", "r": 0},
            "src0": {"k": "sgpr", "r": 2, "n": 2},
        })
        self.assertEqual(res, 32)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(0), 32)

    def test_s_bcnt0_i32_b64_in_place_dst_is_low_sgpr(self):
        # s[4:5] = 0
        self.interp.set_sgpr_u64(4, 0)
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b64",
            "dst": "s4",
            "src0": "s[4:5]",
        })
        self.assertEqual(res, 64)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(4), 64)

    def test_s_bcnt0_i32_b64_lane_masks(self):
        self.interp.exec = 0xAAAAAAAAAAAAAAAA
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b64",
            "dst": "s0",
            "src0": {"k": "special", "t": "exec"},
        })
        self.assertEqual(res, 32)
        self.assertEqual(scc, 1)

        self.interp.vcc = MASK64
        res, scc = self.interp.step({
            "op": "s_bcnt0_i32_b64",
            "dst": "s0",
            "src0": "vcc",
        })
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_s_ff0_i32_b32_registers(self):
        self.interp.set_sgpr_u32(1, 0b10111)  # bit 3 is first zero
        res, _ = self.interp.step({
            "op": "s_ff0_i32_b32",
            "dst": "s0",
            "src0": "s1",
        })
        self.assertEqual(res, 3)
        self.assertEqual(self.interp.get_sgpr_u32(0), 3)

    def test_s_ff0_i32_b32_all_ones_returns_minus_one(self):
        res, _ = self.interp.step({
            "op": "s_ff0_i32_b32",
            "dst": "s0",
            "src0": 0xFFFFFFFF,
        })
        self.assertEqual(res, MASK32)  # -1 represented as unsigned 32-bit
        self.assertEqual(self.interp.get_sgpr_u32(0), 0xFFFFFFFF)

    def test_s_ff0_i32_b32_scc_invariance(self):
        # s_ff0_i32_b32 must preserve preexisting SCC
        for initial_scc in (0, 1):
            self.interp.scc = initial_scc
            self.interp.step({"op": "s_ff0_i32_b32", "dst": "s0", "src0": 0x7FFFFFFF})
            self.assertEqual(
                self.interp.scc,
                initial_scc,
                f"s_ff0_i32_b32 mutated SCC when initial was {initial_scc}",
            )

            # Test also when result is -1 (input is 0xFFFFFFFF)
            self.interp.scc = initial_scc
            self.interp.step({"op": "s_ff0_i32_b32", "dst": "s0", "src0": 0xFFFFFFFF})
            self.assertEqual(
                self.interp.scc,
                initial_scc,
                f"s_ff0_i32_b32 mutated SCC on all-ones when initial was {initial_scc}",
            )

    def test_s_flbit_i32_b32_registers(self):
        self.interp.set_sgpr_u32(1, 0x00020000)  # bit 17 set -> 31 - 17 = 14 leading zeros
        res, _ = self.interp.step({
            "op": "s_flbit_i32_b32",
            "dst": "s0",
            "src0": "s1",
        })
        self.assertEqual(res, 14)
        self.assertEqual(self.interp.get_sgpr_u32(0), 14)

    def test_s_flbit_i32_b32_zero_returns_minus_one(self):
        res, _ = self.interp.step({
            "op": "s_flbit_i32_b32",
            "dst": "s0",
            "src0": 0x0,
        })
        self.assertEqual(res, MASK32)  # -1 as uint32
        self.assertEqual(self.interp.get_sgpr_u32(0), 0xFFFFFFFF)

    def test_s_flbit_i32_b32_scc_invariance(self):
        # s_flbit_i32_b32 must preserve preexisting SCC
        for initial_scc in (0, 1):
            self.interp.scc = initial_scc
            self.interp.step({"op": "s_flbit_i32_b32", "dst": "s0", "src0": 0x80000000})
            self.assertEqual(
                self.interp.scc,
                initial_scc,
                f"s_flbit_i32_b32 mutated SCC when initial was {initial_scc}",
            )

            # Test also when input is 0 (returns -1)
            self.interp.scc = initial_scc
            self.interp.step({"op": "s_flbit_i32_b32", "dst": "s0", "src0": 0x0})
            self.assertEqual(
                self.interp.scc,
                initial_scc,
                f"s_flbit_i32_b32 mutated SCC on zero input when initial was {initial_scc}",
            )

    def test_chained_instruction_sequences(self):
        # Sequence: Priority search and mask clearing loop emulation
        # 1. Initialize s1 = 0b10011 (bits 2, 5, 6, 7 are zero)
        # 2. Find first zero with s_ff0_i32_b32 -> should find bit 2
        # 3. Count zeros with s_bcnt0_i32_b32 -> 32 - 3 = 29 zeros, SCC = 1
        # 4. Find leading zeros with s_flbit_i32_b32 on s1 -> bit 4 is MSB set, 31 - 4 = 27
        seq = [
            {"op": "s_mov_b32", "dst": "s1", "src0": 0b10011},
            {"op": "s_ff0_i32_b32", "dst": "s2", "src0": "s1"},
            {"op": "s_bcnt0_i32_b32", "dst": "s3", "src0": "s1"},
            {"op": "s_flbit_i32_b32", "dst": "s4", "src0": "s1"},
        ]
        self.interp.run_sequence(seq)
        self.assertEqual(self.interp.get_sgpr_u32(2), 2)
        self.assertEqual(self.interp.get_sgpr_u32(3), 29)
        self.assertEqual(self.interp.scc, 1)
        self.assertEqual(self.interp.get_sgpr_u32(4), 27)


class TestRdna2SpirvScalarBcntEmission(unittest.TestCase):
    """Verifies that rdna2spirv generates conforming SPIR-V for all zero-counting ops."""

    def test_s_bcnt0_i32_b32_emission_sgpr(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_bcnt0_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b32_emission_imm(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_bcnt0_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x12345678}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c12345678", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b32_emission_lane_mask(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_bcnt0_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "special", "t": "vcc_lo"}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b64_emission_sgpr_pair(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_bcnt0_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 2, "n": 2}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertEqual(emitted.count("OpNot %uint"), 2)
        self.assertEqual(emitted.count("OpBitCount %uint"), 2)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b64_emission_exec_mask(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_bcnt0_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "special", "t": "exec"}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff0_i32_b32_emission_sgpr(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_ff0_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpStore %scc", emitted)

    def test_s_ff0_i32_b32_emission_imm(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_ff0_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 7}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c7", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpStore %scc", emitted)

    def test_s_flbit_i32_b32_emission_sgpr(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_flbit_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "sgpr", "r": 1}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("FindUMsb", emitted)
        self.assertIn("OpISub %uint %c1f", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("%cffffffff", emitted)
        self.assertNotIn("OpStore %scc", emitted)

    def test_s_flbit_i32_b32_emission_imm(self):
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_flbit_i32_b32",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x40000000}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("FindUMsb", emitted)
        self.assertIn("OpISub %uint %c1f", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertNotIn("OpStore %scc", emitted)


if __name__ == "__main__":
    unittest.main()
