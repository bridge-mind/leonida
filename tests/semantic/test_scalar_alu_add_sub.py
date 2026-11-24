"""Semantic interpreter tests for RDNA2 64-bit scalar addition and subtraction with carry.

Covers:
  * s_add_u32  (D = S0 + S1, SCC = unsigned carry-out)
  * s_addc_u32 (D = S0 + S1 + SCC, SCC = unsigned carry-out)
  * s_sub_u32  (D = S0 - S1, SCC = unsigned borrow-out)
  * s_subb_u32 (D = S0 - S1 - SCC, SCC = unsigned borrow-out)
  * Chained 64-bit scalar addition:
      s_add_u32  d_lo, s0_lo, s1_lo
      s_addc_u32 d_hi, s0_hi, s1_hi
  * Chained 64-bit scalar subtraction:
      s_sub_u32  d_lo, s0_lo, s1_lo
      s_subb_u32 d_hi, s0_hi, s1_hi
  * SPIR-V translation lowering equivalence (OpIAddCarry / OpISubBorrow lowering in tools/rdna2spirv.py).

Run directly:
    python -I tests/semantic/test_scalar_alu_add_sub.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class ScalarALUInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU add/sub operations.

    Tracks 32-bit SGPRs and the 1-bit SCC (Scalar Condition Code) register.
    """

    def __init__(self, num_sgprs=106):
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self):
        self.sgpr = [0] * len(self.sgpr)
        self.scc = 0

    @staticmethod
    def eval_s_add_u32(s0, s1):
        """Execute s_add_u32: D = (S0 + S1) mod 2^32; SCC = (S0 + S1 >= 2^32)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        full = s0 + s1
        d = full & MASK32
        scc = 1 if full >= (1 << 32) else 0
        return d, scc

    @staticmethod
    def eval_s_addc_u32(s0, s1, scc_in):
        """Execute s_addc_u32: D = (S0 + S1 + SCC) mod 2^32; SCC = (S0 + S1 + SCC >= 2^32)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        cin = 1 if scc_in else 0
        full = s0 + s1 + cin
        d = full & MASK32
        scc = 1 if full >= (1 << 32) else 0
        return d, scc

    @staticmethod
    def eval_s_sub_u32(s0, s1):
        """Execute s_sub_u32: D = (S0 - S1) mod 2^32; SCC = (S0 < S1)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        d = (s0 - s1) & MASK32
        scc = 1 if s0 < s1 else 0
        return d, scc

    @staticmethod
    def eval_s_subb_u32(s0, s1, scc_in):
        """Execute s_subb_u32: D = (S0 - S1 - SCC) mod 2^32; SCC = (S0 < S1 + SCC)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        bin_ = 1 if scc_in else 0
        d = (s0 - s1 - bin_) & MASK32
        scc = 1 if s0 < (s1 + bin_) else 0
        return d, scc

    def step(self, insn):
        """Execute an instruction dictionary with 'op', 'dst', 's0', 's1'."""
        op = insn["op"]
        s0 = self._resolve_val(insn["s0"])
        s1 = self._resolve_val(insn["s1"])
        dst_idx = self._resolve_dst(insn["dst"])

        if op == "s_add_u32":
            res, scc = self.eval_s_add_u32(s0, s1)
        elif op == "s_addc_u32":
            res, scc = self.eval_s_addc_u32(s0, s1, self.scc)
        elif op == "s_sub_u32":
            res, scc = self.eval_s_sub_u32(s0, s1)
        elif op == "s_subb_u32":
            res, scc = self.eval_s_subb_u32(s0, s1, self.scc)
        elif op == "s_mov_b32":
            res, scc = s0 & MASK32, self.scc
        else:
            raise NotImplementedError(f"Unsupported op: {op}")

        if dst_idx is not None:
            self.sgpr[dst_idx] = res
        if op in ("s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32"):
            self.scc = scc
        return res, self.scc

    def run_sequence(self, instructions):
        """Execute a list of instruction dictionaries sequentially."""
        last_res = 0
        for insn in instructions:
            last_res, _ = self.step(insn)
        return last_res

    def _resolve_val(self, operand):
        if isinstance(operand, int):
            return operand & MASK32
        if isinstance(operand, dict):
            if operand.get("k") == "imm":
                return operand["v"] & MASK32
            if operand.get("k") == "sgpr":
                return self.sgpr[operand["r"]] & MASK32
        if isinstance(operand, str) and operand.startswith("s"):
            return self.sgpr[int(operand[1:])] & MASK32
        raise ValueError(f"Unknown operand format: {operand}")

    def _resolve_dst(self, operand):
        if isinstance(operand, dict) and operand.get("k") == "sgpr":
            return operand["r"]
        if isinstance(operand, str) and operand.startswith("s"):
            return int(operand[1:])
        if isinstance(operand, int):
            return operand
        return None


def reference_add_64(a_lo, a_hi, b_lo, b_hi):
    """Reference implementation of 64-bit integer addition."""
    a64 = ((a_hi & MASK32) << 32) | (a_lo & MASK32)
    b64 = ((b_hi & MASK32) << 32) | (b_lo & MASK32)
    sum64 = a64 + b64
    carry_out = 1 if sum64 >= (1 << 64) else 0
    res = sum64 & MASK64
    return res & MASK32, (res >> 32) & MASK32, carry_out


def reference_sub_64(a_lo, a_hi, b_lo, b_hi):
    """Reference implementation of 64-bit integer subtraction."""
    a64 = ((a_hi & MASK32) << 32) | (a_lo & MASK32)
    b64 = ((b_hi & MASK32) << 32) | (b_lo & MASK32)
    borrow_out = 1 if a64 < b64 else 0
    diff64 = (a64 - b64) & MASK64
    return diff64 & MASK32, (diff64 >> 32) & MASK32, borrow_out


def spirv_lowering_model_add_sub(op, x, y, scc_in):
    """Mathematical emulation of the SPIR-V lowering emitted by tools/rdna2spirv.py.

    Emulates OpIAddCarry / OpISubBorrow with OpCompositeExtract and OpBitwiseOr.
    """
    x = x & MASK32
    y = y & MASK32
    sub = op.startswith("s_sub")

    if not sub:
        full1 = x + y
        r1 = full1 & MASK32
        c1 = 1 if full1 >= (1 << 32) else 0
    else:
        full1 = x - y
        r1 = full1 & MASK32
        c1 = 1 if x < y else 0

    r, carry = r1, c1

    if op in ("s_addc_u32", "s_subb_u32"):
        cin = 1 if scc_in else 0
        if not sub:
            full2 = r1 + cin
            r2 = full2 & MASK32
            c2 = 1 if full2 >= (1 << 32) else 0
        else:
            full2 = r1 - cin
            r2 = full2 & MASK32
            c2 = 1 if r1 < cin else 0
        cc = c1 | c2
        r, carry = r2, cc

    return r, carry


class TestScalarAddU32(unittest.TestCase):
    """Unit tests for s_add_u32 semantics and condition code updates."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_zero_addition(self):
        res, scc = self.interp.eval_s_add_u32(0, 0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_identity_operand(self):
        for val in (1, 42, 0x12345678, 0xFFFFFFFF):
            with self.subTest(val=val):
                res1, scc1 = self.interp.eval_s_add_u32(val, 0)
                res2, scc2 = self.interp.eval_s_add_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)
                self.assertEqual(res2, val)
                self.assertEqual(scc2, 0)

    def test_basic_addition_no_carry(self):
        res, scc = self.interp.eval_s_add_u32(100, 200)
        self.assertEqual(res, 300)
        self.assertEqual(scc, 0)

    def test_carry_boundary_single_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(scc, 1)

    def test_carry_boundary_just_below_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 0)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

    def test_carry_boundary_high_bit_addition(self):
        res, scc = self.interp.eval_s_add_u32(0x80000000, 0x80000000)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(scc, 1)

    def test_max_inputs_carry_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

    def test_commutativity(self):
        pairs = [
            (0, 0),
            (12345, 67890),
            (0xFFFFFFFF, 2),
            (0x7FFFFFFF, 0x80000001),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                r1, s1 = self.interp.eval_s_add_u32(a, b)
                r2, s2 = self.interp.eval_s_add_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_interpreter_step_and_state(self):
        self.interp.step({"op": "s_mov_b32", "dst": "s0", "s0": 10, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s1", "s0": 25, "s1": 0})
        res, scc = self.interp.step({"op": "s_add_u32", "dst": "s2", "s0": "s0", "s1": "s1"})
        self.assertEqual(res, 35)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.sgpr[2], 35)
        self.assertEqual(self.interp.scc, 0)


class TestScalarAddcU32(unittest.TestCase):
    """Unit tests for s_addc_u32 semantics and carry-in propagation."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_carry_in_zero_matches_add(self):
        cases = [(0, 0), (10, 20), (0xFFFFFFFF, 1), (0xFFFFFFFF, 0xFFFFFFFF)]
        for s0, s1 in cases:
            with self.subTest(s0=s0, s1=s1):
                r_add, s_add = self.interp.eval_s_add_u32(s0, s1)
                r_addc, s_addc = self.interp.eval_s_addc_u32(s0, s1, 0)
                self.assertEqual(r_add, r_addc)
                self.assertEqual(s_add, s_addc)

    def test_carry_in_one_no_overflow(self):
        res, scc = self.interp.eval_s_addc_u32(100, 200, 1)
        self.assertEqual(res, 301)
        self.assertEqual(scc, 0)

    def test_carry_in_one_first_sum_overflows(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 5, 1)
        self.assertEqual(res, 5)
        self.assertEqual(scc, 1)

    def test_carry_in_one_second_sum_overflows(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 0, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

    def test_carry_in_one_split_overflow(self):
        res, scc = self.interp.eval_s_addc_u32(0x7FFFFFFF, 0x80000000, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

    def test_carry_in_all_ones(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_carry_in_only_zero_operands(self):
        res, scc = self.interp.eval_s_addc_u32(0, 0, 1)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 0)


class TestScalarSubU32(unittest.TestCase):
    """Unit tests for s_sub_u32 semantics and condition code updates."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_zero_subtraction(self):
        res, scc = self.interp.eval_s_sub_u32(0, 0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_identity_zero_subtrahend(self):
        for val in (1, 42, 0x12345678, 0xFFFFFFFF):
            with self.subTest(val=val):
                res, scc = self.interp.eval_s_sub_u32(val, 0)
                self.assertEqual(res, val)
                self.assertEqual(scc, 0)

    def test_identical_operands(self):
        for val in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
            with self.subTest(val=val):
                res, scc = self.interp.eval_s_sub_u32(val, val)
                self.assertEqual(res, 0)
                self.assertEqual(scc, 0)

    def test_basic_subtraction_no_borrow(self):
        res, scc = self.interp.eval_s_sub_u32(300, 100)
        self.assertEqual(res, 200)
        self.assertEqual(scc, 0)

    def test_borrow_boundary_zero_minus_one(self):
        res, scc = self.interp.eval_s_sub_u32(0, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_underflow(self):
        res, scc = self.interp.eval_s_sub_u32(10, 20)
        self.assertEqual(res, 0xFFFFFFF6)
        self.assertEqual(scc, 1)

    def test_max_underflow(self):
        res, scc = self.interp.eval_s_sub_u32(0, 0xFFFFFFFF)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

    def test_max_min_difference(self):
        res, scc = self.interp.eval_s_sub_u32(0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(scc, 0)


class TestScalarSubbU32(unittest.TestCase):
    """Unit tests for s_subb_u32 semantics and borrow-in propagation."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_borrow_in_zero_matches_sub(self):
        cases = [(0, 0), (30, 10), (0, 1), (10, 20), (0, 0xFFFFFFFF)]
        for s0, s1 in cases:
            with self.subTest(s0=s0, s1=s1):
                r_sub, s_sub = self.interp.eval_s_sub_u32(s0, s1)
                r_subb, s_subb = self.interp.eval_s_subb_u32(s0, s1, 0)
                self.assertEqual(r_sub, r_subb)
                self.assertEqual(s_sub, s_subb)

    def test_borrow_in_one_no_underflow(self):
        res, scc = self.interp.eval_s_subb_u32(50, 20, 1)
        self.assertEqual(res, 29)
        self.assertEqual(scc, 0)

    def test_borrow_in_one_exact_zero_result(self):
        res, scc = self.interp.eval_s_subb_u32(10, 9, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_borrow_in_one_first_sub_borrows(self):
        res, scc = self.interp.eval_s_subb_u32(0, 5, 1)
        self.assertEqual(res, 0xFFFFFFFA)
        self.assertEqual(scc, 1)

    def test_borrow_in_one_second_sub_borrows(self):
        res, scc = self.interp.eval_s_subb_u32(5, 5, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_in_one_zero_minus_zero(self):
        res, scc = self.interp.eval_s_subb_u32(0, 0, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_in_all_ones(self):
        res, scc = self.interp.eval_s_subb_u32(0xFFFFFFFF, 0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)


class Test64BitScalarArithmetic(unittest.TestCase):
    """Semantic tests for chained 64-bit scalar addition and subtraction with carry/borrow."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def run_add64(self, a_lo, a_hi, b_lo, b_hi):
        seq = [
            {"op": "s_mov_b32", "dst": "s0", "s0": a_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s1", "s0": a_hi, "s1": 0},
            {"op": "s_mov_b32", "dst": "s2", "s0": b_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s3", "s0": b_hi, "s1": 0},
            {"op": "s_add_u32", "dst": "s4", "s0": "s0", "s1": "s2"},
            {"op": "s_addc_u32", "dst": "s5", "s0": "s1", "s1": "s3"},
        ]
        self.interp.run_sequence(seq)
        return self.interp.sgpr[4], self.interp.sgpr[5], self.interp.scc

    def run_sub64(self, a_lo, a_hi, b_lo, b_hi):
        seq = [
            {"op": "s_mov_b32", "dst": "s0", "s0": a_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s1", "s0": a_hi, "s1": 0},
            {"op": "s_mov_b32", "dst": "s2", "s0": b_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s3", "s0": b_hi, "s1": 0},
            {"op": "s_sub_u32", "dst": "s4", "s0": "s0", "s1": "s2"},
            {"op": "s_subb_u32", "dst": "s5", "s0": "s1", "s1": "s3"},
        ]
        self.interp.run_sequence(seq)
        return self.interp.sgpr[4], self.interp.sgpr[5], self.interp.scc

    def test_add64_without_carry(self):
        r_lo, r_hi, scc = self.run_add64(0x00000002, 0x00000001, 0x00000004, 0x00000003)
        self.assertEqual(r_lo, 0x00000006)
        self.assertEqual(r_hi, 0x00000004)
        self.assertEqual(scc, 0)

    def test_add64_low_to_high_carry_propagation(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0x00000001, 0x00000001, 0x00000002)
        self.assertEqual(r_lo, 0x00000000)
        self.assertEqual(r_hi, 0x00000004)
        self.assertEqual(scc, 0)

    def test_add64_full_chain_overflow(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0x00000000)
        self.assertEqual(r_lo, 0x00000000)
        self.assertEqual(r_hi, 0x00000000)
        self.assertEqual(scc, 1)

    def test_add64_max_unsigned(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(r_lo, 0xFFFFFFFE)
        self.assertEqual(r_hi, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_sub64_without_borrow(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000005, 0x00000008, 0x00000002, 0x00000003)
        self.assertEqual(r_lo, 0x00000003)
        self.assertEqual(r_hi, 0x00000005)
        self.assertEqual(scc, 0)

    def test_sub64_low_to_high_borrow_propagation(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000000, 0x00000005, 0x00000001, 0x00000002)
        self.assertEqual(r_lo, 0xFFFFFFFF)
        self.assertEqual(r_hi, 0x00000002)
        self.assertEqual(scc, 0)

    def test_sub64_full_chain_underflow(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000000, 0x00000000, 0x00000001, 0x00000000)
        self.assertEqual(r_lo, 0xFFFFFFFF)
        self.assertEqual(r_hi, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_add64_sub64_roundtrip(self):
        test_vectors = [
            (0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321),
            (0xFFFFFFFF, 0x00000000, 0x00000001, 0x00000000),
            (0x00000000, 0x80000000, 0x00000000, 0x40000000),
            (0xCAFEBABE, 0xDEADBEEF, 0x11111111, 0x22222222),
        ]
        for a_lo, a_hi, b_lo, b_hi in test_vectors:
            with self.subTest(a_lo=a_lo, a_hi=a_hi, b_lo=b_lo, b_hi=b_hi):
                sum_lo, sum_hi, _ = self.run_add64(a_lo, a_hi, b_lo, b_hi)
                res_lo, res_hi, _ = self.run_sub64(sum_lo, sum_hi, b_lo, b_hi)
                self.assertEqual(res_lo, a_lo)
                self.assertEqual(res_hi, a_hi)

    def test_deterministic_pseudorandom_64bit_sweep(self):
        rng = random.Random(42)
        for i in range(250):
            a_lo = rng.getrandbits(32)
            a_hi = rng.getrandbits(32)
            b_lo = rng.getrandbits(32)
            b_hi = rng.getrandbits(32)

            exp_add_lo, exp_add_hi, exp_add_c = reference_add_64(a_lo, a_hi, b_lo, b_hi)
            got_add_lo, got_add_hi, got_add_c = self.run_add64(a_lo, a_hi, b_lo, b_hi)
            self.assertEqual(got_add_lo, exp_add_lo, f"Add iter {i} lo mismatch")
            self.assertEqual(got_add_hi, exp_add_hi, f"Add iter {i} hi mismatch")
            self.assertEqual(got_add_c, exp_add_c, f"Add iter {i} carry mismatch")

            exp_sub_lo, exp_sub_hi, exp_sub_b = reference_sub_64(a_lo, a_hi, b_lo, b_hi)
            got_sub_lo, got_sub_hi, got_sub_b = self.run_sub64(a_lo, a_hi, b_lo, b_hi)
            self.assertEqual(got_sub_lo, exp_sub_lo, f"Sub iter {i} lo mismatch")
            self.assertEqual(got_sub_hi, exp_sub_hi, f"Sub iter {i} hi mismatch")
            self.assertEqual(got_sub_b, exp_sub_b, f"Sub iter {i} borrow mismatch")


class TestRdna2SpirvLoweringSemantics(unittest.TestCase):
    """Tests cross-checking RDNA2 scalar ALU semantics against the SPIR-V lowering emitted by rdna2spirv.py."""

    def test_gen_spirv_emission_add_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_add_u32",
            "args": [{"k": "sgpr", "r": 4}, {"k": "imm", "v": 0xFFFFFFFF}, {"k": "imm", "v": 1}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertIn("OpIAddCarry %umul", body)
        self.assertIn("OpCompositeExtract %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_gen_spirv_emission_addc_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_addc_u32",
            "args": [{"k": "sgpr", "r": 5}, {"k": "imm", "v": 10}, {"k": "imm", "v": 20}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertEqual(body.count("OpIAddCarry %umul"), 2)
        self.assertIn("OpLoad %uint %scc", body)
        self.assertIn("OpBitwiseOr %uint", body)

    def test_gen_spirv_emission_sub_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_sub_u32",
            "args": [{"k": "sgpr", "r": 6}, {"k": "imm", "v": 50}, {"k": "imm", "v": 30}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertIn("OpISubBorrow %umul", body)
        self.assertIn("OpCompositeExtract %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_gen_spirv_emission_subb_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_subb_u32",
            "args": [{"k": "sgpr", "r": 7}, {"k": "imm", "v": 50}, {"k": "imm", "v": 30}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertEqual(body.count("OpISubBorrow %umul"), 2)
        self.assertIn("OpLoad %uint %scc", body)
        self.assertIn("OpBitwiseOr %uint", body)

    def test_spirv_model_equivalence_sweep(self):
        rng = random.Random(1337)
        for _ in range(200):
            x = rng.getrandbits(32)
            y = rng.getrandbits(32)
            cin = rng.choice([0, 1])

            # s_add_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_add_u32(x, y)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_add_u32", x, y, 0)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_addc_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_addc_u32(x, y, cin)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_addc_u32", x, y, cin)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_sub_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_sub_u32(x, y)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_sub_u32", x, y, 0)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_subb_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_subb_u32(x, y, cin)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_subb_u32", x, y, cin)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)


class TestAddressArithmeticUseCases(unittest.TestCase):
    """Simulates real-world 64-bit scalar address arithmetic patterns seen in shaders."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_s_getpc_b64_forward_displacement(self):
        pc_base = 0x00007FF8_10000000
        offset = 0x00000000_00002A40

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_add_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_addc_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base + offset)

    def test_s_getpc_b64_forward_displacement_spanning_4gb_boundary(self):
        pc_base = 0x00007FF8_FFFF8000
        offset = 0x00000000_00010000

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_add_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_addc_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base + offset)

    def test_backward_loop_edge_displacement(self):
        pc_base = 0x00007FF8_00001000
        offset = 0x00000000_00000500

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_sub_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_subb_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base - offset)

    def test_backward_loop_edge_spanning_4gb_boundary(self):
        pc_base = 0x00007FF8_00002000
        offset = 0x00000000_00008000

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_sub_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_subb_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base - offset)


if __name__ == "__main__":
    unittest.main()
