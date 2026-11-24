"""Semantic interpreter tests for structured conditional scalar branching (s_cbranch_scc0, s_cbranch_scc1).

Tests the ISA semantics, scalar execution models, and SPIR-V translator emission for:
  - s_cbranch_scc0: branch if SCC == 0 (Scalar Condition Code is cleared)
  - s_cbranch_scc1: branch if SCC != 0 (Scalar Condition Code is set)

Covers:
  1. Pure mathematical reference oracle for SCC condition evaluation and signed
     two's-complement displacement arithmetic (next_pc + 4 * simm16).
  2. ScalarInterpreter modeling RDNA2 SGPRs, the SCC status flag, scalar comparisons,
     scalar ALU status updates, and graph basic block control flow.
  3. SPIR-V translator emission verification via tools/rdna2spirv.py branch_condition():
     validating OpLoad %scc, OpIEqual / OpINotEqual against constant zero, and verifying
     that SCC branching remains lane-uniform without subgroup voting overhead.
  4. End-to-end SPIR-V assembly execution via StructuredBranchInterp modeling:
     - Forward skip patterns (if-then without else)
     - Symmetric if-then-else diamond control flow
     - Iterative countdown loops with backward conditional branches
     - Two-level nested conditional branching covering all permutations
     - Multi-way conditional dispatch ladders (switch / ladder)
  5. Invariants and duality verification: scc0 and scc1 are strict logical complements.
  6. Negative controls and fault rejection: sensitivity to inverted conditions,
     out-of-bounds branch targets, infinite loops, and invalid opcodes.

Run:
    python -I tests/semantic/test_control_flow_branches.py
    python -m unittest discover -s tests/semantic -p "test_control_flow_branches.py" -v
"""
from __future__ import annotations

import gc
import os
import random
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")
_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpName", "OpSource",
    "OpSelectionMerge", "OpLoopMerge",
})


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode or invalid construct."""


class BranchExecutionLimitError(RuntimeError):
    """Execution step limit reached, detecting an infinite loop or runaway branch."""


# -----------------------------------------------------------------------------
# 1. Reference Semantics (AMD RDNA2 GFX10.3 ISA: Scalar Control Flow & SCC)
# -----------------------------------------------------------------------------

def eval_scc0(scc: int) -> bool:
    """s_cbranch_scc0: taken iff the SCC register bit is zero."""
    return (scc & 1) == 0


def eval_scc1(scc: int) -> bool:
    """s_cbranch_scc1: taken iff the SCC register bit is one."""
    return (scc & 1) != 0


def compute_branch_target(pc: int, simm16: int, insn_len: int = 4) -> int:
    """Calculates branch destination address given branch instruction PC and 16-bit signed displacement.

    Displacement arithmetic:
        next_pc = pc + insn_len
        target_pc = next_pc + 4 * simm16
    """
    v = simm16 & 0xFFFF
    signed_imm = v - 0x10000 if (v & 0x8000) else v
    next_pc = pc + insn_len
    return next_pc + 4 * signed_imm


def evaluate_scc_branch_condition(op: str, scc: int) -> bool:
    """Evaluates whether an SCC conditional branch is taken given the SCC register value."""
    if op == "s_cbranch_scc0":
        return eval_scc0(scc)
    if op == "s_cbranch_scc1":
        return eval_scc1(scc)
    raise ValueError(f"Unknown branch opcode: {op}")


# -----------------------------------------------------------------------------
# 2. Pure Python Reference Scalar Interpreter
# -----------------------------------------------------------------------------

class ScalarInterpreter:
    """Simulates RDNA2 scalar execution across basic blocks.

    Models:
      - 106 SGPRs (s0..s105).
      - 1-bit SCC condition flag.
      - Scalar comparisons (s_cmp_*), bit tests (s_bitcmp*), scalar ALU (s_and, s_or, s_add, s_sub, s_min, s_max).
      - Conditional and unconditional scalar branching (s_cbranch_scc0, s_cbranch_scc1, s_branch).
      - Basic block dispatch, program counter, and step limit detection.
    """

    def __init__(self, initial_scc: int = 0):
        self.sgprs = [0] * 106
        self.scc = initial_scc & 1
        self.trace: list[str] = []

    def set_sgpr(self, reg: int | str, val: int):
        r = self._resolve_reg(reg)
        self.sgprs[r] = val & MASK32

    def get_sgpr(self, reg: int | str) -> int:
        r = self._resolve_reg(reg)
        return self.sgprs[r] & MASK32

    def _resolve_reg(self, reg: int | str) -> int:
        if isinstance(reg, str):
            if reg.startswith("s") and reg[1:].isdigit():
                return int(reg[1:])
            raise ValueError(f"Invalid register name: {reg}")
        return int(reg)

    def _resolve_src(self, operand: int | str, default_reg: bool = False) -> int:
        if isinstance(operand, str):
            if operand.startswith("s") and operand[1:].isdigit():
                return self.get_sgpr(int(operand[1:]))
            raise ValueError(f"Unknown symbolic operand: {operand}")
        if default_reg:
            return self.get_sgpr(operand)
        return operand & MASK32

    def run_program(self, blocks: dict[str, list], start_block: str, max_steps: int = 5000):
        cur: str | None = start_block
        steps = 0
        self.trace = []

        while cur is not None and steps < max_steps:
            steps += 1
            self.trace.append(cur)
            block = blocks.get(cur)
            if block is None:
                raise ValueError(f"Unknown block label: {cur}")

            next_block: str | None = None
            for ins in block:
                if isinstance(ins, str):
                    next_block = ins
                    break

                op = ins[0]
                args = ins[1:]

                if op == "s_endpgm":
                    return

                elif op == "s_branch":
                    next_block = args[0]
                    break

                elif op == "s_cbranch_scc0":
                    target = args[0]
                    if eval_scc0(self.scc):
                        next_block = target
                        break

                elif op == "s_cbranch_scc1":
                    target = args[0]
                    if eval_scc1(self.scc):
                        next_block = target
                        break

                elif op == "s_mov_b32":
                    dst_s, val = args
                    r = self._resolve_reg(dst_s)
                    v = self._resolve_src(val, default_reg=False)
                    self.set_sgpr(r, v)

                elif op == "s_cmp_eq_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 == v1) else 0

                elif op == "s_cmp_lg_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 != v1) else 0

                elif op == "s_cmp_gt_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 > v1) else 0

                elif op == "s_cmp_ge_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 >= v1) else 0

                elif op == "s_cmp_lt_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 < v1) else 0

                elif op == "s_cmp_le_u32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    self.scc = 1 if (v0 <= v1) else 0

                elif op == "s_bitcmp0_b32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    bit_idx = self._resolve_src(s1, default_reg=False) & 31
                    bit = (v0 >> bit_idx) & 1
                    self.scc = 1 if (bit == 0) else 0

                elif op == "s_bitcmp1_b32":
                    s0, s1 = args
                    v0 = self._resolve_src(s0, default_reg=True)
                    bit_idx = self._resolve_src(s1, default_reg=False) & 31
                    bit = (v0 >> bit_idx) & 1
                    self.scc = 1 if (bit == 1) else 0

                elif op == "s_and_b32":
                    dst_s, s0, s1 = args
                    r = self._resolve_reg(dst_s)
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    res = (v0 & v1) & MASK32
                    self.set_sgpr(r, res)
                    self.scc = 1 if (res != 0) else 0

                elif op == "s_or_b32":
                    dst_s, s0, s1 = args
                    r = self._resolve_reg(dst_s)
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    res = (v0 | v1) & MASK32
                    self.set_sgpr(r, res)
                    self.scc = 1 if (res != 0) else 0

                elif op == "s_add_u32":
                    dst_s, s0, s1 = args
                    r = self._resolve_reg(dst_s)
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    s = v0 + v1
                    self.set_sgpr(r, s & MASK32)
                    self.scc = 1 if (s > MASK32) else 0

                elif op == "s_sub_u32":
                    dst_s, s0, s1 = args
                    r = self._resolve_reg(dst_s)
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    res = (v0 - v1) & MASK32
                    self.set_sgpr(r, res)
                    self.scc = 1 if (v0 < v1) else 0

                elif op == "s_min_u32":
                    dst_s, s0, s1 = args
                    r = self._resolve_reg(dst_s)
                    v0 = self._resolve_src(s0, default_reg=True)
                    v1 = self._resolve_src(s1, default_reg=False)
                    if v0 < v1:
                        self.set_sgpr(r, v0)
                        self.scc = 1
                    else:
                        self.set_sgpr(r, v1)
                        self.scc = 0

                else:
                    raise NotImplementedError(f"Unsupported instruction: {op}")

            if next_block is None:
                cur = block[-1] if isinstance(block[-1], str) else None
            else:
                cur = next_block

        if steps >= max_steps:
            raise BranchExecutionLimitError(f"Execution step limit {max_steps} exceeded: potential infinite loop")


# -----------------------------------------------------------------------------
# 3. SPIR-V Structured Control Flow Mini-Interpreter
# -----------------------------------------------------------------------------

class StructuredBranchInterp:
    """Interprets structured SPIR-V assembly emitted by tools/rdna2spirv.py for shaders with branches."""

    def __init__(self, spvasm: str):
        self.spvasm = spvasm
        self.types: dict[str, tuple] = {}
        self.constants: dict[str, int | float | bool | tuple | list] = {}
        self.ptr_types: dict[str, tuple[str, str]] = {}
        self.var_class: dict[str, str] = {}
        self.var_box: dict[str, list] = {}
        self.pointers: dict[str, tuple] = {}
        self.outputs: list[str] = []
        self.insns: list[tuple[str | None, str, list[str], str]] = []
        self.labels: dict[str, int] = {}
        self._parse()

    def zero(self, name: str):
        k = self.types[name]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return 0.0
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(self.zero(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [self.zero(k[1]) for _ in range(k[2])]
        return 0

    def _parse(self):
        in_body = False
        idx = 0
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                continue
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                if op == "OpLabel":
                    self.labels[res] = idx
                self.insns.append((res, op, a, line))
                idx += 1
                continue
            if op == "OpFunction":
                in_body = True
                continue
            if op == "OpTypeInt":
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeFloat":
                self.types[res] = ("float",)
            elif op == "OpTypeBool":
                self.types[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                self.types[res] = ("arr", a[0], int(self.constants[a[1]]))
            elif op == "OpTypeStruct":
                self.types[res] = ("struct", a)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                k = self.types[a[0]]
                if k[0] == "int":
                    self.constants[res] = int(a[1], 0) & MASK32
                elif k[0] == "float":
                    self.constants[res] = float(a[1])
            elif op == "OpConstantTrue":
                self.constants[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = False
            elif op == "OpConstantNull":
                self.constants[res] = self.zero(a[0])
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                self.var_class[res] = cls
                if cls == "Output":
                    self.outputs.append(res)
                    self.var_box[res] = [self.zero(pointee)]
                    self.pointers[res] = (self.var_box[res], 0)
                elif cls == "PushConstant":
                    self.var_box[res] = [None]
                    self.pointers[res] = ("pc", None)

    def run(self, max_steps: int = 20000) -> dict[str, tuple]:
        values = dict(self.constants)
        var_box = dict(self.var_box)
        pointers = dict(self.pointers)

        def val(x: str):
            if x in values:
                return values[x]
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        def deref(p):
            box, i = p
            return box[i]

        ip = 0
        steps = 0
        while ip < len(self.insns):
            steps += 1
            if steps > max_steps:
                raise BranchExecutionLimitError(f"Step limit {max_steps} exceeded: possible infinite loop")
            res, op, a, _ = self.insns[ip]
            if op in _IGNORED or op == "OpLabel":
                ip += 1
                continue
            if op in ("OpReturn", "OpFunctionEnd"):
                break
            if op == "OpBranch":
                target_label = a[0]
                if target_label not in self.labels:
                    raise UnsupportedConstruct(f"OpBranch target {target_label} not found")
                ip = self.labels[target_label]
                continue
            if op == "OpBranchConditional":
                cond = val(a[0])
                if not isinstance(cond, bool):
                    raise UnsupportedConstruct("OpBranchConditional requires a boolean condition")
                target_label = a[1] if cond else a[2]
                if target_label not in self.labels:
                    raise UnsupportedConstruct(f"OpBranchConditional target {target_label} not found")
                ip = self.labels[target_label]
                continue
            if op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                init = values[a[2]] if len(a) > 2 else self.zero(pointee)
                var_box[res] = [list(init) if isinstance(init, list) else init]
                pointers[res] = (var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if self.var_class.get(base) == "PushConstant":
                    pointers[res] = ("pc", None)
                else:
                    if base not in pointers:
                        raise UnsupportedConstruct(f"OpAccessChain on invalid base {base}")
                    p = pointers[base]
                    for idx_id in a[2:]:
                        idx = val(idx_id)
                        container = deref(p)
                        if not isinstance(container, (list, tuple)) or not (0 <= idx < len(container)):
                            raise UnsupportedConstruct(f"OpAccessChain out of bounds index {idx}")
                        p = (container, idx)
                    pointers[res] = p
            elif op == "OpLoad":
                if a[1] not in pointers:
                    raise UnsupportedConstruct(f"OpLoad from invalid pointer {a[1]}")
                p = pointers[a[1]]
                if p[0] == "pc":
                    values[res] = 0
                else:
                    v = deref(p)
                    values[res] = list(v) if isinstance(v, list) else v
            elif op == "OpStore":
                if a[0] not in pointers:
                    raise UnsupportedConstruct(f"OpStore to invalid pointer {a[0]}")
                p = pointers[a[0]]
                if p[0] == "pc":
                    raise UnsupportedConstruct("Store to PushConstant rejected")
                p[0][p[1]] = val(a[1])
            elif op == "OpIEqual":
                values[res] = (val(a[1]) == val(a[2]))
            elif op == "OpINotEqual":
                values[res] = (val(a[1]) != val(a[2]))
            elif op == "OpUGreaterThan":
                values[res] = (val(a[1]) & MASK32) > (val(a[2]) & MASK32)
            elif op == "OpULessThan":
                values[res] = (val(a[1]) & MASK32) < (val(a[2]) & MASK32)
            elif op == "OpUGreaterThanEqual":
                values[res] = (val(a[1]) & MASK32) >= (val(a[2]) & MASK32)
            elif op == "OpULessThanEqual":
                values[res] = (val(a[1]) & MASK32) <= (val(a[2]) & MASK32)
            elif op == "OpLogicalNot":
                c = val(a[1])
                if not isinstance(c, bool):
                    raise UnsupportedConstruct("OpLogicalNot operand must be bool")
                values[res] = not c
            elif op == "OpLogicalAnd":
                values[res] = bool(val(a[1]) and val(a[2]))
            elif op == "OpLogicalOr":
                values[res] = bool(val(a[1]) or val(a[2]))
            elif op == "OpSelect":
                cond = val(a[1])
                if not isinstance(cond, bool):
                    raise UnsupportedConstruct("OpSelect condition must be bool")
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = val(a[1])
            elif op == "OpCompositeConstruct":
                values[res] = tuple(val(x) for x in a[1:])
            elif op == "OpCompositeExtract":
                values[res] = val(a[1])[int(a[2])]
            elif op == "OpIAdd":
                values[res] = (val(a[1]) + val(a[2])) & MASK32
            elif op == "OpISub":
                values[res] = (val(a[1]) - val(a[2])) & MASK32
            elif op == "OpIAddCarry":
                x_val = val(a[1]) & MASK32
                y_val = val(a[2]) & MASK32
                total = x_val + y_val
                values[res] = (total & MASK32, 1 if total > MASK32 else 0)
            elif op == "OpISubBorrow":
                x_val = val(a[1]) & MASK32
                y_val = val(a[2]) & MASK32
                diff = (x_val - y_val) & MASK32
                values[res] = (diff, 1 if x_val < y_val else 0)
            elif op == "OpBitwiseAnd":
                values[res] = (val(a[1]) & val(a[2])) & MASK32
            elif op == "OpBitwiseOr":
                values[res] = (val(a[1]) | val(a[2])) & MASK32
            elif op == "OpBitwiseXor":
                values[res] = (val(a[1]) ^ val(a[2])) & MASK32
            elif op == "OpShiftLeftLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift left >= 32 is undefined in SPIR-V")
                values[res] = (val(a[1]) << sh) & MASK32
            elif op == "OpShiftRightLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift right >= 32 is undefined in SPIR-V")
                values[res] = (val(a[1]) >> sh) & MASK32
            else:
                raise UnsupportedConstruct(f"Opcode not modelled: {op}")
            ip += 1

        result = {}
        for out in self.outputs:
            v = var_box[out][0]
            result[out] = tuple(v if isinstance(v, tuple) else (v,))
        return result


def translate_disassembly_lines(lines: list[str], shader_name: str = "ps_cf_test") -> str:
    """Translates assembly lines into structured SPIR-V assembly text, suppressing ResourceWarnings."""
    with tempfile.TemporaryDirectory() as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        src_path = os.path.join(dis, f"{shader_name}.s")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        dxbc_path = os.path.join(tmp, f"{shader_name}.dxbc")
        with open(dxbc_path, "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            result = rdna2spirv.translate(src_path)
            gc.collect()
        return result


# -----------------------------------------------------------------------------
# Test Suite 1: Reference Truth Tables & Displacement Arithmetic
# -----------------------------------------------------------------------------

class TestBranchDisplacementAndConditionReference(unittest.TestCase):
    """Verifies RDNA2 signed two's-complement displacement arithmetic and SCC branch conditions."""

    def test_scc0_and_scc1_duality(self):
        """s_cbranch_scc0 and s_cbranch_scc1 must be strict logical complements across all SCC states."""
        test_states = [0, 1, 2, 3, 0x80000000, 0xFFFFFFFF]
        for scc in test_states:
            c0 = eval_scc0(scc)
            c1 = eval_scc1(scc)
            with self.subTest(scc=hex(scc)):
                self.assertNotEqual(c0, c1, "scc0 and scc1 must never have identical outcome")
                self.assertEqual(c0, (scc & 1) == 0)
                self.assertEqual(c1, (scc & 1) == 1)

    def test_forward_displacement_calculation(self):
        """Displacement target_pc = next_pc + 4 * simm16 for forward jumps."""
        self.assertEqual(compute_branch_target(pc=8, simm16=2), 20)
        self.assertEqual(compute_branch_target(pc=0, simm16=1), 8)
        self.assertEqual(compute_branch_target(pc=100, simm16=10), 144)

    def test_backward_displacement_calculation(self):
        """Displacement target_pc = next_pc + 4 * simm16 for backward jumps (negative simm16)."""
        self.assertEqual(compute_branch_target(pc=20, simm16=-4), 8)
        self.assertEqual(compute_branch_target(pc=20, simm16=0xFFFC), 8)
        self.assertEqual(compute_branch_target(pc=16, simm16=-1), 16)  # branch to self

    def test_zero_displacement_calculation(self):
        """simm16 = 0 branches to next_pc (effectively a no-op branch)."""
        self.assertEqual(compute_branch_target(pc=12, simm16=0), 16)

    def test_displacement_limits(self):
        """Boundary verification at 16-bit signed integer limits (-32768 to 32767 dwords)."""
        # Maximum positive jump (+32767 dwords)
        self.assertEqual(compute_branch_target(pc=0, simm16=0x7FFF), 4 + 4 * 32767)
        # Maximum negative jump (-32768 dwords)
        self.assertEqual(compute_branch_target(pc=200000, simm16=0x8000), 200004 - 4 * 32768)

    def test_evaluate_scc_branch_condition_dispatches(self):
        self.assertTrue(evaluate_scc_branch_condition("s_cbranch_scc0", 0))
        self.assertFalse(evaluate_scc_branch_condition("s_cbranch_scc0", 1))
        self.assertFalse(evaluate_scc_branch_condition("s_cbranch_scc1", 0))
        self.assertTrue(evaluate_scc_branch_condition("s_cbranch_scc1", 1))

        with self.assertRaises(ValueError):
            evaluate_scc_branch_condition("s_cbranch_vccz", 0)


# -----------------------------------------------------------------------------
# Test Suite 2: Reference Scalar Interpreter (ISA Level)
# -----------------------------------------------------------------------------

class TestScalarInterpreterExecution(unittest.TestCase):
    """Verifies control flow graphs executed on the pure Python scalar interpreter."""

    def test_if_then_skip_pattern_scc0(self):
        """s_cbranch_scc0 skips the then-block when SCC == 0; executes it when SCC == 1."""
        interp_skip = ScalarInterpreter()
        interp_skip.run_program({
            "entry": [
                ("s_mov_b32", 0, 50),
                ("s_mov_b32", 1, 10),
                ("s_cmp_eq_u32", 0, 999),  # 50 == 999 -> False -> SCC = 0
                ("s_cbranch_scc0", "exit"),
                "then_block",
            ],
            "then_block": [
                ("s_mov_b32", 1, 55),
                "exit",
            ],
            "exit": [
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp_skip.trace, ["entry", "exit"])
        self.assertEqual(interp_skip.get_sgpr(1), 10)

        interp_exec = ScalarInterpreter()
        interp_exec.run_program({
            "entry": [
                ("s_mov_b32", 0, 50),
                ("s_mov_b32", 1, 10),
                ("s_cmp_eq_u32", 0, 50),  # 50 == 50 -> True -> SCC = 1
                ("s_cbranch_scc0", "exit"),
                "then_block",
            ],
            "then_block": [
                ("s_mov_b32", 1, 55),
                "exit",
            ],
            "exit": [
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp_exec.trace, ["entry", "then_block", "exit"])
        self.assertEqual(interp_exec.get_sgpr(1), 55)

    def test_if_then_skip_pattern_scc1(self):
        """s_cbranch_scc1 skips the then-block when SCC == 1; executes it when SCC == 0."""
        interp_skip = ScalarInterpreter()
        interp_skip.run_program({
            "entry": [
                ("s_mov_b32", 0, 50),
                ("s_mov_b32", 1, 10),
                ("s_cmp_eq_u32", 0, 50),  # 50 == 50 -> True -> SCC = 1
                ("s_cbranch_scc1", "exit"),
                "then_block",
            ],
            "then_block": [
                ("s_mov_b32", 1, 55),
                "exit",
            ],
            "exit": [
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp_skip.trace, ["entry", "exit"])
        self.assertEqual(interp_skip.get_sgpr(1), 10)

    def test_diamond_if_then_else(self):
        """Symmetric diamond pattern: selects exactly one branch based on comparison."""
        def run_diamond(val: int) -> int:
            interp = ScalarInterpreter()
            interp.run_program({
                "entry": [
                    ("s_mov_b32", 0, val),
                    ("s_cmp_eq_u32", 0, 7),
                    ("s_cbranch_scc0", "else_block"),
                    "then_block",
                ],
                "then_block": [
                    ("s_mov_b32", 1, 100),
                    ("s_branch", "join"),
                ],
                "else_block": [
                    ("s_mov_b32", 1, 200),
                    ("s_branch", "join"),
                ],
                "join": [
                    ("s_endpgm",),
                ],
            }, "entry")
            return interp.get_sgpr(1)

        self.assertEqual(run_diamond(7), 100)
        self.assertEqual(run_diamond(42), 200)

    def test_countdown_loop_execution(self):
        """Iterative loop using backward s_cbranch_scc0 terminating when counter reaches 0."""
        interp = ScalarInterpreter()
        interp.run_program({
            "entry": [
                ("s_mov_b32", 0, 6),   # loop count = 6
                ("s_mov_b32", 1, 0),   # accumulator
                "loop_body",
            ],
            "loop_body": [
                ("s_add_u32", 1, 1, 10),  # acc += 10
                ("s_sub_u32", 0, 0, 1),   # counter -= 1
                ("s_cmp_eq_u32", 0, 0),   # counter == 0 ? SCC=1 : SCC=0
                ("s_cbranch_scc0", "loop_body"),
                "exit",
            ],
            "exit": [
                ("s_endpgm",),
            ],
        }, "entry")
        # 6 iterations * 10 = 60
        self.assertEqual(interp.get_sgpr(1), 60)
        self.assertEqual(interp.get_sgpr(0), 0)
        self.assertEqual(interp.trace.count("loop_body"), 6)

    def test_branch_on_all_comparison_conditions(self):
        """Verifies conditional branches following all standard scalar comparison operations."""
        comparisons = [
            ("s_cmp_eq_u32", 10, 10, True),
            ("s_cmp_eq_u32", 10, 20, False),
            ("s_cmp_lg_u32", 10, 20, True),
            ("s_cmp_lg_u32", 10, 10, False),
            ("s_cmp_gt_u32", 20, 10, True),
            ("s_cmp_gt_u32", 10, 20, False),
            ("s_cmp_ge_u32", 10, 10, True),
            ("s_cmp_ge_u32", 5, 10, False),
            ("s_cmp_lt_u32", 5, 10, True),
            ("s_cmp_lt_u32", 10, 5, False),
            ("s_cmp_le_u32", 10, 10, True),
            ("s_cmp_le_u32", 15, 10, False),
        ]
        for cmp_op, val0, val1, expected_true in comparisons:
            with self.subTest(cmp_op=cmp_op, val0=val0, val1=val1):
                interp = ScalarInterpreter()
                interp.run_program({
                    "entry": [
                        ("s_mov_b32", 0, val0),
                        (cmp_op, 0, val1),
                        ("s_cbranch_scc1", "true_target"),
                        "false_target",
                    ],
                    "true_target": [
                        ("s_mov_b32", 2, 1),
                        ("s_endpgm",),
                    ],
                    "false_target": [
                        ("s_mov_b32", 2, 0),
                        ("s_endpgm",),
                    ],
                }, "entry")
                expected_dest = "true_target" if expected_true else "false_target"
                self.assertIn(expected_dest, interp.trace)
                self.assertEqual(interp.get_sgpr(2), 1 if expected_true else 0)

    def test_branch_on_bitcmp_conditions(self):
        """Verifies branching following bitcmp0 (bit==0 -> SCC=1) and bitcmp1 (bit==1 -> SCC=1)."""
        interp = ScalarInterpreter()
        interp.run_program({
            "entry": [
                ("s_mov_b32", 0, 0b00000100),  # bit 2 is 1, bit 3 is 0
                ("s_bitcmp1_b32", 0, 2),       # bit 2 is 1 -> SCC=1
                ("s_cbranch_scc1", "bit2_is_set"),
                "fail",
            ],
            "bit2_is_set": [
                ("s_bitcmp0_b32", 0, 3),       # bit 3 is 0 -> SCC=1
                ("s_cbranch_scc1", "success"),
                "fail",
            ],
            "fail": [
                ("s_mov_b32", 1, 999),
                ("s_endpgm",),
            ],
            "success": [
                ("s_mov_b32", 1, 777),
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp.trace, ["entry", "bit2_is_set", "success"])
        self.assertEqual(interp.get_sgpr(1), 777)

    def test_branch_on_alu_carry_and_borrow(self):
        """s_add_u32 sets SCC on overflow carry; s_sub_u32 sets SCC on borrow."""
        # 1. Overflow carry: 0xFFFFFFFF + 1 -> sum 0, carry 1 -> s_cbranch_scc1 taken
        interp_carry = ScalarInterpreter()
        interp_carry.run_program({
            "entry": [
                ("s_mov_b32", 0, 0xFFFFFFFF),
                ("s_add_u32", 1, 0, 1),
                ("s_cbranch_scc1", "overflow_taken"),
                "overflow_missed",
            ],
            "overflow_taken": [
                ("s_mov_b32", 2, 42),
                ("s_endpgm",),
            ],
            "overflow_missed": [
                ("s_mov_b32", 2, 0),
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp_carry.trace, ["entry", "overflow_taken"])
        self.assertEqual(interp_carry.get_sgpr(2), 42)

        # 2. Borrow: 5 - 10 -> borrow 1 -> s_cbranch_scc1 taken
        interp_borrow = ScalarInterpreter()
        interp_borrow.run_program({
            "entry": [
                ("s_mov_b32", 0, 5),
                ("s_sub_u32", 1, 0, 10),
                ("s_cbranch_scc1", "borrow_taken"),
                "borrow_missed",
            ],
            "borrow_taken": [
                ("s_mov_b32", 2, 84),
                ("s_endpgm",),
            ],
            "borrow_missed": [
                ("s_mov_b32", 2, 0),
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp_borrow.trace, ["entry", "borrow_taken"])
        self.assertEqual(interp_borrow.get_sgpr(2), 84)

    def test_branch_on_bitwise_logic_scc(self):
        """s_and_b32 and s_or_b32 set SCC to 1 iff the result is non-zero."""
        interp = ScalarInterpreter()
        interp.run_program({
            "entry": [
                ("s_mov_b32", 0, 0x0F0F),
                ("s_and_b32", 1, 0, 0xF0F0),  # 0x0F0F & 0xF0F0 = 0 -> SCC=0
                ("s_cbranch_scc0", "is_zero"),
                "not_zero",
            ],
            "is_zero": [
                ("s_or_b32", 2, 0, 0x0001),   # 0x0F0F | 0x0001 = 0x0F0F != 0 -> SCC=1
                ("s_cbranch_scc1", "is_nonzero"),
                "fail",
            ],
            "not_zero": [
                ("s_mov_b32", 3, 111),
                ("s_endpgm",),
            ],
            "fail": [
                ("s_mov_b32", 3, 222),
                ("s_endpgm",),
            ],
            "is_nonzero": [
                ("s_mov_b32", 3, 333),
                ("s_endpgm",),
            ],
        }, "entry")
        self.assertEqual(interp.trace, ["entry", "is_zero", "is_nonzero"])
        self.assertEqual(interp.get_sgpr(3), 333)

    def test_infinite_loop_step_limit_error(self):
        """Runaway loops trigger BranchExecutionLimitError instead of hanging forever."""
        interp = ScalarInterpreter()
        with self.assertRaises(BranchExecutionLimitError):
            interp.run_program({
                "entry": [
                    ("s_branch", "entry"),
                ],
            }, "entry", max_steps=100)


# -----------------------------------------------------------------------------
# Test Suite 3: SPIR-V Translator Emission Verification
# -----------------------------------------------------------------------------

class TestTranslatorSpirvEmission(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits expected instructions for s_cbranch_scc0/1."""

    def test_branch_condition_scc0_emission(self):
        """s_cbranch_scc0 emits OpLoad %scc and OpIEqual against zero without subgroup voting."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_scc0")

        self.assertFalse(g.uses_vote, "s_cbranch_scc0 must not require subgroup voting")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %uint %scc", body_text)
        self.assertIn(f"{cond} = OpIEqual %bool", body_text)
        self.assertNotIn("OpGroupNonUniformAny", body_text)

    def test_branch_condition_scc1_emission(self):
        """s_cbranch_scc1 emits OpLoad %scc and OpINotEqual against zero without subgroup voting."""
        g = rdna2spirv.Gen("ps")
        cond = rdna2spirv.branch_condition(g, "s_cbranch_scc1")

        self.assertFalse(g.uses_vote, "s_cbranch_scc1 must not require subgroup voting")

        body_text = "\n".join(g.body)
        self.assertIn("OpLoad %uint %scc", body_text)
        self.assertIn(f"{cond} = OpINotEqual %bool", body_text)
        self.assertNotIn("OpGroupNonUniformAny", body_text)

    def test_branches_membership(self):
        """Verifies that s_cbranch_scc0 and s_cbranch_scc1 are registered in rdna2spirv.BRANCHES."""
        self.assertIn("s_cbranch_scc0", rdna2spirv.BRANCHES)
        self.assertIn("s_cbranch_scc1", rdna2spirv.BRANCHES)


# -----------------------------------------------------------------------------
# Test Suite 4: End-to-End SPIR-V Execution (StructuredBranchInterp)
# -----------------------------------------------------------------------------

class TestStructuredSpirvIfThenExecution(unittest.TestCase):
    """Tests forward conditional skipping with translated SPIR-V assembly."""

    def _build_if_then_shader(self, branch_op: str, s0_val: int, compare_val: int) -> list[str]:
        enc_branch = "0x01,0x00,0x84,0xbf" if branch_op == "s_cbranch_scc0" else "0x01,0x00,0x85,0xbf"
        return [
            "; ps_if_then test",
            f"\ts_mov_b32 s0, {s0_val} ; encoding: [0x80,0x03,0x80,0xbe]",
            "\ts_mov_b32 s1, 10 ; encoding: [0x8a,0x03,0x81,0xbe]",
            f"\ts_cmp_eq_u32 s0, {compare_val} ; encoding: [0x80,0x00,0x0c,0xbf]",
            f"\t{branch_op} 1 ; encoding: [{enc_branch}]",
            "\ts_mov_b32 s1, 55 ; encoding: [0xb7,0x03,0x81,0xbe]",
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]

    def test_s_cbranch_scc0_taken_skips_assignment(self):
        # s0 = 2 != 1 -> SCC = 0 -> s_cbranch_scc0 IS taken, skipping s1=55. Result = 10.
        spv = translate_disassembly_lines(self._build_if_then_shader("s_cbranch_scc0", s0_val=2, compare_val=1))
        self.assertIn("OpSelectionMerge", spv)
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (10, 10, 10, 10))

    def test_s_cbranch_scc0_not_taken_executes_assignment(self):
        # s0 = 1 == 1 -> SCC = 1 -> s_cbranch_scc0 NOT taken, executing s1=55. Result = 55.
        spv = translate_disassembly_lines(self._build_if_then_shader("s_cbranch_scc0", s0_val=1, compare_val=1))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (55, 55, 55, 55))

    def test_s_cbranch_scc1_taken_skips_assignment(self):
        # s0 = 1 == 1 -> SCC = 1 -> s_cbranch_scc1 IS taken, skipping s1=55. Result = 10.
        spv = translate_disassembly_lines(self._build_if_then_shader("s_cbranch_scc1", s0_val=1, compare_val=1))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (10, 10, 10, 10))

    def test_s_cbranch_scc1_not_taken_executes_assignment(self):
        # s0 = 2 != 1 -> SCC = 0 -> s_cbranch_scc1 NOT taken, executing s1=55. Result = 55.
        spv = translate_disassembly_lines(self._build_if_then_shader("s_cbranch_scc1", s0_val=2, compare_val=1))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (55, 55, 55, 55))


class TestStructuredSpirvDiamondExecution(unittest.TestCase):
    """Tests symmetric if-then-else diamond control flow where exactly one branch executes."""

    def _build_diamond_shader(self, s0_val: int) -> list[str]:
        return [
            "; ps_diamond test",
            f"\ts_mov_b32 s0, {s0_val} ; encoding: [0x80,0x03,0x80,0xbe]",
            "\ts_cmp_eq_u32 s0, 7 ; encoding: [0x87,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc0 2 ; encoding: [0x02,0x00,0x84,0xbf]",
            "\ts_mov_b32 s1, 100 ; encoding: [0xe4,0x03,0x81,0xbe]",
            "\ts_branch 1 ; encoding: [0x01,0x00,0x82,0xbf]",
            "\ts_mov_b32 s1, 200 ; encoding: [0xc8,0x03,0x81,0xbe]",
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]

    def test_diamond_then_branch_taken(self):
        spv = translate_disassembly_lines(self._build_diamond_shader(s0_val=7))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (100, 100, 100, 100))

    def test_diamond_else_branch_taken(self):
        spv = translate_disassembly_lines(self._build_diamond_shader(s0_val=9))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (200, 200, 200, 200))


class TestStructuredSpirvLoopExecution(unittest.TestCase):
    """Tests structured loop emission and execution with backward conditional branches."""

    def _build_loop_shader(self, iterations: int, step: int) -> list[str]:
        return [
            "; ps_loop test",
            f"\ts_mov_b32 s0, {iterations} ; encoding: [0x83,0x03,0x80,0xbe]",
            "\ts_mov_b32 s1, 0 ; encoding: [0x80,0x03,0x81,0xbe]",
            f"\ts_add_u32 s1, s1, {step} ; encoding: [0x85,0x01,0x01,0x80]",
            "\ts_sub_u32 s0, s0, 1 ; encoding: [0x81,0x00,0x00,0x81]",
            "\ts_cmp_eq_u32 s0, 0 ; encoding: [0x80,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc0 -4 ; encoding: [0xfc,0xff,0x84,0xbf]",
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]

    def test_single_iteration_loop(self):
        spv = translate_disassembly_lines(self._build_loop_shader(iterations=1, step=25))
        self.assertIn("OpLoopMerge", spv)
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (25, 25, 25, 25))

    def test_multi_iteration_loop(self):
        spv = translate_disassembly_lines(self._build_loop_shader(iterations=5, step=7))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (35, 35, 35, 35))

    def test_larger_iteration_loop(self):
        spv = translate_disassembly_lines(self._build_loop_shader(iterations=12, step=10))
        res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(res, (120, 120, 120, 120))


class TestStructuredSpirvNestedBranchControlFlow(unittest.TestCase):
    """Tests 2-level nested conditional branching covering all (outer, inner) permutations."""

    def _build_nested_shader(self, outer_val: int, inner_val: int) -> list[str]:
        # Block layout and displacements:
        # 0:  s_mov s0, outer_val
        # 4:  s_mov s2, inner_val
        # 8:  s_cmp_eq_u32 s0, 1
        # 12: s_cbranch_scc0 6     -> next_pc=16 + 24 = pc 40 (outer else)
        # 16: s_cmp_eq_u32 s2, 1
        # 20: s_cbranch_scc0 2     -> next_pc=24 + 8  = pc 32 (inner else)
        # 24: s_mov s1, 11
        # 28: s_branch 7           -> next_pc=32 + 28 = pc 60 (exit)
        # 32: s_mov s1, 10
        # 36: s_branch 5           -> next_pc=40 + 20 = pc 60 (exit)
        # 40: s_cmp_eq_u32 s2, 1
        # 44: s_cbranch_scc0 2     -> next_pc=48 + 8  = pc 56 (outer else inner else)
        # 48: s_mov s1, 21
        # 52: s_branch 1           -> next_pc=56 + 4  = pc 60 (exit)
        # 56: s_mov s1, 20
        # 60: v_mov v0, s1
        # 64: exp mrt0 v0, v0, v0, v0 done vm
        # 72: s_endpgm
        return [
            "; ps_nested test",
            f"\ts_mov_b32 s0, {outer_val} ; encoding: [0x80,0x03,0x80,0xbe]",
            f"\ts_mov_b32 s2, {inner_val} ; encoding: [0x80,0x03,0x82,0xbe]",
            "\ts_cmp_eq_u32 s0, 1 ; encoding: [0x81,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc0 6 ; encoding: [0x06,0x00,0x84,0xbf]",
            "\ts_cmp_eq_u32 s2, 1 ; encoding: [0x81,0x02,0x0c,0xbf]",
            "\ts_cbranch_scc0 2 ; encoding: [0x02,0x00,0x84,0xbf]",
            "\ts_mov_b32 s1, 11 ; encoding: [0x8b,0x03,0x81,0xbe]",
            "\ts_branch 7 ; encoding: [0x07,0x00,0x82,0xbf]",
            "\ts_mov_b32 s1, 10 ; encoding: [0x8a,0x03,0x81,0xbe]",
            "\ts_branch 5 ; encoding: [0x05,0x00,0x82,0xbf]",
            "\ts_cmp_eq_u32 s2, 1 ; encoding: [0x81,0x02,0x0c,0xbf]",
            "\ts_cbranch_scc0 2 ; encoding: [0x02,0x00,0x84,0xbf]",
            "\ts_mov_b32 s1, 21 ; encoding: [0x95,0x03,0x81,0xbe]",
            "\ts_branch 1 ; encoding: [0x01,0x00,0x82,0xbf]",
            "\ts_mov_b32 s1, 20 ; encoding: [0x94,0x03,0x81,0xbe]",
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]

    def test_all_nested_permutations(self):
        cases = [
            (1, 1, 11),
            (1, 0, 10),
            (0, 1, 21),
            (0, 0, 20),
        ]
        for outer, inner, expected in cases:
            with self.subTest(outer=outer, inner=inner):
                spv = translate_disassembly_lines(self._build_nested_shader(outer, inner))
                res = StructuredBranchInterp(spv).run()["%out_mrt0"]
                self.assertEqual(res, (expected, expected, expected, expected))


class TestMultiWayDispatchLadder(unittest.TestCase):
    """Tests a chained multi-way dispatch ladder (switch emulation via conditional branches)."""

    def _build_dispatch_shader(self, selector: int) -> list[str]:
        # Case 0: s1 = 100
        # Case 1: s1 = 200
        # Case 2: s1 = 300
        # Default: s1 = 400
        return [
            "; ps_dispatch test",
            f"\ts_mov_b32 s0, {selector} ; encoding: [0x80,0x03,0x80,0xbe]",
            # check case 0:
            "\ts_cmp_eq_u32 s0, 0 ; encoding: [0x80,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc1 6 ; encoding: [0x06,0x00,0x85,0xbf]",  # jump to case 0 body
            # check case 1:
            "\ts_cmp_eq_u32 s0, 1 ; encoding: [0x81,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc1 6 ; encoding: [0x06,0x00,0x85,0xbf]",  # jump to case 1 body
            # check case 2:
            "\ts_cmp_eq_u32 s0, 2 ; encoding: [0x82,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc1 6 ; encoding: [0x06,0x00,0x85,0xbf]",  # jump to case 2 body
            # default:
            "\ts_mov_b32 s1, 400 ; encoding: [0xc0,0x03,0x81,0xbe]",
            "\ts_branch 5 ; encoding: [0x05,0x00,0x82,0xbf]",
            # case 0 body:
            "\ts_mov_b32 s1, 100 ; encoding: [0xe4,0x03,0x81,0xbe]",
            "\ts_branch 3 ; encoding: [0x03,0x00,0x82,0xbf]",
            # case 1 body:
            "\ts_mov_b32 s1, 200 ; encoding: [0xc8,0x03,0x81,0xbe]",
            "\ts_branch 1 ; encoding: [0x01,0x00,0x82,0xbf]",
            # case 2 body:
            "\ts_mov_b32 s1, 300 ; encoding: [0xac,0x03,0x81,0xbe]",
            # exit:
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]

    def test_dispatch_ladder_cases(self):
        cases = [
            (0, 100),
            (1, 200),
            (2, 300),
            (3, 400),
            (99, 400),
        ]
        for sel, expected in cases:
            with self.subTest(selector=sel):
                spv = translate_disassembly_lines(self._build_dispatch_shader(sel))
                res = StructuredBranchInterp(spv).run()["%out_mrt0"]
                self.assertEqual(res, (expected, expected, expected, expected))


# -----------------------------------------------------------------------------
# Test Suite 5: Negative Controls & Mutation Sensitivity
# -----------------------------------------------------------------------------

class TestNegativeControlsAndFaultDetection(unittest.TestCase):
    """Negative controls and fault rejection: verifies sabotaged logic is detected."""

    def test_inverted_branch_condition_is_detected(self):
        """Sabotaging OpIEqual to OpINotEqual flips the branch outcome."""
        spv = translate_disassembly_lines([
            "; ps_fault test",
            "\ts_mov_b32 s0, 7 ; encoding: [0x87,0x03,0x80,0xbe]",
            "\ts_cmp_eq_u32 s0, 7 ; encoding: [0x87,0x00,0x0c,0xbf]",
            "\ts_cbranch_scc0 2 ; encoding: [0x02,0x00,0x84,0xbf]",
            "\ts_mov_b32 s1, 100 ; encoding: [0xe4,0x03,0x81,0xbe]",
            "\ts_branch 1 ; encoding: [0x01,0x00,0x82,0xbf]",
            "\ts_mov_b32 s1, 200 ; encoding: [0xc8,0x03,0x81,0xbe]",
            "\tv_mov_b32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ])
        baseline_res = StructuredBranchInterp(spv).run()["%out_mrt0"]
        self.assertEqual(baseline_res, (100, 100, 100, 100))

        # Invert condition by replacing OpIEqual with OpINotEqual
        self.assertIn("OpIEqual", spv)
        bad_spv = spv.replace("OpIEqual", "OpINotEqual", 1)
        bad_res = StructuredBranchInterp(bad_spv).run()["%out_mrt0"]
        self.assertNotEqual(bad_res, baseline_res)
        self.assertEqual(bad_res, (200, 200, 200, 200))

    def test_infinite_loop_triggers_limit_error(self):
        """Unconditional backward branch to self triggers BranchExecutionLimitError."""
        spv = translate_disassembly_lines([
            "; ps_infinite test",
            "\ts_mov_b32 s0, 0 ; encoding: [0x80,0x03,0x80,0xbe]",
            "\ts_branch -1 ; encoding: [0xff,0xff,0x82,0xbf]",
            "\tv_mov_b32 v0, s0 ; encoding: [0x00,0x02,0x00,0x7e]",
            "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ])
        with self.assertRaises(BranchExecutionLimitError):
            StructuredBranchInterp(spv).run(max_steps=500)

    def test_out_of_bounds_branch_target_rejected_by_translator(self):
        """Branch displacement landing outside code buffer is rejected with Unsupported."""
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_disassembly_lines([
                "; ps_bad_target test",
                "\ts_cbranch_scc0 500 ; encoding: [0xf4,0x01,0x84,0xbf]",
                "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
            ])

    def test_unsupported_branch_construct_rejected_by_interp(self):
        """Unrecognized opcode in SPIR-V interpreter raises UnsupportedConstruct."""
        spv = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %nonexistent_label
OpFunctionEnd
"""
        with self.assertRaises(UnsupportedConstruct):
            StructuredBranchInterp(spv).run()


if __name__ == "__main__":
    unittest.main()
