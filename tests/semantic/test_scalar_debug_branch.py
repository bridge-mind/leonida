"""Semantic interpreter tests for AMD RDNA2 scalar debug branching and halt controls.

Covers:
  - s_cbranch_cdbgsys: conditional branch taken iff COND_DBG_SYS is non-zero
  - s_cbranch_cdbguser: conditional branch taken iff COND_DBG_USER is non-zero
  - s_cbranch_cdbgsys_or_user: conditional branch taken iff COND_DBG_SYS or COND_DBG_USER is non-zero
  - s_cbranch_cdbgsys_and_user: conditional branch taken iff both COND_DBG_SYS and COND_DBG_USER are non-zero
  - s_sethalt: set or clear wavefront execution halt state based on simm16 & 1, with fatal halt
    detection (simm16 & 4) and privileged trap handler deferral (PRIV == 1)
  - SOPP binary instruction encoding/decoding and 16-bit signed immediate offset arithmetic
  - STATUS register packing/unpacking and hardware debug flag isolation
  - Control-flow graph execution (skipping, polling loops, if-then-else joins)
  - Privilege mode trap handler execution and pending halt commit upon exception return (s_rfe_b64)
  - Negative controls and mutation sensitivity verification

Standard library only; zero game data.
Compatible with Python 3.9+.

Run:
    python -I tests/semantic/test_scalar_debug_branch.py
    python -m unittest discover -s tests/semantic -v
"""
from __future__ import annotations

import dataclasses
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

import rdna2spirv  # noqa: E402
import shader_profile  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# -----------------------------------------------------------------------------
# 1. SOPP Binary Constants and Encodings (AMD RDNA2 / GFX10 Architecture)
# -----------------------------------------------------------------------------

# SOPP 32-bit instruction header: bits [31:23] = 0b101111111 (0x17F)
SOPP_HEADER = 0x17F

# SOPP Opcodes
SOPP_OP_NOP = 0
SOPP_OP_ENDPGM = 1
SOPP_OP_BRANCH = 2
SOPP_OP_CBRANCH_SCC0 = 4
SOPP_OP_CBRANCH_SCC1 = 5
SOPP_OP_CBRANCH_VCCZ = 6
SOPP_OP_CBRANCH_VCCNZ = 7
SOPP_OP_CBRANCH_EXECZ = 8
SOPP_OP_CBRANCH_EXECNZ = 9
SOPP_OP_SETHALT = 13
SOPP_OP_TRAP = 18
SOPP_OP_CBRANCH_CDBGSYS = 23
SOPP_OP_CBRANCH_CDBGUSER = 24
SOPP_OP_CBRANCH_CDBGSYS_OR_USER = 25
SOPP_OP_CBRANCH_CDBGSYS_AND_USER = 26

SOPP_MNEMONIC_TO_OPCODE = {
    "s_nop": SOPP_OP_NOP,
    "s_endpgm": SOPP_OP_ENDPGM,
    "s_branch": SOPP_OP_BRANCH,
    "s_cbranch_scc0": SOPP_OP_CBRANCH_SCC0,
    "s_cbranch_scc1": SOPP_OP_CBRANCH_SCC1,
    "s_cbranch_vccz": SOPP_OP_CBRANCH_VCCZ,
    "s_cbranch_vccnz": SOPP_OP_CBRANCH_VCCNZ,
    "s_cbranch_execz": SOPP_OP_CBRANCH_EXECZ,
    "s_cbranch_execnz": SOPP_OP_CBRANCH_EXECNZ,
    "s_sethalt": SOPP_OP_SETHALT,
    "s_trap": SOPP_OP_TRAP,
    "s_cbranch_cdbgsys": SOPP_OP_CBRANCH_CDBGSYS,
    "s_cbranch_cdbguser": SOPP_OP_CBRANCH_CDBGUSER,
    "s_cbranch_cdbgsys_or_user": SOPP_OP_CBRANCH_CDBGSYS_OR_USER,
    "s_cbranch_cdbgsys_and_user": SOPP_OP_CBRANCH_CDBGSYS_AND_USER,
}

SOPP_OPCODE_TO_MNEMONIC = {v: k for k, v in SOPP_MNEMONIC_TO_OPCODE.items()}


# -----------------------------------------------------------------------------
# 2. Pure Reference Oracles & Bit Arithmetic
# -----------------------------------------------------------------------------

def decode_simm16(raw16: int) -> int:
    """Decode a 16-bit unsigned value to a signed Python integer in range [-32768, 32767]."""
    raw = raw16 & MASK16
    return raw - 0x10000 if (raw & 0x8000) else raw


def encode_simm16(val: int) -> int:
    """Encode a signed integer in range [-32768, 32767] to a 16-bit unsigned bit pattern."""
    if not -32768 <= val <= 32767:
        raise ValueError(f"Immediate value {val} out of signed 16-bit range [-32768, 32767]")
    return val & MASK16


def compute_branch_target_bytes(current_pc_bytes: int, simm16: int) -> int:
    """Calculate byte-address branch target: target = PC + 4 + (simm16 * 4) mod 2^32."""
    return (current_pc_bytes + 4 + (simm16 * 4)) & MASK32


def compute_branch_target_dwords(current_pc_dwords: int, simm16: int) -> int:
    """Calculate dword-indexed branch target: target = PC_dwords + 1 + simm16."""
    return current_pc_dwords + 1 + simm16


def eval_cbranch_cdbgsys(cond_dbg_sys: bool) -> bool:
    """s_cbranch_cdbgsys: branch taken iff COND_DBG_SYS is non-zero."""
    return bool(cond_dbg_sys)


def eval_cbranch_cdbguser(cond_dbg_user: bool) -> bool:
    """s_cbranch_cdbguser: branch taken iff COND_DBG_USER is non-zero."""
    return bool(cond_dbg_user)


def eval_cbranch_cdbgsys_or_user(cond_dbg_sys: bool, cond_dbg_user: bool) -> bool:
    """s_cbranch_cdbgsys_or_user: branch taken iff COND_DBG_SYS or COND_DBG_USER is non-zero."""
    return bool(cond_dbg_sys or cond_dbg_user)


def eval_cbranch_cdbgsys_and_user(cond_dbg_sys: bool, cond_dbg_user: bool) -> bool:
    """s_cbranch_cdbgsys_and_user: branch taken iff both COND_DBG_SYS and COND_DBG_USER are non-zero."""
    return bool(cond_dbg_sys and cond_dbg_user)


def eval_sethalt(simm16: int, priv: bool = False) -> tuple[bool, bool, bool]:
    """Evaluate s_sethalt immediate against current privilege mode.

    Returns:
      (halt_immediate, pending_halt, fatal_halt)
      - simm16[0] (bit 0): Halt flag (1 = halt, 0 = resume).
      - simm16[2] (bit 2): Fatal halt flag.
      - If priv is True (privileged execution in trap handler), halt cannot take
        immediate effect; it is deferred into pending_halt until exception return.
    """
    halt_req = bool(simm16 & 1)
    fatal_req = bool((simm16 >> 2) & 1)
    if priv:
        return False, halt_req, fatal_req
    return halt_req, False, fatal_req


def encode_sopp(opcode: int, simm16: int) -> int:
    """Encode a 32-bit machine word for an SOPP instruction."""
    if not (0 <= opcode <= 0x7F):
        raise ValueError(f"Opcode {opcode} out of 7-bit range [0, 127]")
    raw16 = encode_simm16(simm16)
    return (SOPP_HEADER << 23) | ((opcode & 0x7F) << 16) | raw16


def decode_sopp(word: int) -> tuple[int, int]:
    """Decode a 32-bit machine word into (opcode, simm16)."""
    header = (word >> 23) & 0x1FF
    if header != SOPP_HEADER:
        raise ValueError(f"Invalid SOPP header: 0x{header:03X} (expected 0x{SOPP_HEADER:03X})")
    opcode = (word >> 16) & 0x7F
    raw16 = word & MASK16
    return opcode, decode_simm16(raw16)


# -----------------------------------------------------------------------------
# 3. Wavefront Status Register & Hardware Simulation
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class WaveStatus:
    """AMD GPU Wavefront STATUS register bitfield representation.

    Bit layout follows AMD RDNA / GCN architecture conventions:
      - Bit 0:  SCC (Scalar Condition Code)
      - Bit 10: HALT (Wavefront execution halted)
      - Bit 12: COND_DBG_SYS (Conditional debug system flag)
      - Bit 13: COND_DBG_USER (Conditional debug user flag)
      - Bit 14: PENDING_HALT (Deferred halt bit while in privileged mode)
      - Bit 16: PRIV (Privileged trap mode active)
      - Bit 20: FATAL_HALT (Fatal halt occurred)
    """
    scc: int = 0
    halt: bool = False
    cond_dbg_sys: bool = False
    cond_dbg_user: bool = False
    pending_halt: bool = False
    priv: bool = False
    fatal_halt: bool = False

    def to_dword(self) -> int:
        val = 0
        if self.scc:
            val |= (1 << 0)
        if self.halt:
            val |= (1 << 10)
        if self.cond_dbg_sys:
            val |= (1 << 12)
        if self.cond_dbg_user:
            val |= (1 << 13)
        if self.pending_halt:
            val |= (1 << 14)
        if self.priv:
            val |= (1 << 16)
        if self.fatal_halt:
            val |= (1 << 20)
        return val

    @classmethod
    def from_dword(cls, dword: int) -> WaveStatus:
        return cls(
            scc=(dword >> 0) & 1,
            halt=bool((dword >> 10) & 1),
            cond_dbg_sys=bool((dword >> 12) & 1),
            cond_dbg_user=bool((dword >> 13) & 1),
            pending_halt=bool((dword >> 14) & 1),
            priv=bool((dword >> 16) & 1),
            fatal_halt=bool((dword >> 20) & 1),
        )


# -----------------------------------------------------------------------------
# 4. Scalar Debug & Control Flow Interpreter
# -----------------------------------------------------------------------------

class ScalarDebugInterpreter:
    """Semantic interpreter modeling scalar debug branching, halt, and trap mechanisms.

    Features:
      1. Dual execution modes:
         - Symbolic CFG mode: executes a directed graph of labelled basic blocks.
         - Binary bytecode mode: executes sequential 32-bit machine dwords with
           real relative PC arithmetic (PC = PC + 1 + simm16).
      2. Hardware register simulation:
         - SGPRs (128 scalar general-purpose registers).
         - STATUS register (COND_DBG_SYS, COND_DBG_USER, HALT, PENDING_HALT, PRIV).
      3. Wavefront control:
         - s_sethalt: immediately halts user execution or defers when in PRIV mode.
         - s_trap: enters privileged mode (PRIV=1), saves PC, jumps to trap handler.
         - s_rfe_b64: exits privileged mode (PRIV=0), restores PC, commits pending halt.
      4. Complete execution tracing and step counter for deterministic verification.
    """

    def __init__(self, num_sgprs: int = 128, status: WaveStatus | None = None):
        self.sgprs = [0] * num_sgprs
        self.status = status if status is not None else WaveStatus()
        self.pc = 0
        self.saved_pc = 0
        self.trap_handler_entry: str | int | None = None
        self.trace: list[str | int] = []
        self.step_count = 0

    def reset_registers(self):
        self.sgprs = [0] * len(self.sgprs)
        self.pc = 0
        self.saved_pc = 0
        self.trace.clear()
        self.step_count = 0

    def unhalt(self):
        """External debugger intervention: clear wavefront halt state and resume."""
        self.status.halt = False
        self.status.pending_halt = False
        self.status.fatal_halt = False

    def is_halted(self) -> bool:
        return self.status.halt

    # --- Symbolic Basic Block Execution Mode ---

    def run_program(
        self,
        blocks: dict[str, list[tuple[str, ...] | str]],
        start_block: str,
        max_steps: int = 2000,
    ):
        """Execute a graph of basic blocks until s_endpgm, halt, or max_steps."""
        cur = start_block
        self.trace = []
        self.step_count = 0

        while cur is not None and self.step_count < max_steps:
            if self.status.halt:
                break

            self.step_count += 1
            self.trace.append(cur)
            block = blocks.get(cur)
            if block is None:
                raise ValueError(f"Unknown block label: {cur}")

            next_block = None
            for ins in block:
                if isinstance(ins, str):
                    next_block = ins
                    break

                op = ins[0]
                args = ins[1:]

                if op == "s_endpgm":
                    return

                elif op == "s_branch":
                    next_block = str(args[0])
                    break

                elif op == "s_cbranch_cdbgsys":
                    target = str(args[0])
                    if eval_cbranch_cdbgsys(self.status.cond_dbg_sys):
                        next_block = target
                        break

                elif op == "s_cbranch_cdbguser":
                    target = str(args[0])
                    if eval_cbranch_cdbguser(self.status.cond_dbg_user):
                        next_block = target
                        break

                elif op == "s_cbranch_cdbgsys_or_user":
                    target = str(args[0])
                    if eval_cbranch_cdbgsys_or_user(self.status.cond_dbg_sys, self.status.cond_dbg_user):
                        next_block = target
                        break

                elif op == "s_cbranch_cdbgsys_and_user":
                    target = str(args[0])
                    if eval_cbranch_cdbgsys_and_user(self.status.cond_dbg_sys, self.status.cond_dbg_user):
                        next_block = target
                        break

                elif op == "s_sethalt":
                    imm = int(args[0])
                    halt_now, pending, fatal = eval_sethalt(imm, priv=self.status.priv)
                    if halt_now:
                        self.status.halt = True
                    if pending:
                        self.status.pending_halt = True
                    if not (imm & 1):
                        self.status.halt = False
                        self.status.pending_halt = False
                    if fatal:
                        self.status.fatal_halt = True
                    if self.status.halt:
                        return

                elif op == "s_trap":
                    trap_id = int(args[0])
                    self.status.priv = True
                    # Fallthrough target within the block or next block becomes saved_pc
                    if self.trap_handler_entry is not None:
                        next_block = str(self.trap_handler_entry)
                        break

                elif op == "s_rfe_b64":
                    self.status.priv = False
                    if self.status.pending_halt:
                        self.status.halt = True
                        self.status.pending_halt = False
                        return
                    if args:
                        next_block = str(args[0])
                        break

                elif op == "s_mov_b32":
                    dst_s = int(args[0][1:]) if isinstance(args[0], str) and args[0].startswith("s") else int(args[0])
                    if isinstance(args[1], str) and args[1].startswith("s"):
                        val = self.sgprs[int(args[1][1:])]
                    else:
                        val = int(args[1]) & MASK32
                    self.sgprs[dst_s] = val

                elif op == "s_add_u32":
                    dst_s = int(args[0][1:]) if isinstance(args[0], str) and args[0].startswith("s") else int(args[0])
                    s0 = int(args[1][1:]) if isinstance(args[1], str) and args[1].startswith("s") else int(args[1])
                    val0 = self.sgprs[s0]
                    if isinstance(args[2], str) and args[2].startswith("s"):
                        val1 = self.sgprs[int(args[2][1:])]
                    else:
                        val1 = int(args[2]) & MASK32
                    self.sgprs[dst_s] = (val0 + val1) & MASK32

                elif op == "s_nop":
                    pass

                else:
                    raise ValueError(f"Unsupported instruction in interpreter: {op}")

            cur = next_block

        if self.step_count >= max_steps:
            raise RuntimeError(f"Program exceeded maximum execution steps ({max_steps}); potential infinite loop.")

    # --- Binary Bytecode Execution Mode ---

    def run_bytecode(
        self,
        bytecode: list[int | tuple[str, ...]],
        entry_pc: int = 0,
        max_steps: int = 2000,
    ):
        """Execute a linear sequence of machine instructions (dwords or decoded tuples)."""
        self.pc = entry_pc
        self.trace = []
        self.step_count = 0

        while 0 <= self.pc < len(bytecode) and self.step_count < max_steps:
            if self.status.halt:
                break

            self.step_count += 1
            curr_pc = self.pc
            self.trace.append(curr_pc)
            entry = bytecode[curr_pc]

            # Decode instruction if represented as raw 32-bit machine word
            if isinstance(entry, int):
                opcode, simm16 = decode_sopp(entry)
                mnemonic = SOPP_OPCODE_TO_MNEMONIC.get(opcode)
                if mnemonic is None:
                    raise ValueError(f"Unknown SOPP opcode: {opcode}")
                op = mnemonic
                args = (simm16,)
            else:
                op = entry[0]
                args = entry[1:]

            next_pc = curr_pc + 1

            if op == "s_endpgm":
                self.pc = len(bytecode)
                return

            elif op == "s_nop":
                pass

            elif op == "s_branch":
                simm16 = int(args[0])
                next_pc = compute_branch_target_dwords(curr_pc, simm16)

            elif op == "s_cbranch_cdbgsys":
                simm16 = int(args[0])
                if eval_cbranch_cdbgsys(self.status.cond_dbg_sys):
                    next_pc = compute_branch_target_dwords(curr_pc, simm16)

            elif op == "s_cbranch_cdbguser":
                simm16 = int(args[0])
                if eval_cbranch_cdbguser(self.status.cond_dbg_user):
                    next_pc = compute_branch_target_dwords(curr_pc, simm16)

            elif op == "s_cbranch_cdbgsys_or_user":
                simm16 = int(args[0])
                if eval_cbranch_cdbgsys_or_user(self.status.cond_dbg_sys, self.status.cond_dbg_user):
                    next_pc = compute_branch_target_dwords(curr_pc, simm16)

            elif op == "s_cbranch_cdbgsys_and_user":
                simm16 = int(args[0])
                if eval_cbranch_cdbgsys_and_user(self.status.cond_dbg_sys, self.status.cond_dbg_user):
                    next_pc = compute_branch_target_dwords(curr_pc, simm16)

            elif op == "s_sethalt":
                simm16 = int(args[0])
                halt_now, pending, fatal = eval_sethalt(simm16, priv=self.status.priv)
                if halt_now:
                    self.status.halt = True
                if pending:
                    self.status.pending_halt = True
                if not (simm16 & 1):
                    self.status.halt = False
                    self.status.pending_halt = False
                if fatal:
                    self.status.fatal_halt = True
                if self.status.halt:
                    self.pc = next_pc
                    return

            elif op == "s_trap":
                trap_imm = int(args[0])
                self.saved_pc = next_pc
                self.status.priv = True
                if self.trap_handler_entry is not None:
                    next_pc = int(self.trap_handler_entry)

            elif op == "s_rfe_b64":
                self.status.priv = False
                if self.status.pending_halt:
                    self.status.halt = True
                    self.status.pending_halt = False
                    self.pc = self.saved_pc
                    return
                next_pc = self.saved_pc

            elif op == "s_mov_b32":
                dst_s = int(args[0][1:]) if isinstance(args[0], str) and args[0].startswith("s") else int(args[0])
                if isinstance(args[1], str) and args[1].startswith("s"):
                    val = self.sgprs[int(args[1][1:])]
                else:
                    val = int(args[1]) & MASK32
                self.sgprs[dst_s] = val

            elif op == "s_add_u32":
                dst_s = int(args[0][1:]) if isinstance(args[0], str) and args[0].startswith("s") else int(args[0])
                s0 = int(args[1][1:]) if isinstance(args[1], str) and args[1].startswith("s") else int(args[1])
                val0 = self.sgprs[s0]
                if isinstance(args[2], str) and args[2].startswith("s"):
                    val1 = self.sgprs[int(args[2][1:])]
                else:
                    val1 = int(args[2]) & MASK32
                self.sgprs[dst_s] = (val0 + val1) & MASK32

            else:
                raise ValueError(f"Unknown instruction in bytecode interpreter: {op}")

            self.pc = next_pc

        if self.step_count >= max_steps:
            raise RuntimeError(f"Bytecode execution exceeded {max_steps} steps.")


# =============================================================================
# Unit Tests
# =============================================================================

class TestDebugBranchReferenceOracles(unittest.TestCase):
    """Mathematical reference oracle and truth table verification."""

    def test_cbranch_cdbgsys_truth_table(self):
        """s_cbranch_cdbgsys strictly follows the COND_DBG_SYS bit."""
        self.assertFalse(eval_cbranch_cdbgsys(False))
        self.assertTrue(eval_cbranch_cdbgsys(True))

    def test_cbranch_cdbguser_truth_table(self):
        """s_cbranch_cdbguser strictly follows the COND_DBG_USER bit."""
        self.assertFalse(eval_cbranch_cdbguser(False))
        self.assertTrue(eval_cbranch_cdbguser(True))

    def test_cbranch_cdbgsys_or_user_truth_table(self):
        """s_cbranch_cdbgsys_or_user behaves as logical OR."""
        self.assertFalse(eval_cbranch_cdbgsys_or_user(False, False))
        self.assertTrue(eval_cbranch_cdbgsys_or_user(False, True))
        self.assertTrue(eval_cbranch_cdbgsys_or_user(True, False))
        self.assertTrue(eval_cbranch_cdbgsys_or_user(True, True))

    def test_cbranch_cdbgsys_and_user_truth_table(self):
        """s_cbranch_cdbgsys_and_user behaves as logical AND."""
        self.assertFalse(eval_cbranch_cdbgsys_and_user(False, False))
        self.assertFalse(eval_cbranch_cdbgsys_and_user(False, True))
        self.assertFalse(eval_cbranch_cdbgsys_and_user(True, False))
        self.assertTrue(eval_cbranch_cdbgsys_and_user(True, True))

    def test_de_morgan_duality(self):
        """Verifies Boolean algebraic duality between OR and AND forms."""
        for sys_flag in (False, True):
            for user_flag in (False, True):
                or_res = eval_cbranch_cdbgsys_or_user(sys_flag, user_flag)
                and_res = eval_cbranch_cdbgsys_and_user(sys_flag, user_flag)
                # De Morgan: not (A or B) == (not A and not B)
                self.assertEqual(not or_res, (not sys_flag) and (not user_flag))
                # De Morgan: not (A and B) == (not A or not B)
                self.assertEqual(not and_res, (not sys_flag) or (not user_flag))

    def test_simm16_offset_decoding_boundaries(self):
        """Signed 16-bit offset decoding at extreme numerical limits."""
        self.assertEqual(decode_simm16(0x0000), 0)
        self.assertEqual(decode_simm16(0x0001), 1)
        self.assertEqual(decode_simm16(0x7FFF), 32767)
        self.assertEqual(decode_simm16(0x8000), -32768)
        self.assertEqual(decode_simm16(0xFFFF), -1)
        self.assertEqual(decode_simm16(0xFFFE), -2)

    def test_simm16_offset_encoding_boundaries(self):
        """Signed 16-bit offset encoding at extreme numerical limits."""
        self.assertEqual(encode_simm16(0), 0x0000)
        self.assertEqual(encode_simm16(1), 0x0001)
        self.assertEqual(encode_simm16(32767), 0x7FFF)
        self.assertEqual(encode_simm16(-32768), 0x8000)
        self.assertEqual(encode_simm16(-1), 0xFFFF)
        self.assertEqual(encode_simm16(-2), 0xFFFE)

        with self.assertRaises(ValueError):
            encode_simm16(32768)
        with self.assertRaises(ValueError):
            encode_simm16(-32769)

    def test_branch_target_byte_calculation(self):
        """AMD ISA byte-address target arithmetic: target = PC + 4 + (simm16 * 4)."""
        # PC at 0x1000, branch +0 -> next instruction at 0x1004
        self.assertEqual(compute_branch_target_bytes(0x1000, 0), 0x1004)
        # Forward branch by 5 instructions (20 bytes)
        self.assertEqual(compute_branch_target_bytes(0x1000, 5), 0x1018)
        # Backward branch by 2 instructions (-8 bytes) -> 0x1004 - 8 = 0x0FFC
        self.assertEqual(compute_branch_target_bytes(0x1000, -2), 0x0FFC)
        # Wraparound around 32-bit address boundary
        self.assertEqual(compute_branch_target_bytes(0xFFFFFFFC, 0), 0x00000000)


class TestSoppBinaryEncoding(unittest.TestCase):
    """Verifies SOPP machine instruction binary layout and field encoding."""

    def test_sopp_header_prefix(self):
        """Bits [31:23] of SOPP instructions must match header 0b101111111 (0x17F)."""
        encoded = encode_sopp(SOPP_OP_CBRANCH_CDBGSYS, 0)
        self.assertEqual((encoded >> 23) & 0x1FF, SOPP_HEADER)
        self.assertEqual(encoded & 0xFF800000, 0xBF800000)

    def test_sopp_opcode_extraction(self):
        """Verify opcodes are correctly packed into bits [22:16]."""
        ops_to_test = [
            ("s_cbranch_cdbgsys", SOPP_OP_CBRANCH_CDBGSYS),
            ("s_cbranch_cdbguser", SOPP_OP_CBRANCH_CDBGUSER),
            ("s_cbranch_cdbgsys_or_user", SOPP_OP_CBRANCH_CDBGSYS_OR_USER),
            ("s_cbranch_cdbgsys_and_user", SOPP_OP_CBRANCH_CDBGSYS_AND_USER),
            ("s_sethalt", SOPP_OP_SETHALT),
            ("s_branch", SOPP_OP_BRANCH),
            ("s_nop", SOPP_OP_NOP),
        ]
        for name, op_val in ops_to_test:
            word = encode_sopp(op_val, 42)
            extracted_op, extracted_imm = decode_sopp(word)
            self.assertEqual(extracted_op, op_val, f"Mismatch for opcode {name}")
            self.assertEqual(extracted_imm, 42)

    def test_sopp_negative_immediate_roundtrip(self):
        """Preserves two's-complement sign extension across pack/unpack."""
        for imm in (-1, -16, -256, -1024, -32768, 0, 1, 127, 4096, 32767):
            word = encode_sopp(SOPP_OP_CBRANCH_CDBGSYS, imm)
            _, recovered_imm = decode_sopp(word)
            self.assertEqual(recovered_imm, imm)

    def test_sopp_invalid_header_rejection(self):
        """decode_sopp must reject words missing the valid SOPP prefix."""
        bad_word = 0x00000000  # Zero header
        with self.assertRaises(ValueError):
            decode_sopp(bad_word)


class TestWaveStatusRegister(unittest.TestCase):
    """Verifies STATUS register bit packing, unpacking, and flag isolation."""

    def test_status_roundtrip_default(self):
        st = WaveStatus()
        dword = st.to_dword()
        self.assertEqual(dword, 0)
        st_recovered = WaveStatus.from_dword(dword)
        self.assertEqual(st, st_recovered)

    def test_status_individual_bits(self):
        """Verify each flag maps to its exact expected hardware bit position."""
        self.assertEqual(WaveStatus(scc=1).to_dword(), 1 << 0)
        self.assertEqual(WaveStatus(halt=True).to_dword(), 1 << 10)
        self.assertEqual(WaveStatus(cond_dbg_sys=True).to_dword(), 1 << 12)
        self.assertEqual(WaveStatus(cond_dbg_user=True).to_dword(), 1 << 13)
        self.assertEqual(WaveStatus(pending_halt=True).to_dword(), 1 << 14)
        self.assertEqual(WaveStatus(priv=True).to_dword(), 1 << 16)
        self.assertEqual(WaveStatus(fatal_halt=True).to_dword(), 1 << 20)

    def test_status_compound_packing(self):
        """Verify multiple active status bits unpack without crosstalk."""
        st = WaveStatus(
            scc=1,
            halt=False,
            cond_dbg_sys=True,
            cond_dbg_user=False,
            pending_halt=True,
            priv=True,
            fatal_halt=False,
        )
        dword = st.to_dword()
        st2 = WaveStatus.from_dword(dword)
        self.assertEqual(st, st2)
        self.assertTrue(st2.cond_dbg_sys)
        self.assertFalse(st2.cond_dbg_user)
        self.assertTrue(st2.priv)
        self.assertTrue(st2.pending_halt)


class TestScalarDebugBranchInterpreterCFG(unittest.TestCase):
    """Verifies scalar debug branching within symbolic control-flow graphs."""

    def test_cdbgsys_branch_taken_when_flag_set(self):
        """When COND_DBG_SYS is True, s_cbranch_cdbgsys jumps to target block."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 100),
                ("s_cbranch_cdbgsys", "debug_block"),
                "release_block",
            ],
            "release_block": [
                ("s_mov_b32", 1, 200),
                ("s_endpgm",),
            ],
            "debug_block": [
                ("s_mov_b32", 1, 999),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=True))
        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "debug_block"])
        self.assertEqual(interp.sgprs[0], 100)
        self.assertEqual(interp.sgprs[1], 999)

    def test_cdbgsys_fallthrough_when_flag_clear(self):
        """When COND_DBG_SYS is False, s_cbranch_cdbgsys falls through."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 100),
                ("s_cbranch_cdbgsys", "debug_block"),
                "release_block",
            ],
            "release_block": [
                ("s_mov_b32", 1, 200),
                ("s_endpgm",),
            ],
            "debug_block": [
                ("s_mov_b32", 1, 999),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=False))
        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "release_block"])
        self.assertEqual(interp.sgprs[0], 100)
        self.assertEqual(interp.sgprs[1], 200)

    def test_cdbguser_branch_taken_when_flag_set(self):
        """When COND_DBG_USER is True, s_cbranch_cdbguser jumps to target block."""
        blocks = {
            "entry": [
                ("s_mov_b32", 5, 50),
                ("s_cbranch_cdbguser", "user_debug"),
                "user_normal",
            ],
            "user_normal": [
                ("s_mov_b32", 6, 1),
                ("s_endpgm",),
            ],
            "user_debug": [
                ("s_mov_b32", 6, 2),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_user=True))
        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "user_debug"])
        self.assertEqual(interp.sgprs[5], 50)
        self.assertEqual(interp.sgprs[6], 2)

    def test_flag_isolation_sys_does_not_trigger_user_branch(self):
        """COND_DBG_SYS=True must not trigger s_cbranch_cdbguser."""
        blocks = {
            "entry": [
                ("s_cbranch_cdbguser", "user_target"),
                "fallthrough",
            ],
            "user_target": [
                ("s_mov_b32", 0, 0xBAD),
                ("s_endpgm",),
            ],
            "fallthrough": [
                ("s_mov_b32", 0, 0x600D),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=True, cond_dbg_user=False))
        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "fallthrough"])
        self.assertEqual(interp.sgprs[0], 0x600D)

    def test_flag_isolation_user_does_not_trigger_sys_branch(self):
        """COND_DBG_USER=True must not trigger s_cbranch_cdbgsys."""
        blocks = {
            "entry": [
                ("s_cbranch_cdbgsys", "sys_target"),
                "fallthrough",
            ],
            "sys_target": [
                ("s_mov_b32", 0, 0xBAD),
                ("s_endpgm",),
            ],
            "fallthrough": [
                ("s_mov_b32", 0, 0x600D),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=False, cond_dbg_user=True))
        interp.run_program(blocks, "entry")

        self.assertEqual(interp.trace, ["entry", "fallthrough"])
        self.assertEqual(interp.sgprs[0], 0x600D)

    def test_cdbgsys_or_user_branch_combinations(self):
        """s_cbranch_cdbgsys_or_user branches if either or both flags are active."""
        for sys_flag in (False, True):
            for user_flag in (False, True):
                blocks = {
                    "entry": [
                        ("s_cbranch_cdbgsys_or_user", "target"),
                        "fallthrough",
                    ],
                    "target": [
                        ("s_mov_b32", 0, 1),
                        ("s_endpgm",),
                    ],
                    "fallthrough": [
                        ("s_mov_b32", 0, 0),
                        ("s_endpgm",),
                    ],
                }

                interp = ScalarDebugInterpreter(
                    status=WaveStatus(cond_dbg_sys=sys_flag, cond_dbg_user=user_flag)
                )
                interp.run_program(blocks, "entry")

                expected_taken = sys_flag or user_flag
                expected_trace = ["entry", "target"] if expected_taken else ["entry", "fallthrough"]
                expected_val = 1 if expected_taken else 0

                self.assertEqual(interp.trace, expected_trace)
                self.assertEqual(interp.sgprs[0], expected_val)

    def test_cdbgsys_and_user_branch_combinations(self):
        """s_cbranch_cdbgsys_and_user branches only when both flags are active."""
        for sys_flag in (False, True):
            for user_flag in (False, True):
                blocks = {
                    "entry": [
                        ("s_cbranch_cdbgsys_and_user", "target"),
                        "fallthrough",
                    ],
                    "target": [
                        ("s_mov_b32", 0, 1),
                        ("s_endpgm",),
                    ],
                    "fallthrough": [
                        ("s_mov_b32", 0, 0),
                        ("s_endpgm",),
                    ],
                }

                interp = ScalarDebugInterpreter(
                    status=WaveStatus(cond_dbg_sys=sys_flag, cond_dbg_user=user_flag)
                )
                interp.run_program(blocks, "entry")

                expected_taken = sys_flag and user_flag
                expected_trace = ["entry", "target"] if expected_taken else ["entry", "fallthrough"]
                expected_val = 1 if expected_taken else 0

                self.assertEqual(interp.trace, expected_trace)
                self.assertEqual(interp.sgprs[0], expected_val)

    def test_structured_if_then_else_convergence(self):
        """Diamond control-flow where debug and release paths re-converge."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 10),
                ("s_cbranch_cdbgsys", "debug_path"),
                "release_path",
            ],
            "release_path": [
                ("s_add_u32", 0, 0, 5),   # 10 + 5 = 15
                ("s_branch", "join"),
            ],
            "debug_path": [
                ("s_add_u32", 0, 0, 100),  # 10 + 100 = 110
                ("s_branch", "join"),
            ],
            "join": [
                ("s_add_u32", 0, 0, 1),   # Final increment
                ("s_endpgm",),
            ],
        }

        # Release mode run
        interp_rel = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=False))
        interp_rel.run_program(blocks, "entry")
        self.assertEqual(interp_rel.trace, ["entry", "release_path", "join"])
        self.assertEqual(interp_rel.sgprs[0], 16)

        # Debug mode run
        interp_dbg = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=True))
        interp_dbg.run_program(blocks, "entry")
        self.assertEqual(interp_dbg.trace, ["entry", "debug_path", "join"])
        self.assertEqual(interp_dbg.sgprs[0], 111)


class TestScalarSetHaltSemantics(unittest.TestCase):
    """Verifies s_sethalt wavefront halting, resumption, and privilege handling."""

    def test_immediate_halt_stops_wave(self):
        """s_sethalt 1 terminates instruction progression immediately."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 42),
                ("s_sethalt", 1),
                ("s_mov_b32", 0, 999),  # Should never be reached
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter()
        interp.run_program(blocks, "entry")

        self.assertTrue(interp.is_halted())
        self.assertEqual(interp.sgprs[0], 42)
        self.assertEqual(interp.trace, ["entry"])

    def test_sethalt_zero_does_not_halt(self):
        """s_sethalt 0 allows execution to proceed normally."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 42),
                ("s_sethalt", 0),
                ("s_mov_b32", 1, 84),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter()
        interp.run_program(blocks, "entry")

        self.assertFalse(interp.is_halted())
        self.assertEqual(interp.sgprs[0], 42)
        self.assertEqual(interp.sgprs[1], 84)

    def test_sethalt_bitmask_evaluation(self):
        """Only bit 0 dictates halt state (odd values halt, even values do not)."""
        test_immediates = [
            (1, True, False),       # Halt, not fatal
            (0, False, False),      # Resume, not fatal
            (3, True, False),       # Halt (bits 0 and 1)
            (5, True, True),        # Halt and Fatal (bits 0 and 2)
            (4, False, True),       # Fatal request without halt bit
            (0xFFFF, True, True),   # All bits set
            (0xFFFE, False, True),  # Bit 0 clear
        ]
        for imm, expect_halt, expect_fatal in test_immediates:
            halt, _, fatal = eval_sethalt(imm, priv=False)
            self.assertEqual(halt, expect_halt, f"Halt mismatch for imm={hex(imm)}")
            self.assertEqual(fatal, expect_fatal, f"Fatal mismatch for imm={hex(imm)}")

    def test_unhalt_and_resumption(self):
        """An external unhalt intervention allows execution to resume."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 1),
                ("s_sethalt", 1),
                "resumed",
            ],
            "resumed": [
                ("s_mov_b32", 0, 2),
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter()
        interp.run_program(blocks, "entry")
        self.assertTrue(interp.is_halted())
        self.assertEqual(interp.sgprs[0], 1)

        # Debugger unhalts the wave and continues execution from resumed block
        interp.unhalt()
        self.assertFalse(interp.is_halted())
        interp.run_program(blocks, "resumed")
        self.assertEqual(interp.sgprs[0], 2)

    def test_privileged_trap_halt_deferral(self):
        """s_sethalt 1 inside privileged trap mode defers halt until exception return."""
        blocks = {
            "entry": [
                ("s_mov_b32", 0, 10),
                ("s_trap", 1),
                "after_trap",
            ],
            "trap_handler": [
                ("s_mov_b32", 1, 20),
                ("s_sethalt", 1),       # Inside PRIV mode: deferred!
                ("s_mov_b32", 1, 30),   # Trap handler continues executing
                ("s_rfe_b64", "after_trap"),  # Return from exception: commits halt!
            ],
            "after_trap": [
                ("s_mov_b32", 0, 999),  # Should NOT execute because wave halted upon rfe
                ("s_endpgm",),
            ],
        }

        interp = ScalarDebugInterpreter()
        interp.trap_handler_entry = "trap_handler"
        interp.run_program(blocks, "entry")

        # The wave should have executed entry -> trap_handler, then halted at rfe
        self.assertTrue(interp.is_halted())
        self.assertEqual(interp.sgprs[0], 10)  # User code did not advance to 999
        self.assertEqual(interp.sgprs[1], 30)  # Trap handler ran fully to completion
        self.assertIn("trap_handler", interp.trace)
        self.assertNotIn("after_trap", interp.trace)


class TestLinearBytecodeExecution(unittest.TestCase):
    """Verifies binary SOPP bytecode execution and PC-relative jump arithmetic."""

    def test_bytecode_forward_branch_cdbgsys(self):
        """Verify binary s_cbranch_cdbgsys skipping instructions when taken."""
        # Program:
        # 0: s_mov_b32 s0, 10
        # 1: s_cbranch_cdbgsys +2 (jumps to instruction 1 + 1 + 2 = 4)
        # 2: s_mov_b32 s0, 20   (skipped if taken)
        # 3: s_mov_b32 s1, 30   (skipped if taken)
        # 4: s_mov_b32 s2, 40   (target)
        # 5: s_endpgm
        insn_branch = encode_sopp(SOPP_OP_CBRANCH_CDBGSYS, 2)
        insn_end = encode_sopp(SOPP_OP_ENDPGM, 0)

        bytecode: list[int | tuple[str, ...]] = [
            ("s_mov_b32", 0, 10),
            insn_branch,
            ("s_mov_b32", 0, 20),
            ("s_mov_b32", 1, 30),
            ("s_mov_b32", 2, 40),
            insn_end,
        ]

        # Case A: Debug flag True -> branch taken, jumps over 2 and 3
        interp_taken = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=True))
        interp_taken.run_bytecode(bytecode)
        self.assertEqual(interp_taken.trace, [0, 1, 4, 5])
        self.assertEqual(interp_taken.sgprs[0], 10)
        self.assertEqual(interp_taken.sgprs[1], 0)   # Not executed
        self.assertEqual(interp_taken.sgprs[2], 40)

        # Case B: Debug flag False -> branch not taken, falls through
        interp_fallthrough = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=False))
        interp_fallthrough.run_bytecode(bytecode)
        self.assertEqual(interp_fallthrough.trace, [0, 1, 2, 3, 4, 5])
        self.assertEqual(interp_fallthrough.sgprs[0], 20)  # Overwritten
        self.assertEqual(interp_fallthrough.sgprs[1], 30)
        self.assertEqual(interp_fallthrough.sgprs[2], 40)

    def test_bytecode_backward_branch_loop(self):
        """Verify binary backward branch looping with s_branch."""
        # Program: Count down s0 from 3 to 0
        # 0: s_mov_b32 s0, 3
        # 1: s_add_u32 s0, s0, -1 (in two's complement: 0xFFFFFFFF)
        # 2: s_cbranch_cdbgsys -2 (if debug is active, loop back to 1; target = 2 + 1 - 2 = 1)
        # 3: s_endpgm
        loop_branch = encode_sopp(SOPP_OP_CBRANCH_CDBGSYS, -2)
        bytecode: list[int | tuple[str, ...]] = [
            ("s_mov_b32", 0, 3),
            ("s_add_u32", 0, 0, MASK32),  # Subtract 1
            loop_branch,
            encode_sopp(SOPP_OP_ENDPGM, 0),
        ]

        # Single step through without loop (flag False)
        interp = ScalarDebugInterpreter(status=WaveStatus(cond_dbg_sys=False))
        interp.run_bytecode(bytecode)
        self.assertEqual(interp.trace, [0, 1, 2, 3])
        self.assertEqual(interp.sgprs[0], 2)

    def test_bytecode_sethalt_binary_encoded(self):
        """Verify binary encoded s_sethalt execution."""
        insn_halt = encode_sopp(SOPP_OP_SETHALT, 1)
        bytecode = [
            ("s_mov_b32", 0, 777),
            insn_halt,
            ("s_mov_b32", 0, 888),
            encode_sopp(SOPP_OP_ENDPGM, 0),
        ]

        interp = ScalarDebugInterpreter()
        interp.run_bytecode(bytecode)

        self.assertTrue(interp.is_halted())
        self.assertEqual(interp.sgprs[0], 777)
        self.assertEqual(interp.trace, [0, 1])


class TestToolingIntegrationAndClassification(unittest.TestCase):
    """Verifies that debug branch instructions integrate properly with shader tooling."""

    def test_shader_profile_classification(self):
        """shader_profile must classify scalar debug branches as 'branch'."""
        self.assertEqual(shader_profile.classify_instruction("s_cbranch_cdbgsys"), "branch")
        self.assertEqual(shader_profile.classify_instruction("s_cbranch_cdbguser"), "branch")
        self.assertEqual(shader_profile.classify_instruction("s_cbranch_cdbgsys_or_user"), "branch")
        self.assertEqual(shader_profile.classify_instruction("s_cbranch_cdbgsys_and_user"), "branch")

        self.assertTrue(shader_profile.is_branch("s_cbranch_cdbgsys"))
        self.assertTrue(shader_profile.is_branch("s_cbranch_cdbguser"))
        self.assertTrue(shader_profile.is_branch("s_cbranch_cdbgsys_or_user"))
        self.assertTrue(shader_profile.is_branch("s_cbranch_cdbgsys_and_user"))

    def test_rdna2spirv_branches_separation(self):
        """Debug branches are hardware/debugger specific and excluded from standard SPIR-V CFG branches."""
        self.assertNotIn("s_cbranch_cdbgsys", rdna2spirv.BRANCHES)
        self.assertNotIn("s_cbranch_cdbguser", rdna2spirv.BRANCHES)
        self.assertIn("s_branch", rdna2spirv.BRANCHES)
        self.assertIn("s_cbranch_scc0", rdna2spirv.BRANCHES)
        self.assertIn("s_cbranch_scc1", rdna2spirv.BRANCHES)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Verifies test assertions detect mutations or inverted logic."""

    def test_inverted_cdbgsys_condition_detected(self):
        """Mutating eval_cbranch_cdbgsys to invert logic must fail."""
        def inverted_eval(flag: bool) -> bool:
            return not flag

        self.assertNotEqual(eval_cbranch_cdbgsys(True), inverted_eval(True))
        self.assertNotEqual(eval_cbranch_cdbgsys(False), inverted_eval(False))

    def test_inverted_sethalt_bit_detected(self):
        """Mutating eval_sethalt to check bit 1 instead of bit 0 must fail on odd values."""
        def bad_sethalt(imm: int) -> bool:
            return bool((imm >> 1) & 1)

        self.assertNotEqual(eval_sethalt(1)[0], bad_sethalt(1))
        self.assertNotEqual(eval_sethalt(2)[0], bad_sethalt(2))

    def test_unsigned_offset_interpretation_detected(self):
        """Treating SIMM16 as unsigned converts negative jumps into out-of-range positive jumps."""
        def unsigned_decode(raw: int) -> int:
            return raw & 0xFFFF

        negative_pattern = 0xFFFF  # -1 in two's complement
        self.assertEqual(decode_simm16(negative_pattern), -1)
        self.assertEqual(unsigned_decode(negative_pattern), 65535)
        self.assertNotEqual(decode_simm16(negative_pattern), unsigned_decode(negative_pattern))

    def test_missing_trap_deferral_detected(self):
        """If halt is not deferred in privileged mode, trap handlers would freeze prematurely."""
        halt_immediate, pending, _ = eval_sethalt(1, priv=True)
        self.assertFalse(halt_immediate, "Halt must not be immediate when in privileged trap handler")
        self.assertTrue(pending, "Halt must be stored as pending when in privileged trap handler")

    def test_infinite_loop_guard(self):
        """Interpreter must raise RuntimeError if branch loop exceeds max_steps."""
        blocks = {
            "loop": [
                ("s_branch", "loop"),
            ],
        }
        interp = ScalarDebugInterpreter()
        with self.assertRaises(RuntimeError):
            interp.run_program(blocks, "loop", max_steps=50)


if __name__ == "__main__":
    unittest.main()
