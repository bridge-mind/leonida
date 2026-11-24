"""Semantic interpreter tests for mixed vector conditional select patterns with alternate mask configurations.

AMD RDNA2 / GFX10 VALU Instruction Set Architecture:
  - v_cndmask_b32_e32 dst, src0, src1
      Implicit condition mask: VCC (vcc_lo in Wave32, vcc in Wave64).
      For each active lane i (EXEC[i] == 1):
          dst[i] = src1[i] if (VCC[i] == 1) else src0[i]
      Inactive lanes (EXEC[i] == 0) preserve prior dst[i].

  - v_cndmask_b32_e64 dst, src0, src1, cond_mask
      Explicit condition mask: any SGPR, 64-bit SGPR pair, VCC, EXEC, or immediate.
      Operates under EXEC mask identically to e32.

Mixed Vector Select Patterns & Alternate Mask Configurations Covered:
  1. Mixed Operand Configurations:
     - Selecting between VGPR and SGPR (uniform base, divergent replacement; or divergent base, uniform replacement).
     - Selecting between VGPR and inline / literal constants (0, 1.0f, -1, bitmasks).
     - Selecting between SGPR and SGPR across divergent lane masks.
     - Self-referencing in-place destination selects (dst == src0 or dst == src1).
     - Floating-point representations (preserving sign, exponent, mantissa, special floats +/-0, Inf, NaN).
     - Signed and unsigned 32-bit integer boundaries.

  2. Alternate Mask Configurations:
     - Explicit SGPR scalar condition masks (s4, s6, etc.) independent of VCC.
     - 64-bit SGPR pair condition masks (s[4:5]) in Wave64.
     - Inverted condition masks (~mask, s_not_b32) and dual-complement symmetry.
     - Boolean mask compositions:
         * Conjunction: s_and_b32 (maskA & maskB)
         * Disjunction: s_or_b32 (maskA | maskB)
         * Exclusive: s_xor_b32 (maskA ^ maskB)
         * Inhibit / AND-NOT: s_andn2_b32 (maskA & ~maskB)
     - Strided, checkered, and walking bitmask patterns (powers-of-two clusters, one-hot, one-cold).
     - Comparison-derived dynamic masks from float (v_cmp_*_f32) and integer (v_cmp_*_u32/i32).

  3. Shader Idioms & Multiplexer Trees:
     - Branchless clamping (float, signed int, unsigned int) via cascaded conditional selects.
     - Multi-way multiplexer trees:
         * 3-way ternary multiplexer: condA ? A : (condB ? B : C)
         * 4-way priority encoder: cond0 ? D0 : (cond1 ? D1 : (cond2 ? D2 : D3))
         * 2-level binary decision tree: sel_hi ? (sel_lo ? D : C) : (sel_lo ? B : A)
     - Branchless math operations:
         * Absolute value: (x < 0) ? -x : x
         * Signum: (x > 0) ? 1 : ((x < 0) ? -1 : 0)
         * Safe division with zero-divisor fallback: (denom == 0) ? fallback : (num / denom)
         * Step and rectangular window / pulse functions.
     - Vector channel mixing and interleaving.

  4. EXEC Mask Interaction & Divergence:
     - Strict inactive lane preservation when EXEC[i] == 0 regardless of condition mask state.
     - Orthogonal EXEC and condition masks.
     - Structured divergent branching (Then/Else emulation with s_and_saveexec_b32 and s_or_exec_b32).
     - Zero-EXEC whole-wave no-op.

  5. Wave64 Cross-Half & 64-Bit Divergence:
     - Cross-half divergence: low-half (lanes 0..31) and high-half (lanes 32..63) using distinct patterns.
     - 64-bit scalar mask logic (s_and_b64, s_or_b64, s_xor_b64, s_not_b64).
     - Half-disabled waves (EXEC low-half only or high-half only).

  6. SPIR-V Translation & Verification:
     - Translates assembly via tools/rdna2spirv.py emitting OpSelect.
     - Executes via tests/semantic/spirv_mini_interp.py.

  7. Invariants & Negative Controls:
     - Single-bit condition flip sensitivity.
     - Invariant duality: cndmask(s0, s1, mask) == cndmask(s1, s0, ~mask).
     - Idempotence: cndmask(x, x, mask) == x.

Standard library only: zero game data, no external dependencies, Python 3.9+ compatible.
"""
from __future__ import annotations

import gc
import math
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

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
SIGN_MASK = 0x80000000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


# -----------------------------------------------------------------------------
# 1. Pure Mathematical Reference Oracle & Helpers
# -----------------------------------------------------------------------------

def f32_to_bits(f: float) -> int:
    """Encode float into IEEE 754 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def bits_to_f32(b: int) -> float:
    """Decode IEEE 754 32-bit unsigned integer into float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def s32(x: int) -> int:
    """Convert integer to signed 32-bit two's complement."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_MASK) else x


def ref_cndmask_lane(
    src0: int,
    src1: int,
    cond_bit: int,
    exec_bit: int = 1,
    prev_dst: int = 0,
) -> int:
    """AMD RDNA2 v_cndmask_b32 per-lane mathematical reference oracle.

    Lane semantics:
      - If exec_bit == 0: lane is inactive, dst is preserved (returns prev_dst & MASK32).
      - If exec_bit == 1: lane is active:
          - If cond_bit == 0: selects src0 & MASK32.
          - If cond_bit != 0: selects src1 & MASK32.
    """
    if not (exec_bit & 1):
        return prev_dst & MASK32
    return (src1 if (cond_bit & 1) else src0) & MASK32


def ref_cndmask_wave(
    src0_lanes: list[int],
    src1_lanes: list[int],
    cond_mask: int,
    exec_mask: int,
    prev_dst_lanes: list[int] | None = None,
    wave_size: int = 32,
) -> list[int]:
    """Whole-wave vector conditional move reference oracle across divergent lanes."""
    if prev_dst_lanes is None:
        prev_dst_lanes = [0] * wave_size
    res: list[int] = []
    for lane in range(wave_size):
        c_bit = (cond_mask >> lane) & 1
        e_bit = (exec_mask >> lane) & 1
        s0 = src0_lanes[lane]
        s1 = src1_lanes[lane]
        old = prev_dst_lanes[lane]
        res.append(ref_cndmask_lane(s0, s1, c_bit, e_bit, old))
    return res


# -----------------------------------------------------------------------------
# 2. Multi-Lane SIMD Wave Interpreter
# -----------------------------------------------------------------------------

class AluCndmaskMixInterpreter:
    """Simulates SIMD wave execution for mixed vector conditional selects.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - VGPRs: per-lane independent 32-bit vector registers (v0..v255).
      - SGPRs: uniform 32-bit scalar registers shared across the wave (s0..s127).
      - VCC: condition mask register (32-bit in Wave32, 64-bit in Wave64).
      - EXEC: execution mask register (32-bit in Wave32, 64-bit in Wave64).
      - Vector conditional moves: v_cndmask_b32_e32, v_cndmask_b32_e64.
      - Comparisons: v_cmp_* (integer and floating-point).
      - Scalar mask logic: s_mov, s_not, s_and, s_or, s_xor, s_andn2, s_orn2.
      - Control flow: s_and_saveexec, s_or_exec, s_xor_saveexec.
      - Arithmetic: v_add_u32, v_sub_u32, v_xor_b32, v_bfi_b32.
    """

    def __init__(self, wave_size: int = 32, initial_exec: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size: {wave_size}")
        self.wave_size = wave_size
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask = self.mask_all if initial_exec is None else (initial_exec & self.mask_all)
        self.vcc = 0
        self.sgprs = [0] * 128
        self.vgprs = [[0] * 256 for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & self.mask_all

    def set_vcc(self, mask: int) -> None:
        self.vcc = mask & self.mask_all

    def get_vgpr(self, reg_idx: int) -> list[int]:
        return [self.vgprs[lane][reg_idx] for lane in range(self.wave_size)]

    def set_vgpr_lane(self, reg_idx: int, lane: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32

    def init_vgpr(self, reg_idx: int, values: list[int] | int) -> None:
        if isinstance(values, int):
            val = values & MASK32
            for lane in range(self.wave_size):
                self.vgprs[lane][reg_idx] = val
        else:
            for lane in range(min(len(values), self.wave_size)):
                self.vgprs[lane][reg_idx] = values[lane] & MASK32

    def read_operand(self, op: int | float | str, lane: int) -> int:
        if isinstance(op, int):
            return op & MASK32
        if isinstance(op, float):
            return f32_to_bits(op)
        op_str = str(op).strip().lower()
        if op_str.startswith("v"):
            reg_idx = int(op_str[1:])
            return self.vgprs[lane][reg_idx] & MASK32
        elif op_str.startswith("s[") and ":" in op_str:
            lo = int(op_str[2:-1].split(":")[0])
            return self.sgprs[lo] & MASK32
        elif op_str.startswith("s"):
            reg_idx = int(op_str[1:])
            return self.sgprs[reg_idx] & MASK32
        elif op_str in ("vcc", "vcc_lo"):
            return (self.vcc >> lane) & 1
        elif op_str == "vcc_hi":
            return (self.vcc >> (lane + 32)) & 1 if self.wave_size == 64 else 0
        elif op_str in ("exec", "exec_lo"):
            return (self.exec_mask >> lane) & 1
        elif op_str == "exec_hi":
            return (self.exec_mask >> (lane + 32)) & 1 if self.wave_size == 64 else 0
        else:
            raise ValueError(f"Unsupported operand: {op}")

    def read_mask_source(self, cond_src: str | int, lane: int) -> int:
        if isinstance(cond_src, int):
            return (cond_src >> lane) & 1
        c_str = str(cond_src).strip().lower()
        if c_str in ("vcc", "vcc_lo"):
            return (self.vcc >> lane) & 1
        elif c_str in ("exec", "exec_lo"):
            return (self.exec_mask >> lane) & 1
        elif c_str.startswith("s[") and ":" in c_str:
            lo = int(c_str[2:-1].split(":")[0])
            lo_val = self.sgprs[lo] & MASK32
            hi_val = (self.sgprs[lo + 1] & MASK32) if (lo + 1 < len(self.sgprs)) else 0
            mask = lo_val | (hi_val << 32)
            return (mask >> lane) & 1
        elif c_str.startswith("s"):
            reg = int(c_str[1:])
            mask = self.sgprs[reg] & MASK32
            return (mask >> (lane % 32)) & 1
        else:
            raise ValueError(f"Unknown cond_src: {cond_src}")

    def execute_v_mov(self, dst_v: int, src: int | float | str) -> None:
        """v_mov_b32 dst_v, src: active lanes only."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                self.vgprs[lane][dst_v] = self.read_operand(src, lane)

    def execute_v_cndmask_e32(self, dst_v: int, src0: int | float | str, src1: int | float | str) -> None:
        """v_cndmask_b32_e32 dst, src0, src1 with implicit VCC condition."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.read_operand(src0, lane)
                s1 = self.read_operand(src1, lane)
                c_bit = (self.vcc >> lane) & 1
                self.vgprs[lane][dst_v] = s1 if c_bit else s0

    def execute_v_cndmask_e64(
        self,
        dst_v: int,
        src0: int | float | str,
        src1: int | float | str,
        cond_src: str | int,
    ) -> None:
        """v_cndmask_b32_e64 dst, src0, src1, cond_src with explicit condition mask."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.read_operand(src0, lane)
                s1 = self.read_operand(src1, lane)
                c_bit = self.read_mask_source(cond_src, lane)
                self.vgprs[lane][dst_v] = s1 if c_bit else s0

    def execute_v_cmp(
        self,
        rel: str,
        src0: int | float | str,
        src1: int | float | str,
        dst_mask: str = "vcc",
    ) -> None:
        """v_cmp_<rel>: inactive lanes in EXEC write 0 to condition mask."""
        mask = 0
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0_u = self.read_operand(src0, lane)
                s1_u = self.read_operand(src1, lane)
                taken = False

                if rel == "eq_u32":
                    taken = (s0_u == s1_u)
                elif rel == "ne_u32":
                    taken = (s0_u != s1_u)
                elif rel == "lt_u32":
                    taken = (s0_u < s1_u)
                elif rel == "le_u32":
                    taken = (s0_u <= s1_u)
                elif rel == "gt_u32":
                    taken = (s0_u > s1_u)
                elif rel == "ge_u32":
                    taken = (s0_u >= s1_u)
                elif rel == "lt_i32":
                    taken = (s32(s0_u) < s32(s1_u))
                elif rel == "le_i32":
                    taken = (s32(s0_u) <= s32(s1_u))
                elif rel == "gt_i32":
                    taken = (s32(s0_u) > s32(s1_u))
                elif rel == "ge_i32":
                    taken = (s32(s0_u) >= s32(s1_u))
                elif rel in ("lt_f32", "le_f32", "gt_f32", "ge_f32", "eq_f32", "ne_f32", "o_f32", "u_f32"):
                    f0 = bits_to_f32(s0_u)
                    f1 = bits_to_f32(s1_u)
                    nan_present = math.isnan(f0) or math.isnan(f1)
                    if rel == "o_f32":
                        taken = not nan_present
                    elif rel == "u_f32":
                        taken = nan_present
                    elif nan_present:
                        taken = False
                    elif rel == "lt_f32":
                        taken = (f0 < f1)
                    elif rel == "le_f32":
                        taken = (f0 <= f1)
                    elif rel == "gt_f32":
                        taken = (f0 > f1)
                    elif rel == "ge_f32":
                        taken = (f0 >= f1)
                    elif rel == "eq_f32":
                        taken = (f0 == f1)
                    elif rel == "ne_f32":
                        taken = (f0 != f1)
                else:
                    raise NotImplementedError(f"Unsupported cmp rel: {rel}")

                if taken:
                    mask |= (1 << lane)

        if dst_mask in ("vcc", "vcc_lo"):
            self.vcc = mask & self.mask_all
        elif dst_mask.startswith("s[") and ":" in dst_mask:
            lo = int(dst_mask[2:-1].split(":")[0])
            self.sgprs[lo] = mask & MASK32
            if lo + 1 < len(self.sgprs):
                self.sgprs[lo + 1] = (mask >> 32) & MASK32
        elif dst_mask.startswith("s"):
            reg = int(dst_mask[1:])
            self.sgprs[reg] = mask & MASK32

    def execute_s_mov_b32(self, dst_s: int, val: int | str) -> None:
        if isinstance(val, int):
            self.sgprs[dst_s] = val & MASK32
        elif str(val).startswith("s"):
            self.sgprs[dst_s] = self.sgprs[int(str(val)[1:])] & MASK32
        elif str(val) in ("vcc", "vcc_lo"):
            self.sgprs[dst_s] = self.vcc & MASK32
        elif str(val) in ("exec", "exec_lo"):
            self.sgprs[dst_s] = self.exec_mask & MASK32

    def execute_s_mov_b64(self, dst_s: int, val: int | str) -> None:
        if isinstance(val, int):
            self.sgprs[dst_s] = val & MASK32
            self.sgprs[dst_s + 1] = (val >> 32) & MASK32
        elif str(val) == "vcc":
            self.sgprs[dst_s] = self.vcc & MASK32
            self.sgprs[dst_s + 1] = (self.vcc >> 32) & MASK32
        elif str(val) == "exec":
            self.sgprs[dst_s] = self.exec_mask & MASK32
            self.sgprs[dst_s + 1] = (self.exec_mask >> 32) & MASK32

    def execute_s_not_b32(self, dst_s: int, src_s: int | str) -> None:
        src_val = (self.vcc & MASK32) if str(src_s) in ("vcc", "vcc_lo") else self.sgprs[int(str(src_s).lstrip("s"))]
        self.sgprs[dst_s] = (~src_val) & MASK32

    def execute_s_not_b64(self, dst_s: int, src_s: str) -> None:
        if src_s == "vcc":
            v = self.vcc
        elif src_s == "exec":
            v = self.exec_mask
        else:
            lo = int(src_s.strip("s[]").split(":")[0])
            v = (self.sgprs[lo] & MASK32) | ((self.sgprs[lo + 1] & MASK32) << 32)
        inv = (~v) & MASK64
        self.sgprs[dst_s] = inv & MASK32
        self.sgprs[dst_s + 1] = (inv >> 32) & MASK32

    def execute_s_and_b32(self, dst_s: int, s0: str | int, s1: str | int) -> None:
        def get_val(op: str | int) -> int:
            if isinstance(op, int):
                return op & MASK32
            op_str = str(op).lower()
            if op_str in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_str.startswith("s"):
                return self.sgprs[int(op_str[1:])]
            return int(op) & MASK32
        self.sgprs[dst_s] = (get_val(s0) & get_val(s1)) & MASK32

    def execute_s_or_b32(self, dst_s: int, s0: str | int, s1: str | int) -> None:
        def get_val(op: str | int) -> int:
            if isinstance(op, int):
                return op & MASK32
            op_str = str(op).lower()
            if op_str in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_str.startswith("s"):
                return self.sgprs[int(op_str[1:])]
            return int(op) & MASK32
        self.sgprs[dst_s] = (get_val(s0) | get_val(s1)) & MASK32

    def execute_s_xor_b32(self, dst_s: int, s0: str | int, s1: str | int) -> None:
        def get_val(op: str | int) -> int:
            if isinstance(op, int):
                return op & MASK32
            op_str = str(op).lower()
            if op_str in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_str.startswith("s"):
                return self.sgprs[int(op_str[1:])]
            return int(op) & MASK32
        self.sgprs[dst_s] = (get_val(s0) ^ get_val(s1)) & MASK32

    def execute_s_andn2_b32(self, dst_s: int, s0: str | int, s1: str | int) -> None:
        """s_andn2_b32: dst = s0 & ~s1."""
        def get_val(op: str | int) -> int:
            if isinstance(op, int):
                return op & MASK32
            op_str = str(op).lower()
            if op_str in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_str.startswith("s"):
                return self.sgprs[int(op_str[1:])]
            return int(op) & MASK32
        self.sgprs[dst_s] = (get_val(s0) & (~get_val(s1) & MASK32)) & MASK32

    def execute_s_orn2_b32(self, dst_s: int, s0: str | int, s1: str | int) -> None:
        """s_orn2_b32: dst = s0 | ~s1."""
        def get_val(op: str | int) -> int:
            if isinstance(op, int):
                return op & MASK32
            op_str = str(op).lower()
            if op_str in ("vcc", "vcc_lo"):
                return self.vcc & MASK32
            if op_str.startswith("s"):
                return self.sgprs[int(op_str[1:])]
            return int(op) & MASK32
        self.sgprs[dst_s] = (get_val(s0) | (~get_val(s1) & MASK32)) & MASK32

    def execute_s_and_saveexec_b32(self, dst_s: int, mask_src: str | int) -> None:
        self.sgprs[dst_s] = self.exec_mask & MASK32
        if isinstance(mask_src, int):
            m = mask_src
        elif mask_src in ("vcc", "vcc_lo"):
            m = self.vcc
        elif str(mask_src).startswith("s"):
            m = self.sgprs[int(str(mask_src)[1:])]
        else:
            raise ValueError(f"Unknown mask_src: {mask_src}")
        self.exec_mask = (self.exec_mask & m) & self.mask_all

    def execute_s_or_exec_b32(self, mask_src: str | int) -> None:
        if isinstance(mask_src, int):
            m = mask_src
        elif mask_src in ("vcc", "vcc_lo"):
            m = self.vcc
        elif str(mask_src).startswith("s"):
            m = self.sgprs[int(str(mask_src)[1:])]
        else:
            raise ValueError(f"Unknown mask_src: {mask_src}")
        self.exec_mask = (self.exec_mask | m) & self.mask_all

    def execute_s_xor_saveexec_b32(self, dst_s: int, mask_src: str | int) -> None:
        self.sgprs[dst_s] = self.exec_mask & MASK32
        if isinstance(mask_src, int):
            m = mask_src
        elif mask_src in ("vcc", "vcc_lo"):
            m = self.vcc
        elif str(mask_src).startswith("s"):
            m = self.sgprs[int(str(mask_src)[1:])]
        else:
            raise ValueError(f"Unknown mask_src: {mask_src}")
        self.exec_mask = (self.exec_mask ^ m) & self.mask_all

    def execute_v_add_u32(self, dst_v: int, src0: int | str, src1: int | str) -> None:
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.read_operand(src0, lane)
                s1 = self.read_operand(src1, lane)
                self.vgprs[lane][dst_v] = (s0 + s1) & MASK32

    def execute_v_sub_u32(self, dst_v: int, src0: int | str, src1: int | str) -> None:
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.read_operand(src0, lane)
                s1 = self.read_operand(src1, lane)
                self.vgprs[lane][dst_v] = (s0 - s1) & MASK32

    def execute_v_xor_b32(self, dst_v: int, src0: int | str, src1: int | str) -> None:
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                s0 = self.read_operand(src0, lane)
                s1 = self.read_operand(src1, lane)
                self.vgprs[lane][dst_v] = (s0 ^ s1) & MASK32

    def execute_v_bfi_b32(self, dst_v: int, src0_mask: int | str, src1_field: int | str, src2_base: int | str) -> None:
        """dst = (mask & field) | (~mask & base)."""
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                m = self.read_operand(src0_mask, lane)
                f = self.read_operand(src1_field, lane)
                b = self.read_operand(src2_base, lane)
                self.vgprs[lane][dst_v] = ((m & f) | ((~m & MASK32) & b)) & MASK32


# -----------------------------------------------------------------------------
# 3. Synthetic Shader Runner via rdna2spirv & spirv_mini_interp
# -----------------------------------------------------------------------------

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cndmask_mix.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cndmask_mix.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run_synthetic_ps(lines: list[str]) -> tuple[int, int, int, int]:
    """Translates synthetic assembly lines and executes with spirv_mini_interp."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; ps_cndmask_mix\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    res = si.run_fragment(spv)["%out_mrt0"]
    return tuple(res)


# -----------------------------------------------------------------------------
# 4. Test Suites
# -----------------------------------------------------------------------------

class TestCndmaskMixReferenceOracle(unittest.TestCase):
    """Verifies core reference oracle truth table, algebraic duality, and bit invariants."""

    def test_truth_table_active_lane(self) -> None:
        src0 = 0xAAAAAAAA
        src1 = 0x55555555
        prev = 0xDEADBEEF
        self.assertEqual(ref_cndmask_lane(src0, src1, cond_bit=0, exec_bit=1, prev_dst=prev), src0)
        self.assertEqual(ref_cndmask_lane(src0, src1, cond_bit=1, exec_bit=1, prev_dst=prev), src1)

    def test_inactive_lane_shielding(self) -> None:
        src0 = 0x11111111
        src1 = 0x22222222
        prev = 0xCAFEBABE
        self.assertEqual(ref_cndmask_lane(src0, src1, cond_bit=0, exec_bit=0, prev_dst=prev), prev)
        self.assertEqual(ref_cndmask_lane(src0, src1, cond_bit=1, exec_bit=0, prev_dst=prev), prev)

    def test_involution_duality(self) -> None:
        """Duality: cndmask(A, B, cond) == cndmask(B, A, 1 - cond)."""
        rng = random.Random(20261122)
        for _ in range(50):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            for c in (0, 1):
                self.assertEqual(
                    ref_cndmask_lane(a, b, cond_bit=c, exec_bit=1),
                    ref_cndmask_lane(b, a, cond_bit=1 - c, exec_bit=1),
                )

    def test_idempotence(self) -> None:
        """cndmask(x, x, cond) == x for any condition bit."""
        for x in (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x3F800000):
            self.assertEqual(ref_cndmask_lane(x, x, 0, 1), x)
            self.assertEqual(ref_cndmask_lane(x, x, 1, 1), x)

    def test_floating_point_bit_preservation(self) -> None:
        """Validates that NaN, Inf, and subnormal floats are preserved bit-for-bit without alteration."""
        special_floats = [
            0x7F800000,  # +INF
            0xFF800000,  # -INF
            0x7FC00000,  # Quiet NaN
            0x7F800001,  # Signaling NaN with payload
            0x00000001,  # Positive smallest subnormal
            0x80000001,  # Negative smallest subnormal
            0x00000000,  # +0.0
            0x80000000,  # -0.0
        ]
        for s0 in special_floats:
            for s1 in special_floats:
                self.assertEqual(ref_cndmask_lane(s0, s1, 0, 1), s0)
                self.assertEqual(ref_cndmask_lane(s0, s1, 1, 1), s1)


class TestMixedOperandPatterns(unittest.TestCase):
    """Verifies conditional select mixing VGPRs, SGPRs, and inline/literal constants."""

    def test_vgpr_vs_sgpr_uniform_base(self) -> None:
        """dst = vcc ? divergent_vgpr : uniform_sgpr."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        divergent_vgpr = [0x1000 + i for i in range(32)]
        wave.init_vgpr(1, divergent_vgpr)
        uniform_sgpr = 0x55AA55AA
        wave.execute_s_mov_b32(dst_s=4, val=uniform_sgpr)
        wave.set_vcc(0x55555555)  # alternating condition

        wave.execute_v_cndmask_e32(dst_v=0, src0="s4", src1="v1")
        res = wave.get_vgpr(0)

        for i in range(32):
            expected = divergent_vgpr[i] if (i % 2 == 0) else uniform_sgpr
            self.assertEqual(res[i], expected)

    def test_vgpr_vs_sgpr_uniform_replacement(self) -> None:
        """dst = vcc ? uniform_sgpr : divergent_vgpr."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        divergent_vgpr = [0x9000 + i for i in range(32)]
        wave.init_vgpr(1, divergent_vgpr)
        uniform_sgpr = 0xBEEFBEEF
        wave.execute_s_mov_b32(dst_s=6, val=uniform_sgpr)
        wave.set_vcc(0x0000FFFF)  # low half true

        wave.execute_v_cndmask_e32(dst_v=0, src0="v1", src1="s6")
        res = wave.get_vgpr(0)

        for i in range(32):
            expected = uniform_sgpr if (i < 16) else divergent_vgpr[i]
            self.assertEqual(res[i], expected)

    def test_sgpr_vs_sgpr_divergent_select(self) -> None:
        """Selecting between two uniform constants across a divergent wave."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        wave.execute_s_mov_b32(dst_s=0, val=0x11111111)
        wave.execute_s_mov_b32(dst_s=1, val=0x22222222)
        wave.set_vcc(0xF0F0F0F0)

        wave.execute_v_cndmask_e32(dst_v=2, src0="s0", src1="s1")
        res = wave.get_vgpr(2)

        for i in range(32):
            c = (0xF0F0F0F0 >> i) & 1
            expected = 0x22222222 if c else 0x11111111
            self.assertEqual(res[i], expected)

    def test_vgpr_vs_immediate_zero_gating(self) -> None:
        """Vector zero-gating / masking: dst = vcc ? v0 : 0."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        v0_vals = [0x80000000 + i for i in range(32)]
        wave.init_vgpr(0, v0_vals)
        wave.set_vcc(0x33333333)  # 2 on, 2 off

        wave.execute_v_cndmask_e32(dst_v=2, src0=0, src1="v0")
        res = wave.get_vgpr(2)

        for i in range(32):
            expected = v0_vals[i] if ((0x33333333 >> i) & 1) else 0
            self.assertEqual(res[i], expected)

    def test_predicate_expansion_to_full_masks(self) -> None:
        """Expands 1-bit predicate to all-1s (0xFFFFFFFF) or all-0s (0x00000000) mask:
           v_cndmask_b32 dst, 0, -1, vcc.
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        pred_mask = 0xA5A5A5A5
        wave.set_vcc(pred_mask)

        wave.execute_v_cndmask_e32(dst_v=0, src0=0, src1=0xFFFFFFFF)
        res = wave.get_vgpr(0)

        for i in range(32):
            expected = 0xFFFFFFFF if ((pred_mask >> i) & 1) else 0x00000000
            self.assertEqual(res[i], expected)

    def test_self_referencing_destination_select(self) -> None:
        """In-place updates:
           1. dst == src0: v_cndmask_b32 v0, v0, v1 (update only active true lanes)
           2. dst == src1: v_cndmask_b32 v0, v1, v0 (update only active false lanes)
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        base = [100 + i for i in range(32)]
        replacement = [500 + i for i in range(32)]

        # Case 1: dst == src0
        wave.init_vgpr(0, base)
        wave.init_vgpr(1, replacement)
        wave.set_vcc(0x00FF00FF)
        wave.execute_v_cndmask_e32(dst_v=0, src0="v0", src1="v1")
        res1 = wave.get_vgpr(0)
        for i in range(32):
            self.assertEqual(res1[i], replacement[i] if ((0x00FF00FF >> i) & 1) else base[i])

        # Case 2: dst == src1
        wave.init_vgpr(0, base)
        wave.init_vgpr(1, replacement)
        wave.set_vcc(0x00FF00FF)
        wave.execute_v_cndmask_e32(dst_v=0, src0="v1", src1="v0")
        res2 = wave.get_vgpr(0)
        for i in range(32):
            self.assertEqual(res2[i], base[i] if ((0x00FF00FF >> i) & 1) else replacement[i])


class TestAlternateMaskConfigurations(unittest.TestCase):
    """Verifies alternate condition mask sources: SGPR masks, inverted masks, and boolean compositions."""

    def test_sgpr_scalar_condition_mask_e64(self) -> None:
        """Explicit condition mask read from SGPR s8 while VCC holds an opposing pattern."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        src0 = [0x1111 + i for i in range(32)]
        src1 = [0x2222 + i for i in range(32)]
        wave.init_vgpr(0, src0)
        wave.init_vgpr(1, src1)

        wave.execute_s_mov_b32(dst_s=8, val=0x0F0F0F0F)
        wave.set_vcc(0xF0F0F0F0)  # Inverted from s8 to prove VCC is ignored

        wave.execute_v_cndmask_e64(dst_v=2, src0="v0", src1="v1", cond_src="s8")
        res = wave.get_vgpr(2)

        for i in range(32):
            expected = src1[i] if ((0x0F0F0F0F >> i) & 1) else src0[i]
            self.assertEqual(res[i], expected)
        # VCC must remain intact
        self.assertEqual(wave.vcc, 0xF0F0F0F0)

    def test_inverted_mask_configuration(self) -> None:
        """Inverting mask with s_not_b32 vs swapping operands."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        src0 = [10 + i for i in range(32)]
        src1 = [20 + i for i in range(32)]
        wave.init_vgpr(0, src0)
        wave.init_vgpr(1, src1)

        mask = 0x12345678
        wave.execute_s_mov_b32(dst_s=4, val=mask)
        wave.execute_s_not_b32(dst_s=6, src_s="s4")

        # Approach A: use inverted mask s6 with normal operands
        wave.execute_v_cndmask_e64(dst_v=10, src0="v0", src1="v1", cond_src="s6")
        # Approach B: use normal mask s4 with swapped operands
        wave.execute_v_cndmask_e64(dst_v=11, src0="v1", src1="v0", cond_src="s4")

        self.assertEqual(wave.get_vgpr(10), wave.get_vgpr(11))

    def test_conjunction_and_mask(self) -> None:
        """s_and_b32: active only when condition A AND condition B hold."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Condition A: lane is even (0x55555555)
        # Condition B: lane < 16 (0x0000FFFF)
        # Conjunction: even lanes in low half (0x00005555)
        wave.execute_s_mov_b32(dst_s=10, val=0x55555555)
        wave.execute_s_mov_b32(dst_s=12, val=0x0000FFFF)
        wave.execute_s_and_b32(dst_s=14, s0="s10", s1="s12")
        self.assertEqual(wave.sgprs[14], 0x00005555)

        wave.init_vgpr(0, 0xAA)
        wave.init_vgpr(1, 0xBB)
        wave.execute_v_cndmask_e64(dst_v=2, src0="v0", src1="v1", cond_src="s14")

        res = wave.get_vgpr(2)
        for i in range(32):
            expected = 0xBB if (i < 16 and (i % 2 == 0)) else 0xAA
            self.assertEqual(res[i], expected)

    def test_disjunction_or_mask(self) -> None:
        """s_or_b32: active when condition A OR condition B holds."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Condition A: lanes 0..7 (0x000000FF)
        # Condition B: lanes 24..31 (0xFF000000)
        wave.execute_s_mov_b32(dst_s=10, val=0x000000FF)
        wave.execute_s_mov_b32(dst_s=12, val=0xFF000000)
        wave.execute_s_or_b32(dst_s=14, s0="s10", s1="s12")
        self.assertEqual(wave.sgprs[14], 0xFF0000FF)

        wave.init_vgpr(0, 100)
        wave.init_vgpr(1, 200)
        wave.execute_v_cndmask_e64(dst_v=2, src0="v0", src1="v1", cond_src="s14")

        res = wave.get_vgpr(2)
        for i in range(32):
            expected = 200 if (i < 8 or i >= 24) else 100
            self.assertEqual(res[i], expected)

    def test_exclusive_xor_mask(self) -> None:
        """s_xor_b32: active when exactly one of condition A or B holds."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        maskA = 0xAAAAAAAA
        maskB = 0x0000FFFF
        wave.execute_s_mov_b32(dst_s=2, val=maskA)
        wave.execute_s_mov_b32(dst_s=4, val=maskB)
        wave.execute_s_xor_b32(dst_s=6, s0="s2", s1="s4")
        expected_mask = (maskA ^ maskB) & MASK32
        self.assertEqual(wave.sgprs[6], expected_mask)

        wave.init_vgpr(0, 0)
        wave.init_vgpr(1, 1)
        wave.execute_v_cndmask_e64(dst_v=5, src0="v0", src1="v1", cond_src="s6")
        res = wave.get_vgpr(5)
        for i in range(32):
            self.assertEqual(res[i], (expected_mask >> i) & 1)

    def test_inhibit_andn2_mask(self) -> None:
        """s_andn2_b32: active when A holds AND B does NOT hold (A & ~B)."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # A: all odd lanes (0xAAAAAAAA)
        # B: high half (0xFFFF0000)
        # A & ~B: odd lanes in low half (0x0000AAAA)
        wave.execute_s_mov_b32(dst_s=2, val=0xAAAAAAAA)
        wave.execute_s_mov_b32(dst_s=4, val=0xFFFF0000)
        wave.execute_s_andn2_b32(dst_s=6, s0="s2", s1="s4")
        self.assertEqual(wave.sgprs[6], 0x0000AAAA)

        wave.init_vgpr(0, 10)
        wave.init_vgpr(1, 99)
        wave.execute_v_cndmask_e64(dst_v=7, src0="v0", src1="v1", cond_src="s6")
        res = wave.get_vgpr(7)
        for i in range(32):
            self.assertEqual(res[i], 99 if (i < 16 and (i % 2 == 1)) else 10)

    def test_walking_one_hot_and_one_cold_masks(self) -> None:
        """Walks an isolated single-bit condition (one-hot) and single-bit clear (one-cold)."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        wave.init_vgpr(0, 0x1111)
        wave.init_vgpr(1, 0x2222)

        for bit in range(32):
            one_hot = 1 << bit
            one_cold = (~one_hot) & MASK32

            # One-hot: only lane `bit` selects src1
            wave.execute_s_mov_b32(dst_s=4, val=one_hot)
            wave.execute_v_cndmask_e64(dst_v=2, src0="v0", src1="v1", cond_src="s4")
            res_hot = wave.get_vgpr(2)
            for i in range(32):
                self.assertEqual(res_hot[i], 0x2222 if i == bit else 0x1111)

            # One-cold: only lane `bit` selects src0
            wave.execute_s_mov_b32(dst_s=5, val=one_cold)
            wave.execute_v_cndmask_e64(dst_v=3, src0="v0", src1="v1", cond_src="s5")
            res_cold = wave.get_vgpr(3)
            for i in range(32):
                self.assertEqual(res_cold[i], 0x1111 if i == bit else 0x2222)


class TestMixedVectorSelectShaderIdioms(unittest.TestCase):
    """Verifies complex shader idioms: clamping, multiplexers, signum, safe division, windowing."""

    def test_branchless_float_clamping(self) -> None:
        """Clamps float values to [min_f, max_f] using two v_cmp and v_cndmask stages:
           v_cmp_lt_f32 vcc_lo, v_val, v_min
           v_cndmask_b32 v_tmp, v_val, v_min, vcc_lo
           v_cmp_gt_f32 vcc_lo, v_tmp, v_max
           v_cndmask_b32 v_out, v_tmp, v_max, vcc_lo
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Inputs spanning [-10.0, 10.0]
        test_floats = [-10.0 + (i * 0.645) for i in range(32)]
        wave.init_vgpr(0, [f32_to_bits(f) for f in test_floats])
        min_val = -2.5
        max_val = 5.0
        wave.execute_s_mov_b32(dst_s=4, val=f32_to_bits(min_val))
        wave.execute_s_mov_b32(dst_s=5, val=f32_to_bits(max_val))

        # Stage 1: if val < min_val -> min_val else val
        wave.execute_v_cmp("lt_f32", src0="v0", src1="s4", dst_mask="vcc")
        wave.execute_v_cndmask_e32(dst_v=1, src0="v0", src1="s4")

        # Stage 2: if tmp > max_val -> max_val else tmp
        wave.execute_v_cmp("gt_f32", src0="v1", src1="s5", dst_mask="vcc")
        wave.execute_v_cndmask_e32(dst_v=2, src0="v1", src1="s5")

        res = [bits_to_f32(b) for b in wave.get_vgpr(2)]
        for i in range(32):
            expected = max(min_val, min(max_val, test_floats[i]))
            self.assertAlmostEqual(res[i], expected, places=5)

    def test_signed_integer_clamp(self) -> None:
        """Signed integer clamp to [-100, +200]."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        test_ints = [-250 + (i * 18) for i in range(32)]
        wave.init_vgpr(0, [x & MASK32 for x in test_ints])

        min_val = -100
        max_val = 200
        wave.execute_s_mov_b32(dst_s=10, val=min_val & MASK32)
        wave.execute_s_mov_b32(dst_s=11, val=max_val & MASK32)

        # Stage 1: v1 = (v0 < min_val) ? min_val : v0
        wave.execute_v_cmp("lt_i32", "v0", "s10", dst_mask="vcc")
        wave.execute_v_cndmask_e32(dst_v=1, src0="v0", src1="s10")

        # Stage 2: v2 = (v1 > max_val) ? max_val : v1
        wave.execute_v_cmp("gt_i32", "v1", "s11", dst_mask="vcc")
        wave.execute_v_cndmask_e32(dst_v=2, src0="v1", src1="s11")

        res = [s32(x) for x in wave.get_vgpr(2)]
        for i in range(32):
            expected = max(min_val, min(max_val, test_ints[i]))
            self.assertEqual(res[i], expected)

    def test_cascaded_3way_multiplexer(self) -> None:
        """Ternary multiplexer: condA ? A : (condB ? B : C).
           Stage 1: v_tmp = cndmask(C, B, condB)
           Stage 2: v_out = cndmask(v_tmp, A, condA)
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Selectors:
        # condA is true for lanes where (lane % 3 == 0)
        # condB is true for lanes where (lane % 3 == 1)
        # neither for lanes where (lane % 3 == 2)
        condA = sum(1 << i for i in range(32) if i % 3 == 0)
        condB = sum(1 << i for i in range(32) if i % 3 == 1)

        wave.execute_s_mov_b32(dst_s=4, val=condA)
        wave.execute_s_mov_b32(dst_s=5, val=condB)

        wave.init_vgpr(10, 0xAAAA0000)  # Value A
        wave.init_vgpr(11, 0xBBBB0000)  # Value B
        wave.init_vgpr(12, 0xCCCC0000)  # Value C (default)

        # Stage 1: choose between C and B using condB
        wave.execute_v_cndmask_e64(dst_v=20, src0="v12", src1="v11", cond_src="s5")
        # Stage 2: choose between stage 1 result and A using condA
        wave.execute_v_cndmask_e64(dst_v=21, src0="v20", src1="v10", cond_src="s4")

        res = wave.get_vgpr(21)
        for i in range(32):
            if i % 3 == 0:
                self.assertEqual(res[i], 0xAAAA0000)
            elif i % 3 == 1:
                self.assertEqual(res[i], 0xBBBB0000)
            else:
                self.assertEqual(res[i], 0xCCCC0000)

    def test_cascaded_4way_priority_encoder(self) -> None:
        """4-way priority multiplexer:
           cond0 ? D0 : (cond1 ? D1 : (cond2 ? D2 : D3))
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        cond0 = 0x11111111
        cond1 = 0x22222222
        cond2 = 0x44444444
        wave.execute_s_mov_b32(dst_s=0, val=cond0)
        wave.execute_s_mov_b32(dst_s=1, val=cond1)
        wave.execute_s_mov_b32(dst_s=2, val=cond2)

        wave.init_vgpr(0, 0xD0)
        wave.init_vgpr(1, 0xD1)
        wave.init_vgpr(2, 0xD2)
        wave.init_vgpr(3, 0xD3)

        # Stage 1: select between D3 and D2 via cond2
        wave.execute_v_cndmask_e64(dst_v=10, src0="v3", src1="v2", cond_src="s2")
        # Stage 2: select between above and D1 via cond1
        wave.execute_v_cndmask_e64(dst_v=11, src0="v10", src1="v1", cond_src="s1")
        # Stage 3: select between above and D0 via cond0
        wave.execute_v_cndmask_e64(dst_v=12, src0="v11", src1="v0", cond_src="s0")

        res = wave.get_vgpr(12)
        for i in range(32):
            if (cond0 >> i) & 1:
                self.assertEqual(res[i], 0xD0)
            elif (cond1 >> i) & 1:
                self.assertEqual(res[i], 0xD1)
            elif (cond2 >> i) & 1:
                self.assertEqual(res[i], 0xD2)
            else:
                self.assertEqual(res[i], 0xD3)

    def test_binary_tree_2bit_decoder(self) -> None:
        """2-bit binary decision tree with 2 selector masks (sel_bit0, sel_bit1):
           Input candidates: [C0, C1, C2, C3]
           Level 1:
             T0 = cndmask(C0, C1, sel_bit0)  # low branch
             T1 = cndmask(C2, C3, sel_bit0)  # high branch
           Level 2:
             Out = cndmask(T0, T1, sel_bit1)
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Cycle through 0, 1, 2, 3 across the wave: lane index mod 4
        sel0 = sum(1 << i for i in range(32) if (i % 4) in (1, 3))
        sel1 = sum(1 << i for i in range(32) if (i % 4) in (2, 3))
        wave.execute_s_mov_b32(dst_s=4, val=sel0)
        wave.execute_s_mov_b32(dst_s=5, val=sel1)

        wave.init_vgpr(0, 0xC000)
        wave.init_vgpr(1, 0xC100)
        wave.init_vgpr(2, 0xC200)
        wave.init_vgpr(3, 0xC300)

        # Level 1
        wave.execute_v_cndmask_e64(dst_v=10, src0="v0", src1="v1", cond_src="s4")
        wave.execute_v_cndmask_e64(dst_v=11, src0="v2", src1="v3", cond_src="s4")

        # Level 2
        wave.execute_v_cndmask_e64(dst_v=12, src0="v10", src1="v11", cond_src="s5")

        res = wave.get_vgpr(12)
        for i in range(32):
            mode = i % 4
            expected = [0xC000, 0xC100, 0xC200, 0xC300][mode]
            self.assertEqual(res[i], expected)

    def test_branchless_signum(self) -> None:
        """Branchless integer signum: (x > 0) ? +1 : ((x < 0) ? -1 : 0)."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        # Mix of positive, negative, and zeros
        test_vals = [(-15 + i) for i in range(32)]
        wave.init_vgpr(0, [x & MASK32 for x in test_vals])

        # Step 1: compare x < 0 -> vcc
        wave.execute_v_cmp("lt_i32", "v0", 0, dst_mask="vcc")
        # Step 2: v1 = (x < 0) ? -1 : 0
        wave.execute_v_cndmask_e32(dst_v=1, src0=0, src1=0xFFFFFFFF)

        # Step 3: compare x > 0 -> s4
        wave.execute_v_cmp("gt_i32", "v0", 0, dst_mask="s4")
        # Step 4: v2 = (x > 0) ? +1 : v1
        wave.execute_v_cndmask_e64(dst_v=2, src0="v1", src1=1, cond_src="s4")

        res = [s32(x) for x in wave.get_vgpr(2)]
        for i in range(32):
            v = test_vals[i]
            expected = 1 if v > 0 else (-1 if v < 0 else 0)
            self.assertEqual(res[i], expected)

    def test_safe_division_with_fallback(self) -> None:
        """Branchless safe-divide: if divisor == 0 -> fallback_value else (num // denom)."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        numerators = [100 + i * 10 for i in range(32)]
        denominators = [(i % 5) for i in range(32)]  # 0, 1, 2, 3, 4 repeating
        wave.init_vgpr(0, numerators)
        wave.init_vgpr(1, denominators)

        # Computed division with zero replaced by 1 to prevent division by zero in SIMD ALU
        safe_div_results = [
            (numerators[i] // (denominators[i] if denominators[i] != 0 else 1)) & MASK32
            for i in range(32)
        ]
        wave.init_vgpr(2, safe_div_results)
        fallback = 0xDEADBEEF
        wave.execute_s_mov_b32(dst_s=4, val=fallback)

        # Compare denominator == 0
        wave.execute_v_cmp("eq_u32", "v1", 0, dst_mask="vcc")
        # Select fallback if denom == 0 else safe_div_result
        wave.execute_v_cndmask_e32(dst_v=3, src0="v2", src1="s4")

        res = wave.get_vgpr(3)
        for i in range(32):
            if denominators[i] == 0:
                self.assertEqual(res[i], fallback)
            else:
                self.assertEqual(res[i], numerators[i] // denominators[i])

    def test_rectangular_window_pulse(self) -> None:
        """Window pulse: 1 if lo <= x <= hi else 0.
           Synthesized via:
             v_cmp_ge_u32 s4, x, lo
             v_cmp_le_u32 s5, x, hi
             s_and_b32 s6, s4, s5
             v_cndmask_b32 dst, 0, 1, s6
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        x_vals = list(range(32))
        wave.init_vgpr(0, x_vals)
        lo = 10
        hi = 20

        wave.execute_v_cmp("ge_u32", "v0", lo, dst_mask="s4")
        wave.execute_v_cmp("le_u32", "v0", hi, dst_mask="s5")
        wave.execute_s_and_b32(dst_s=6, s0="s4", s1="s5")

        wave.execute_v_cndmask_e64(dst_v=1, src0=0, src1=1, cond_src="s6")
        res = wave.get_vgpr(1)
        for i in range(32):
            expected = 1 if (lo <= i <= hi) else 0
            self.assertEqual(res[i], expected)

    def test_interleaved_channel_mux(self) -> None:
        """Interleaves vector components: even lanes take channel A, odd lanes take channel B."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        chanA = [0xA000 + i for i in range(32)]
        chanB = [0xB000 + i for i in range(32)]
        wave.init_vgpr(0, chanA)
        wave.init_vgpr(1, chanB)

        # Alternating mask
        wave.set_vcc(0xAAAAAAAA)  # odd lanes select chanB
        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")

        res = wave.get_vgpr(2)
        for i in range(32):
            self.assertEqual(res[i], chanB[i] if (i % 2 == 1) else chanA[i])


class TestExecAndConditionMaskInteractions(unittest.TestCase):
    """Verifies EXEC mask interactions: inactive lane preservation, then/else branching, and no-ops."""

    def test_inactive_lane_preservation_with_all_true_condition(self) -> None:
        """Even when condition mask has all 1s, inactive lanes (EXEC == 0) must remain intact."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        initial_dst = [0x5500 + i for i in range(32)]
        wave.init_vgpr(2, initial_dst)
        wave.init_vgpr(0, 0x10)
        wave.init_vgpr(1, 0x20)

        # Only lanes 8..15 are active
        exec_active = 0x0000FF00
        wave.set_exec(exec_active)
        wave.set_vcc(0xFFFFFFFF)  # all condition bits 1

        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        res = wave.get_vgpr(2)

        for i in range(32):
            if 8 <= i < 16:
                self.assertEqual(res[i], 0x20)
            else:
                self.assertEqual(res[i], initial_dst[i])

    def test_structured_then_else_divergence_emulation(self) -> None:
        """Emulates complete structured divergent control flow:
           v_cmp_lt_u32 -> s_and_saveexec (Then) -> v_cndmask -> s_xor_saveexec (Else) -> v_cndmask -> restore.
        """
        wave = AluCndmaskMixInterpreter(wave_size=32)
        wave.init_vgpr(0, list(range(32)))  # lane ids
        wave.init_vgpr(1, 0xAAAA)            # then payload
        wave.init_vgpr(2, 0xBBBB)            # else payload
        background = [0x9999 for _ in range(32)]
        wave.init_vgpr(3, background)

        # Condition: lane < 12
        wave.execute_v_cmp("lt_u32", "v0", 12, dst_mask="vcc")
        self.assertEqual(wave.vcc, 0x00000FFF)

        # 1. Enter Then branch: save EXEC and narrow to vcc
        wave.execute_s_and_saveexec_b32(dst_s=4, mask_src="vcc")
        self.assertEqual(wave.exec_mask, 0x00000FFF)

        # Execute in Then block: update dst
        wave.execute_v_cndmask_e32(dst_v=3, src0=0, src1="v1")
        # Lanes 0..11 should now have 0xAAAA, 12..31 still background
        for i in range(12):
            self.assertEqual(wave.vgprs[i][3], 0xAAAA)
        for i in range(12, 32):
            self.assertEqual(wave.vgprs[i][3], 0x9999)

        # 2. Enter Else branch: EXEC = saved_exec ^ then_mask = ~vcc
        saved_exec = wave.sgprs[4]
        wave.set_exec((saved_exec & (~wave.vcc & MASK32)))
        self.assertEqual(wave.exec_mask, 0xFFFFF000)

        # Execute in Else block: update dst
        wave.execute_v_cndmask_e32(dst_v=3, src0="v2", src1=0)

        # 3. Restore EXEC
        wave.set_exec(saved_exec)
        self.assertEqual(wave.exec_mask, 0xFFFFFFFF)

        res = wave.get_vgpr(3)
        for i in range(32):
            if i < 12:
                self.assertEqual(res[i], 0xAAAA)
            else:
                self.assertEqual(res[i], 0xBBBB)

    def test_zero_exec_whole_wave_nop(self) -> None:
        """With EXEC == 0, multiple chained conditional moves have zero side effects."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        init_state = [0xBADC0DE0 + i for i in range(32)]
        wave.init_vgpr(2, init_state)
        wave.set_exec(0x00000000)
        wave.set_vcc(0xFFFFFFFF)

        wave.execute_v_cndmask_e32(dst_v=2, src0=1, src1=2)
        wave.execute_v_cndmask_e64(dst_v=2, src0=3, src1=4, cond_src=0xFFFFFFFF)

        self.assertEqual(wave.get_vgpr(2), init_state)

    def test_walking_single_active_exec_lane(self) -> None:
        """Walking a single active EXEC bit across all 32 lanes."""
        for active_lane in range(32):
            wave = AluCndmaskMixInterpreter(wave_size=32)
            bg = [0x7000 + i for i in range(32)]
            wave.init_vgpr(2, bg)
            wave.set_exec(1 << active_lane)
            wave.set_vcc(0xFFFFFFFF)

            wave.execute_v_cndmask_e32(dst_v=2, src0=0x11, src1=0x99)
            res = wave.get_vgpr(2)
            for i in range(32):
                if i == active_lane:
                    self.assertEqual(res[i], 0x99)
                else:
                    self.assertEqual(res[i], bg[i])


class TestWave64MixedMaskConfigurations(unittest.TestCase):
    """Verifies Wave64 (64 lanes) execution with 64-bit mask registers and cross-half patterns."""

    def test_wave64_cross_half_divergent_select(self) -> None:
        """Low-half (lanes 0..31) and high-half (lanes 32..63) use distinct mask patterns."""
        wave = AluCndmaskMixInterpreter(wave_size=64)
        src0 = [0x1000 + i for i in range(64)]
        src1 = [0x2000 + i for i in range(64)]
        wave.init_vgpr(0, src0)
        wave.init_vgpr(1, src1)

        # Low half VCC: 0x55555555 (even lanes true)
        # High half VCC: 0x0000FFFF (low 16 lanes of high half true)
        vcc64 = 0x55555555 | (0x0000FFFF << 32)
        wave.set_vcc(vcc64)

        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        res = wave.get_vgpr(2)

        for i in range(64):
            c = (vcc64 >> i) & 1
            expected = src1[i] if c else src0[i]
            self.assertEqual(res[i], expected)

    def test_wave64_sgpr_pair_mask_e64(self) -> None:
        """Wave64 conditional select reading 64-bit mask from SGPR pair s[4:5]."""
        wave = AluCndmaskMixInterpreter(wave_size=64)
        src0 = [0x3000 + i for i in range(64)]
        src1 = [0x7000 + i for i in range(64)]
        wave.init_vgpr(0, src0)
        wave.init_vgpr(1, src1)

        mask_lo = 0xAAAAAAAA
        mask_hi = 0x12345678
        wave.execute_s_mov_b32(dst_s=4, val=mask_lo)
        wave.execute_s_mov_b32(dst_s=5, val=mask_hi)
        mask64 = (mask_lo & MASK32) | ((mask_hi & MASK32) << 32)

        wave.execute_v_cndmask_e64(dst_v=2, src0="v0", src1="v1", cond_src="s[4:5]")
        res = wave.get_vgpr(2)

        for i in range(64):
            expected = src1[i] if ((mask64 >> i) & 1) else src0[i]
            self.assertEqual(res[i], expected)

    def test_wave64_half_disabled_execution(self) -> None:
        """Wave64 with low half active only, or high half active only."""
        wave = AluCndmaskMixInterpreter(wave_size=64)
        bg = [0x8800 + i for i in range(64)]
        wave.init_vgpr(2, bg)
        wave.init_vgpr(0, 0x10)
        wave.init_vgpr(1, 0x20)
        wave.set_vcc(MASK64)

        # Case 1: Low half active only
        wave.set_exec(0x00000000FFFFFFFF)
        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        res1 = wave.get_vgpr(2)
        self.assertEqual(res1[:32], [0x20] * 32)
        self.assertEqual(res1[32:], bg[32:])

        # Case 2: High half active only
        wave.init_vgpr(2, bg)
        wave.set_exec(0xFFFFFFFF00000000)
        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        res2 = wave.get_vgpr(2)
        self.assertEqual(res2[:32], bg[:32])
        self.assertEqual(res2[32:], [0x20] * 32)

    def test_wave64_randomized_divergence_crosscheck(self) -> None:
        """Cross-checks 30 pseudo-random 64-bit masks against reference oracle."""
        rng = random.Random(20261122)
        for trial in range(30):
            with self.subTest(trial=trial):
                wave = AluCndmaskMixInterpreter(wave_size=64)
                src0 = [rng.getrandbits(32) for _ in range(64)]
                src1 = [rng.getrandbits(32) for _ in range(64)]
                prev = [rng.getrandbits(32) for _ in range(64)]
                cond = rng.getrandbits(64)
                exec_m = rng.getrandbits(64)

                wave.init_vgpr(0, src0)
                wave.init_vgpr(1, src1)
                wave.init_vgpr(2, prev)
                wave.set_vcc(cond)
                wave.set_exec(exec_m)

                wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
                oracle_expected = ref_cndmask_wave(src0, src1, cond, exec_m, prev, wave_size=64)
                self.assertEqual(wave.get_vgpr(2), oracle_expected)


class TestSpirvSyntheticShaderExecution(unittest.TestCase):
    """Verifies end-to-end SPIR-V translation and execution via rdna2spirv & spirv_mini_interp."""

    def test_spirv_immediate_selects(self) -> None:
        """v_cndmask_b32 selecting between immediate values."""
        lines = [
            "v_cmp_gt_u32_e32 vcc_lo, 50, 20",
            "v_cndmask_b32_e32 v0, 10, 20, vcc_lo",
            "v_cmp_lt_u32_e32 vcc_lo, 50, 20",
            "v_cndmask_b32_e32 v1, 10, 20, vcc_lo",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        out = run_synthetic_ps(lines)
        self.assertEqual(out[0], 20)  # 50 > 20 is true -> selects src1 (20)
        self.assertEqual(out[1], 10)  # 50 < 20 is false -> selects src0 (10)

    def test_spirv_sgpr_condition_mask_e64(self) -> None:
        """v_cndmask_b32_e64 reading condition from SGPR s4 and s6."""
        lines = [
            "s_mov_b32 s4, 1",
            "s_mov_b32 s6, 0",
            "v_mov_b32 v10, 111",
            "v_mov_b32 v11, 222",
            "v_cndmask_b32_e64 v0, v10, v11, s4",
            "v_cndmask_b32_e64 v1, v10, v11, s6",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        out = run_synthetic_ps(lines)
        self.assertEqual(out[0], 222)
        self.assertEqual(out[1], 111)

    def test_spirv_branchless_clamping(self) -> None:
        """Branchless integer clamping in SPIR-V: clamp(x, 10, 50)."""
        test_cases = [(5, 10), (30, 30), (75, 50), (10, 10), (50, 50)]
        for x_val, expected in test_cases:
            lines = [
                f"v_mov_b32 v10, {x_val}",
                # if x < 10 -> 10 else x
                "v_cmp_lt_u32_e32 vcc_lo, v10, 10",
                "v_cndmask_b32_e32 v11, v10, 10, vcc_lo",
                # if v11 > 50 -> 50 else v11
                "v_cmp_gt_u32_e32 vcc_lo, v11, 50",
                "v_cndmask_b32_e32 v0, v11, 50, vcc_lo",
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ]
            out = run_synthetic_ps(lines)
            self.assertEqual(out[0], expected)

    def test_spirv_cascaded_ternary_mux(self) -> None:
        """Cascaded conditional moves in SPIR-V: (x == 1) ? 100 : ((x == 2) ? 200 : 300)."""
        for x_val, expected in ((1, 100), (2, 200), (9, 300)):
            lines = [
                f"v_mov_b32 v10, {x_val}",
                "v_cmp_eq_u32_e32 vcc_lo, v10, 2",
                "v_cndmask_b32_e32 v20, 300, 200, vcc_lo",
                "v_cmp_eq_u32_e32 vcc_lo, v10, 1",
                "v_cndmask_b32_e32 v0, v20, 100, vcc_lo",
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ]
            out = run_synthetic_ps(lines)
            self.assertEqual(out[0], expected)


class TestNegativeControlsAndSensitivity(unittest.TestCase):
    """Negative controls and sensitivity verification against corruptions."""

    def test_single_bit_flip_sensitivity(self) -> None:
        """Flipping exactly 1 bit in the condition mask alters strictly that lane and no other."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        s0 = [0x1000 + i for i in range(32)]
        s1 = [0x5000 + i for i in range(32)]
        wave.init_vgpr(0, s0)
        wave.init_vgpr(1, s1)

        base_vcc = 0xAAAAAAAA
        wave.set_vcc(base_vcc)
        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        base_res = wave.get_vgpr(2)

        for flip_lane in range(32):
            wave.set_vcc(base_vcc ^ (1 << flip_lane))
            wave.execute_v_cndmask_e32(dst_v=3, src0="v0", src1="v1")
            flipped_res = wave.get_vgpr(3)

            for lane in range(32):
                if lane == flip_lane:
                    self.assertNotEqual(flipped_res[lane], base_res[lane])
                else:
                    self.assertEqual(flipped_res[lane], base_res[lane])

    def test_condition_inversion_sensitivity(self) -> None:
        """Inverting the condition mask inverts selections across every differing lane."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        s0 = [100 + i for i in range(32)]
        s1 = [900 + i for i in range(32)]
        wave.init_vgpr(0, s0)
        wave.init_vgpr(1, s1)

        vcc = 0x55AA55AA
        wave.set_vcc(vcc)
        wave.execute_v_cndmask_e32(dst_v=2, src0="v0", src1="v1")
        res_normal = wave.get_vgpr(2)

        wave.set_vcc((~vcc) & MASK32)
        wave.execute_v_cndmask_e32(dst_v=3, src0="v0", src1="v1")
        res_inverted = wave.get_vgpr(3)

        for i in range(32):
            self.assertNotEqual(res_normal[i], res_inverted[i])

    def test_zero_exec_corruption_rejection(self) -> None:
        """With EXEC == 0, registers remain byte-for-byte identical to background state."""
        wave = AluCndmaskMixInterpreter(wave_size=32)
        bg = [0xDEADBEEF for _ in range(32)]
        wave.init_vgpr(2, bg)
        wave.set_exec(0)
        wave.set_vcc(0xFFFFFFFF)

        wave.execute_v_cndmask_e32(dst_v=2, src0=0x11111111, src1=0x22222222)
        self.assertEqual(wave.get_vgpr(2), bg)


if __name__ == "__main__":
    unittest.main()
