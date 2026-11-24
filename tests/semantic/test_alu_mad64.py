"""Semantic interpreter tests for RDNA2 v_mad_u64_u32 and v_mad_i64_i32.

{sdst, D.u64} = S0 * S1 + S2 as a 65-bit result (RDNA ISA pseudocode): the 32 x 32 -> 64-bit product, unsigned for
v_mad_u64_u32 and signed for v_mad_i64_i32, plus the 64-bit S2. D is the low 64 bits; sdst, a lane mask, gets bit 64:
the carry out of the unsigned sum, and for the signed one the sign of the exact sum. Each case translates a synthetic
pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a
plain-Python reference. v[0:1] and the sdst bit leave through exp as 16-bit halves ORed into 0x40000000, so no
export channel is a NaN or denormal pattern that a float round trip could change.
"""
from __future__ import annotations

import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

# Polyfill int.bit_count for Python 3.9 compatibility without modifying other files
if not hasattr(int, "bit_count"):
    import ctypes
    _mp = int.__dict__
    _dict_ptr = ctypes.c_void_p.from_address(id(_mp) + 16).value
    _d = ctypes.cast(_dict_ptr, ctypes.py_object).value
    _d["bit_count"] = lambda self: bin(self).count("1")
    ctypes.pythonapi.PyType_Modified(ctypes.py_object(int))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def u32(x: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return x & M32


def u64(x: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return x & M64


def pack_u64_pair(lo: int, hi: int) -> int:
    """Pack low 32-bit and high 32-bit dwords into a 64-bit integer."""
    return (lo & M32) | ((hi & M32) << 32)


def unpack_u64_pair(val: int) -> tuple[int, int]:
    """Unpack a 64-bit integer into a destination register pair (v[D], v[D+1])."""
    return val & M32, (val >> 32) & M32


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def ref_mad_u64_u32(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 v_mad_u64_u32: (D.u64, carry_out_bit 64)."""
    full_sum = (s0 & M32) * (s1 & M32) + (s2 & M64)
    return full_sum & M64, (full_sum >> 64) & 1


def ref_mad_u64_components(s0: int, s1: int, s2: int) -> tuple[int, int, int]:
    """Return (dst_lo, dst_hi, carry_out_bit) for v_mad_u64_u32."""
    res_u64, carry = ref_mad_u64_u32(s0, s1, s2)
    lo, hi = unpack_u64_pair(res_u64)
    return lo, hi, carry


def ref_mad_i64_i32(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 v_mad_i64_i32: (D.u64, sign_bit 64)."""
    exact_sum = s32(s0) * s32(s1) + s64(s2)
    return exact_sum & M64, 1 if exact_sum < 0 else 0


def ref_mad_i64_components(s0: int, s1: int, s2: int) -> tuple[int, int, int]:
    """Return (dst_lo, dst_hi, sign_bit 64) for v_mad_i64_i32."""
    res_u64, sign = ref_mad_i64_i32(s0, s1, s2)
    lo, hi = unpack_u64_pair(res_u64)
    return lo, hi, sign


def ref(op: str, x: int, y: int, z: int) -> tuple:
    """(D.u64, sdst bit) from the 65-bit result."""
    if op == "v_mad_u64_u32":
        return ref_mad_u64_u32(x, y, z)
    return ref_mad_i64_i32(x, y, z)


def parse_vgpr_pair(token: str) -> int:
    """Parse a VGPR pair token like 'v[0:1]' into the base register index."""
    token = token.strip().rstrip(",").lower()
    if not (token.startswith("v[") and token.endswith("]")):
        raise ValueError(f"Invalid VGPR pair syntax: {token}")
    parts = token[2:-1].split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid VGPR pair format: {token}")
    start, end = int(parts[0]), int(parts[1])
    if end != start + 1:
        raise ValueError(f"VGPR pair must be contiguous pair v[X:X+1]: {token}")
    return start


def parse_sgpr_pair(token: str) -> int | None:
    """Parse an SGPR pair token like 's[10:11]' or 'null'."""
    token = token.strip().rstrip(",").lower()
    if token == "null":
        return None
    if token in ("vcc", "vcc_lo"):
        return 106
    if not (token.startswith("s[") and token.endswith("]")):
        raise ValueError(f"Invalid SGPR pair syntax: {token}")
    parts = token[2:-1].split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid SGPR pair format: {token}")
    start, end = int(parts[0]), int(parts[1])
    if end != start + 1:
        raise ValueError(f"SGPR pair must be contiguous pair s[X:X+1]: {token}")
    return start


class Mad64SimdInterpreter:
    """Deterministic SIMD wavefront interpreter for RDNA2 64-bit widening multiply-add instructions."""

    def __init__(self, num_lanes: int = 64, num_vgprs: int = 256, num_sgprs: int = 108) -> None:
        self.num_lanes = num_lanes
        self.num_vgprs = num_vgprs
        self.num_sgprs = num_sgprs
        self.vgprs: list[list[int]] = [[0] * num_vgprs for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * num_sgprs
        self.exec_mask: int = (1 << num_lanes) - 1

    def reset(self) -> None:
        for lane in range(self.num_lanes):
            self.vgprs[lane] = [0] * self.num_vgprs
        self.sgprs = [0] * self.num_sgprs
        self.exec_mask = (1 << self.num_lanes) - 1

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of bounds")
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR v{reg_idx} out of bounds")
        self.vgprs[lane][reg_idx] = u32(val)

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of bounds")
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR v{reg_idx} out of bounds")
        return self.vgprs[lane][reg_idx]

    def set_vgpr_pair(self, lane: int, reg_idx: int, val: int) -> None:
        self.set_vgpr(lane, reg_idx, val & M32)
        self.set_vgpr(lane, reg_idx + 1, (val >> 32) & M32)

    def get_vgpr_pair(self, lane: int, reg_idx: int) -> int:
        lo = self.get_vgpr(lane, reg_idx)
        hi = self.get_vgpr(lane, reg_idx + 1)
        return pack_u64_pair(lo, hi)

    def set_sgpr(self, reg_idx: int, val: int) -> None:
        if not (0 <= reg_idx < self.num_sgprs):
            raise IndexError(f"SGPR s{reg_idx} out of bounds")
        self.sgprs[reg_idx] = u32(val)

    def get_sgpr(self, reg_idx: int) -> int:
        if not (0 <= reg_idx < self.num_sgprs):
            raise IndexError(f"SGPR s{reg_idx} out of bounds")
        return self.sgprs[reg_idx]

    def set_sgpr_pair(self, reg_idx: int, val: int) -> None:
        self.set_sgpr(reg_idx, val & M32)
        self.set_sgpr(reg_idx + 1, (val >> 32) & M32)

    def get_sgpr_pair(self, reg_idx: int) -> int:
        lo = self.get_sgpr(reg_idx)
        hi = self.get_sgpr(reg_idx + 1)
        return pack_u64_pair(lo, hi)

    def parse_src32(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        low = token.lower()
        if low.startswith("v") and low[1:].isdigit():
            return self.get_vgpr(lane, int(low[1:]))
        if low.startswith("s") and low[1:].isdigit():
            return self.get_sgpr(int(low[1:]))
        if low.startswith("0x") or low.startswith("-0x"):
            return u32(int(token, 16))
        return u32(int(token, 10))

    def parse_src64(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        low = token.lower()
        if low.startswith("v[") and low.endswith("]"):
            idx = parse_vgpr_pair(token)
            return self.get_vgpr_pair(lane, idx)
        if low.startswith("s[") and low.endswith("]"):
            idx = parse_sgpr_pair(token)
            if idx is None:
                return 0
            return self.get_sgpr_pair(idx)
        val = int(token, 16) if (low.startswith("0x") or low.startswith("-0x")) else int(token, 10)
        return u64(val)

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return
        parts = line.replace(",", " ").split()
        op = parts[0].lower()
        operands = parts[1:]

        if op == "v_mov_b32":
            dst_token = operands[0].lower()
            if not (dst_token.startswith("v") and dst_token[1:].isdigit()):
                raise ValueError(f"v_mov_b32 dst must be VGPR: {dst_token}")
            dst_idx = int(dst_token[1:])
            for lane in range(self.num_lanes):
                if (self.exec_mask & (1 << lane)) != 0:
                    val = self.parse_src32(operands[1], lane)
                    self.set_vgpr(lane, dst_idx, val)

        elif op == "s_mov_b32":
            dst_token = operands[0].lower()
            if dst_token.startswith("s") and dst_token[1:].isdigit():
                val = self.parse_src32(operands[1], 0)
                self.set_sgpr(int(dst_token[1:]), val)
            else:
                raise ValueError(f"Unsupported s_mov_b32 dst: {dst_token}")

        elif op == "s_mov_b64":
            dst_token = operands[0].lower()
            val = self.parse_src64(operands[1], 0)
            if dst_token == "exec":
                self.exec_mask = val & ((1 << self.num_lanes) - 1)
            else:
                idx = parse_sgpr_pair(dst_token)
                if idx is not None:
                    self.set_sgpr_pair(idx, val)

        elif op == "v_mad_u64_u32":
            dst_idx = parse_vgpr_pair(operands[0])
            sdst_idx = parse_sgpr_pair(operands[1])
            carry_mask = 0
            for lane in range(self.num_lanes):
                if (self.exec_mask & (1 << lane)) != 0:
                    s0 = self.parse_src32(operands[2], lane)
                    s1 = self.parse_src32(operands[3], lane)
                    s2 = self.parse_src64(operands[4], lane)
                    res_u64, carry = ref_mad_u64_u32(s0, s1, s2)
                    self.set_vgpr_pair(lane, dst_idx, res_u64)
                    if carry:
                        carry_mask |= (1 << lane)
            if sdst_idx is not None:
                self.set_sgpr_pair(sdst_idx, carry_mask)

        elif op == "v_mad_i64_i32":
            dst_idx = parse_vgpr_pair(operands[0])
            sdst_idx = parse_sgpr_pair(operands[1])
            sign_mask = 0
            for lane in range(self.num_lanes):
                if (self.exec_mask & (1 << lane)) != 0:
                    s0 = self.parse_src32(operands[2], lane)
                    s1 = self.parse_src32(operands[3], lane)
                    s2 = self.parse_src64(operands[4], lane)
                    res_u64, sign = ref_mad_i64_i32(s0, s1, s2)
                    self.set_vgpr_pair(lane, dst_idx, res_u64)
                    if sign:
                        sign_mask |= (1 << lane)
            if sdst_idx is not None:
                self.set_sgpr_pair(sdst_idx, sign_mask)

        elif op == "s_endpgm":
            return

        else:
            raise ValueError(f"Unknown instruction: {op}")

    def run_program(self, lines: list[str]) -> None:
        for line in lines:
            self.execute_line(line)


SPECIAL32 = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678]
SPECIAL64 = [0, 1, M64, 1 << 63, (1 << 63) - 1, 0xFFFFFFFF, 1 << 32, 0xFFFFFFFE00000001]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mad64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


HALVES = ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_and_b32 v7, 0xffff, v1",
          "v_lshrrev_b32 v8, 16, v1", "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6",
          "v_or_b32 v7, 0x40000000, v7", "v_or_b32 v8, 0x40000000, v8"]


def run_lines(lines, sdst: str = "s[10:11]") -> tuple:
    """Run the lines, return (v[0:1] as one 64-bit value, 1 if lane 0's bit of sdst is set)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = HALVES + [f"v_cndmask_b32 v9, 0x40000000, 0x40000001, {sdst}"]
    exports = "\texp mrt0 v5, v6, v7, v8 vm\n\texp mrt1 v9, v9, v9, v9 done vm\n"
    asm = "; test_mad64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail) + exports + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    w, flag = out["%out_mrt0"], out["%out_mrt1"][0]
    assert all(x >> 16 == 0x4000 for x in w) and flag >> 1 == 0x20000000, ([hex(x) for x in w], hex(flag))
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w)), flag & 1


def run_op(op: str, x: int, y: int, z: int) -> tuple:
    return run_lines([f"v_mov_b32 v2, {x:#x}", f"v_mov_b32 v3, {y:#x}", f"v_mov_b32 v12, {z & M32:#x}",
                      f"v_mov_b32 v13, {z >> 32:#x}", "s_mov_b32 s10, 0", "s_mov_b32 s11, 0",
                      f"{op} v[0:1], s[10:11], v2, v3, v[12:13]"])


class MadU64ReferenceOracleTests(unittest.TestCase):
    """Mathematical reference oracle tests for unsigned 64-bit widening multiply-add v_mad_u64_u32."""

    def test_zero_operands(self):
        # 0 * y + z = z
        for z in (0, 1, 0x1234567890ABCDEF, M64):
            self.assertEqual(ref_mad_u64_u32(0, 0xDEADBEEF, z), (z, 0))
            self.assertEqual(ref_mad_u64_u32(0xCAFEBABE, 0, z), (z, 0))
            self.assertEqual(ref_mad_u64_u32(0, 0, z), (z, 0))

    def test_identity_multiplication(self):
        # S0 * 1 + S2 = S0 + S2
        self.assertEqual(ref_mad_u64_u32(12345, 1, 0), (12345, 0))
        self.assertEqual(ref_mad_u64_u32(M32, 1, 0), (M32, 0))
        self.assertEqual(ref_mad_u64_u32(1, M32, 1), (M32 + 1, 0))

    def test_widening_pure_multiplication(self):
        # 32-bit x 32-bit -> 64-bit result without accumulator (S2 = 0)
        self.assertEqual(ref_mad_u64_u32(0x10000, 0x10000, 0), (0x100000000, 0))
        self.assertEqual(ref_mad_u64_u32(0x12345678, 0x87654321, 0), (0x12345678 * 0x87654321, 0))

    def test_max_32bit_square(self):
        # (2^32 - 1)^2 = 2^64 - 2*2^32 + 1 = 0xFFFFFFFE00000001
        res, carry = ref_mad_u64_u32(M32, M32, 0)
        self.assertEqual(res, 0xFFFFFFFE00000001)
        self.assertEqual(carry, 0)

    def test_high_bit_powers_of_two(self):
        # High bit set in 32-bit operand is interpreted as unsigned (not sign-extended)
        res, carry = ref_mad_u64_u32(0x80000000, 2, 0)
        self.assertEqual(res, 0x100000000)
        self.assertEqual(carry, 0)

        res, carry = ref_mad_u64_u32(0x80000000, 0x80000000, 0)
        self.assertEqual(res, 0x4000000000000000)
        self.assertEqual(carry, 0)

    def test_destination_register_pair_decomposition(self):
        # Register pair v[D:D+1] receiving (v[D]=low 32-bits, v[D+1]=high 32-bits)
        lo, hi, carry = ref_mad_u64_components(M32, M32, 0)
        self.assertEqual(lo, 0x00000001)
        self.assertEqual(hi, 0xFFFFFFFE)
        self.assertEqual(carry, 0)
        self.assertEqual(pack_u64_pair(lo, hi), 0xFFFFFFFE00000001)

    def test_accumulation_carry_out_boundary(self):
        # S0 * S1 + S2: carry out occurs when exact sum >= 2^64
        # Just below boundary: carry = 0
        res, carry = ref_mad_u64_u32(1, 1, M64 - 1)
        self.assertEqual(res, M64)
        self.assertEqual(carry, 0)

        # Exact boundary: 1 * 1 + M64 = 2^64 -> res = 0, carry = 1
        res, carry = ref_mad_u64_u32(1, 1, M64)
        self.assertEqual(res, 0)
        self.assertEqual(carry, 1)

    def test_max_possible_65bit_sum(self):
        # (2^32 - 1)^2 + (2^64 - 1) = 0xFFFFFFFE00000001 + 0xFFFFFFFFFFFFFFFF = 0x1FFFFFFFE00000000
        res, carry = ref_mad_u64_u32(M32, M32, M64)
        self.assertEqual(res, 0xFFFFFFFE00000000)
        self.assertEqual(carry, 1)

    def test_prng_seeded_u64_verification(self):
        rng = random.Random(0x42)
        for _ in range(100):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(64)
            expected_sum = (s0 * s1) + s2
            expected_res = expected_sum & M64
            expected_carry = (expected_sum >> 64) & 1
            self.assertEqual(ref_mad_u64_u32(s0, s1, s2), (expected_res, expected_carry))


class MadI64ReferenceOracleTests(unittest.TestCase):
    """Mathematical reference oracle tests for signed 64-bit widening multiply-add v_mad_i64_i32."""

    def test_signed_zero_and_identity(self):
        # 0 * y + z = z
        for z in (0, 1, 0x1234567890ABCDEF, M64):
            self.assertEqual(ref_mad_i64_i32(0, 0x7FFFFFFF, z), (z & M64, 1 if s64(z) < 0 else 0))
            self.assertEqual(ref_mad_i64_i32(0x80000000, 0, z), (z & M64, 1 if s64(z) < 0 else 0))

        # +1 identity multiplier
        self.assertEqual(ref_mad_i64_i32(12345, 1, 0), (12345, 0))
        self.assertEqual(ref_mad_i64_i32(0xFFFFFFFF, 1, 0), (M64, 1))  # -1 * 1 = -1

        # -1 multiplier (0xFFFFFFFF)
        self.assertEqual(ref_mad_i64_i32(50, M32, 0), ((-50) & M64, 1))
        self.assertEqual(ref_mad_i64_i32(M32, M32, 0), (1, 0))  # (-1) * (-1) = +1

    def test_positive_times_negative(self):
        # 100 * (-5) = -500
        res, sign = ref_mad_i64_i32(100, (-5) & M32, 0)
        self.assertEqual(res, (-500) & M64)
        self.assertEqual(sign, 1)

    def test_negative_times_negative(self):
        # (-10) * (-20) = +200
        res, sign = ref_mad_i64_i32((-10) & M32, (-20) & M32, 0)
        self.assertEqual(res, 200)
        self.assertEqual(sign, 0)

    def test_extreme_signed_min_product(self):
        # (-2^31) * (-2^31) = +2^62 = 0x4000000000000000
        res, sign = ref_mad_i64_i32(0x80000000, 0x80000000, 0)
        self.assertEqual(res, 0x4000000000000000)
        self.assertEqual(sign, 0)

        # (-2^31) * (2^31 - 1) = -2^62 + 2^31 = -0x3FFFFFFF80000000 = 0xC000000080000000
        res, sign = ref_mad_i64_i32(0x80000000, 0x7FFFFFFF, 0)
        self.assertEqual(res, 0xC000000080000000)
        self.assertEqual(sign, 1)

    def test_signed_destination_register_pair(self):
        # Register pair v[D:D+1] receiving two's complement sign-extended high dword
        lo, hi, sign = ref_mad_i64_components(M32, 1, 0)  # -1
        self.assertEqual(lo, 0xFFFFFFFF)
        self.assertEqual(hi, 0xFFFFFFFF)
        self.assertEqual(sign, 1)
        self.assertEqual(pack_u64_pair(lo, hi), M64)

        lo, hi, sign = ref_mad_i64_components(1, 1, 0)  # +1
        self.assertEqual(lo, 1)
        self.assertEqual(hi, 0)
        self.assertEqual(sign, 0)
        self.assertEqual(pack_u64_pair(lo, hi), 1)

    def test_signed_accumulator_cancellation(self):
        # Exact cancellation to 0
        res, sign = ref_mad_i64_i32((-5) & M32, 4, 20)
        self.assertEqual(res, 0)
        self.assertEqual(sign, 0)

        # Negative sum by 1
        res, sign = ref_mad_i64_i32((-5) & M32, 4, 19)
        self.assertEqual(res, M64)
        self.assertEqual(sign, 1)

    def test_prng_seeded_i64_verification(self):
        rng = random.Random(0x84)
        for _ in range(100):
            s0_raw = rng.getrandbits(32)
            s1_raw = rng.getrandbits(32)
            s2_raw = rng.getrandbits(64)
            s0_val = s32(s0_raw)
            s1_val = s32(s1_raw)
            s2_val = s64(s2_raw)
            exact = s0_val * s1_val + s2_val
            expected_res = exact & M64
            expected_sign = 1 if exact < 0 else 0
            self.assertEqual(ref_mad_i64_i32(s0_raw, s1_raw, s2_raw), (expected_res, expected_sign))


class Mad64SignedUnsignedDivergenceTests(unittest.TestCase):
    """Direct comparison of v_mad_u64_u32 vs v_mad_i64_i32 on identical bit patterns."""

    def test_high_bit_square_divergence(self):
        # Raw operands 0xFFFFFFFF x 0xFFFFFFFF + 0
        u_res, u_carry = ref_mad_u64_u32(M32, M32, 0)
        i_res, i_sign = ref_mad_i64_i32(M32, M32, 0)

        self.assertEqual(u_res, 0xFFFFFFFE00000001)
        self.assertEqual(u_carry, 0)
        self.assertEqual(i_res, 0x0000000000000001)
        self.assertEqual(i_sign, 0)

        u_lo, u_hi = unpack_u64_pair(u_res)
        i_lo, i_hi = unpack_u64_pair(i_res)
        self.assertEqual(u_lo, i_lo)  # low dwords both 1 (since (-1)*(-1) == 1 mod 2^32)
        self.assertNotEqual(u_hi, i_hi)  # high dwords diverge: 0xFFFFFFFE vs 0x00000000
        self.assertEqual(u_hi, 0xFFFFFFFE)
        self.assertEqual(i_hi, 0x00000000)

    def test_high_bit_unit_multiplier_divergence(self):
        # Raw operands 0xFFFFFFFF x 1 + 0
        u_res, u_carry = ref_mad_u64_u32(M32, 1, 0)
        i_res, i_sign = ref_mad_i64_i32(M32, 1, 0)

        self.assertEqual(u_res, 0x00000000FFFFFFFF)
        self.assertEqual(u_carry, 0)
        self.assertEqual(i_res, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(i_sign, 1)

        u_lo, u_hi = unpack_u64_pair(u_res)
        i_lo, i_hi = unpack_u64_pair(i_res)
        self.assertEqual(u_lo, i_lo)  # low dwords both 0xFFFFFFFF
        self.assertEqual(u_hi, 0x00000000)
        self.assertEqual(i_hi, 0xFFFFFFFF)

    def test_msb_scale_divergence(self):
        # Raw operands 0x80000000 x 2 + 0
        u_res, u_carry = ref_mad_u64_u32(0x80000000, 2, 0)
        i_res, i_sign = ref_mad_i64_i32(0x80000000, 2, 0)

        self.assertEqual(u_res, 0x0000000100000000)
        self.assertEqual(u_carry, 0)
        self.assertEqual(i_res, 0xFFFFFFFF00000000)
        self.assertEqual(i_sign, 1)

        u_lo, u_hi = unpack_u64_pair(u_res)
        i_lo, i_hi = unpack_u64_pair(i_res)
        self.assertEqual(u_lo, i_lo)  # low dwords both 0
        self.assertEqual(u_hi, 0x00000001)
        self.assertEqual(i_hi, 0xFFFFFFFF)

    def test_msb_positive_max_divergence(self):
        # Raw operands 0x80000000 x 0x7FFFFFFF + 0
        u_res, u_carry = ref_mad_u64_u32(0x80000000, 0x7FFFFFFF, 0)
        i_res, i_sign = ref_mad_i64_i32(0x80000000, 0x7FFFFFFF, 0)

        self.assertEqual(u_res, 0x3FFFFFFF80000000)
        self.assertEqual(u_carry, 0)
        self.assertEqual(i_res, 0xC000000080000000)
        self.assertEqual(i_sign, 1)

    def test_shader_execution_divergence(self):
        # Verify divergence in compiled SPIR-V fragment execution
        # 0xFFFFFFFF x 0xFFFFFFFF + 0
        u_out = run_op("v_mad_u64_u32", M32, M32, 0)
        i_out = run_op("v_mad_i64_i32", M32, M32, 0)
        self.assertEqual(u_out, (0xFFFFFFFE00000001, 0))
        self.assertEqual(i_out, (1, 0))

        # 0xFFFFFFFF x 1 + 0
        u_out = run_op("v_mad_u64_u32", M32, 1, 0)
        i_out = run_op("v_mad_i64_i32", M32, 1, 0)
        self.assertEqual(u_out, (M32, 0))
        self.assertEqual(i_out, (M64, 1))

        # 0x80000000 x 2 + 0
        u_out = run_op("v_mad_u64_u32", 0x80000000, 2, 0)
        i_out = run_op("v_mad_i64_i32", 0x80000000, 2, 0)
        self.assertEqual(u_out, (0x100000000, 0))
        self.assertEqual(i_out, (0xFFFFFFFF00000000, 1))


class Mad64OverflowModuloTests(unittest.TestCase):
    """64-bit overflow and two's complement modulo arithmetic wrapping tests."""

    def test_unsigned_wrap_modulo_64(self):
        # 1 * 1 + (2^64 - 1) = 2^64 -> wraps to 0 with carry 1
        res, carry = ref_mad_u64_u32(1, 1, M64)
        self.assertEqual(res, 0)
        self.assertEqual(carry, 1)

        # 5 * 1 + (2^64 - 1) = 2^64 + 4 -> wraps to 4 with carry 1
        res, carry = ref_mad_u64_u32(5, 1, M64)
        self.assertEqual(res, 4)
        self.assertEqual(carry, 1)

        # (2^32 - 1)^2 = 2^64 - 2^33 + 1
        # Adding (2^33 - 1) gives exactly 2^64 -> wraps to 0
        acc = (1 << 33) - 1
        res, carry = ref_mad_u64_u32(M32, M32, acc)
        self.assertEqual(res, 0)
        self.assertEqual(carry, 1)

    def test_unsigned_large_product_wrap(self):
        # Product + accumulator > 2^64
        res, carry = ref_mad_u64_u32(M32, M32, M64)
        self.assertEqual(res, 0xFFFFFFFE00000000)
        self.assertEqual(carry, 1)

    def test_signed_negative_underflow_wrap(self):
        # Product is -2^32, accumulator is -2^63 (0x8000000000000000)
        # Sum is -2^63 - 2^32. In 64-bit two's complement: 2^63 - 2^32 = 0x7FFFFFFF00000000
        # Exact sum is negative, so sign bit 64 is 1
        res, sign = ref_mad_i64_i32(0x80000000, 2, 1 << 63)
        self.assertEqual(res, (1 << 63) - (1 << 32))
        self.assertEqual(sign, 1)

    def test_chained_accumulation_modulo_64(self):
        # Verify deterministic wrapping over repeated steps
        acc = 0xFFFFFFFFFFFFFFFE
        for step in range(10):
            acc, carry = ref_mad_u64_u32(2, 2, acc)
        # Each step adds 4. Total added = 40.
        # (2^64 - 2) + 40 = 2^64 + 38 = 38
        self.assertEqual(acc, 38)


class Mad64SimdDivergenceTests(unittest.TestCase):
    """Multi-lane SIMD divergence and execution mask tests using Mad64SimdInterpreter."""

    def test_all_lanes_active(self):
        interp = Mad64SimdInterpreter(num_lanes=64)
        for lane in range(64):
            interp.set_vgpr(lane, 2, lane)
            interp.set_vgpr(lane, 3, 10)
            interp.set_vgpr_pair(lane, 12, 1000)
        interp.run_program([
            "s_mov_b64 exec, -1",
            "v_mad_u64_u32 v[0:1], s[10:11], v2, v3, v[12:13]",
        ])
        for lane in range(64):
            expected = lane * 10 + 1000
            self.assertEqual(interp.get_vgpr_pair(lane, 0), expected)
        # No carry out, so s[10:11] is 0
        self.assertEqual(interp.get_sgpr_pair(10), 0)

    def test_exec_mask_inactive_preservation(self):
        # Even lanes active, odd lanes inactive
        interp = Mad64SimdInterpreter(num_lanes=64)
        sentinel = 0xDEADBEEFCAFEBABE
        for lane in range(64):
            interp.set_vgpr_pair(lane, 0, sentinel)
            interp.set_vgpr(lane, 2, 7)
            interp.set_vgpr(lane, 3, 11)
            interp.set_vgpr_pair(lane, 12, 0)

        # 0x5555555555555555 has bit=1 for even lanes (0, 2, 4...)
        checkerboard = 0x5555555555555555
        interp.run_program([
            f"s_mov_b64 exec, {checkerboard:#x}",
            "v_mad_u64_u32 v[0:1], null, v2, v3, v[12:13]",
        ])
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr_pair(lane, 0), 77, f"Lane {lane} should be updated")
            else:
                self.assertEqual(interp.get_vgpr_pair(lane, 0), sentinel, f"Lane {lane} should be untouched")

    def test_exec_mask_upper_half_disabled(self):
        interp = Mad64SimdInterpreter(num_lanes=64)
        sentinel = 0x1122334455667788
        for lane in range(64):
            interp.set_vgpr_pair(lane, 6, sentinel)
            interp.set_vgpr(lane, 2, 100)
            interp.set_vgpr(lane, 3, 2)
            interp.set_vgpr_pair(lane, 4, 5)

        # Lanes 0..31 active
        interp.run_program([
            "s_mov_b64 exec, 0x00000000FFFFFFFF",
            "v_mad_u64_u32 v[6:7], null, v2, v3, v[4:5]",
        ])
        for lane in range(64):
            if lane < 32:
                self.assertEqual(interp.get_vgpr_pair(lane, 6), 205)
            else:
                self.assertEqual(interp.get_vgpr_pair(lane, 6), sentinel)

    def test_exec_mask_zero_no_op(self):
        interp = Mad64SimdInterpreter(num_lanes=64)
        sentinel = 0x0102030405060708
        for lane in range(64):
            interp.set_vgpr_pair(lane, 0, sentinel)
            interp.set_vgpr(lane, 2, 999)
            interp.set_vgpr(lane, 3, 888)
            interp.set_vgpr_pair(lane, 4, 777)
        interp.run_program([
            "s_mov_b64 exec, 0",
            "v_mad_u64_u32 v[0:1], null, v2, v3, v[4:5]",
        ])
        for lane in range(64):
            self.assertEqual(interp.get_vgpr_pair(lane, 0), sentinel)

    def test_single_lane_exec(self):
        interp = Mad64SimdInterpreter(num_lanes=64)
        target_lane = 42
        for lane in range(64):
            interp.set_vgpr_pair(lane, 0, 0)
            interp.set_vgpr(lane, 2, 50)
            interp.set_vgpr(lane, 3, 20)
            interp.set_vgpr_pair(lane, 4, 3)

        interp.run_program([
            f"s_mov_b64 exec, {1 << target_lane:#x}",
            "v_mad_u64_u32 v[0:1], null, v2, v3, v[4:5]",
        ])
        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr_pair(lane, 0), 1003)
            else:
                self.assertEqual(interp.get_vgpr_pair(lane, 0), 0)

    def test_divergent_lane_mask_generation(self):
        # Lanes 5 and 55 generate carry out; other lanes do not
        interp = Mad64SimdInterpreter(num_lanes=64)
        for lane in range(64):
            interp.set_vgpr(lane, 2, M32)
            interp.set_vgpr(lane, 3, 1 if lane in (5, 55) else 0)
            interp.set_vgpr_pair(lane, 4, M64)

        interp.run_program([
            "s_mov_b64 exec, -1",
            "v_mad_u64_u32 v[0:1], s[10:11], v2, v3, v[4:5]",
        ])
        expected_mask = (1 << 5) | (1 << 55)
        self.assertEqual(interp.get_sgpr_pair(10), expected_mask)


class Mad64NegativeControlsTests(unittest.TestCase):
    """Negative controls, textual sabotage, and invalid syntax rejection tests."""

    def test_unsigned_versus_signed_mismatch_control(self):
        # Negative control: running signed logic on high-bit unsigned inputs MUST diverge
        for x in (0xFFFFFFFF, 0x80000000):
            u_res, _ = ref_mad_u64_u32(x, 2, 0)
            i_res, _ = ref_mad_i64_i32(x, 2, 0)
            self.assertNotEqual(u_res, i_res)

    def test_invalid_register_pair_tokens(self):
        interp = Mad64SimdInterpreter(num_lanes=32)
        # Not a pair
        with self.assertRaises(ValueError):
            interp.execute_line("v_mad_u64_u32 v0, null, v1, v2, v[3:4]")
        # Out of bounds pair
        with self.assertRaises(IndexError):
            interp.execute_line("v_mad_u64_u32 v[300:301], null, v1, v2, v[3:4]")

    def test_unaligned_register_pair(self):
        interp = Mad64SimdInterpreter(num_lanes=32)
        # Non-contiguous pair v[0:2]
        with self.assertRaises(ValueError):
            interp.execute_line("v_mad_u64_u32 v[0:2], null, v1, v2, v[3:4]")
        # Inverted pair v[2:1]
        with self.assertRaises(ValueError):
            interp.execute_line("v_mad_u64_u32 v[2:1], null, v1, v2, v[3:4]")

    def test_unknown_instruction(self):
        interp = Mad64SimdInterpreter(num_lanes=32)
        with self.assertRaises(ValueError):
            interp.execute_line("v_mad_bogus_op v[0:1], null, v2, v3, 0")

    def test_zero_multiplicand_invariance(self):
        # Invariant: if S1 is 0, result must equal S2 regardless of S0
        for s0 in (0, 1, 0x12345678, M32):
            for s2 in (0, 1, 0xABCDEF0123456789, M64):
                res, carry = ref_mad_u64_u32(s0, 0, s2)
                self.assertEqual(res, s2)
                self.assertEqual(carry, 0)


class Mad64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_mad_u64_u32", M32, M32, 0), (0xFFFFFFFE00000001, 0))
        self.assertEqual(ref("v_mad_u64_u32", M32, M32, M64), (0xFFFFFFFE00000000, 1))      # carry out
        self.assertEqual(ref("v_mad_i64_i32", M32, 1, 0), (M64, 1))                         # -1: sign set
        self.assertEqual(ref("v_mad_i64_i32", 0x80000000, 0x80000000, (1 << 63) - 1),     # 2^62 + 2^63 - 1
                         ((1 << 62) + (1 << 63) - 1, 0))

    def test_edge_values(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            for x in SPECIAL32:
                for y in SPECIAL32[::2]:
                    for z in SPECIAL64[::3]:
                        with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                            self.assertEqual(run_op(op, x, y, z), ref(op, x, y, z))

    def test_seeded(self):
        rng = random.Random(0x64)
        for _ in range(40):
            op = rng.choice(["v_mad_u64_u32", "v_mad_i64_i32"])
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(64)
            with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(op, x, y, z), ref(op, x, y, z))

    def test_inline_accumulator_and_null_sdst(self):
        # S2 = -1 as a 64-bit inline constant (sign-extended), and sdst = null: only D is written.
        got = run_lines(["v_mov_b32 v2, 3", "v_mov_b32 v3, 5", "s_mov_b32 s10, 0", "s_mov_b32 s11, 0",
                         "v_mad_u64_u32 v[0:1], null, v2, v3, -1"])
        self.assertEqual(got, (14, 0))
        got = run_lines(["v_mov_b32 v2, 3", "v_mov_b32 v3, 5", "v_mad_u64_u32 v[0:1], s[10:11], v2, v3, -1"])
        self.assertEqual(got, (14, 1))                                     # 15 + (2^64 - 1) carries out

    def test_carry_feeds_v_add_co_ci(self):
        # The sdst mask is a real carry: add it into a third word.
        got = run_lines(["v_mov_b32 v2, -1", "v_mov_b32 v3, -1", "v_mov_b32 v12, -1", "v_mov_b32 v13, -1",
                         "v_mad_u64_u32 v[0:1], vcc_lo, v2, v3, v[12:13]", "v_mov_b32 v14, 41",
                         "v_add_co_ci_u32 v1, vcc_lo, v14, 0, vcc_lo", "v_mov_b32 v0, 0", "s_mov_b32 s10, 0",
                         "s_mov_b32 s11, 0"])
        self.assertEqual(got[0] >> 32, 42)


class Mad64WaveMaskTests(unittest.TestCase):
    """lane_masks.py must track v_mad_u64_u32 / v_mad_i64_i32 sdst as a lane mask across all 64 lanes."""

    def run_wave64_cs(self, name: str, body_lines):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body_lines) + "\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2: 0x8088 -> 4 user SGPRs (s[0:3] = output V#), TGID_X, 64x1x1 threads.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
        pc = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc, memory={},
        )
        return mem

    def test_mad_u64_carry_mask_ballot_wave64(self):
        # Only lanes 3 and 40 carry out (v2=0xffffffff, v3=1, acc=0xffffffff_ffffffff -> carry=1);
        # all other lanes have v3=0 -> carry=0. s_bcnt1_i32_b64 must return 2 and s_ff1_i32_b64 must return 3
        # uniformly in every lane (without lane_masks tracking, lanes 3/40 saw 64 and the rest saw 0).
        mem = self.run_wave64_cs("cs_mad_u64_mask", [
            "s_mov_b64 exec, -1",
            "v_cmp_eq_u32_e64 s[4:5], v0, 3",
            "v_cmp_eq_u32_e64 s[6:7], v0, 40",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cndmask_b32_e64 v3, 0, 1, s[4:5]",
            "v_mov_b32 v2, 0xffffffff",
            "v_mov_b32 v12, 0xffffffff",
            "v_mov_b32 v13, 0xffffffff",
            "v_mad_u64_u32 v[6:7], s[10:11], v2, v3, v[12:13]",
            "s_bcnt1_i32_b64 s12, s[10:11]",
            "s_ff1_i32_b64 s13, s[10:11]",
            "v_mov_b32 v8, s12",
            "v_mov_b32 v9, s13",
            "v_lshlrev_b32 v10, 3, v0",
            "buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen",
        ])
        for lane in range(64):
            self.assertEqual(mem[lane * 8], 2, f"lane {lane} bcnt1")
            self.assertEqual(mem[lane * 8 + 4], 3, f"lane {lane} ff1")

    def test_mad_i64_sign_mask_ballot_wave64(self):
        # Only lanes 7, 35 and 63 get v2 = -1 (with v3 = 1, acc = 0 -> exact sum = -1, bit 64 = 1);
        # all other lanes get v2 = 1 -> sum = +1, bit 64 = 0.
        mem = self.run_wave64_cs("cs_mad_i64_mask", [
            "s_mov_b64 exec, -1",
            "v_cmp_eq_u32_e64 s[4:5], v0, 7",
            "v_cmp_eq_u32_e64 s[6:7], v0, 35",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cmp_eq_u32_e64 s[6:7], v0, 63",
            "s_or_b64 s[4:5], s[4:5], s[6:7]",
            "v_cndmask_b32_e64 v2, 1, -1, s[4:5]",
            "v_mov_b32 v3, 1",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
            "v_mad_i64_i32 v[6:7], s[10:11], v2, v3, v[12:13]",
            "s_bcnt1_i32_b64 s12, s[10:11]",
            "s_ff1_i32_b64 s13, s[10:11]",
            "v_mov_b32 v8, s12",
            "v_mov_b32 v9, s13",
            "v_lshlrev_b32 v10, 3, v0",
            "buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen",
        ])
        for lane in range(64):
            self.assertEqual(mem[lane * 8], 3, f"lane {lane} bcnt1")
            self.assertEqual(mem[lane * 8 + 4], 7, f"lane {lane} ff1")


if __name__ == "__main__":
    unittest.main()
