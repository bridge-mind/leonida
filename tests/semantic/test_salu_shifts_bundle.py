"""Bundle 28: SALU 32/64-bit Shift Operations Suite.

Tests for 6 SALU shift instructions:
- s_lshl_b32: 32-bit scalar logical shift left (shift count & 31, SCC = result != 0)
- s_lshr_b32: 32-bit scalar logical shift right (shift count & 31, SCC = result != 0)
- s_ashr_i32: 32-bit scalar arithmetic shift right (shift count & 31, sign-extended, SCC = result != 0)
- s_lshl_b64: 64-bit scalar logical shift left (shift count & 63, SCC = result != 0)
- s_lshr_b64: 64-bit scalar logical shift right (shift count & 63, SCC = result != 0)
- s_ashr_i64: 64-bit scalar arithmetic shift right (shift count & 63, sign-extended, SCC = result != 0)

Standard library only; zero game data.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

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
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle28") -> list[int]:
    """Helper to translate and run a compute shader; stores outputs at out[lane*32 + word*4]."""
    prologue = ["v_lshlrev_b32_e32 v30, 5, v0"]
    full_body = prologue + list(body) + ["s_endpgm"]
    spv = support.translate_lines(name, full_body, fssm=FSSM)
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    return [mem.get(i * 32, 0) for i in range(num_words)]


class ASTNode:
    """Base class for symbolic AST nodes."""
    pass

def ref_s_lshl_b32(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshl_b32 -> (result, scc)."""
    sh = shift & 31
    res = (src << sh) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc


class TestSLshlB32Oracle(unittest.TestCase):
    def test_basic_shifts(self) -> None:
        self.assertEqual(ref_s_lshl_b32(1, 0), (1, 1))
        self.assertEqual(ref_s_lshl_b32(1, 4), (16, 1))
        self.assertEqual(ref_s_lshl_b32(0x12345678, 8), (0x34567800, 1))

    def test_zero_result_and_scc(self) -> None:
        self.assertEqual(ref_s_lshl_b32(0, 5), (0, 0))
        self.assertEqual(ref_s_lshl_b32(0x80000000, 1), (0, 0))

    def test_shift_count_masking(self) -> None:
        # Shift count is masked with 31: 32 -> 0, 33 -> 1
        self.assertEqual(ref_s_lshl_b32(1, 32), (1, 1))
        self.assertEqual(ref_s_lshl_b32(1, 33), (2, 1))

class SLshlB32Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_lshl_b32(self.src, self.shift)


class TestSLshlB32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshlB32Expr(0x0F, 4)
        self.assertEqual(expr.eval(), (0xF0, 1))
        expr_zero = SLshlB32Expr(0, 10)
        self.assertEqual(expr_zero.eval(), (0, 0))

class TestSLshlB32SCCAndMask(unittest.TestCase):
    def test_scc_condition_code(self) -> None:
        # Shift off all bits
        res, scc = ref_s_lshl_b32(0x00000001, 31)
        self.assertEqual((res, scc), (0x80000000, 1))
        # High bit shifted out by another bit
        res, scc = ref_s_lshl_b32(0x80000000, 1)
        self.assertEqual((res, scc), (0, 0))

    def test_shift_counts_boundary(self) -> None:
        for sh in range(32):
            res, scc = ref_s_lshl_b32(1, sh)
            self.assertEqual(res, 1 << sh)
            self.assertEqual(scc, 1)

class TestSLshlB32Translator(unittest.TestCase):
    def test_translator_s_lshl_b32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x12345678",
            "s_lshl_b32 s5, s4, 4",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshl_b32")
        for l in range(32):
            self.assertEqual(results[l], 0x23456780)

def ref_s_lshr_b32(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshr_b32 -> (result, scc)."""
    sh = shift & 31
    res = (src & MASK32) >> sh
    scc = 1 if res != 0 else 0
    return res, scc


class TestSLshrB32Oracle(unittest.TestCase):
    def test_basic_shifts(self) -> None:
        self.assertEqual(ref_s_lshr_b32(16, 4), (1, 1))
        self.assertEqual(ref_s_lshr_b32(0x12345678, 8), (0x00123456, 1))

    def test_zero_result_and_scc(self) -> None:
        self.assertEqual(ref_s_lshr_b32(0, 4), (0, 0))
        self.assertEqual(ref_s_lshr_b32(1, 1), (0, 0))

    def test_shift_count_masking(self) -> None:
        self.assertEqual(ref_s_lshr_b32(0x10, 32), (0x10, 1))
        self.assertEqual(ref_s_lshr_b32(0x10, 33), (0x08, 1))

class SLshrB32Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_lshr_b32(self.src, self.shift)


class TestSLshrB32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshrB32Expr(0xF0, 4)
        self.assertEqual(expr.eval(), (0x0F, 1))
        expr_zero = SLshrB32Expr(1, 1)
        self.assertEqual(expr_zero.eval(), (0, 0))

class TestSLshrB32SCCAndMask(unittest.TestCase):
    def test_scc_condition_code(self) -> None:
        res, scc = ref_s_lshr_b32(0x80000000, 31)
        self.assertEqual((res, scc), (1, 1))
        res, scc = ref_s_lshr_b32(0x80000000, 32)  # shift 0
        self.assertEqual((res, scc), (0x80000000, 1))

    def test_logical_zero_fill(self) -> None:
        # Highest bit cleared on right shift
        res, _ = ref_s_lshr_b32(0x80000000, 1)
        self.assertEqual(res, 0x40000000)

class TestSLshrB32Translator(unittest.TestCase):
    def test_translator_s_lshr_b32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x12345678",
            "s_lshr_b32 s5, s4, 4",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshr_b32")
        for l in range(32):
            self.assertEqual(results[l], 0x01234567)

def ref_s_ashr_i32(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_ashr_i32 -> (result, scc)."""
    sh = shift & 31
    val = src & MASK32
    sval = val - 0x100000000 if val & 0x80000000 else val
    res = (sval >> sh) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc


class TestSAshrI32Oracle(unittest.TestCase):
    def test_positive_numbers(self) -> None:
        self.assertEqual(ref_s_ashr_i32(0x1000, 4), (0x0100, 1))

    def test_negative_sign_extension(self) -> None:
        # 0x80000000 >> 4 = 0xF8000000
        self.assertEqual(ref_s_ashr_i32(0x80000000, 4), (0xF8000000, 1))
        # -1 (0xFFFFFFFF) >> any = -1 (0xFFFFFFFF)
        self.assertEqual(ref_s_ashr_i32(0xFFFFFFFF, 10), (0xFFFFFFFF, 1))

class SAshrI32Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_ashr_i32(self.src, self.shift)


class TestSAshrI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SAshrI32Expr(0x80000000, 1)
        self.assertEqual(expr.eval(), (0xC0000000, 1))
        expr_pos = SAshrI32Expr(4, 2)
        self.assertEqual(expr_pos.eval(), (1, 1))

class TestSAshrI32SignAndSCC(unittest.TestCase):
    def test_sign_bit_propagation(self) -> None:
        val = 0x80000000
        for sh in range(32):
            res, scc = ref_s_ashr_i32(val, sh)
            # Bit 31 must stay 1
            self.assertEqual(res & 0x80000000, 0x80000000)
            self.assertEqual(scc, 1)

    def test_zero_result(self) -> None:
        res, scc = ref_s_ashr_i32(0, 5)
        self.assertEqual((res, scc), (0, 0))

class TestSAshrI32Translator(unittest.TestCase):
    def test_translator_s_ashr_i32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x80000000",
            "s_ashr_i32 s5, s4, 4",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_ashr_i32")
        for l in range(32):
            self.assertEqual(results[l], 0xF8000000)

def ref_s_lshl_b64(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshl_b64 -> (result, scc)."""
    sh = shift & 63
    res = (src << sh) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


class TestSLshlB64Oracle(unittest.TestCase):
    def test_cross_dword_shift(self) -> None:
        # Shift lower dword into upper dword by 32 bits
        self.assertEqual(ref_s_lshl_b64(0x0000000012345678, 32), (0x1234567800000000, 1))

    def test_shift_count_masking_64(self) -> None:
        self.assertEqual(ref_s_lshl_b64(1, 64), (1, 1))
        self.assertEqual(ref_s_lshl_b64(1, 65), (2, 1))

class SLshlB64Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_lshl_b64(self.src, self.shift)


class TestSLshlB64AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshlB64Expr(0x1, 40)
        self.assertEqual(expr.eval(), (1 << 40, 1))

class TestSLshlB64CrossDwordAndSCC(unittest.TestCase):
    def test_scc_zero_result(self) -> None:
        res, scc = ref_s_lshl_b64(0, 10)
        self.assertEqual((res, scc), (0, 0))
        # 1 shifted left by 63 is non-zero, but shift by another bit wraps via mask 63
        res, scc = ref_s_lshl_b64(1 << 63, 1)
        self.assertEqual((res, scc), (0, 0))

    def test_dword_boundary_transitions(self) -> None:
        val = 0x00000000FFFFFFFF
        res, scc = ref_s_lshl_b64(val, 16)
        self.assertEqual(res, 0x0000FFFFFFFF0000)
        self.assertEqual(scc, 1)

class TestSLshlB64Translator(unittest.TestCase):
    def test_translator_s_lshl_b64_execution(self) -> None:
        body = [
            "s_mov_b64 s[4:5], 0x123456789abcdef0",
            "s_lshl_b64 s[6:7], s[4:5], 8",
            "v_mov_b32_e32 v1, s6",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshl_b64")
        for l in range(32):
            self.assertEqual(results[l], 0xBCDEF000)

def ref_s_lshr_b64(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshr_b64 -> (result, scc)."""
    sh = shift & 63
    res = (src & MASK64) >> sh
    scc = 1 if res != 0 else 0
    return res, scc


class TestSLshrB64Oracle(unittest.TestCase):
    def test_cross_dword_shift(self) -> None:
        # Shift upper dword into lower dword
        self.assertEqual(ref_s_lshr_b64(0x1234567800000000, 32), (0x12345678, 1))

    def test_shift_count_masking_64(self) -> None:
        self.assertEqual(ref_s_lshr_b64(0x100, 64), (0x100, 1))
        self.assertEqual(ref_s_lshr_b64(0x100, 65), (0x80, 1))

class SLshrB64Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_lshr_b64(self.src, self.shift)


class TestSLshrB64AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshrB64Expr(0x10000000000, 40)
        self.assertEqual(expr.eval(), (1, 1))

class TestSLshrB64CrossDwordAndSCC(unittest.TestCase):
    def test_scc_condition_code(self) -> None:
        res, scc = ref_s_lshr_b64(1, 1)
        self.assertEqual((res, scc), (0, 0))

    def test_upper_dword_drain(self) -> None:
        val = 0x8000000000000000
        res, scc = ref_s_lshr_b64(val, 63)
        self.assertEqual((res, scc), (1, 1))

class TestSLshrB64Translator(unittest.TestCase):
    def test_translator_s_lshr_b64_execution(self) -> None:
        body = [
            "s_mov_b64 s[4:5], 0x123456789abcdef0",
            "s_lshr_b64 s[6:7], s[4:5], 8",
            "v_mov_b32_e32 v1, s6",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshr_b64")
        for l in range(32):
            self.assertEqual(results[l], 0x789ABCDE)

def ref_s_ashr_i64(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_ashr_i64 -> (result, scc)."""
    sh = shift & 63
    val = src & MASK64
    sval = val - (1 << 64) if val & (1 << 63) else val
    res = (sval >> sh) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


class TestSAshrI64Oracle(unittest.TestCase):
    def test_positive_numbers(self) -> None:
        self.assertEqual(ref_s_ashr_i64(0x1000000000, 4), (0x0100000000, 1))

    def test_negative_sign_extension(self) -> None:
        # 0x8000000000000000 >> 8 = 0xFF80000000000000
        self.assertEqual(ref_s_ashr_i64(0x8000000000000000, 8), (0xFF80000000000000, 1))
        # -1 (0xFFFFFFFFFFFFFFFF) >> any = -1
        self.assertEqual(ref_s_ashr_i64(MASK64, 20), (MASK64, 1))

class SAshrI64Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_ashr_i64(self.src, self.shift)


class TestSAshrI64AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SAshrI64Expr(0x8000000000000000, 1)
        self.assertEqual(expr.eval(), (0xC000000000000000, 1))

class TestSAshrI64SignAndSCC(unittest.TestCase):
    def test_64bit_sign_propagation(self) -> None:
        val = 0x8000000000000000
        for sh in range(64):
            res, scc = ref_s_ashr_i64(val, sh)
            self.assertEqual(res & (1 << 63), 1 << 63)
            self.assertEqual(scc, 1)

    def test_zero_result(self) -> None:
        res, scc = ref_s_ashr_i64(0, 10)
        self.assertEqual((res, scc), (0, 0))

class TestSAshrI64Translator(unittest.TestCase):
    def test_translator_s_ashr_i64_execution(self) -> None:
        body = [
            "s_mov_b64 s[4:5], 0x8000000000000000",
            "s_ashr_i64 s[6:7], s[4:5], 8",
            "v_mov_b32_e32 v1, s7",  # upper dword
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_ashr_i64")
        for l in range(32):
            self.assertEqual(results[l], 0xFF800000)


if __name__ == "__main__":
    unittest.main()
