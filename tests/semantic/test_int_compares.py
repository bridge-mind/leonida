"""Semantic interpreter tests for RDNA2 integer compares: v_cmp_{eq,ne,lt,le,gt,ge}_{u16,i16,u32,i32,u64,i64}
(e32 into VCC and e64 into an SGPR pair, read back with v_cndmask_b32) and s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32},
s_cmp_{eq,lg}_u64 (SCC, read back with s_cselect_b32).

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the result with Python's own comparison of the operands, read at the instruction's width and signedness
(16-bit compares ignore the high half of each 32-bit source).
"""
import itertools
import os
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

RELATIONS = {
    "eq": lambda x, y: x == y, "ne": lambda x, y: x != y, "lg": lambda x, y: x != y,
    "lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "gt": lambda x, y: x > y, "ge": lambda x, y: x >= y,
}


def as_type(value: int, ty: str) -> int:
    bits = int(ty[1:])
    value &= (1 << bits) - 1
    if ty[0] == "i" and value >> (bits - 1):
        value -= 1 << bits
    return value


# Operand pairs at the sign and width boundaries of each type. 32-bit pairs carry high-half noise so the 16-bit
# compares must ignore it.
PAIRS32 = [(0, 0), (1, 2), (2, 1), (0x7FFFFFFF, 0x80000000), (0x80000000, 0x7FFFFFFF), (0xFFFFFFFF, 0),
           (0, 0xFFFFFFFF), (0x12347FFF, 0xABCD8000), (0xABCD8000, 0x12347FFF), (0xFFFF0001, 0x00010001)]
PAIRS64 = [(0, 0), (1, 1 << 32), (1 << 32, 1), (0x7FFFFFFFFFFFFFFF, 0x8000000000000000),
           (0x8000000000000000, 0x7FFFFFFFFFFFFFFF), ((1 << 64) - 1, 0), (0xFFFFFFFF, 0x100000000),
           (0x1234567800000000, 0x1234567800000000)]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cmp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cmp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_int_compares\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def v_cmp32(op: str, x: int, y: int, form: str) -> int:
    lines = [f"v_mov_b32 v2, {x:#x}", f"v_mov_b32 v3, {y:#x}"]
    if form == "e32":
        lines += [f"{op}_e32 vcc_lo, v2, v3", "v_cndmask_b32_e32 v0, 0, 1, vcc_lo"]
    else:
        lines += [f"{op}_e64 s4, v2, v3", "v_cndmask_b32_e64 v0, 0, 1, s4"]
    return run_lines(lines)


def v_cmp64(op: str, x: int, y: int) -> int:
    return run_lines([f"v_mov_b32 v2, {x & MASK32:#x}", f"v_mov_b32 v3, {x >> 32:#x}",
                      f"v_mov_b32 v4, {y & MASK32:#x}", f"v_mov_b32 v5, {y >> 32:#x}",
                      f"{op}_e64 s4, v[2:3], v[4:5]", "v_cndmask_b32_e64 v0, 0, 1, s4"])


def s_cmp(op: str, x: int, y: int, wide: bool = False) -> int:
    if wide:
        lines = [f"s_mov_b32 s2, {x & MASK32:#x}", f"s_mov_b32 s3, {x >> 32:#x}",
                 f"s_mov_b32 s4, {y & MASK32:#x}", f"s_mov_b32 s5, {y >> 32:#x}", f"{op} s[2:3], s[4:5]"]
    else:
        lines = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}", f"{op} s2, s3"]
    return run_lines(lines + ["s_cselect_b32 s0, 1, 0", "v_mov_b32 v0, s0"])


class VectorCompares(unittest.TestCase):
    def test_16_and_32_bit(self):
        for rel, ty, form in itertools.product(("eq", "ne", "lt", "le", "gt", "ge"), ("u16", "i16", "u32", "i32"),
                                               ("e32", "e64")):
            op = f"v_cmp_{rel}_{ty}"
            for x, y in PAIRS32:
                with self.subTest(op=op, form=form, x=hex(x), y=hex(y)):
                    want = int(RELATIONS[rel](as_type(x, ty), as_type(y, ty)))
                    self.assertEqual(v_cmp32(op, x, y, form), want)

    def test_64_bit(self):
        for rel, ty in itertools.product(("eq", "ne", "lt", "le", "gt", "ge"), ("u64", "i64")):
            op = f"v_cmp_{rel}_{ty}"
            for x, y in PAIRS64:
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    want = int(RELATIONS[rel](as_type(x, ty), as_type(y, ty)))
                    self.assertEqual(v_cmp64(op, x, y), want)


class ScalarCompares(unittest.TestCase):
    def test_32_bit(self):
        for rel, ty in itertools.product(("eq", "lg", "lt", "le", "gt", "ge"), ("u32", "i32")):
            op = f"s_cmp_{rel}_{ty}"
            for x, y in PAIRS32:
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    want = int(RELATIONS[rel](as_type(x, ty), as_type(y, ty)))
                    self.assertEqual(s_cmp(op, x, y), want)

    def test_64_bit_equality(self):
        for rel in ("eq", "lg"):
            op = f"s_cmp_{rel}_u64"
            for x, y in PAIRS64:
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    self.assertEqual(s_cmp(op, x, y, wide=True), int(RELATIONS[rel](x, y)))


if __name__ == "__main__":
    unittest.main()
