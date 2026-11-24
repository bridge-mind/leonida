"""Semantic interpreter tests for the register-register scalar compares s_cmp_{eq,lg,gt,ge,lt,le}_{i32,u32}.

RDNA2 ISA (SOPC): SCC = S0 <cond> S1, compared as signed 32-bit values for _i32 and unsigned for _u32. Only eq, lg,
gt, ge, lt and le on _u32 and gt/lt on _i32 had a translation test before; the signed ordering of operands that differ
in bit 31 (0x7FFFFFFF vs 0x80000000, -1 vs 0) is what a wrong signedness gets wrong.
Each case sets SCC, copies it with s_cselect_b32 into one bit of a mask, and the 32-bit mask leaves through exp as two
16-bit halves ORed into 0x40000000 (no NaN or denormal export pattern). The operands are set with s_mov_b32. The
expected bits come from plain-Python integer comparisons.
"""
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


CONDS = {
    "eq": lambda a, b: a == b, "lg": lambda a, b: a != b, "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
}

PAIRS = [(0, 0), (1, 0), (0, 1), (M32, 0), (0, M32), (M32, 1), (1, M32), (0x7FFFFFFF, 0x80000000),
         (0x80000000, 0x7FFFFFFF), (0x80000000, 0x80000000), (0x80000000, M32), (M32, M32), (5, 5),
         (0x7FFFFFFF, 0x7FFFFFFF), (0xFFFFFFFE, M32), (0x00010000, 0x0000FFFF)]

_TMP = None
_SRC = None


def setUpModule():
    global _TMP, _SRC
    _TMP = tempfile.TemporaryDirectory()
    d = os.path.join(_TMP.name, "disasm")
    os.mkdir(d)
    _SRC = os.path.join(d, "ps_salu_cmp.s")
    with open(os.path.join(_TMP.name, "ps_salu_cmp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_mask(lines):
    """Run lines that leave their result bits in s5; return s5."""
    if _TMP is None:
        setUpModule()
    tail = ["s_and_b32 s10, s5, 0xffff", "s_lshr_b32 s11, s5, 16", "s_or_b32 s10, s10, 0x40000000",
            "s_or_b32 s11, s11, 0x40000000", "v_mov_b32 v0, s10", "v_mov_b32 v1, s11"]
    asm = ("; test_salu_cmp_signed\n\ts_mov_b32 s5, 0\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v0, v1, v1, v1 done vm\n\ts_endpgm\n")
    with open(_SRC, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def load(reg, value, dynamic=False):
    return [f"s_mov_b32 s{reg}, {value & M32:#x}"]


def collect(k):
    return [f"s_cselect_b32 s4, 1, 0", f"s_lshl_b32 s4, s4, {k}", "s_or_b32 s5, s5, s4"]


def bits(flags):
    return sum(1 << i for i, f in enumerate(flags) if f)


class ScalarCompareTests(unittest.TestCase):
    def check(self, make_lines, expect, label):
        lines = []
        for k in range(len(expect)):
            lines += make_lines(k, False)
        with self.subTest(case=label):
            self.assertEqual(run_mask(lines), bits(expect), label)

    def test_s_cmp_i32_u32(self):
        for ty in ("i32", "u32"):
            conv = s32 if ty == "i32" else (lambda x: x & M32)
            for cond, fn in CONDS.items():
                def make(k, dynamic, ty=ty, cond=cond):
                    a, b = PAIRS[k]
                    return (load(2, a, dynamic) + load(3, b, dynamic)
                            + [f"s_cmp_{cond}_{ty} s2, s3"] + collect(k))
                expect = [fn(conv(a), conv(b)) for a, b in PAIRS]
                self.check(make, expect, f"s_cmp_{cond}_{ty}")


if __name__ == "__main__":
    unittest.main()
