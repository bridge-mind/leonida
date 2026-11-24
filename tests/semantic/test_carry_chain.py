"""Semantic interpreter tests for RDNA2 carry and borrow: v_add_co_u32, v_sub_co_u32, v_subrev_co_u32 and their
_co_ci forms (e32 through VCC, e64 through SGPRs), s_add_u32/s_addc_u32 and s_sub_u32/s_subb_u32 (through SCC), and
the 64-bit add and subtract a compiler builds from them.

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the result and the carry-out (read back with v_cndmask_b32 or s_cselect_b32) with plain Python. The
carry-out of a subtract is its borrow: 1 when the subtrahend (plus the carry-in) is larger than the minuend.
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
MASK64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def add(x, y, ci):
    return (x + y + ci) & MASK32, (x + y + ci) >> 32


def sub(x, y, ci):
    return (x - y - ci) & MASK32, int(y + ci > x)


# op: (reference(src0, src1, carry_in), takes a carry-in)
VECTOR = {
    "v_add_co_u32": (add, False), "v_sub_co_u32": (sub, False),
    "v_subrev_co_u32": (lambda x, y, ci: sub(y, x, ci), False),
    "v_add_co_ci_u32": (add, True), "v_sub_co_ci_u32": (sub, True),
    "v_subrev_co_ci_u32": (lambda x, y, ci: sub(y, x, ci), True),
}
SCALAR = {"s_add_u32": (add, False), "s_sub_u32": (sub, False), "s_addc_u32": (add, True), "s_subb_u32": (sub, True)}

# Pairs at the carry and borrow boundaries (x + y around 2^32, x - y around 0).
PAIRS = [(0, 0), (1, 0), (0, 1), (0xFFFFFFFF, 1), (1, 0xFFFFFFFF), (0xFFFFFFFF, 0), (0xFFFFFFFF, 0xFFFFFFFF),
         (0x80000000, 0x80000000), (0x7FFFFFFF, 0x80000000), (0x12345678, 0x12345678), (0x12345679, 0x12345678)]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_carry.s")
    with open(os.path.join(_TMP_DIR.name, "ps_carry.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, export v0 and v1, return those two words."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_carry_chain\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def vector_op(op, x, y, ci, form):
    with_ci = VECTOR[op][1]
    lines = [f"v_mov_b32 v2, {x:#x}", f"v_mov_b32 v3, {y:#x}", f"v_mov_b32 v4, {ci}"]
    if form == "e32":
        lines += ["v_cmp_ne_u32_e32 vcc_lo, 0, v4", f"{op}_e32 v0, vcc_lo, v2, v3" + (", vcc_lo" if with_ci else ""),
                  "v_cndmask_b32_e32 v1, 0, 1, vcc_lo"]
    else:
        lines += ["v_cmp_ne_u32_e64 s8, 0, v4", f"{op}_e64 v0, s10, v2, v3" + (", s8" if with_ci else ""),
                  "v_cndmask_b32_e64 v1, 0, 1, s10"]
    return run_lines(lines)


def scalar_op(op, x, y, ci):
    return run_lines([f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}", f"s_cmp_eq_u32 {ci}, 1",   # SCC = carry-in
                      f"{op} s0, s2, s3", "s_cselect_b32 s1, 1, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])


class VectorCarry(unittest.TestCase):
    def test_result_and_carry_out(self):
        for (op, (ref, with_ci)), form in itertools.product(VECTOR.items(), ("e32", "e64")):
            for (x, y), ci in itertools.product(PAIRS, (0, 1) if with_ci else (0,)):
                with self.subTest(op=op, form=form, x=hex(x), y=hex(y), ci=ci):
                    self.assertEqual(vector_op(op, x, y, ci, form), ref(x, y, ci))

    def test_64_bit_add_and_subtract(self):
        for x, y in ((0xFFFFFFFF, 1), (0x00000001FFFFFFFF, 0x00000000FFFFFFFF), (MASK64, 1), (0, 1),
                     (0x123456789ABCDEF0, 0x0FEDCBA987654321)):
            for op_lo, op_hi, want in (("v_add_co_u32", "v_add_co_ci_u32", (x + y) & MASK64),
                                       ("v_sub_co_u32", "v_sub_co_ci_u32", (x - y) & MASK64)):
                with self.subTest(op=op_lo, x=hex(x), y=hex(y)):
                    got = run_lines([f"v_mov_b32 v2, {x & MASK32:#x}", f"v_mov_b32 v3, {x >> 32:#x}",
                                     f"v_mov_b32 v4, {y & MASK32:#x}", f"v_mov_b32 v5, {y >> 32:#x}",
                                     f"{op_lo}_e64 v0, s10, v2, v4", f"{op_hi}_e64 v1, s12, v3, v5, s10"])
                    self.assertEqual(got[0] | got[1] << 32, want)


class ScalarCarry(unittest.TestCase):
    def test_result_and_scc(self):
        for op, (ref, with_ci) in SCALAR.items():
            for (x, y), ci in itertools.product(PAIRS, (0, 1) if with_ci else (0,)):
                with self.subTest(op=op, x=hex(x), y=hex(y), ci=ci):
                    self.assertEqual(scalar_op(op, x, y, ci), ref(x, y, ci))

    def test_64_bit_add_and_subtract(self):
        for x, y in ((0xFFFFFFFF, 1), (MASK64, 1), (0, 1), (0x123456789ABCDEF0, 0x0FEDCBA987654321)):
            for op_lo, op_hi, want in (("s_add_u32", "s_addc_u32", (x + y) & MASK64),
                                       ("s_sub_u32", "s_subb_u32", (x - y) & MASK64)):
                with self.subTest(op=op_lo, x=hex(x), y=hex(y)):
                    got = run_lines([f"s_mov_b32 s2, {x & MASK32:#x}", f"s_mov_b32 s3, {x >> 32:#x}",
                                     f"s_mov_b32 s4, {y & MASK32:#x}", f"s_mov_b32 s5, {y >> 32:#x}",
                                     f"{op_lo} s0, s2, s4", f"{op_hi} s1, s3, s5", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
                    self.assertEqual(got[0] | got[1] << 32, want)


if __name__ == "__main__":
    unittest.main()
