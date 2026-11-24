"""Semantic interpreter tests for RDNA2 float compares: v_cmp_{lt,le,gt,ge,eq,lg,neq,nlg,nlt,ngt,nle,nge}_{f32,f16}.

lt/le/gt/ge/eq/lg are ordered (false when either source is NaN); neq/nlg/nlt/ngt/nle/nge are their unordered negations
(true when either source is NaN). +0 and -0 compare equal. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py, reads the compare back with v_cndmask_b32 and
compares it with Python's own comparison of the operands. f16 sources take the low half of the 32-bit register; the
high half is noise the compare must ignore.
"""
import itertools
import math
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

ORDERED = {"lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "gt": lambda x, y: x > y,
           "ge": lambda x, y: x >= y, "eq": lambda x, y: x == y,
           "lg": lambda x, y: x != y and not (math.isnan(x) or math.isnan(y))}
UNORDERED = {"neq": "eq", "nlg": "lg", "nlt": "lt", "ngt": "gt", "nle": "le", "nge": "ge"}   # n<c> = not <c>


def relation(cond: str, x: float, y: float) -> bool:
    if cond in UNORDERED:
        return not ORDERED[UNORDERED[cond]](x, y)
    return ORDERED[cond](x, y)


CONDS = list(ORDERED) + list(UNORDERED)
F32 = {"0": 0x00000000, "-0": 0x80000000, "1": 0x3F800000, "-1": 0xBF800000, "2.5": 0x40200000,
       "inf": 0x7F800000, "-inf": 0xFF800000, "nan": 0x7FC00000}
F16 = {"0": 0x0000, "-0": 0x8000, "1": 0x3C00, "-1": 0xBC00, "2.5": 0x4100,
       "inf": 0x7C00, "-inf": 0xFC00, "nan": 0x7E00}


def f32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f16_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_fcmp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_fcmp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_cmp(op: str, x_bits: int, y_bits: int, form: str = "e32", srcs: str = "v2, v3") -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v2, {x_bits:#x}", f"v_mov_b32 v3, {y_bits:#x}"]
    if form == "e32":
        lines += [f"{op}_e32 vcc_lo, {srcs}", "v_cndmask_b32_e32 v0, 0, 1, vcc_lo"]
    else:
        lines += [f"{op}_e64 s4, {srcs}", "v_cndmask_b32_e64 v0, 0, 1, s4"]
    asm = "; test_float_compares\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    def test_nan_and_signed_zero(self):
        nan = float("nan")
        for cond in ORDERED:
            self.assertFalse(relation(cond, nan, 1.0), cond)
        for cond in UNORDERED:
            self.assertTrue(relation(cond, nan, 1.0), cond)
        self.assertTrue(relation("eq", 0.0, -0.0))
        self.assertFalse(relation("lg", 0.0, -0.0))


class F32Compares(unittest.TestCase):
    def test_every_condition_on_special_values(self):
        for cond, form in itertools.product(CONDS, ("e32", "e64")):
            op = f"v_cmp_{cond}_f32"
            for (xn, xb), (yn, yb) in itertools.product(F32.items(), repeat=2):
                with self.subTest(op=op, form=form, x=xn, y=yn):
                    want = int(relation(cond, f32_value(xb), f32_value(yb)))
                    self.assertEqual(run_cmp(op, xb, yb, form), want)

    def test_source_modifiers(self):
        # e64 neg/abs apply before the compare: -x < |y|
        for (xn, xb), (yn, yb) in itertools.product(F32.items(), repeat=2):
            with self.subTest(x=xn, y=yn):
                want = int(relation("lt", -f32_value(xb), abs(f32_value(yb))))
                self.assertEqual(run_cmp("v_cmp_lt_f32", xb, yb, "e64", "-v2, |v3|"), want)


class F16Compares(unittest.TestCase):
    def test_every_condition_on_special_values(self):
        for cond in CONDS:
            op = f"v_cmp_{cond}_f16"
            for (xn, xb), (yn, yb) in itertools.product(F16.items(), repeat=2):
                with self.subTest(op=op, x=xn, y=yn):
                    want = int(relation(cond, f16_value(xb), f16_value(yb)))
                    self.assertEqual(run_cmp(op, 0xABCD0000 | xb, 0x12340000 | yb), want)


class CmpxNlgTests(unittest.TestCase):
    def run_cmpx(self, op: str, x_bits: int, y_bits: int):
        if _TMP_DIR is None:
            setUpModule()
        lines = [
            f"v_mov_b32 v2, {x_bits:#x}",
            f"v_mov_b32 v3, {y_bits:#x}",
            "v_mov_b32 v0, 0x3f800000",
            f"{op}_e32 v2, v3",
        ]
        asm = "; test_cmpx_nlg\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        return "%out_mrt0" in si.run_fragment(spv)

    def test_v_cmpx_nlg_f32_and_f16(self):
        # nlg = unordered equal: active iff x == y or either operand is NaN.
        for op, tbl in (("v_cmpx_nlg_f32", F32), ("v_cmpx_nlg_f16", F16)):
            for xn, yn, want in (("1", "1", True), ("0", "-0", True), ("nan", "1", True), ("1", "2.5", False)):
                with self.subTest(op=op, x=xn, y=yn):
                    self.assertEqual(self.run_cmpx(op, tbl[xn], tbl[yn]), want)


if __name__ == "__main__":
    unittest.main()
