"""Semantic interpreter tests for RDNA2 v_cvt_f16_u16, v_cvt_f16_i16, v_cvt_u16_f16 and v_cvt_i16_f16.

The int -> f16 forms convert the low 16-bit half and round to nearest even (65520 and above give +infinity); the
f16 -> int forms truncate toward zero, saturate to the 16-bit range (an f16 infinity too) and give 0 for NaN, as the
32-bit conversions do (docs/design.md, "Float-to-int saturation"). op_sel:[src,dst] picks the high half of the
source / of the destination in the VOP3 form; the other half of the destination is kept. Each case translates a
synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported word with a
plain-Python reference. Not measured on hardware: the kept half and the behaviour of `clamp` and output
multipliers, which the translator rejects.
"""
import itertools
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, old: int = OLD) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v0, {old:#x}", line]
    asm = "; test_cvt16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def merge(old: int, h: int, dst_hi: int) -> int:
    h &= 0xFFFF
    return (h << 16 | old & 0xFFFF) if dst_hi else (old & 0xFFFF0000 | h)


def ref_to_int(op: str, h: int) -> int:
    x = h2f(h)
    lo, hi = (-32768, 32767) if op == "v_cvt_i16_f16" else (0, 65535)
    if math.isnan(x):
        return 0
    if math.isinf(x):
        return (hi if x > 0 else lo) & 0xFFFF
    return max(lo, min(hi, math.trunc(x))) & 0xFFFF


def ref_to_f16(op: str, v: int) -> int:
    v &= 0xFFFF
    if op == "v_cvt_f16_i16" and v & 0x8000:
        v -= 0x10000
    return f2h(float(v))


def half(word: int, hi: int) -> int:
    return (word >> 16 if hi else word) & 0xFFFF


class IntToF16Tests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(run("v_cvt_f16_u16 v0, v1", 0x1234_0003), merge(OLD, 0x4200, 0))        # 3.0
        self.assertEqual(run("v_cvt_f16_i16 v0, v1", 0x0000_FFFD), merge(OLD, 0xC200, 0))        # -3.0
        self.assertEqual(run("v_cvt_f16_u16 v0, v1", 0xFFFF), merge(OLD, 0x7C00, 0))             # 65535 -> +inf
        self.assertEqual(run("v_cvt_f16_u16 v0, v1", 0xFFEF), merge(OLD, 0x7BFF, 0))             # 65519 -> 65504
        self.assertEqual(run("v_cvt_f16_u16 v0, v1", 0xFFF0), merge(OLD, 0x7C00, 0))             # 65520 -> +inf
        self.assertEqual(run("v_cvt_f16_i16 v0, v1", 0x8000), merge(OLD, 0xF800, 0))             # -32768

    def test_op_sel_and_keeps_the_other_half(self):
        v = 0x0005_FFFE
        for op in ("v_cvt_f16_u16", "v_cvt_f16_i16"):
            for bits in itertools.product((0, 1), repeat=2):
                line = f"{op}_e64 v0, v1 op_sel:[{bits[0]},{bits[1]}]"
                with self.subTest(line=line):
                    want = merge(OLD, ref_to_f16(op, half(v, bits[0])), bits[1])
                    self.assertEqual(run(line, v), want)

    def test_random_against_the_reference(self):
        rng = random.Random(0xC16)
        for op in ("v_cvt_f16_u16", "v_cvt_f16_i16"):
            for _ in range(60):
                v = rng.randrange(0x10000) if rng.random() < 0.8 else rng.choice([0, 1, 0x7FFF, 0x8000, 0xFFEF, 0xFFF0])
                with self.subTest(op=op, v=hex(v)):
                    self.assertEqual(run(f"{op} v0, v1", v | 0xBEEF0000), merge(OLD, ref_to_f16(op, v), 0))


class F16ToIntTests(unittest.TestCase):
    def test_known_values(self):
        for op, h, want in (("v_cvt_u16_f16", 0x4248, 3), ("v_cvt_i16_f16", 0xC248, -3 & 0xFFFF),         # 3.14 / -3.14
                            ("v_cvt_u16_f16", 0xBC00, 0), ("v_cvt_u16_f16", 0x7BFF, 0xFFE0),               # -1.0, 65504 (in range)
                            ("v_cvt_i16_f16", 0x7BFF, 0x7FFF), ("v_cvt_i16_f16", 0xFBFF, 0x8000),
                            ("v_cvt_u16_f16", 0x7C00, 0xFFFF), ("v_cvt_u16_f16", 0xFC00, 0),
                            ("v_cvt_i16_f16", 0x7C00, 0x7FFF), ("v_cvt_i16_f16", 0xFC00, 0x8000),
                            ("v_cvt_u16_f16", 0x7E00, 0), ("v_cvt_i16_f16", 0x7E00, 0),                  # NaN
                            ("v_cvt_i16_f16", 0x3800, 0), ("v_cvt_i16_f16", 0xB800, 0)):                  # +-0.5 truncates to 0
            with self.subTest(op=op, h=hex(h)):
                self.assertEqual(run(f"{op} v0, v1", h), merge(OLD, want, 0))

    def test_op_sel_and_keeps_the_other_half(self):
        v = 0xC248_4248                                    # lo 3.14, hi -3.14
        for op in ("v_cvt_u16_f16", "v_cvt_i16_f16"):
            for bits in itertools.product((0, 1), repeat=2):
                line = f"{op}_e64 v0, v1 op_sel:[{bits[0]},{bits[1]}]"
                with self.subTest(line=line):
                    self.assertEqual(run(line, v), merge(OLD, ref_to_int(op, half(v, bits[0])), bits[1]))

    def test_random_against_the_reference(self):
        rng = random.Random(0x16C)
        special = [0x0000, 0x8000, 0x0001, 0x03FF, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x5BFF, 0x7800]
        for op in ("v_cvt_u16_f16", "v_cvt_i16_f16"):
            for _ in range(80):
                h = rng.choice(special) if rng.random() < 0.3 else rng.randrange(0x10000)
                with self.subTest(op=op, h=hex(h)):
                    self.assertEqual(run(f"{op} v0, v1", 0x5555_0000 | h), merge(OLD, ref_to_int(op, h), 0))

    def test_source_neg_abs(self):
        self.assertEqual(run("v_cvt_i16_f16_e64 v0, -v1", 0x4248), merge(OLD, -3 & 0xFFFF, 0))
        self.assertEqual(run("v_cvt_i16_f16_e64 v0, |v1|", 0xC248), merge(OLD, 3, 0))

    def test_output_modifiers_are_rejected(self):
        for line in ("v_cvt_u16_f16_e64 v0, v1 clamp", "v_cvt_i16_f16_e64 v0, v1 mul:2"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                run(line, 0x4248)


if __name__ == "__main__":
    unittest.main()
