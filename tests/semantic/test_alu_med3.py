"""Semantic interpreter tests for RDNA2 three-operand median instructions:
v_med3_f32, v_med3_i32, and v_med3_u32.

Each test translates a synthetic pixel shader with tools/rdna2spirv.py, executes it in
tests/semantic/spirv_mini_interp.py, and validates the exported mrt0 word against a plain-Python reference.

The test matrix covers:
- Signed vs unsigned integer median boundaries (v_med3_i32 vs v_med3_u32):
  Values such as 0xFFFFFFFF (-1 as i32, UINT_MAX as u32), 0x80000000 (INT_MIN as i32,
  2^31 as u32), 0x7FFFFFFF (INT_MAX), 0, and ordinary positive/negative integers where signed
  and unsigned orderings diverge.
- Floating-point 3-way median (v_med3_f32):
  Infinities (-inf, +inf), signed zeros (+0.0, -0.0), positive/negative finite floats,
  denormals, and large magnitudes.
- Commutativity / operand order invariance:
  The hardware/SPIR-V translation evaluates median using min/max/clamp combinations, e.g.
  clamp(z, min(x, y), max(x, y)). Testing all 3! = 6 permutations ensures operand placement
  does not alter the median result.
- Inline constants and input modifiers:
  Operands supplied directly as inline immediate values (e.g. integer or floating-point constants)
  and floating-point input modifiers (negation '-' and absolute value '| |').
"""
from __future__ import annotations

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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
INT_OPS = ("v_med3_i32", "v_med3_u32")

INT_TRIPLES = (
    (5, 0x80000000, 7),
    (0xFFFFFFFF, 0, 1),
    (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
    (0x80000001, 0x80000000, 0x7FFFFFFE),
    (1, 0xFFFFFFFE, 0xFFFFFFFF),
    (9, 9, 0x80000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0),
    (3, 3, 3),
    (0, 0x80000000, 0x7FFFFFFF),
    (0xFFFFFFF0, 0x00000010, 0x00000000),
    (0x7FFFFFFF, 100, 0x80000001),
)

FLOAT_TRIPLES = (
    (-1.0, 0.0, 1.0),
    (-float("inf"), 0.0, float("inf")),
    (-float("inf"), -100.0, float("inf")),
    (0.5, 2.5, 1.5),
    (-10.0, -20.0, -15.0),
    (3.0, 3.0, 1.0),
    (1.0, 3.0, 3.0),
    (2.0, 2.0, 2.0),
    (1e-30, 1e30, 1.0),
    (-1e30, 0.0, 1e30),
    (3.4028234663852886e38, 1.0, -3.4028234663852886e38),
    (1.1754943508222875e-38, 0.5, 2.0),
)


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def f32_from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def bits_from_f32(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def ref_i32(x: int, y: int, z: int) -> int:
    return sorted([x, y, z], key=s32)[1] & MASK32


def ref_u32(x: int, y: int, z: int) -> int:
    return sorted([x, y, z], key=lambda v: v & MASK32)[1] & MASK32


def ref_f32(x_bits: int, y_bits: int, z_bits: int) -> int:
    xf = f32_from_bits(x_bits)
    yf = f32_from_bits(y_bits)
    zf = f32_from_bits(z_bits)
    # The median with -0 ordered below +0, as measured on RDNA2 hardware (AMD Radeon RX 6800, gfx1030):
    # med3(-0, +0, 1.0) = +0 and med3(+0, -0, -1.0) = -0, in every operand order (docs/design.md, float
    # min/max/med3 with NaN). Python's min/max return their first argument for equal zeros, which is not that.
    r = sorted((xf, yf, zf), key=lambda v: (v, math.copysign(1.0, v)))[1]
    return bits_from_f32(r)


def ref(op: str, x: int, y: int, z: int) -> int:
    if op == "v_med3_i32":
        return ref_i32(x, y, z)
    elif op == "v_med3_u32":
        return ref_u32(x, y, z)
    elif op == "v_med3_f32":
        return ref_f32(x, y, z)
    raise ValueError(f"Unknown op {op}")


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_med3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_med3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(
    op: str,
    *xs: int | str,
    inline: tuple[int, ...] = (),
    mods: dict[int, str] | None = None,
) -> int:
    """Translate and run `op v0, src0, src1, src2`.

    Operands are passed as 32-bit raw words (or str if inline). Unless listed in `inline`,
    operands are loaded into v1..v3 via v_mov_b32. If an index is in `inline`, the value is
    written directly into the instruction operand list.
    `mods` maps operand index to a modifier format string or prefix (e.g. `0: "-{}"` or `0: "|{}|"`).
    """
    if _TMP_DIR is None:
        setUpModule()
    lines, operands = [], []
    for i, x in enumerate(xs):
        mod_fmt = (mods or {}).get(i, "{}")
        if i in inline:
            val_str = str(x) if isinstance(x, str) else (f"{x:#x}" if (isinstance(x, int) and x >= 0) else str(x))
            operands.append(mod_fmt.format(val_str))
        else:
            v_reg = f"v{i + 1}"
            raw = x if isinstance(x, int) else int(x, 0)
            lines.append(f"\tv_mov_b32 {v_reg}, {raw & MASK32:#x}")
            operands.append(mod_fmt.format(v_reg))
    ops_str = ", ".join(operands)
    lines.append(f"\t{op} v0, {ops_str}")
    asm = "; test_alu_med3\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    """Validate plain-Python reference against known hand-calculated results."""

    def test_reference_integer(self):
        self.assertEqual(ref("v_med3_i32", 5, 0x80000000, 7), 5)
        self.assertEqual(ref("v_med3_u32", 5, 0x80000000, 7), 7)

        self.assertEqual(ref("v_med3_i32", 0xFFFFFFFF, 0, 1), 0)
        self.assertEqual(ref("v_med3_u32", 0xFFFFFFFF, 0, 1), 1)

        self.assertEqual(ref("v_med3_i32", 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref("v_med3_u32", 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF), 0x80000000)

        self.assertEqual(ref("v_med3_i32", 42, 42, 0x80000000), 42)
        self.assertEqual(ref("v_med3_u32", 42, 42, 0x80000000), 42)
        self.assertEqual(ref("v_med3_i32", 7, 7, 7), 7)

    def test_reference_float(self):
        self.assertEqual(ref("v_med3_f32", bits_from_f32(-1.0), bits_from_f32(0.0), bits_from_f32(1.0)), bits_from_f32(0.0))
        self.assertEqual(ref("v_med3_f32", bits_from_f32(-float("inf")), bits_from_f32(2.5), bits_from_f32(float("inf"))), bits_from_f32(2.5))
        self.assertEqual(ref("v_med3_f32", bits_from_f32(100.0), bits_from_f32(-50.0), bits_from_f32(0.0)), bits_from_f32(0.0))
        self.assertEqual(ref("v_med3_f32", bits_from_f32(3.0), bits_from_f32(3.0), bits_from_f32(1.0)), bits_from_f32(3.0))
        self.assertEqual(ref("v_med3_f32", bits_from_f32(1e-30), bits_from_f32(1e30), bits_from_f32(1.0)), bits_from_f32(1.0))


class Med3IntegerTests(unittest.TestCase):
    """Semantic interpreter tests for v_med3_i32 and v_med3_u32."""

    def test_every_order(self):
        for op in INT_OPS:
            for xs in INT_TRIPLES:
                for perm in sorted(set(itertools.permutations(xs))):
                    with self.subTest(op=op, xs=[hex(v) for v in perm]):
                        self.assertEqual(run(op, *perm), ref(op, *perm))

    def test_sweep(self):
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for op in INT_OPS:
            for xs in itertools.product(values, repeat=3):
                with self.subTest(op=op, xs=[hex(v) for v in xs]):
                    self.assertEqual(run(op, *xs), ref(op, *xs))

    def test_inline_constants(self):
        cases = [
            ("v_med3_i32", (0x7FFFFFFF, -16, 64), 64),
            ("v_med3_u32", (0x7FFFFFFF, -16, 64), 0x7FFFFFFF),
            ("v_med3_i32", (0, 1, -1), 0),
            ("v_med3_u32", (0, 1, -1), 1),
            ("v_med3_i32", (0x80000000, 10, -10), -10 & MASK32),
            ("v_med3_u32", (0x80000000, 10, -10), 0x80000000),
        ]
        for op, xs, want in cases:
            raw_xs = tuple(x & MASK32 for x in xs)
            with self.subTest(op=op, xs=xs):
                self.assertEqual(want, ref(op, *raw_xs))
                self.assertEqual(run(op, *xs, inline=(1, 2)), want)


class Med3FloatTests(unittest.TestCase):
    """Semantic interpreter tests for v_med3_f32."""

    def test_every_order(self):
        for triple in FLOAT_TRIPLES:
            bits = tuple(bits_from_f32(v) for v in triple)
            for perm in sorted(set(itertools.permutations(bits))):
                with self.subTest(perm=[hex(b) for b in perm]):
                    self.assertEqual(run("v_med3_f32", *perm), ref("v_med3_f32", *perm))

    def test_sweep(self):
        values = (-float("inf"), -1.0, 0.0, 1.0, float("inf"))
        bits_values = tuple(bits_from_f32(v) for v in values)
        for perm in itertools.product(bits_values, repeat=3):
            with self.subTest(xs=[hex(b) for b in perm]):
                self.assertEqual(run("v_med3_f32", *perm), ref("v_med3_f32", *perm))

    def test_signed_zero(self):
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        one = bits_from_f32(1.0)
        neg_one = bits_from_f32(-1.0)

        for triple in [
            (pos_zero, neg_zero, one),
            (pos_zero, neg_zero, neg_one),
            (neg_zero, pos_zero, one),
        ]:
            for perm in sorted(set(itertools.permutations(triple))):
                with self.subTest(perm=[hex(b) for b in perm]):
                    got = run("v_med3_f32", *perm)
                    want = ref("v_med3_f32", *perm)
                    self.assertEqual(got, want)

    def test_inline_constants(self):
        cases = [
            ((bits_from_f32(3.0), "1.0", "2.0"), bits_from_f32(2.0)),
            ((bits_from_f32(-5.0), "0.0", "5.0"), bits_from_f32(0.0)),
            ((bits_from_f32(0.5), "-1.0", "1.0"), bits_from_f32(0.5)),
        ]
        for xs, want in cases:
            with self.subTest(xs=xs):
                self.assertEqual(run("v_med3_f32", *xs, inline=(1, 2)), want)

    def test_input_modifiers(self):
        r_neg = run(
            "v_med3_f32",
            bits_from_f32(3.0),
            bits_from_f32(1.0),
            bits_from_f32(2.0),
            mods={0: "-{}"},
        )
        self.assertEqual(r_neg, bits_from_f32(1.0))

        r_abs = run(
            "v_med3_f32",
            bits_from_f32(-4.0),
            bits_from_f32(1.0),
            bits_from_f32(2.0),
            mods={0: "|{}|"},
        )
        self.assertEqual(r_abs, bits_from_f32(2.0))


if __name__ == "__main__":
    unittest.main()
