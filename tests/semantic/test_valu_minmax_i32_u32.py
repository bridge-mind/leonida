"""Semantic interpreter unit tests for RDNA2 32-bit integer VALU min/max instructions.

Covers all four ops end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py:
  v_min_i32   D.i32 = min(S0.i32, S1.i32)   (signed 32-bit two's complement)
  v_max_i32   D.i32 = max(S0.i32, S1.i32)   (signed 32-bit two's complement)
  v_min_u32   D.u32 = min(S0.u32, S1.u32)   (unsigned 32-bit)
  v_max_u32   D.u32 = max(S0.u32, S1.u32)   (unsigned 32-bit)

Tests signed vs unsigned ordering on high-bit values (0x80000000 = INT32_MIN, 0x7FFFFFFF = INT32_MAX,
0xFFFFFFFF = -1 / UINT32_MAX, 0, 1), full boundary pair matrices, equal/negative/positive pairs,
inline constants, 32-bit literal constants, SGPR operands, VOP2 (_e32 / bare), VOP3 (_e64), and SDWA
(_sdwa) encodings, register aliasing, inactive EXEC lane preservation, chained min/max clamping,
and sabotaged-SPIR-V negative controls.
"""
import gc
import itertools
import os
import re
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xCAFEBABE

OPS = ("v_min_i32", "v_max_i32", "v_min_u32", "v_max_u32")

INT32_MIN = 0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF

BOUNDARY_VALUES = (
    0x00000000,  # 0
    0x00000001,  # +1
    0x00000040,  # +64
    0x7FFFFFFE,  # INT32_MAX - 1
    INT32_MAX,   # +2147483647
    INT32_MIN,   # -2147483648 (i32) / 2147483648 (u32)
    0x80000001,  # INT32_MIN + 1
    0xFFFFFFFE,  # -2 (i32) / UINT32_MAX - 1 (u32)
    UINT32_MAX,  # -1 (i32) / 4294967295 (u32)
)


def u32(x: int) -> int:
    return x & MASK32


def s32(x: int) -> int:
    v = x & MASK32
    return v - (1 << 32) if (v & 0x80000000) else v


def ref_op(op: str, x: int, y: int) -> int:
    """Plain-Python RDNA2 ISA reference; operands and result are raw 32-bit words."""
    ux, uy = u32(x), u32(y)
    if op == "v_min_i32":
        return min(s32(ux), s32(uy)) & MASK32
    if op == "v_max_i32":
        return max(s32(ux), s32(uy)) & MASK32
    if op == "v_min_u32":
        return min(ux, uy)
    if op == "v_max_u32":
        return max(ux, uy)
    raise ValueError(f"unknown op {op!r}")


def ref_clamp(signed: bool, x: int, lo: int, hi: int) -> int:
    """Plain-Python clamp(x, lo, hi) = min(max(x, lo), hi) as raw 32-bit word."""
    if signed:
        return min(max(s32(x), s32(lo)), s32(hi)) & MASK32
    return min(max(u32(x), u32(lo)), u32(hi))


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_valu_minmax_i32_u32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_valu_minmax_i32_u32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_valu_minmax_i32_u32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_mrt0(lines, regs=("v0", "v0", "v0", "v0")) -> list[int]:
    spv = translate_lines(list(lines) + [f"exp mrt0 {', '.join(regs)} done vm"])
    return list(si.run_fragment(spv)["%out_mrt0"])


def run_pair(mnemonic: str, x: int, y: int) -> int:
    return run_mrt0([
        f"v_mov_b32 v1, {u32(x):#x}",
        f"v_mov_b32 v2, {u32(y):#x}",
        f"{mnemonic} v0, v1, v2",
    ])[0]


class ReferenceOracleTests(unittest.TestCase):
    """Cross-check the plain-Python reference oracle against hand-worked values."""

    def test_hand_worked_values(self):
        cases = [
            # High-bit signed vs unsigned disagreements
            ("v_min_i32", INT32_MIN, INT32_MAX, INT32_MIN),
            ("v_max_i32", INT32_MIN, INT32_MAX, INT32_MAX),
            ("v_min_u32", INT32_MIN, INT32_MAX, INT32_MAX),
            ("v_max_u32", INT32_MIN, INT32_MAX, INT32_MIN),
            ("v_min_i32", UINT32_MAX, 0, UINT32_MAX),   # -1 < 0
            ("v_max_i32", UINT32_MAX, 0, 0),            # max(-1, 0) = 0
            ("v_min_u32", UINT32_MAX, 0, 0),            # min(2^32-1, 0) = 0
            ("v_max_u32", UINT32_MAX, 0, UINT32_MAX),   # max(2^32-1, 0) = 2^32-1
            ("v_min_i32", UINT32_MAX, 1, UINT32_MAX),   # -1 < 1
            ("v_max_i32", UINT32_MAX, 1, 1),
            ("v_min_u32", UINT32_MAX, 1, 1),
            ("v_max_u32", UINT32_MAX, 1, UINT32_MAX),
            # Both negative (high bit set on both): signed and unsigned orders coincide
            ("v_min_i32", INT32_MIN, UINT32_MAX, INT32_MIN),
            ("v_max_i32", INT32_MIN, UINT32_MAX, UINT32_MAX),
            ("v_min_u32", INT32_MIN, UINT32_MAX, INT32_MIN),
            ("v_max_u32", INT32_MIN, UINT32_MAX, UINT32_MAX),
            # Equal operands
            ("v_min_i32", INT32_MIN, INT32_MIN, INT32_MIN),
            ("v_max_u32", UINT32_MAX, UINT32_MAX, UINT32_MAX),
        ]
        for op, x, y, want in cases:
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                self.assertEqual(ref_op(op, x, y), want)


class ValuMinMaxCoreTests(unittest.TestCase):
    """Core semantic value tests across signed/unsigned disagreements, boundaries, and categories."""

    def test_signed_vs_unsigned_high_bit_disagreements(self):
        # Every pair has one high-bit value (negative as i32, >= 2^31 as u32) and one non-negative value,
        # so signed and unsigned min/max must strictly disagree on both min and max.
        pairs = [
            (INT32_MIN, INT32_MAX),
            (INT32_MAX, INT32_MIN),
            (UINT32_MAX, 0),
            (0, UINT32_MAX),
            (UINT32_MAX, 1),
            (1, UINT32_MAX),
            (INT32_MIN, 0),
            (0, INT32_MIN),
            (INT32_MIN, 1),
            (0xFFFFFFF0, 0x00000040),
            (0x80000001, 0x7FFFFFFE),
        ]
        for x, y in pairs:
            with self.subTest(x=hex(x), y=hex(y)):
                min_i, max_i, min_u, max_u = run_mrt0([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    "v_min_i32 v10, v1, v2",
                    "v_max_i32 v11, v1, v2",
                    "v_min_u32 v12, v1, v2",
                    "v_max_u32 v13, v1, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(min_i, ref_op("v_min_i32", x, y))
                self.assertEqual(max_i, ref_op("v_max_i32", x, y))
                self.assertEqual(min_u, ref_op("v_min_u32", x, y))
                self.assertEqual(max_u, ref_op("v_max_u32", x, y))
                self.assertNotEqual(min_i, min_u)
                self.assertNotEqual(max_i, max_u)

    def test_boundary_matrix_all_ops(self):
        # All 81 ordered pairs of BOUNDARY_VALUES x 4 ops, batched 4 per shader in mrt0 xyzw.
        pairs = list(itertools.product(BOUNDARY_VALUES, repeat=2))
        for op in OPS:
            for start in range(0, len(pairs), 4):
                chunk = pairs[start:start + 4]
                lines = []
                expected = []
                for slot, (x, y) in enumerate(chunk):
                    dst = f"v{10 + slot}"
                    lines += [
                        f"v_mov_b32 v1, {x:#x}",
                        f"v_mov_b32 v2, {y:#x}",
                        f"{op} {dst}, v1, v2",
                    ]
                    expected.append(ref_op(op, x, y))
                while len(expected) < 4:
                    lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                    expected.append(0)
                with self.subTest(op=op, batch=start // 4):
                    self.assertEqual(run_mrt0(lines, ("v10", "v11", "v12", "v13")), expected)

    def test_equal_operands(self):
        values = (0, 1, 42, INT32_MAX, INT32_MIN, 0xDEADBEEF, UINT32_MAX)
        for v in values:
            with self.subTest(v=hex(v)):
                words = run_mrt0([
                    f"v_mov_b32 v1, {v:#x}",
                    f"v_mov_b32 v2, {v:#x}",
                    "v_min_i32 v10, v1, v2",
                    "v_max_i32 v11, v1, v2",
                    "v_min_u32 v12, v1, v2",
                    "v_max_u32 v13, v1, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(words, [v, v, v, v])

    def test_negative_vs_negative_and_positive_vs_positive(self):
        # When both operands share sign bit (both negative or both non-negative), signed and unsigned agree.
        same_sign_pairs = [
            # Negative vs negative
            (INT32_MIN, UINT32_MAX),       # -2147483648 vs -1
            (0xFFFFFFF0, 0xFFFFFFFC),      # -16 vs -4
            (0x80000001, 0x80000000),      # -2147483647 vs -2147483648
            (0xC0000000, 0xBFFFFFFF),      # -1073741824 vs -1073741825
            # Positive vs positive
            (0, 1),
            (1, 64),
            (0x12345678, 0x76543210),
            (0x7FFFFFFE, INT32_MAX),
        ]
        for x, y in same_sign_pairs:
            for a, b in ((x, y), (y, x)):
                with self.subTest(a=hex(a), b=hex(b)):
                    min_i, max_i, min_u, max_u = run_mrt0([
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        "v_min_i32 v10, v1, v2",
                        "v_max_i32 v11, v1, v2",
                        "v_min_u32 v12, v1, v2",
                        "v_max_u32 v13, v1, v2",
                    ], ("v10", "v11", "v12", "v13"))
                    self.assertEqual(min_i, ref_op("v_min_i32", a, b))
                    self.assertEqual(max_i, ref_op("v_max_i32", a, b))
                    self.assertEqual(min_u, ref_op("v_min_u32", a, b))
                    self.assertEqual(max_u, ref_op("v_max_u32", a, b))
                    self.assertEqual(min_i, min_u)
                    self.assertEqual(max_i, max_u)


class ValuMinMaxOperandsAndEncodingsTests(unittest.TestCase):
    """Inline constants, 32-bit literals, SGPR sources, VOP2/VOP3/SDWA forms, aliasing, and EXEC masking."""

    def test_inline_constants_src0_and_e64_src1(self):
        # Inline constants (0, -1, 1, 64, -16) as src0 in VOP2, and as src1 in VOP3 (_e64).
        v_val = 0x00000005
        v_neg = 0xFFFFFFFC  # -4
        cases = [
            # (insn, x, y, op)
            ("v_min_i32_e32 v0, 0, v1", 0, v_neg, "v_min_i32"),
            ("v_max_i32_e32 v0, 0, v1", 0, v_neg, "v_max_i32"),
            ("v_min_u32_e32 v0, 0, v1", 0, v_neg, "v_min_u32"),
            ("v_max_u32_e32 v0, 0, v1", 0, v_neg, "v_max_u32"),
            ("v_min_i32_e32 v0, -1, v2", UINT32_MAX, v_val, "v_min_i32"),
            ("v_max_i32_e32 v0, -1, v2", UINT32_MAX, v_val, "v_max_i32"),
            ("v_min_u32_e32 v0, -1, v2", UINT32_MAX, v_val, "v_min_u32"),
            ("v_max_u32_e32 v0, -1, v2", UINT32_MAX, v_val, "v_max_u32"),
            ("v_min_i32_e32 v0, 1, v1", 1, v_neg, "v_min_i32"),
            ("v_max_i32_e32 v0, 1, v2", 1, v_val, "v_max_i32"),
            ("v_min_i32_e32 v0, 64, v2", 64, v_val, "v_min_i32"),
            ("v_max_u32_e32 v0, 64, v2", 64, v_val, "v_max_u32"),
            # VOP3 (_e64) allows inline constants in src1 (or both src0 and src1).
            ("v_min_i32_e64 v0, v2, -1", v_val, UINT32_MAX, "v_min_i32"),
            ("v_max_i32_e64 v0, v1, 0", v_neg, 0, "v_max_i32"),
            ("v_min_u32_e64 v0, v1, 64", v_neg, 64, "v_min_u32"),
            ("v_max_u32_e64 v0, v2, -1", v_val, UINT32_MAX, "v_max_u32"),
            ("v_min_i32_e64 v0, -16, 64", 0xFFFFFFF0, 64, "v_min_i32"),
            ("v_min_u32_e64 v0, -16, 64", 0xFFFFFFF0, 64, "v_min_u32"),
            ("v_max_i32_e64 v0, -16, 64", 0xFFFFFFF0, 64, "v_max_i32"),
            ("v_max_u32_e64 v0, -16, 64", 0xFFFFFFF0, 64, "v_max_u32"),
        ]
        for insn, x, y, op in cases:
            with self.subTest(insn=insn):
                got = run_mrt0([
                    f"v_mov_b32 v1, {v_neg:#x}",
                    f"v_mov_b32 v2, {v_val:#x}",
                    insn,
                ])[0]
                self.assertEqual(got, ref_op(op, x, y))

    def test_literal_32bit_constants_src0(self):
        # 32-bit literal constants (outside inline constant range) as src0 in VOP2/bare/e32.
        v_val = 0x00001000
        literals = (INT32_MIN, INT32_MAX, 0xFFFFFF00, 0x12345678, 0xDEADBEEF)
        for lit in literals:
            with self.subTest(lit=hex(lit)):
                words = run_mrt0([
                    f"v_mov_b32 v2, {v_val:#x}",
                    f"v_min_i32_e32 v10, {lit:#x}, v2",
                    f"v_max_i32_e32 v11, {lit:#x}, v2",
                    f"v_min_u32 v12, {lit:#x}, v2",
                    f"v_max_u32 v13, {lit:#x}, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(words, [
                    ref_op("v_min_i32", lit, v_val),
                    ref_op("v_max_i32", lit, v_val),
                    ref_op("v_min_u32", lit, v_val),
                    ref_op("v_max_u32", lit, v_val),
                ])

    def test_sgpr_operands_vop2_and_e64(self):
        # SGPR src0 in VOP2 (_e32 / bare) and SGPR src0/src1 in VOP3 (_e64).
        s_val = INT32_MIN
        v_val = INT32_MAX
        s_other = UINT32_MAX
        for suffix in ("", "_e32", "_e64"):
            with self.subTest(form=f"sgpr_src0{suffix}"):
                words = run_mrt0([
                    f"s_mov_b32 s4, {s_val:#x}",
                    f"v_mov_b32 v2, {v_val:#x}",
                    f"v_min_i32{suffix} v10, s4, v2",
                    f"v_max_i32{suffix} v11, s4, v2",
                    f"v_min_u32{suffix} v12, s4, v2",
                    f"v_max_u32{suffix} v13, s4, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(words, [
                    ref_op("v_min_i32", s_val, v_val),
                    ref_op("v_max_i32", s_val, v_val),
                    ref_op("v_min_u32", s_val, v_val),
                    ref_op("v_max_u32", s_val, v_val),
                ])

        # VOP3 (_e64) also permits SGPR in src1 or in both src0 and src1 (same SGPR or distinct SGPRs).
        words_e64 = run_mrt0([
            f"s_mov_b32 s4, {s_val:#x}",
            f"s_mov_b32 s5, {s_other:#x}",
            f"v_mov_b32 v2, {v_val:#x}",
            "v_min_i32_e64 v10, v2, s4",
            "v_max_u32_e64 v11, v2, s4",
            "v_min_i32_e64 v12, s4, s5",
            "v_max_u32_e64 v13, s4, s5",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words_e64, [
            ref_op("v_min_i32", v_val, s_val),
            ref_op("v_max_u32", v_val, s_val),
            ref_op("v_min_i32", s_val, s_other),
            ref_op("v_max_u32", s_val, s_other),
        ])

    def test_vop2_e32_and_vop3_e64_equivalence(self):
        pairs = [
            (INT32_MIN, INT32_MAX),
            (UINT32_MAX, 0),
            (0xFFFFFF00, 0x00000100),
            (0x80000001, 0x80000000),
        ]
        for op in OPS:
            for x, y in pairs:
                want = ref_op(op, x, y)
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    bare = run_pair(op, x, y)
                    e32 = run_pair(f"{op}_e32", x, y)
                    e64 = run_pair(f"{op}_e64", x, y)
                    self.assertEqual(bare, want)
                    self.assertEqual(e32, want)
                    self.assertEqual(e64, want)

    def test_sdwa_source_and_destination_modifiers(self):
        # v1 = 0x8001007F -> BYTE_0 = 0x7F (+127), BYTE_2 = 0x01, BYTE_3 = 0x80 (-128 sext / 128 uext),
        #                    WORD_0 = 0x007F, WORD_1 = 0x8001 (-32767 sext / 32769 uext).
        # v2 = 0x0002FFFE -> BYTE_0 = 0xFE (-2 sext / 254 uext), WORD_0 = 0xFFFE (-2 sext / 65534 uext).
        v1, v2 = 0x8001007F, 0x0002FFFE
        words = run_mrt0([
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v2, {v2:#x}",
            f"v_mov_b32 v12, {OLD:#x}",
            # sext(BYTE_3(v1)) = -128 (0xFFFFFF80) vs sext(BYTE_0(v2)) = -2 (0xFFFFFFFE) -> min_i32 = -128
            "v_min_i32_sdwa v10, sext(v1), sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_0",
            # uext(BYTE_3(v1)) = 128 vs uext(BYTE_0(v2)) = 254 -> max_u32 = 254
            "v_max_u32_sdwa v11, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_0",
            # sext(WORD_1(v1)) = -32767 (0xFFFF8001) vs uext(WORD_0(v1)) = 0x7F -> min_i32 = 0xFFFF8001;
            # placed into WORD_1 with UNUSED_PRESERVE keeps OLD[15:0] = 0xBABE -> 0x8001BABE.
            "v_min_i32_sdwa v12, sext(v1), v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0",
            # sext(BYTE_3(v1)) = -128 vs sext(BYTE_0(v2)) = -2 -> max_i32 = -2 (0xFFFFFFFE);
            # dst_sel:BYTE_0 with UNUSED_SEXT sign-extends low byte 0xFE back to 0xFFFFFFFE.
            "v_max_i32_sdwa v13, sext(v1), sext(v2) dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_0",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [
            0xFFFFFF80,  # min_i32(-128, -2) = -128
            0x000000FE,  # max_u32(128, 254) = 254
            0x8001BABE,  # low 16 bits of 0xFFFF8001 inserted at WORD_1, preserving 0xBABE
            0xFFFFFFFE,  # max_i32(-128, -2) = -2 -> BYTE_0 = 0xFE, UNUSED_SEXT -> 0xFFFFFFFE
        ])

    def test_dst_aliases_sources(self):
        a, b = INT32_MIN, 0x00000007
        for op in OPS:
            for order, x, y in (
                ("v0, v0, v1", a, b),
                ("v0, v1, v0", b, a),
                ("v0, v0, v0", a, a),
            ):
                with self.subTest(op=op, order=order):
                    got = run_mrt0([
                        f"v_mov_b32 v0, {a:#x}",
                        f"v_mov_b32 v1, {b:#x}",
                        f"{op} {order}",
                    ])[0]
                    self.assertEqual(got, ref_op(op, x, y))

    def test_inactive_exec_lane_preserves_destination(self):
        for op in OPS:
            for suffix in ("", "_e32", "_e64"):
                insn = f"{op}{suffix}"
                with self.subTest(insn=insn):
                    got = run_mrt0([
                        f"v_mov_b32 v0, {OLD:#x}",
                        f"v_mov_b32 v1, {INT32_MIN:#x}",
                        f"v_mov_b32 v2, {INT32_MAX:#x}",
                        "v_cmpx_eq_u32_e32 vcc_lo, 0, 1",
                        f"{insn} v0, v1, v2",
                        "s_mov_b32 exec_lo, -1",
                    ])[0]
                    self.assertEqual(got, OLD)


class ValuMinMaxClampingTests(unittest.TestCase):
    """Chaining v_min/v_max as signed and unsigned clamp(x, lo, hi)."""

    def test_signed_clamp_chain(self):
        # Signed range [-10, +10] across below-min, at-min, interior, at-max, above-max, and extremes.
        lo = (-10) & MASK32
        hi = 10
        inputs = (INT32_MIN, (-11) & MASK32, lo, (-1) & MASK32, 0, 5, hi, 11, INT32_MAX)
        for x in inputs:
            want = ref_clamp(True, x, lo, hi)
            with self.subTest(x=hex(x)):
                # Test both min(max(x, lo), hi) and max(min(x, hi), lo), with VOP2 and _e64.
                got_min_max, got_max_min, got_e64, _ = run_mrt0([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    # Chain 1: min(max(x, lo), hi)
                    "v_max_i32 v4, v1, v2",
                    "v_min_i32 v10, v4, v3",
                    # Chain 2: max(min(x, hi), lo)
                    "v_min_i32 v5, v1, v3",
                    "v_max_i32 v11, v5, v2",
                    # Chain 3: inline-constant bounds in VOP3 _e64
                    "v_max_i32_e64 v6, v1, -10",
                    "v_min_i32_e64 v12, v6, 10",
                    "v_mov_b32 v13, 0",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(got_min_max, want)
                self.assertEqual(got_max_min, want)
                self.assertEqual(got_e64, want)

    def test_unsigned_clamp_chain_straddling_sign_bit(self):
        # Unsigned range [0x7FFFFFF0, 0x80000010] straddles the i32 sign boundary:
        # signed clamping would invert lo > hi, so this proves unsigned ordering end-to-end.
        lo = 0x7FFFFFF0
        hi = 0x80000010
        inputs = (0, 1, 0x7FFFFFEF, lo, INT32_MAX, INT32_MIN, hi, 0x80000011, UINT32_MAX)
        for x in inputs:
            want = ref_clamp(False, x, lo, hi)
            with self.subTest(x=hex(x)):
                got_min_max, got_max_min, _, _ = run_mrt0([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    "v_max_u32 v4, v1, v2",
                    "v_min_u32 v10, v4, v3",
                    "v_min_u32_e64 v5, v1, v3",
                    "v_max_u32_e64 v11, v5, v2",
                    "v_mov_b32 v12, 0",
                    "v_mov_b32 v13, 0",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(got_min_max, want)
                self.assertEqual(got_max_min, want)


class ValuMinMaxNegativeControlsTests(unittest.TestCase):
    """Sabotage translated SPIR-V and confirm the tests detect signed/unsigned and min/max swaps."""

    def test_sabotaged_spirv_is_detected(self):
        controls = [
            ("v_min_i32 swapped to UMin", "v_min_i32 v0, v1, v2", INT32_MIN, INT32_MAX,
             r"glsl SMin", "glsl UMin", ref_op("v_min_i32", INT32_MIN, INT32_MAX)),
            ("v_max_i32 swapped to UMax", "v_max_i32 v0, v1, v2", INT32_MIN, INT32_MAX,
             r"glsl SMax", "glsl UMax", ref_op("v_max_i32", INT32_MIN, INT32_MAX)),
            ("v_min_u32 swapped to SMin", "v_min_u32 v0, v1, v2", UINT32_MAX, 0,
             r"glsl UMin", "glsl SMin", ref_op("v_min_u32", UINT32_MAX, 0)),
            ("v_max_u32 swapped to SMax", "v_max_u32 v0, v1, v2", UINT32_MAX, 0,
             r"glsl UMax", "glsl SMax", ref_op("v_max_u32", UINT32_MAX, 0)),
            ("v_min_i32 swapped to SMax", "v_min_i32 v0, v1, v2", 0xFFFFFFFE, 5,
             r"glsl SMin", "glsl SMax", ref_op("v_min_i32", 0xFFFFFFFE, 5)),
            ("v_max_u32 swapped to UMin", "v_max_u32 v0, v1, v2", 1, 64,
             r"glsl UMax", "glsl UMin", ref_op("v_max_u32", 1, 64)),
        ]
        for label, insn, v1, v2, pattern, repl, want in controls:
            with self.subTest(what=label):
                good = translate_lines([
                    f"v_mov_b32 v1, {v1:#x}",
                    f"v_mov_b32 v2, {v2:#x}",
                    insn,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                ])
                self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], want)
                bad, count = re.subn(pattern, repl, good)
                self.assertEqual(count, 1, f"unexpected match count for {pattern}")
                self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], want)


if __name__ == "__main__":
    unittest.main()
