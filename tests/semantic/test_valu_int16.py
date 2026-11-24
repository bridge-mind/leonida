"""Semantic value tests for RDNA2 16-bit integer VALU instructions.

Covers all 6 ops end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py:
  v_min_u16       D.u16 = min(S0.u16, S1.u16)           (unsigned 16-bit)
  v_max_u16       D.u16 = max(S0.u16, S1.u16)           (unsigned 16-bit)
  v_min_i16       D.i16 = min(S0.i16, S1.i16)           (signed 16-bit two's complement)
  v_max_i16       D.i16 = max(S0.i16, S1.i16)           (signed 16-bit two's complement)
  v_mul_lo_u16    D.u16 = (S0.u16 * S1.u16) & 0xffff    (low 16 bits, mod 2^16)
  v_sub_nc_u16    D.u16 = (S0.u16 - S1.u16) & 0xffff    (no carry, mod 2^16; VOP3-only on GFX10)

16-bit destination and op_sel semantics (AMD RDNA2 ISA, VOP3 OP_SEL; docs/design.md "16-bit results"):
  - VOP3 (_e64, 8-byte encoding, or VOP3-only v_sub_nc_u16): op_sel:[s0,s1,dst] picks the low (0) or high (1)
    16-bit half of src0, src1 and dst; the untouched half of dst is preserved.
  - VOP2 (_e32 / encoding-less non-_e64 mnemonic for the other five ops): reads low halves, writes dst[15:0]
    and zeroes dst[31:16]; source op_sel is rejected as Unsupported.
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

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555

ALL_OPS = (
    "v_min_u16",
    "v_max_u16",
    "v_min_i16",
    "v_max_i16",
    "v_mul_lo_u16",
    "v_sub_nc_u16",
)
VOP2_MODELLED_OPS = tuple(op for op in ALL_OPS if op != "v_sub_nc_u16")

# 8-byte VOP3 encoding comment (LLVM 21 disassembles VOP3 16-bit integer ops without _e64, with 8-byte encoding).
VOP3_ENC8 = "; encoding: [0x00,0x00,0x00,0xd5,0x01,0x05,0x02,0x00]"
# 12-byte GFX10 VOP3 encoding with a 32-bit literal constant (byte 3 bits [31:26] = 0b110101 = 0x35 -> 0xd4..0xd7).
VOP3_ENC12 = "; encoding: [0x00,0x00,0x03,0xd7,0xff,0x02,0x02,0x00,0x34,0x12,0x00,0x00]"
# 8-byte VOP2 encoding with a 32-bit literal constant (byte 3 bit 31 = 0 -> not VOP3).
VOP2_ENC8_LIT = "; encoding: [0xff,0x02,0x00,0x64,0x34,0x12,0x00,0x00]"

BOUNDARY_HALVES = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF)

# Distinct halves across all four slots: unsigned order 0x1234 < 0x7FFF < 0x8001 < 0xABCD,
# signed order 0x8001 (-32767) < 0xABCD (-21555) < 0x1234 (+4660) < 0x7FFF (+32767).
OPSEL_A = 0x12348001
OPSEL_B = 0xABCD7FFF


def u16(x: int) -> int:
    return x & MASK16


def s16(x: int) -> int:
    v = x & MASK16
    return v - 0x10000 if (v & 0x8000) else v


def half16(word: int, hi: int) -> int:
    return ((word >> 16) & MASK16) if hi else (word & MASK16)


def merge16(old: int, res16: int, dst_hi: int = 0, preserve_high: bool = True) -> int:
    res16 &= MASK16
    if dst_hi:
        return ((res16 << 16) | (old & MASK16)) & MASK32
    if preserve_high:
        return ((old & 0xFFFF0000) | res16) & MASK32
    return res16


def ref_op16(op: str, x: int, y: int) -> int:
    """Plain-Python RDNA2 ISA reference for the 16-bit result in [0, 0xFFFF]."""
    ux, uy = u16(x), u16(y)
    if op == "v_min_u16":
        return min(ux, uy)
    if op == "v_max_u16":
        return max(ux, uy)
    if op == "v_min_i16":
        return min(s16(ux), s16(uy)) & MASK16
    if op == "v_max_i16":
        return max(s16(ux), s16(uy)) & MASK16
    if op == "v_mul_lo_u16":
        return (ux * uy) & MASK16
    if op == "v_sub_nc_u16":
        return (ux - uy) & MASK16
    raise ValueError(f"unknown op {op!r}")


def ref_word(op: str, src0: int, src1: int, old: int = OLD, sel=(0, 0, 0), preserve_high: bool = True) -> int:
    """Plain-Python 32-bit destination word reference including op_sel and half preservation."""
    res16 = ref_op16(op, half16(src0, sel[0]), half16(src1, sel[1]))
    return merge16(old, res16, dst_hi=sel[2], preserve_high=preserve_high)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_valu_int16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_valu_int16.dxbc"), "wb") as f:
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
    asm = "; test_valu_int16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_mrt0(lines, regs=("v0", "v0", "v0", "v0")) -> list[int]:
    spv = translate_lines(list(lines) + [f"exp mrt0 {', '.join(regs)} done vm"])
    return list(si.run_fragment(spv)["%out_mrt0"])


def run_one(line: str, v1: int, v2: int, old: int = OLD) -> int:
    return run_mrt0([f"v_mov_b32 v1, {v1 & MASK32:#x}", f"v_mov_b32 v2, {v2 & MASK32:#x}",
                     f"v_mov_b32 v0, {old & MASK32:#x}", line])[0]


def vop3_mnemonic(op: str) -> str:
    return op if op == "v_sub_nc_u16" else f"{op}_e64"


class ReferenceOracleTests(unittest.TestCase):
    """Cross-check plain-Python reference against hand-worked constants."""

    def test_hand_worked_constants(self):
        cases = [
            ("v_min_u16", 0x11117FFF, 0x22228000, (0, 0, 0), True, 0xAAAA7FFF),
            ("v_max_u16", 0x11117FFF, 0x22228000, (0, 0, 0), True, 0xAAAA8000),
            ("v_min_i16", 0x11117FFF, 0x22228000, (0, 0, 0), True, 0xAAAA8000),
            ("v_max_i16", 0x11117FFF, 0x22228000, (0, 0, 0), True, 0xAAAA7FFF),
            ("v_min_i16", 0x11110000, 0x2222FFFF, (0, 0, 0), True, 0xAAAAFFFF),
            ("v_max_u16", 0x11110000, 0x2222FFFF, (0, 0, 0), True, 0xAAAAFFFF),
            ("v_mul_lo_u16", 0x1111FFFF, 0x2222FFFF, (0, 0, 0), True, 0xAAAA0001),
            ("v_mul_lo_u16", 0x11118000, 0x22220002, (0, 0, 0), True, 0xAAAA0000),
            ("v_sub_nc_u16", 0x11110000, 0x22220001, (0, 0, 0), True, 0xAAAAFFFF),
            ("v_sub_nc_u16", 0x11110005, 0x2222000A, (0, 0, 1), True, 0xFFFB5555),
            ("v_min_u16", 0x1111AAAA, 0x22225555, (0, 0, 0), False, 0x00005555),
        ]
        for op, x, y, sel, keep_hi, want in cases:
            with self.subTest(op=op, x=hex(x), y=hex(y), sel=sel, keep_hi=keep_hi):
                self.assertEqual(ref_word(op, x, y, OLD, sel, keep_hi), want)


class ValuInt16BoundaryTests(unittest.TestCase):
    """Exhaustive boundary matrix and high-half noise isolation for all 10 ops."""

    def test_boundary_matrix_all_ops(self):
        # Batch 4 pairs per shader (mrt0 xyzw) to test all 49 boundary pairs x 10 ops cleanly and fast.
        pairs = list(itertools.product(BOUNDARY_HALVES, repeat=2))
        for op in ALL_OPS:
            mnemonic = vop3_mnemonic(op)
            for start in range(0, len(pairs), 4):
                chunk = pairs[start:start + 4]
                lines = []
                expected = []
                for slot, (hx, hy) in enumerate(chunk):
                    dst = f"v{10 + slot}"
                    s0 = 0xCAFE0000 | hx
                    s1 = 0xBEEF0000 | hy
                    lines += [
                        f"v_mov_b32 v1, {s0:#x}",
                        f"v_mov_b32 v2, {s1:#x}",
                        f"v_mov_b32 {dst}, {OLD:#x}",
                        f"{mnemonic} {dst}, v1, v2",
                    ]
                    expected.append(ref_word(op, s0, s1, OLD))
                while len(expected) < 4:
                    lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                    expected.append(0)
                with self.subTest(op=op, batch=start // 4):
                    self.assertEqual(run_mrt0(lines, ("v10", "v11", "v12", "v13")), expected)

    def test_signed_vs_unsigned_minmax_disagreements(self):
        # Pairs where signed i16 and unsigned u16 order invert; high halves carry misleading signs.
        cases = [
            (0x00000000, 0x0000FFFF),  # 0 vs -1 (i16) / 65535 (u16)
            (0x00007FFF, 0x00008000),  # +32767 vs -32768 (i16) / 32768 (u16)
            (0x80007FFF, 0x7FFF8000),  # opposite signs in high halves must be ignored
            (0xFFFF0001, 0x00018000),  # +1 vs -32768 (i16) / 32768 (u16)
            (0x12348000, 0x56780001),  # -32768 vs +1 (i16) / 32768 vs 1 (u16)
        ]
        for x, y in cases:
            with self.subTest(x=hex(x), y=hex(y)):
                min_u, max_u, min_i, max_i = run_mrt0([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    f"v_mov_b32 v10, {OLD:#x}",
                    f"v_mov_b32 v11, {OLD:#x}",
                    f"v_mov_b32 v12, {OLD:#x}",
                    f"v_mov_b32 v13, {OLD:#x}",
                    "v_min_u16_e64 v10, v1, v2",
                    "v_max_u16_e64 v11, v1, v2",
                    "v_min_i16_e64 v12, v1, v2",
                    "v_max_i16_e64 v13, v1, v2",
                ], ("v10", "v11", "v12", "v13"))
                self.assertEqual(min_u, ref_word("v_min_u16", x, y))
                self.assertEqual(max_u, ref_word("v_max_u16", x, y))
                self.assertEqual(min_i, ref_word("v_min_i16", x, y))
                self.assertEqual(max_i, ref_word("v_max_i16", x, y))
                self.assertNotEqual(min_u, min_i)
                self.assertNotEqual(max_u, max_i)

    def test_mul_lo_u16_wraparound_and_high_half_isolation(self):
        cases = [
            (0xABCD0100, 0xDCBA0100, 0x0000),  # 256 * 256 = 65536 -> 0 mod 2^16
            (0x1111FFFF, 0x2222FFFF, 0x0001),  # 65535^2 = 0xFFFE0001 -> 1
            (0x33338001, 0x44440003, 0x8003),  # 32769 * 3 = 0x18003 -> 0x8003
            (0x55551234, 0x66665678, 0x0060),  # 0x1234 * 0x5678 = 0x06260060 -> 0x0060
        ]
        for x, y, want16 in cases:
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(ref_op16("v_mul_lo_u16", x, y), want16)
                self.assertEqual(run_one("v_mul_lo_u16_e64 v0, v1, v2", x, y), (OLD & 0xFFFF0000) | want16)

    def test_sub_nc_u16_wraparound_and_high_half_isolation(self):
        pairs = [(0x12340000, 0x56780001), (0xAAAA000A, 0xBBBB0005), (0xFFFF7FFF, 0x00008000), (0xDEAD8001, 0xBEEF0001)]
        for x, y in pairs:
            with self.subTest(x=hex(x), y=hex(y)):
                sub = run_one("v_sub_nc_u16 v0, v1, v2", x, y)
                self.assertEqual(sub, ref_word("v_sub_nc_u16", x, y))

    def test_min_max_sub_does_not_leak_high_half_bits(self):
        # Upper halves are all-1s / disjoint bits; only low 16 bits participate and dst[31:16] stays OLD.
        x, y = 0xFFFF0F5A, 0xFFFFF03C
        words = run_mrt0([
            f"v_mov_b32 v1, {x:#x}",
            f"v_mov_b32 v2, {y:#x}",
            f"v_mov_b32 v10, {OLD:#x}",
            f"v_mov_b32 v11, {OLD:#x}",
            f"v_mov_b32 v12, {OLD:#x}",
            f"v_mov_b32 v13, {OLD:#x}",
            "v_min_u16_e64 v10, v1, v2",
            "v_max_u16_e64 v11, v1, v2",
            "v_min_i16_e64 v12, v1, v2",
            "v_sub_nc_u16 v13, v1, 0",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [
            ref_word("v_min_u16", x, y),
            ref_word("v_max_u16", x, y),
            ref_word("v_min_i16", x, y),
            ref_word("v_sub_nc_u16", x, 0),
        ])


class ValuInt16OpSelAndEncodingTests(unittest.TestCase):
    """VOP3 op_sel:[s0,s1,dst], 8-byte VOP3 disassembly encodings, VOP3-only v_sub_nc_u16, and VOP2 _e32."""

    def test_all_ops_vop3_op_sel_combinations(self):
        for op in ALL_OPS:
            mnemonic = vop3_mnemonic(op)
            for bits in itertools.product((0, 1), repeat=3):
                line = f"{mnemonic} v0, v1, v2 op_sel:[{bits[0]},{bits[1]},{bits[2]}]"
                with self.subTest(op=op, op_sel=bits):
                    self.assertEqual(run_one(line, OPSEL_A, OPSEL_B), ref_word(op, OPSEL_A, OPSEL_B, OLD, bits))

    def test_bare_mnemonic_with_8byte_vop3_encoding_keeps_other_half(self):
        # LLVM 21 prints VOP3 16-bit ops without _e64, distinguished by an 8-byte (16 hex char) encoding.
        for op in ALL_OPS:
            with self.subTest(op=op):
                got_lo = run_one(f"{op} v0, v1, v2 {VOP3_ENC8}", OPSEL_A, OPSEL_B)
                got_hi = run_one(f"{op} v0, v1, v2 op_sel:[1,0,1] {VOP3_ENC8}", OPSEL_A, OPSEL_B)
                self.assertEqual(got_lo, ref_word(op, OPSEL_A, OPSEL_B, OLD, (0, 0, 0), preserve_high=True))
                self.assertEqual(got_hi, ref_word(op, OPSEL_A, OPSEL_B, OLD, (1, 0, 1), preserve_high=True))

    def test_v_sub_nc_u16_is_always_vop3_without_suffix_or_encoding(self):
        # v_sub_nc_u16 is VOP3-only on GFX10, so even bare without _e64/encoding it keeps the untouched half.
        self.assertEqual(run_one("v_sub_nc_u16 v0, v1, v2", OPSEL_A, OPSEL_B),
                         ref_word("v_sub_nc_u16", OPSEL_A, OPSEL_B, OLD, (0, 0, 0), preserve_high=True))
        self.assertEqual(run_one("v_sub_nc_u16 v0, v1, v2 op_sel:[1,1,1]", OPSEL_A, OPSEL_B),
                         ref_word("v_sub_nc_u16", OPSEL_A, OPSEL_B, OLD, (1, 1, 1), preserve_high=True))

    def test_vop2_e32_zeroes_high_half_and_rejects_source_op_sel(self):
        for op in VOP2_MODELLED_OPS:
            with self.subTest(op=op, mode="e32"):
                self.assertEqual(run_one(f"{op}_e32 v0, v1, v2", OPSEL_A, OPSEL_B),
                                 ref_word(op, OPSEL_A, OPSEL_B, OLD, (0, 0, 0), preserve_high=False))
            for sel in ("op_sel:[1,0,0]", "op_sel:[0,1,0]", "op_sel:[0,0,1]"):
                with self.subTest(op=op, mode=sel):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_lines([
                            "v_mov_b32 v0, 0xaaaa5555",
                            "v_mov_b32 v1, 0x12348001",
                            "v_mov_b32 v2, 0xabcd7fff",
                            f"{op}_e32 v0, v1, v2 {sel}",
                            "exp mrt0 v0, v0, v0, v0 done vm",
                        ])

    def test_12byte_vop3_literal_encoding_preserves_other_half_and_honours_op_sel(self):
        # GFX10 VOP3 instructions with a 32-bit literal constant have a 12-byte (24-hex-char) encoding
        # with byte 3 bits [31:26] == 0x35; they must still be recognized as VOP3 (e64=True).
        lit = 0x56781234
        for op in ALL_OPS:
            with self.subTest(op=op, mode="lo"):
                got = run_one(f"{op} v0, {lit:#x}, v2 {VOP3_ENC12}", OPSEL_A, OPSEL_B)
                self.assertEqual(got, ref_word(op, lit, OPSEL_B, OLD, (0, 0, 0), preserve_high=True))
            with self.subTest(op=op, mode="dst_hi"):
                got = run_one(f"{op} v0, {lit:#x}, v2 op_sel:[0,0,1] {VOP3_ENC12}", OPSEL_A, OPSEL_B)
                self.assertEqual(got, ref_word(op, lit, OPSEL_B, OLD, (0, 0, 1), preserve_high=True))
            with self.subTest(op=op, mode="src1_hi"):
                got = run_one(f"{op} v0, {lit:#x}, v2 op_sel:[0,1,0] {VOP3_ENC12}", OPSEL_A, OPSEL_B)
                self.assertEqual(got, ref_word(op, lit, OPSEL_B, OLD, (0, 1, 0), preserve_high=True))
        # Also verify VOP3-only v_add_nc_u16 with a 12-byte literal encoding.
        self.assertEqual(
            run_one(f"v_add_nc_u16 v0, {lit:#x}, v2 {VOP3_ENC12}", OPSEL_A, OPSEL_B),
            merge16(OLD, (u16(lit) + u16(OPSEL_B)) & MASK16, dst_hi=0, preserve_high=True),
        )
        self.assertEqual(
            run_one(f"v_add_nc_u16 v0, {lit:#x}, v2 op_sel:[0,1,1] {VOP3_ENC12}", OPSEL_A, OPSEL_B),
            merge16(OLD, (u16(lit) + half16(OPSEL_B, 1)) & MASK16, dst_hi=1, preserve_high=True),
        )

    def test_8byte_vop2_literal_encoding_stays_vop2(self):
        # An 8-byte VOP2 encoding with a 32-bit literal has byte 3 bit 31 == 0 (not VOP3 0x35),
        # so it must stay VOP2 (zeroing dst[31:16] and rejecting op_sel).
        lit = 0x00001234
        for op in VOP2_MODELLED_OPS:
            with self.subTest(op=op, mode="e32_lit"):
                got = run_one(f"{op}_e32 v0, {lit:#x}, v2 {VOP2_ENC8_LIT}", OPSEL_A, OPSEL_B)
                self.assertEqual(got, ref_word(op, lit, OPSEL_B, OLD, (0, 0, 0), preserve_high=False))
            with self.subTest(op=op, mode="e32_lit_opsel"):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines([
                        f"v_mov_b32 v0, {OLD:#x}",
                        f"v_mov_b32 v2, {OPSEL_B:#x}",
                        f"{op}_e32 v0, {lit:#x}, v2 op_sel:[0,1,0] {VOP2_ENC8_LIT}",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                    ])


class ValuInt16OperandsAndExecTests(unittest.TestCase):
    """Register aliasing, SGPR and inline-constant operands, and EXEC lane masking."""

    def test_dst_aliases_source_with_op_sel(self):
        # v0 starts as 0x1234ABCD and v1 as 0x567800FF; reading v0 before writing must keep the other half intact.
        init_v0, init_v1 = 0x1234ABCD, 0x567800FF
        for op in ALL_OPS:
            mnemonic = vop3_mnemonic(op)
            for order, sel in (("v0, v0, v1", (1, 0, 0)), ("v0, v1, v0", (0, 0, 1)), ("v0, v0, v0", (0, 1, 1))):
                s0 = init_v0 if order.split(", ")[1] == "v0" else init_v1
                s1 = init_v0 if order.split(", ")[2] == "v0" else init_v1
                with self.subTest(op=op, order=order, sel=sel):
                    got = run_mrt0([
                        f"v_mov_b32 v0, {init_v0:#x}",
                        f"v_mov_b32 v1, {init_v1:#x}",
                        f"{mnemonic} {order} op_sel:[{sel[0]},{sel[1]},{sel[2]}]",
                    ])[0]
                    self.assertEqual(got, ref_word(op, s0, s1, init_v0, sel))

    def test_sgpr_and_inline_constant_operands(self):
        # -16 is 0xFFFFFFF0 (lo=0xFFF0, hi=0xFFFF); 64 is 0x00000040; -1 is 0xFFFFFFFF.
        s4_val = 0xCAFE8005
        v1_val = 0xBEEF0010
        cases = [
            ("v_min_i16", "v_min_i16_e64 v0, s4, -16", s4_val, 0xFFFFFFF0, (0, 0, 0)),
            ("v_min_u16", "v_min_u16_e64 v0, s4, -16", s4_val, 0xFFFFFFF0, (0, 0, 0)),
            ("v_max_i16", "v_max_i16_e64 v0, -1, v1", 0xFFFFFFFF, v1_val, (0, 0, 0)),
            ("v_max_u16", "v_max_u16_e64 v0, v1, -1", v1_val, 0xFFFFFFFF, (0, 0, 0)),
            ("v_mul_lo_u16", "v_mul_lo_u16_e64 v0, s4, 64", s4_val, 64, (0, 0, 0)),
            ("v_sub_nc_u16", "v_sub_nc_u16 v0, 1, s4", 1, s4_val, (0, 0, 0)),
            ("v_sub_nc_u16", "v_sub_nc_u16 v0, s4, 1", s4_val, 1, (0, 0, 0)),
            ("v_min_u16", "v_min_u16_e64 v0, s4, -16 op_sel:[1,0,1]", s4_val, 0xFFFFFFF0, (1, 0, 1)),
            ("v_max_u16", "v_max_u16_e64 v0, 64, s4", 64, s4_val, (0, 0, 0)),
            ("v_mul_lo_u16", "v_mul_lo_u16_e64 v0, v1, -1", v1_val, 0xFFFFFFFF, (0, 0, 0)),
        ]
        for op, insn, x, y, sel in cases:
            with self.subTest(insn=insn):
                got = run_mrt0([
                    f"s_mov_b32 s4, {s4_val:#x}",
                    f"v_mov_b32 v1, {v1_val:#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    insn,
                ])[0]
                self.assertEqual(got, ref_word(op, x, y, OLD, sel))

    def test_inactive_exec_lane_preserves_destination(self):
        for op in ALL_OPS:
            for suffix in (vop3_mnemonic(op), f"{op}_e32") if op != "v_sub_nc_u16" else (op,):
                with self.subTest(insn=suffix):
                    got = run_mrt0([
                        f"v_mov_b32 v0, {OLD:#x}",
                        "v_mov_b32 v1, 0x12345678",
                        "v_mov_b32 v2, 0x9abcdef0",
                        "v_cmpx_eq_u32_e32 vcc_lo, 0, 1",
                        f"{suffix} v0, v1, v2",
                        "s_mov_b32 exec_lo, -1",
                    ])[0]
                    self.assertEqual(got, OLD)


class ValuInt16NegativeControlsTests(unittest.TestCase):
    """Sabotage translated SPIR-V output and confirm the checks detect each mutation."""

    def test_sabotaged_spirv_is_detected(self):
        controls = [
            ("v_min_i16 treated as unsigned extract",
             "v_min_i16_e64 v0, v1, v2", 0x11117FFF, 0x22228000,
             r"OpBitFieldSExtract", "OpBitFieldUExtract", 2,
             ref_word("v_min_i16", 0x11117FFF, 0x22228000)),
            ("v_max_i16 using unsigned max",
             "v_max_i16_e64 v0, v1, v2", 0x11117FFF, 0x22228000,
             r"glsl SMax", "glsl UMax", 1,
             ref_word("v_max_i16", 0x11117FFF, 0x22228000)),
            ("v_min_u16 swapped to max",
             "v_min_u16_e64 v0, v1, v2", 0x11117FFF, 0x22228000,
             r"glsl UMin", "glsl UMax", 1,
             ref_word("v_min_u16", 0x11117FFF, 0x22228000)),
            ("v_sub_nc_u16 drops operand order",
             "v_sub_nc_u16 v0, v1, v2", 0x0000000A, 0x00000005,
             r"OpISub %uint (%\w+) (%\w+)", r"OpISub %uint \2 \1", 1,
             ref_word("v_sub_nc_u16", 0x0000000A, 0x00000005)),
            ("v_mul_lo_u16 drops 16-bit result mask",
             "v_mul_lo_u16_e64 v0, v1, v2", 0x00000101, 0x00000101,
             r"(%\w+) = OpIMul %uint (%\w+) (%\w+)\n(\s*%\w+) = OpBitwiseAnd %uint \1 %cffff",
             r"\1 = OpIMul %uint \2 \3\n\4 = OpCopyObject %uint \1", 1,
             ref_word("v_mul_lo_u16", 0x00000101, 0x00000101)),
            ("v_min_u16 drops high-half preservation",
             "v_min_u16_e64 v0, v1, v2", 0x00000F0F, 0x000000FF,
             r"OpBitwiseAnd %uint (%\w+) %cffff0000", r"OpBitwiseAnd %uint \1 %c0", 1,
             ref_word("v_min_u16", 0x00000F0F, 0x000000FF)),
        ]
        for label, line, v1, v2, pattern, repl, want_count, want in controls:
            with self.subTest(what=label):
                good = translate_lines([
                    f"v_mov_b32 v1, {v1:#x}",
                    f"v_mov_b32 v2, {v2:#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    line,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                ])
                self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], want)
                bad, count = re.subn(pattern, repl, good)
                self.assertEqual(count, want_count, f"unexpected match count for {pattern}")
                self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], want)


if __name__ == "__main__":
    unittest.main()
