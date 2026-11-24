"""Semantic interpreter tests for RDNA2 packed 16-bit integer VOP3P ops.

v_pk_add/sub_u16/i16, v_pk_mul_lo_u16, v_pk_mad_u16/i16 and v_pk_max/min_u16/i16. Result half 0 takes each source's
half from op_sel[i], result half 1 from op_sel_hi[i] (defaults [0,..], [1,..]); each half wraps to 16 bits, and clamp
saturates add/sub/mad to the type's range while acting as a hardware no-op on mul_lo/max/min. A constant operand reads as a 32-bit dword under op_sel/op_sel_hi
(op_sel=0 selects bits [15:0], op_sel=1 selects bits [31:16], measured on gfx1036). Each case translates a synthetic pixel shader with tools/rdna2spirv.py and runs it in
tests/semantic/spirv_mini_interp.py; D leaves through exp as 16-bit halves ORed into 0x40000000. Standard library
only; no GPU, no game files.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


def ref(op, srcs, sel=None, selhi=None, clamp=False):
    ns = len(srcs)
    sel, selhi = sel or [0] * ns, selhi or [1] * ns
    signed = op.endswith("i16")
    kind = op[len("v_pk_"):].rsplit("_", 1)[0]
    out = 0
    for h, sl in enumerate((sel, selhi)):
        v = [(srcs[i] >> (16 * sl[i])) & 0xFFFF for i in range(ns)]
        if signed:
            v = [s16(x) for x in v]
        r = {"add": lambda: v[0] + v[1], "sub": lambda: v[0] - v[1], "mul_lo": lambda: v[0] * v[1],
             "mad": lambda: v[0] * v[1] + v[2], "max": lambda: max(v[0], v[1]), "min": lambda: min(v[0], v[1])}[kind]()
        if clamp and kind in ("add", "sub", "mad"):
            r = max(-32768, min(32767, r)) if signed else max(0, min(0xFFFF, r))
        out |= (r & 0xFFFF) << (16 * h)
    return out


_TMP = None


def translate(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_pk_int16.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    src = os.path.join(_TMP.name, "disasm", "ps_pk_int16.s")
    with open(src, "w", encoding="utf-8") as f:
        f.write("; ps_pk_int16\n" + "".join(f"\t{ln}\n" for ln in lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def run(line, srcs):
    lines = [f"v_mov_b32 v{1 + i}, {v:#x}" for i, v in enumerate(srcs)] + ["v_mov_b32 v0, 0xdeadbeef", line,
             "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_or_b32 v5, 0x40000000, v5",
             "v_or_b32 v6, 0x40000000, v6", "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm"]
    lo, hi = si.run_fragment(translate(lines))["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


TWO = ["v_pk_add_u16", "v_pk_add_i16", "v_pk_sub_u16", "v_pk_sub_i16", "v_pk_mul_lo_u16", "v_pk_max_u16",
       "v_pk_max_i16", "v_pk_min_u16", "v_pk_min_i16"]
THREE = ["v_pk_mad_u16", "v_pk_mad_i16"]
WORDS = [0, 0x00010002, 0xFFFF0001, 0x80007FFF, 0x7FFF8000, 0x12345678, 0xFFFFFFFF, 0x00FF0100]


def opsel(name, bits):
    return f" {name}:[{','.join(map(str, bits))}]"


class PkInt16Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_pk_add_u16", [0xFFFF0001, 0x00010002]), 0x00000003)       # hi wraps
        self.assertEqual(ref("v_pk_add_u16", [0xFFFF0001, 0x00010002], clamp=True), 0xFFFF0003)
        self.assertEqual(ref("v_pk_sub_u16", [0x00010001, 0x00020000], clamp=True), 0x00000001)
        self.assertEqual(ref("v_pk_add_i16", [0x7FFF8000, 0x0001FFFF], clamp=True), 0x7FFF8000)
        self.assertEqual(ref("v_pk_max_i16", [0x8000FFFF, 0x00010000]), 0x00010000)

    def test_two_source_ops(self):
        for op in TWO:
            for x in WORDS:
                for y in WORDS[::2]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(run(f"{op} v0, v1, v2", [x, y]), ref(op, [x, y]))

    def test_mad(self):
        rng = random.Random(0x9A16)
        for op in THREE:
            for _ in range(12):
                srcs = [rng.getrandbits(32) for _ in range(3)]
                with self.subTest(op=op, srcs=[hex(s) for s in srcs]):
                    self.assertEqual(run(f"{op} v0, v1, v2, v3", srcs), ref(op, srcs))

    def test_op_sel(self):
        x, y, z = 0x11112222, 0x33334444, 0x55556666
        for sel, selhi in (([1, 0], [0, 1]), ([1, 1], [0, 0]), ([0, 1], [1, 0])):
            for op in ("v_pk_add_u16", "v_pk_sub_i16", "v_pk_max_u16"):
                with self.subTest(op=op, sel=sel, selhi=selhi):
                    line = f"{op} v0, v1, v2" + opsel("op_sel", sel) + opsel("op_sel_hi", selhi)
                    self.assertEqual(run(line, [x, y]), ref(op, [x, y], sel, selhi))
        line = "v_pk_mad_u16 v0, v1, v2, v3" + opsel("op_sel", [1, 0, 1]) + opsel("op_sel_hi", [0, 1, 0])
        self.assertEqual(run(line, [x, y, z]), ref("v_pk_mad_u16", [x, y, z], [1, 0, 1], [0, 1, 0]))

    def test_clamp(self):
        for op in ("v_pk_add_u16", "v_pk_add_i16", "v_pk_sub_u16", "v_pk_sub_i16",
                   "v_pk_mul_lo_u16", "v_pk_max_u16", "v_pk_min_u16", "v_pk_max_i16", "v_pk_min_i16"):
            for x in WORDS:
                for y in WORDS[1::2]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(run(f"{op} v0, v1, v2 clamp", [x, y]), ref(op, [x, y], clamp=True))
        for op in THREE:
            srcs = [0x7FFFFFFF, 0x7FFF0002, 0x7FFF8000]
            with self.subTest(op=op):
                self.assertEqual(run(f"{op} v0, v1, v2, v3 clamp", srcs), ref(op, srcs, clamp=True))

    def test_constant_operands_op_sel(self):
        # Integer constants are treated as 32-bit dwords: op_sel selects bits [15:0] (0) or [31:16] (1).
        # Positive inline constants zero-extend (5 -> 0x00000005: lo=5, hi=0).
        # Default op_sel:[0,0] op_sel_hi:[1,1]: half 0 adds 5, half 1 adds 0.
        self.assertEqual(run("v_pk_add_u16 v0, v1, 5", [0x00100020]), 0x00100025)
        # Explicit op_sel_hi:[1,0]: both halves add 5.
        self.assertEqual(run("v_pk_add_u16 v0, v1, 5 op_sel_hi:[1,0]", [0x00100020]), 0x00150025)
        # Positive inline constant 64 (0x00000040: lo=64, hi=0)
        self.assertEqual(run("v_pk_add_u16 v0, v1, 64", [0x00100020]), 0x00100060)
        # Negative inline constant -1 sign-extends to 0xFFFFFFFF (lo=0xFFFF, hi=0xFFFF).
        self.assertEqual(run("v_pk_sub_i16 v0, v1, -1", [0x00100020]), 0x00110021)
        self.assertEqual(run("v_pk_add_i16 v0, v1, -1", [0x00100020]), 0x000F001F)
        # Negative inline constant -16 sign-extends to 0xFFFFFFF0 (lo=0xFFF0, hi=0xFFFF).
        # op_sel:[0,0] op_sel_hi:[1,1]: half 0 subtracts 0xFFF0 (-16, adds 16 -> 0x0030), half 1 subtracts 0xFFFF (-1, adds 1 -> 0x0011).
        self.assertEqual(run("v_pk_sub_i16 v0, v1, -16", [0x00100020]), 0x00110030)
        # 32-bit literal 0x12345678: lo=0x5678, hi=0x1234.
        self.assertEqual(run("v_pk_add_u16 v0, v1, 0x12345678", [0x00100020]), 0x12445698)
        # 32-bit literal with op_sel permutation
        self.assertEqual(run("v_pk_add_u16 v0, v1, 0x12345678 op_sel:[1,0] op_sel_hi:[1,0]", [0x00100020]), 0x56885688)
        # 32-bit literal 0x80017fff: lo=0x7fff, hi=0x8001
        self.assertEqual(run("v_pk_add_i16 v0, v1, 0x80017fff", [0x00100020]), 0x8011801f)
        # clamp with constant operand
        self.assertEqual(run("v_pk_add_u16 v0, v1, 5 clamp", [0xFFFF0001]), 0xFFFF0006)
        self.assertEqual(run("v_pk_sub_u16 v0, v1, 5 clamp", [0x00040010]), 0x0004000b)
        self.assertEqual(run("v_pk_sub_u16 v0, v1, 5 op_sel_hi:[1,0] clamp", [0x00040010]), 0x0000000b)
        self.assertEqual(run("v_pk_mad_u16 v0, v1, 5, v2 clamp", [0x00100020, 0x00010002]), 0x000100a2)

    def test_rejected_forms(self):
        for line in ("v_pk_add_u16 v0, v1, v2 neg_lo:[1,0]",
                     "v_pk_mul_lo_u16 v0, v1, v2 neg_hi:[0,1]", "v_pk_max_u16 v0, v1, v2 mul:2"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", line, "exp mrt0 v0, v0, v0, v0 done vm",
                               "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
