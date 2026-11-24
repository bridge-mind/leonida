"""Semantic interpreter tests for RDNA2 v_mad_u32_u16 and v_mad_i32_i16.

RDNA ISA pseudocode: D = S0.half * S1.half + S2, wrapping at 32 bits. The 16-bit halves are zero-extended for
v_mad_u32_u16 and sign-extended for v_mad_i32_i16; op_sel bit 0 / 1 reads the high half of S0 / S1 (the low half by
default). S2 and D are whole dwords. Measured on RDNA2 (gfx1036, EGEN-HW-MAD-U32-U16): op_sel bit 2 adds S2[31:16]
zero-extended (in the i16 op too), op_sel bit 3 writes the result shifted left by 16, and clamp saturates the sum to
[0, 0xffffffff] (u16) or [0x80000000, 0x7fffffff] (i16) before that shift. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it
in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python reference. D leaves through exp as
two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a float round trip
could change.
"""
import itertools
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def half(x: int, hi: int, signed: bool) -> int:
    h = (x >> (16 * hi)) & 0xFFFF
    return h - 0x10000 if signed and h >> 15 else h


def ref(op: str, x: int, y: int, z: int, sel=(0, 0), s2hi=0, dhi=0, clamp=False) -> int:
    signed = op == "v_mad_i32_i16"
    c = z >> 16 if s2hi else z - (1 << 32) if signed and z >> 31 else z
    r = half(x, sel[0], signed) * half(y, sel[1], signed) + c
    if clamp:
        r = max(-(1 << 31), min((1 << 31) - 1, r)) if signed else min(r, M32)
    r &= M32
    return (r << 16) & M32 if dhi else r


OPS = ("v_mad_u32_u16", "v_mad_i32_i16")
VALUES = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0xFFFF0000, 0x80007FFF, 0x12345678, 0xFFFFFFFF]
ACCS = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mad_32_16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad_32_16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_mad_32_16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int, z: int, sel=None) -> int:
    mod = f" op_sel:[{sel[0]},{sel[1]},0,0]" if sel else ""
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"v_mov_b32 v3, {z:#x}",
                      f"{op} v0, v1, v2, v3{mod}"])


class Mad3216Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_mad_u32_u16", 0xFFFF, 0xFFFF, 0), 0xFFFE0001)
        self.assertEqual(ref("v_mad_i32_i16", 0xFFFF, 0xFFFF, 0), 1)                  # -1 * -1
        self.assertEqual(ref("v_mad_i32_i16", 0x8000, 0x7FFF, 0), (-32768 * 32767) & M32)
        self.assertEqual(ref("v_mad_u32_u16", 0x00020000, 3, 1, sel=(1, 0)), 7)        # high half of S0
        self.assertEqual(ref("v_mad_u32_u16", 0xFFFF, 0xFFFF, 0x0001FFFF), 0)          # wraps

    def test_values(self):
        for op in OPS:
            for x in VALUES:
                for y in VALUES[::2]:
                    z = ACCS[(x ^ y) % len(ACCS)]
                    with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                        self.assertEqual(run_op(op, x, y, z), ref(op, x, y, z))

    def test_op_sel_picks_the_source_halves(self):
        x, y, z = 0x8001FFFE, 0x0003FFFD, 0x10
        for op in OPS:
            for sel in itertools.product((0, 1), repeat=2):
                with self.subTest(op=op, op_sel=sel):
                    self.assertEqual(run_op(op, x, y, z, sel), ref(op, x, y, z, sel))

    def test_seeded(self):
        rng = random.Random(0x3216)
        for _ in range(30):
            op, sel = rng.choice(OPS), (rng.getrandbits(1), rng.getrandbits(1))
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z), op_sel=sel):
                self.assertEqual(run_op(op, x, y, z, sel), ref(op, x, y, z, sel))

    def test_reference_measured_modifiers(self):
        # Words measured on gfx1036 (raw kernels, wave32 = wave64).
        self.assertEqual(ref("v_mad_i32_i16", 0xFFFF, 0xFFFF, 0xFFFF8000, s2hi=1), 0x00010000)   # 1 + 0xffff
        self.assertEqual(ref("v_mad_u32_u16", 0x8003FFFE, 0x7FFF0005, 0x12345678, dhi=1), 0x566E0000)
        self.assertEqual(ref("v_mad_u32_u16", 0xFFFE, 0xFFFE, 0xFFFFFFF0, clamp=True), 0xFFFFFFFF)
        self.assertEqual(ref("v_mad_i32_i16", 0x8000, 0x8000, 0x7FFFFFFF, clamp=True, dhi=1), 0xFFFF0000)
        self.assertEqual(ref("v_mad_i32_i16", 0x8000, 0x7FFF, 0x80000000, clamp=True), 0x80000000)

    def test_op_sel_s2_d_and_clamp_seeded(self):
        rng = random.Random(0x3217)
        edges = [0, 0xFFFF, 0x8000, 0x7FFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFF8000]
        for _ in range(40):
            op = rng.choice(OPS)
            sel = (rng.getrandbits(1), rng.getrandbits(1))
            s2hi, dhi, clamp = rng.getrandbits(1), rng.getrandbits(1), rng.random() < 0.6
            x, y, z = (rng.choice(edges) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(3))
            mod = f" op_sel:[{sel[0]},{sel[1]},{s2hi},{dhi}]" + (" clamp" if clamp else "")
            with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z), mod=mod):
                got = run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"v_mov_b32 v3, {z:#x}",
                                 f"{op} v0, v1, v2, v3{mod}"])
                self.assertEqual(got, ref(op, x, y, z, sel, s2hi, dhi, clamp))


if __name__ == "__main__":
    unittest.main()
