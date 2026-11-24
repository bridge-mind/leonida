"""Semantic interpreter tests for RDNA2 v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32.

RDNA ISA pseudocode: D[15:0] = sat16(S0), D[31:16] = sat16(S1), where v_cvt_pk_u16_u32 saturates the unsigned 32-bit
value to [0, 0xFFFF] (u32_to_u16) and v_cvt_pk_i16_i32 the signed value to [-32768, 32767] (i32_to_i16). The whole
dword is written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python reference. D leaves through exp as two
16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could
change.
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
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sat_u16(x: int) -> int:
    return min(x & M32, 0xFFFF)


def sat_i16(x: int) -> int:
    return max(-32768, min(32767, s32(x))) & 0xFFFF


OPS = {"v_cvt_pk_u16_u32": sat_u16, "v_cvt_pk_i16_i32": sat_i16}
VALUES = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFF8000,
          0xFFFF7FFF, 0xFFFFFFFE]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_pk_16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_pk_16.dxbc"), "wb") as f:
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
    asm = "; test_cvt_pk_16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", "v_mov_b32 v0, 0xdeadbeef", f"{op} v0, v1, v2"])


class CvtPk16Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(sat_u16(0x10000), 0xFFFF)
        self.assertEqual(sat_u16(0xFFFFFFFF), 0xFFFF)                # unsigned: -1 is huge
        self.assertEqual(sat_i16(0xFFFFFFFF), 0xFFFF)                # signed: -1 stays -1
        self.assertEqual(sat_i16(0x8000), 0x7FFF)
        self.assertEqual(sat_i16(0xFFFF7FFF), 0x8000)                # below -32768

    def test_values(self):
        for op, sat in OPS.items():
            for x in VALUES:
                for y in VALUES[::3]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(op, x, y), sat(x) | sat(y) << 16)

    def test_seeded(self):
        rng = random.Random(0xC16)
        for _ in range(30):
            op, x, y = rng.choice(list(OPS)), rng.getrandbits(32) >> rng.randrange(32), rng.getrandbits(32)
            if rng.getrandbits(1):
                x = (-x) & M32
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y), OPS[op](x) | OPS[op](y) << 16)

    def test_scalar_and_inline_operands(self):
        self.assertEqual(run_lines(["s_mov_b32 s2, 0x12345", "v_cvt_pk_i16_i32 v0, s2, -1"]), 0x7FFF | 0xFFFF << 16)
        self.assertEqual(run_lines(["s_mov_b32 s2, 0x12345", "v_cvt_pk_u16_u32 v0, s2, 64"]), 0xFFFF | 64 << 16)

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mov_b32 v1, 1", "v_cvt_pk_u16_u32 v0, v1, v1 clamp"])


if __name__ == "__main__":
    unittest.main()
