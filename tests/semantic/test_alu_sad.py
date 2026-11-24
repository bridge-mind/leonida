"""Semantic interpreter tests for RDNA2 v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32 and v_msad_u8.

Sums of absolute differences, unsigned, wrapping at 32 bits (RDNA ISA pseudocode):
  v_sad_u8:    D = S2 + sum over the 4 bytes of |S0.byte - S1.byte|
  v_sad_hi_u8: D = S2 + (the same byte sum << 16)
  v_sad_u16:   D = S2 + sum over the 2 halves of |S0.half - S1.half|
  v_sad_u32:   D = S2 + |S0 - S1|
  v_msad_u8:   v_sad_u8, skipping each byte where S1's byte is 0 (masked SAD)
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the result with a plain-Python reference. The result leaves through exp as two 16-bit halves, each ORed
into 0x40000000, so an export channel is never a NaN or a denormal pattern that a float round trip could change.
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def parts(x: int, width: int) -> list:
    return [(x >> off) & ((1 << width) - 1) for off in range(0, 32, width)]


def sad(x: int, y: int, width: int, masked: bool = False) -> int:
    return sum(abs(p - q) for p, q in zip(parts(x, width), parts(y, width)) if not (masked and q == 0))


OPS = {
    "v_sad_u8": lambda x, y, z: (sad(x, y, 8) + z) & MASK32,
    "v_sad_hi_u8": lambda x, y, z: ((sad(x, y, 8) << 16) + z) & MASK32,
    "v_sad_u16": lambda x, y, z: (sad(x, y, 16) + z) & MASK32,
    "v_sad_u32": lambda x, y, z: (sad(x, y, 32) + z) & MASK32,
    "v_msad_u8": lambda x, y, z: (sad(x, y, 8, masked=True) + z) & MASK32,
}
VALUES = [0, 0xFFFFFFFF, 0x00FF00FF, 0xFF00FF00, 0x01020304, 0x80808080, 0x7F7F7F7F, 0xFFFF0000, 0x12345678]
ACCS = [0, 1, 0xFFFFFFFF, 0xFFFFFC00, 0x7FFF0000]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sad.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sad.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    """Run the lines, return v0 (exported as two 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_sad\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int, z: int) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"v_mov_b32 v3, {z:#x}", f"{op} v0, v1, v2, v3"])


class SadTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(OPS["v_sad_u8"](0x01020304, 0x04030201, 10), 10 + 3 + 1 + 1 + 3)
        self.assertEqual(OPS["v_sad_hi_u8"](0xFF, 0, 1), (255 << 16) + 1)
        self.assertEqual(OPS["v_sad_u16"](0x0000FFFF, 0xFFFF0000, 0), 2 * 0xFFFF)
        self.assertEqual(OPS["v_sad_u32"](0, 0xFFFFFFFF, 1), 0)                    # wraps at 32 bits
        self.assertEqual(OPS["v_msad_u8"](0x10203040, 0x00200030, 0), 0x10 + 0)    # bytes with S1 = 0 skipped

    def test_edge_values(self):
        for op, fn in OPS.items():
            for x in VALUES:
                for y in VALUES:
                    z = ACCS[(x ^ y) % len(ACCS)]
                    with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                        self.assertEqual(run_op(op, x, y, z), fn(x, y, z))

    def test_seeded(self):
        rng = random.Random(0x5AD)
        for _ in range(40):
            op = rng.choice(list(OPS))
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            if op == "v_msad_u8" and rng.random() < 0.5:
                y &= rng.choice([0xFF00FF00, 0x00FFFF00, 0xFFFFFF00])       # zero some reference bytes
            with self.subTest(op=op, x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(op, x, y, z), OPS[op](x, y, z))

    def test_scalar_and_inline_operands(self):
        self.assertEqual(run_lines(["s_mov_b32 s2, 0x01010101", "v_mov_b32 v1, 0x03030303", "v_sad_u8 v0, s2, v1, 7"]),
                         7 + 4 * 2)

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mov_b32 v1, 1", "v_sad_u8 v0, v1, v1, v1 clamp"])


if __name__ == "__main__":
    unittest.main()
