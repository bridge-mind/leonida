"""Semantic interpreter tests for RDNA2 v_mul_hi_u32_u24 and v_mul_hi_i32_i24.

Both multiply the low 24 bits of each source into a 48-bit product and return its high 32 bits (bits 32-47, so at
most 16 significant bits): v_mul_hi_u32_u24 zero-extends the operands, v_mul_hi_i32_i24 sign-extends them from
bit 23 and the result is the signed high word. Bits 24-31 of the sources are ignored. The low words are
v_mul_u32_u24 / v_mul_i32_i24, so the pair must give back the whole product. Each case translates a synthetic
pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported
words with a plain-Python reference.
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
VALUES = [0, 1, 2, 0x7FFFFF, 0x800000, 0xFFFFFF, 0x1000000, 0xFF800000, 0xFFFFFFFF, 0x12345678, 0x00ABCDEF,
          0x80000000, 0x7FFFFFFF]


def u24(x: int) -> int:
    return x & 0xFFFFFF


def i24(x: int) -> int:
    x &= 0xFFFFFF
    return x - (1 << 24) if x >> 23 else x


def ref(op: str, x: int, y: int) -> int:
    p = u24(x) * u24(y) if op == "v_mul_hi_u32_u24" else i24(x) * i24(y)
    return (p >> 32) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mul_hi_24.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mul_hi_24.dxbc"), "wb") as f:
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
    asm = "; test_mul_hi_24\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def hi_lo(op: str, x: int, y: int) -> tuple:
    lo_op = "v_mul_u32_u24" if op == "v_mul_hi_u32_u24" else "v_mul_i32_i24"
    return run_lines([f"v_mov_b32 v2, {x:#x}", f"v_mov_b32 v3, {y:#x}", f"{op} v1, v2, v3", f"{lo_op}_e32 v0, v2, v3"])


class MulHi24Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_mul_hi_u32_u24", 0xFFFFFF, 0xFFFFFF), 0xFFFF)      # (2^24-1)^2 >> 32
        self.assertEqual(ref("v_mul_hi_i32_i24", 0xFFFFFF, 0x7FFFFF), 0xFFFFFFFF)  # -1 * (2^23-1): high word -1
        self.assertEqual(ref("v_mul_hi_i32_i24", 0x800000, 0x800000), 0x4000)      # (-2^23)^2 = 2^46
        self.assertEqual(ref("v_mul_hi_u32_u24", 0xFF000001, 0xFF000001), 0)       # bits 24-31 ignored

    def test_edge_values(self):
        for op in ("v_mul_hi_u32_u24", "v_mul_hi_i32_i24"):
            for x in VALUES:
                for y in VALUES[::3]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(hi_lo(op, x, y)[1], ref(op, x, y))

    def test_hi_and_lo_give_the_whole_product(self):
        rng = random.Random(24)
        for op in ("v_mul_hi_u32_u24", "v_mul_hi_i32_i24"):
            for _ in range(25):
                x, y = rng.getrandbits(32), rng.getrandbits(32)
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    lo, hi = hi_lo(op, x, y)
                    want = u24(x) * u24(y) if op == "v_mul_hi_u32_u24" else i24(x) * i24(y)
                    self.assertEqual(hi << 32 | lo, want & 0xFFFFFFFFFFFFFFFF)

    def test_vop3_form_with_scalar_and_inline_operands(self):
        got = run_lines(["s_mov_b32 s2, 0xfff000", "v_mov_b32 v2, 0x800001",
                         "v_mul_hi_i32_i24_e64 v1, s2, v2", "v_mul_hi_u32_u24_e64 v0, -1, v2"])
        self.assertEqual(got, (ref("v_mul_hi_u32_u24", MASK32, 0x800001), ref("v_mul_hi_i32_i24", 0xFFF000, 0x800001)))


if __name__ == "__main__":
    unittest.main()
