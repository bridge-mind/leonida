"""Semantic interpreter tests for RDNA2 64-bit vector logical shift left v_lshlrev_b64.

1. Reference Oracle: Plain-Python RDNA2 ISA reference for v_lshlrev_b64 (D = S1 << (S0 & 63)).
2. SIMD Multi-Lane: Tests across multiple VGPR register pairs and values.
3. Zero-fill extremes, shift count modulo 64, and negative controls.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

def ref_lshlrev_b64(n: int, x: int) -> int:
    """Plain-Python RDNA2 ISA reference for v_lshlrev_b64."""
    return (x << (n & 63)) & M64

_TMP_DIR = None
_SRC_PATH = None

def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_lshl_b64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_lshl_b64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)

def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None

EXPORT_HALVES = [
    "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1", "v_lshrrev_b32 v8, 16, v1",
    "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7", "v_or_b32 v8, 0x40000000, v8"
]

def run_lines(lines) -> int:
    if _TMP_DIR is None:
        setUpModule()
    body = list(lines) + EXPORT_HALVES
    asm = "; test_lshl_b64\n" + "".join(f"\t{ln}\n" for ln in body) + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(x >> 16 == 0x4000 for x in w), [hex(x) for x in w]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w))

def run_op(n: int, x: int) -> int:
    return run_lines([
        f"v_mov_b32 v4, {n:#x}",
        f"v_mov_b32 v2, {x & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {x >> 32:#x}",
        "v_lshlrev_b64 v[0:1], v4, v[2:3]"
    ])

class ALUIVectorLshlB64Tests(unittest.TestCase):
    def test_reference_oracle(self):
        self.assertEqual(ref_lshlrev_b64(0, 1), 1)
        self.assertEqual(ref_lshlrev_b64(1, 1), 2)
        self.assertEqual(ref_lshlrev_b64(32, 1), 1 << 32)
        self.assertEqual(ref_lshlrev_b64(63, 1), 1 << 63)
        self.assertEqual(ref_lshlrev_b64(64, 1), 1)  # mod 64 -> 0
        self.assertEqual(ref_lshlrev_b64(0x80000020, 1), 1 << 32)

    def test_v_lshlrev_b64_basic(self):
        self.assertEqual(run_op(1, 0x0000000100000001), 0x0000000200000002)

    def test_simd_multilane_pairs(self):
        pairs = [
            (4, 2, 3),
            (6, 8, 9),
        ]
        for dst_reg, src_lo, src_hi in pairs:
            n = 5
            x = 0x123456789ABCDEF0
            lines = [
                f"v_mov_b32 v10, {n:#x}",
                f"v_mov_b32 v{src_lo}, {x & 0xFFFFFFFF:#x}",
                f"v_mov_b32 v{src_hi}, {x >> 32:#x}",
                f"v_lshlrev_b64 v[{dst_reg}:{dst_reg+1}], v10, v[{src_lo}:{src_hi}]",
                f"v_mov_b32 v0, v{dst_reg}",
                f"v_mov_b32 v1, v{dst_reg+1}"
            ]
            self.assertEqual(run_lines(lines), ref_lshlrev_b64(n, x))

    def test_extremes_and_modulo(self):
        shifts = [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 0xFFFFFFFF]
        values = [0, 1, M64, 1 << 63, 0x0123456789ABCDEF]
        for n in shifts:
            for x in values:
                with self.subTest(n=n, x=hex(x)):
                    self.assertEqual(run_op(n, x), ref_lshlrev_b64(n, x))

    def test_negative_controls(self):
        # Negative control: verify that incorrect shift or wrong mask fails
        n = 4
        x = 0x0000000100000001
        correct = ref_lshlrev_b64(n, x)
        actual = run_op(n, x)
        self.assertEqual(actual, correct)
        # Explicitly ensure it does NOT equal an unshifted or wrongly shifted value
        self.assertNotEqual(actual, x)
        self.assertNotEqual(actual, ref_lshlrev_b64(n + 1, x))

if __name__ == "__main__":
    unittest.main()
