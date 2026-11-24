"""Semantic interpreter tests for RDNA2 s_flbit_i32_b64 and s_flbit_i32_i64.

Leading-bit counts of the 64-bit SGPR pair S0, from bit 63 down:
  s_flbit_i32_b64: the number of 0 bits before the first 1; -1 (0xFFFFFFFF) when S0 is 0.
  s_flbit_i32_i64: the number of bits before the first one that differs from the sign bit (bit 63 itself counts,
                   as for v_ffbh_i32 / s_flbit_i32); -1 when every bit equals the sign (0 and -1).
SCC is not written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python reference. D leaves through exp as
two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a float round trip
could change; SCC comes back through s_cselect_b32.
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
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def flbit_b64(x: int) -> int:
    x &= M64
    return M32 if x == 0 else 64 - x.bit_length()


def flbit_i64(x: int) -> int:
    x &= M64
    return flbit_b64(~x & M64 if x >> 63 else x)


OPS = {"s_flbit_i32_b64": flbit_b64, "s_flbit_i32_i64": flbit_i64}
VALUES = [0, 1, M64, 1 << 63, (1 << 63) - 1, 1 << 32, 0xFFFFFFFF, 0x80000000, 0x00000001FFFFFFFF,
          0xFFFFFFFF00000000, 0xFFFFFFFE00000000, 0xC000000000000000, 0x4000000000000000, 0xFFFFFFFFFFFFFFFE]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_flbit64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_flbit64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, return (s0, SCC)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = ["s_cselect_b32 s20, 0x40000001, 0x40000000", "s_and_b32 s10, s0, 0xffff", "s_lshr_b32 s11, s0, 16",
            "s_or_b32 s10, s10, 0x40000000", "s_or_b32 s11, s11, 0x40000000",
            "v_mov_b32 v0, s10", "v_mov_b32 v1, s11", "v_mov_b32 v2, s20"]
    asm = ("; test_salu_flbit64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


def run_op(op: str, x: int, scc_in: int = 0) -> tuple:
    return run_lines([f"s_mov_b32 s2, {x & M32:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"s_mov_b32 s4, {scc_in}",
                      "s_cmp_eq_u32 s4, 1", f"{op} s0, s[2:3]"])


class Flbit64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(flbit_b64(1), 63)
        self.assertEqual(flbit_b64(1 << 32), 31)
        self.assertEqual(flbit_b64(0), M32)
        self.assertEqual(flbit_i64(1), 63)
        self.assertEqual(flbit_i64(0xFFFFFFFFFFFFFFFE), 63)            # -2: bit 0 is the first to differ
        self.assertEqual(flbit_i64(1 << 63), 1)
        self.assertEqual(flbit_i64(M64), M32)

    def test_values_and_scc_kept(self):
        for op, fn in OPS.items():
            for x in VALUES:
                for scc_in in (0, 1):
                    with self.subTest(op=op, x=hex(x), scc=scc_in):
                        self.assertEqual(run_op(op, x, scc_in), (fn(x), scc_in))

    def test_seeded(self):
        rng = random.Random(0xF1B)
        for _ in range(30):
            op = rng.choice(list(OPS))
            x = rng.getrandbits(64) >> rng.randrange(64)
            if rng.getrandbits(1):
                x = ~x & M64
            with self.subTest(op=op, x=hex(x)):
                self.assertEqual(run_op(op, x)[0], OPS[op](x))

    def test_inline_source(self):
        self.assertEqual(run_lines(["s_flbit_i32_b64 s0, 1"])[0], 63)
        self.assertEqual(run_lines(["s_flbit_i32_i64 s0, -1"])[0], M32)

    def test_lane_mask_source_is_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["s_flbit_i32_b64 s0, exec"])


if __name__ == "__main__":
    unittest.main()
