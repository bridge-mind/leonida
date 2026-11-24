"""Semantic interpreter tests for RDNA2 s_pack_ll_b32_b16, s_pack_lh_b32_b16 and s_pack_hh_b32_b16.

D = {S1 half, S0 half}: the low half of the result comes from S0 and the high half from S1. The two letters name
the half taken from each source, S0 first (l = bits 15:0, h = bits 31:16):
  s_pack_ll_b32_b16: D = S1[15:0] << 16 | S0[15:0]
  s_pack_lh_b32_b16: D = S1[31:16] << 16 | S0[15:0]
  s_pack_hh_b32_b16: D = S1[31:16] << 16 | S0[31:16]
SCC is not written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported words with a plain-Python reference.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = {
    "s_pack_ll_b32_b16": lambda x, y: (y & 0xFFFF) << 16 | (x & 0xFFFF),
    "s_pack_lh_b32_b16": lambda x, y: (y & 0xFFFF0000) | (x & 0xFFFF),
    "s_pack_hh_b32_b16": lambda x, y: (y & 0xFFFF0000) | (x >> 16),
}
VALUES = [0, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000, 0x12345678, 0x9ABCDEF0, 0x80008000, 0x00010001]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_pack.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_pack.dxbc"), "wb") as f:
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
    asm = "; test_salu_pack\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def run_op(op: str, x: int, y: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = op(s2, s3); return (s0, SCC after)."""
    return run_lines([f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}", f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1",
                      f"{op} s0, s2, s3", "s_cselect_b32 s1, 1, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])


class PackTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(OPS["s_pack_ll_b32_b16"](0x11112222, 0x33334444), 0x44442222)
        self.assertEqual(OPS["s_pack_lh_b32_b16"](0x11112222, 0x33334444), 0x33332222)
        self.assertEqual(OPS["s_pack_hh_b32_b16"](0x11112222, 0x33334444), 0x33331111)

    def test_values_and_scc_kept(self):
        for op, fn in OPS.items():
            for x in VALUES:
                for y in VALUES:
                    for scc_in in (0, 1):
                        with self.subTest(op=op, x=hex(x), y=hex(y), scc=scc_in):
                            self.assertEqual(run_op(op, x, y, scc_in), (fn(x, y), scc_in))

    def test_seeded(self):
        rng = random.Random(0xB16)
        for _ in range(30):
            op = rng.choice(list(OPS))
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y, 0)[0], OPS[op](x, y))

    def test_literal_and_inline_operands(self):
        got = run_lines(["s_pack_ll_b32_b16 s0, 0x1234, -1", "s_pack_hh_b32_b16 s1, 0xabcd0000, 0x5678ffff",
                         "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
        self.assertEqual(got, (0xFFFF1234, 0x5678ABCD))


if __name__ == "__main__":
    unittest.main()
