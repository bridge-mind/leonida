"""Semantic interpreter tests for RDNA2 s_cmov_b32 and s_cmovk_i32.

s_cmov_b32: D = SCC ? S0 : D. s_cmovk_i32: D = SCC ? sign_extend(simm16) : D. Both read SCC and leave it unchanged.
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the result with a plain-Python reference. D leaves through exp as two 16-bit halves ORed into
0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could change; SCC comes back
through s_cselect_b32.
"""
import os
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
OLD = 0xDEADBEEF
VALUES = [0, 1, 0x7FFFFFFF, 0x80000000, M32, 0x12345678, 0x7F800001]
SIMM16 = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xFF80]


def sext16(v: int) -> int:
    v &= 0xFFFF
    return (v - 0x10000 if v & 0x8000 else v) & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_cmov.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_cmov.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(line: str, x: int, scc_in: int) -> tuple:
    """s0 = OLD, s2 = x, SCC = scc_in, then ``line``; return (s0, SCC after)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"s_mov_b32 s0, {OLD:#x}", f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1", line,
             "s_cselect_b32 s1, 0x40000001, 0x40000000", "s_and_b32 s5, s0, 0xffff", "s_lshr_b32 s6, s0, 16",
             "s_or_b32 s5, s5, 0x40000000", "s_or_b32 s6, s6, 0x40000000",
             "v_mov_b32 v0, s5", "v_mov_b32 v1, s6", "v_mov_b32 v2, s1"]
    asm = "; test_salu_cmov\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v2 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, scc = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == hi >> 16 == 0x4000 and scc >> 1 == 0x20000000, (hex(lo), hex(hi), hex(scc))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16, scc & 1


class CmovTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(sext16(0x8000), 0xFFFF8000)
        self.assertEqual(sext16(0x7FFF), 0x7FFF)

    def test_s_cmov_b32(self):
        for x in VALUES:
            for scc_in in (0, 1):
                with self.subTest(x=hex(x), scc=scc_in):
                    self.assertEqual(run_op("s_cmov_b32 s0, s2", x, scc_in), (x if scc_in else OLD, scc_in))

    def test_s_cmov_b32_constant_source(self):
        self.assertEqual(run_op("s_cmov_b32 s0, -1", 0, 1), (M32, 1))
        self.assertEqual(run_op("s_cmov_b32 s0, 0x12345678", 0, 0), (OLD, 0))

    def test_s_cmovk_i32(self):
        for k in SIMM16:
            for scc_in in (0, 1):
                with self.subTest(k=hex(k), scc=scc_in):
                    self.assertEqual(run_op(f"s_cmovk_i32 s0, {k:#x}", 0, scc_in), (sext16(k) if scc_in else OLD, scc_in))

    def test_scc_from_a_compare(self):
        # The usual shape: a compare sets SCC, then the move depends on it.
        for x, moved in ((9, True), (5, False)):
            with self.subTest(x=x):
                got = run_op("s_cmp_gt_u32 s2, 5\n\ts_cmovk_i32 s0, 7", x, 0)
                self.assertEqual(got, (7 if moved else OLD, int(moved)))


if __name__ == "__main__":
    unittest.main()
