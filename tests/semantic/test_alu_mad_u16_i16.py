"""Semantic interpreter tests for RDNA2 v_mad_u16 and v_mad_i16.

D.u16 = S0.u16 * S1.u16 + S2.u16 on the low halves (VOP3 only). op_sel bits 0..2 read the high half of a source, bit 3
writes the result to the high half of the destination; the other half of the destination is kept (RDNA2 ISA; Mesa ACO
can_use_opsel, as for the other opsel-capable 16-bit VOP3 ops, docs/design.md "16-bit results"). With `clamp` the
full-width result saturates to [0, 65535] (u16) or [-32768, 32767] (i16). Each case translates a synthetic pixel
shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported word with a
plain-Python reference. The hardware behaviour of `clamp` on the multiply-add and of the kept half is not measured.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
A, B, C = 0x0003_0007, 0x0005_000B, 0x0002_000D      # lo / hi halves: (7, 3), (11, 5), (13, 2)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mad16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mad16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int, v3: int, old: int = OLD) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v2, {v2:#x}", f"v_mov_b32 v3, {v3:#x}", f"v_mov_b32 v0, {old:#x}", line]
    asm = "; test_mad16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def half(word: int, hi: int, signed: bool) -> int:
    h = (word >> 16 if hi else word) & 0xFFFF
    return h - 0x10000 if signed and h & 0x8000 else h


def ref(op: str, v1, v2, v3, sel=(0, 0, 0), dst_hi=0, clamp=False, old=OLD) -> int:
    signed = op.endswith("i16")
    x, y, z = (half(w, s, signed) for w, s in zip((v1, v2, v3), sel))
    r = x * y + z
    if clamp:
        lo, hi = (-0x8000, 0x7FFF) if signed else (0, 0xFFFF)
        r = max(lo, min(hi, r))
    r &= 0xFFFF
    return (r << 16 | old & 0xFFFF) if dst_hi else (old & 0xFFFF0000 | r)


class MadTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_mad_u16", A, B, C), 0xAAAA0000 | (7 * 11 + 13))
        self.assertEqual(ref("v_mad_i16", 0xFFFF, 3, 0), 0xAAAA0000 | 0xFFFD)

    def test_known_values(self):
        self.assertEqual(run("v_mad_u16 v0, v1, v2, v3", A, B, C), 0xAAAA0000 | 90)
        self.assertEqual(run("v_mad_i16 v0, v1, v2, v3", 0xFFFE, 3, 5), 0xAAAA0000 | (-1 & 0xFFFF))   # -2*3+5
        self.assertEqual(run("v_mad_u16 v0, v1, v2, v3", 0xFFFF, 0xFFFF, 0xFFFF), 0xAAAA0000 | ((0xFFFF * 0xFFFF + 0xFFFF) & 0xFFFF))

    def test_keeps_the_high_half_of_the_destination(self):
        self.assertEqual(run("v_mad_u16 v0, v1, v2, v3", A, B, C, old=0x12345678) >> 16, 0x1234)

    def test_op_sel_all_sixteen_patterns(self):
        for op in ("v_mad_u16", "v_mad_i16"):
            for bits in itertools.product((0, 1), repeat=4):
                line = f"{op} v0, v1, v2, v3 op_sel:[{bits[0]},{bits[1]},{bits[2]},{bits[3]}]"
                with self.subTest(line=line):
                    self.assertEqual(run(line, A, B, C), ref(op, A, B, C, bits[:3], bits[3]))

    def test_clamp_saturates(self):
        self.assertEqual(run("v_mad_u16 v0, v1, v2, v3 clamp", 0x100, 0x100, 5), 0xAAAA0000 | 0xFFFF)
        self.assertEqual(run("v_mad_u16 v0, v1, v2, v3 clamp", 3, 4, 5), 0xAAAA0000 | 17)
        self.assertEqual(run("v_mad_i16 v0, v1, v2, v3 clamp", 0x7FFF, 2, 0), 0xAAAA0000 | 0x7FFF)
        self.assertEqual(run("v_mad_i16 v0, v1, v2, v3 clamp", 0x7FFF, 0xFFFE, 0), 0xAAAA0000 | 0x8000)   # -65534 -> -32768
        self.assertEqual(run("v_mad_i16 v0, v1, v2, v3 clamp", 0xFFFF, 4, 0), 0xAAAA0000 | 0xFFFC)       # -4 unchanged

    def test_random_against_the_reference(self):
        rng = random.Random(0x16)
        special = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0xFFFE]
        pick = lambda: rng.choice(special) if rng.random() < 0.3 else rng.randrange(0x10000)
        for op in ("v_mad_u16", "v_mad_i16"):
            for clamp in (False, True):
                for _ in range(25):
                    v1, v2, v3 = (pick() | pick() << 16 for _ in range(3))
                    sel = tuple(rng.randrange(2) for _ in range(3))
                    line = f"{op} v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]},0]" + (" clamp" if clamp else "")
                    with self.subTest(line=line, v=(hex(v1), hex(v2), hex(v3))):
                        self.assertEqual(run(line, v1, v2, v3), ref(op, v1, v2, v3, sel, 0, clamp))


if __name__ == "__main__":
    unittest.main()
