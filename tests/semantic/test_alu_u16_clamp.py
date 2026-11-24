"""Semantic interpreter tests for the VOP3 clamp bit on the 16-bit unsigned add/sub (GFX10).

v_add_nc_u16 / v_sub_nc_u16 (and the v_add_u16_e64 / v_sub_u16_e64 names the translator also takes; v_subrev_u16 is
a GFX9 spelling the translator no longer accepts, #3677): D.u16 = S0.u16 + S1.u16 / S0.u16 - S1.u16, wrapping at 16 bits. With the clamp bit the result saturates to
[0, 0xFFFF], as for the 32-bit ops in test_alu_add_nc_clamp.py (llvm-mc gfx1030 encodes the clamp bit on
these ops, and LLVM selects i16 uaddsat/usubsat to them). The VOP3 form keeps the high half
of the destination. D leaves through exp as two 16-bit halves ORed into 0x40000000.
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


def ref(op: str, x: int, y: int, clamp: bool, old: int) -> int:
    x, y = x & 0xFFFF, y & 0xFFFF
    if "subrev" in op:
        x, y = y, x
    r = x - y if "sub" in op else x + y
    if clamp:
        r = min(max(r, 0), 0xFFFF)
    return (old & 0xFFFF0000) | (r & 0xFFFF)


OPS = ["v_add_nc_u16", "v_sub_nc_u16", "v_add_u16_e64", "v_sub_u16_e64"]
VALUES = [0, 1, 2, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF, 0x1234FFFF, 0xABCD0001]
OLD = 0x5A5A0000

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_u16_clamp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_u16_clamp.dxbc"), "wb") as f:
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
    asm = "; test_u16_clamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16

def run_op(op: str, x: int, y: int, clamp: bool) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"v_mov_b32 v0, {OLD:#x}",
                      f"{op} v0, v1, v2" + (" clamp" if clamp else "")])


class U16ClampTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_add_nc_u16", 0xFFFF, 1, False, 0), 0)
        self.assertEqual(ref("v_add_nc_u16", 0xFFFF, 1, True, 0), 0xFFFF)
        self.assertEqual(ref("v_sub_nc_u16", 1, 2, True, 0), 0)

    def test_values(self):
        for op in OPS:
            for clamp in (False, True):
                for x in VALUES:
                    for y in VALUES:
                        with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                            self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp, OLD))

    def test_seeded(self):
        rng = random.Random(0x16C)
        for _ in range(40):
            op, clamp = rng.choice(OPS), bool(rng.getrandbits(1))
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp, OLD))


if __name__ == "__main__":
    unittest.main()
