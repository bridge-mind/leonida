"""Semantic interpreter tests for RDNA2 v_add_nc_i16 and v_sub_nc_i16 (VOP3-only, GFX10).

D.i16 = S0.i16 +/- S1.i16; the low 16 bits wrap like v_add_nc_u16 / v_sub_nc_u16, and clamp saturates the exact
signed result to [-32768, 32767]. op_sel:[s0,s1,dst] picks each source's half and the destination half; the other
half of D is kept (opsel-capable VOP3, as for the u16 forms). Each case translates a synthetic pixel shader and runs it
in tests/semantic/spirv_mini_interp.py; D leaves through exp as 16-bit halves ORed into 0x40000000, so no channel is a
NaN pattern a float round trip could change. Standard library only; no GPU, no game files.
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
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
_TMP = None


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


def ref(op, a, b, sel=(0, 0, 0), clamp=False, old=OLD):
    x, y = s16(a >> (16 * sel[0])), s16(b >> (16 * sel[1]))
    r = x + y if op == "v_add_nc_i16" else x - y
    if clamp:
        r = max(-32768, min(32767, r))
    r &= 0xFFFF
    return (old & 0xFFFF0000) | r if not sel[2] else (old & 0xFFFF) | r << 16


def translate(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_nci16.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    src = os.path.join(_TMP.name, "disasm", "ps_nci16.s")
    with open(src, "w", encoding="utf-8") as f:
        f.write("; ps_nci16\n" + "".join(f"\t{ln}\n" for ln in lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def run(line, a, b, old=OLD):
    lines = [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"v_mov_b32 v0, {old:#x}", line,
             "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_or_b32 v5, 0x40000000, v5",
             "v_or_b32 v6, 0x40000000, v6", "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm"]
    lo, hi = si.run_fragment(translate(lines))["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def mods(sel, clamp):
    m = f" op_sel:[{sel[0]},{sel[1]},{sel[2]}]" if any(sel) else ""
    return m + (" clamp" if clamp else "")


OPS = ("v_add_nc_i16", "v_sub_nc_i16")
VALUES = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x7FFE, 0x8001, 0x1234, 0x7FFF8000, 0x80017FFF]


class AddSubNcI16(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_add_nc_i16", 0x7FFF, 1) & 0xFFFF, 0x8000)                   # wraps
        self.assertEqual(ref("v_add_nc_i16", 0x7FFF, 1, clamp=True) & 0xFFFF, 0x7FFF)       # saturates
        self.assertEqual(ref("v_sub_nc_i16", 0x8000, 1, clamp=True) & 0xFFFF, 0x8000)
        self.assertEqual(ref("v_sub_nc_i16", 0x0000, 0x8000, clamp=True) & 0xFFFF, 0x7FFF)  # 0 - (-32768)

    def test_values_with_and_without_clamp(self):
        for op in OPS:
            for a in VALUES:
                for b in VALUES[::2]:
                    for clamp in (False, True):
                        with self.subTest(op=op, a=hex(a), b=hex(b), clamp=clamp):
                            self.assertEqual(run(f"{op} v0, v1, v2" + mods((0, 0, 0), clamp), a, b),
                                             ref(op, a, b, clamp=clamp))

    def test_op_sel(self):
        a, b = 0x7FFF1234, 0x80000001
        for op in OPS:
            for sel in ((1, 0, 0), (0, 1, 0), (1, 1, 1), (0, 0, 1)):
                for clamp in (False, True):
                    with self.subTest(op=op, sel=sel, clamp=clamp):
                        self.assertEqual(run(f"{op} v0, v1, v2" + mods(sel, clamp), a, b),
                                         ref(op, a, b, sel, clamp))

    def test_without_clamp_same_as_u16(self):
        rng = random.Random(0x16A)
        for _ in range(20):
            a, b = rng.getrandbits(32), rng.getrandbits(32)
            for i16, u16 in (("v_add_nc_i16", "v_add_nc_u16"), ("v_sub_nc_i16", "v_sub_nc_u16")):
                with self.subTest(op=i16, a=hex(a), b=hex(b)):
                    self.assertEqual(run(f"{i16} v0, v1, v2", a, b), run(f"{u16} v0, v1, v2", a, b))

    def test_sgpr_and_inline_constants(self):
        for op in OPS:
            with self.subTest(op=op):
                # s4 as src0, 1 as src1
                lines = [
                    "s_mov_b32 s4, 0x00000005",
                    "v_mov_b32 v0, 0xAAAA0000",
                    f"{op} v0, s4, 1",
                    "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                    "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6",
                    "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm",
                ]
                lo, hi = si.run_fragment(translate(lines))["%out_mrt0"][:2]
                got = (lo & 0xFFFF) | (hi & 0xFFFF) << 16
                want_lo = 6 if op == "v_add_nc_i16" else 4
                self.assertEqual(got, 0xAAAA0000 | want_lo)

    def test_literal_and_high_half_preservation(self):
        for op in OPS:
            with self.subTest(op=op):
                # 32-bit literal operand preserving high half (0xAAAA)
                lines = [
                    "v_mov_b32 v1, 0x00000002",
                    "v_mov_b32 v0, 0xAAAA0000",
                    f"{op} v0, 0x1234, v1",
                    "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                    "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6",
                    "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm",
                ]
                lo, hi = si.run_fragment(translate(lines))["%out_mrt0"][:2]
                got = (lo & 0xFFFF) | (hi & 0xFFFF) << 16
                want_lo = (0x1234 + 2) if op == "v_add_nc_i16" else (0x1234 - 2)
                self.assertEqual(got, 0xAAAA0000 | want_lo)

    def test_non_vgpr_opsel_dst_rejected(self):
        for op in OPS:
            with self.subTest(op=op):
                lines = [
                    "v_mov_b32 v1, 1",
                    "v_mov_b32 v2, 2",
                    f"{op} s4, v1, v2 op_sel:[0,0,1]",
                    "s_endpgm",
                ]
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(lines)


if __name__ == "__main__":
    unittest.main()
