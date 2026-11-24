"""v_fmamk_f16 / v_fmaak_f16 (VOP2 with a literal K) against v_fma_f16, run through spirv_mini_interp on the CPU.

RDNA2 ISA: v_fmamk_f16 D = S0 * K + S1 (printed `vdst, src0, K, vsrc1`), v_fmaak_f16 D = S0 * S1 + K (printed `vdst,
src0, vsrc1, K`), one rounding to f16; the literal's low 16 bits are K, and the GFX10 f16 VOP2 write keeps D's high
half. Each case must give the bits v_fma_f16 gives with K in a VGPR (v_fma_f16's rounding is tested elsewhere), in the
low half, with the high half kept. Halves are exported ORed into 0x40000000, so no channel is a NaN pattern a float
round trip could change. Standard library only; no GPU, no game files.
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
OLD = 0xABCD0000
_TMP = None


def translate(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_fmakf16.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    src = os.path.join(_TMP.name, "disasm", "ps_fmakf16.s")
    with open(src, "w", encoding="utf-8") as f:
        f.write("; ps_fmakf16\n" + "".join(f"\t{ln}\n" for ln in lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def run(line, regs):
    lines = [f"v_mov_b32 v{r}, {v:#x}" for r, v in regs.items()] + [f"v_mov_b32 v0, {OLD:#x}", line,
             "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_or_b32 v5, 0x40000000, v5",
             "v_or_b32 v6, 0x40000000, v6", "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm"]
    lo, hi = si.run_fragment(translate(lines))["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


F16 = [0x3C00, 0xBC00, 0x3800, 0x4900, 0x7BFF, 0x0001, 0x8400, 0x3555, 0xC500, 0x0000, 0x8000, 0x7C00]


def fma_ref(x, y, z):
    """v_fma_f16 v0, v1, v2, v3 with those f16 halves: the reference both literal forms must match."""
    return run("v_fma_f16 v0, v1, v2, v3", {1: x, 2: y, 3: z})


class FmamkFmaakF16(unittest.TestCase):
    def test_fmamk_is_s0_times_k_plus_s1(self):
        for x in F16[::2]:
            for k in F16:
                for y in F16[::3]:
                    with self.subTest(x=hex(x), k=hex(k), y=hex(y)):
                        got = run(f"v_fmamk_f16 v0, v1, {k:#x}, v2", {1: x, 2: y})
                        self.assertEqual(got, fma_ref(x, k, y))
                        self.assertEqual(got >> 16, OLD >> 16)          # high half kept

    def test_fmaak_is_s0_times_s1_plus_k(self):
        for x in F16[::2]:
            for y in F16[1::3]:
                for k in F16:
                    with self.subTest(x=hex(x), y=hex(y), k=hex(k)):
                        got = run(f"v_fmaak_f16 v0, v1, v2, {k:#x}", {1: x, 2: y})
                        self.assertEqual(got, fma_ref(x, y, k))

    def test_float_literal_k_and_operand_order(self):
        # A float literal is the f16 of its value; fmamk multiplies by K, fmaak adds it.
        two, three, half = 0x4000, 0x4200, 0x3800
        self.assertEqual(run("v_fmamk_f16 v0, v1, 0.5, v2", {1: two, 2: three}) & 0xFFFF, 0x4400)   # 2*0.5+3 = 4
        self.assertEqual(run("v_fmaak_f16 v0, v1, v2, 0.5", {1: two, 2: three}) & 0xFFFF, 0x4680)   # 2*3+0.5 = 6.5
        self.assertEqual(run(f"v_fmamk_f16 v0, v1, {half:#x}, v2", {1: two, 2: three}) & 0xFFFF, 0x4400)

    def test_seeded(self):
        rng = random.Random(0xF16A)
        for _ in range(25):
            x, y, k = (rng.getrandbits(16) & 0x7BFF | rng.getrandbits(1) << 15 for _ in range(3))
            with self.subTest(x=hex(x), y=hex(y), k=hex(k)):
                self.assertEqual(run(f"v_fmamk_f16 v0, v1, {k:#x}, v2", {1: x, 2: y}), fma_ref(x, k, y))


if __name__ == "__main__":
    unittest.main()
