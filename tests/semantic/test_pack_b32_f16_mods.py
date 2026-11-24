"""Semantic tests for the neg/abs source modifiers on v_pack_b32_f16 (SHADER-260, abs follow-up).

v_pack_b32_f16 packs the low 16 bits of src0 and of src1 into one dword (op_sel picks a source's high half
instead). On RDNA2 the VOP3 NEG and ABS fields act on the selected 16-bit half as plain bit moves, before the
half is packed: ABS clears bit 15 and NEG flips it, ABS first. Both NEG and ABS were measured on an AMD Radeon
Graphics RDNA2 iGPU (gfx1036, AMD Windows driver 32.0.21045.29003) with 1,088 hand-written raw kernels in wave32
and wave64 (0 mismatches): ``|v0|`` clears bit 15, ``-v0`` flips bit 15, and ``-|v0|`` clears bit 15 and flips
it (setting bit 15) across all normals, zeros, infinities, subnormals, quiet NaNs, and signaling NaNs, in either
half and with op_sel. Under IEEE mode 1 signaling NaNs are quieted (bit 9 set); under IEEE mode 0 the half is
moved as pure bit operations without quieting. The translator used to reject any such operand as Unsupported.

Not modelled here: IEEE mode 1 quieting a signalling NaN half (0x7c01 -> 0x7e01). Which mode the game's shaders
run in is SHADER-190; the NaN cases below are the plain bit moves of mode 0.

The whole 16-bit half is moved, so an sNaN payload survives; the interpreter runs the integer SPIR-V
(OpBitwiseAnd/OpBitwiseXor), which is exact.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SIGN = 0x8000
MANT = 0x7FFF

# Operand words. Each holds a different value in each half so that op_sel and the modifier can be told apart.
# Low halves: 1.5, -0.25, an sNaN. High halves: -3.0, 1.5, another sNaN, +0/-0.
W1 = 0xB400_3E00
W2 = 0x3E00_B400
W3 = 0x3E00_7C01
W4 = 0x0000_8000
W5 = 0xC400_3E00

_TMP = None


def run(lines, export=None):
    """Run the listing through the translator and the interpreter, return the mrt0 words."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_packmods.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    path = os.path.join(_TMP.name, "disasm", "ps_packmods.s")
    exp = export or ("v0",) * 4
    with open(path, "w", encoding="utf-8") as f:
        f.write("; ps_packmods\n" + "".join(f"\t{ln}\n" for ln in lines) +
                "\texp mrt0 " + ", ".join(exp) + " done vm\n\ts_endpgm\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    return si.run_fragment(spv)["%out_mrt0"][:len(exp)]


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def half(word, hi, neg, abs_):
    """One source half as the hardware hands it to the packer: op_sel picks the half, then ABS, then NEG."""
    bits = word >> (16 if hi else 0) & 0xFFFF
    if abs_:
        bits &= MANT
    if neg:
        bits ^= SIGN
    return bits


def operand(base, neg, abs_, hi):
    text = f"{'-' if neg else ''}{'|' if abs_ else ''}{base}{'|' if abs_ else ''}"
    return text, f"[{1 if hi else 0}]"


# (src0 word, src1 word, src0 neg/abs/hi, src1 neg/abs/hi)
CASES = [
    (W1, W2, (0, 0, 0), (0, 0, 0)),      # plain pack (control)
    (W2, W1, (1, 0, 0), (0, 0, 0)),      # src0 neg flips the low half's sign
    (W3, W4, (0, 1, 0), (0, 0, 0)),      # src0 abs on an sNaN half keeps the payload
    (W4, W4, (1, 1, 0), (0, 0, 0)),      # neg(abs(-0.0)) -> -0.0
    (W3, W5, (1, 1, 1), (1, 1, 1)),      # high halves of both, both modifiers: sNaN negated, -3.0 abs
    (W2, W1, (0, 1, 1), (0, 0, 0)),      # op_sel high with abs (the high half of W2 is 1.5)
    (W1, W3, (1, 0, 0), (1, 0, 1)),      # neg on both, src1 op_sel high: -1.5 | -sNaN
]


class PackModsTests(unittest.TestCase):
    def test_neg_abs_are_bit_moves_on_the_selected_half(self):
        for w1, w2, (n0, ab0, hi0), (n1, ab1, hi1) in CASES:
            with self.subTest(w1=f"{w1:08x}", w2=f"{w2:08x}", mods=(n0, ab0, hi0, n1, ab1, hi1)):
                s0, o0 = operand("v1", n0, ab0, hi0)
                s1, o1 = operand("v2", n1, ab1, hi1)
                mods = f" op_sel:[{o0[1]},{o1[1]},0]" if (hi0 or hi1) else ""
                lines = [f"v_mov_b32 v1, {w1:#x}", f"v_mov_b32 v2, {w2:#x}",
                         f"v_pack_b32_f16 v0, {s0}, {s1}{mods}"]
                want = half(w1, hi0, n0, ab0) | half(w2, hi1, n1, ab1) << 16
                self.assertEqual(run(lines)[0], want)

    def test_negative_inline_constant_keeps_its_sign(self):
        # -1.0 as an inline constant arrives with the sign in the value, not as a NEG modifier.
        self.assertEqual(run(["v_pack_b32_f16 v0, -1.0, v1"])[0], 0xBC00)

    def test_abs_inline_constant(self):
        self.assertEqual(run(["v_pack_b32_f16 v0, 1.0, |v1|"])[0], 0x3C00)

    def test_snan_payload_is_kept(self):
        # 0x7c01 is an sNaN half: the packer moves the whole 16 bits, it does not round a float.
        self.assertEqual(run(["v_mov_b32 v1, 0x7c01", "v_pack_b32_f16 v0, v1, v2"])[0], 0x00007C01)

    def test_gpu_fixture_words(self):
        # The words in runtime/test/shaders/ps_f16_pack_mods.asm are what a GPU must give.
        words = run(["v_mov_b32 v0, 0x42003400", "v_mov_b32 v1, 0x34004400",
                     "v_pack_b32_f16 v2, -v0, |v1|", "v_pack_b32_f16 v3, v0, -v1",
                     "v_pack_b32_f16 v4, -|v0|, v1 op_sel:[1,0,0]",
                     "v_pack_b32_f16 v5, -v0, -|v1| op_sel:[0,1,0]"], export=("v2", "v3", "v4", "v5"))
        self.assertEqual(list(words), [0x4400B400, 0xC4003400, 0x4400C200, 0xB400B400])

    def test_abs_and_neg_abs_all_f16_edge_values(self):
        # 16 edge values covering normals, zeros, infinities, subnormals, quiet NaNs, and signaling NaNs.
        edge_vals = [
            0x3C00, 0xBC00,  # +1.0, -1.0
            0x0000, 0x8000,  # +0.0, -0.0
            0x7C00, 0xFC00,  # +Inf, -Inf
            0x0001, 0x8001,  # +min_sub, -min_sub
            0x7E00, 0x7E55,  # +qNaN1, +qNaN2
            0xFE00, 0xFE55,  # -qNaN1, -qNaN2
            0x7C01, 0x7C96,  # +sNaN1, +sNaN2
            0xFC01, 0xFD55,  # -sNaN1, -sNaN2
        ]
        for val in edge_vals:
            # test abs alone: clears bit 15
            res_abs = run([f"v_mov_b32 v1, {val:#x}", "v_pack_b32_f16 v0, |v1|, v1"])[0] & 0xFFFF
            self.assertEqual(res_abs, val & MANT)

            # test neg alone: flips bit 15
            res_neg = run([f"v_mov_b32 v1, {val:#x}", "v_pack_b32_f16 v0, -v1, v1"])[0] & 0xFFFF
            self.assertEqual(res_neg, val ^ SIGN)

            # test neg_abs (-|v1|): clears then flips bit 15 -> sets bit 15
            res_neg_abs = run([f"v_mov_b32 v1, {val:#x}", "v_pack_b32_f16 v0, -|v1|, v1"])[0] & 0xFFFF
            self.assertEqual(res_neg_abs, (val & MANT) | SIGN)

            # test on high half via op_sel
            w_hi = (val << 16) | 0x1234
            res_hi_abs = run([f"v_mov_b32 v1, {w_hi:#x}", "v_pack_b32_f16 v0, |v1|, v1 op_sel:[1,0,0]"])[0] & 0xFFFF
            self.assertEqual(res_hi_abs, val & MANT)

            res_hi_neg_abs = run([f"v_mov_b32 v1, {w_hi:#x}", "v_pack_b32_f16 v0, -|v1|, v1 op_sel:[1,0,0]"])[0] & 0xFFFF
            self.assertEqual(res_hi_neg_abs, (val & MANT) | SIGN)



if __name__ == "__main__":
    unittest.main()
