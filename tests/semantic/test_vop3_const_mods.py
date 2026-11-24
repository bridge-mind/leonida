"""Semantic interpreter tests for VOP3 FP input modifiers on numeric constants and literals.

LLVM AMDGPUInstPrinter::printOperandAndFPInputMods prints VOP3 FP source modifiers on inline constants and 32-bit
literals as ``neg(<imm>)``, ``|<imm>|`` and ``-|<imm>|`` (and the assembler also accepts ``abs(<imm>)`` and
``neg(abs(<imm>))``). In hardware, ``abs`` clears the sign bit first and ``neg`` flips the sign bit second:

  * ``-|1.0|``, ``-|-2.0|``, ``neg(|1.0|)`` -> ``-1.0`` / ``-2.0`` / ``-1.0`` (previously ``shader_ir`` folded the
    outer ``-`` into ``v`` with ``abs=True`` and no ``neg``, so ``FAbs`` turned them positive).
  * ``neg(0x3f800000)`` -> ``0xbf800000`` (``-1.0f``) and ``neg(0x3c00)`` -> ``0xbc00`` (``-1.0h``) (previously
    ``shader_ir`` did two's-complement negation, producing ``-4.0f`` / ``-4.0h``).
  * ``neg(1)`` -> ``0x80000001`` (negative f32 subnormal), NOT ``-1`` (``0xffffffff``, quiet NaN).
  * ``|0xbf800000|``, ``-|0x3f800000|``, ``-|0|`` preserve ``abs`` and ``neg`` on integer/hex literals (previously
    ``shader_ir`` dropped ``abs`` on ``imm`` operands).

Each case translates a synthetic pixel shader with ``tools/rdna2spirv.py`` and executes it in
``tests/semantic/spirv_mini_interp.py`` (no llvm-mc, SPIRV-Tools, GPU or game data needed).
"""
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_vop3_const_mods.s")
    with open(os.path.join(_TMP_DIR.name, "ps_vop3_const_mods.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines, export="v0, v1, v2, v3"):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_vop3_const_mods\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + f"\texp mrt0 {export} done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run(lines, export="v0, v1, v2, v3"):
    return list(si.run_fragment(translate(lines, export=export))["%out_mrt0"])


class F32ConstModifierTests(unittest.TestCase):
    def test_fimm_abs_and_neg_combinations(self):
        # -|1.0| = -1.0, -|-2.0| = -2.0, neg(|1.0|) = -1.0, -|0.15915494| = -1/(2*pi)
        got = run([
            "v_add_f32_e64 v0, 0, -|1.0|",
            "v_add_f32_e64 v1, 0, -|-2.0|",
            "v_add_f32_e64 v2, 0, neg(|1.0|)",
            "v_add_f32_e64 v3, 0, -|0.15915494|",
        ])
        want_inv2pi = struct.unpack("<I", struct.pack("<f", -0.15915494))[0]
        self.assertEqual(got, [0xBF800000, 0xC0000000, 0xBF800000, want_inv2pi])

    def test_hex_literal_fp_modifiers(self):
        # neg(0x3f800000) = -1.0f (not two's-complement -4.0f = 0xc0800000),
        # |0xbf800000| = +1.0f, -|0x3f800000| = -1.0f, neg(abs(0x3f800000)) = -1.0f.
        got = run([
            "v_add_f32_e64 v0, 0, neg(0x3f800000)",
            "v_add_f32_e64 v1, 0, |0xbf800000|",
            "v_add_f32_e64 v2, 0, -|0x3f800000|",
            "v_add_f32_e64 v3, 0, neg(abs(0x3f800000))",
        ])
        self.assertEqual(got, [0xBF800000, 0x3F800000, 0xBF800000, 0xBF800000])

    def test_abs_func_form_and_signed_zero(self):
        # -|0| = -0.0, neg(0) = -0.0, abs(0xbf800000) = +1.0, -abs(1.0) = -1.0.
        # Multiply by +1.0 preserves -0.0 in IEEE binary32.
        got = run([
            "v_mul_f32_e64 v0, 1.0, -|0|",
            "v_mul_f32_e64 v1, 1.0, neg(0)",
            "v_add_f32_e64 v2, 0, abs(0xbf800000)",
            "v_add_f32_e64 v3, 0, -abs(1.0)",
        ])
        self.assertEqual(got, [0x80000000, 0x80000000, 0x3F800000, 0xBF800000])

    def test_neg_of_inline_int_differs_from_negative_literal(self):
        # LLVM AMDGPUInstPrinter.cpp explicitly notes "-1 is not the same value as neg(1)":
        #   -1      = 0xffffffff -> quiet NaN (class bit 1 = 0x002)
        #   neg(1)  = 0x80000001 -> negative subnormal (class bit 4 = 0x010)
        #   neg(-1) = 0x7fffffff -> positive quiet NaN (class bit 1 = 0x002)
        #   -|0|    = 0x80000000 -> negative zero (class bit 5 = 0x020)
        got = run([
            "v_cmp_class_f32_e64 s4, neg(1), 0x10",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_class_f32_e64 s5, -1, 0x10",
            "v_cndmask_b32_e64 v1, 0, 1, s5",
            "v_cmp_class_f32_e64 s6, neg(-1), 0x2",
            "v_cndmask_b32_e64 v2, 0, 1, s6",
            "v_cmp_class_f32_e64 s7, -|0|, 0x20",
            "v_cndmask_b32_e64 v3, 0, 1, s7",
        ])
        self.assertEqual(got, [1, 0, 1, 1])

    def test_compare_with_modified_constants(self):
        got = run([
            "v_mov_b32 v4, 0xbf800000",          # -1.0f
            "v_cmp_eq_f32_e64 s4, v4, -|1.0|",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_eq_f32_e64 s5, v4, neg(0x3f800000)",
            "v_cndmask_b32_e64 v1, 0, 1, s5",
            "v_cmp_eq_f32_e64 s6, 1.0, |0xbf800000|",
            "v_cndmask_b32_e64 v2, 0, 1, s6",
            "v_cmp_u_f32_e64 s7, neg(1), 0.0",   # neg(1) is a subnormal, NOT NaN
            "v_cndmask_b32_e64 v3, 0, 1, s7",
        ])
        self.assertEqual(got, [1, 1, 1, 0])


class F16AndMixConstModifierTests(unittest.TestCase):
    def test_f16_alu_and_compare_with_modified_constants(self):
        # f16 inline and literal constants: 0x3c00 = +1.0h, 0xbc00 = -1.0h.
        got = run([
            "v_add_f16_e64 v0, 0, -|1.0|",
            "v_add_f16_e64 v1, 0, neg(0x3c00)",
            "v_add_f16_e64 v2, 0, |0xbc00|",
            "v_add_f16_e64 v3, 0, -|0x3c00|",
        ])
        self.assertEqual([w & 0xFFFF for w in got], [0xBC00, 0xBC00, 0x3C00, 0xBC00])

    def test_f16_cvt_compare_and_fma_mix(self):
        got = run([
            "v_cvt_f32_f16_e64 v0, -|0.5|",
            "v_cvt_f32_f16_e64 v1, neg(0x3c00)",
            "v_fma_mix_f32 v2, -|0x3c00|, 2.0, 0.5 op_sel:[0,0,0] op_sel_hi:[1,0,0]",
            "v_cmp_eq_f16_e64 s4, -1.0, -|1.0|",
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        # -0.5f, -1.0f, -1.0h * 2.0f + 0.5f = -1.5f, compare true = 1
        self.assertEqual(got, [0xBF000000, 0xBF800000, 0xBFC00000, 1])

    def test_v_pack_b32_f16_neg_flips_the_packed_half(self):
        # SHADER-260, measured on gfx1036: `neg` on this instruction is a plain sign flip of the half it packs,
        # before any quieting (0x7c01 -> 0xfc01 in IEEE mode 0 and -> 0xfe01 in mode 1), in either half and with
        # op_sel. It used to be refused, together with every other modifier (and before that, neg(0x3c00) silently
        # folded to -0x3c00 = 0xc400, -4.0h, bypassing the modifier check).
        got = run([
            "v_mov_b32 v1, 0x7c01",
            "v_pack_b32_f16 v0, neg(0x3c00), 0x4000",     # -1.0h in the low half, 2.0h in the high one
            "v_pack_b32_f16 v2, neg(v1), v1",            # sign flip on the low half only; no quieting is modelled
        ])
        self.assertEqual(got[:3], [0x4000BC00, 0x00007C01, 0x7C01FC01])

    def test_v_pack_b32_f16_abs_clears_the_sign_of_the_packed_half(self):
        # `abs` is the ISA's sign clear (|S|), applied before `neg`; it has no hardware run of this instruction yet.
        got = run([
            "v_mov_b32 v1, 0xbc00",
            "v_pack_b32_f16 v0, abs(0xbc00), 0x4000",     # +1.0h in the low half, 2.0h in the high one
            "v_pack_b32_f16 v2, -abs(v1), v1",            # clear then flip the low half; src1 stays -1.0h
        ])
        self.assertEqual(got[:3], [0x40003C00, 0x0000BC00, 0xBC00BC00])

class F64ConstModifierStructureTests(unittest.TestCase):
    def test_unary_f64_literals_reach_the_operation_as_high_words(self) -> None:
        fixture = Path(HERE) / "fixtures" / "unary_f64_literals.asm"
        unary = [line for line in fixture.read_text(encoding="ascii").splitlines() if "_e32 " in line]
        for line in unary[:3]:
            with self.subTest(line=line):
                spv = translate([line])
                literal = int(line.rsplit(" ", 1)[1], 16)
                self.assertIn(f"OpCompositeConstruct %v2uint_d %c0 %c{literal:x}", spv)

        # frexp uses integer words alone. Execute the mantissa and exponent paths and read both destination words.
        for spelling, expected in (("0x40180000", [0, 0x3fe80000, 3]), ("2.0", [0, 0x3fe00000, 2]),
                                   ("1", [0, 0x3fe00000, (-1073) & 0xffffffff])):  # 2^-1074 = 0.5 * 2^-1073
            with self.subTest(spelling=spelling):
                lines = [op.replace("0x40180000", spelling) for op in unary[3:]]
                got = run(lines, export="v8, v9, v7, v7")
                self.assertEqual(got[:3], expected)

    def test_f64_literal_and_inline_source_words(self) -> None:
        fixture = Path(HERE) / "fixtures" / "f64_literals.asm"
        instructions = [line for line in fixture.read_text(encoding="utf-8").splitlines()
                        if line and not line.startswith(";")]
        spv = translate(instructions)
        words = {name: int(value) for name, value in
                 re.findall(r"(%\w+) = OpConstant %uint (\d+)", spv)}
        pairs = {name: (words[high] << 32) | words[low] for name, low, high in
                 re.findall(r"(%\w+) = OpCompositeConstruct %v2uint_d (%\w+) (%\w+)", spv)}
        doubles = {name: pairs[pair] for name, pair in
                   re.findall(r"(%\w+) = OpBitcast %double (%\w+)", spv)}
        conversions = re.findall(r"OpFConvert %float (%\w+)", spv)
        additions = re.findall(r"OpFAdd %double (%\w+) (%\w+)", spv)
        self.assertEqual([doubles[source] for source in conversions],
                         [0x3FF8000000000000, 0xFFF0000000000000, 64, 0xFFFFFFFFFFFFFFFF])
        self.assertEqual([(doubles[left], doubles[right]) for left, right in additions],
                         [(0x0000000100000000, 1), (0x3FF0000000000000, 0xBFF8000000000000)])

    def test_f64_emits_fabs_then_fnegate_for_neg_abs_const(self):
        spv = translate([
            "v_cvt_f32_f64 v0, -|1.0|",
            "v_cvt_f32_f64 v1, neg(0)",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        # For -|1.0|: FAbs %double must precede OpFNegate %double feeding OpFConvert %float.
        m_abs = re.search(r"(%\w+) = OpExtInst %double %glsl FAbs (%\w+)", spv)
        self.assertIsNotNone(m_abs, "expected double FAbs for -|1.0|")
        abs_id = m_abs.group(1)
        m_neg = re.search(rf"(%\w+) = OpFNegate %double {re.escape(abs_id)}\b", spv)
        self.assertIsNotNone(m_neg, "expected double OpFNegate after FAbs for -|1.0|")
        # For neg(0): OpFNegate %double on bitcast of (0, 0).
        self.assertGreaterEqual(len(re.findall(r"= OpFNegate %double ", spv)), 2)


if __name__ == "__main__":
    unittest.main()
