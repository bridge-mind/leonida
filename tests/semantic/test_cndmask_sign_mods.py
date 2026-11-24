"""v_cndmask_b32_e64 source modifiers are sign-bit moves, not float ops.

v_cndmask_b32 only moves bits, so a VOP3 `abs` on a source clears bit 31 and `neg` flips it (abs first), exactly, for
every bit pattern: a signalling NaN stays signalling and a denormal isn't flushed. The translator used to read the
sources through `Gen.modify` (SPIR-V `FAbs`/`OpFNegate`), which may quiet a signalling NaN; the mini interpreter does,
so `-v1` of 0x7F800001 came out 0xFFC00001. The same input modifiers on `v_cmp_class_f32` are measured as sign-bit moves
on gfx1036 (docs/design.md, "Floating-point class compare").

Hand-written assembly only (no game files): each case loads a word into v1, selects it with a modifier through
v_cndmask_b32_e64, and exports it. Runs on the CPU through tests/semantic/spirv_mini_interp.py; with spirv-val on PATH
the module is validated too.

    python -m unittest discover -s tests/semantic -p test_cndmask_sign_mods.py
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for path in (TOOLS_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Signalling and quiet NaNs of both signs, denormals, zeros, infinities and normal numbers.
WORDS = (0x7F800001, 0xFF800001, 0x7FBFFFFF, 0x7FC00000, 0xFFC00001, 0x00000001, 0x80000001, 0x007FFFFF,
         0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000, 0x12345678)
MODIFIERS = {
    "v1": lambda w: w,
    "-v1": lambda w: w ^ 0x80000000,
    "|v1|": lambda w: w & 0x7FFFFFFF,
    "-|v1|": lambda w: (w & 0x7FFFFFFF) ^ 0x80000000,
}
OTHER = 0x0BADF00D  # the operand that isn't selected


class CndmaskSignBitModifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(cls.tmp.name, "disasm"))
        cls.src = os.path.join(cls.tmp.name, "disasm", "ps_cndmask_mods.s")
        with open(os.path.join(cls.tmp.name, "ps_cndmask_mods.dxbc"), "wb") as f:
            f.write(STUB_DXBC)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def translate(self, word, operand, cond, modified_is_src1):
        """The SPIR-V text of a pixel shader exporting v_cndmask_b32_e64 of `operand` (v1 = word) and v2 = OTHER."""
        srcs = (f"v2, {operand}" if modified_is_src1 else f"{operand}, v2")
        lines = [f"v_mov_b32 v1, 0x{word:08x}", f"v_mov_b32 v2, 0x{OTHER:08x}", f"s_mov_b32 s4, {cond}",
                 f"v_cndmask_b32_e64 v0, {srcs}, s4", "v_mov_b32 v3, 0"]
        asm = ("; ps_cndmask_mods\n" + "".join(f"\t{line}\n" for line in lines)
               + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n")
        with open(self.src, "w", encoding="utf-8", newline="\n") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(self.src)

    def test_modifiers_move_only_the_sign_bit(self):
        for operand, rule in MODIFIERS.items():
            for word in WORDS:
                # The modified operand as src0 (selected when the condition is 0) and as src1 (selected when 1).
                for modified_is_src1 in (False, True):
                    with self.subTest(operand=operand, word=f"{word:08x}", src1=modified_is_src1):
                        spv = self.translate(word, operand, int(modified_is_src1), modified_is_src1)
                        got = si.run_fragment(spv)["%out_mrt0"][0]
                        self.assertEqual(got, rule(word), f"got 0x{got:08x}, want 0x{rule(word):08x}")

    def test_unselected_modified_operand_does_not_leak(self):
        for operand in MODIFIERS:
            with self.subTest(operand=operand):
                spv = self.translate(0x7F800001, operand, 1, False)  # condition 1 selects v2, the plain operand
                self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], OTHER)

    def test_no_float_ops_are_emitted(self):
        for operand in MODIFIERS:
            with self.subTest(operand=operand):
                spv = self.translate(0x7F800001, operand, 0, False)
                self.assertNotIn("OpFNegate", spv)
                self.assertNotIn("FAbs", spv)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "spirv-as/spirv-val not on PATH")
    def test_modules_validate(self):
        for operand in MODIFIERS:
            with self.subTest(operand=operand):
                spv = self.translate(0x7F800001, operand, 0, False)
                asm = os.path.join(self.tmp.name, "m.spvasm")
                binary = os.path.join(self.tmp.name, "m.spv")
                with open(asm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                assembled = subprocess.run(["spirv-as", "--target-env", "vulkan1.3", asm, "-o", binary],
                                           capture_output=True, text=True)
                self.assertEqual(assembled.returncode, 0, assembled.stderr)
                validated = subprocess.run(["spirv-val", "--target-env", "vulkan1.3", binary],
                                           capture_output=True, text=True)
                self.assertEqual(validated.returncode, 0, validated.stderr)


if __name__ == "__main__":
    unittest.main()
