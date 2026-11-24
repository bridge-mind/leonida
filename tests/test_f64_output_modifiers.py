"""Check the SPIR-V data flow for output modifiers (omod, then clamp) on v_add_f64.

LLVM defines v_add_f64 as VOP3Inst<"v_add_f64", VOP3_Profile<VOP_F64_F64_F64>>; VOPProfile gives it HasClamp
(f64 is a modifier type) and HasOMod (its destination is floating point). runtime/test/shaders/ps_f64_mods.asm is
the matching GPU test.
"""

import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

from listing import translate_listing

# a = 0.75 in v[0:1], b = 0.5 in v[6:7] (low word first); the sum goes to v[4:5].
BODY = """v_mov_b32 v0, 0
v_mov_b32 v1, 0x3fe80000
v_mov_b32 v6, 0
v_mov_b32 v7, 0x3fe00000
v_add_f64 v[4:5], v[0:1], v[6:7]"""


def words(x):
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits & 0xFFFFFFFF, bits >> 32


class F64OutputModifiersTest(unittest.TestCase):
    def test_add_f64_output_modifiers(self):
        cases = (
            ("", None, False),
            ("mul:2", 2.0, False),
            ("mul:4", 4.0, False),
            ("div:2", 0.5, False),
            ("clamp", None, True),
            ("clamp div:2", 0.5, True),
            ("mul:4 clamp", 4.0, True),
        )
        with tempfile.TemporaryDirectory() as directory:
            for modifiers, scale, clamp in cases:
                with self.subTest(modifiers=modifiers):
                    spv = translate_listing(Path(directory), f"{BODY} {modifiers}".rstrip())
                    self.check_add(spv, scale, clamp)

    def check_add(self, spv, scale, clamp):
        def result(instruction):
            match = re.search(rf"^ *(%\w+) = {instruction}$", spv, re.M)
            self.assertIsNotNone(match, f"Missing SPIR-V instruction: {instruction}")
            return match.group(1)

        def double(x):
            lo, hi = words(x)
            pair = result(rf"OpCompositeConstruct %v2uint_d {result(f'OpConstant %uint {lo}')} "
                          rf"{result(f'OpConstant %uint {hi}')}")
            return result(rf"OpBitcast %double {pair}")

        value = result(r"OpFAdd %double %\w+ %\w+")
        if scale is not None:
            value = result(rf"OpFMul %double {value} {double(scale)}")
        else:
            self.assertNotRegex(spv, r"OpFMul %double")
        if clamp:
            value = result(rf"OpExtInst %double %glsl FClamp {value} {double(0.0)} {double(1.0)}")
        else:
            self.assertNotIn("FClamp", spv)
        pair = result(rf"OpBitcast %v2uint_d {value}")
        for index, register in ((0, "%c4"), (1, "%c5")):
            word = result(rf"OpCompositeExtract %uint {pair} {index}")
            dest = result(rf"OpAccessChain %_ptr_Function_uint %vgpr {register}")
            old_word = result(rf"OpLoad %uint {dest}")
            masked_word = result(rf"OpSelect %uint %\w+ {word} {old_word}")
            self.assertRegex(spv, rf"(?m)^ *OpStore {dest} {masked_word}$")
        # Every f64 add and scale keeps its own rounding (no_contraction decorates all float arithmetic).
        for op in re.findall(r"(?m)^ *(%\w+) = Op(?:FAdd|FMul) %double ", spv):
            self.assertRegex(spv, rf"(?m)^ *OpDecorate {op} NoContraction$")

    def test_gpu_fixture_expectation(self):
        """The words written in ps_f64_mods.asm follow from the ISA rule (omod, then clamp), computed in Python."""
        def mods(x, scale=1.0, clamp=False):
            x *= scale
            return min(max(x, 0.0), 1.0) if clamp else x

        a, b = 0.75, 0.5
        want = tuple(words(v)[1] for v in (mods(a + b, 2.0), mods(a + b, clamp=True),
                                           mods(a + b, 0.5, True), mods(a - b, 4.0)))
        asm = (TEST_DIR.parent / "runtime" / "test" / "shaders" / "ps_f64_mods.asm").read_text(encoding="utf-8")
        line = re.search(r"Expected mrt0 bits: (.*)", asm).group(1)
        self.assertEqual(tuple(int(w, 16) for w in line.split(", ")), want)


if __name__ == "__main__":
    unittest.main()
