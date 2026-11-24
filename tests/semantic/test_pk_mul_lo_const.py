"""v_pk_mul_lo_u16 with a constant operand against the words measured natively on RDNA2 (cs_pk_mul_lo_const.asm).

The low 16 bits of each product are formed as ((x << 16) * y) >> 16: the AMD Windows driver dropped the `& 0xffff` of
(x * y) & 0xffff when the other half of the result was a constant 0. The interpreter cannot show that, the GPU test does;
this checks the values and that no mask-after-multiply is left."""
import os
import re
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import PK_MUL_LO_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 512, 0x31016FAC]   # the output V# in s[0:3]: base 0, 512 bytes, FORMAT 32_FLOAT, DST_SEL xyzw
FILL = 0xEEEEEEEE


class PkMulLoConstTest(unittest.TestCase):
    def test_hardware_words(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_pk_mul_lo_const.asm")
        spv = support.translate_fixture("cs_pk_mul_lo_const", fssm=bytes(compute_header(asm)))
        # Structure: no multiplication result goes straight through a 0xffff mask (the 16-bit wrap is the shift pair)
        self.assertNotRegex(spv, r"(%\S+) = OpIMul %uint \S+ \S+\n\s+%\S+ = OpBitwiseAnd %uint \1 %cffff\b")
        self.assertGreaterEqual(len(re.findall(r"OpIMul %uint", spv)), 8)
        for lane in (0, 1):
            with self.subTest(lane=lane):
                memory = {16 * lane + 4 * k: FILL for k in range(4)}   # the buffer fill the shader reads back
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                    push_constants=OUT_V + [0] * 28, memory=memory)
                self.assertEqual([memory[16 * lane + 4 * k] for k in range(4)], PK_MUL_LO_WORDS[:4])


if __name__ == "__main__":
    unittest.main()
