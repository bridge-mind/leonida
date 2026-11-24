"""v_cvt_pk_u8_f32 against the words measured natively on RDNA2 hardware (cs_cvt_pk_u8.asm)."""
import os
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import CVT_PK_U8_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 32, 0x31016FAC]   # the output V# in s[0:3]: base 0, 32 bytes, FORMAT 32_FLOAT, DST_SEL xyzw


class CvtPkU8Test(unittest.TestCase):
    def test_hardware_words(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_cvt_pk_u8.asm")
        spv = support.translate_fixture("cs_cvt_pk_u8", fssm=bytes(compute_header(asm)))
        for lane, want in enumerate(CVT_PK_U8_WORDS):
            with self.subTest(lane=lane):
                memory = {}
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                    push_constants=OUT_V + [0] * 28, memory=memory)
                self.assertEqual(memory, {4 * lane: want})


if __name__ == "__main__":
    unittest.main()
