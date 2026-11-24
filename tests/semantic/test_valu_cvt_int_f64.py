"""v_cvt_i32_f64 and v_cvt_u32_f64 are not implemented in tools/rdna2spirv.py. v_cvt_f64_i32 and v_cvt_f64_u32 are
since #2521, and tests/semantic/test_int32_to_f64.py runs them through the translator.

The previous version of this file tested a private Python model of the four conversions and never called the
translator, so it passed whatever the translator did.  Until the ops are lowered, the honest test is that the
translator refuses each of them by name (instead of emitting wrong code); when one gets implemented, this test fails
and is replaced by a run through the translator against the ISA semantics (exact int->f64, truncating f64->int with
saturation, NaN -> 0).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402

CASES = [
    ("v_cvt_i32_f64", "v_cvt_i32_f64_e32 v2, v[0:1]"),
    ("v_cvt_u32_f64", "v_cvt_u32_f64_e32 v2, v[0:1]"),
    ("v_cvt_i32_f64", "v_cvt_i32_f64_e64 v2, -|v[0:1]|"),
]


class CvtIntF64Unsupported(unittest.TestCase):
    def test_each_op_is_refused_by_name(self):
        for name, ins in CASES:
            with self.subTest(ins=ins):
                lines = ["v_mov_b32 v1, 5", ins, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_cvt_int_f64", lines)
                self.assertEqual(str(refused.exception), name)

    def test_a_listing_without_them_still_translates(self):
        lines = ["v_mov_b32 v1, 5", "v_cvt_f32_i32_e32 v2, v1", "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
        self.assertIn("OpFunction", support.translate_lines("ps_cvt_i32_ok", lines))


if __name__ == "__main__":
    unittest.main()
