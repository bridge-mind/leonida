"""v_dot4acc_u32_u8 is not an RDNA2 opcode the translator implements: it must stay refused, not guessed.

This replaces the one translator-level check of the deleted test_alu_dot4acc_u8.py. test_valu_dot4_int.py (#3472) pins
the same refusal once it lands; until then this keeps it. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402


class Dot4AccRefused(unittest.TestCase):
    def test_refused_in_each_operand_form(self):
        for src in ("v1, v2, v3", "s1, s2, s3", "v1, v2, 0"):
            with self.subTest(src=src):
                lines = ["v_mov_b32 v1, 1", "v_mov_b32 v2, 1", "v_mov_b32 v3, 0", "s_mov_b32 s1, 1", "s_mov_b32 s2, 1",
                         "s_mov_b32 s3, 0", f"v_dot4acc_u32_u8 v8, {src}", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
                with self.assertRaises(Unsupported) as cm:
                    support.translate_lines("ps_dot4acc_refused", lines)
                self.assertIn("v_dot4acc_u32_u8", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
