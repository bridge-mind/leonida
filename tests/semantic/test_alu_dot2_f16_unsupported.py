"""`v_dot2_f16_f16` and `v_dot2acc_f16_f16` are refused, not translated.

Scope, measured with llvm-mc (AOMP 16.0-45 / LLVM 17.0.0git, `-arch=amdgcn`, four-operand
VOP3P form):

  | instruction         | gfx1100            | gfx1030                   | gfx1013                   |
  |---------------------|--------------------|---------------------------|---------------------------|
  | `v_dot2_f16_f16`    | encodes            | not supported on this GPU | not supported on this GPU |
  | `v_dot2acc_f16_f16` | invalid instruction| invalid instruction       | invalid instruction       |
  | `v_dot2c_f16_f16`   | invalid instruction| invalid instruction       | invalid instruction       |

The packed half-precision dot product therefore exists on GFX11 only: it is not part of the
GFX10/RDNA2 VALU this project translates, and the accumulating spellings do not assemble at
all. The translator has to refuse them rather than invent a lowering, and that refusal is
what this file pins. `docs/design.md` carries no row for them; the packed-integer dot
products (`v_dot2_i32_i16`, `v_dot2c_i32_i16`, ...) are different instructions with their own
rows.

This file replaces `tests/semantic/test_alu_dot2_f16.py`, which built its own SPIR-V with its
own reference oracle and executed that in `spirv_mini_interp`: no line of it went through
`rdna2spirv`, so nothing in it could fail when the translator changed (TEST-603), and its
docstring attributed the instruction to the RDNA2/GFX10 ISA. The sibling
`tests/semantic/test_alu_dot2acc_f16.py` is the same class and is removed by #3098.

Standard library only, no game data, no GPU.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

for path in (ROOT, TOOLS_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

DOT2 = [
    "v_mov_b32 v1, 0x3c003c00",
    "v_mov_b32 v2, 0x40004000",
    "v_dot2_f16_f16 v0, v1, v2",
    "exp mrt0 v0, v0, v0, v0 done vm",
    "s_endpgm",
]

DOT2ACC = [
    "v_mov_b32 v1, 0x3c003c00",
    "v_mov_b32 v2, 0x40004000",
    "v_mov_b32 v3, 0",
    "v_dot2acc_f16_f16 v0, v1, v2, v3",
    "exp mrt0 v0, v0, v0, v0 done vm",
    "s_endpgm",
]


class Dot2Unsupported(unittest.TestCase):
    def assert_refused(self, name, lines, instruction):
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines(name, lines)
        self.assertIn(instruction, str(caught.exception))

    def test_the_packed_half_dot_product_is_refused(self):
        self.assert_refused("ps_dot2_unsupported", DOT2, "v_dot2_f16_f16")

    def test_the_accumulating_spelling_is_refused(self):
        self.assert_refused("ps_dot2acc_unsupported", DOT2ACC, "v_dot2acc_f16_f16")


if __name__ == "__main__":
    unittest.main()
