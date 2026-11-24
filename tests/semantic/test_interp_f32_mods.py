"""VOP3 output modifiers (clamp, mul:2, mul:4, div:2) on v_interp_p2_f32_e64 and v_interp_mov_f32_e64.

LLVM llvm/test/MC/AMDGPU/gfx10_asm_vop3.s encodes clamp, mul:2, mul:4 and div:2 on the VOP3 forms of both
v_interp_p2_f32 and v_interp_mov_f32, with omod applied before clamp as for every other VOP3 float op.
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from test_value_semantics import PS_FLAT_DXBC   # noqa: E402 — attr0.xy and attr1.x, f32


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def translate(asm, name="ps_interp_f32_mods"):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join("\t" + ln + "\n" for ln in asm.splitlines() if ln.strip()))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(PS_FLAT_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            result = support.rdna2spirv.translate(s_path)
            gc.collect()
    return result


class InterpF32OutputModifierTests(unittest.TestCase):
    def test_interp_p2_f32_clamp_and_omod(self):
        # attr0 = (1.5, -0.25, 0.375, 0.625):
        #   v2: clamp(1.5) = 1.0
        #   v3: clamp(-0.25) = 0.0
        #   v4: 0.375 * 2 = 0.75 (mul:2)
        #   v5: clamp(0.625 * 2) = 1.0 (mul:2 clamp -> omod before clamp)
        shader = """
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32_e64 v2, v1, attr0.x clamp
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32_e64 v3, v1, attr0.y clamp
v_interp_p1_f32 v4, v0, attr0.z
v_interp_p2_f32_e64 v4, v1, attr0.z mul:2
v_interp_p1_f32 v5, v0, attr0.w
v_interp_p2_f32_e64 v5, v1, attr0.w mul:2 clamp
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""
        words = support.mrt0_words(
            translate(shader),
            inputs={"%in_attr0": (1.5, -0.25, 0.375, 0.625), "%in_attr1": (0.0,) * 4},
        )
        self.assertEqual(words, [fbits(1.0), fbits(0.0), fbits(0.75), fbits(1.0)])

    def test_interp_p2_mul4_div2_and_mov_f32(self):
        # attr0 = (0.25, 3.0, 0.0, 0.0), attr1 = (1.5, -0.5, 0.25, 4.0) (attr1 read only via mov -> Flat)
        shader = """
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32_e64 v2, v1, attr0.x mul:4
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32_e64 v3, v1, attr0.y div:2
v_interp_mov_f32_e64 v4, p0, attr1.x clamp
v_interp_mov_f32_e64 v5, p0, attr1.x div:2 clamp
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""
        words = support.mrt0_words(
            translate(shader, name="ps_interp_mov_f32_mods"),
            inputs={"%in_attr0": (0.25, 3.0, 0.0, 0.0), "%in_attr1": (1.5, -0.5, 0.25, 4.0)},
        )
        self.assertEqual(words, [fbits(1.0), fbits(1.5), fbits(1.0), fbits(0.75)])

    def test_plain_interp_without_mods_unchanged(self):
        shader = """
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_interp_mov_f32 v4, p0, attr1.x
v_mov_b32 v5, 0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""
        words = support.mrt0_words(
            translate(shader, name="ps_interp_plain"),
            inputs={"%in_attr0": (1.5, -0.25, 0.0, 0.0), "%in_attr1": (2.5, 0.0, 0.0, 0.0)},
        )
        self.assertEqual(words, [fbits(1.5), fbits(-0.25), fbits(2.5), 0])


if __name__ == "__main__":
    unittest.main()
