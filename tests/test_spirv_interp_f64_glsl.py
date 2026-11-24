"""Regression tests for 64-bit %double and %v2double GLSL.std.450 execution in tests/spirv_interp.py."""

import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

from listing import translate_listing  # noqa: E402
import spirv_interp  # noqa: E402


def f64_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def f64_words(x):
    bits = f64_bits(x)
    return bits & 0xFFFFFFFF, bits >> 32


def f32_round(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def run_f64_glsl(consts, body_lines, out_type="%double"):
    const_lines = []
    for name, ty, val in consts:
        const_lines.append(f"{name} = OpConstant {ty} {val}")
    spv = "\n".join(
        [
            "%uint = OpTypeInt 32 0",
            "%float = OpTypeFloat 32",
            "%double = OpTypeFloat 64",
            "%v2double = OpTypeVector %double 2",
            "%_c0 = OpConstant %uint 0",
            "%_c1 = OpConstant %uint 1",
            "%_arr_u1 = OpTypeArray %uint %_c1",
            "%_ptr_fn_arr = OpTypePointer Function %_arr_u1",
            "%_ptr_out = OpTypePointer Output " + out_type,
            "%_null = OpConstantNull %_arr_u1",
            *const_lines,
            "%vgpr = OpVariable %_ptr_fn_arr Function %_null",
            "%sgpr = OpVariable %_ptr_fn_arr Function %_null",
            "%out_res = OpVariable %_ptr_out Output",
            "%main = OpFunction %uint None %uint",
            "%entry = OpLabel",
            *body_lines,
            "OpStore %out_res %res",
            "OpReturn",
            "OpFunctionEnd",
        ]
    )
    return spirv_interp.run(spv).outputs["out_res"]


class SpirvInterpF64GlslTests(unittest.TestCase):
    def test_scalar_f64_glsl_preserves_53_bit_precision(self):
        tiny = 2.0**-50
        x_neg = -(1.0 + tiny)
        x_pos = 1.0 + tiny
        x_half = 0.5 + tiny
        x_sq = (1.0 + tiny) * (1.0 + tiny)
        self.assertNotEqual(f64_bits(x_pos), f64_bits(f32_round(x_pos)))

        cases = [
            (
                "FAbs",
                [("%a", "%double", repr(x_neg))],
                "%res = OpExtInst %double %glsl FAbs %a",
                x_pos,
            ),
            (
                "FClamp",
                [("%a", "%double", repr(x_half)), ("%lo", "%double", "0.0"), ("%hi", "%double", "1.0")],
                "%res = OpExtInst %double %glsl FClamp %a %lo %hi",
                x_half,
            ),
            (
                "FMin",
                [("%a", "%double", repr(x_pos)), ("%b", "%double", repr(1.0 + 2.0 * tiny))],
                "%res = OpExtInst %double %glsl FMin %a %b",
                x_pos,
            ),
            (
                "FMax",
                [("%a", "%double", repr(1.0)), ("%b", "%double", repr(x_pos))],
                "%res = OpExtInst %double %glsl FMax %a %b",
                x_pos,
            ),
            (
                "Sqrt",
                [("%a", "%double", repr(x_sq))],
                "%res = OpExtInst %double %glsl Sqrt %a",
                math.sqrt(x_sq),
            ),
            (
                "InverseSqrt",
                [("%a", "%double", repr(x_sq))],
                "%res = OpExtInst %double %glsl InverseSqrt %a",
                1.0 / math.sqrt(x_sq),
            ),
            (
                "Fract",
                [("%a", "%double", repr(x_pos))],
                "%res = OpExtInst %double %glsl Fract %a",
                x_pos - math.floor(x_pos),
            ),
            (
                "Exp2",
                [("%a", "%double", repr(x_half))],
                "%res = OpExtInst %double %glsl Exp2 %a",
                2.0**x_half,
            ),
            (
                "Log2",
                [("%a", "%double", repr(x_pos))],
                "%res = OpExtInst %double %glsl Log2 %a",
                math.log2(x_pos),
            ),
        ]
        for name, consts, inst, expected in cases:
            with self.subTest(op=name):
                got = run_f64_glsl(consts, [inst])
                self.assertEqual(f64_bits(got), f64_bits(expected))

    def test_v2double_glsl_preserves_53_bit_precision(self):
        tiny = 2.0**-48
        a0, a1 = -(1.0 + tiny), 0.25 + tiny
        b0, b1 = 1.0 + 2.0 * tiny, 0.25 + 3.0 * tiny
        lo0, lo1 = 0.0, 0.0
        hi0, hi1 = 2.0, 1.0

        got_abs = run_f64_glsl(
            [("%a0", "%double", repr(a0)), ("%a1", "%double", repr(a1))],
            [
                "%va = OpCompositeConstruct %v2double %a0 %a1",
                "%res = OpExtInst %v2double %glsl FAbs %va",
            ],
            out_type="%v2double",
        )
        self.assertEqual(tuple(f64_bits(v) for v in got_abs), (f64_bits(abs(a0)), f64_bits(abs(a1))))

        got_min = run_f64_glsl(
            [
                ("%a0", "%double", repr(1.0 + tiny)),
                ("%a1", "%double", repr(a1)),
                ("%b0", "%double", repr(b0)),
                ("%b1", "%double", repr(b1)),
            ],
            [
                "%va = OpCompositeConstruct %v2double %a0 %a1",
                "%vb = OpCompositeConstruct %v2double %b0 %b1",
                "%res = OpExtInst %v2double %glsl FMin %va %vb",
            ],
            out_type="%v2double",
        )
        self.assertEqual(tuple(f64_bits(v) for v in got_min), (f64_bits(1.0 + tiny), f64_bits(a1)))

        got_max = run_f64_glsl(
            [
                ("%a0", "%double", repr(1.0 + tiny)),
                ("%a1", "%double", repr(a1)),
                ("%b0", "%double", repr(b0)),
                ("%b1", "%double", repr(b1)),
            ],
            [
                "%va = OpCompositeConstruct %v2double %a0 %a1",
                "%vb = OpCompositeConstruct %v2double %b0 %b1",
                "%res = OpExtInst %v2double %glsl FMax %va %vb",
            ],
            out_type="%v2double",
        )
        self.assertEqual(tuple(f64_bits(v) for v in got_max), (f64_bits(b0), f64_bits(b1)))

        got_clamp = run_f64_glsl(
            [
                ("%a0", "%double", repr(1.0 + tiny)),
                ("%a1", "%double", repr(a1)),
                ("%lo0", "%double", repr(lo0)),
                ("%lo1", "%double", repr(lo1)),
                ("%hi0", "%double", repr(hi0)),
                ("%hi1", "%double", repr(hi1)),
            ],
            [
                "%va = OpCompositeConstruct %v2double %a0 %a1",
                "%vlo = OpCompositeConstruct %v2double %lo0 %lo1",
                "%vhi = OpCompositeConstruct %v2double %hi0 %hi1",
                "%res = OpExtInst %v2double %glsl FClamp %va %vlo %vhi",
            ],
            out_type="%v2double",
        )
        self.assertEqual(tuple(f64_bits(v) for v in got_clamp), (f64_bits(1.0 + tiny), f64_bits(a1)))

    def test_f32_glsl_still_rounds_to_binary32(self):
        x = -(1.0 + 2.0**-25)
        got = run_f64_glsl(
            [("%a", "%float", repr(x))],
            ["%res = OpExtInst %float %glsl FAbs %a"],
            out_type="%float",
        )
        self.assertEqual(got, f32_round(abs(x)))
        self.assertEqual(got, 1.0)

    def test_end_to_end_v_add_f64_abs_and_clamp(self):
        a = -(0.5 + 2.0**-45)
        b = -(0.25 + 2.0**-46)
        alo, ahi = f64_words(a)
        blo, bhi = f64_words(b)
        expected = min(max(abs(a) + abs(b), 0.0), 1.0)
        elo, ehi = f64_words(expected)

        listing = "\n".join(
            [
                f"v_mov_b32 v0, {alo:#x}",
                f"v_mov_b32 v1, {ahi:#x}",
                f"v_mov_b32 v2, {blo:#x}",
                f"v_mov_b32 v3, {bhi:#x}",
                "v_add_f64 v[4:5], |v[0:1]|, |v[2:3]| clamp",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            spv = translate_listing(Path(directory), listing)
        res = spirv_interp.run(spv)
        self.assertEqual((res.vgpr[4], res.vgpr[5]), (elo, ehi))


if __name__ == "__main__":
    unittest.main()
