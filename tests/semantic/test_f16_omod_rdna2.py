"""Hardware gating test suite for RDNA2 VOP3 output modifiers (omod) and clamp on FP16 destinations.

Verified on AMD Radeon(TM) Graphics RDNA2 iGPU (gfx1036, driver 32.0.21045.29003).
Under Leonida's default shader MODE (denorm16_64=3, denormals kept; IEEE=0):
- f16 destination operations completely ignore omod scaling (mul:2, mul:4, div:2).
- clamp saturation to [0.0, 1.0] remains active and functional.
- f32 destination operations (e.g. v_cvt_f32_f16, v_add_f32) scale with omod under IEEE=0.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_ps(lines):
    spv = translate_lines("ps_f16_omod_rdna2", lines)
    validate_spirv(spv)
    return mrt0_words(spv)


def f32_to_bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


class F16OmodRdna2Tests(unittest.TestCase):
    def test_class1_arithmetic_triples(self):
        # 0.75h = 0x3A00, 1.0h = 0x3C00, 0.0h = 0x0000
        words = run_ps([
            "v_mov_b32 v0, 0xaaaa3800",           # 0.5h
            "v_add_f16 v0, v0, 0.25 mul:2",       # 0.5 + 0.25 = 0.75h (mul:2 ignored)
            "v_mov_b32 v1, 0xaaaa3c00",           # 1.0h
            "v_sub_f16 v1, v1, 0.25 mul:4",       # 1.0 - 0.25 = 0.75h (mul:4 ignored)
            "v_mov_b32 v2, 0xaaaa3e00",           # 1.5h
            "v_mul_f16 v2, v2, 0.5 div:2",        # 1.5 * 0.5 = 0.75h (div:2 ignored)
            "v_mov_b32 v3, 0xaaaa3c00",           # 1.0h
            "v_fma_f16 v3, 1.0, 0.5, 0.25 mul:2", # 1.0 * 0.5 + 0.25 = 0.75h (mul:2 ignored)
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words[0], 0xAAAA3A00)
        self.assertEqual(words[1], 0xAAAA3A00)
        self.assertEqual(words[2], 0xAAAA3A00)
        self.assertEqual(words[3], 0xAAAA3A00)

        # Min/max/med3 triples
        words_triples = run_ps([
            "v_mov_b32 v0, 0xaaaa3a00",           # 0.75h
            "v_min3_f16 v0, v0, 1.0, 2.0 mul:2",  # min(0.75, 1.0, 2.0) = 0.75h (mul:2 ignored)
            "v_mov_b32 v1, 0xaaaa3a00",           # 0.75h
            "v_max3_f16 v1, v1, 0.5, 0.25 div:2", # max(0.75, 0.5, 0.25) = 0.75h (div:2 ignored)
            "v_mov_b32 v2, 0xaaaa3a00",           # 0.75h
            "v_med3_f16 v2, 0.5, v2, 1.0 mul:4",  # med(0.5, 0.75, 1.0) = 0.75h (mul:4 ignored)
            "v_mov_b32 v3, 0xaaaa3c00",           # 1.0h
            "v_add_f16 v3, 1.0, 1.0 mul:2 clamp", # 1.0 + 1.0 = 2.0 -> clamped to 1.0 (0x3C00)
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words_triples[0], 0xAAAA3A00)
        self.assertEqual(words_triples[1], 0xAAAA3A00)
        self.assertEqual(words_triples[2], 0xAAAA3A00)
        self.assertEqual(words_triples[3], 0xAAAA3C00)

    def test_class2_transcendental_math(self):
        # 0.5h = 0x3800, 2.0h = 0x4000, 0.75h = 0x3A00, 1.0h = 0x3C00
        words = run_ps([
            "v_mov_b32 v0, 0xaaaa4000",           # 2.0h
            "v_rcp_f16 v0, v0 mul:2",             # rcp(2.0) = 0.5h (mul:2 ignored)
            "v_mov_b32 v1, 0xaaaa4400",           # 4.0h
            "v_sqrt_f16 v1, v1 mul:2",            # sqrt(4.0) = 2.0h (mul:2 ignored)
            "v_mov_b32 v2, 0xaaaa3f00",           # 1.75h
            "v_fract_f16 v2, v2 mul:4",           # fract(1.75) = 0.75h (mul:4 ignored)
            "v_mov_b32 v3, 0xaaaa4400",           # 4.0h
            "v_sqrt_f16 v3, v3 mul:2 clamp",      # sqrt(4.0) = 2.0 -> clamp -> 1.0h (0x3C00)
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words[0], 0xAAAA3800)
        self.assertEqual(words[1], 0xAAAA4000)
        self.assertEqual(words[2], 0xAAAA3A00)
        self.assertEqual(words[3], 0xAAAA3C00)

    def test_class3_conversions(self):
        # v_cvt_f16_u16 and v_cvt_f16_i16 use clamp_f16 (omod ignored):
        # 3 u16 -> 3.0h (0x4200); 3 i16 -> 3.0h (0x4200); 1 u16 clamp -> 1.0h (0x3C00).
        words = run_ps([
            "v_mov_b32 v0, 3",                    # 3 u16
            "v_cvt_f16_u16 v0, v0 mul:2",         # cvt(3) = 3.0h (mul:2 ignored)
            "v_mov_b32 v1, 3",                    # 3 i16
            "v_cvt_f16_i16 v1, v1 div:2",         # cvt(3) = 3.0h (div:2 ignored)
            "v_mov_b32 v2, 5",                    # 5 u16
            "v_cvt_f16_u16 v2, v2 mul:4 clamp",   # cvt(5) = 5.0h -> clamp -> 1.0h (0x3C00)
            "v_mov_b32 v3, 0xffff",               # -1 i16
            "v_cvt_f16_i16 v3, v3 mul:2 clamp",   # cvt(-1) = -1.0h -> clamp -> 0.0h (0x0000)
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words[0] & 0xFFFF, 0x4200)
        self.assertEqual(words[1] & 0xFFFF, 0x4200)
        self.assertEqual(words[2] & 0xFFFF, 0x3C00)
        self.assertEqual(words[3] & 0xFFFF, 0x0000)

    def test_class4_control_f32_dest_scales(self):
        # v_cvt_f32_f16 produces f32 destination; under IEEE=0, omod scales:
        # cvt(2.0h) = 2.0f -> mul:2 = 4.0f, mul:4 = 8.0f, div:2 = 1.0f
        words = run_ps([
            "v_mov_b32 v0, 0x4000",               # 2.0h
            "v_cvt_f32_f16 v0, v0 mul:2",         # 2.0 * 2 = 4.0f32
            "v_mov_b32 v1, 0x4000",               # 2.0h
            "v_cvt_f32_f16 v1, v1 mul:4",         # 2.0 * 4 = 8.0f32
            "v_mov_b32 v2, 0x4000",               # 2.0h
            "v_cvt_f32_f16 v2, v2 div:2",         # 2.0 / 2 = 1.0f32
            "v_mov_b32 v3, 0x4000",               # 2.0h
            "v_cvt_f32_f16 v3, v3 clamp",         # 2.0 -> clamp -> 1.0f32
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(bits_to_f32(words[0]), 4.0)
        self.assertEqual(bits_to_f32(words[1]), 8.0)
        self.assertEqual(bits_to_f32(words[2]), 1.0)
        self.assertEqual(bits_to_f32(words[3]), 1.0)


if __name__ == "__main__":
    unittest.main()
