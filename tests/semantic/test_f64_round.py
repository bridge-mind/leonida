"""Value regressions for RDNA2's binary64 round-to-integral instructions."""
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import support


def words(value):
    return struct.unpack("<II", struct.pack("<d", value))


def fixture():
    path = Path(__file__).with_name("fixtures") / "ps_f64_round.asm"
    return support.translate_lines(path.stem, path.read_text(encoding="ascii").splitlines())


def run_op(op, value, *, inactive=False):
    low, high = words(value)
    return support.mrt0_words(support.translate_lines("ps_f64_case", [
        f"v_mov_b32 v0, {low:#x}", f"v_mov_b32 v1, {high:#x}",
        "v_mov_b32 v4, 0x11223344", "v_mov_b32 v5, 0x55667788",
        *(["s_mov_b32 exec_lo, 0"] if inactive else []),
        f"{op} v[4:5], v[0:1]",
        *(["s_mov_b32 exec_lo, -1"] if inactive else []),
        "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
    ]))[:2]


class F64RoundTests(unittest.TestCase):
    def test_fixture(self):
        text = fixture()
        self.assertIn("OpCapability Float64", text)
        result = support.run_fragment(text)
        self.assertEqual(tuple(result["%out_mrt0"]), (0, 0xBFF00000, 0, 0xC0000000))
        self.assertEqual(tuple(result["%out_mrt1"]), (0, 0xC0000000, 0, 0xC0000000))

    def test_binary64_precision_and_ties(self):
        cases = (
            ("v_trunc_f64", -0.75, -0.0),
            ("v_ceil_f64", -0.25, -0.0),
            ("v_floor_f64", math.nextafter(1.0, 0.0), 0.0),
            ("v_ceil_f64", math.nextafter(1.0, 2.0), 2.0),
            ("v_trunc_f64", 2**40 + 1.75, float(2**40 + 1)),
            ("v_rndne_f64", 2.5, 2.0),
            ("v_rndne_f64", 3.5, 4.0),
            ("v_rndne_f64", -3.5, -4.0),
            ("v_rndne_f64", -0.5, -0.0),
        )
        for op, value, want in cases:
            with self.subTest(op=op, value=value):
                self.assertEqual(tuple(run_op(op, value)), words(want))

    def test_vop3_modifiers(self):
        low, high = words(-2.75)
        text = support.translate_lines("ps_f64_mod", [
            f"v_mov_b32 v0, {low:#x}", f"v_mov_b32 v1, {high:#x}",
            "v_trunc_f64_e64 v[2:3], |v[0:1]| clamp div:2",
            "v_floor_f64_e64 v[4:5], -v[0:1] mul:2",
            "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm",
        ])
        self.assertEqual(tuple(support.mrt0_words(text)), words(1.0) + words(4.0))

    def test_rejects_bad_operand_forms(self):
        for instruction in ("v_floor_f64 v4, v[0:1]", "v_round_f64 v[4:5], v[0:1]",
                            "v_trunc_f64 v[4:5], v0", "v_ceil_f64 v[4:5], 0x100000000"):
            with self.subTest(instruction=instruction):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    support.translate_lines("ps_f64_bad", [instruction, "s_endpgm"])

    def test_inline_float_and_exec(self):
        text = support.translate_lines("ps_f64_inline", [
            "v_ceil_f64_e32 v[0:1], 0.5",
            "v_floor_f64_e32 v[2:3], -0.5",
            "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm",
        ])
        self.assertEqual(tuple(support.mrt0_words(text)), words(1.0) + words(-1.0))
        self.assertEqual(run_op("v_rndne_f64", 3.5, inactive=True), [0x11223344, 0x55667788])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not installed")
    def test_spirv_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            asm = Path(directory) / "shader.spvasm"
            binary = asm.with_suffix(".spv")
            asm.write_text(fixture(), encoding="ascii")
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", str(asm), "-o", str(binary)], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(binary)], check=True)
