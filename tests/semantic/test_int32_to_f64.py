"""Exact binary64 conversion of 32-bit signed and unsigned register values."""
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import translate_fixture, translate_lines
import rdna2spirv
import spirv_interface


class Int32ToFloat64(unittest.TestCase):
    @unittest.skipUnless(all(shutil.which(t) for t in ("spirv-as", "spirv-opt", "spirv-dis")), "needs SPIRV-Tools")
    def test_constant_folded_mrt_words(self):
        # Integer-to-double is exact. Float controls can prevent optimizers folding even these constants.
        default = translate_fixture("ps_int32_to_f64")
        with patch.object(rdna2spirv, "FLOAT_CONTROLS", False):
            foldable = translate_fixture("ps_int32_to_f64")
        float_controls = {
            "OpCapability SignedZeroInfNanPreserve", "OpCapability DenormPreserve",
            "OpExecutionMode %main SignedZeroInfNanPreserve 32", "OpExecutionMode %main DenormPreserve 16",
            # SHADER-597 (#3290): a module that declares Float64 also preserves f64 signed zeros, infs and NaNs
            "OpExecutionMode %main SignedZeroInfNanPreserve 64",
        }
        self.assertTrue(float_controls.issubset(line.strip() for line in default.splitlines()))
        self.assertEqual([line for line in default.splitlines() if line.strip() not in float_controls],
                         foldable.splitlines())
        with tempfile.TemporaryDirectory() as tmp:
            binary, folded = Path(tmp) / "int32.spv", Path(tmp) / "folded.spv"
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", binary, "-"],
                           input=foldable, text=True, check=True)
            subprocess.run(["spirv-opt", "-O", binary, "-o", folded], check=True)
            listing = subprocess.check_output(["spirv-dis", "--raw-id", folded], text=True)
            stored, = re.findall(r"OpStore %\d+ (%\d+)", listing)
            components, = re.findall(rf"{stored} = OpConstantComposite %\d+ ((?:%\d+ ){{3}}%\d+)", listing)
            # Read constant payloads directly to preserve every bit of the exported float channels.
            words = spirv_interface.parse_words(folded.read_bytes())
            constants = {args[1]: args[2] for op, args, _ in spirv_interface.instructions(words)
                         if op == spirv_interface.OP_CONSTANT}
            actual = tuple(constants[int(c[1:])] for c in components.split())
        expected = struct.unpack("<4I", struct.pack("<2d", -2147483647, 2147483649))
        self.assertEqual(actual, expected)

    def test_signed_and_unsigned_conversion_reach_both_destination_words(self):
        text = translate_fixture("ps_int32_to_f64")
        for opcode, first in (("OpConvertSToF", 4), ("OpConvertUToF", 6)):
            with self.subTest(opcode=opcode):
                value, = re.findall(rf"(%\w+) = {opcode} %double ", text)
                pair, = re.findall(rf"(%\w+) = OpBitcast %v2uint_d {value}\b", text)
                for i in range(2):
                    word, = re.findall(rf"(%\w+) = OpCompositeExtract %uint {pair} {i}\b", text)
                    sel = re.findall(rf"(%\w+) = OpSelect %uint %\w+ {word} %\w+", text)
                    selected, = sel or [word]
                    pointer, = re.findall(rf"OpStore (%\w+) {selected}\b", text)
                    self.assertIn(f"{pointer} = OpAccessChain %_ptr_Function_uint %vgpr %c{first + i}\n", text)

    def test_vop3_clamps_the_double_and_ignores_the_scaling(self):
        # Measured on gfx1036 through tools/hw_oracle (wave32 and wave64 identical on every case,
        # tools/f64_omod_probe.py): a binary64 destination ignores mul:2/mul:4/div:2 while
        # denorm_mode_16_64 = 3, the MODE the translator assumes, and clamp still saturates.
        # See docs/design.md, "32-bit integers to binary64".
        for suffix, opcode in (("i32", "OpConvertSToF"), ("u32", "OpConvertUToF")):
            with self.subTest(suffix=suffix):
                text = translate_lines("ps_int32_to_f64_mods", [
                    "v_mov_b32 v4, 3",
                    f"v_cvt_f64_{suffix}_e64 v[4:5], v4 clamp div:2",
                    "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
                ])
                value, = re.findall(rf"(%\w+) = {opcode} %double ", text)
                self.assertNotIn("OpFMul %double", text)
                clamped, = re.findall(rf"(%\w+) = OpExtInst %double %glsl FClamp {value} ", text)
                self.assertIn(f"OpBitcast %v2uint_d {clamped}\n", text)
                omod_only = translate_lines("ps_int32_to_f64_omod", [
                    "v_mov_b32 v4, 3",
                    f"v_cvt_f64_{suffix}_e64 v[4:5], v4 mul:2",
                    "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
                ])
                self.assertNotRegex(omod_only, r"OpFMul %double")
                self.assertNotIn("FClamp", omod_only)
