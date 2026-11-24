"""Check SPIR-V emission and computed values for v_rcp_iflag_f32 source and output modifiers."""

import re
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

import spirv_interp as si
from listing import translate_listing


def fbits(x):
    return si.f2u(si.f32(x))


def parse_defs(spv):
    return dict(re.findall(r"^\s*(%\w+) = (Op[^\n]+)", spv, re.M))


class RcpIflagModsTest(unittest.TestCase):
    def translate_rcp(self, tmp_dir, insn, v1_val):
        body = f"v_mov_b32 v1, {fbits(v1_val):#x}\n{insn}\nv_mov_b32 v3, v0"
        return translate_listing(tmp_dir, body)

    def assert_rcp_pipeline(self, spv, *, neg=False, abs_=False, scale_bits=None, clamp=False):
        defs = parse_defs(spv)
        fdiv_matches = [
            (res, expr)
            for res, expr in defs.items()
            if expr.startswith("OpFDiv %float %f1 ")
        ]
        self.assertEqual(len(fdiv_matches), 1, f"Expected one OpFDiv %float %f1, got {fdiv_matches}")
        value, fdiv_expr = fdiv_matches[0]
        src = fdiv_expr.split()[-1]

        # Verify source modifier chain feeding OpFDiv.
        if neg or abs_:
            # The modified float, as the id itself: a bitcast to uint and back is the original id (Gen.tof/tou).
            self.assertNotRegex(defs[src], r"^OpBitcast ")
            if neg:
                self.assertRegex(defs[src], r"^OpFNegate %float %\w+$")
                src = defs[src].split()[-1]
            else:
                self.assertNotIn("OpFNegate", spv)
            if abs_:
                self.assertRegex(defs[src], r"^OpExtInst %float %glsl FAbs %\w+$")
                src = defs[src].split()[-1]
            else:
                self.assertNotIn("FAbs", spv)
        else:
            self.assertNotIn("OpFNegate", spv)
            self.assertNotIn("FAbs", spv)
        self.assertRegex(defs[src], r"^OpBitcast %float %\w+$")
        loaded_u = defs[src].split()[-1]
        self.assertRegex(defs[loaded_u], r"^OpLoad %uint %\w+$")
        ptr = defs[loaded_u].split()[-1]
        self.assertRegex(defs[ptr], r"^OpAccessChain %_ptr_Function_uint %vgpr %c1$")

        # Verify output modifiers (omod before clamp) feeding v0 store.
        def match_result(pattern):
            m = re.search(rf"^ *(%\w+) = {pattern}$", spv, re.M)
            self.assertIsNotNone(m, f"Missing SPIR-V instruction: {pattern}")
            return m.group(1)

        if scale_bits is not None:
            const = match_result(rf"OpConstant %uint {scale_bits}")
            scale = match_result(rf"OpBitcast %float {const}")
            value = match_result(rf"OpFMul %float {value} {scale}")
        else:
            self.assertNotRegex(spv, r"\bOpFMul %float\b")

        if clamp:
            value = match_result(rf"OpExtInst %float %glsl FClamp {value} %f0 %f1")
        else:
            self.assertNotIn("FClamp", spv)

        word = match_result(rf"OpBitcast %uint {value}")
        dest = match_result(r"OpAccessChain %_ptr_Function_uint %vgpr %c0")
        if not re.search(rf"(?m)^ *OpStore {re.escape(dest)} {re.escape(word)}$", spv):
            old_word = match_result(rf"OpLoad %uint {dest}")
            masked = match_result(rf"OpSelect %uint %\w+ {word} {old_word}")
            self.assertRegex(spv, rf"(?m)^ *OpStore {dest} {masked}$")

    def test_no_output_modifiers(self):
        cases = (
            ("v_rcp_iflag_f32_e32 v0, v1", 0.5, 2.0),
            ("v_rcp_iflag_f32_e32 v0, v1", -0.25, -4.0),
            ("v_rcp_iflag_f32_e64 v0, v1", 0.5, 2.0),
            ("v_rcp_iflag_f32_e64 v0, v1", -0.25, -4.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            for insn, src_val, want in cases:
                with self.subTest(insn=insn, src=src_val):
                    spv = self.translate_rcp(Path(directory), insn, src_val)
                    self.assert_rcp_pipeline(spv, scale_bits=None, clamp=False)
                    self.assertEqual(si.run(spv).vgpr[0], fbits(want))

    def test_clamp_modifier(self):
        cases = (
            (0.5, 1.0),    # 1 / 0.5 = 2.0 -> clamped to 1.0
            (-0.5, 0.0),   # 1 / -0.5 = -2.0 -> clamped to 0.0
            (2.0, 0.5),    # 1 / 2.0 = 0.5 -> stays 0.5
        )
        with tempfile.TemporaryDirectory() as directory:
            for src_val, want in cases:
                with self.subTest(src=src_val):
                    spv = self.translate_rcp(Path(directory), "v_rcp_iflag_f32_e64 v0, v1 clamp", src_val)
                    self.assert_rcp_pipeline(spv, scale_bits=None, clamp=True)
                    self.assertEqual(si.run(spv).vgpr[0], fbits(want))

    def test_omod_multipliers(self):
        cases = (
            ("mul:2", 0x40000000, 0.5, 4.0),
            ("mul:4", 0x40800000, 0.5, 8.0),
            ("div:2", 0x3F000000, 0.5, 1.0),
            ("mul:2", 0x40000000, -4.0, -0.5),
            ("mul:4", 0x40800000, 4.0, 1.0),
            ("div:2", 0x3F000000, 2.0, 0.25),
        )
        with tempfile.TemporaryDirectory() as directory:
            for mod, scale_bits, src_val, want in cases:
                with self.subTest(mod=mod, src=src_val):
                    spv = self.translate_rcp(Path(directory), f"v_rcp_iflag_f32_e64 v0, v1 {mod}", src_val)
                    self.assert_rcp_pipeline(spv, scale_bits=scale_bits, clamp=False)
                    self.assertEqual(si.run(spv).vgpr[0], fbits(want))

    def test_omod_before_clamp(self):
        cases = (
            ("mul:2 clamp", 0x40000000, 0.5, 1.0),   # clamp(2.0 * 2.0) = 1.0
            ("mul:2 clamp", 0x40000000, 4.0, 0.5),   # clamp(0.25 * 2.0) = 0.5
            ("clamp div:2", 0x3F000000, 0.5, 1.0),   # clamp(2.0 * 0.5) = 1.0 (clamp-first would give 0.5)
            ("clamp div:2", 0x3F000000, -0.5, 0.0),  # clamp(-2.0 * 0.5) = 0.0
        )
        with tempfile.TemporaryDirectory() as directory:
            for mods, scale_bits, src_val, want in cases:
                with self.subTest(mods=mods, src=src_val):
                    spv = self.translate_rcp(Path(directory), f"v_rcp_iflag_f32_e64 v0, v1 {mods}", src_val)
                    self.assert_rcp_pipeline(spv, scale_bits=scale_bits, clamp=True)
                    self.assertEqual(si.run(spv).vgpr[0], fbits(want))

    def test_source_and_output_modifiers_combined(self):
        cases = (
            ("v_rcp_iflag_f32_e64 v0, -v1 clamp", True, False, None, True, -0.5, 1.0),
            ("v_rcp_iflag_f32_e64 v0, -v1 mul:2", True, False, 0x40000000, False, 0.5, -4.0),
            ("v_rcp_iflag_f32_e64 v0, |v1| mul:2", False, True, 0x40000000, False, -0.5, 4.0),
            ("v_rcp_iflag_f32_e64 v0, |v1| clamp", False, True, None, True, -0.5, 1.0),
            ("v_rcp_iflag_f32_e64 v0, -|v1| mul:2 clamp", True, True, 0x40000000, True, 0.5, 0.0),
            ("v_rcp_iflag_f32_e64 v0, -|v1| mul:2", True, True, 0x40000000, False, -0.5, -4.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            for insn, neg, abs_, scale_bits, clamp, src_val, want in cases:
                with self.subTest(insn=insn, src=src_val):
                    spv = self.translate_rcp(Path(directory), insn, src_val)
                    self.assert_rcp_pipeline(
                        spv, neg=neg, abs_=abs_, scale_bits=scale_bits, clamp=clamp
                    )
                    self.assertEqual(si.run(spv).vgpr[0], fbits(want))


if __name__ == "__main__":
    unittest.main()
