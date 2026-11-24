"""Tests for tools/shader_ir.py, built from synthetic AMDGPU disassembly (no game data).

    python -I tests/test_shader_ir.py
"""
import contextlib
import errno
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "shader_ir.py")
    spec = importlib.util.spec_from_file_location("shader_ir_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shader_ir = load_module()


class OperandTests(unittest.TestCase):
    def test_single_registers(self):
        self.assertEqual(
            shader_ir.operand("s5"),
            {"k": "sgpr", "r": 5, "n": 1, "neg": False, "abs": False},
        )
        self.assertEqual(
            shader_ir.operand("v3"),
            {"k": "vgpr", "r": 3, "n": 1, "neg": False, "abs": False},
        )

    def test_register_ranges(self):
        self.assertEqual(
            shader_ir.operand("s[4:7]"),
            {"k": "sgpr", "r": 4, "n": 4, "neg": False, "abs": False},
        )
        self.assertEqual(
            shader_ir.operand("v[0:0]"),
            {"k": "vgpr", "r": 0, "n": 1, "neg": False, "abs": False},
        )

    def test_negation_and_absolute_value(self):
        self.assertTrue(shader_ir.operand("-v2")["neg"])
        self.assertTrue(shader_ir.operand("|s1|")["abs"])
        both = shader_ir.operand("-|v2|")
        self.assertTrue(both["neg"])
        self.assertTrue(both["abs"])

    def test_inline_and_float_constants(self):
        self.assertEqual(shader_ir.operand("0x10"), {"k": "imm", "v": 16})
        self.assertEqual(shader_ir.operand("-5"), {"k": "imm", "v": -5})
        self.assertEqual(shader_ir.operand("1.5"), {"k": "fimm", "v": 1.5, "abs": False})
        self.assertEqual(shader_ir.operand("-2.5e1"), {"k": "fimm", "v": -25.0, "abs": False})
        self.assertEqual(shader_ir.operand("|1.0|"), {"k": "fimm", "v": 1.0, "abs": True})
        # The regex accepts leading-zero decimals; int(core, 0) used to reject them and abort the run.
        self.assertEqual(shader_ir.operand("010"), {"k": "imm", "v": 10})
        self.assertEqual(shader_ir.operand("0001"), {"k": "imm", "v": 1})
        self.assertEqual(shader_ir.operand("08"), {"k": "imm", "v": 8})
        self.assertEqual(shader_ir.operand("-010"), {"k": "imm", "v": -10})

    def test_hardware_fp_modifiers_on_constants(self):
        # LLVM AMDGPUInstPrinter::printOperandAndFPInputMods emits neg(<imm>), |<imm>|, -|<imm>| and the
        # assembler also accepts abs(<imm>) / neg(abs(<imm>)). Keep abs/neg as VOP3 FP modifier flags rather
        # than folding neg into two's-complement negation (neg(1) is bit-31 flip 0x80000001, not -1 = 0xffffffff)
        # or folding outer '-' of -|x| into v before FAbs.
        self.assertEqual(shader_ir.operand("neg(0)"), {"k": "imm", "v": 0, "neg": True})
        self.assertEqual(shader_ir.operand("neg(1)"), {"k": "imm", "v": 1, "neg": True})
        self.assertEqual(shader_ir.operand("neg(-1)"), {"k": "imm", "v": -1, "neg": True})
        self.assertEqual(shader_ir.operand("neg(5)"), {"k": "imm", "v": 5, "neg": True})
        self.assertEqual(shader_ir.operand("neg(0x3f800000)"), {"k": "imm", "v": 0x3F800000, "neg": True})
        self.assertEqual(shader_ir.operand("neg(1.0)"), {"k": "fimm", "v": 1.0, "abs": False, "neg": True})
        self.assertEqual(shader_ir.operand("neg(-1.0)"), {"k": "fimm", "v": -1.0, "abs": False, "neg": True})
        self.assertEqual(shader_ir.operand("-|1.0|"), {"k": "fimm", "v": 1.0, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("-|-2.0|"), {"k": "fimm", "v": -2.0, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("-|0.15915494|"), {"k": "fimm", "v": 0.15915494, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("neg(|1.0|)"), {"k": "fimm", "v": 1.0, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("|0xbf800000|"), {"k": "imm", "v": 0xBF800000, "abs": True})
        self.assertEqual(shader_ir.operand("-|0x3f800000|"), {"k": "imm", "v": 0x3F800000, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("-|0|"), {"k": "imm", "v": 0, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("|-0x10|"), {"k": "imm", "v": -16, "abs": True})
        self.assertEqual(shader_ir.operand("abs(0xbf800000)"), {"k": "imm", "v": 0xBF800000, "abs": True})
        self.assertEqual(shader_ir.operand("neg(abs(0x3f800000))"), {"k": "imm", "v": 0x3F800000, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("abs(1.0)"), {"k": "fimm", "v": 1.0, "abs": True})
        self.assertEqual(shader_ir.operand("-abs(1.0)"), {"k": "fimm", "v": 1.0, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("neg(neg(1.0))"), {"k": "fimm", "v": 1.0, "abs": False})
        self.assertEqual(shader_ir.operand("-neg(0x3f800000)"), {"k": "imm", "v": 0x3F800000})

    def test_negation_and_abs_of_registers_compose(self):
        self.assertTrue(shader_ir.operand("neg(v1)")["neg"])
        self.assertFalse(shader_ir.operand("neg(neg(v1))")["neg"])
        self.assertFalse(shader_ir.operand("-neg(v1)")["neg"])
        self.assertEqual(
            shader_ir.operand("|neg(v1)|"),
            {"k": "vgpr", "r": 1, "n": 1, "neg": False, "abs": True},
        )
        self.assertEqual(
            shader_ir.operand("-|neg(v1)|"),
            {"k": "vgpr", "r": 1, "n": 1, "neg": True, "abs": True},
        )
        self.assertEqual(
            shader_ir.operand("abs(v1)"),
            {"k": "vgpr", "r": 1, "n": 1, "neg": False, "abs": True},
        )
        self.assertEqual(
            shader_ir.operand("-abs(v1)"),
            {"k": "vgpr", "r": 1, "n": 1, "neg": True, "abs": True},
        )
        self.assertEqual(
            shader_ir.operand("abs(neg(v1))"),
            {"k": "vgpr", "r": 1, "n": 1, "neg": False, "abs": True},
        )

    def test_sign_extension(self):
        self.assertTrue(shader_ir.operand("sext(v1)")["sext"])
        self.assertFalse(shader_ir.operand("sext(v1)")["neg"])
        self.assertTrue(shader_ir.operand("-sext(v1)")["neg"])
        self.assertTrue(shader_ir.operand("sext(neg(v1))")["neg"])
        self.assertTrue(shader_ir.operand("|sext(v1)|")["abs"])
        self.assertTrue(shader_ir.operand("|sext(v1)|")["sext"])

    def test_ttmp_registers(self):
        self.assertEqual(shader_ir.operand("ttmp5"), {"k": "ttmp", "t": "ttmp5"})
        self.assertEqual(shader_ir.operand("ttmp[4:5]"), {"k": "ttmp", "t": "ttmp[4:5]"})

    def test_non_contiguous_vgpr_list(self):
        self.assertEqual(shader_ir.operand("[v1, v2, v3]"), {"k": "vlist", "r": [1, 2, 3]})

    def test_attribute_and_interpolation(self):
        self.assertEqual(shader_ir.operand("attr2.w"), {"k": "attr", "a": 2, "c": 3})
        self.assertEqual(shader_ir.operand("p10"), {"k": "interp", "t": "p10"})

    def test_export_targets(self):
        self.assertEqual(shader_ir.operand("pos0"), {"k": "target", "t": "pos0"})
        self.assertEqual(shader_ir.operand("mrt1"), {"k": "target", "t": "mrt1"})
        self.assertEqual(shader_ir.operand("mrtz"), {"k": "target", "t": "mrtz"})

    def test_special_registers_including_null(self):
        for name in ("vcc_lo", "exec", "m0", "scc", "null", "off"):
            with self.subTest(name=name):
                self.assertEqual(shader_ir.operand(name)["k"], "special")
                self.assertEqual(shader_ir.operand(name)["t"], name)

    def test_hardware_register_and_counter_forms(self):
        self.assertEqual(shader_ir.operand("hwreg(0x1234)")["k"], "special")
        self.assertEqual(shader_ir.operand("vmcnt(3)")["k"], "special")
        self.assertEqual(shader_ir.operand("sendmsg(MSG_GS)")["k"], "special")

    def test_unknown_operand_falls_back_to_raw(self):
        self.assertEqual(shader_ir.operand("weirdthing"), {"k": "raw", "t": "weirdthing"})

    def test_llvm23_lit_is_the_constant(self):
        # LLVM 23 prints lit(V) for a literal it could also show as an inline constant (v_fma_mix* sources).
        for text, plain in (("lit(0x42200000)", "0x42200000"), ("|lit(0x3f800000)|", "|0x3f800000|"),
                            ("neg(lit(0x3f800000))", "neg(0x3f800000)"), ("-lit(0x3f800000)", "neg(0x3f800000)")):
            with self.subTest(text=text):
                self.assertEqual(shader_ir.operand(text), shader_ir.operand(plain))

    def test_llvm23_lit_of_a_register_is_not_unwrapped(self):
        # lit() wraps a constant; a register inside it is not a literal and must not parse as the register (or as a
        # constant): it keeps the lit(...) text, which the translator refuses as before lit() was read.
        for text in ("lit(v2)", "lit(s4)", "-lit(v2)", "lit(vcc_lo)"):
            with self.subTest(text=text):
                op = shader_ir.operand(text)
                self.assertNotIn(op["k"], ("vgpr", "sgpr", "imm", "fimm"))
                self.assertIn("lit(", op.get("t", ""))


class SplitArgsTests(unittest.TestCase):
    def test_commas_inside_parentheses_stay_together(self):
        self.assertEqual(
            shader_ir.split_args("a, b(1, 2), c"),
            ["a", " b(1, 2)", " c"],
        )

    def test_commas_inside_brackets_stay_together(self):
        self.assertEqual(shader_ir.split_args("[v1, v2], v3"), ["[v1, v2]", " v3"])

    def test_trailing_comma_is_dropped(self):
        self.assertEqual(shader_ir.split_args("v0,"), ["v0"])

    def test_empty_text(self):
        self.assertEqual(shader_ir.split_args(""), [])


class ParseLineTests(unittest.TestCase):
    def test_operation_and_operands(self):
        ins = shader_ir.parse_line("\tv_add_f32 v0, v1, v2")
        self.assertEqual(ins["op"], "v_add_f32")
        self.assertEqual([a["k"] for a in ins["args"]], ["vgpr", "vgpr", "vgpr"])
        self.assertEqual(ins["mods"], [])
        self.assertIsNone(ins["enc"])

    def test_encoding_bytes(self):
        ins = shader_ir.parse_line("\tv_mov_b32 v0, v1 ; encoding: [0x0a,0xff]")
        self.assertEqual(ins["enc"], "0aff")
        self.assertEqual(len(ins["args"]), 2)

    def test_leading_zero_decimal_operand_does_not_abort_a_line(self):
        ins = shader_ir.parse_line("\tv_mov_b32_e32 v0, 010 ; encoding: [0x00,0x00,0x02,0x7e]")
        self.assertEqual(ins["op"], "v_mov_b32_e32")
        self.assertEqual(ins["args"][1], {"k": "imm", "v": 10})

    def test_comment_or_blank_lines_are_skipped(self):
        self.assertIsNone(shader_ir.parse_line("; only a comment"))
        self.assertIsNone(shader_ir.parse_line(""))
        self.assertIsNone(shader_ir.parse_line("   \t  "))

    def test_modifiers_are_removed_from_the_operands(self):
        ins = shader_ir.parse_line(
            "\tbuffer_load_dword v5, v[0:3], s[4:7], 0 offen offset:8 glc slc"
        )
        self.assertEqual(ins["mods"], ["offen", "offset:8", "glc", "slc"])
        self.assertEqual([a["k"] for a in ins["args"]], ["vgpr", "vgpr", "sgpr", "imm"])
        self.assertEqual(ins["args"][1]["n"], 4)
        self.assertEqual(ins["args"][2]["n"], 4)

    def test_dpp_modifiers_keep_their_bracket_lists(self):
        ins = shader_ir.parse_line(
            "\tv_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        )
        self.assertEqual(
            ins["mods"],
            ["quad_perm:[1,0,3,2]", "row_mask:0xf", "bank_mask:0xf"],
        )
        self.assertEqual(len(ins["args"]), 2)

    def test_ds_swizzle_offset_modifier_with_spaces_stays_intact(self):
        for mod in (
            "offset:swizzle(QUAD_PERM, 3, 2, 1, 0)",
            'offset:swizzle(BITMASK_PERM, "00p01")',
            "offset:swizzle(BROADCAST, 4, 0)",
            "offset:swizzle(SWAP, 1)",
            "offset:swizzle(REVERSE, 4)",
        ):
            with self.subTest(mod=mod):
                ins = shader_ir.parse_line(f"\tds_swizzle_b32 v0, v1 {mod} gds")
                self.assertEqual(ins["op"], "ds_swizzle_b32")
                self.assertEqual(ins["mods"], [mod, "gds"])
                self.assertEqual(
                    ins["args"],
                    [
                        {"k": "vgpr", "r": 0, "n": 1, "neg": False, "abs": False},
                        {"k": "vgpr", "r": 1, "n": 1, "neg": False, "abs": False},
                    ],
                )

    def test_exp_target_is_split_from_its_operands(self):
        ins = shader_ir.parse_line("\texp mrt0 v0, v1, v2, v3")
        self.assertEqual(len(ins["args"]), 5)
        self.assertEqual(ins["args"][0], {"k": "target", "t": "mrt0"})
        self.assertEqual(ins["args"][1]["k"], "vgpr")

    def test_exp_null_off_target(self):
        ins = shader_ir.parse_line("\texp null off, v0, v1, v2")
        self.assertEqual([a["t"] for a in ins["args"][:2]], ["null", "off"])
        self.assertEqual(len(ins["args"]), 5)

    def test_sendmsg_commas_stay_in_one_operand(self):
        ins = shader_ir.parse_line("\ts_sendmsg sendmsg(MSG_GS, GS_OP_NOP)")
        self.assertEqual(len(ins["args"]), 1)
        self.assertEqual(ins["args"][0]["k"], "special")

    def test_no_operands(self):
        self.assertEqual(shader_ir.parse_line("\ts_endpgm")["args"], [])


class ParseFileTests(unittest.TestCase):
    def write(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "shader_abc.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def test_header_start_and_pc_stepping(self):
        path = self.write(
            "; shader.cs_5_0 code 0x00000100..0x00000110\n"
            "\tv_mov_b32 v0, 1.0 ; encoding: [0x01,0x02]\n"
            "\tv_add_f32 v1, v0, v2 ; encoding: [0x03,0x04]\n"
            "\tv_sub_f32 v3, v1, v0 ; no encoding bytes here\n"
            "\t.directive skipped\n"
            "plain text is not an instruction\n"
            "; ---\n"
            "\tv_mul_f32 v4, v3, v2 ; encoding: [0x05,0x06]\n"
            "; invalid instruction encoding [0xdead]\n"
        )
        info = shader_ir.parse_file(path)
        self.assertEqual(info["name"], "shader_abc")
        self.assertEqual(info["code_start"], 0x100)
        self.assertEqual([i["op"] for i in info["insns"]],
                         ["v_mov_b32", "v_add_f32", "v_sub_f32"])
        # pc advances by the encoding length; an instruction without bytes keeps the offset.
        self.assertEqual([i["pc"] for i in info["insns"]], [0x100, 0x102, 0x104])
        self.assertEqual(info["invalid"], 1)

    def test_tab_between_mnemonic_and_operands(self):
        path = self.write(
            "; sample code 0x0..0x4 (header)\n"
            "\ts_mov_b32\ts3, 1 ; encoding: [0x81,0x00,0x83,0xbe]\n"
        )
        ins = shader_ir.parse_file(path)["insns"][0]
        self.assertEqual(ins["op"], "s_mov_b32")
        self.assertEqual(ins["args"], [
            {"k": "sgpr", "r": 3, "n": 1, "neg": False, "abs": False},
            {"k": "imm", "v": 1},
        ])

    def test_file_without_a_header_starts_at_zero(self):
        path = self.write("\tv_mov_b32 v0, v1 ; encoding: [0xaa,0xbb]\n")
        info = shader_ir.parse_file(path)
        self.assertEqual(info["code_start"], 0)
        self.assertEqual(info["insns"][0]["pc"], 0)
        self.assertEqual(info["insns"][0]["enc"], "aabb")

    def test_header_only_file_has_no_instructions(self):
        path = self.write("; shader.cs_6_0 code 0x00001000..0x00001008\n")
        info = shader_ir.parse_file(path)
        self.assertEqual(info["code_start"], 0x1000)
        self.assertEqual(info["insns"], [])
        self.assertEqual(info["invalid"], 0)


class CliInputTest(unittest.TestCase):
    """The tool read sys.argv by hand: no arguments, a bad --limit or a missing folder ended in a traceback."""

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "shader_ir.py"), *args],
                              capture_output=True, text=True, timeout=60)

    def test_census_reports_input_io_errors_without_a_partial_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.s")
            second = os.path.join(tmp, "second.s")
            for path in (first, second):
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\ts_endpgm\n")
            parse_file = shader_ir.parse_file

            def fail_second(path):
                if path == second:
                    raise PermissionError(errno.EACCES, "permission denied", second)
                return parse_file(path)

            for owner, attr, effect, path, diagnostic in (
                    (shader_ir.os, "scandir", OSError(errno.EIO, "I/O error", tmp), tmp, "cannot list"),
                    (shader_ir, "parse_file", fail_second, second, "cannot read")):
                with self.subTest(operation=attr):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with mock.patch.object(owner, attr, side_effect=effect), \
                            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        self.assertEqual(shader_ir.main([tmp]), 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertEqual(len(stderr.getvalue().strip().splitlines()), 1)
                    self.assertIn(diagnostic, stderr.getvalue())
                    self.assertIn(path, stderr.getvalue())

    def test_bad_arguments_are_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            not_a_folder = os.path.join(tmp, "one.s")
            with open(not_a_folder, "w", encoding="utf-8") as f:
                f.write("s_endpgm\n")
            cases = ((), (tmp, "--limit", "x"), (tmp, "--limit", "0"),
                     (os.path.join(tmp, "missing"),), (not_a_folder,))
            for args in cases:
                with self.subTest(args=args):
                    p = self.run_tool(*args)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn("usage:", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)

    def test_s_directory_does_not_count_toward_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "a_cache.s"))
            with open(os.path.join(tmp, "b_shader.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            result = self.run_tool(tmp, "--limit", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 shaders, 1 instructions", result.stdout)

    def test_folder_with_limit_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.s", "b.s"):
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                    f.write("  s_endpgm  ; encoding: [0x00,0x00,0x81,0xbf]\n")
            p = self.run_tool(tmp, "--limit", "1")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(p.stdout.startswith("1 shaders,"), p.stdout)  # --limit 1 of the two files


if __name__ == "__main__":
    unittest.main()
