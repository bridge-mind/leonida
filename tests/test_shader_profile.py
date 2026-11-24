"""Unit tests for tools/shader_profile.py.

Validates instruction classification (ALU vs memory vs branch), operation ratio counting,
register footprint tracking (VGPR, SGPR, ranges, special registers), LLVM warning words, and CLI corpus profiling.
Built entirely with synthetic disassembly text; no game data or proprietary assets.

Run:
    python -I tests/test_shader_profile.py
    python -m unittest discover -s tests -v
"""
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_profile as P  # noqa: E402


def diag(line, col, src, kind="warning: invalid instruction encoding"):
    """One llvm-mc diagnostic as LLVM prints it: position and message, echoed source line, caret."""
    return f"<stdin>:{line}:{col}: {kind}\n{src}\n{' ' * (col - 1)}^\n"


def messages(*diags):
    """The llvm-mc messages section shader_disasm.py appends to a .s file."""
    return "\n; --- llvm-mc messages ---\n; " + "".join(diags).strip().replace("\n", "\n; ") + "\n"


def echo(hex_bytes):
    """llvm-mc's input as shader_disasm.py writes it: space-separated 0xNN tokens."""
    return " ".join(f"0x{b:02x}" for b in bytes.fromhex(hex_bytes))


class TestFamilyClassification(unittest.TestCase):
    def test_known_families(self):
        cases = [
            ("s_buffer_load_dword", "smem"),
            ("s_load_dwordx4", "smem"),
            ("s_mov_b32", "scalar"),
            ("s_add_u32", "scalar"),
            ("v_mov_b32", "vector"),
            ("v_fma_f32", "vector"),
            ("ds_read_b32", "lds"),
            ("ds_write_b32", "lds"),
            ("buffer_load_dword", "buffer"),
            ("tbuffer_load_format_x", "buffer"),
            ("image_sample", "image"),
            ("image_load", "image"),
            ("global_load_dword", "global"),
            ("global_store_dword", "global"),
            ("flat_load_dword", "flat"),
            ("flat_store_dword", "flat"),
            ("scratch_load_dword", "scratch"),
            ("scratch_store_dword", "scratch"),
            ("exp", "export"),
            ("unknown_instruction", "other"),
        ]
        for mnemonic, expected_family in cases:
            with self.subTest(mnemonic=mnemonic):
                self.assertEqual(P.family(mnemonic), expected_family)


class TestInstructionClassification(unittest.TestCase):
    def test_alu_instructions(self):
        alu_cases = [
            "v_mov_b32", "v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32",
            "v_cmp_gt_f32", "v_cndmask_b32", "v_cvt_f32_u32", "v_and_b32",
            "v_lshlrev_b32", "v_med3_f32", "v_bfi_b32", "v_bfe_u32",
            "s_mov_b32", "s_add_u32", "s_sub_u32", "s_and_b64", "s_or_b64",
            "s_xor_b64", "s_lshl_b64", "s_cmp_eq_u32", "s_cselect_b32",
        ]
        for m in alu_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "alu")
                self.assertTrue(P.is_alu(m))
                self.assertFalse(P.is_memory(m))
                self.assertFalse(P.is_branch(m))

    def test_memory_instructions(self):
        mem_cases = [
            "s_load_dword", "s_load_dwordx4", "s_buffer_load_dword",
            "ds_read_b32", "ds_write_b32", "buffer_load_dword",
            "buffer_store_dword", "tbuffer_load_format_x",
            "tbuffer_store_format_xyzw", "image_sample", "image_sample_lz",
            "image_load", "image_store", "global_load_dword",
            "global_store_dword", "flat_load_dword", "flat_store_dword",
            "scratch_load_dword", "scratch_store_dword",
        ]
        for m in mem_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "memory")
                self.assertTrue(P.is_memory(m))
                self.assertFalse(P.is_alu(m))
                self.assertFalse(P.is_branch(m))

    def test_branch_instructions(self):
        branch_cases = [
            "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
            "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz",
            "s_cbranch_execnz", "s_setpc_b64", "s_swappc_b64",
            "s_rfe_b64", "s_call_b64",
        ]
        for m in branch_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "branch")
                self.assertTrue(P.is_branch(m))
                self.assertFalse(P.is_alu(m))
                self.assertFalse(P.is_memory(m))

    def test_export_and_other_instructions(self):
        self.assertEqual(P.classify_instruction("exp"), "export")
        self.assertEqual(P.classify_instruction("s_nop"), "other")
        self.assertEqual(P.classify_instruction("s_endpgm"), "other")
        self.assertEqual(P.classify_instruction("s_barrier"), "other")
        self.assertEqual(P.classify_instruction("s_waitcnt"), "other")
        self.assertEqual(P.classify_instruction("s_waitcnt_vscnt"), "other")
        self.assertEqual(P.classify_instruction("s_sendmsg"), "other")
        self.assertEqual(P.classify_instruction(""), "other")
        self.assertEqual(P.classify_instruction("nonexistent_opcode"), "other")


class TestRatiosAndCategoryCounting(unittest.TestCase):
    def test_count_categories_from_list(self):
        mnemonics = [
            "v_mov_b32", "v_add_f32", "s_load_dword", "s_cbranch_scc0",
            "exp", "s_nop", "s_endpgm"
        ]
        counts = P.count_categories(mnemonics)
        self.assertEqual(counts["alu"], 2)
        self.assertEqual(counts["memory"], 1)
        self.assertEqual(counts["branch"], 1)
        self.assertEqual(counts["export"], 1)
        self.assertEqual(counts["other"], 2)
        self.assertEqual(counts["total"], 7)

    def test_count_categories_from_dict(self):
        mnemonics = {
            "v_add_f32": 10,
            "v_mul_f32": 5,
            "buffer_load_dword": 5,
            "s_branch": 2,
        }
        counts = P.count_categories(mnemonics)
        self.assertEqual(counts["alu"], 15)
        self.assertEqual(counts["memory"], 5)
        self.assertEqual(counts["branch"], 2)
        self.assertEqual(counts["export"], 0)
        self.assertEqual(counts["other"], 0)
        self.assertEqual(counts["total"], 22)

    def test_compute_ratios_balanced(self):
        counts = {"alu": 50, "memory": 25, "branch": 25, "export": 0, "other": 0, "total": 100}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_ratio"], 0.50)
        self.assertEqual(r["memory_ratio"], 0.25)
        self.assertEqual(r["branch_ratio"], 0.25)
        self.assertEqual(r["alu_to_memory"], 2.0)

    def test_compute_ratios_empty(self):
        counts = {"alu": 0, "memory": 0, "branch": 0, "export": 0, "other": 0, "total": 0}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_ratio"], 0.0)
        self.assertEqual(r["memory_ratio"], 0.0)
        self.assertEqual(r["branch_ratio"], 0.0)
        self.assertEqual(r["alu_to_memory"], 0.0)

    def test_compute_ratios_zero_memory(self):
        counts = {"alu": 10, "memory": 0, "branch": 2, "export": 0, "other": 0, "total": 12}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_to_memory"], float("inf"))


class TestRegisterFootprint(unittest.TestCase):
    def test_single_registers(self):
        line = "\tv_add_f32 v5, v2, v0"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {0, 2, 5})
        self.assertEqual(regs["sgprs"], set())
        self.assertEqual(regs["special"], set())

    def test_register_ranges(self):
        line = "\ts_load_dwordx4 s[4:7], s[0:1], 0x0"
        regs = P.parse_registers(line)
        self.assertEqual(regs["sgprs"], {0, 1, 4, 5, 6, 7})
        self.assertEqual(regs["vgprs"], set())

    def test_mixed_vgpr_and_sgpr(self):
        line = "\timage_sample v[0:3], v[8:9], s[16:23], s[24:27] dmask:0xf"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {0, 1, 2, 3, 8, 9})
        self.assertEqual(regs["sgprs"], {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27})

    def test_special_registers(self):
        line = "\tv_cndmask_b32 v0, v1, v2, vcc"
        regs = P.parse_registers(line)
        self.assertIn("vcc", regs["special"])
        self.assertEqual(regs["vgprs"], {0, 1, 2})

        line2 = "\ts_mov_b64 exec, s[0:1]"
        regs2 = P.parse_registers(line2)
        self.assertIn("exec", regs2["special"])
        self.assertEqual(regs2["sgprs"], {0, 1})

    def test_comments_stripped_for_registers(self):
        line = "\tv_mov_b32 v1, 1.0 ; encoding: [0x00,0x00,0x80,0xbf] v255 s99"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {1})
        self.assertEqual(regs["sgprs"], set())

    def test_footprint_calculation(self):
        shader = (
            "; synthetic pixel shader\n"
            "\ts_load_dwordx2 s[0:1], s[2:3], 0x10\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v31, v0, v0\n"
            "\ts_cbranch_scc0 0x10\n"
            "\ts_endpgm\n"
        )
        fp = P.register_footprint(shader)
        self.assertEqual(fp["max_vgpr"], 31)
        self.assertEqual(fp["vgpr_count"], 32)
        self.assertEqual(fp["unique_vgprs"], 2)
        self.assertEqual(fp["vgprs"], [0, 31])

        self.assertEqual(fp["max_sgpr"], 3)
        self.assertEqual(fp["sgpr_count"], 4)
        self.assertEqual(fp["unique_sgprs"], 4)
        self.assertEqual(fp["sgprs"], [0, 1, 2, 3])

    def test_footprint_empty(self):
        fp = P.register_footprint([])
        self.assertEqual(fp["max_vgpr"], -1)
        self.assertEqual(fp["max_sgpr"], -1)
        self.assertEqual(fp["vgpr_count"], 0)
        self.assertEqual(fp["sgpr_count"], 0)
        self.assertEqual(fp["vgprs"], [])
        self.assertEqual(fp["sgprs"], [])


class TestProfileShader(unittest.TestCase):
    def test_profile_shader_synthetic(self):
        text = (
            "; ps_sample code 0x0..0x40\n"
            "\ts_load_dwordx4 s[0:3], s[4:5], 0x0\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v1, v0, v0\n"
            "\tv_mul_f32 v2, v1, v0\n"
            "\ts_cbranch_scc0 0x20\n"
            "\texp mrt0 v0, v1, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        prof = P.profile_shader(text)
        self.assertEqual(prof["total_instructions"], 7)
        cats = prof["categories"]
        self.assertEqual(cats["alu"], 3)     # v_mov_b32, v_add_f32, v_mul_f32
        self.assertEqual(cats["memory"], 1)  # s_load_dwordx4
        self.assertEqual(cats["branch"], 1)  # s_cbranch_scc0
        self.assertEqual(cats["export"], 1)  # exp
        self.assertEqual(cats["other"], 1)   # s_endpgm

        ratios = prof["ratios"]
        self.assertAlmostEqual(ratios["alu_ratio"], 3 / 7, places=5)
        self.assertAlmostEqual(ratios["memory_ratio"], 1 / 7, places=5)
        self.assertAlmostEqual(ratios["branch_ratio"], 1 / 7, places=5)
        self.assertEqual(ratios["alu_to_memory"], 3.0)

        regs = prof["registers"]
        self.assertEqual(regs["max_vgpr"], 2)
        self.assertEqual(regs["vgpr_count"], 3)
        self.assertEqual(regs["max_sgpr"], 5)
        self.assertEqual(regs["sgpr_count"], 6)


class TestWarningWords(unittest.TestCase):
    def test_positions_in_echo(self):
        self.assertEqual(P.warning_words(messages(diag(1, 1, echo("feffffff")))), (["fe ff ff ff"], 0))
        # Standalone LLVM: line 2 of the input, indented two spaces, so column 3.
        self.assertEqual(P.warning_words(messages(diag(2, 3, "  " + echo("ffffffd7 000081bf")))),
                         (["ff ff ff d7"], 0))
        long_src = echo("00" * 2048 + "feffffff")
        self.assertEqual(P.warning_words(messages(diag(1, 10241, long_src))), (["fe ff ff ff"], 0))

    def test_unresolved_positions_and_tokens(self):
        src = echo("070080bf feffffff")
        self.assertEqual(P.warning_words(messages(diag(1, 21, src))), (["fe ff ff ff"], 0))
        for pos in ("0:21", "1:0", "1:2", "1:20", "1:22", "1:26", "1:36", "1:40", "1:" + "9" * 5000, "x:21",
                    "1:21:5", None):
            with self.subTest(pos=str(pos)[:12]):
                head = "<stdin>" + (f":{pos}" if pos else "") + ": warning: invalid instruction encoding"
                self.assertEqual(P.warning_words(messages(f"{head}\n{src}\n")), ([], 1))
        for src in ("0xfe 0xff 0xff", "0xfe 0xff 0xff 0xf", "0xfe 0xff 0xff 0xfff", "0xfe,0xff,0xff,0xff",
                    "0xfe 0xff 0xff 0xzz", "x0xfe 0xff 0xff 0xff"):
            with self.subTest(src=src):
                self.assertEqual(P.warning_words(messages(diag(1, 1, src))), ([], 1))

    def test_missing_or_truncated_echo_stays_in_its_diagnostic(self):
        warning = "<stdin>:1:1: warning: invalid instruction encoding\n"
        tail = diag(1, 21, echo("070080bf ffff"))    # LLVM's genuine warning for a two-byte tail
        self.assertEqual(P.warning_words(messages(warning)), ([], 1))
        self.assertEqual(P.warning_words(messages(warning, diag(1, 1, echo("ffffffd7")))), (["ff ff ff d7"], 1))
        self.assertEqual(P.warning_words(messages(tail, diag(1, 1, echo("ffff0000")))), (["ff ff 00 00"], 1))
        # A missing echo must not borrow bytes mentioned by the next diagnostic.
        note = "<stdin>:1:1: note: " + echo("00112233")
        self.assertEqual(P.warning_words(messages(f"<stdin>:1:20: warning: invalid instruction encoding\n{note}\n")),
                         ([], 1))

    def test_only_the_messages_section_and_warning_class(self):
        src = echo("feffffff")
        stray = "\ts_nop 0\n; <stdin>:1:1: warning: invalid instruction encoding\n; " + src + "\n"
        section = messages(diag(1, 1, echo("ffffffd7")))
        self.assertEqual(P.warning_words(stray + section), (["ff ff ff d7"], 0))
        for header in (";  --- llvm-mc messages ---", "; --- llvm-mc messages --- ", "; --- llvm-mc messages"):
            with self.subTest(header=header):
                text = stray + section.replace("; --- llvm-mc messages ---", header)
                self.assertEqual(P.warning_words(text), ([], 0))
        unrelated = messages(diag(1, 1, src, "error: invalid instruction encoding"),
                             diag(1, 1, src, "note: text mentioning: warning: invalid instruction encoding"),
                             diag(1, 1, src, "warning: potentially undefined instruction encoding"),
                             diag(1, 1, src, "warning: invalid instruction encoding, again"),
                             "; <stdin>:1:1: warning: invalid instruction encoding\n" + src + "\n")
        self.assertEqual(P.warning_words(unrelated), ([], 0))
        self.assertEqual(P.warning_words(""), ([], 0))


class TestCommandLineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.disasm_dir = os.path.join(self.tmp.name, "disasm")
        os.makedirs(self.disasm_dir, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        with open(os.path.join(self.disasm_dir, name + ".s"), "w", encoding="utf-8") as f:
            f.write(text)

    def run_profile(self, extra=()):
        out_json = os.path.join(self.tmp.name, "custom_profile.json")
        cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "shader_profile.py"), self.disasm_dir, "--out", out_json]
        res = subprocess.run(cmd + list(extra), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
        with open(out_json, encoding="utf-8") as f:
            return res.stdout, json.load(f, parse_constant=self.fail)

    def test_cli_default_report_is_next_to_disasm_directory(self):
        shader_dir = os.path.join(self.tmp.name, "work", "shaders", "dxbc")
        disasm_dir = os.path.join(shader_dir, "disasm")
        os.makedirs(disasm_dir)
        tool = os.path.join(ROOT, "tools", "shader_profile.py")
        result = subprocess.run([sys.executable, "-I", tool, disasm_dir], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(os.path.join(shader_dir, "profile.json"), encoding="utf-8") as report:
            self.assertEqual(json.load(report)["shaders"], 0)

    def test_cli_bad_source_or_out_is_one_line(self):
        tool = os.path.join(ROOT, "tools", "shader_profile.py")
        a_file = os.path.join(self.disasm_dir, "ps_001.s")
        with open(a_file, "w", encoding="utf-8") as f:
            f.write("\tv_mov_b32 v0, 1.0\n")
        missing = os.path.join(self.tmp.name, "missing")
        out_in_missing = os.path.join(self.tmp.name, "no_dir", "profile.json")
        out_is_dir = os.path.join(self.tmp.name, "out_dir")
        os.mkdir(out_is_dir)
        for args, message in (([missing], "does not exist"), ([a_file], "is not a folder"),
                              ([self.disasm_dir, "--out", out_in_missing], "folder for"),
                              ([self.disasm_dir, "--out", out_is_dir], "cannot write")):
            with self.subTest(case=message):
                res = subprocess.run([sys.executable, "-I", tool, *args], capture_output=True, text=True)
                self.assertEqual(res.returncode, 2)
                self.assertIn(message, res.stderr)
                self.assertNotIn("Traceback", res.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "profile.json")))

    def test_cli_alu_without_memory_writes_null(self):
        with open(os.path.join(self.disasm_dir, "ps_001.s"), "w", encoding="utf-8") as f:
            f.write("\tv_mov_b32 v0, 1.0\n\ts_endpgm\n")
        _, data = self.run_profile()
        self.assertEqual(data["categories"]["alu"], 1)
        self.assertEqual(data["categories"]["memory"], 0)
        self.assertIsNone(data["ratios"]["alu_to_memory"])

    def test_cli_counts_category_occurrences_by_stage(self):
        self.write("ps_first", "\tv_mov_b32 v0, 1\n" * 2)
        self.write("ps_second", "\tv_mov_b32 v1, 2\n\ts_endpgm\n")
        self.write("vs_first", "\ts_load_dword s0, s[2:3], 0\n")
        self.write("unclassified", "")
        self.write("zz_foo", "\tv_mov_b32 v2, 1\n")
        self.write("LIB_helper", "\ts_endpgm\n")
        _, data = self.run_profile()
        stages = data["categories_by_stage"]
        self.assertEqual(list(stages), ["?", "lib", "ps", "vs"])
        self.assertEqual(stages["ps"], {"alu": 3, "memory": 0, "branch": 0,
                                        "export": 0, "other": 1, "total": 4})
        self.assertEqual(stages["vs"]["memory"], 1)
        self.assertEqual(stages["?"]["alu"], 1)
        self.assertEqual(stages["lib"]["total"], 1)
        for key, count in data["categories"].items():
            self.assertEqual(sum(stage[key] for stage in stages.values()), count)

    def test_cli_empty_corpus(self):
        stdout, data = self.run_profile()
        self.assertEqual(data["shaders"], 0)
        self.assertEqual(data["categories_by_stage"], {})
        self.assertEqual(data["ratios"]["alu_to_memory"], 0.0)
        self.assertEqual(data["undecoded"], {"warnings": 0, "unresolved": 0, "by_word": []})
        self.assertIn("  undecoded: 0 LLVM warnings, 0 distinct words, 0 unresolved", stdout)

    def test_cli_takes_the_stage_from_the_carve_index(self):
        """Most corpus names carry no stage prefix; shader_carve.py's index.json (next to the disasm folder) has
        the stage from each container's program header, so it wins over the name. Names it doesn't list keep
        the name-prefix fallback."""
        with open(os.path.join(self.tmp.name, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "Foo", "file": "Foo_1.dxbc", "stage": "cs"},
                       {"name": "CS_Bar", "file": "CS_Bar.dxbc", "stage": "ms"}], f)
        for name in ("Foo_1", "CS_Bar", "vs_unlisted"):
            self.write(name, "\tv_mov_b32 v0, 1\n\ts_endpgm\n")
        _, data = self.run_profile(["--per-shader"])
        self.assertEqual({row["name"]: row["stage"] for row in data["per_shader"]},
                         {"Foo_1": "cs", "CS_Bar": "ms", "vs_unlisted": "vs"})
        self.assertEqual(sorted(data["registers"]), ["cs", "ms", "vs"])

    def test_cli_counts_only_instructions(self):
        with open(os.path.join(self.disasm_dir, "ps_001.s"), "w", encoding="utf-8") as f:
            f.write("\t.text\n\t.long 0\n\t; comment with v7\n\t\t\n\tv_mov_b32 v0, 1\n\ts_endpgm\n")
        _, data = self.run_profile(["--per-shader"])
        self.assertEqual(data["categories"]["total"], 2)
        self.assertEqual({row["m"] for row in data["by_mnemonic"]}, {"v_mov_b32", "s_endpgm"})
        self.assertEqual(data["per_shader"][0]["instructions"], 2)
        self.assertEqual(data["per_shader"][0]["vgpr_count"], 1)

    def test_cli_execution_and_json_output(self):
        s1 = (
            "; ps_001 code 0x0..0x20\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v1, v0, v0\n"
            "\ts_load_dword s0, s[2:3], 0x0\n"
            "\ts_endpgm\n"
        )
        s2 = (
            "; vs_002 code 0x0..0x20\n"
            "\ts_branch 0x10\n"
            "\texp pos0 v0, v1, v2, v3 done vm\n"
            "\ts_nop 0\n"
            "; <stdin>:5:1: warning: invalid instruction encoding\n"
            "; 0x12 0x34 0x56 0x78\n"
        )
        with open(os.path.join(self.disasm_dir, "ps_001.s"), "w", encoding="utf-8") as f:
            f.write(s1)
        with open(os.path.join(self.disasm_dir, "vs_002.s"), "w", encoding="utf-8") as f:
            f.write(s2)

        stdout, data = self.run_profile()
        self.assertIn("2 shaders", stdout)
        self.assertIn("ALU:", stdout)
        self.assertIn("Memory:", stdout)
        self.assertIn("Branch:", stdout)

        self.assertEqual(data["shaders"], 2)
        self.assertIn("families", data)
        self.assertIn("categories", data)
        self.assertIn("ratios", data)
        self.assertIn("by_mnemonic", data)
        # vs_002's warning lines are not in an llvm-mc messages section.
        self.assertEqual(data["undecoded"], {"warnings": 0, "unresolved": 0, "by_word": []})

        cats = data["categories"]
        self.assertEqual(cats["alu"], 2)
        self.assertEqual(cats["memory"], 1)
        self.assertEqual(cats["branch"], 1)
        self.assertEqual(cats["export"], 1)
        self.assertEqual(cats["other"], 2)

        ratios = data["ratios"]
        self.assertGreater(ratios["alu_ratio"], 0.0)
        self.assertGreater(ratios["memory_ratio"], 0.0)
        self.assertEqual(ratios["alu_to_memory"], 2.0)

        # Register pressure per stage: ps_001 uses v0-v1 and s0-s3, vs_002 uses v0-v3 and no SGPR.
        regs = data["registers"]
        self.assertEqual(regs["ps"], {"shaders": 1, "vgpr_max": 2, "vgpr_mean": 2.0, "sgpr_max": 4,
                                      "sgpr_mean": 4.0, "vgpr_hist": {"32": 1}})
        self.assertEqual(regs["vs"]["vgpr_max"], 4)
        self.assertEqual(regs["vs"]["sgpr_max"], 0)
        self.assertIn("ps registers: VGPR max 2", stdout)
        self.assertNotIn("per_shader", data)

        _, data = self.run_profile(["--per-shader"])
        self.assertEqual(data["per_shader"], [
            {"name": "ps_001", "stage": "ps", "instructions": 4, "vgpr_count": 2, "sgpr_count": 4},
            {"name": "vs_002", "stage": "vs", "instructions": 3, "vgpr_count": 4, "sgpr_count": 0},
        ])

    def test_cli_ignores_directories_with_shader_suffix(self):
        self.write("cs_main", "\ts_endpgm\n")
        os.mkdir(os.path.join(self.disasm_dir, "cs_cache.s"))

        _, data = self.run_profile(["--per-shader"])

        self.assertEqual(data["shaders"], 1)
        self.assertEqual([shader["name"] for shader in data["per_shader"]], ["cs_main"])

    def test_profile_rankings_do_not_depend_on_directory_order(self) -> None:
        self.write("ps_vector", "\n".join((
            "\tv_add_f32 v0, v0, v1",
            "\tv_add_f32 v0, v0, v1",
            "\tv_and_b32 v0, v0, v1",
            "\tv_max_f32 v0, v0, v1",
            "\tv_min_f32 v0, v0, v1",
            "\tv_mov_b32 v0, v1",
            "\tv_or_b32 v0, v0, v1",
            "\tv_xor_b32 v0, v0, v1",
            "\ts_endpgm",
        )))
        self.write("ps_scalar", "\n".join((
            "\ts_add_u32 s0, s0, s1",
            "\ts_and_b32 s0, s0, s1",
            "\ts_mov_b32 s0, s1",
            "\ts_nop 0",
            "\ts_or_b32 s0, s0, s1",
            "\ts_xor_b32 s0, s0, s1",
            "\ts_endpgm",
        )))
        with os.scandir(self.disasm_dir) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        report = os.path.join(self.tmp.name, "profile.json")
        args = ["shader_profile.py", self.disasm_dir, "--out", report]

        def profile(order: list[os.DirEntry[str]]) -> tuple[str, bytes]:
            stdout = io.StringIO()
            with (patch.object(P.os, "scandir", return_value=nullcontext(order)),
                  patch.object(sys, "argv", args), redirect_stdout(stdout)):
                self.assertIsNone(P.main())
            with open(report, "rb") as output:
                return stdout.getvalue(), output.read()

        forward, data = profile(entries)
        reverse, reversed_data = profile(list(reversed(entries)))
        self.assertEqual(data, reversed_data)
        self.assertEqual(forward, reverse)
        rankings = json.loads(data)["by_mnemonic"]
        self.assertEqual(rankings[0], {"m": "s_endpgm", "shaders": 2, "count": 2})
        self.assertEqual([row["m"] for row in rankings[1:]], sorted(row["m"] for row in rankings[1:]))
        most_widespread = next(line for line in forward.splitlines() if "most widespread:" in line)
        self.assertEqual(most_widespread.removeprefix("  most widespread: "),
                         ", ".join(f"{row['m']} ({row['shaders']})" for row in rankings[:12]))

    def test_cli_warning_words_from_llvm_messages(self):
        a = echo("070080bf feffffff ffffffd7 050080bf feffffff 000081bf")
        b = echo("090080bf ffffffd7 000081bf")
        nop9, endpgm = "\ts_nop 9 ; encoding: [0x09,0x00,0x80,0xbf]\n", "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"
        self.write("cs_probe", "; cs_probe  code 0x128..0x140 (header), gfx1013, 3 invalid words\n"
                   "\ts_nop 7 ; encoding: [0x07,0x00,0x80,0xbf]\n\ts_nop 5 ; encoding: [0x05,0x00,0x80,0xbf]\n"
                   + endpgm + messages(diag(1, 21, a), diag(1, 41, a), diag(1, 81, a)))
        self.write("ps_probe", "; ps_probe  code 0x128..0x134 (header), gfx1013, 1 invalid words\n" + nop9 + endpgm
                   + messages(diag(1, 21, b)))
        self.write("vs_clean", "; vs_clean  code 0x128..0x130 (header), gfx1013, 0 invalid words\n" + nop9 + endpgm)
        rows = [{"word": "ff ff ff d7", "shaders": 2, "count": 2}, {"word": "fe ff ff ff", "shaders": 1, "count": 2}]

        stdout, data = self.run_profile()
        self.assertEqual(data["undecoded"], {"warnings": 4, "unresolved": 0, "by_word": rows})
        self.assertEqual({r["m"]: (r["shaders"], r["count"]) for r in data["by_mnemonic"]},
                         {"s_nop": (3, 4), "s_endpgm": (3, 3)})
        self.assertIn("\n  undecoded: 4 LLVM warnings, 2 distinct words, 0 unresolved (words only in the JSON report)\n",
                      stdout)
        per_stdout, per_data = self.run_profile(["--per-shader"])
        self.assertEqual(per_data["undecoded"], data["undecoded"])
        self.assertEqual([e["name"] for e in per_data["per_shader"]], ["cs_probe", "ps_probe", "vs_clean"])
        for out in (stdout, per_stdout):
            for private in ("ff ff", "fe ff", "cs_probe", "ps_probe", "vs_clean"):
                self.assertNotIn(private, out)

        self.write("gs_tail", "; gs_tail  code 0x128..0x132 (header), gfx1013, 1 invalid words\n"
                   "\ts_nop 7 ; encoding: [0x07,0x00,0x80,0xbf]\n" + messages(diag(1, 21, echo("070080bf ffff"))))
        stdout, data = self.run_profile()
        self.assertEqual(data["undecoded"], {"warnings": 5, "unresolved": 1, "by_word": rows})
        self.assertIn("  undecoded: 5 LLVM warnings, 2 distinct words, 1 unresolved", stdout)

    def test_cli_ranks_words_by_count_then_shaders_then_word(self):
        self.write("cs_1", "\ts_nop 0\n" + messages(diag(1, 1, echo("01" * 4)), diag(2, 1, echo("aa" * 4)),
                                                    diag(3, 1, echo("aa" * 4))))
        self.write("ps_2", "\ts_nop 0\n" + messages(diag(1, 1, echo("bb" * 4)), diag(2, 1, echo("00" * 4))))
        self.write("vs_3", "\ts_nop 0\n" + messages(diag(1, 1, echo("bb" * 4))))
        expected = {"warnings": 6, "unresolved": 0, "by_word": [
            {"word": "bb bb bb bb", "shaders": 2, "count": 2},
            {"word": "aa aa aa aa", "shaders": 1, "count": 2},
            {"word": "00 00 00 00", "shaders": 1, "count": 1},
            {"word": "01 01 01 01", "shaders": 1, "count": 1}]}
        for extra in ([], ["--per-shader"]):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_profile(extra)[1]["undecoded"], expected)

    def test_register_summary_histogram(self):
        summary = P.register_summary({"cs": [(10, 4), (40, 8), (100, 20), (300, 2)]})
        self.assertEqual(summary["cs"]["vgpr_hist"], {"32": 1, "64": 1, "128": 1, ">256": 1})
        self.assertEqual(summary["cs"]["vgpr_max"], 300)
        self.assertEqual(summary["cs"]["sgpr_mean"], 8.5)
        self.assertEqual(P.register_summary({}), {})

    def test_same_output_path_identity(self):
        a = os.path.join(self.disasm_dir, "a.json")
        self.assertTrue(P.same_output_path(a, a))
        b = os.path.join(self.disasm_dir, "b.json")
        self.assertFalse(P.same_output_path(a, b))

    def assert_carve_index_output_refused(self, output):
        tool = os.path.join(ROOT, "tools", "shader_profile.py")
        res = subprocess.run(
            [sys.executable, "-I", tool, self.disasm_dir, "--out", output],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("ERROR: output would overwrite the shader carve index", res.stderr)

    def test_cli_reserves_the_carve_index_path(self):
        self.write("cs_main", "\ts_endpgm\n")
        index = os.path.join(self.tmp.name, "index.json")
        self.assert_carve_index_output_refused(index)
        self.assertFalse(os.path.exists(index))
        original = '[{"file": "cs_main.dxbc", "stage": "cs"}]'
        with open(index, "w", encoding="utf-8") as f:
            f.write(original)
        self.assert_carve_index_output_refused(index)
        with open(index, encoding="utf-8") as f:
            self.assertEqual(f.read(), original)

    def test_cli_refuses_a_hard_link_to_the_carve_index(self):
        self.write("cs_main", "\ts_endpgm\n")
        index = os.path.join(self.tmp.name, "index.json")
        original = '[{"file": "cs_main.dxbc", "stage": "cs"}]'
        with open(index, "w", encoding="utf-8") as f:
            f.write(original)
        alias = os.path.join(self.tmp.name, "alias.json")
        try:
            os.link(index, alias)
        except OSError as error:
            self.skipTest(f"hard links are unavailable: {error}")
        self.assert_carve_index_output_refused(alias)
        with open(index, encoding="utf-8") as f:
            self.assertEqual(f.read(), original)

    def test_cli_refuses_output_overwriting_input_shader(self):
        tool = os.path.join(ROOT, "tools", "shader_profile.py")
        shader_file = os.path.join(self.disasm_dir, "ps_001.s")
        original_content = "\tv_mov_b32 v0, 1.0\n\ts_endpgm\n"
        with open(shader_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        res = subprocess.run(
            [sys.executable, "-I", tool, self.disasm_dir, "--out", shader_file],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("ERROR: output would overwrite input shader", res.stderr)
        with open(shader_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), original_content)

    def test_unreadable_listing_reports_its_path_and_preserves_report(self):
        from io import StringIO
        from unittest.mock import patch

        self.write("cs_main", "\ts_endpgm\n")
        listing = os.path.join(self.disasm_dir, "cs_main.s")
        report = os.path.join(self.tmp.name, "profile.json")
        original = b'{"previous": true}\n'
        with open(report, "wb") as f:
            f.write(original)

        def open_listing(path, *args, **kwargs):
            if path == listing:
                raise PermissionError(13, "Cannot read listing", listing)
            return open(path, *args, **kwargs)

        stderr = StringIO()
        with (patch.object(P, "open", open_listing, create=True),
              patch.object(sys, "argv", ["shader_profile.py", self.disasm_dir, "--out", report]),
              patch.object(sys, "stderr", stderr)):
            self.assertEqual(P.main(), 2)

        self.assertEqual(stderr.getvalue().splitlines(), [f"ERROR: cannot read {listing}: Cannot read listing"])
        with open(report, "rb") as f:
            self.assertEqual(f.read(), original)


if __name__ == "__main__":
    unittest.main()
