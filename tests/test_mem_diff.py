"""Tests for tools/mem_diff.py, the FUZZ-5 memory differential fuzzer (standard library only; no game data, GPU or
llvm-mc).

    python -I tests/test_mem_diff.py
"""
import os
import random
import sys
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import mem_diff as md  # noqa: E402


def program(lines, memory):
    p = md.Program()
    p.lines, p.memory = lines, dict(memory)
    return p


class Harness(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertTrue(md.self_test())

    def test_generation_is_deterministic(self):
        a = md.generate_program(random.Random(42), 3)
        b = md.generate_program(random.Random(42), 3)
        self.assertEqual(a.lines, b.lines)
        self.assertEqual(a.memory, b.memory)

    def test_tbuffer_case_uses_a_bound_descriptor(self):
        rng = random.Random(20261122)
        prog = md.generate_program(rng, rng.randrange(1, 3), families=["buffer_format"])
        self.assertTrue(any(line.startswith("tbuffer_load_format_") for line in prog.lines))
        descriptor = next(line for line in prog.lines if line.startswith("s_mov_b32 s3,"))
        self.assertNotEqual(int(descriptor.split(",")[1], 0) >> 12 & 0x7f, 0)

    def test_every_family_runs_clean(self):
        for family in sorted(md.GENERATORS):
            with self.subTest(family=family):
                counts, first = md.run_suite(60, seed=20261122, max_ops=2, families=[family])
                self.assertEqual(counts["mismatch"], 0, first)
                self.assertEqual(counts["harness-error"], 0, first)
                self.assertEqual(counts["skip"], {}, counts["skip"])
                self.assertEqual(counts["pass"], 60)


class EverythingSkipped(unittest.TestCase):
    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        import contextlib
        import io
        from unittest import mock
        out = io.StringIO()
        with mock.patch.object(md, "run_one", lambda *args: ("skip", "unsupported form")), \
                contextlib.redirect_stdout(out):
            rc = md.main(["--programs", "2", "--seed", "1"])
        self.assertEqual(rc, 2, out.getvalue())
        self.assertIn("no program ran: every one was skipped", out.getvalue())


class Reference(unittest.TestCase):
    """The reference model on hand-made cases (the expected words are worked out by hand)."""

    def run_ref(self, lines, memory):
        ref = md.RefMachine(dict(memory))
        ref.run(lines)
        return ref

    def test_structured_buffer_address(self):
        # base 0x3000, stride 16, index 2, voffset 4, soffset 8, offset:4 -> 0x3000 + 8 + 4 + 32 + 4 = 0x3030
        ref = self.run_ref(["s_mov_b32 s0, 0x3000", "s_mov_b32 s1, 0x100000", "s_mov_b32 s2, 4", "s_mov_b32 s3, 0",
                            "v_mov_b32 v2, 2", "v_mov_b32 v3, 4",
                            "buffer_load_dword v4, v[2:3], s[0:3], 8 idxen offen offset:4"], {0x3030: 0xABCD})
        self.assertEqual(ref.v[4], 0xABCD)

    def test_s_buffer_load_masks_the_descriptor_flags(self):
        # V# word1 bits 31:16 (stride/flags) are not part of the base address.
        ref = self.run_ref(["s_mov_b32 s0, 0x4000", "s_mov_b32 s1, 0xFFFF0001", "s_mov_b32 s2, 0x100", "s_mov_b32 s3, 0",
                            "s_buffer_load_dword s8, s[0:3], 0x8"], {(1 << 32) + 0x4008: 0x55})
        self.assertEqual(ref.s[8], 0x55)

    def test_global_negative_offset_and_sign_extension(self):
        ref = self.run_ref(["v_mov_b32 v2, 0x5010", "v_mov_b32 v3, 0",
                            "global_load_sbyte v4, v[2:3], off offset:-15"], {0x5000: 0x0000_8000})
        self.assertEqual(ref.v[4], 0xFFFFFF80)       # byte 1 of the word at 0x5000, sign-extended

    def test_cmpswap_and_glc(self):
        memory = {0x6000: 7}
        lines = ["s_mov_b32 s0, 0x6000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x10", "s_mov_b32 s3, 0",
                 "v_mov_b32 v4, 9", "v_mov_b32 v5, 7", "buffer_atomic_cmpswap v[4:5], off, s[0:3], 0 glc"]
        ref = self.run_ref(lines, memory)
        self.assertEqual((ref.mem[0x6000], ref.v[4]), (9, 7))

    def test_buffer_load_sbyte_lane_offset(self):
        lines = ["s_mov_b32 s0, 0x7000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x10", "s_mov_b32 s3, 0",
                 "buffer_load_sbyte v4, off, s[0:3], 0 offset:2"]
        ref = self.run_ref(lines, {0x7000: 0x0080_0000})
        self.assertEqual(ref.v[4], 0xFFFFFF80)

    def test_global_atomic_cmpswap_glc_and_no_glc(self):
        lines = ["s_mov_b32 s4, 0x8000", "s_mov_b32 s5, 0", "v_mov_b32 v2, 0x14",
                 "v_mov_b32 v4, 11", "v_mov_b32 v5, 5",
                 "global_atomic_cmpswap v6, v2, v[4:5], s[4:5] offset:-4 glc",
                 "v_mov_b32 v4, 22",
                 "global_atomic_add v2, v4, s[4:5] offset:-4"]
        ref = self.run_ref(lines, {0x8010: 5})
        self.assertEqual((ref.v[6], ref.mem[0x8010]), (5, 33))


class FormatDecode(unittest.TestCase):
    """The reference decode of typed-buffer formats, from the GFX10 names (expected words worked out by hand)."""

    def test_names_list_widths_from_the_top_bit(self):
        self.assertEqual(md.parse_format("GFX10_FORMAT_2_10_10_10_UNORM"), ([10, 10, 10, 2], "UNORM"))
        self.assertEqual(md.parse_format("GFX10_FORMAT_16_16_SNORM"), ([16, 16], "SNORM"))
        self.assertIsNone(md.parse_format("GFX10_FORMAT_10_11_11_FLOAT"))     # packed float: not modelled

    def test_correct_rounding(self):
        self.assertEqual(md.f32_round(Fraction(3, 255)), 0x3C40C0C1)     # the multiply-by-reciprocal gives ...C2
        self.assertEqual(md.f32_round(Fraction(1, 1)), 0x3F800000)
        self.assertEqual(md.f32_round(Fraction(1, 3)), 0x3EAAAAAB)

    def test_decode_examples(self):
        # 8_8_8_8_UINT: X in the low byte; four channels, nothing missing.
        self.assertEqual(md.decode_element(0xDDCCBBAA, [8, 8, 8, 8], "UINT"), [0xAA, 0xBB, 0xCC, 0xDD])
        # 16_16_SNORM: X = 0x8000 clamps to -1.0, Y = 0x7FFF is 1.0; Z/W repeat XYXY (SHADER-86).
        self.assertEqual(md.decode_element(0x7FFF8000, [16, 16], "SNORM"),
                         [0xBF800000, 0x3F800000, 0xBF800000, 0x3F800000])
        # 2_10_10_10_UNORM: X10 at bit 0 ... W2 at bit 30. All ones in X, W = 3 -> 1.0, Y = Z = 0.
        self.assertEqual(md.decode_element(0xC00003FF, [10, 10, 10, 2], "UNORM"),
                         [0x3F800000, 0, 0, 0x3F800000])
        # 32_UINT: missing Y/Z/W repeat X (SHADER-86).
        self.assertEqual(md.decode_element(7, [32], "UINT"), [7, 7, 7, 7])

    def test_mtbuf_bound_unbound_and_identity(self):
        # Seed 20261122's first format load: gfx1036 returned zero for INVALID, 2/3 for a bound descriptor.
        for fmt, want in ((0, 0), (20, 0x3f2aaaab)):
            with self.subTest(fmt=fmt):
                ref = md.RefMachine({0x728c: 0xaededaaa})
                ref.run(["s_mov_b32 s0, 0x7000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000",
                         f"s_mov_b32 s3, {md.OOB_RAW | md.XYZW | fmt << 12:#x}", "v_mov_b32 v2, 0x28c",
                         "tbuffer_load_format_x v8, v2, s[0:3], 0 format:[BUF_FMT_2_10_10_10_UNORM] offen"])
                self.assertEqual(ref.v[8], want)
        ref = md.RefMachine({0x7000: 42})
        ref.run(["s_mov_b32 s0, 0x7000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000",
                 f"s_mov_b32 s3, {md.OOB_RAW | 20 << 12:#x}",
                 "tbuffer_load_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_FMT_32_UINT]"])
        self.assertEqual([ref.v[i] for i in range(8, 12)], [42, 0, 0, 0])

    def test_mtbuf_default_format_matches_explicit_8_unorm(self) -> None:
        descriptor = md.OOB_RAW | md.XYZW | (20 << 12)
        setup = ["s_mov_b32 s0, 0x7000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000",
                 f"s_mov_b32 s3, {descriptor:#x}"]
        for modifier in (" format:[BUF_FMT_8_UNORM]", ""):
            with self.subTest(modifier=modifier):
                load = "tbuffer_load_format_xyzw v[8:11], off, s[0:3], 0" + modifier
                probe = program(setup + [load, "s_waitcnt vmcnt(0)"], {0x7000: 0xFF})
                ref = md.RefMachine(dict(probe.memory))
                ref.run(probe.lines)
                self.assertEqual([ref.v[reg] for reg in md.EXPORT], [0x3F800000, 0, 0, 0])
                status, detail = md.run_one("ps_mem_default_mtbuf", probe)
                self.assertEqual(status, "match", detail)

    def test_shader86_closed_open_questions(self):
        """Missing W repeats X; UNORM raw 3 matches correctly rounded 0x3c40c0c1 (no 1-ULP split)."""
        base = ["s_mov_b32 s0, 0x7000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100"]
        w32 = program(base + [f"s_mov_b32 s3, {22 << 12 | md.XYZW | md.OOB_RAW:#x}", "v_mov_b32 v2, 0",
                              "buffer_load_format_xyzw v[4:7], v2, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
                              "v_mov_b32 v8, v4", "v_mov_b32 v9, v5", "v_mov_b32 v10, v6", "v_mov_b32 v11, v7"],
                      {0x7000: 0x40000000})
        self.assertEqual(md.run_one("ps_mem_w32", w32)[0], "match")
        unorm = program(base + [f"s_mov_b32 s3, {56 << 12 | md.XYZW | md.OOB_RAW:#x}", "v_mov_b32 v2, 0",
                                "buffer_load_format_x v4, v2, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
                                "v_mov_b32 v8, v4", "v_mov_b32 v9, v4", "v_mov_b32 v10, v4", "v_mov_b32 v11, v4"],
                        {0x7000: 3})
        self.assertEqual(md.run_one("ps_mem_unorm3", unorm)[0], "match")


class GlobalOffsetRegression(unittest.TestCase):
    """llvm-mc prints a gfx10 global_* immediate as signed decimal ('offset:-16'). The translation must add -16, not
    sign-extend the printed value a second time (which reads 4 KiB lower)."""

    def test_negative_global_offset(self):
        memory = {0x8000 - 16: 0x11111111, 0x8000 - 16 - 0x1000: 0x22222222}
        for form in (["v_mov_b32 v2, 0x8000", "v_mov_b32 v3, 0", "global_load_dword v8, v[2:3], off offset:-16"],
                     ["s_mov_b32 s4, 0x8000", "s_mov_b32 s5, 0", "v_mov_b32 v2, 0",
                      "global_load_dword v8, v2, s[4:5] offset:-16"]):
            with self.subTest(form=form[-1]):
                status, detail = md.run_one("ps_mem_global_neg", program(
                    form + ["s_waitcnt vmcnt(0)", "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"], memory))
                self.assertEqual(status, "match", detail)

    def test_negative_global_store_offset(self):
        status, detail = md.run_one("ps_mem_global_neg_store", program(
            ["v_mov_b32 v2, 0x9000", "v_mov_b32 v3, 0", "v_mov_b32 v4, 0xC0FFEE",
             "global_store_dword v[2:3], v4, off offset:-2048", "v_mov_b32 v8, 0", "v_mov_b32 v9, 0",
             "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"], {}))
        self.assertEqual(status, "match", detail)


if __name__ == "__main__":
    unittest.main()
