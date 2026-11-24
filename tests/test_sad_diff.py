"""tools/sad_diff.py: the SAD-family reference against hand-pinned ISA words, the source-kind sweep,
and a short seeded differential run. Standard library only; no GPU, no game files.

The family is v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32, v_msad_u8, v_qsad_pk_u16_u8 and
v_mqsad_pk_u16_u8 (docs/design.md's "Sum of absolute differences" and v_qsad contract rows).
"""
import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import sad_diff as d  # noqa: E402


def run(*body):
    """The reference's four exported words for a program."""
    return d.reference(list(body))


class Reference(unittest.TestCase):
    def test_plain_ops_pinned_words(self):
        # 0x01020304 against 0xFF00FF00: |01-FF|+|02-00|+|03-FF|+|04-00| = 254+2+252+4 = 512
        # the op overwrites v8 (its own S0/S2), v11 is never written and reads 0
        self.assertEqual(run("v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF00FF00",
                             "v_mov_b32 v10, 5", "v_sad_u8 v8, v8, v9, v10"),
                         [517, 0xFF00FF00, 5, 0])
        self.assertEqual(run("v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF00FF00",
                             "v_mov_b32 v10, 5", "v_sad_hi_u8 v11, v8, v9, v10")[3],
                         0x02000005)
        # halves 0xFFFF vs 0x0001 twice: 65534 + 65534 = 0x1FFFC
        self.assertEqual(run("v_mov_b32 v8, 0x0001FFFF", "v_mov_b32 v9, 0xFFFF0001",
                             "v_sad_u16 v11, v8, v9, 0")[3], 0x0001FFFC)
        self.assertEqual(run("v_mov_b32 v8, 0", "v_mov_b32 v9, 0xFFFFFFFF",
                             "v_sad_u32 v11, v8, v9, 1")[3], 0)               # wraps
        self.assertEqual(run("v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF000000",
                             "v_mov_b32 v10, 5", "v_msad_u8 v11, v8, v9, v10")[3],
                         259)                                                  # only the FF byte

    def test_quad_windows_slide_one_byte(self):
        # S0 = 0x08070605_04030201 against S1 = 0x04030201: the four rolling windows give 0, 4, 8, 12;
        # the result lands in v[10:11], overwriting the S1 seed that lived in v10
        self.assertEqual(run("v_mov_b32 v8, 0x04030201", "v_mov_b32 v9, 0x08070605",
                             "v_mov_b32 v10, 0x04030201", "v_qsad_pk_u16_u8 v[10:11], v[8:9], v10, 0"),
                         [0x04030201, 0x08070605, 0x00040000, 0x000C0008])
        # S1 = 0 masks every byte (each window adds nothing), so D = S2: v11 unwritten gives the
        # accumulator's high word 0, and the result overwrites the v[10:11] it read
        self.assertEqual(run("v_mov_b32 v8, 0xFFFFFFFF", "v_mov_b32 v9, 0xFFFFFFFF",
                             "v_mov_b32 v10, 5",
                             "v_mqsad_pk_u16_u8 v[10:11], v[8:9], 0, v[10:11]")[2], 5)

    def test_64bit_constant_rule(self):
        # A hex literal as a 64-bit operand has a zero high word; an inline constant sign-extends.
        lit = run("v_qsad_pk_u16_u8 v[8:9], 0xFFFF8000, 0, 0")
        pair = run("v_mov_b32 v8, 0xFFFF8000", "v_qsad_pk_u16_u8 v[8:9], v[8:9], 0, 0")
        self.assertEqual(lit, [0x027E027E, 0x00FF01FE, 0, 0])   # lanes 638, 638, 510, 255
        self.assertEqual(lit, pair)
        self.assertEqual(run("v_qsad_pk_u16_u8 v[8:9], -16, 0, 0"),
                         [0x03FC03ED, 0x03FC03FC, 0, 0])        # lanes 1005, 1020, 1020, 1020

    def test_hi_accumulator_wraps_at_32_bits(self):
        # byte sum 512 (<< 16 = 0x02000000) plus 0xFFFF0000 wraps at 32 bits to 0x01FF0000
        self.assertEqual(run("v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF00FF00",
                             "v_sad_hi_u8 v11, v8, v9, 0xFFFF0000")[3], 0x01FF0000)


class Generator(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(d.program(random.Random(5), 6), d.program(random.Random(5), 6))
        self.assertNotEqual(d.program(random.Random(5), 6), d.program(random.Random(6), 6))

    def test_ops_subset_is_respected(self):
        rng = random.Random(1)
        ops = ["v_sad_u8", "v_qsad_pk_u16_u8"]
        for _ in range(30):
            lines, used = d.program(rng, 4, ops)
            self.assertTrue(set(used) <= set(ops))
            self.assertTrue(lines)                       # seeds and ops only, no exp/s_endpgm
        # the sweep over the same subset enumerates only those ops, every kind combination
        cases = d.sweep_ops(ops)
        self.assertEqual(set(op for op, _ in cases), set(ops))
        self.assertEqual(len(cases), 125 + 80)

    def test_results_land_in_exported_registers(self):
        # every op's destination (whole register or pair) is inside the exported v8..v11
        dst = re.compile(r"^v(_?\S+) ")
        rng = random.Random(2)
        for _ in range(30):
            lines, _ = d.program(rng, 4)
            for line in lines:
                first = line.split()[0]
                if first not in d.OP_FORMS:
                    continue
                m = re.match(r"^v\[(\d+):(\d+)\]", line.split()[1])
                if m:
                    lo, hi = int(m.group(1)), int(m.group(2))
                    self.assertTrue(8 <= lo and hi <= 11)
                else:
                    self.assertGreaterEqual(int(line.split()[1].rstrip(",").lstrip("v")), 8)
                    self.assertLessEqual(int(line.split()[1].rstrip(",").lstrip("v")), 11)


class Translated(unittest.TestCase):
    def test_harness_canaries(self):
        self.assertTrue(d.self_test())

    def test_short_seeded_run_matches(self):
        counts, failure = d.run_suite(15, 3, 5)
        self.assertEqual(failure, None)
        self.assertEqual(counts["skip"], {})
        self.assertEqual(counts["pass"], 15)

    def test_sweep_covers_the_whole_kind_space(self):
        # 5 kinds ^ 3 positions for a plain op, 4 x 5 x 4 for a quad op's 64-bit S0/S2
        counts, failure = d.run_sweep(4, ["v_sad_u8"])
        self.assertEqual((failure, counts["skip"]), (None, {}))
        self.assertEqual(counts["pass"], 125)
        counts, failure = d.run_sweep(4, ["v_qsad_pk_u16_u8"])
        self.assertEqual((failure, counts["skip"]), (None, {}))
        self.assertEqual(counts["pass"], 80)

    def test_a_wrong_translation_is_caught(self):
        lines = ["v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF00FF00", "v_mov_b32 v10, 5",
                 "v_sad_u8 v8, v8, v9, v10"]
        real = d.translate_listing
        try:
            d.translate_listing = lambda l, n, t: [516] + real(l, n, t)[1:]
            self.assertIn("expected 0x00000205, actual 0x00000204", d.check(lines))
        finally:
            d.translate_listing = real


class EverythingSkipped(unittest.TestCase):
    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        import contextlib
        import io
        from unittest import mock

        def refuse(lines, name, tmpdir):
            raise d.rdna2spirv.Unsupported("unsupported form")
        out = io.StringIO()
        with (mock.patch.object(d, "translate_listing", refuse), mock.patch.object(sys, "argv", ["sad_diff.py", "--programs", "3"]),
              contextlib.redirect_stdout(out)):
            rc = d.main()
        self.assertEqual(rc, 2, out.getvalue())
        self.assertIn("no program ran: every one was skipped", out.getvalue())

if __name__ == "__main__":
    unittest.main()
