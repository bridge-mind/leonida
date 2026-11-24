"""Negative and out-of-range indices for tools/zero_span.py.

The helper documents that its counts match len(buf[start:end]) and that all_zero() is true only for an empty
or all-zero slice. Python slices normalize negative indices and clamp out-of-range ones; the helper used to
clip only the upper bound, so the two disagreed for ranges such as buf[-5:10]. Synthetic buffers only.

    python -I tests/test_zero_span_negative.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import zero_span  # noqa: E402

DATA = b"ABCDEFGHIJ" * 20          # 200 non-zero bytes
ZEROS = bytes(200)
RANGES = [
    (-30, 200), (-5, 10), (-200, -195), (-250, 250),
    (0, -5), (0, 200), (195, 205), (190, 1000), (10, 5),
]


class ZeroSpanNegativeIndices(unittest.TestCase):
    def test_counts_match_the_slice_for_negative_and_clamped_ranges(self):
        for buf in (DATA, ZEROS):
            for start, end in RANGES:
                with self.subTest(buf=buf[:1], start=start, end=end):
                    zero, unread = zero_span.gap_split(buf, start, end)
                    self.assertEqual(zero + unread, len(buf[start:end]))
                    expected_zero = len(buf[start:end]) if not any(buf[start:end]) else 0
                    self.assertEqual((zero, unread), (expected_zero, len(buf[start:end]) - expected_zero))

    def test_all_zero_agrees_with_the_slice(self):
        for buf in (DATA, ZEROS):
            for start, end in RANGES:
                with self.subTest(buf=buf[:1], start=start, end=end):
                    self.assertEqual(zero_span.all_zero(buf, start, end), not any(buf[start:end]))

    def test_exact_regressions(self):
        self.assertEqual(zero_span.gap_split(DATA, -5, 10), (0, 0))       # slice is empty
        self.assertEqual(zero_span.gap_split(DATA, -30, 200), (0, 30))
        self.assertEqual(zero_span.gap_split(ZEROS, -30, 200), (30, 0))
        self.assertFalse(zero_span.all_zero(DATA, 0, -5))                 # 195 non-zero bytes
        self.assertTrue(zero_span.all_zero(ZEROS, 0, -5))

    def test_positive_ranges_are_unchanged(self):
        self.assertEqual(zero_span.gap_split(DATA, 0, 10), (0, 10))
        self.assertEqual(zero_span.gap_split(ZEROS, 0, 10), (10, 0))
        self.assertEqual(zero_span.gap_split(DATA, 0, 1000), (0, 200))


if __name__ == "__main__":
    unittest.main()
