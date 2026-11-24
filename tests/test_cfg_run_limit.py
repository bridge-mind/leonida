"""tools/cfg_structure.run stops after `limit` blocks, and a limit of zero or less runs none (standard library only).

    python -I tests/test_cfg_run_limit.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cfg_structure  # noqa: E402


def straight_line(n):
    """A tree of n blocks, 0..n-1, where each block goes to the next one."""
    return [("block", i) for i in range(n)], lambda b: b + 1


class RunLimitTest(unittest.TestCase):
    def test_runs_everything_under_the_limit(self):
        tree, choose = straight_line(3)
        self.assertEqual(cfg_structure.run(tree, choose, 10), [0, 1, 2])

    def test_stops_after_limit_blocks(self):
        tree, choose = straight_line(5)
        self.assertEqual(cfg_structure.run(tree, choose, 2), [0, 1])
        self.assertEqual(cfg_structure.run(tree, choose, 1), [0])

    def test_limit_equal_to_the_length_runs_everything(self):
        tree, choose = straight_line(3)
        self.assertEqual(cfg_structure.run(tree, choose, 3), [0, 1, 2])

    def test_zero_or_negative_limit_runs_nothing(self):
        tree, choose = straight_line(3)
        self.assertEqual(cfg_structure.run(tree, choose, 0), [])
        self.assertEqual(cfg_structure.run(tree, choose, -4), [])


if __name__ == "__main__":
    unittest.main()
