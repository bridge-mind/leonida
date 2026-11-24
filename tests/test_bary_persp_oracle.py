"""check.py's reference for ps_persp_bary (vs_persp's triangle with clip w = 1, 2, 4): a perspective-correct image
passes, the affine (NoPerspective) one fails on every covered pixel, and a stray pixel outside the triangle is caught.
No GPU needed.

    python -I tests/test_bary_persp_oracle.py
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import check

CLEAR = (-1.0, -1.0, -1.0, -1.0)


def image(perspective):
    """The 64x64 mrt0 image ps_persp_bary should draw, with perspective-correct or affine weights."""
    (x0, y0), (x1, y1), (x2, y2) = check.PERSP_SCREEN
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    px = []
    for i in range(64 * 64):
        x, y = i % 64 + 0.5, i // 64 + 0.5
        l1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / area
        l2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / area
        l = (1 - l1 - l2, l1, l2)
        if min(l) < 0:
            px.append(CLEAR)
            continue
        p = [lk / wk for lk, wk in zip(l, check.PERSP_W)] if perspective else list(l)
        p = [pk / sum(p) for pk in p]
        px.append((sum(pk * a[0] for pk, a in zip(p, check.PERSP_ATTR)),
                   sum(pk * a[1] for pk, a in zip(p, check.PERSP_ATTR)), p[1], p[2]))
    return px


class BaryPerspReference(unittest.TestCase):
    def test_perspective_correct_image_passes(self):
        self.assertEqual(check.bary_persp_problem(image(True)), "")

    def test_affine_weights_fail_every_covered_pixel(self):
        problem = check.bary_persp_problem(image(False))
        self.assertTrue(problem.startswith("1158 pixels off"), problem)

    def test_a_pixel_outside_the_triangle_is_caught(self):
        px = image(True)
        px[0] = (0.0, 0.0, 0.0, 0.0)   # (0, 0) is far outside: it must keep the clear value
        self.assertIn("1 pixels off", check.bary_persp_problem(px))

    def test_a_centre_next_to_an_edge_is_judged(self):
        px = image(True)
        px[16 * 64 + 56] = (0.0, 0.0, 0.0, 0.0)   # (56, 16): covered, 6.5e-4 of the way in from an edge
        self.assertIn("1 pixels off", check.bary_persp_problem(px))

    def test_runs_in_both_runners(self):
        self.assertIn("ps_persp_bary", check.PIXEL_TESTS)
        self.assertEqual(check.PIXEL_VERTEX["ps_persp_bary"], "vs_persp")
        for name in ("ps_persp_bary", "vs_persp"):
            self.assertTrue((ROOT / "runtime" / "test" / "shaders" / f"{name}.asm").is_file())


if __name__ == "__main__":
    unittest.main()
