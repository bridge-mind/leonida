"""tools/ps_helper_probe.py: the analysis on a CPU model of what gfx1036 does (each check, and that it fails when the
model breaks that rule), the programs' shape, and (with LEONIDA_PS_ORACLE_BUILD set to a runtime build
and an AMD GPU selected) the measurement itself. Standard library only."""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ps_helper_probe as H  # noqa: E402

W, Hh = H.SIZE


def u32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def model(helpers_at_entry=False, empty_quads=False, swap_lanes=False, helper_export=False, flat_helpers=False):
    """What the measured gfx1036 gives, built from the geometry: pixel centres inside the triangle are covered, the
    quads with a covered pixel are packed 16 to a wave64 in raster order. Each flag breaks one measured rule."""
    def inside(p):
        i, j = H.bary(p)
        return i > 0 and j > 0 and i + j < 1
    cov = {(x, y) for y in range(Hh) for x in range(W) if inside((x, y))}
    quads = sorted({(x // 2, y // 2) for x, y in cov}, key=lambda q: (q[1], q[0]))
    if empty_quads:
        quads.insert(0, (99, 99))          # a quad with no covered pixel takes lanes 0..3
    lane, wave_of = {}, {}
    for n, q in enumerate(quads):
        for k in range(4):
            p = (2 * q[0] + (k & 1), 2 * q[1] + (k >> 1))
            if swap_lanes:
                k ^= 1
            lane[p], wave_of[p] = (n % 16) * 4 + k, n // 16
    entry, wqm = {}, {}
    for p in cov:
        w = wave_of[p]
        entry[w] = entry.get(w, 0) | 1 << lane[p]
        if helpers_at_entry:
            for k in (1, 2, 3):
                entry[w] |= 1 << lane[H.neighbour(p, k)]
    for p in lane:
        wqm[wave_of[p]] = wqm.get(wave_of[p], 0) | 1 << lane[p]
    res = {n: {(x, y): H.CLEAR for y in range(Hh) for x in range(W)} for n in H.PROGRAMS}
    for p in cov:
        e, q = entry[wave_of[p]], wqm[wave_of[p]]
        x, y = H.ndc(p)
        res["masks"][p] = (lane[p], e & 0xFFFFFFFF, e >> 32, q & 0xFFFFFFFF)
        res["self"][p] = (lane[p], q >> 32, u32(x), u32(y))
        nb = {k: H.ndc(H.neighbour(p, k)) for k in (1, 2, 3)}
        if flat_helpers:
            nb = {k: (x, y) if H.neighbour(p, k) not in cov else v for k, v in nb.items()}
        res["nb12"][p] = (u32(nb[1][0]), u32(nb[1][1]), u32(nb[2][0]), u32(nb[2][1]))
        res["nb3"][p] = (u32(nb[3][0]), u32(nb[3][1])) * 2
        i, j = H.bary(H.neighbour(p, 1))
        res["nowqm_ij"][p] = (u32(i), u32(j), 0xDEADBEEF, H.SENTINEL)
        res["nowqm_masks"][p] = (lane[p], e & 0xFFFFFFFF, e >> 32, e & 0xFFFFFFFF)
        res["helper_export"][p] = (0x3F800000,) * 4
    if helper_export:
        outside = next(p for p in lane if p not in cov)
        res["helper_export"][outside] = (0x3F800000,) * 4
    return res


class Analysis(unittest.TestCase):
    def failing(self, res):
        return [n for n, ok, _d in H.analyze(res) if not ok]

    def test_the_measured_behaviour_passes_every_check(self):
        checks = H.analyze(model())
        self.assertEqual([n for n, ok, _d in checks if not ok], [])
        self.assertEqual(len(checks), 10)

    def test_each_rule_broken_fails_its_check(self):
        for flag, check in (("helpers_at_entry", "entry EXEC"), ("empty_quads", "quads packed"),
                            ("swap_lanes", "lane % 4"), ("helper_export", "an exp"),
                            ("flat_helpers", "after s_wqm a helper")):
            with self.subTest(flag=flag):
                failing = self.failing(model(**{flag: True}))
                self.assertTrue(any(n.startswith(check) for n in failing), failing)

    def test_full_coverage_has_no_helpers_to_measure(self):
        res = model()
        for p in res["masks"]:
            if res["masks"][p] == H.CLEAR:
                for name in H.PROGRAMS:
                    res[name][p] = res[name][next(iter(q for q in res["masks"] if res["masks"][q] != H.CLEAR))]
        self.assertIn("partial quads at the edge", self.failing(res))

    def test_geometry(self):
        self.assertEqual(H.bary((0, 0)), H.bary((0, 0)))
        i, j = H.bary((0, 0))
        self.assertTrue(0 < i < 0.1 and 0 < j < 0.1)
        self.assertEqual([H.neighbour((4, 6), k) for k in (1, 2, 3)], [(5, 6), (4, 7), (5, 7)])
        self.assertEqual(H.ndc((0, 0)), (-1 + 1 / 16, -1 + 1 / 16))


class Programs(unittest.TestCase):
    def test_every_program_ends_and_exports_once(self):
        for name in H.PROGRAMS:
            with self.subTest(name=name):
                lines = H.program(name)
                self.assertEqual(lines[-1], "s_endpgm")
                self.assertEqual(sum(ln.startswith("exp ") for ln in lines), 1)
                self.assertEqual("s_wqm_b64 exec, exec" in lines, H.PROGRAMS[name][0])


@unittest.skipUnless(os.environ.get("LEONIDA_PS_ORACLE_BUILD"), "set LEONIDA_PS_ORACLE_BUILD to a runtime build "
                     "and select an AMD GPU (LEONIDA_VK_DEVICE) to measure helper lanes on hardware")
class OnHardware(unittest.TestCase):
    def test_helper_lanes_at_a_triangle_edge(self):
        try:
            _device, res = H.measure(os.environ["LEONIDA_PS_ORACLE_BUILD"])
        except H.ps_oracle.Unavailable as e:
            self.skipTest(str(e))
        self.assertEqual([(n, d) for n, ok, d in H.analyze(res) if not ok], [])


if __name__ == "__main__":
    unittest.main()
