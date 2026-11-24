"""tools/image_query_probe.py: the resinfo and get_lod rules against the values measured on gfx1036, the descriptor
bits, the kernels, and the comparison and its exit codes (through a fake oracle that plays the GPU with the rules).
The last class runs on the hardware when the oracle tools are present and the GPU is GFX10, and is skipped otherwise.
Standard library only."""
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import image_query_probe as pr  # noqa: E402
from hw_oracle import runner  # noqa: E402


class Rules(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)

    def test_a_2d_depth_word_is_zero_whatever_word4_holds(self):
        for depth in (0, 5, 0x3F, 0x1FFF):
            self.assertEqual(pr.resinfo_model(16, 4, pr.TYPE_2D, depth, 0, 4, 0)[2], 0)

    def test_the_layer_count_is_minified_from_the_base_level(self):
        got = [pr.resinfo_model(256, 256, pr.TYPE_2D_ARRAY, 99, base, 8, level)[2]
               for base, level in ((0, 0), (0, 1), (0, 3), (1, 0), (1, 5))]
        self.assertEqual(got, [100, 50, 12, 50, 1])

    def test_past_the_last_level_the_size_is_zero_and_the_level_count_stays(self):
        self.assertEqual(pr.resinfo_model(16, 4, pr.TYPE_2D, 0, 1, 4, 4), [0, 0, 0, 4])
        self.assertEqual(pr.resinfo_model(16, 4, pr.TYPE_2D_ARRAY, 5, 0, 2, 9), [0, 0, 0, 3])

    def test_clamped_lod_per_mip_filter(self):
        # computed LOD 2.75 + bias 0, MIN_LOD 0.5, MAX_LOD 2.25, five levels
        self.assertEqual(pr.lod_model(2.75, pr.MIP_NONE, 0.5, 2.25, 0, 5), (0.0, 2.75))
        self.assertEqual(pr.lod_model(2.75, pr.MIP_POINT, 0.5, 2.25, 0, 5), (2.0, 2.75))
        self.assertEqual(pr.lod_model(2.75, pr.MIP_LINEAR, 0.5, 2.25, 0, 5), (2.25, 2.75))

    def test_point_rounds_ties_up_and_min_lod_wins_over_max_lod(self):
        self.assertEqual(pr.lod_model(0.5, pr.MIP_POINT, 0, 15, 0, 5)[0], 1)
        self.assertEqual(pr.lod_model(7.0, pr.MIP_LINEAR, 3, 2, 0, 5)[0], 3)
        self.assertEqual(pr.lod_model(0.0, pr.MIP_LINEAR, 5.5, 15, 0, 5)[0], 4)    # capped by the level count


class Descriptors(unittest.TestCase):
    def test_tsharp_type_levels_and_depth(self):
        t = pr.tsharp_words("gfx1036", 16, 4, pr.TYPE_2D_ARRAY, depth_field=5, base=1, last=4)
        self.assertEqual(t[3] >> 28, pr.TYPE_2D_ARRAY)
        self.assertEqual((t[3] >> 12) & 0xF, 1)
        self.assertEqual((t[3] >> 16) & 0xF, 4)
        self.assertEqual(t[4] & 0x1FFF, 5)
        self.assertEqual(t[2] >> 31, 1)                  # GFX10 RESOURCE_LEVEL

    def test_ssharp_fields(self):
        s = pr.ssharp_words(pr.MIP_LINEAR, 0.5, 2.25, -0.5)
        self.assertEqual(s[1] & 0xFFF, 128)
        self.assertEqual(s[1] >> 12 & 0xFFF, 576)
        self.assertEqual(s[2] & 0x3FFF, 0x3FFF & -128)
        self.assertEqual(s[2] >> 26 & 3, pr.MIP_LINEAR)

    def test_kernels_use_the_instruction_and_dim(self):
        k = pr.resinfo_kernel(pr.tsharp_words("gfx1036", 16, 4, pr.TYPE_2D_ARRAY, 5), 2, pr.TYPE_2D_ARRAY, 64)
        self.assertIn("image_get_resinfo v[8:11], v2, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY", k.lines)
        self.assertIn("v_mov_b32 v2, 2", k.lines)
        self.assertEqual((k.wave, k.threads, k.words), (64, 1, 4))
        k = pr.lod_kernel(pr.tsharp_words("gfx1036", 16, 16), pr.ssharp_words(pr.MIP_POINT, 0, 15), 0.25, 0.25, 32)
        self.assertIn("image_get_lod v[8:9], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D", k.lines)
        self.assertEqual((k.threads, k.words), (4, 16))

    def test_every_lod_case_is_exact_in_binary(self):
        # the probe compares raw LODs exactly, so each step must be a power of two times a quarter-step LOD
        for name, w, h, last, du, dv, samp, lod in pr.lod_cases():
            self.assertEqual(lod * 4, int(lod * 4), name)


class FakeOracle:
    """Plays the GPU with the probe's own rules, optionally wrong in one place."""
    arch = "gfx1036"

    def __init__(self, depth_2d=0):
        self.depth_2d = depth_2d

    def run(self, kernels, words=4):
        out = []
        for k in kernels:
            text = "\n".join(k.lines)
            sgprs = [int(ln.split(", ")[1], 16) for ln in k.lines if ln.startswith("s_mov_b32 s")]
            t, s = sgprs[:8], sgprs[8:12]
            typ, base, last = t[3] >> 28, (t[3] >> 12) & 0xF, (t[3] >> 16) & 0xF
            width = (t[1] >> 30 | (t[2] & 0xFFF) << 2) + 1
            height = (t[2] >> 14 & 0xFFFF) + 1
            if "image_get_resinfo" in text:
                level = int(next(ln for ln in k.lines if ln.startswith("v_mov_b32 v2,")).split(", ")[1])
                r = pr.resinfo_model(width, height, typ, t[4] & 0x1FFF, base, last, level)
                if typ == pr.TYPE_2D and r[0]:
                    r[2] = self.depth_2d
                out.append(r)
            else:
                du = pr.u2f(int(next(ln for ln in k.lines if ln.startswith("v_mul_f32 v2")).split()[2][:-1], 16))
                dv = pr.u2f(int(next(ln for ln in k.lines if ln.startswith("v_mul_f32 v3")).split()[2][:-1], 16))
                lod = round(math.log2(max(du * width, dv * height)) * 256) / 256   # the LOD has 8 fraction bits
                bias = ((s[2] & 0x3FFF) ^ 0x2000) - 0x2000
                c, raw = pr.lod_model(lod, s[2] >> 26 & 3, (s[1] & 0xFFF) / 256, (s[1] >> 12 & 0xFFF) / 256,
                                      bias / 256, last - base + 1)
                out.append([pr.f2u(c), pr.f2u(raw), 0, 0] * 4)
        return out

    def close(self):
        pass


class Comparison(unittest.TestCase):
    def test_rules_played_back_match(self):
        with redirect_stdout(io.StringIO()):
            bad, report = pr.run(FakeOracle(), [32])
        self.assertEqual(bad, 0)
        self.assertEqual(len(report["resinfo"]), len(pr.resinfo_cases()))

    def test_a_gpu_that_writes_1_for_the_2d_depth_is_caught(self):
        with redirect_stdout(io.StringIO()) as out:
            bad, _ = pr.run(FakeOracle(depth_2d=1), [32])
        self.assertGreater(bad, 0)
        self.assertIn("MISMATCH resinfo", out.getvalue())

    def test_main_exit_codes(self):
        with redirect_stdout(io.StringIO()):
            with mock.patch.object(runner, "open_oracle", return_value=FakeOracle()):
                self.assertEqual(pr.main(["--waves", "32"]), 0)
            with mock.patch.object(runner, "open_oracle", return_value=FakeOracle(depth_2d=1)):
                self.assertEqual(pr.main(["--waves", "32"]), 1)
            with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no GPU")):
                self.assertEqual(pr.main([]), 77)
            other = FakeOracle()
            other.arch = "gfx1150"
            with mock.patch.object(runner, "open_oracle", return_value=other):
                self.assertEqual(pr.main([]), 77)

    def test_json_output_creates_its_parent_and_reports_write_errors(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner, "open_oracle", return_value=FakeOracle()):
            report = Path(tmp) / "new" / "report.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(pr.main(["--waves", "32", "--json", str(report)]), 0)
            self.assertTrue(report.is_file())

    def test_a_report_that_cannot_be_written_still_shows_the_result_and_exits_2(self):
        # Exit 1 means a result off the rules; an unwritable --json must not look like one, nor hide the summary.
        for oracle, summary in ((FakeOracle(), r"follow the rules, 0 do not"),
                                (FakeOracle(depth_2d=1), r"follow the rules, [1-9]\d* do not")):
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(runner, "open_oracle", return_value=oracle):
                report = Path(tmp) / "directory"
                report.mkdir()
                out, error = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(error):
                    self.assertEqual(pr.main(["--waves", "32", "--json", str(report)]), 2)
                self.assertRegex(out.getvalue(), summary)
                self.assertIn(str(report), error.getvalue())
                self.assertIn("cannot write", error.getvalue())


class Hardware(unittest.TestCase):
    def test_on_the_gpu(self):
        try:
            oracle = runner.open_oracle()
        except runner.Unavailable as e:
            self.skipTest(f"hardware oracle unavailable: {e}")
        try:
            if not oracle.arch.startswith("gfx10"):
                self.skipTest(f"{oracle.arch} is not GFX10")
            with redirect_stdout(io.StringIO()):
                bad, _ = pr.run(oracle, [32, 64])
            self.assertEqual(bad, 0)
        finally:
            oracle.close()


if __name__ == "__main__":
    unittest.main()
