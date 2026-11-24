"""tools/image_d16_probe.py: the rules, the T# words and kernels per dimension, the comparison and its exit codes
(through a fake oracle that plays the GPU), and the exit-77 contract. The last class runs on the hardware when the
oracle tools are present and the GPU is GFX10, and is skipped otherwise. Standard library only."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import image_d16_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class Rules(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)

    def test_the_edges_separate_saturation_from_truncation(self):
        self.assertEqual(pr.rules(0x00010000), {"trunc": 0x0000, "sat_u": 0xFFFF, "sat_s": 0x7FFF})
        self.assertEqual(pr.rules(0xFFFF7FFF), {"trunc": 0x7FFF, "sat_u": 0xFFFF, "sat_s": 0x8000})
        self.assertEqual(pr.rules(0x0000FFFF), {"trunc": 0xFFFF, "sat_u": 0xFFFF, "sat_s": 0x7FFF})
        self.assertEqual(pr.rules(0xFFFFFFFF), {"trunc": 0xFFFF, "sat_u": 0xFFFF, "sat_s": 0xFFFF})


class Descriptors(unittest.TestCase):
    def test_type_and_depth_word_per_dimension(self):
        for dim, kind, word4 in (("1d", 8, 63), ("2d", 9, 63), ("2d_array", 13, 2), ("3d", 10, 2)):
            t = pr.image_words(dim, "r32_uint")
            self.assertEqual(t[3] >> 28, kind, dim)
            self.assertEqual(t[4], word4, dim)                 # pitch - 1 for 1D/2D, last layer or depth - 1
            self.assertEqual(t[2] >> 31, 1, dim)               # RESOURCE_LEVEL

    def test_two_channel_formats_keep_256_byte_rows(self):
        self.assertEqual(pr.geometry("2d", "r32g32_sint"), (32, 2, 1, 64))
        self.assertEqual(pr.geometry("3d", "r32_sint"), (64, 2, 3, 384))

    def test_each_kernel_reads_with_its_own_dimension(self):
        lines = pr.kernel_lines("2d_array", "r32g32_uint")
        loads = [ln for ln in lines if ln.startswith("image_load")]
        self.assertEqual(loads, [
            "image_load v8, v[2:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm",
            "image_load v9, v[2:4], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY unorm d16",
            "image_load_mip v10, v[2:5], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY unorm d16",
            "image_load v11, v[2:4], s[8:15] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY unorm"])
        self.assertIn("image_load v8, v2, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm", pr.kernel_lines("1d", "r32_sint"))

    def test_kernels_assemble_for_gfx1036(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        kernels = []
        for uid, (d, f) in enumerate((d, f) for d in pr.DIMS for f in pr.FORMATS):
            n = pr.geometry(d, f)[3]
            kernels.append(asm.Kernel(pr.kernel_lines(d, f, uid), 32, n, n * 4, raw=True))
        text, _, _ = asm.module_text(kernels, "gfx1036", wave=32)
        with tempfile.TemporaryDirectory() as tmp:
            s = Path(tmp) / "k.s"
            s.write_text(text, encoding="utf-8")
            r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1036", "-c", str(s), "-o",
                                str(Path(tmp) / "k.o")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:800])
        self.assertIn("global_store_dwordx4 v6, v[8:11], s[2:3]", text)


class FakeOracle:
    """Plays the GPU: finds each kernel's dimension and format and answers with a rule's halves."""
    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, rule="sat", bad_control=False, mip_off=False):
        self.rule, self.bad_control, self.mip_off = rule, bad_control, mip_off

    def run(self, kernels, words=4):
        by_code = {code: f for f, (code, _, _) in pr.FORMATS.items()}
        by_dim = {name: d for d, (_, name, _, _) in pr.DIMS.items()}
        out = []
        for k in kernels:
            word1 = int(next(ln for ln in k.lines if ln.startswith("s_or_b32 s9")).split("0x")[1], 16)
            fmt = by_code[(word1 >> 20) & 0x1FF]
            dim = by_dim[next(ln for ln in k.lines if ln.startswith("image_load")).split("dim:")[1].split()[0]]
            signed = pr.FORMATS[fmt][2]
            rule = ("sat_s" if signed else "sat_u") if self.rule == "sat" else self.rule
            res = []
            for i, texel in enumerate(pr.texel_values(fmt, pr.geometry(dim, fmt)[3])):
                h = [pr.rules(v)[rule] for v in texel]
                d16 = h[0] | ((h[1] << 16) if len(h) == 2 else pr.SEED & 0xFFFF0000)
                mip = d16 ^ 1 if self.mip_off and i == 5 else d16
                x = texel[0] ^ 0x100 if self.bad_control and i == 3 else texel[0]
                res += [x, d16, mip, texel[1] if len(texel) == 2 else pr.SEED]
            out.append(res)
        return out

    def close(self):
        pass


def run_main(oracle, *args):
    buf = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf):
        rc = pr.main(list(args))
    return rc, buf.getvalue()


class Compare(unittest.TestCase):
    def test_saturating_hardware_matches_its_rule_on_every_case(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle(), "--json", j)
            with open(j, encoding="utf-8") as f:
                rep = json.load(f)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(rep["cases"]), 16)
        e = rep["cases"]["3d r32_sint"]
        self.assertEqual(e["forms"]["image_load"]["match"]["sat_s"], 384)
        self.assertLess(e["forms"]["image_load"]["match"]["trunc"], 384)
        self.assertEqual(e["forms"]["image_load_mip"]["high_half"], {"kept": 384})

    def test_truncating_hardware_is_a_finding_not_a_failure(self):
        rc, out = run_main(FakeOracle("trunc"), "--dims", "1d", "--formats", "r32_uint")
        self.assertEqual(rc, 0, out)
        self.assertIn("trunc 64/64", out)

    def test_a_wrong_control_word_is_a_harness_error(self):
        rc, out = run_main(FakeOracle(bad_control=True), "--dims", "2d_array", "--formats", "r32_uint")
        self.assertEqual(rc, 2, out)
        self.assertIn("texel 3: x read back", out)

    def test_load_and_load_mip_disagreeing_exits_1(self):
        rc, out = run_main(FakeOracle(mip_off=True), "--dims", "3d", "--formats", "r32g32_sint")
        self.assertEqual(rc, 1, out)

    def test_not_gfx10_exits_77(self):
        o = FakeOracle()
        o.arch = "gfx1150"
        rc, out = run_main(o)
        self.assertEqual(rc, 77, out)


class Cli(unittest.TestCase):
    def test_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "image_d16_probe.py"), "--dims", "1d"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_unknown_dimension_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "image_d16_probe.py"), "--dims", "cube"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("cube", r.stderr)

    def test_bad_json_paths_rejected_before_open_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing_dir = os.path.join(tmp, "dir")
            os.mkdir(existing_dir)
            file_ancestor = os.path.join(tmp, "file_ancestor")
            with open(file_ancestor, "w", encoding="utf-8") as f:
                f.write("x")

            bad_paths = [
                "",
                "   ",
                "bad\x00out.json",
                existing_dir,
                os.path.join(tmp, "trailing") + os.sep,
                os.path.join(file_ancestor, "sub", "out.json"),
            ]
            for bad in bad_paths:
                with self.subTest(json_path=bad):
                    stderr = io.StringIO()
                    with mock.patch.object(runner, "open_oracle") as open_mock, \
                         mock.patch.object(sys, "stderr", stderr):
                        rc = pr.main(["--json", bad])
                    self.assertEqual(rc, 2)
                    self.assertIn("error:", stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
                    open_mock.assert_not_called()

    def test_valid_nested_json_does_not_create_dirs_when_oracle_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested_dir = os.path.join(tmp, "a", "b")
            json_path = os.path.join(nested_dir, "report.json")
            stdout = io.StringIO()
            with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no gpu")), \
                 mock.patch.object(sys, "stdout", stdout):
                rc = pr.main(["--json", json_path])
            self.assertEqual(rc, 77)
            self.assertFalse(os.path.exists(os.path.join(tmp, "a")))

    def test_write_json_report_creates_parents_and_writes_lf_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "nested", "dir", "report.json")
            payload = {"device": "fake", "arch": "gfx1036", "cases": {}}
            err = pr.write_json_report(out_path, payload)
            self.assertIsNone(err)
            with open(out_path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw.decode("utf-8")), payload)

    def test_write_json_oserror_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.json")
            with mock.patch("builtins.open", side_effect=PermissionError("denied")):
                err = pr.write_json_report(out_path, {"ok": True})
            self.assertIsNotNone(err)
            self.assertIn("cannot write --json", err)

            stderr = io.StringIO()
            with mock.patch("builtins.open", side_effect=OSError("disk full")), \
                 mock.patch.object(sys, "stderr", stderr):
                rc, _ = run_main(FakeOracle(), "--dims", "1d", "--formats", "r32_uint", "--json", out_path)
            self.assertEqual(rc, 2)
            self.assertIn("error:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None
        if asm.family(cls.oracle.arch) != "gfx10":
            cls.oracle.close()
            raise unittest.SkipTest(f"{cls.oracle.arch} is not GFX10")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_controls_hold_on_every_dimension(self):
        cases = [(d, "r32_sint") for d in pr.DIMS]
        kernels = []
        for uid, (d, f) in enumerate(cases):
            n = pr.geometry(d, f)[3]
            kernels.append(asm.Kernel(pr.kernel_lines(d, f, uid), 32, n, n * 4, raw=True))
        for (d, f), got in zip(cases, self.oracle.run(kernels)):
            self.assertIsInstance(got, list, got)
            e = pr.classify(d, f, got)
            self.assertEqual(e["control"], [], d)
            self.assertEqual(e["forms_differ"], 0, d)


if __name__ == "__main__":
    unittest.main()
