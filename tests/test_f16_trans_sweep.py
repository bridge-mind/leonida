"""tools/f16_trans_sweep.py: the f16 reference and its rounding, which input each kernel thread computes and where it
stores it, the classification, and the exit-77 contract. The last test runs on the hardware when the oracle tools
are present and is skipped otherwise. Standard library only."""
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import f16_trans_sweep as sw  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class Reference(unittest.TestCase):
    def test_self_test(self):
        self.assertEqual(sw.self_test(), 0)

    def test_rounding_to_f16(self):
        self.assertEqual(sw.from_dec(Decimal(1) + Decimal(2) ** -11), 0x3C00)                  # tie: to even
        self.assertEqual(sw.from_dec(Decimal(1) + Decimal(2) ** -11 * 3), 0x3C02)              # tie: to even
        self.assertEqual(sw.from_dec(Decimal(1) + Decimal(2) ** -11 + Decimal(2) ** -40), 0x3C01)
        self.assertEqual(sw.from_dec(-(Decimal(1) + Decimal(2) ** -10 * Decimal("0.99")), rtz=True), 0xBC00)
        self.assertEqual(sw.from_dec(Decimal(2) ** -14 - Decimal(2) ** -25), 0x0400)           # rounds up to normal
        self.assertEqual(sw.from_dec(Decimal(-70000)), 0xFC00)
        self.assertEqual(sw.from_dec(Decimal(-70000), rtz=True), 0xFBFF)

    def test_undefined_inputs_are_the_ones_spirv_leaves_undefined(self):
        undef = {op: [h for h, r in enumerate(sw.reference(op)) if r is None] for op in ("v_rcp_f16", "v_sqrt_f16")}
        self.assertEqual(undef["v_rcp_f16"], [0x0000, 0x8000])
        self.assertEqual(len(undef["v_sqrt_f16"]), 31744)                 # every negative but -0, -inf included
        self.assertNotIn(0x8000, undef["v_sqrt_f16"])
        self.assertIn(0xFC00, undef["v_sqrt_f16"])
        self.assertEqual(sw.exact("v_log_f16", 0x8000), "undef")
        self.assertEqual(sw.exact("v_rsq_f16", 0x0000), "undef")
        self.assertEqual(sw.exact("v_exp_f16", 0x7E00), "nan")

    def test_the_sine_reduces_in_turns_and_the_f32_product_does_not(self):
        # x = 1000.5: sin(2 pi x) = sin(pi) = 0, but 6.2831855 * x in f32 is 6286.3726, about 1.6e-3 rad off
        h = 0x63D1
        self.assertEqual(float(sw.to_dec(h)), 1000.5)
        self.assertEqual(sw.to_bits(sw.exact("v_sin_f16", h)), 0)
        self.assertNotEqual(sw.to_bits(sw.product_turns("v_sin_f16", h)), 0)


class Kernels(unittest.TestCase):
    def test_every_input_is_computed_once_and_stored_at_its_own_half(self):
        seen = {}
        for j in range(65536 // sw.PER_KERNEL):
            base = j * sw.PER_KERNEL
            lines = sw.kernel_lines("v_exp_f16", base)
            self.assertEqual(lines[1], f"v_add_nc_u32 v1, 0x{base:x}, v1")
            for t in range(sw.THREADS):
                for k in range(sw.PER_THREAD):
                    word = base // 2 + 4 * t + k // 2        # global_store_b128 at 16 * t, v8 + k // 2
                    seen[word * 2 + k % 2] = base + 8 * t + k
        self.assertEqual(sorted(seen), list(range(65536)))
        self.assertTrue(all(seen[i] == i for i in seen))

    def test_kernels_assemble_for_rdna2(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        kernels = [asm.Kernel(sw.kernel_lines(op, 0, "gfx1030"), w, sw.THREADS, sw.PER_KERNEL // 2, raw=True,
                              mode=sw.MODES["keep16"]) for op in sw.OPS for w in (32,)]
        text, names, _ = asm.module_text(kernels, "gfx1030", wave=32)
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "k.s"
            s.write_text(text, encoding="utf-8")
            r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1030", "-c", str(s), "-o",
                                str(Path(d) / "k.o")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:800])
        self.assertEqual(len(names), len(sw.OPS))
        self.assertIn(".amdhsa_float_denorm_mode_16_64 3", text)
        self.assertIn("global_store_dwordx4 v24, v[8:11], s[2:3]", text)


class Classify(unittest.TestCase):
    def setUp(self):
        self.ref = sw.reference("v_rcp_f16")

    def test_the_rule_itself_is_all_exact(self):
        hw = [r if r is not None else 0x7E00 for r in self.ref]
        c, worst = sw.classify("v_rcp_f16", hw, self.ref)
        self.assertEqual(c["exact"] + c["nan_ok"] + c["undefined"], 65536)
        self.assertEqual(c["undefined"], 2)
        self.assertEqual(worst, [])

    def test_each_kind_of_difference_is_counted_apart(self):
        hw = [r if r is not None else 0 for r in self.ref]
        hw[0x0000], hw[0x8000] = 0x7C00, 0xFC00     # undefined in the rule: the hardware's +-inf
        hw[0x3C00] = 0x3C01                       # rcp(1) off by 1 ulp
        hw[0x4000] = 0x3802                       # rcp(2) off by 2 ulp
        hw[0x03FF] = 0x7C00                       # rcp(largest denormal) as the hardware's rcp(+0)
        hw[0x7BFF] = 0x0000                       # rcp(65504) = 2^-16 * (1 + ...) is a denormal, flushed
        hw[0x7E00] = 0x3C00                       # NaN in, a number out
        hw[0x7D00] = 0x7F00                       # a signalling NaN quieted
        hw[0x7C01] = 0x7C01                       # a signalling NaN passed through
        c, worst = sw.classify("v_rcp_f16", hw, self.ref)
        self.assertTrue(sw.is_denorm(self.ref[0x7BFF]))
        self.assertEqual((c["ulp1"], c["ulp_more"], c["max_ulp"]), (1, 1, 2))
        self.assertEqual((c["denorm_in_as_zero"], c["denorm_out_flushed"], c["other"]), (1, 1, 1))
        self.assertEqual((c["snan_quieted"], c["snan_kept"], c["snan_other"]), (1, 1, 1020))
        self.assertEqual({w[0] for w in worst}, {0x4000, 0x7E00})


class Cli(unittest.TestCase):
    def test_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "f16_trans_sweep.py"), "--ops", "v_exp_f16"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_unknown_names_are_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "f16_trans_sweep.py"), "--ops", "v_exp_f32"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("v_exp_f32", r.stderr)

    def test_empty_and_malformed_selections_fail_before_opening_oracle(self):
        cases = [
            (["--ops", ""], "--ops"),
            (["--modes", ","], "--modes"),
            (["--waves", ""], "--waves"),
            (["--waves", "32,nope"], "--waves"),
            (["--waves", "16"], "16"),
        ]
        for argv, message in cases:
            with self.subTest(argv=argv), mock.patch.object(runner, "open_oracle") as open_oracle:
                error = io.StringIO()
                with self.assertRaises(SystemExit) as raised, redirect_stderr(error):
                    sw.main(argv)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())
                open_oracle.assert_not_called()

    def test_json_empty_string_exits_1_before_opening_oracle(self):
        err = io.StringIO()
        with mock.patch.object(runner, "open_oracle") as open_oracle, redirect_stderr(err):
            rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", ""])
        self.assertEqual(rc, 1)
        self.assertIn("error:", err.getvalue())
        open_oracle.assert_not_called()

    def test_json_existing_directory_exits_1_before_sweep_or_probe(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.object(runner, "open_oracle") as open_oracle, redirect_stderr(err):
                rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", d])
            self.assertEqual(rc, 1)
            self.assertIn("error:", err.getvalue())
            open_oracle.assert_not_called()

            env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
            r = subprocess.run([sys.executable, "-I", str(TOOLS / "f16_trans_sweep.py"),
                                "--ops", "v_exp_f16", "--json", d],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("error:", r.stderr)

    def test_json_creates_nested_parent_and_writes_lf_endings(self):
        ref = [0] * 65536
        fake_oracle = mock.MagicMock(device="fake-gpu", arch="gfx1030")
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "nested" / "deep" / "report.json"
            self.assertFalse(out_path.parent.exists())
            out = io.StringIO()
            err = io.StringIO()
            with mock.patch.object(runner, "open_oracle", return_value=fake_oracle), \
                    mock.patch.object(sw, "check_modes"), \
                    mock.patch.object(sw, "reference", return_value=ref), \
                    mock.patch.object(sw, "sweep", return_value=list(ref)), \
                    redirect_stdout(out), redirect_stderr(err):
                rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", str(out_path)])
            self.assertEqual(rc, 0, err.getvalue())
            self.assertEqual(err.getvalue(), "")
            self.assertTrue(out_path.is_file())
            raw = out_path.read_bytes()
            self.assertIn(b"\n", raw)
            self.assertNotIn(b"\r\n", raw)
            data = json.loads(raw.decode("utf-8"))
            self.assertEqual(data["device"], "fake-gpu")
            self.assertEqual(data["arch"], "gfx1030")
            self.assertIn("keep16", data["ops"]["v_exp_f16"])
            fake_oracle.close.assert_called_once()

    def test_json_blocked_by_regular_file_parent_exits_1(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "blocker"
            blocker.write_text("not a dir\n", encoding="utf-8", newline="\n")
            for target in (blocker / "report.json", blocker / "sub" / "report.json"):
                err = io.StringIO()
                with mock.patch.object(runner, "open_oracle") as open_oracle, redirect_stderr(err):
                    rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", str(target)])
                self.assertEqual(rc, 1)
                self.assertIn("error:", err.getvalue())
                open_oracle.assert_not_called()

            env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
            r = subprocess.run([sys.executable, "-I", str(TOOLS / "f16_trans_sweep.py"),
                                "--ops", "v_exp_f16", "--json", str(blocker / "sub" / "report.json")],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("error:", r.stderr)

    def test_json_write_oserror_reports_cleanly_and_exits_1(self):
        ref = [0] * 65536
        fake_oracle = mock.MagicMock(device="fake-gpu", arch="gfx1030")
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "nested" / "report.json"
            with mock.patch.object(runner, "open_oracle", return_value=fake_oracle), \
                    mock.patch.object(sw, "check_modes"), \
                    mock.patch.object(sw, "reference", return_value=ref), \
                    mock.patch.object(sw, "sweep", return_value=list(ref)), \
                    mock.patch.object(sw.os, "makedirs", side_effect=PermissionError(13, "Permission denied")):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", str(out_path)])
            self.assertEqual(rc, 1)
            self.assertIn("error:", err.getvalue())
            self.assertIn("Permission denied", err.getvalue())

            with mock.patch.object(runner, "open_oracle", return_value=fake_oracle), \
                    mock.patch.object(sw, "check_modes"), \
                    mock.patch.object(sw, "reference", return_value=ref), \
                    mock.patch.object(sw, "sweep", return_value=list(ref)), \
                    mock.patch("builtins.open", side_effect=OSError(28, "No space left on device")):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    rc = sw.main(["--ops", "v_exp_f16", "--modes", "keep16", "--waves", "32", "--json", str(out_path)])
            self.assertEqual(rc, 1)
            self.assertIn("error:", err.getvalue())
            self.assertIn("No space left on device", err.getvalue())


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None
        why = asm.rdna23_skip_reason(cls.oracle.arch)
        if why:
            cls.oracle.close()
            raise unittest.SkipTest(why)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_exp_sweep_is_the_same_in_both_wave_sizes_and_exact_on_integers(self):
        sw.check_modes(self.oracle, ["keep16"])
        w32 = sw.sweep(self.oracle, "v_exp_f16", "keep16", 32)
        w64 = sw.sweep(self.oracle, "v_exp_f16", "keep16", 64)
        self.assertEqual(w32, w64)
        for x, want in ((0x0000, 0x3C00), (0x3C00, 0x4000), (0x4000, 0x4400), (0xBC00, 0x3800), (0x7C00, 0x7C00),
                        (0xFC00, 0x0000)):
            self.assertEqual(w32[x], want, f"exp2(0x{x:04x})")


if __name__ == "__main__":
    unittest.main()
