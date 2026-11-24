"""scripts/run_gpu_tests.py: how it judges the output of a compute "dump" test (no device needed)."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt
import check   # runtime/test/shaders, on the path once run_gpu_tests is imported

WANT = [55, 48]
DEVICE_32 = "device: X (subgroup 32..32)\n"
DEVICE_64 = "device: X (subgroup 32..64)\n"
CONFIRMED = "compute subgroup size: 64\n"


class BarycentricFallbackSelection(unittest.TestCase):
    def setUp(self):
        rgt._BARYCENTRIC_FALLBACK.clear()

    def test_env_override(self):
        with patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": "1"}, clear=False):
            self.assertTrue(rgt.bary_fallback())
        with patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": "0"}, clear=False):
            proc = SimpleNamespace(stdout='{"barycentric_fallback": true}\n')
            with patch.object(rgt.subprocess, "run", return_value=proc):
                self.assertFalse(rgt.bary_fallback())

    def test_anv_probe_is_selected_when_env_is_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEONIDA_BARY_FALLBACK", None)
            proc = SimpleNamespace(stdout='{"barycentric_fallback": true}\n')
            with patch.object(rgt.subprocess, "run", return_value=proc):
                self.assertTrue(rgt.bary_fallback())

    def test_env_override_beats_the_probe(self):
        """#5933: =0 brings back native barycentrics on a device the probe flags (gfx90c on AMD's Windows driver),
        and =1 does not need the probe at all."""
        proc = SimpleNamespace(stdout='{"barycentric_fallback": true, "vendor_id": 4098, "device_id": 5607, '
                                      '"driver_id": 1}\n')
        with patch.object(rgt.subprocess, "run", return_value=proc) as run:
            with patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": "0"}, clear=False):
                self.assertIs(rgt.bary_fallback(), False)
            with patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": "1"}, clear=False):
                self.assertIs(rgt.bary_fallback(), True)
            run.assert_not_called()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LEONIDA_BARY_FALLBACK", None)
                self.assertIs(rgt.bary_fallback(), True)
            run.assert_called_once()

    def test_probe_failure_is_not_a_fallback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEONIDA_BARY_FALLBACK", None)
            with patch.object(rgt.subprocess, "run", side_effect=OSError("no Vulkan")):
                self.assertIs(rgt.bary_fallback(), False)


def dump(words, pad=8):
    return "out[0..7] = " + " ".join(str(w) for w in (list(words) + [0] * pad)[:8]) + "\n"


def judge(rc, msg, **kw):
    kw = {"wave64": False, "partial": False, "emulated": False, **kw}
    return rgt.judge_cs_dump(rc, msg, WANT, **kw)


class DumpWords(unittest.TestCase):
    def test_eight_decimal_words(self):
        self.assertEqual(rgt.dump_words("noise\n" + dump(WANT) + "tail\n"), WANT + [0] * 6)

    def test_anything_else_is_not_a_dump(self):
        for line in ("out[0..7] = 55 48\n", "out[0..7] = 55 48 BAD 0 0 0 0 0\n", "out[0..7] = 1 BAD 3\n",
                     "out[0..7] = 55 48 0 0 0 0 0\n", "out[0..7] = 55 48 0 0 0 0 0 0 0\n",
                     "out[0..7] = 55 48 0 0 0 0 0 0 x\n", "out[0..7] = \n", "no dump here\n"):
            with self.subTest(line=line):
                self.assertIsNone(rgt.dump_words(line))


class PackedFormatFixture(unittest.TestCase):
    def test_filtered_suite_counts_a_packed_store_mismatch(self):
        name = "cs_fmt_1010102"
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")

            def translate(_cmd, env=None, gpu=True):
                (Path(tmp) / "spirv" / f"{name}.spv").touch()
                return 0, ""

            output = io.StringIO()
            problem = "word 15: expected 0xffe00003, got 0xe00003ff"
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=translate),
                  patch.object(rgt, "fmt_1010102_test", return_value=problem) as check_fmt,
                  contextlib.redirect_stdout(output)):
                self.assertEqual(rgt.run_suite(args, tmp), 1)
            check_fmt.assert_called_once()
            self.assertIn(problem, output.getvalue())
            self.assertIn("0/1 passed", output.getvalue())


class FrameReadbackTest(unittest.TestCase):
    def test_only_a_frame_written_by_this_run_can_pass(self):
        name = "ps_ops"
        pixels = struct.pack("<4I", 0x48804300, 5, 7, 0xBFFFFFFC) * 4096
        for existing, writes_frame, expected in ((True, False, 1), (False, False, 1), (True, True, 0)):
            with self.subTest(existing=existing, writes_frame=writes_frame), tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp, "raw", f"{name}.raw")
                if existing:
                    raw.parent.mkdir()
                    raw.write_bytes(pixels)
                args = SimpleNamespace(filter="ps_ops.asm", build=tmp, llvm_mc="unused")

                def launch(cmd, **kwargs):
                    if cmd[:3] == [sys.executable, "-I", rgt.TRANSLATOR]:
                        (Path(tmp) / "spirv" / f"{name}.spv").touch()
                    elif writes_frame:
                        raw.write_bytes(pixels)
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"),
                      patch.object(rgt, "bary_fallback", return_value=False),
                      patch.object(rgt, "f16_interp_fallback", return_value=False),
                      patch.object(rgt.subprocess, "run", side_effect=launch),
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), expected, output.getvalue())
                if expected:
                    self.assertIn("missing readback", output.getvalue())
                    self.assertFalse(raw.exists())
                else:
                    self.assertEqual(raw.read_bytes(), pixels)


class TextureTestOnASmallPushConstantDevice(unittest.TestCase):
    """vk_texture_test exits 77 on a device with 128 push constant bytes: a SKIP, not a FAIL."""
    UNSUPPORTED_MSG = "unsupported: need 144 push constant bytes, device has 128\n"

    def run_one(self, name, require_pass=False):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused", require_pass=require_pass)

            def launch(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
                if cmd[:3] == [sys.executable, "-I", rgt.TRANSLATOR]:
                    (Path(tmp) / "spirv" / f"{name}.spv").touch()
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                return SimpleNamespace(returncode=77, stdout="", stderr=self.UNSUPPORTED_MSG)

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt.subprocess, "run", side_effect=launch),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                rc = rgt.run_suite(args, tmp)
            return rc, output.getvalue()

    def test_bindless_texture_and_gather_tests_skip(self):
        for name in ("ps_cube_sample", "ps_texture_sample_d16", "ps_gather_wrap", "ps_array_resinfo"):
            with self.subTest(name=name):
                rc, out = self.run_one(name)
                self.assertEqual(rc, 0, out)
                self.assertIn("SKIP (unsupported: need 144 push constant bytes, device has 128)", out)
                self.assertIn("0/1 passed, 1 skipped", out)
                self.assertNotIn("no runnable", out)

    def test_require_pass_rejects_a_skip_only_run(self):
        rc, out = self.run_one("ps_gather_wrap", require_pass=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("0/1 passed, 1 skipped", out)
        self.assertIn("--require-pass: no GPU tests passed", out)


class BindlessTestOnASmallPushConstantDevice(unittest.TestCase):
    """vk_bindless_test (ps_tex) exits 77 below its 144 push constant bytes (TEST-149): a SKIP with that reason."""

    def run_ps_tex(self, rc, msg):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter="ps_tex.asm", build=tmp, llvm_mc="unused")   # not ps_texoff etc.

            def run(cmd, env=None, gpu=True):
                if Path(cmd[0]).stem == "vk_bindless_test":
                    return rc, msg
                (Path(tmp) / "spirv" / "ps_tex.spv").touch()   # rdna2spirv.py "translating" the shader
                return 0, ""

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=run),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                suite_rc = rgt.run_suite(args, tmp)
            return suite_rc, output.getvalue()

    def test_too_few_push_constant_bytes_skip(self):
        rc, out = self.run_ps_tex(77, "device: X\nunsupported: need 144 push constant bytes, device has 128")
        self.assertEqual(rc, 0, out)
        self.assertIn("SKIP (unsupported: need 144 push constant bytes, device has 128)", out)
        self.assertIn("0/1 passed, 1 skipped", out)

    def test_other_failures_still_fail(self):
        rc, out = self.run_ps_tex(1, "device: X\nps_tex: 12 pixels differ")
        self.assertEqual(rc, 1, out)
        self.assertIn("0/1 passed", out)
        self.assertNotIn("skipped", out)


class StoreHelperUnsupported(unittest.TestCase):
    """vk_store_test exits 77 when the device lacks a feature (TEST-469): a SKIP that quotes why, not a FAIL."""

    def run_store(self, shader, rc, msg):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=f"{shader}.asm", build=tmp, llvm_mc="unused")

            def run(cmd, env=None, gpu=True):
                if Path(cmd[0]).stem == "vk_store_test":
                    return rc, msg
                (Path(tmp) / "spirv" / f"{shader}.spv").touch()   # rdna2spirv.py "translating" the shader
                return 0, ""

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=run),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                suite_rc = rgt.run_suite(args, tmp)
            return suite_rc, output.getvalue()

    def test_unsupported_is_a_skip_with_its_reason(self):
        for shader in ("cs_imgstore", "cs_imgload", "cs_imgload_d16"):
            with self.subTest(shader=shader):
                rc, out = self.run_store(shader, 77, "unsupported: shaderStorageImageWriteWithoutFormat=0\ndevice: X")
                self.assertEqual(rc, 0, out)
                self.assertIn("SKIP (unsupported: shaderStorageImageWriteWithoutFormat=0)", out)
                self.assertIn("0/1 passed, 1 skipped", out)

    def test_other_failures_still_fail(self):
        rc, out = self.run_store("cs_imgstore", 1, "device: X\nFAILED: texel 3 is 0x0")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED: texel 3 is 0x0", out)
        self.assertNotIn("skipped", out)


INT64 = "unsupported: m.spv declares Int64 and the device has no shaderInt64\n"
INT64_SKIP = "SKIP (unsupported: m.spv declares Int64 and the device has no shaderInt64)"
# judge_cs_dump quotes the reason without the "unsupported:" prefix (87da93277, which wins over the wave64 skip).
INT64_JUDGE_SKIP = "SKIP (m.spv declares Int64 and the device has no shaderInt64)"


class ComputeFeatureUnsupported(unittest.TestCase):
    """vk_compute_test exits 77 with an "unsupported: ..." line when the module needs a feature the device lacks
    (Int64, Float64, OpFmaKHR): a SKIP that quotes it in every compute test, not only under --require-wave64."""

    def test_judge_skips_a_named_feature_in_every_mode(self):
        self.assertEqual(judge(77, DEVICE_32 + INT64), (None, INT64_JUDGE_SKIP))
        self.assertEqual(judge(77, DEVICE_32 + INT64, partial=True), (None, INT64_JUDGE_SKIP))
        # A 64-wide device without Int64: the feature is the cause, not the subgroup width.
        self.assertEqual(judge(77, DEVICE_64 + INT64, wave64=True), (None, INT64_JUDGE_SKIP))

    def test_judge_keeps_the_wave64_wording_and_fails_a_bare_77(self):
        self.assertEqual(judge(77, DEVICE_32 + "full 64-wide compute subgroups unavailable\n", wave64=True),
                         (None, "SKIP (no 64-wide compute subgroup on this device)"))
        ok, _ = judge(77, DEVICE_32 + "something else went wrong\n")
        self.assertIs(ok, False)

    def run_compute(self, shader, rc, msg):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=f"{shader}.asm", build=tmp, llvm_mc="unused")

            def run(cmd, env=None, gpu=True):
                if Path(cmd[0]).stem == "vk_compute_test":
                    calls.append(cmd)
                    return rc, msg
                (Path(tmp) / "spirv" / f"{shader}.spv").touch()
                return 0, ""

            output = io.StringIO()
            with (patch.dict(rgt.os.environ, {"LEONIDA_W64_EMULATE": ""}, clear=False),
                  patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=run),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                suite_rc = rgt.run_suite(args, tmp)
            return suite_rc, output.getvalue(), calls

    def test_every_compute_call_site_skips(self):
        # One shader per way the runner reads vk_compute_test: judge_cs_dump (cs_bounds), its own checks (cs_lds,
        # cs_workgroup_yz, cs_heap_init), one dispatch per input (cs_cvt_f32_sat), cs_f64_add and wavemask_check.
        for shader in ("cs_bounds", "cs_lds", "cs_workgroup_yz", "cs_heap_init", "cs_cvt_f32_sat", "cs_f64_add",
                       "cs_wavemask"):
            with self.subTest(shader=shader):
                rc, out, _ = self.run_compute(shader, 77, "device: X (subgroup 32..32)\n" + INT64)
                self.assertEqual(rc, 0, out)
                self.assertIn(INT64_SKIP, out)
                self.assertIn("0/1 passed, 1 skipped", out)

    def test_one_refused_input_stops_the_per_input_loop(self):
        _, _, calls = self.run_compute("cs_cvt_f32_sat", 77, INT64)
        self.assertEqual(len(calls), 1)

    def test_check_py_helpers_skip_a_named_feature(self):
        # check.py's own runner and the cs_fmt_* rows go through dump_test/compute_test, not run_cmd.
        with tempfile.TemporaryDirectory() as tmp, patch.object(check, "run", return_value=(77, INT64)):
            for result in (check.dump_test(tmp, tmp, "cs_fmt_1010102", [1], 1),
                           check.compute_test(tmp, tmp, "cs_bounds", "1", "dump")):
                self.assertIsInstance(result, check.Skip)
                self.assertEqual(result, INT64_SKIP)
        with tempfile.TemporaryDirectory() as tmp, patch.object(check, "run", return_value=(77, "no reason")):
            self.assertEqual(check.dump_test(tmp, tmp, "cs_fmt_1010102", [1], 1), "exit 77: no reason")

    def test_a_check_py_skip_counts_as_skipped(self):
        # The cs_fmt_* rows record (name, not problem, problem): a Skip there is a skip, not a failure.
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter="cs_fmt_1010102.asm", build=tmp, llvm_mc="unused")

            def translate(_cmd, env=None, gpu=True):
                (Path(tmp) / "spirv" / "cs_fmt_1010102.spv").touch()
                return 0, ""

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=translate),
                  patch.object(rgt, "fmt_1010102_test", return_value=check.Skip(INT64_SKIP)),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                rc = rgt.run_suite(args, tmp)
        self.assertEqual(rc, 0, output.getvalue())
        self.assertIn("SKIP  cs_fmt_1010102", output.getvalue())
        self.assertIn("0/1 passed, 1 skipped", output.getvalue())

    def test_a_bare_77_still_fails(self):
        for shader in ("cs_bounds", "cs_lds", "cs_heap_init"):
            with self.subTest(shader=shader):
                rc, out, _ = self.run_compute(shader, 77, "device: X (subgroup 32..32)\nsomething else went wrong")
                self.assertEqual(rc, 1, out)
                self.assertNotIn("skipped", out)


class FilterWithoutRunnableCases(unittest.TestCase):
    """A filter selecting only an input shader (cs_bench_mix) must not report 0/0 passed with exit 0."""

    def run_filter(self, name, translate_rc=0, spv=True):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")

            def translate(_cmd, env=None, gpu=True):
                if spv:
                    (Path(tmp) / "spirv" / f"{name}.spv").touch()
                return translate_rc, ""

            out, err = io.StringIO(), io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()) as asm,
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=translate) as run,
                  patch.object(rgt, "fmt_1010102_test", return_value=None),
                  contextlib.redirect_stdout(out), contextlib.redirect_stderr(err)):
                rc = rgt.run_suite(args, tmp)
            return rc, out.getvalue(), err.getvalue(), asm, run

    def test_benchmark_input_reports_no_runnable_cases(self):
        rc, out, err, _, _ = self.run_filter("cs_bench_mix")
        self.assertEqual(rc, 2, out + err)
        self.assertIn("--filter 'cs_bench_mix' selected cs_bench_mix but no runnable GPU test cases", err)
        self.assertIn("inputs without a GPU test row", err)
        self.assertIn("benchmark tool", err)
        self.assertIn("=== rdna2spirv.py ===", out)
        self.assertNotIn("0/0 passed", out)

    def test_translation_failure_stays_a_failure(self):
        rc, out, err, _, _ = self.run_filter("cs_bench_mix", translate_rc=1)
        self.assertEqual((rc, err), (1, ""))
        rc, out, err, _, _ = self.run_filter("cs_bench_mix", spv=False)
        self.assertEqual((rc, err), (1, ""))
        self.assertIn("missing .spv (translation failed)", out)
        self.assertIn("0/1 passed", out)

    def test_supported_case_still_passes(self):
        rc, out, err, _, _ = self.run_filter("cs_fmt_1010102")
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("1/1 passed", out)

    def test_unmatched_filter_fails_before_the_tools_run(self):
        rc, out, err, asm, run = self.run_filter("unmatched_nicofilter")
        self.assertEqual(rc, 2)
        self.assertEqual(err, "error: --filter 'unmatched_nicofilter' matched no GPU test cases\n")
        asm.assert_not_called()
        run.assert_not_called()


class WholeWaveDump(unittest.TestCase):
    """wave64: run with --require-wave64 (the whole-wave mask tests)."""
    def judge(self, rc, msg):
        return judge(rc, msg, wave64=True)

    def test_passes_with_the_words_and_the_confirmation(self):
        self.assertEqual(self.judge(0, DEVICE_64 + CONFIRMED + dump(WANT)), (True, "55 48"))

    def test_a_device_without_64_wide_subgroups_skips(self):
        ok, detail = self.judge(77, DEVICE_32 + "full 64-wide compute subgroups unavailable\n")
        self.assertIs(ok, None)
        self.assertTrue(detail.startswith("SKIP"))

    def test_a_77_with_an_unsupported_reason_keeps_that_reason(self):
        """A 64-wide device, a module that needs OpFmaKHR the device lacks: the skip names OpFmaKHR, not the subgroup."""
        msg = ("unsupported: cs_x.spv needs OpFmaKHR (VK_KHR_shader_fma, shaderFmaFloat32) and the device lacks it; "
               "translate with --fma glsl\n")
        ok, detail = self.judge(77, DEVICE_64 + msg)
        self.assertIs(ok, None)
        self.assertTrue(detail.startswith("SKIP (cs_x.spv needs OpFmaKHR"), detail)
        self.assertNotIn("64-wide", detail)

    def test_right_words_without_the_confirmation_fail(self):    # a vk_compute_test that ignores the flag
        ok, detail = self.judge(0, DEVICE_64 + dump(WANT))
        self.assertIs(ok, False)
        self.assertIn("not confirmed", detail)

    def test_wrong_word_and_harness_errors_fail(self):
        for i in range(len(WANT)):
            wrong = list(WANT)
            wrong[i] ^= 1
            with self.subTest(word=i):
                self.assertIs(self.judge(0, CONFIRMED + dump(wrong))[0], False)
        self.assertIs(self.judge(1, "Vulkan pipeline failed\n")[0], False)       # an error is not a skip
        self.assertIs(self.judge(1, CONFIRMED + dump(WANT))[0], False)

    def test_malformed_dump_lines_fail_without_raising(self):
        for line in ("out[0..7] = 55 48\n", "out[0..7] = 55 48 BAD 0 0 0 0 0\n", "out[0..7] = 1 BAD 3\n",
                     "out[0..7] = 55 48 0 0 0 0 0 0 0\n"):
            with self.subTest(line=line):
                ok, detail = self.judge(0, CONFIRMED + line)
                self.assertIs(ok, False)
                self.assertIn(line.strip(), detail)


class NotWholeWave(unittest.TestCase):
    """Dumps run without --require-wave64: cs_store, emulated wave64 (#152) and partial-group tests."""
    def test_plain_dump_needs_no_confirmation_and_a_77_is_a_failure(self):
        self.assertEqual(judge(0, dump(WANT)), (True, "55 48"))
        self.assertIs(judge(77, "unavailable\n")[0], False)

    def test_a_77_with_an_unsupported_reason_is_a_skip(self):
        """A module that needs OpFmaKHR on a device without VK_KHR_shader_fma: vk_compute_test exits 77 with a reason."""
        msg = ("unsupported: cs_x.spv needs OpFmaKHR (VK_KHR_shader_fma, shaderFmaFloat32) and the device lacks it; "
               "translate with --fma glsl\n")
        for kw in ({}, {"emulated": True}, {"partial": True}):
            with self.subTest(**kw):
                ok, detail = judge(77, DEVICE_32 + msg, **kw)
                self.assertIs(ok, None)
                self.assertTrue(detail.startswith("SKIP (cs_x.spv needs OpFmaKHR"), detail)
        self.assertIs(judge(77, DEVICE_32 + "no reason given\n")[0], False)       # a bare 77 stays a failure

    def test_hex_words(self):
        ok, detail = rgt.judge_cs_dump(0, dump([0x400080FF, 7]), [0x400080FF, 7], wave64=False, partial=False,
                                       emulated=False, hexwords=True)
        self.assertEqual((ok, detail), (True, "0x400080ff 0x7"))

    def test_emulated_wave64_runs_on_any_subgroup_size_without_the_confirmation(self):
        self.assertEqual(judge(0, DEVICE_32 + dump(WANT), emulated=True), (True, "55 48"))
        self.assertIs(judge(77, DEVICE_32, emulated=True)[0], False)             # no skip: it must run here
        self.assertIs(judge(0, DEVICE_32 + dump([1, 2]), emulated=True)[0], False)

    def test_partial_group_skips_unless_the_device_reports_a_64_subgroup(self):
        ok, detail = judge(0, DEVICE_32 + dump(WANT), partial=True)
        self.assertIs(ok, None)
        self.assertTrue(detail.startswith("SKIP"))
        self.assertEqual(judge(0, DEVICE_64 + dump(WANT), partial=True), (True, "55 48"))   # no confirmation line
        self.assertIs(judge(0, DEVICE_64 + dump([1, 2]), partial=True)[0], False)

    def test_partial_group_emulated_runs_everywhere(self):
        self.assertEqual(judge(0, DEVICE_32 + dump(WANT), partial=True, emulated=True), (True, "55 48"))

    def test_malformed_dump_lines_fail_in_every_mode(self):
        for kw in ({}, {"emulated": True}, {"partial": True, "emulated": True}):
            with self.subTest(**kw):
                self.assertIs(judge(0, DEVICE_64 + "out[0..7] = 55 48 BAD 0 0 0 0 0\n", **kw)[0], False)



class UniformBranchCheck(unittest.TestCase):
    """check_uniform_branch (ps_uniform_branch, ps_scc_mask32) must reject a readback in which no bit=0 pixel got
    s5 = 8: that is what a translator deciding the branch per lane draws (#477, measured on RTX 5080 and RX 9070 XT:
    1856 (8,1) / 192 (8,0) / 2048 (7,0) with the vote, 1856 / 0 / 2240 without)."""

    @staticmethod
    def judge(rows):
        import os
        import struct
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".raw")
        try:
            with os.fdopen(fd, "wb") as f:   # the rows repeated over a full 64x64 frame (a short one fails, TEST-29)
                for k in range(64 * 64):
                    s5, bit, bary = rows[k % len(rows)]
                    f.write(struct.pack("<2I2f", s5, bit, bary, 0.0))
            return rgt.check_uniform_branch(path)
        finally:
            os.remove(path)

    def test_wave_vote_passes(self):
        ok, detail = self.judge([(8, 1, 0.5), (8, 0, 0.1), (7, 0, 0.1)])
        self.assertTrue(ok, detail)

    def test_per_lane_decision_fails(self):
        ok, detail = self.judge([(8, 1, 0.5), (7, 0, 0.1), (7, 0, 0.1)])
        self.assertFalse(ok)
        self.assertIn("no bit=0 pixel got s5=8", detail)

    def test_set_bit_without_8_fails(self):
        self.assertFalse(self.judge([(7, 1, 0.5), (8, 0, 0.1), (7, 0, 0.1)])[0])


class Wave32LaneDumps(unittest.TestCase):
    """run_gpu_tests.py skips cs_wqm_w32 and cs_w32_lane_wrap below 32 lanes with check.py's rule."""
    def test_process_exit_takes_precedence_in_both_runners(self):
        import check
        name = "cs_w32_lane_wrap"
        narrow = "device: synthetic (subgroup 8..16)\n"
        for code, msg, skipped in ((1, narrow, False), (0, narrow, True),
                                   (77, "unsupported: device feature unavailable\n", True)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                with patch.object(check, "run", return_value=(code, msg)):
                    verdict = check.compute_test(tmp, tmp, name, "1", "dump", needs_32_lanes=True)
                with self.subTest(runner="check.py"):
                    self.assertEqual(isinstance(verdict, check.Skip), skipped, verdict)
                    if not skipped:
                        self.assertIn("exit 1", verdict)

                def launch(cmd, **kwargs):
                    if cmd[:3] == [sys.executable, "-I", rgt.TRANSLATOR]:
                        (Path(tmp) / "spirv" / f"{name}.spv").touch()
                        return 0, ""
                    return code, msg

                args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")
                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda mc, src, dst, n: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=launch),
                      contextlib.redirect_stdout(output)):
                    status = rgt.run_suite(args, tmp)
                self.assertEqual(status, 0 if skipped else 1, output.getvalue())
                self.assertIn(f"{'SKIP' if skipped else 'FAIL'}  {name}", output.getvalue())

    def test_uses_check_py_rule(self):
        import check
        self.assertIs(rgt.narrow_subgroup_skip, check.narrow_subgroup_skip)
        self.assertIs(rgt.wave32_lane_skip, check.wave32_lane_skip)
        self.assertIs(rgt.WAVE32_LANE_DUMPS, check.WAVE32_LANE_DUMPS)
        self.assertIs(rgt.WAVE32_LANE_ENV, check.WAVE32_LANE_ENV)

    def test_unpinned_rule(self):
        self.assertEqual(rgt.wave32_lane_skip("device: X (subgroup 8..32)\ncompute subgroup size: 32 not available\n"),
                         "SKIP (the device can't run this shader at 32 lanes on request)")
        self.assertIsNone(rgt.wave32_lane_skip("device: X (subgroup 8..32)\ncompute subgroup size: 32\n"))
        observed = "compute subgroup size: 32 (observed: all 20 invocations of a 20x1x1 group)\n"
        self.assertIsNone(rgt.wave32_lane_skip("device: Apple M3 Max (subgroup 4..32)\n" + observed))
        wrong = "compute subgroup size: 32 observed wrong in 4 of 64 invocations; invocation 60 has SubgroupSize 32\n"
        self.assertIsNone(rgt.wave32_lane_skip("device: X (subgroup 4..32)\n" + wrong))
        ok, detail = rgt.judge_cs_dump(1, "device: X (subgroup 4..32)\n" + wrong, [1], wave64=False, partial=False,
                                       emulated=False)
        self.assertIs(ok, False)
        self.assertIn("observed wrong", detail)

    def test_rule(self):
        self.assertEqual(rgt.narrow_subgroup_skip("device: llvmpipe (subgroup 8..8)\n"),
                         "SKIP (subgroup 8 is narrower than 32 lanes)")
        for out in ("device: X (subgroup 32..32)\n", "device: X (subgroup 8..32)\n", "device: X (subgroup 32..64)\n",
                    "no device line\n"):
            with self.subTest(out=out):
                self.assertIsNone(rgt.narrow_subgroup_skip(out))


class SubgroupPinSkip(unittest.TestCase):
    """A device that cannot pin the wave size a shader was translated for skips, not fails. GCN3/GCN4 report
    64..64, so cs_barrier_loop's wave32 fixture runs at 64 lanes and its words say nothing; check.py skips it
    already (unpinned_subgroup_skip), the runner used to judge it and report a failure."""
    MSG = "device: X (subgroup 64..64)\ncompute subgroup size: 32 not available\n" + dump(WANT)

    def test_unpinned_size_is_a_skip(self):
        self.assertEqual(rgt.subgroup_pin_skip(self.MSG, {"VKC_SUBGROUP_SIZE": "32"}),
                         "SKIP (the device can't run this shader at 32 lanes on request)")

    def test_no_pin_requested_is_judged(self):
        for env in (None, {}, {"VKC_SGPRS": "0x1"}, {"VKH_SUBGROUP_SIZE": "64"}):
            with self.subTest(env=env):
                self.assertIsNone(rgt.subgroup_pin_skip(self.MSG, env))

    def test_a_honoured_pin_is_judged(self):
        out = "device: X (subgroup 32..64)\ncompute subgroup size: 32\n" + dump(WANT)
        self.assertIsNone(rgt.subgroup_pin_skip(out, {"VKC_SUBGROUP_SIZE": "32"}))
        self.assertIsNone(rgt.subgroup_pin_skip("device: X (subgroup 32..64)\n" + dump(WANT),
                                                {"VKC_SUBGROUP_SIZE": "32"}))

    def test_only_wave_width_fixtures_carry_the_rule(self):
        self.assertIn("cs_barrier_loop", rgt.WAVE32_PIN_DUMPS)
        # everything else is judged at the device's own width, as before, so a 64-only device keeps testing it
        for name in ("cs_store", "cs_smem_soffset_imm", "cs_w32_int_ext", "cs_barrier_divergent",
                     "cs_barrier_early_end", "cs_w32_lds_u8_d16"):
            with self.subTest(name=name):
                self.assertNotIn(name, rgt.WAVE32_PIN_DUMPS)


class ComputedIJWithoutPerVertexInputs(unittest.TestCase):
    """ps_ij_computed (SHADER-177): without PerVertexKHR inputs (--bary-fallback, --f16-interp-fallback, the latter on
    by default on MoltenVK) the translator refuses the computed I/J pair, so there is no .spv. Both runners SKIP it
    there, before their missing-.spv check; a missing .spv without a fallback stays a translation failure."""
    NAME = "ps_ij_computed"

    def run_suite(self, bary, f16):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=self.NAME, build=tmp, llvm_mc="unused")
            env = {"LEONIDA_BARY_FALLBACK": "1" if bary else "0", "LEONIDA_F16_INTERP_FALLBACK": "1" if f16 else "0"}
            out = io.StringIO()
            with (patch.dict(rgt.os.environ, env),
                  patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", return_value=(0, "1 shaders: 0 translated, 0 pass spirv-val")),   # refused
                  contextlib.redirect_stdout(out), contextlib.redirect_stderr(out)):
                rc = rgt.run_suite(args, tmp)
            return rc, out.getvalue()

    def test_run_gpu_tests_skips_under_either_fallback(self):
        for bary, f16 in ((True, False), (False, True)):
            with self.subTest(bary=bary, f16=f16):
                rc, out = self.run_suite(bary, f16)
                self.assertIn("SKIP (computed I/J pairs need PerVertexKHR inputs", out)
                self.assertIn("0/1 passed, 1 skipped", out)
                self.assertNotIn("missing .spv", out)
                self.assertEqual(rc, 0, out)

    def test_run_gpu_tests_missing_spv_without_fallback_still_fails(self):
        rc, out = self.run_suite(False, False)
        self.assertIn("missing .spv (translation failed)", out)
        self.assertEqual(rc, 1, out)

    def test_check_py_skips_before_its_spv_check(self):
        import check
        with tempfile.TemporaryDirectory() as tmp:
            for bary, f16 in ((True, False), (False, True)):
                with self.subTest(bary=bary, f16=f16), \
                        patch.object(check, "BARY_FALLBACK", bary), patch.object(check, "F16_HALVES", f16):
                    self.assertIsInstance(check.pixel_test("unused", tmp, self.NAME), check.Skip)
            with patch.object(check, "BARY_FALLBACK", False), patch.object(check, "F16_HALVES", False):
                self.assertEqual(check.pixel_test("unused", tmp, self.NAME), f"missing {self.NAME}.spv")


class F16InterpUnderBaryFallback(unittest.TestCase):
    """ps_interp_f16_high and ms_f16_pack under LEONIDA_BARY_FALLBACK=1. --bary-fallback alone has no PerVertexKHR
    inputs, so the f16 halves can't be interpolated and both are skipped. With --f16-interp-fallback as well (on by
    itself where the device has no fragmentShaderBarycentric, the usual reason for the bary fallback) the pixel shader
    reads the decoded halves its producer exports, so both run (llvmpipe, which has neither: both pass)."""
    SKIP = "SKIP (per-vertex f16 interpolants need barycentrics)"
    DRAWN = "unsupported: drawn by the stub harness"

    def run_suite(self, name, bary, f16):
        drawn = []

        def run_cmd(cmd, env=None, gpu=True):
            if rgt.TRANSLATOR in cmd:   # the translator publishes a module for every input
                out = cmd[cmd.index("--out") + 1]
                for s in os.listdir(cmd[cmd.index(rgt.TRANSLATOR) + 1]):
                    Path(out, Path(s).stem + ".spv").touch()
                return 0, ""
            if os.path.basename(cmd[0]).startswith("vk_harness"):
                drawn.append(cmd)
                return rgt.UNSUPPORTED, self.DRAWN
            return 0, ""

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")
            env = {"LEONIDA_BARY_FALLBACK": "1" if bary else "0", "LEONIDA_F16_INTERP_FALLBACK": "1" if f16 else "0"}
            out = io.StringIO()
            with (patch.dict(rgt.os.environ, env),
                  patch.dict(sys.modules, {"bary_fallback_gs": SimpleNamespace(build_for_index=lambda _d: [])}),
                  patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=run_cmd),
                  contextlib.redirect_stdout(out), contextlib.redirect_stderr(out)):
                rc = rgt.run_suite(args, tmp)
        return rc, out.getvalue(), drawn

    def test_run_gpu_tests_draws_them_with_the_f16_fallback(self):
        for name in ("ps_interp_f16_high", "ms_f16_pack"):
            with self.subTest(name=name):
                rc, out, drawn = self.run_suite(name, bary=True, f16=True)
                self.assertNotIn(self.SKIP, out)
                self.assertIn(self.DRAWN, out)
                self.assertEqual(len(drawn), 1, out)
                self.assertTrue(drawn[0][2].endswith("ps_interp_f16_high.spv"), drawn)
                self.assertEqual(rc, 0, out)

    def test_run_gpu_tests_skips_them_under_the_bary_fallback_alone(self):
        for name in ("ps_interp_f16_high", "ms_f16_pack"):
            with self.subTest(name=name):
                rc, out, drawn = self.run_suite(name, bary=True, f16=False)
                self.assertIn(self.SKIP, out)
                self.assertEqual(drawn, [], out)
                self.assertEqual(rc, 0, out)

    def test_check_py_draws_it_with_the_f16_fallback(self):
        import check
        name = "ps_interp_f16_high"
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, name + ".spv").touch()
            with patch.object(check, "pixels", return_value=(check.Skip("x"), self.DRAWN)):
                for bary, f16, want in ((True, True, f"SKIP ({self.DRAWN})"),
                                        (True, False, "SKIP (per-vertex interpolants need "
                                                      "VK_KHR_fragment_shader_barycentric)"),
                                        (False, True, f"SKIP ({self.DRAWN})")):
                    with self.subTest(bary=bary, f16=f16), \
                            patch.object(check, "BARY_FALLBACK", bary), patch.object(check, "F16_HALVES", f16):
                        self.assertEqual(check.pixel_test("unused", tmp, name), want)


class TestProgramThatDiesBeforeItsVerdict(unittest.TestCase):
    """The rows whose test program judges itself show its last output line. A program that died first (a driver
    crash while it creates a pipeline: exit -11 on Linux, 0xC0000005 on Windows) has only printed its device or
    extension line, or nothing, so the FAIL line must carry the exit status to say what happened."""
    DIED = {   # one row each: what its program had printed when it died
        "cs_lds": "device: X\n", "cs_exec64": "device: X\n", "cs_bvh": "device: X\n", "cs_wavemask": "device: X\n",
        "cs_imgstore": "device: X\n", "cs_imgload": "device: X\n", "cs_sample_lz_int_d16": "device: X\n",
        "ps_texture_sample": "device: X\nenabled VK_KHR_maintenance8\n",
        "ps_texture_sample_d16": "device: X\nenabled VK_KHR_maintenance8\n",
        "ps_cube_sample": "device: X\n", "ps_gather_wrap": "device: X\n",
        "ps_tex": "", "ps_imgatomic": "",
    }

    def run_one(self, name, code, printed):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name + ".asm", build=tmp, llvm_mc="unused", require_pass=False)   # one row

            def launch(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
                if cmd[:3] == [sys.executable, "-I", rgt.TRANSLATOR]:
                    (Path(tmp) / "spirv" / f"{name}.spv").touch()
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                return SimpleNamespace(returncode=code, stdout="", stderr=printed)

            output = io.StringIO()
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt.subprocess, "run", side_effect=launch),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                rc = rgt.run_suite(args, tmp)
        rows = [ln.split(None, 2) for ln in output.getvalue().splitlines() if ln.startswith("  FAIL")]
        return rc, rows, output.getvalue()

    def test_the_fail_line_says_the_program_died(self):
        for code in (-11, 0xC0000005):
            for name, printed in self.DIED.items():
                with self.subTest(name=name, code=code):
                    rc, rows, out = self.run_one(name, code, printed)
                    self.assertEqual(rc, 1, out)
                    self.assertEqual(len(rows), 1, out)
                    last = printed.splitlines()[-1] if printed else "no test output"
                    self.assertEqual(rows[0][1:], [name, f"exit {code}: {last}"], out)

    def test_no_pixel_count_for_a_frame_that_was_never_read(self):
        _rc, rows, out = self.run_one("ps_gather_wrap", -11, "device: X\n")
        self.assertNotIn("pixels differ", rows[0][2], out)

    def test_a_verdict_keeps_its_line(self):
        self.assertEqual(rgt.verdict_line(0, "device: X\ncs_lds: 64 lanes mirrored"), "cs_lds: 64 lanes mirrored")
        self.assertEqual(rgt.verdict_line(0, ""), "")
        self.assertEqual(rgt.verdict_line(1, "device: X\ncs_lds: lane 3 differs"), "exit 1: cs_lds: lane 3 differs")
        rc, rows, out = self.run_one("cs_imgstore", 0, "device: X\ncs_imgstore: 64 texels match\n")
        self.assertEqual((rc, rows), (0, []), out)
        self.assertIn("PASS  cs_imgstore           cs_imgstore: 64 texels match", out)


class FmaFusedSuite(unittest.TestCase):
    """run_gpu_tests.py runs cs_fma_fused under --filter, honoring LEONIDA_FMA=khr vs default glsl."""
    RADV_SPLIT = [0x00000000, 0xA8800000, 0x3A000000, 0x00000000, 0x00000000, 0x00000000, 0x28800000, 0x00007351]

    def run_fma(self, rc, msg, fma_env=None, fma_mode=None):
        with tempfile.TemporaryDirectory() as tmp:
            if fma_mode:
                (Path(tmp) / "spirv").mkdir(parents=True, exist_ok=True)
                (Path(tmp) / "spirv" / "index.json").write_text(
                    json.dumps({"cs_fma_fused": {"fma": fma_mode}}), encoding="utf-8")
            args = SimpleNamespace(filter="cs_fma_fused", build=tmp, llvm_mc="unused")

            def fake_run(cmd, env=None, gpu=True):
                if Path(cmd[0]).stem == "vk_compute_test":
                    self.assertEqual((env or {}).get("VKC_SGPRS"), rgt.FMA_FUSED_ENV["VKC_SGPRS"])
                    return rc, msg
                (Path(tmp) / "spirv" / "cs_fma_fused.spv").touch()
                return 0, ""

            output = io.StringIO()
            env_patch = {"LEONIDA_FMA": fma_env} if fma_env is not None else {}
            with (patch.dict(rgt.os.environ, env_patch, clear=False),
                  patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=fake_run),
                  contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                if fma_env is None:
                    rgt.os.environ.pop("LEONIDA_FMA", None)
                code = rgt.run_suite(args, tmp)
            return code, output.getvalue()

    def test_glsl_accepts_radv_split_and_khr_rejects_it(self):
        rc, out = self.run_fma(0, dump(self.RADV_SPLIT), fma_env=None)
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS  cs_fma_fused", out)
        self.assertIn("5/7 split", out)
        rc, out = self.run_fma(0, dump(self.RADV_SPLIT), fma_env="khr")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL  cs_fma_fused", out)

    def test_recorded_exact_mode_makes_the_gpu_suite_strict(self):
        # TEST-1147: run_gpu_tests must read the same index field as check.py, or an exact build judged
        # in a fresh shell accepts split words that the GPU driver really produced.
        import check
        rc, out = self.run_fma(0, dump(check.FMA_FUSED), fma_mode="exact")
        self.assertEqual(rc, 0, out)
        rc, out = self.run_fma(0, dump(self.RADV_SPLIT), fma_mode="exact")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL  cs_fma_fused", out)

    def test_recorded_glsl_mode_ignores_stale_exact_env(self):
        rc, out = self.run_fma(0, dump(self.RADV_SPLIT), fma_mode="glsl", fma_env="exact")
        self.assertEqual(rc, 0, out)
        self.assertIn("5/7 split", out)

    def test_khr_fused_passes_and_exit_77_skips(self):
        import check
        rc, out = self.run_fma(0, dump(check.FMA_FUSED), fma_env="khr")
        self.assertEqual(rc, 0, out)
        self.assertIn("8/8 fused exact (single-rounded FMA: OpFmaKHR or --fma exact)", out)
        rc, out = self.run_fma(77, "unsupported: no VK_KHR_shader_fma", fma_env="khr")
        self.assertEqual(rc, 0, out)
        self.assertIn("SKIP (unsupported: no VK_KHR_shader_fma)", out)


if __name__ == "__main__":
    unittest.main()
