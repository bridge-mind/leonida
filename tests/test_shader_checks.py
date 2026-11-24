"""Exercise the synthetic Vulkan test runner without a Vulkan device."""
import ast
import contextlib
import io
import json
import re
import struct
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import check


GOOD_FMT = struct.pack("<64I", *check.FMT_1010102_WORDS, *[0xEEEEEEEE] * 48)
GOOD_XY_XYZ = struct.pack("<192I", *check.FMT_XY_XYZ_WORDS, *[0xEEEEEEEE] * 108)
GOOD_TBUFFER_STORE = struct.pack("<192I", *check.TBUFFER_STORE_WORDS, *[0xEEEEEEEE] * 140)
FULL_WAVE = ((0, "compute subgroup size: 64\n0 mismatches\n"),
             (0, "compute subgroup size: 64\nout[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"))

# The other wave64-only dump tests (#73's cs_wm_*, the cs_w64x_* cross-half tests): name -> expected out[] words.
WAVE64_DUMPS = {n: w for n, w in check.CS_DUMPS + check.W64X_DUMPS + check.W64X_PARTIAL
                if n not in (*check.WAVE32_DUMPS, "cs_wave64_ballot")}
# A group smaller than a wave can't ask for full subgroups: it asks for subgroup size 64 and reads the device's range.
PARTIAL = {n for n, _ in check.W64X_PARTIAL}
WAVE32_WORDS = {n: w for n, w in check.CS_DUMPS if n in check.WAVE32_DUMPS and n not in ("cs_store", "cs_smem_soffset_imm")}


# All check.py cases, including the four vk_store_test shaders and the three mesh fixtures, and TEST-184's
# vk_store_test "words" fixtures and image_get_lod draws.
TOTAL = (len(check.COMPUTE) + 7 + len(check.CS_DUMPS) + len(check.W64X_DUMPS) + len(check.W64X_PARTIAL) + 4
         + len(check.PIXEL_TESTS) + len(check.CLIP_TESTS) + len(check.BOUNDS) + len(check.GLOBAL_OFFSET)
         + len(check.MIP_TEST_PASSES) + len(check.GRADIENT_TESTS)
         + len(check.STORE_WORDS) + len(check.GET_LOD_TESTS)
         + len(check.TEXTURE_CONSTANTS) + 15)  # three mesh fixtures, ps_tex, ps_imgatomic, six compute fixtures,
         # ps_ssharp, vs_base_ids, main's standalone cs_fmt_d16_xyz / cs_heap_init dumps (net one more case) and cs_fmt_all
CVT = {bits: [u, s] for bits, u, s in check.CVT_F32_SAT_CASES}


def words_output(name, flip=None):
    """What vk_store_test prints in "words" mode when the shader left check.store_words(name) in the three images;
    flip = (image, word index) changes one word."""
    lines = ["device: fake"]
    for image, words in zip(check.STORE_IMAGES, check.store_words(name)):
        words = [w ^ 1 if flip == (image, i) else w for i, w in enumerate(words)]
        lines.append(f"{image} words: " + " ".join(f"{w:#010x}" for w in words))
    return "\n".join(lines) + "\n"


def cvt_answer(env, wrong_for=None):
    """A device running cs_cvt_f32_sat on the word in VKC_SGPRS (s4 reads 0 without it)."""
    bits = int((env or {}).get("VKC_SGPRS", "0"), 0)
    words = [0xBAD, 0xBAD] if bits == wrong_for else CVT.get(bits, [0, 0])
    return 0, "out[0..7] = " + " ".join(map(str, words + [0xEEEEEEEE] * 6)) + "\n"


class Wave64Checks(unittest.TestCase):
    def run_checks(self, wavemask, ballot, *, missing_ballot=False, emulated=False, fmt=(0, GOOD_FMT)):
        def harness(_tool, shader, *args, env=None):
            name = Path(shader).stem
            if name == "cs_fmt_1010102":
                self.assertEqual(args, ("1", "dump"))
                self.fmt_env, dump = env, Path(env["VKC_DUMP"])
                self.assertFalse(dump.exists())
                if fmt[1] is not None:
                    dump.write_bytes(fmt[1])
                return fmt[0], "out[0..7] = 0 0 0 0 0 0 0 0\n"   # stdout is not the oracle
            if name == "cs_fmt_sint32":
                self.assertEqual((args, env["VKC_OUT_WORDS"]), (("1", "dump"), "64"))
                Path(env["VKC_DUMP"]).write_bytes(struct.pack("<64I", *check.FMT_SINT32_WORDS))
                return 0, ""
            if name == "cs_fmt_d16_xyz":
                self.assertEqual((args, env["VKC_OUT_WORDS"]), (("1", "dump"), "80"))
                Path(env["VKC_DUMP"]).write_bytes(struct.pack("<80I", *check.FMT_D16_XYZ_WORDS))
                return 0, ""
            if name == "cs_fmt_int_narrow":
                self.assertEqual((args, env["VKC_OUT_WORDS"]), (("1", "dump"), "136"))
                Path(env["VKC_DUMP"]).write_bytes(struct.pack("<136I", *check.FMT_INT_NARROW_WORDS))
                return 0, ""
            if name in ("cs_fmt_xy_xyz", "cs_tbuffer_store"):
                self.assertEqual((args, env["VKC_OUT_WORDS"]), (("1", "dump"), "192"))
                Path(env["VKC_DUMP"]).write_bytes(GOOD_XY_XYZ if name == "cs_fmt_xy_xyz" else GOOD_TBUFFER_STORE)
                return 0, "out[0..7] = 0 0 0 0 0 0 0 0\n"
            if name == "cs_fmt_all":
                self.assertEqual((args, env["VKC_OUT_WORDS"]), (("1", "dump"), str(len(check.FMT_ALL_WORDS))))
                Path(env["VKC_DUMP"]).write_bytes(struct.pack(f"<{len(check.FMT_ALL_WORDS)}I", *check.FMT_ALL_WORDS))
                return 0, "out[0..7] = 0 0 0 0 0 0 0 0\n"
            if name == "cs_heap_init":
                # The seeded words come back only if check.py wrote the init file and sized the buffer for it.
                path = (env or {}).get("VKC_HEAP_INIT")
                seeded = bool(path) and Path(path).read_bytes() == check.HEAP_INIT_FILE and \
                    env.get("VKC_OUT_WORDS") == "68"
                words = check.HEAP_INIT_EXPECTED if seeded else [0xEEEEEEEE] * 8
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name == "cs_user_sgprs":
                # The seeded words come back only if check.py passed them in VKC_SGPRS.
                seeded = (env or {}).get("VKC_SGPRS") == check.USER_SGPRS_ENV["VKC_SGPRS"]
                words = check.USER_SGPRS_WORDS if seeded else [0] * 4 + [0xEEEEEEEE] * 4
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name == "cs_cvt_f32_sat":
                return cvt_answer(env)
            if name == "cs_fma_fused":
                # The operands come in through VKC_SGPRS; without them the fused words are not produced.
                self.assertEqual((env or {}).get("VKC_SGPRS"), check.FMA_FUSED_ENV["VKC_SGPRS"])
                return 0, "out[0..7] = " + " ".join(map(str, check.FMA_FUSED)) + "\n"
            self.assertNotIn("VKC_SGPRS", env or {})
            if name in PARTIAL:
                self.assertNotIn("--require-wave64", args)
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                narrow = wavemask[0] == 77
                return 0, f"device: gpu (subgroup 32..{32 if narrow else 64})\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if emulated:
                self.assertNotIn("--require-wave64", args)
                if name in WAVE64_DUMPS:
                    words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                    return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
                if name == "cs_wavemask":
                    return 0, "0 mismatches\n"
                if name == "cs_wave64_ballot":
                    return 0, "out[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"
            if name in WAVE64_DUMPS:
                self.assertIn("--require-wave64", args)
                if wavemask[0] == 77:          # same device as cs_wavemask: no 64-wide subgroups
                    return wavemask
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                return 0, "compute subgroup size: 64\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if name in ("cs_wavemask", "cs_wave64_ballot"):
                self.assertIn("--require-wave64", args)
                if name == "cs_wavemask":
                    return wavemask
                self.assertFalse(missing_ballot, "missing shader was sent to the harness")
                return ballot
            self.assertNotIn("--require-wave64", args)
            if name == "cs_bounds":
                # 0x12345678, the in-bounds reload, the OOB load's 0, then the untouched 0xEEEEEEEE fill.
                return 0, "out[0..7] = 305419896 305419896 0 4008636142 4008636142 4008636142 4008636142 4008636142\n"
            if name == "cs_cmpx_class_f32":
                return 0, "out[0..7] = " + " ".join(map(str, check.CMPX_CLASS_WORDS)) + "\n"
            if name in {n for n, _, _ in check.GLOBAL_OFFSET}:
                groups, words = next((g, w) for n, g, w in check.GLOBAL_OFFSET if n == name)
                self.assertEqual(args[0], groups)
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name in dict(check.BOUNDS):    # the SMEM / raw MUBUF rule tests: their words, then the 0xee fill
                words = (dict(check.BOUNDS)[name] + [0xEEEEEEEE] * 8)[:8]
                return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
            if name == "cs_cvt_pk_u8":
                return 0, "out[0..7] = " + " ".join(map(str, check.CVT_PK_U8_WORDS)) + "\n"
            if name == "cs_pk_mul_lo_const":
                return 0, "out[0..7] = " + " ".join(map(str, check.PK_MUL_LO_WORDS)) + "\n"
            if name == "cs_store":
                return 0, "out[0..7] = " + " ".join(map(str, (check.CS_STORE + [0] * 8)[:8])) + "\n"
            if name == "cs_smem_soffset_imm":
                return 0, "out[0..7] = " + " ".join(map(str, check.CS_SMEM)) + "\n"
            if name in WAVE32_WORDS:           # wave32 dumps run on any subgroup size
                return 0, "out[0..7] = " + " ".join(map(str, (WAVE32_WORDS[name] + [0] * 8)[:8])) + "\n"
            if name == "cs_workgroup_yz":     # #186: vk_compute_test's 2x3x4 Y/Z histogram line
                return 0, "2x3x4 workgroup Y/Z: 0 mismatches\n"
            if name in ("cs_imgstore", "cs_imgload", "cs_imgload_d16", "cs_sample_lz_int_d16"):   # vk_store_test checks
                return 0, "0 bad\n0 bad\n0 bad\n"
            if name in check.STORE_WORDS:     # vk_store_test "words": the three images, judged by check.py
                self.assertEqual(args, ("words",))
                return 0, words_output(name)
            return 0, "0 mismatches\n"

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / "cs_wavemask.spv").touch()
            for name in (*WAVE64_DUMPS, *check.STORE_WORDS):
                (spirv / f"{name}.spv").touch()
            if not missing_ballot:
                (spirv / "cs_wave64_ballot.spv").touch()
            output = io.StringIO()
            with (patch.object(sys, "argv", ["check.py", tmp]), patch.object(check, "run", side_effect=harness),
                  patch.object(check, "clip_test", return_value=""),
                  patch.object(check, "pixel_test", return_value=""), patch.object(check, "texture_mip_test", return_value=""),
                  patch.object(check, "bindless_test", return_value=""), patch.object(check, "W64_EMULATED", emulated),
                  patch.object(check, "texture_gradient_a16_test", return_value=""),
                  patch.object(check, "texture_constant_test", return_value=""),
                  patch.object(check, "base_ids_test", return_value=""),
                  patch.object(check, "get_lod_test", return_value=""),
                  patch.object(check, "sampler_test", return_value=""),
                  contextlib.redirect_stdout(output)):
                status = check.main()
        return status, output.getvalue()

    def test_wave32_only_device_skips_full_wave_checks(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wavemask +SKIP")
        self.assertRegex(out, r"cs_wave64_ballot +SKIP")
        self.assertRegex(out, r"cs_exec64 +ok")
        self.assertIn(f"{TOTAL - 5 - len(WAVE64_DUMPS)} of {TOTAL} pass, {5 + len(WAVE64_DUMPS)} skipped", out)

    def test_full_wave_checks_pass_with_expected_results(self):
        status, out = self.run_checks(*FULL_WAVE)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wave64_ballot +ok")
        self.assertIn(f"{TOTAL - 3} of {TOTAL} pass, 3 skipped", out)

    def test_emulated_wave64_runs_without_a_64_wide_subgroup(self):
        # LEONIDA_W64_EMULATE=1: the shaders were translated with --wave64-emulate, so no wave64 test asks the
        # harness for a 64-wide subgroup, and a 32-wide device runs them all.
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, emulated=True)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_w64x_wg128_branches +ok")
        self.assertIn(f"{TOTAL - 3} of {TOTAL} pass, 3 skipped", out)

    def test_ballot_mismatches_and_harness_errors_fail(self):
        wavemask = 0, "compute subgroup size: 64\n0 mismatches\n"
        for ballot, detail in (
            ((0, "compute subgroup size: 64\nout[0..7] = 16 16 0 64 0 64 0 0\n"), "exit 0"),
            ((0, "compute subgroup size: 64\nout[0..7] = corrupt\n"), "exit 0"),
            ((0, "out[0..7] = 32 16 24 40 8 48 0 0\n"), "64-wide subgroup was not confirmed"),
            ((1, "Vulkan pipeline failed\n"), "exit 1"),
        ):
            with self.subTest(ballot=ballot):
                status, out = self.run_checks(wavemask, ballot)
                self.assertEqual(status, 1, out)
                self.assertRegex(out, rf"cs_wave64_ballot +{detail}")
                self.assertIn(f"{TOTAL - 4} of {TOTAL} pass, 3 skipped", out)

    def test_fmt_1010102_compares_the_whole_dump(self):
        words = list(struct.unpack("<64I", GOOD_FMT))
        for fmt, detail in (
            ((0, struct.pack("<64I", *words[:15], 0xE00003FF, *words[16:])), "word 15: expected 0xffe00003, got 0xe00003ff"),
            ((0, struct.pack("<64I", *words[:63], 0)), "word 63: expected 0xeeeeeeee, got 0x00000000"),
            ((0, GOOD_FMT[:-4]), "expected 256 dump bytes, got 252"),
            ((0, GOOD_FMT + bytes(4)), "expected 256 dump bytes, got 260"),
            ((0, None), "no VKC_DUMP file written"),
            ((1, GOOD_FMT), "exit 1"),
        ):
            with self.subTest(detail=detail):
                status, out = self.run_checks(*FULL_WAVE, fmt=fmt)
                self.assertEqual(status, 1, out)
                self.assertRegex(out, rf"cs_fmt_1010102 +{detail}")
                self.assertIn(f"{TOTAL - 4} of {TOTAL} pass, 3 skipped", out)

    def test_fmt_1010102_child_env_is_private(self):
        parent = {"VK_DRIVER_FILES": "driver.json", "VKC_OUT_WORDS": "128", "VKC_DUMP": "caller.bin"}
        with patch.dict(check.os.environ, parent):
            status, out = self.run_checks(*FULL_WAVE)
            for key, value in parent.items():
                self.assertEqual(check.os.environ[key], value)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_fmt_1010102 +ok")
        self.assertEqual((self.fmt_env["VK_DRIVER_FILES"], self.fmt_env["VKC_OUT_WORDS"]), ("driver.json", "64"))
        self.assertNotEqual(self.fmt_env["VKC_DUMP"], parent["VKC_DUMP"])
        self.assertFalse(Path(self.fmt_env["VKC_DUMP"]).exists())

    def test_missing_wave64_fixture_fails_even_on_wave32(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, missing_ballot=True)
        self.assertEqual(status, 1, out)
        self.assertIn("missing cs_wave64_ballot.spv", out)


class FloatToIntSweep(unittest.TestCase):
    """check.cvt_f32_sat_test sends each input in VKC_SGPRS and judges both words of every dispatch."""

    def sweep(self, wrong_for=None):
        sent = []

        def harness(_tool, shader, *_args, env=None):
            self.assertEqual(Path(shader).stem, "cs_cvt_f32_sat")
            sent.append(int(env["VKC_SGPRS"], 0))
            return cvt_answer(env, wrong_for)

        with patch.object(check, "run", side_effect=harness):
            return check.cvt_f32_sat_test("tools", "spirv"), sent

    def test_every_input_is_sent_and_exact(self):
        problem, sent = self.sweep()
        self.assertEqual(problem, "")
        self.assertEqual(sent, [bits for bits, _, _ in check.CVT_F32_SAT_CASES])

    def test_one_wrong_input_fails_and_is_named(self):
        problem, _ = self.sweep(wrong_for=0x4F000000)
        self.assertIn("1 of 16 inputs wrong", problem)
        self.assertIn("0x4f000000", problem)

    def test_a_device_that_cannot_pin_the_wave_size_skips_instead_of_failing(self):
        # Before: on a 4..32 device (Apple under MoltenVK) every dispatch came back as unpinned_subgroup_skip's Skip,
        # and the sweep reported it as "16 of 16 inputs wrong, first 0x00000000: SKIP (...)", a failure.
        calls = []

        def harness(_tool, _shader, *_args, env=None):
            calls.append(env.get("VKC_SUBGROUP_SIZE"))
            return 1, "compute subgroup size: 32 not available\n"

        with patch.object(check, "run", side_effect=harness), \
                patch.object(check, "subgroup_size", return_value=32):
            problem = check.cvt_f32_sat_test("tools", "spirv")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("can't run this shader at 32 lanes", problem)
        self.assertNotIn("inputs wrong", problem)
        self.assertEqual(calls, ["32"])   # the first dispatch decides; the other 15 would say the same

    def test_inputs_match_the_interpreter_test(self):
        # The GPU sweep and tests/semantic/test_float_to_int.py must judge the same inputs the same way.
        src = (Path(__file__).resolve().parent / "semantic" / "test_float_to_int.py").read_text(encoding="utf-8")
        cases = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CASES")
        self.assertEqual(tuple(cases), check.CVT_F32_SAT_CASES)


class MeshWorkgroupChecks(unittest.TestCase):
    def test_every_group_has_its_own_pixel(self):
        pixels = [struct.pack("<4f", 32 * g, 32 * g + 1, 32 * g + 31, 1) for g in range(24)]   # #209's v5
        self.assertEqual(check.check_mesh_v5(b"".join(pixels)), "")
        pixels[6], pixels[2] = pixels[2], pixels[6]
        self.assertIn("2/24 mismatches; first (0,1,0)", check.check_mesh_v5(b"".join(pixels)))
        self.assertIn("384 bytes", check.check_mesh_v5(b"".join(pixels[:-1])))

    def test_a_previous_groups_file_cannot_pass_the_mesh_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp)
            (spirv / "ms_v5_workgroup.spv").touch()
            (spirv / "fs_passthru.spv").touch()
            raw = spirv / "groups.raw"
            raw.write_bytes(b"".join(struct.pack("<4f", 32 * g, 32 * g + 1, 32 * g + 31, 1)
                                     for g in range(24)))
            with patch.object(check, "run", return_value=(0, "")):
                self.assertIn("missing readback: groups.raw", check.mesh_v5_test(tmp, spirv, raw))
            self.assertFalse(raw.exists())

    def test_readback_cleanup_failure_does_not_start_the_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "groups.raw"
            raw.mkdir()
            with patch.object(check, "run") as launch:
                status, message = check.run_readback(raw, launch, "unused")
                launch.assert_not_called()
            self.assertEqual(status, 1)
            self.assertIn("could not remove the previous readback", message)
            self.assertTrue(raw.is_dir())


class PixelDeviceLimits(unittest.TestCase):
    """pixel_test with a fake harness: the device's limits decide SKIP instead of a failure."""

    def run_pixel(self, name, code, out, data=b""):
        def harness(_tool, _vs, _fs, raw, env=None):
            if data:
                Path(raw).write_bytes(data)
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            with patch.object(check, "run", side_effect=harness):
                return check.pixel_test(tmp, str(spirv), name)

    @staticmethod
    def bitmask_pixels(subgroup):
        lanes = min(subgroup, 32)
        return b"".join(struct.pack("<4I", i % lanes, (i % lanes) ^ 9, (((i % lanes) & 5) | 16) ^ 1, 0)
                        for i in range(64 * 64))

    def test_shader_needing_missing_barycentrics_skips(self):
        problem = self.run_pixel("ps_bary", 77,
                                 "unsupported: ps_bary.spv reads BaryCoordKHR and the device has no "
                                 "VK_KHR_fragment_shader_barycentric\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("BaryCoordKHR", problem)

    def test_other_harness_errors_still_fail(self):
        problem = self.run_pixel("ps_ops", 1, "FAILED vkCreateDevice -> -9\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("vk_harness failed", problem)

    def test_nan_payload_exception_is_limited_to_kosmickrisp(self):
        payloads = self.readlane_pixels((0x7fc12345, 0xffc54321, 0x3f800000, 0x3f000000))
        canonical = self.readlane_pixels((0x7fc00000, 0x7fc00000, 0x3f800000, 0x3f000000))
        for driver in ("driver: KosmicKrisp\n", "driver: MoltenVK\n", ""):
            with self.subTest(driver=driver):
                out = f"device: Apple M1 Pro (subgroup 32..32)\n{driver}ok 64x64\n"
                self.assertEqual(self.run_pixel("ps_nan_payload", 0, out, payloads), "")
                problem = self.run_pixel("ps_nan_payload", 0, out, canonical)
                if driver == "driver: KosmicKrisp\n":
                    self.assertEqual(problem, "")
                else:
                    self.assertIn("unexpected words", problem)

    def test_kosmickrisp_still_rejects_bad_nan_and_finite_words(self):
        out = "driver: KosmicKrisp\nok 64x64\n"
        for words in ((0x7fc00001, 0x7fc00000, 0x3f800000, 0x3f000000),
                      (0x7f800000, 0x7fc00000, 0x3f800000, 0x3f000000),
                      (0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3f000000)):
            with self.subTest(words=words):
                problem = self.run_pixel("ps_nan_payload", 0, out, self.readlane_pixels(words))
                self.assertIn("unexpected words", problem)

    def test_bitmask_perm_skips_below_32_lanes(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=self.bitmask_pixels(8))
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("narrower than 32 lanes", problem)

    def test_bitmask_perm_is_judged_at_32_lanes(self):
        out = "device: fake (subgroup 32..32)\nok 64x64\n"
        self.assertEqual(self.run_pixel("ps_bitmask_perm", 0, out, data=self.bitmask_pixels(32)), "")

    def test_bitmask_perm_skips_when_the_frame_is_short_below_32_lanes(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=b"short")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("narrower than 32 lanes", problem)

    def test_bitmask_perm_skips_without_a_subgroup_width(self):
        problem = self.run_pixel("ps_bitmask_perm", 0, "ok 64x64\n", data=self.bitmask_pixels(32))
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("no subgroup width", problem)

    @staticmethod
    def readlane_pixels(words=(0x40000079, 0x40000010, 0x40000064, 0x400000be)):
        return struct.pack("<4I", *words) * (64 * 64)

    def test_wave64_pixel_shader_skips_where_64_cannot_be_required(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_w64x_readlane", 0, out, data=self.readlane_pixels((0, 0, 0, 0)))
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("64-wide fragment subgroup", problem)
        self.assertIn("8..8", problem)

    def test_wave64_pixel_shader_skips_where_the_harness_says_64_is_not_available(self):
        # vk_harness.cpp's line for a size the device could not take; it must not read as "64 was required".
        out = "device: fake (subgroup 32..32)\nfragment subgroup size: 64 not available\nok 64x64\n"
        problem = self.run_pixel("ps_w64x_readlane", 0, out, data=self.readlane_pixels())
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("64-wide fragment subgroup", problem)

    def test_wave64_pixel_shader_is_judged_when_64_was_required(self):
        # The line as vk_harness.cpp prints it when the size took effect: "fragment subgroup size: %u\n".
        out = "device: fake (subgroup 32..64)\nfragment subgroup size: 64\nok 64x64\n"
        self.assertEqual(self.run_pixel("ps_w64x_readlane", 0, out, data=self.readlane_pixels()), "")
        problem = self.run_pixel("ps_w64x_readlane", 0, out, data=self.readlane_pixels((0, 0, 0, 0)))
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("pixels differ", problem)

    def test_wide_pixel_skip_reads_the_line_vk_harness_prints(self):
        # Keeps the check and the harness in step: the format string is read from vk_harness.cpp, not written here.
        harness = Path(__file__).resolve().parents[1] / "runtime" / "test" / "vk_harness.cpp"
        source = harness.read_text(encoding="utf-8")
        formats = re.findall(r'"(fragment subgroup size: %u[^"]*)\\n"', source)
        self.assertEqual(sorted(formats), ["fragment subgroup size: %u", "fragment subgroup size: %u not available"])
        taken, refused = (f.replace("%u", "64") + "\n" for f in sorted(formats))
        self.assertIsNone(check.wide_pixel_skip("ps_w64x_readlane", "device: fake (subgroup 32..64)\n" + taken))
        self.assertIsInstance(check.wide_pixel_skip("ps_w64x_readlane", "device: fake (subgroup 32..32)\n" + refused),
                              check.Skip)

    def test_bitmask_perm_skips_when_variable_device_runs_narrow_fragment_subgroup(self):
        out = "device: Intel(R) HD Graphics 530 (subgroup 8..32)\nok 64x64\n"
        for width in (8, 16):
            with self.subTest(width=width):
                # In SIMD8/SIMD16, "10p0i" targets lanes >= 16 (out of bounds), returning 0 or wrapped values.
                data = b"".join(struct.pack("<4I", i % width, (i % width) ^ 9, 0, 0) for i in range(64 * 64))
                problem = self.run_pixel("ps_bitmask_perm", 0, out, data=data)
                self.assertIsInstance(problem, check.Skip)
                self.assertIn(f"{width} lanes", problem)

    def test_bitmask_perm_narrow_run_still_judges_the_sources_that_run(self):
        # At 16 lanes "pippi" (lane ^ 9) stays inside the running lanes, so a wrong word there is a failure.
        out = "device: Intel(R) HD Graphics 530 (subgroup 8..32)\nok 64x64\n"
        bad = b"".join(struct.pack("<4I", i % 16, i % 16, 0, 0) for i in range(64 * 64))
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=bad)
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("wrong", problem)
        # At 8 lanes every "pippi" source is lane 8 or above, so nothing can be judged and any word skips.
        any8 = b"".join(struct.pack("<4I", i % 8, 12345, 0, 0) for i in range(64 * 64))
        self.assertIsInstance(self.run_pixel("ps_bitmask_perm", 0, out, data=any8), check.Skip)

    def test_bitmask_perm_judges_variable_device_at_32_lanes(self):
        out = "device: Intel(R) HD Graphics 530 (subgroup 8..32)\nok 64x64\n"
        self.assertEqual(self.run_pixel("ps_bitmask_perm", 0, out, data=self.bitmask_pixels(32)), "")
        bad = b"".join(struct.pack("<4I", i % 32, (i % 32) ^ 9, 0, 0) for i in range(64 * 64))
        problem = self.run_pixel("ps_bitmask_perm", 0, out, data=bad)
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("wrong", problem)

    def test_bitmask_perm_fails_broken_lanes_instead_of_skipping(self):
        # Fixed 32..32 device producing only lanes 0..15, or 8..32 device producing incomplete/too-narrow lanes.
        simd16 = b"".join(struct.pack("<4I", i % 16, (i % 16) ^ 9, 0, 0) for i in range(64 * 64))
        problem = self.run_pixel("ps_bitmask_perm", 0, "device: fake (subgroup 32..32)\nok 64x64\n", data=simd16)
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("wrong", problem)

        out = "device: fake (subgroup 8..32)\nok 64x64\n"
        for lanes in (1, 4, 15):
            with self.subTest(lanes=lanes):
                bad = b"".join(struct.pack("<4I", i % lanes, (i % lanes) ^ 9, 0, 0) for i in range(64 * 64))
                problem = self.run_pixel("ps_bitmask_perm", 0, out, data=bad)
                self.assertNotIsInstance(problem, check.Skip)
                self.assertIn("wrong", problem)

    @staticmethod
    def vcmpx_pixels(words=(0xf05b2a48, 0xf05e8a12, 0xb174be18, 0x15550000)):
        return b"".join(struct.pack("<4I", *words) for _ in range(64 * 64))

    def test_vcmpx_forms_skips_on_variable_fragment_subgroup(self):
        # TEST-205: no lane IDs in mrt0, so the skip is from the reported range (same class as #1318).
        out = "device: Intel UHD 770 (subgroup 8..32)\nok 64x64\n"
        problem = self.run_pixel("ps_vcmpx_forms", 0, out, data=self.vcmpx_pixels())
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("8..32", problem)
        self.assertIn("needs 32", problem)

    def test_vcmpx_forms_skips_below_32_lanes(self):
        out = "device: fake (subgroup 8..8)\nok 64x64\n"
        problem = self.run_pixel("ps_vcmpx_forms", 0, out, data=self.vcmpx_pixels())
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("narrower than 32 lanes", problem)

    def test_vcmpx_forms_is_judged_at_32_lanes(self):
        out = "device: NVIDIA (subgroup 32..32)\nok 64x64\n"
        self.assertEqual(self.run_pixel("ps_vcmpx_forms", 0, out, data=self.vcmpx_pixels()), "")
        bad = self.vcmpx_pixels((0, 0, 0, 0xffff))  # lane-mask-looking word3, as on UHD 770
        problem = self.run_pixel("ps_vcmpx_forms", 0, out, data=bad)
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("differ", problem)

    def test_vcmpx_forms_skips_without_a_subgroup_width(self):
        problem = self.run_pixel("ps_vcmpx_forms", 0, "ok 64x64\n", data=self.vcmpx_pixels())
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("no subgroup width", problem)


class SeededPixelTests(unittest.TestCase):
    """ps_vcc_x2 reads a V# from s[0:3]; check.py hands vk_harness that V# and the heap words behind it (TEST-19)."""

    def run_pixel_test(self, words, code=0, out=""):
        seen = {}

        def harness(_tool, _vs, _fs, raw, env=None):
            seen["env"] = env
            if not code:
                Path(raw).write_bytes(struct.pack("<4I", *words) * (64 * 64))
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_vcc_x2.spv").touch()
            with patch.object(check, "run", side_effect=harness):
                problem = check.pixel_test(tmp, tmp, "ps_vcc_x2")
        return problem, seen["env"] or {}

    def test_v_sharp_and_heap_reach_the_harness(self):
        problem, env = self.run_pixel_test((0x11223344, 0x55667788, 0, 0))
        self.assertEqual(problem, "")
        self.assertEqual(env.get("VKH_SGPRS"), "0x1000,0,16,0")
        self.assertEqual(env.get("VKH_HEAP"), "0x1000=0x11223344,0x1004=0x55667788")

    def test_unseeded_result_fails(self):
        problem, _ = self.run_pixel_test((0, 0, 0, 0))
        self.assertIn("4096 of 4096 pixels differ", problem)

    def test_device_without_buffer_addresses_skips(self):
        problem, _ = self.run_pixel_test((), code=77, out="unsupported: VKH_HEAP needs bufferDeviceAddress\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("needs bufferDeviceAddress", problem)


class SignedZeroUnjudgedWords(unittest.TestCase):
    """cs_signed_zero_nan's -0 products (words 0, 2 and 3) aren't judged on Apple GPUs, which return +0 for them with
    or without SignedZeroInfNanPreserve (#1203), and its 0 * inf NaN check (word 5) isn't judged on Mesa llvmpipe,
    which returns 0 there with or without it (#664). Their other words still are, and every other device judges all."""
    APPLE_WORDS = [0, 0x80000000, 0, 0, 1, 1, 1, 0x7C00]   # what MoltenVK returned in #664
    LLVMPIPE = "llvmpipe (LLVM 21.1.8, 256 bits)"
    LLVMPIPE_WORDS = [0x80000000, 0x80000000, 0x80008000, 0x8000, 1, 0, 1, 0x7C00]   # what llvmpipe returned in #664

    def compute(self, device, words, subgroup="32..32"):
        out = f"device: {device} (subgroup {subgroup})\nout[0..7] = " + " ".join(map(str, words)) + "\n"
        with patch.object(check, "run", return_value=(0, out)):
            return check.compute_test("tools", "spirv", "cs_signed_zero_nan", "1", "dump",
                                      expected_words=check.SIGNED_ZERO_NAN)

    def test_apple_does_not_judge_the_minus_zero_products(self):
        self.assertEqual(self.compute("Apple M1 Max", self.APPLE_WORDS), "")

    def test_apple_still_judges_the_nan_checks(self):
        words = list(self.APPLE_WORDS)
        words[4] = 0   # an x != x NaN check folded to false, as without the execution mode
        self.assertNotEqual(self.compute("Apple M1 Max", words), "")

    def test_llvmpipe_does_not_judge_the_zero_times_inf_check(self):
        self.assertEqual(self.compute(self.LLVMPIPE, self.LLVMPIPE_WORDS, "8..8"), "")

    def test_llvmpipe_still_judges_the_other_words(self):
        for i, wrong in ((4, 0), (6, 0), (0, 0), (7, 0)):   # a folded NaN check, a lost -0, a lost NaN conversion
            words = list(self.LLVMPIPE_WORDS)
            words[i] = wrong
            self.assertNotEqual(self.compute(self.LLVMPIPE, words, "8..8"), "", i)

    def test_other_devices_judge_every_word(self):
        self.assertNotEqual(self.compute("AMD Radeon RX 9070 XT", self.APPLE_WORDS), "")
        self.assertNotEqual(self.compute("AMD Radeon RX 9070 XT", self.LLVMPIPE_WORDS), "")
        self.assertEqual(self.compute("AMD Radeon RX 9070 XT", check.SIGNED_ZERO_NAN), "")

    def test_only_cs_signed_zero_nan_has_unjudged_words(self):
        out = "device: Apple M1 Max (subgroup 32..32)\n"
        self.assertEqual(check.judged_words("cs_store", out, check.CS_STORE), check.CS_STORE)

    def test_run_gpu_tests_uses_the_same_rule(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_gpu_tests
        for device, words in (("Apple M1 Max (subgroup 32..32)", self.APPLE_WORDS),
                              (self.LLVMPIPE + " (subgroup 8..8)", self.LLVMPIPE_WORDS)):
            msg = f"device: {device}\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            want = check.judged_words("cs_signed_zero_nan", msg, check.SIGNED_ZERO_NAN)
            ok, _ = run_gpu_tests.judge_cs_dump(0, msg, want, wave64=False, partial=False, emulated=False)
            self.assertTrue(ok, device)
            ok, _ = run_gpu_tests.judge_cs_dump(0, msg, check.SIGNED_ZERO_NAN, wave64=False, partial=False,
                                                emulated=False)
            self.assertFalse(ok, device)


class TextureMipChecks(unittest.TestCase):
    def test_mip_draw_reports_unsupported_device_and_preserves_other_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / ("vk_texture_test" + check.EXE)).touch()
            for shader in (check.TEXTURE_MIP_TEST, *check.TEXTURE_MIP_DEPENDENCIES):
                (directory / (shader + ".spv")).touch()
            with patch.object(check, "run", return_value=(77, "unsupported device")):
                self.assertIsInstance(check.texture_mip_test(tmp, tmp), check.Skip)
            with patch.object(check, "run", return_value=(1, "device lost")):
                failure = check.texture_mip_test(tmp, tmp)
                self.assertNotIsInstance(failure, check.Skip)
                self.assertIn("device lost", failure)

    def test_implicit_mip_checks_every_channel_through_last_pixel(self):
        pixel = struct.pack("<4f", 0, 1, 1, 1)
        image = pixel * (64 * 64)
        self.assertEqual(check.texture_mip_mismatch(image, 1), "")
        for channel in range(4):
            damaged = bytearray(image)
            struct.pack_into("<f", damaged, len(image) - 16 + 4 * channel, 0.5)
            self.assertTrue(check.texture_mip_mismatch(damaged, 1).startswith("(63,63):"))
        self.assertTrue(check.texture_mip_mismatch(image[:-1], 1))


class GradientSampleChecks(unittest.TestCase):
    def test_device_without_min_lod_is_a_skip(self):
        answer = 77, "unsupported: shaderResourceMinLod (the shader clamps its LOD with MinLod)\n"
        with patch.object(check, "run", return_value=answer):
            result = check.texture_gradient_a16_test("tools", "spirv", "ps_texture_grad_cl_a16")
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported: shaderResourceMinLod (the shader clamps its LOD with MinLod))")
        with patch.object(check, "run", return_value=(1, "device lost")):
            failure = check.texture_gradient_a16_test("tools", "spirv", "ps_texture_grad_cl_a16")
        self.assertNotIsInstance(failure, check.Skip)
        self.assertIn("device lost", failure)

class StoreWordsChecks(unittest.TestCase):
    """TEST-184: the image atomics and image_get_resinfo fixtures, judged from vk_store_test's three word lines."""

    def test_words_of_the_asm_headers(self):
        uint, sint, volume = check.store_words("cs_imgatomic")
        self.assertEqual(uint[:17], [0x54, 0xFFFFFF9D, 6, 0xF067, 0x97, 0xFFFFFFF0, 106, 107, 200, 109, 5,
                                     0xFFFFFFF0, 112, 0xC0FFEE, 114, 122, 180])
        self.assertEqual(uint[17:], [100 + i for i in range(17, 64)])
        self.assertEqual(sint[:16], [*range(100, 115), 7])
        self.assertEqual(sint[16:], [0x100000000 - 2 * i for i in range(16, 64)])
        self.assertEqual(volume, [200 + i for i in range(64)])
        uint, sint, volume = check.store_words("cs_imgatomic3d")
        self.assertEqual(uint, [100 + i for i in range(64)])
        self.assertEqual(sint[:6], [257, 219, 244, 206, 230, 0xFF])
        # texel i = 16z + 4y + x of the six cases: (1,2,3), (3,0,1), (0,3,2), (2,1,0), (2,3,1), (3,1,2)
        self.assertEqual({i: w for i, w in enumerate(volume) if w != 200 + i},
                         {57: 257 + 16, 19: 3, 6: 0xD00D, 39: 239 ^ 0xFF})
        self.assertEqual((volume[44], volume[30]), (244, 230))      # smax with -16 and the missed cmpswap keep theirs
        uint, sint, volume = check.store_words("cs_resinfo")
        self.assertEqual(uint[:24], [16, 4, 0, 5, 8, 2, 0, 5, 2, 1, 0, 5, 4, 1, 0xABCD, 115, 2, 5, 5, 119, 0, 0, 0, 5])
        self.assertEqual((sint, volume), ([-2 * i & 0xFFFFFFFF for i in range(64)], [200 + i for i in range(64)]))

    def test_expected_words_pass_and_any_other_word_fails(self):
        for name in check.STORE_WORDS:
            with self.subTest(name=name):
                self.assertEqual(check.store_words_problem(words_output(name), name), "")
                for image in check.STORE_IMAGES:
                    for index in (0, 20, 63):     # a word the fixture changes or one it must leave alone
                        problem = check.store_words_problem(words_output(name, flip=(image, index)), name)
                        self.assertIn(f"1 of 64 {image} words differ; first word {index}:", problem)

    def test_one_fixture_is_not_another(self):
        for printed, judged_as in (("cs_resinfo", "cs_imgatomic"), ("cs_imgatomic", "cs_imgatomic3d"),
                                   ("cs_imgatomic3d", "cs_resinfo")):
            with self.subTest(printed=printed, judged_as=judged_as):
                self.assertIn("uint words differ", check.store_words_problem(words_output(printed), judged_as))

    def test_missing_or_short_lines_fail(self):
        full = words_output("cs_imgatomic")
        self.assertIn("printed no volume words", check.store_words_problem(full.rsplit("volume", 1)[0], "cs_imgatomic"))
        self.assertIn("printed no uint words", check.store_words_problem("device: fake\n", "cs_imgatomic"))
        short = full.replace(" 0x00000054", "", 1)                   # 63 words in the first line
        self.assertIn("printed no uint words", check.store_words_problem(short, "cs_imgatomic"))

    def test_the_tool_runs_in_words_mode_and_its_exit_codes_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cs_imgatomic.spv").touch()
            with patch.object(check, "run", return_value=(0, words_output("cs_imgatomic"))) as run:
                self.assertEqual(check.store_words_test("tools", tmp, "cs_imgatomic"), "")
            self.assertIn("vk_store_test", run.call_args.args[0])
            self.assertEqual(run.call_args.args[2:], ("words",))
            answer = 77, "unsupported: shaderStorageImageWriteWithoutFormat=0 shaderInt64=1\n"
            with patch.object(check, "run", return_value=answer):
                skip = check.store_words_test("tools", tmp, "cs_imgatomic")
            self.assertIsInstance(skip, check.Skip)
            self.assertIn("shaderStorageImageWriteWithoutFormat=0", skip)
            with patch.object(check, "run", return_value=(1, "FAILED vkCreateComputePipelines -> -3\n")):
                failure = check.store_words_test("tools", tmp, "cs_imgatomic")
            self.assertNotIsInstance(failure, check.Skip)
            self.assertIn("exit 1", failure)
            self.assertEqual(check.store_words_test("tools", tmp, "cs_resinfo"), "missing cs_resinfo.spv")


class GetLodChecks(unittest.TestCase):
    """TEST-184: image_get_lod frames, (clamped, calculated) pairs within a sixteenth of a level."""

    @staticmethod
    def frame(pixel):
        return struct.pack("<4f", *pixel) * (64 * 64)

    def run_test(self, name, frames, code=0, out=""):
        """get_lod_test with a fake vk_texture_test that writes frames[S# word 1]; returns (result, calls)."""
        calls = []

        def harness(_tool, _vs, _fs, raw, *args, env=None):
            calls.append(args)
            if not code:
                Path(raw).write_bytes(frames[args[2]])
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            for shader in ("vk_texture_test" + check.EXE, "vs_ids.spv", name + ".spv"):
                (Path(tmp) / shader).touch()
            with patch.object(check, "run", side_effect=harness):
                return check.get_lod_test(tmp, tmp, name), calls

    def test_expected_frames_pass_within_the_tolerance(self):
        for name, draws in check.GET_LOD_TESTS.items():
            with self.subTest(name=name):
                frames = {limits: self.frame(want) for limits, _, want in draws}
                result, calls = self.run_test(name, frames)
                self.assertEqual(result, "")
                self.assertEqual([c[2] for c in calls], [limits for limits, _, _ in draws])
                self.assertEqual(["--two-mip" in c for c in calls], [two_mip for _, two_mip, _ in draws])
        want = check.GET_LOD_TESTS["ps_get_lod"][0][2]
        near = tuple(v + 1 / 32 for v in want)
        self.assertEqual(check.get_lod_problem(self.frame(near), want), "")
        self.assertIn("4096 of 4096 pixels differ", check.get_lod_problem(self.frame(tuple(v + 1 / 8 for v in want)), want))

    def test_swapped_results_an_unclamped_lod_and_bad_frames_fail(self):
        clamped_first = check.GET_LOD_TESTS["ps_get_lod"][1][2]           # (0, 1, 3, 0) under MAX_LOD 0.25
        for what, pixel in (("calculated LOD first", (1.0, 0.0, 3.0, 0.0)),
                            ("sampler limit not applied", (1.0, 1.0, 3.0, 1.0)),
                            ("LOD of the unscaled coordinate", (0.0, 1.0, 1.0, 0.0)),
                            ("NaN", (0.0, float("nan"), 3.0, 0.0))):
            with self.subTest(what):
                self.assertIn("pixels differ", check.get_lod_problem(self.frame(pixel), clamped_first))
        one_bad = bytearray(self.frame(clamped_first))
        struct.pack_into("<f", one_bad, 16 * (64 * 5 + 7) + 4, 0.5)
        self.assertIn("1 of 4096 pixels differ", check.get_lod_problem(bytes(one_bad), clamped_first))
        self.assertIn("first at (7,5)", check.get_lod_problem(bytes(one_bad), clamped_first))
        self.assertIn("got 65520 bytes", check.get_lod_problem(self.frame(clamped_first)[:-16], clamped_first))

    def test_second_draw_is_judged_and_named(self):
        draws = check.GET_LOD_TESTS["ps_get_lod"]
        frames = {limits: self.frame(draws[0][2]) for limits, _, _ in draws}    # the wide sampler's frame twice
        result, _ = self.run_test("ps_get_lod", frames)
        self.assertTrue(result.startswith("S# word 1 40000: 4096 of 4096 pixels differ"), result)

    def test_unsupported_device_skips_and_other_failures_fail(self):
        result, _ = self.run_test("ps_get_lod", {}, code=77,
                                  out="unsupported: need 144 push constant bytes, device has 128\n")
        self.assertIsInstance(result, check.Skip)
        self.assertIn("144 push constant bytes", result)
        result, _ = self.run_test("ps_get_lod", {}, code=1, out="FAILED vkCreateGraphicsPipelines -> -3\n")
        self.assertNotIsInstance(result, check.Skip)
        self.assertIn("exit 1", result)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check.get_lod_test(tmp, tmp, "ps_get_lod"), "missing vk_texture_test" + check.EXE)


class TexelOffsetChecks(unittest.TestCase):
    """ps_texoff (image_sample*_o, image_gather4_lz_o): the judgement of a dump, and the skip without maintenance8."""
    LIMITS = "device: gpu\ntexel offset limits: -64..63, gather -32..31\n"

    @staticmethod
    def dump(words=check.texoff_words):
        return b"".join(struct.pack("<4I", *words(i % 64, i // 64)) for i in range(64 * 64))

    def test_hand_derived_words_of_the_asm_header(self):
        self.assertEqual(check.texoff_words(0, 0), (0x00, 0x0761, 0x0000, 0x0000))
        self.assertEqual(check.texoff_words(20, 50), (0x62, 0x0761, 0x1221, 0x5566))
        self.assertEqual(check.texoff_words(63, 63), (0x77, 0x0761, 0x6776, 0x6677))

    def test_expected_dump_passes_and_wrong_offsets_fail(self):
        self.assertEqual(check.texoff_problem(self.dump(), self.LIMITS), ("", "offsets +16/-32 judged"))
        for what, words in (
            ("offset dropped", lambda x, y: (0x44, *check.texoff_words(x, y)[1:])),
            ("x and y swapped", lambda x, y: (y >> 3 | x >> 3 << 4, *check.texoff_words(x, y)[1:])),
            ("constant offset wrong", lambda x, y: (x >> 3 | y >> 3 << 4, 0x0744, *check.texoff_words(x, y)[2:])),
            ("gather offset dropped", lambda x, y: (*check.texoff_words(x, y)[:2], 0x3443, 0x3344)),
            ("descriptor miss", lambda x, y: (0x08, 0x0808, 0x8888, 0x0000)),
        ):
            with self.subTest(what):
                self.assertIn("pixels differ", check.texoff_problem(self.dump(words), self.LIMITS)[0])
        self.assertIn("pixel count", check.texoff_problem(self.dump()[:-16], self.LIMITS)[0])

    def test_wide_offsets_are_judged_only_inside_the_device_limits(self):
        wide_wrong = self.dump(lambda x, y: (x >> 3 | y >> 3 << 4, 0x4461, *check.texoff_words(x, y)[2:]))
        self.assertIn("pixels differ", check.texoff_problem(wide_wrong, self.LIMITS)[0])
        for out in ("texel offset limits: -8..7, gather -8..7\n", "no limits printed\n"):
            problem, covered = check.texoff_problem(wide_wrong, out)
            self.assertEqual(problem, "")
            self.assertIn("not judged", covered)
            self.assertIn("pixels differ", check.texoff_problem(self.dump(lambda x, y: (0x44, 0x0761, 0, 0)), out)[0])

    def test_device_without_maintenance8_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_texoff.spv").touch()
            answer = 77, "unsupported: VK_KHR_maintenance8 (non-constant texel Offset)\n"
            with patch.object(check, "run", return_value=answer) as run:
                result = check.pixel_test("tools", tmp, "ps_texoff")
            self.assertEqual(run.call_args.kwargs["env"]["VKT_NEED_MAINTENANCE8"], "1")
            self.assertIn("vk_texture_test", run.call_args.args[0])
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported: VK_KHR_maintenance8 (non-constant texel Offset))")


class Wave32LaneDumps(unittest.TestCase):
    """cs_wqm_w32 and cs_w32_lane_wrap need 32 lanes: skipped where the widest subgroup is narrower (llvmpipe: 8)."""
    def run_dump(self, name, code, out):
        want = dict(check.CS_DUMPS)[name]
        with patch.object(check, "run", return_value=(code, out)):
            return check.compute_test("tools", "spirv", name, "1", "dump", expected_words=want,
                                      needs_32_lanes=name in check.WAVE32_LANE_DUMPS)

    @staticmethod
    def words(name, flip=False):
        w = list(dict(check.CS_DUMPS)[name])
        if flip:
            w[0] ^= 1
        return "out[0..7] = " + " ".join(map(str, (w + [0] * 8)[:8])) + "\n"

    def test_lane_dumps_are_wave32_dumps(self):
        self.assertTrue(set(check.WAVE32_LANE_DUMPS) <= set(check.WAVE32_DUMPS))
        self.assertTrue(set(check.WAVE32_LANE_DUMPS) <= set(dict(check.CS_DUMPS)))

    def test_skipped_below_32_lanes_whatever_the_words(self):
        for name in check.WAVE32_LANE_DUMPS:
            for flip in (False, True):
                with self.subTest(name=name, flip=flip):
                    problem = self.run_dump(name, 0, "device: llvmpipe (subgroup 8..8)\n" + self.words(name, flip))
                    self.assertIsInstance(problem, check.Skip)
                    self.assertEqual(problem, "SKIP (subgroup 8 is narrower than 32 lanes)")

    def test_judged_on_32_lanes_and_wider(self):
        for name in check.WAVE32_LANE_DUMPS:
            for dev in ("device: gpu (subgroup 32..32)\n", "device: gpu (subgroup 32..64)\n",
                        "device: gpu (subgroup 8..32)\n", ""):
                with self.subTest(name=name, dev=dev):
                    self.assertEqual(self.run_dump(name, 0, dev + self.words(name)), "")
                    problem = self.run_dump(name, 0, dev + self.words(name, flip=True))
                    self.assertTrue(problem)
                    self.assertNotIsInstance(problem, check.Skip)

    def test_other_dumps_are_not_skipped_on_narrow_devices(self):
        out = "device: llvmpipe (subgroup 8..8)\nout[0..7] = 1 2 3 4 5 6 7 8\n"
        with patch.object(check, "run", return_value=(0, out)):
            problem = check.compute_test("tools", "spirv", "cs_store", "1", "dump", expected_words=check.CS_STORE)
        self.assertTrue(problem)
        self.assertNotIsInstance(problem, check.Skip)

    def test_lane_dumps_request_32_lanes(self):
        # #774: like run_gpu_tests.py, so an 8..32 device (Intel ANV) doesn't judge the dump at a narrower size.
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name), patch.object(check, "run", return_value=(0, self.words(name))) as run:
                check.compute_test("tools", "spirv", name, "1", "dump", expected_words=dict(check.CS_DUMPS)[name],
                                   needs_32_lanes=True)
                self.assertEqual(run.call_args.kwargs["env"]["VKC_SUBGROUP_SIZE"], "32")

    def test_other_dumps_keep_the_default_size(self):
        with patch.object(check, "run", return_value=(0, self.words("cs_store"))) as run:
            check.compute_test("tools", "spirv", "cs_store", "1", "dump", expected_words=check.CS_STORE)
        self.assertNotIn("env", run.call_args.kwargs)

    def test_recorded_wave_size_is_requested(self):
        # cs_barrier_loop is wave32 but not a lane dump: left to the default it ran 64-wide on an RX 9070 (32..64),
        # two waves instead of four, and failed in check.py while run_gpu_tests.py (which reads index.json) passed.
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "index.json").write_text(json.dumps({"cs_barrier_loop": {"subgroup_size": 32},
                                                           "cs_store": {}}), encoding="utf-8")
            with patch.object(check, "run", return_value=(0, "out[0..7] = 128 96 64 32 128 96 64 0\n")) as run:
                self.assertEqual(check.compute_test("tools", tmp, "cs_barrier_loop", "1", "dump",
                                                    expected_words=[128, 96, 64, 32, 128, 96, 64]), "")
            self.assertEqual(run.call_args.kwargs["env"]["VKC_SUBGROUP_SIZE"], "32")
            with patch.object(check, "run", return_value=(0, self.words("cs_store"))) as run:
                check.compute_test("tools", tmp, "cs_store", "1", "dump", expected_words=check.CS_STORE)
            self.assertNotIn("env", run.call_args.kwargs)                 # nothing recorded: the default size
            with patch.object(check, "run", return_value=(0, "out[0..7] = 0 0 0 0 0 0 0 0\n")) as run:
                check.compute_test("tools", tmp, "cs_barrier_loop", "1", "dump", expected_words=[1],
                                   env={"VKC_SUBGROUP_SIZE": "64"})
            self.assertEqual(run.call_args.kwargs["env"]["VKC_SUBGROUP_SIZE"], "64")   # a caller's choice wins

    def test_recorded_wave_size_that_cannot_be_pinned_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "index.json").write_text(json.dumps({"cs_barrier_loop": {"subgroup_size": 32}}),
                                               encoding="utf-8")
            out = "compute subgroup size: 32 not available\nout[0..7] = 0 0 0 0 0 0 0 0\n"
            with patch.object(check, "run", return_value=(0, out)):
                problem = check.compute_test("tools", tmp, "cs_barrier_loop", "1", "dump", expected_words=[128])
        self.assertIsInstance(problem, check.Skip)

    def test_skipped_when_32_lanes_cannot_be_pinned(self):
        for name in check.WAVE32_LANE_DUMPS:
            for flip in (False, True):
                with self.subTest(name=name, flip=flip):
                    out = ("device: gpu (subgroup 8..32)\ncompute subgroup size: 32 not available\n"
                           + self.words(name, flip))
                    problem = self.run_dump(name, 0, out)
                    self.assertIsInstance(problem, check.Skip)
                    self.assertIn("at 32 lanes on request", problem)

    def test_judged_when_32_lanes_are_pinned(self):
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name):
                out = "device: gpu (subgroup 8..32)\ncompute subgroup size: 32\n"
                self.assertEqual(self.run_dump(name, 0, out + self.words(name)), "")
                self.assertNotIsInstance(self.run_dump(name, 0, out + self.words(name, flip=True)), check.Skip)

    def test_judged_when_32_lanes_are_observed(self):
        # MoltenVK: no compute stage in requiredSubgroupSizeStages, but vk_compute_test's cs_wave_probe saw 32 lanes.
        out = "device: Apple M3 Max (subgroup 4..32)\ncompute subgroup size: 32 (observed: all 64 invocations of a " \
              "64x1x1 group, full subgroups)\n"
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name):
                self.assertEqual(self.run_dump(name, 0, out + self.words(name)), "")
                problem = self.run_dump(name, 0, out + self.words(name, flip=True))
                self.assertTrue(problem)
                self.assertNotIsInstance(problem, check.Skip)

    def test_failed_when_the_observed_size_is_wrong(self):
        out = ("device: gpu (subgroup 4..32)\ncompute subgroup size: 32 observed wrong in 64 of 64 invocations; "
               "invocation 0 has SubgroupSize 32, lane 0, subgroup 0, a ballot of 16 lanes\n")
        for name in check.WAVE32_LANE_DUMPS:
            with self.subTest(name=name):
                problem = self.run_dump(name, 1, out)
                self.assertNotIsInstance(problem, check.Skip)
                self.assertIn("observed wrong", problem)


class SamplerMipChecks(unittest.TestCase):
    """The S#/T# oracle check.py judges vk_sampler_test with, against hand-worked levels."""

    def test_levels_follow_the_s_and_t_fields(self):
        want = {"point": (0, 1), "point max_lod 0": (0, 0), "point min_lod 1": (1, 1), "point bias +1": (1, 1),
                "point bias -1": (0, 0), "none": (0, 0), "none min_lod 1": (0, 0), "point base 1": (1, 1)}
        self.assertEqual({name for name, _, _ in check.SAMPLER_CASES}, set(want))
        for name, words, view in check.SAMPLER_CASES:
            with self.subTest(name=name):
                self.assertEqual((check.sampler_level(words, view, 0), check.sampler_level(words, view, 1)), want[name])

    def test_s_words_put_the_fields_where_mesa_does(self):
        self.assertEqual(check.ssharp_words(1, min_lod=1, max_lod=15, bias=-1), (0x92, 0xf00100, 0x04003f00, 0))

    @staticmethod
    def image(level_lz, level_implicit):
        return b"".join(struct.pack("<4f", *check.sampler_texel(level_lz, x, y), *check.sampler_texel(level_implicit, x, y))
                        for y in range(64) for x in range(64))

    def test_every_channel_and_the_last_pixel_are_judged(self):
        words, view = check.ssharp_words(1), (0, 1)
        good = self.image(0, 1)
        self.assertEqual(check.sampler_mismatch(good, words, view), "")
        for channel in range(4):
            bad = bytearray(good)
            struct.pack_into("<f", bad, len(good) - 16 + 4 * channel, 0.25)
            self.assertTrue(check.sampler_mismatch(bytes(bad), words, view).startswith("(63,63):"))
        self.assertIn("bytes", check.sampler_mismatch(good[:-16], words, view))

    def test_nan_in_any_sampler_channel_fails(self) -> None:
        words, view = check.ssharp_words(1), (0, 1)
        good = self.image(0, 1)
        self.assertEqual(check.sampler_mismatch(good, words, view), "")
        for channel in range(4):
            with self.subTest(channel=channel):
                bad = bytearray(good)
                struct.pack_into("<I", bad, len(good) - 16 + 4 * channel, 0x7FC00000)
                self.assertTrue(check.sampler_mismatch(bytes(bad), words, view).startswith("(63,63):"))

    def test_a_wrong_level_or_swapped_uv_fails_where_predicted(self):
        words, view = check.ssharp_words(1), (0, 1)
        self.assertTrue(check.sampler_mismatch(self.image(1, 1), words, view).startswith("(0,0):"))
        self.assertTrue(check.sampler_mismatch(self.image(0, 0), words, view).startswith("(0,0):"))
        swapped = b"".join(struct.pack("<4f", *check.sampler_texel(0, y, x), *check.sampler_texel(1, y, x))
                           for y in range(64) for x in range(64))
        self.assertTrue(check.sampler_mismatch(swapped, words, view).startswith("(32,0):"))


class FmaFusedChecks(unittest.TestCase):
    """cs_fma_fused: strict single-rounded words under --fma khr, fused-or-split f32 under --fma glsl (Mesa RADV)."""
    RADV_SPLIT = [0x00000000, 0xA8800000, 0x3A000000, 0x00000000, 0x00000000, 0x00000000, 0x28800000, 0x00007351]

    def run_fma(self, words, *, khr_index=False, fma_mode=None, env_fma=None, code=0, raw_out=None):
        out = raw_out if raw_out is not None else "out[0..7] = " + " ".join(map(str, words)) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            if khr_index or fma_mode:
                meta = {"features": ["shaderFmaFloat32"]} if khr_index else {}
                if fma_mode:
                    meta["fma"] = fma_mode
                Path(tmp, "index.json").write_text(
                    json.dumps({"cs_fma_fused": meta}), encoding="utf-8")
            with patch.dict(check.os.environ, {}, clear=False):
                check.os.environ.pop("LEONIDA_FMA", None)
                if env_fma:
                    check.os.environ["LEONIDA_FMA"] = env_fma
                with patch.object(check, "run", return_value=(code, out)) as run:
                    res = check.fma_fused_test("tools", tmp)
                self.assertEqual(run.call_args.kwargs["env"]["VKC_SGPRS"], check.FMA_FUSED_ENV["VKC_SGPRS"])
        return res

    def test_fused_passes_under_both_modes(self):
        self.assertEqual(self.run_fma(check.FMA_FUSED, khr_index=False), "")
        self.assertEqual(self.run_fma(check.FMA_FUSED, khr_index=True), "")

    def test_radv_split_passes_under_glsl_and_fails_under_khr(self):
        self.assertEqual(self.run_fma(self.RADV_SPLIT, khr_index=False), "")
        ok, detail = check.judge_fma_fused(self.RADV_SPLIT, khr=False)
        self.assertTrue(ok)
        self.assertIn("5/7 split", detail)
        problem = self.run_fma(self.RADV_SPLIT, khr_index=True)
        self.assertIn("word 0", problem)
        self.assertIn("OpFmaKHR", problem)

    def test_fma_mode_recorded_in_index_is_honoured(self):
        # TEST-1147: an --fma exact build has no FMAKHR capability, so without the recorded mode a fresh shell
        # judges split words as a legal GLSL Fma result instead of a regression.
        for mode in ("glsl", "khr", "exact"):
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "index.json").write_text(
                    json.dumps({"cs_fma_fused": {"fma": mode}}), encoding="utf-8")
                self.assertEqual(check.shader_fma_mode(tmp, "cs_fma_fused"), mode)
        self.assertEqual(self.run_fma(check.FMA_FUSED, fma_mode="exact"), "")
        problem = self.run_fma(self.RADV_SPLIT, fma_mode="exact")
        self.assertIn("word 0", problem)
        self.assertNotIn("LEONIDA_FMA=khr", check.judge_fma_fused(self.RADV_SPLIT, khr=True)[1])

    def test_recorded_mode_overrides_stale_environment(self):
        # When index.json records the mode it is authoritative; a leftover LEONIDA_FMA from the build shell
        # must not switch a glsl build to strict judging (or an exact build back to lenient).
        self.assertEqual(self.run_fma(self.RADV_SPLIT, fma_mode="glsl", env_fma="exact"), "")
        problem = self.run_fma(self.RADV_SPLIT, fma_mode="exact", env_fma="glsl")
        self.assertIn("word 0", problem)

    def test_older_index_still_honors_exact_env(self):
        # An index written before the fma field existed can only be judged strictly via the environment.
        self.assertEqual(self.run_fma(check.FMA_FUSED, env_fma="exact"), "")
        problem = self.run_fma(self.RADV_SPLIT, env_fma="exact")
        self.assertIn("word 0", problem)
        self.assertIn("--fma exact", problem)

    def test_wrong_f16_or_garbage_f32_fails_under_both_modes(self):
        bad_f16 = check.FMA_FUSED[:7] + [0x00007352]
        bad_f32 = [0x12345678] + check.FMA_FUSED[1:]
        for khr in (False, True):
            self.assertIn("word 7", self.run_fma(bad_f16, khr_index=khr))
            self.assertIn("word 0", self.run_fma(bad_f32, khr_index=khr))

    def test_missing_shader_fma_extension_skips(self):
        res = self.run_fma([], khr_index=True, code=77, raw_out="unsupported: cs_fma_fused.spv needs VK_KHR_shader_fma\n")
        self.assertIsInstance(res, check.Skip)
        self.assertIn("VK_KHR_shader_fma", res)

if __name__ == "__main__":
    unittest.main()


class TexelOffsetFormsChecks(unittest.TestCase):
    """ps_texoff_forms (image_sample_o, image_sample_b_o, image_gather4_l_o): the judgement of a dump."""

    @staticmethod
    def dump(words):
        return b"".join(struct.pack("<4I", *words(i % 64, i // 64)) for i in range(64 * 64))

    def test_the_rule_passes_and_both_samples_are_judged(self):
        good = self.dump(lambda px, py: check.texoff_words(px, py, "ps_texoff_forms"))
        problem, covered = check.texoff_problem(good, "device: gpu\n", name="ps_texoff_forms")
        self.assertEqual(problem, "")
        self.assertIn("image_sample_b_o", covered)
        self.assertEqual(check.texoff_words(20, 50, "ps_texoff_forms"), (0x62, 0x62, 0x1221, 0x5566))

    def test_a_dropped_or_swapped_offset_fails(self):
        for label, broken in (("dropped in .y", lambda px, py: (lambda w: (w[0], 0x44, w[2], w[3]))(
                                  check.texoff_words(px, py, "ps_texoff_forms"))),
                              ("swapped in .x", lambda px, py: (lambda w: ((py >> 3) | (px >> 3) << 4, *w[1:]))(
                                  check.texoff_words(px, py, "ps_texoff_forms")))):
            with self.subTest(label):
                problem, _ = check.texoff_problem(self.dump(broken), "device: gpu\n", name="ps_texoff_forms")
                self.assertIn("pixels differ", problem)


class PixelTestRegistry(unittest.TestCase):
    def test_pixel_tests_include_every_pixel_fixture(self):
        # A repeated PIXEL_TESTS assignment silently dropped ps_texoff once; keep the registry explicit.
        for name in ("ps_texoff", "ps_texoff_forms", "ps_wqm", "ps_scc_mask32"):
            with self.subTest(name=name):
                self.assertIn(name, check.PIXEL_TESTS)
        self.assertEqual(len(check.PIXEL_TESTS), len(set(check.PIXEL_TESTS)))

    def test_pixel_test_runs_the_harness_once(self):
        # A duplicated pixels() line ran the whole shader twice and could hide a first-call failure behind a
        # second-call success.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_legacy_mul.spv").touch()
            want = check.PIXEL_BITS["ps_legacy_mul"]
            with patch.object(check, "pixels", return_value=([want], "")) as pixels:
                self.assertEqual(check.pixel_test("tools", tmp, "ps_legacy_mul"), "")
            self.assertEqual(pixels.call_count, 1)
