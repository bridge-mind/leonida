"""make_game.py --wave64-emulate auto|on|off (SHADER-11, #94): when the build translates with --wave64-emulate.

Standard library only; no GPU, no Vulkan SDK, no game data: vulkaninfo and the translator are mocked, and the
vulkaninfo text below is a hand-written excerpt of its format.

    python -I tests/test_make_game_wave64.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def vulkaninfo_text(*max_sizes):
    """vulkaninfo's plain-text layout: one block per device, each with a maxSubgroupSize line."""
    blocks = []
    for i, size in enumerate(max_sizes):
        blocks.append(f"GPU{i}:\n"
                      "VkPhysicalDeviceSubgroupSizeControlProperties:\n"
                      "-----------------------------------------------\n"
                      f"\tminSubgroupSize              = {min(size, 32)}\n"
                      f"\tmaxSubgroupSize              = {size}\n"
                      "\tmaxComputeWorkgroupSubgroups = 1024\n")
    return "\n".join(blocks)


class SmallestMaxSubgroupTest(unittest.TestCase):
    def test_one_device(self):
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(32)), 32)
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64)), 64)

    def test_several_devices_give_the_smallest(self):
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64, 32)), 32)
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64, 64)), 64)

    def test_json_output(self):
        self.assertEqual(make_game.smallest_max_subgroup('{"minSubgroupSize": 32, "maxSubgroupSize": 64}'), 64)

    def test_no_subgroup_size_gives_none(self):
        self.assertIsNone(make_game.smallest_max_subgroup(""))
        self.assertIsNone(make_game.smallest_max_subgroup("ERROR: vkCreateInstance failed\n"))
        self.assertIsNone(make_game.smallest_max_subgroup("\tminSubgroupSize = 32\n"))


class Wave64EmulateTest(unittest.TestCase):
    """wave64_emulate(): on and off are taken as given; auto asks vulkaninfo and falls back to on."""

    def decide(self, mode, which="vulkaninfo", stdout="", error=None):
        def fake_run(args, **kw):
            if error:
                raise error
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
        with mock.patch.object(make_game.shutil, "which", return_value=which), \
                mock.patch.object(make_game.subprocess, "run", side_effect=fake_run) as run, \
                mock.patch.object(make_game, "log", lambda *a: None):
            result = make_game.wave64_emulate(mode)
        asked_vulkaninfo = any(call.args[0] == [which] for call in run.call_args_list)
        return result, asked_vulkaninfo

    def test_on_and_off_never_ask_vulkaninfo(self):
        self.assertEqual(self.decide("on", stdout=vulkaninfo_text(64)), (True, False))
        self.assertEqual(self.decide("off", stdout=vulkaninfo_text(32)), (False, False))

    def test_auto_32_wide_turns_it_on(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(32)), (True, True))

    def test_auto_64_wide_leaves_it_off(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(64)), (False, True))

    def test_auto_with_a_64_and_a_32_wide_gpu_turns_it_on(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(64, 32))[0], True)

    def test_auto_without_vulkaninfo_turns_it_on(self):
        self.assertEqual(self.decide("auto", which=None), (True, False))

    def test_auto_when_vulkaninfo_cannot_run_turns_it_on(self):
        self.assertTrue(self.decide("auto", error=OSError("cannot start"))[0])
        self.assertTrue(self.decide("auto", error=subprocess.TimeoutExpired("vulkaninfo", 120))[0])

    def test_auto_when_vulkaninfo_prints_no_subgroup_size_turns_it_on(self):
        self.assertTrue(self.decide("auto", stdout="ERROR: no Vulkan device\n")[0])


REAL_FORMAT_BLOCK = """VkPhysicalDeviceProperties:
	deviceName        = {name}
VkPhysicalDeviceVulkan13Properties:
-----------------------------------
	minSubgroupSize                                                               = 32
	maxSubgroupSize                                                               = {size}
	maxComputeWorkgroupSubgroups                                                  = 4294967295
	requiredSubgroupSizeStages: count = {n}
{stages}
	maxInlineUniformBlockSize                                                     = 65536
"""


def real_format(*devices):
    """vulkaninfo's text as a real run prints it (excerpt): per device its name and the 1.3 properties block with
    the required-stage list. devices: (name, maxSubgroupSize, [stage names])."""
    return "\n".join(REAL_FORMAT_BLOCK.format(name=n, size=s, n=len(st), stages="\n".join("\t\t" + x for x in st))
                     for n, s, st in devices)


ALL = ["SHADER_STAGE_FRAGMENT_BIT", "SHADER_STAGE_COMPUTE_BIT", "SHADER_STAGE_TASK_BIT_EXT",
       "SHADER_STAGE_MESH_BIT_EXT"]


class DetectionGapsTest(unittest.TestCase):
    """Where auto's detection is less strict than vk_pipeline_check's can_require_sg, native shaders would run
    inexact without any error."""

    def test_a_device_that_prints_no_subgroup_block_is_not_ignored(self):
        text = real_format(("Big GPU", 64, ALL)) + "VkPhysicalDeviceProperties:\n\tdeviceName        = Old GPU\n"
        self.assertIsNone(make_game.smallest_max_subgroup(text))      # one device says nothing: unknown, so on

    def test_a_layered_api_name_is_not_another_device(self):
        # VK_KHR_maintenance7 lists a layered driver's underlying API (MoltenVK on Metal) with its own deviceName,
        # indented deeper and printed before the subgroup block: it is the same device, not one without a size.
        layered = ("GPU0:\nVkPhysicalDeviceProperties:\n\tdeviceName        = Apple M2 Max\n"
                   "VkPhysicalDeviceLayeredApiPropertiesListKHR:\n\tpLayeredApis: count = 1\n\t\t0:\n"
                   "\t\t\tlayeredAPI = PHYSICAL_DEVICE_LAYERED_API_METAL_KHR\n\t\t\tdeviceName = Apple M2 Max\n"
                   "VkPhysicalDeviceSubgroupSizeControlProperties:\n\tmaxSubgroupSize = 32\n")
        self.assertEqual(make_game.smallest_max_subgroup(layered), 32)
        layered_json = ('{"VkPhysicalDeviceProperties": {\n\t\t"deviceName": "Apple M2 Max"},\n'
                        '\t"VkPhysicalDeviceLayeredApiPropertiesListKHR": {"pLayeredApis": [\n'
                        '\t\t\t{"layeredAPI": 3,\n\t\t\t "deviceName": "Apple M2 Max"}]},\n'
                        '\t"VkPhysicalDeviceSubgroupSizeControlProperties": {\n\t\t"maxSubgroupSize": 32}}')
        self.assertEqual(make_game.smallest_max_subgroup(layered_json), 32)

    def test_a_device_printing_the_size_twice_does_not_hide_one_that_prints_none(self):
        twice = ("VkPhysicalDeviceProperties:\n\tdeviceName        = A\n"
                 "VkPhysicalDeviceSubgroupSizeControlProperties:\n\tmaxSubgroupSize = 64\n"
                 "VkPhysicalDeviceVulkan13Properties:\n\tmaxSubgroupSize = 64\n")
        never = "VkPhysicalDeviceProperties:\n\tdeviceName        = B\n"
        self.assertEqual(make_game.smallest_max_subgroup(twice), 64)
        self.assertIsNone(make_game.smallest_max_subgroup(twice + never))
        self.assertIsNone(make_game.smallest_max_subgroup(never + twice))

    def test_a_device_printing_the_size_twice_counts_once(self):
        a = ("VkPhysicalDeviceProperties:\n\tdeviceName = A\n\tmaxSubgroupSize = 64\n\tmaxSubgroupSize = 64\n")
        b = "VkPhysicalDeviceProperties:\n\tdeviceName = B\n\tmaxSubgroupSize = 32\n"
        self.assertEqual(make_game.smallest_max_subgroup(a + b), 32)

    def test_real_format_still_reads(self):
        self.assertEqual(make_game.smallest_max_subgroup(real_format(("A", 64, ALL))), 64)
        self.assertEqual(make_game.smallest_max_subgroup(real_format(("A", 64, ALL), ("B", 32, ALL))), 32)

    def decide(self, vulkaninfo, tool, *, mesh_compute=False):
        def fake_run(args, **kw):
            if os.path.basename(str(args[-1])) == "vk_subgroup_info.py":
                return subprocess.CompletedProcess(args, 0 if "native64" in tool else 1, stdout=tool, stderr="")
            return subprocess.CompletedProcess(args, 0, stdout=vulkaninfo, stderr="")
        with mock.patch.object(make_game.shutil, "which", return_value="vulkaninfo"), \
                mock.patch.object(make_game.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(make_game, "log", lambda *a: None):
            return make_game.wave64_emulate("auto", mesh_compute=mesh_compute)

    def test_64_wide_but_without_the_mesh_stage_bit_turns_emulation_on(self):
        no_mesh = [s for s in ALL if "MESH" not in s]
        tool = '{"native64": false, "mesh_in_required_stages": false, "compute_in_required_stages": true}'
        self.assertTrue(self.decide(real_format(("A", 64, no_mesh)), tool))

    def test_64_wide_without_the_subgroup_size_control_feature_turns_emulation_on(self):
        self.assertTrue(self.decide(real_format(("A", 64, ALL)), '{"native64": false, "subgroup_size_control": false}'))

    def test_64_wide_and_confirmed_by_the_runtime_check_leaves_it_off(self):
        self.assertFalse(self.decide(real_format(("A", 64, ALL)), '{"native64": true}'))

    def test_mesh_compute_uses_compute_wave64_capability(self):
        text = real_format(("Compute-only GPU", 64, ["SHADER_STAGE_COMPUTE_BIT"]))
        self.assertFalse(self.decide(text, '{"native64": false, "native64_compute": true}', mesh_compute=True))
        self.assertTrue(self.decide(text, '{"native64": false, "native64_compute": false}', mesh_compute=True))
        self.assertTrue(self.decide(text, '{"native64": false, "native64_compute": true}'))

    def test_an_unreadable_runtime_check_does_not_change_the_vulkaninfo_answer(self):
        self.assertFalse(self.decide(real_format(("A", 64, ALL)), '{"error": "no Vulkan loader found"}'))
        self.assertFalse(self.decide(real_format(("A", 64, ALL)), "not json"))
        self.assertTrue(self.decide(real_format(("A", 32, ALL)), '{"native64": true}'))     # vulkaninfo wins: on


class Rdna2AmdWindowsTest(unittest.TestCase):
    """#857: AMD's Windows compiler for RDNA2 crashes on emulated LDS compute shaders. When the runtime's GPU is one
    and runs wave64 natively, auto leaves emulation off even if another device (an Intel iGPU) is 32-wide."""

    RX6700 = '{"device": "AMD Radeon RX 6700", "native64": true, "rdna2_amd_windows": true}'

    def decide(self, vulkaninfo, tool, mode="auto"):
        def fake_run(args, **kw):
            if os.path.basename(str(args[-1])) == "vk_subgroup_info.py":
                return subprocess.CompletedProcess(args, 0 if "native64" in tool else 1, stdout=tool, stderr="")
            return subprocess.CompletedProcess(args, 0, stdout=vulkaninfo, stderr="")
        logged = []
        with mock.patch.object(make_game.shutil, "which", return_value="vulkaninfo"), \
                mock.patch.object(make_game.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(make_game, "log", logged.append):
            return make_game.wave64_emulate(mode), logged

    def test_intel_igpu_and_rdna2_card_leaves_it_off(self):
        on, logged = self.decide(real_format(("Intel iGPU", 32, ALL), ("AMD Radeon RX 6700", 64, ALL)), self.RX6700)
        self.assertFalse(on)
        self.assertIn("#857", logged[-1])
        self.assertIn("AMD Radeon RX 6700", logged[-1])

    def test_other_runtime_gpus_keep_emulation_on(self):
        two = real_format(("Intel iGPU", 32, ALL), ("Big GPU", 64, ALL))
        for label, tool in (("rdna3", '{"native64": true, "rdna2_amd_windows": false}'),
                            ("older_tool_without_the_key", '{"native64": true}'),
                            ("rdna2_without_native64", '{"native64": false, "rdna2_amd_windows": true}'),
                            ("unreadable", '{"error": "no Vulkan loader found"}'),
                            ("not_json", "not json")):
            with self.subTest(case=label):
                self.assertTrue(self.decide(two, tool)[0])

    def test_rdna2_only_pc_is_unchanged(self):
        self.assertFalse(self.decide(real_format(("AMD Radeon RX 6700", 64, ALL)), self.RX6700)[0])

    def test_rdna2_guard_when_vulkaninfo_is_missing_or_unreadable(self):
        for exe in (None, "vulkaninfo"):
            with self.subTest(vulkaninfo=exe), \
                    mock.patch.object(make_game.shutil, "which", return_value=exe), \
                    mock.patch.object(make_game, "runtime_gpu_info", return_value=json.loads(self.RX6700)), \
                    mock.patch.object(make_game.subprocess, "run", side_effect=OSError("cannot start")), \
                    mock.patch.object(make_game, "log") as log:
                self.assertEqual(make_game.wave64_settings("auto"), (False, False))
                self.assertIn("#857", log.call_args.args[0])

    def test_explicit_on_is_taken_as_given(self):
        self.assertTrue(self.decide(real_format(("AMD Radeon RX 6700", 64, ALL)), self.RX6700, mode="on")[0])


class KeepEmulatedFlagTest(unittest.TestCase):
    """Only the default auto may keep emulated shaders; an explicit on or off is taken as given."""

    def settings(self, mode, gpu=None, max_sizes=(64,), *, mesh_compute=False):
        def fake_run(args, **kwargs):
            if os.path.basename(str(args[-1])) == "vk_subgroup_info.py":
                stdout = json.dumps(gpu or {})
            else:
                stdout = vulkaninfo_text(*max_sizes)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            os.makedirs(game)
            argv = ["make_game.py", "--game", game, "--out", os.path.join(tmp, "o"), "--work", os.path.join(tmp, "w"),
                    "--stages", "shaders", "--no-tui", "--wave64-emulate", mode]
            if mesh_compute:
                argv.append("--mesh-compute")
            with mock.patch.object(sys, "argv", argv), mock.patch.object(make_game, "below_normal"), \
                    mock.patch.object(make_game, "log", lambda *a: None), \
                    mock.patch.object(make_game, "check", return_value={"expected_exe": "g.exe"}), \
                    mock.patch.object(make_game, "preflight") as preflight, \
                    mock.patch.object(make_game.shutil, "which", return_value="vulkaninfo"), \
                    mock.patch.object(make_game.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(make_game, "runtime_gpu_per_vertex", return_value=True), \
                    mock.patch.object(make_game, "runtime_gpu_max_push_constants", return_value=256), \
                    mock.patch.object(make_game, "runtime_gpu_mesh_draws_crash", return_value=False), \
                    mock.patch.object(make_game, "shaders") as sh:
                make_game.main()
            keys = ("emulate", "keep_emulated", "fallbacks")
            self.assertEqual(tuple(preflight.call_args.kwargs[key] for key in keys),
                             tuple(sh.call_args.kwargs[key] for key in keys))
            self.assertEqual(sh.call_args.kwargs["fallbacks"], ("--mesh-compute",) if mesh_compute else ())
            return sh.call_args.kwargs["emulate"], sh.call_args.kwargs["keep_emulated"]

    def test_only_auto_keeps(self):
        self.assertEqual(self.settings("auto"), (False, True))
        self.assertEqual(self.settings("off"), (False, False))
        self.assertEqual(self.settings("on"), (True, False))

    def test_auto_does_not_keep_emulated_outputs_on_native_rdna2_on_amd_windows(self):
        gpu = {"device": "AMD Radeon RX 6700", "native64": True, "rdna2_amd_windows": True}
        for max_sizes in ((64,), (64, 32)):
            with self.subTest(max_sizes=max_sizes):
                self.assertEqual(self.settings("auto", gpu, max_sizes), (False, False))
        self.assertEqual(self.settings("auto", dict(gpu, rdna2_amd_windows=False)), (False, True))

    def test_rdna1_amd_windows_uses_native_compute_and_rebuilds_emulated_outputs(self):
        gpu = {"device": "AMD Radeon RX 5700", "rdna1_amd_windows": True, "native64": False,
               "native64_compute": True, "mesh_in_supported_stages": False}
        for mesh_compute in (False, True):
            for max_sizes in ((), (64,), (64, 32)):
                with self.subTest(mesh_compute=mesh_compute, max_sizes=max_sizes):
                    self.assertEqual(self.settings("auto", gpu, max_sizes, mesh_compute=mesh_compute),
                                     (False, False))
        self.assertEqual(self.settings("auto", dict(gpu, native64_compute=False), mesh_compute=True), (True, True))
        self.assertEqual(self.settings("on", gpu, mesh_compute=True), (True, False))

    def test_mesh_compute_flag_reaches_auto_decision(self):
        gpu = {"native64": False, "native64_compute": True, "mesh_in_supported_stages": False}
        self.assertEqual(self.settings("auto", gpu, mesh_compute=True), (False, True))
        self.assertEqual(self.settings("auto", gpu), (True, True))


class SubgroupInfoToolTest(unittest.TestCase):
    def test_enumeration_failures_do_not_select_from_a_partial_list(self):
        import ctypes
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import vk_subgroup_info as v

        for name, responses in (("count error", [(-3, 0)]),
                                ("partial list", [(0, 1), (5, 1), (0, 2), (-3, 0)])):
            with self.subTest(name=name):
                vk = mock.Mock()
                vk.vkCreateInstance.return_value = 0
                results = iter(responses)

                def enumerate_devices(_instance, count, _devices):
                    status, written = next(results)
                    ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = written
                    return status

                vk.vkEnumeratePhysicalDevices.side_effect = enumerate_devices
                with mock.patch.object(v, "load_vulkan", return_value=vk):
                    with self.assertRaisesRegex(RuntimeError, "vkEnumeratePhysicalDevices.*-3"):
                        v.query()
                self.assertEqual(vk.vkEnumeratePhysicalDevices.call_count, len(responses))
                vk.vkGetPhysicalDeviceProperties2.assert_not_called()
                vk.vkDestroyInstance.assert_called_once()

    def test_sizes_and_offsets_match_the_vulkan_headers_on_x64(self):
        # Literal values from vulkan_core.h (sizeof/offsetof on x86-64), not derived from the tool's own constants
        import ctypes
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import vk_subgroup_info as v
        self.assertEqual(v.PROPS_SIZE, 824)
        self.assertEqual((v.PROPS_API_VERSION, v.PROPS_DEVICE_TYPE, v.PROPS_DEVICE_NAME), (0, 16, 20))
        self.assertEqual((v.FEATURES2_SIZE, v.F12_SIZE, v.F13_SIZE), (240, 208, 80))
        self.assertEqual((v.F12_SEPARATE_DEPTH_STENCIL_LAYOUTS, v.F12_TIMELINE_SEMAPHORE), (156, 164))
        self.assertEqual((v.F13_SUBGROUP_SIZE_CONTROL, v.F13_SYNCHRONIZATION2, v.F13_DYNAMIC_RENDERING), (44, 52, 64))
        self.assertEqual(v.QUEUE_FAMILY_SIZE, 24)
        self.assertEqual((ctypes.sizeof(v.SubgroupSizeControl), ctypes.sizeof(v.AppInfo),
                          ctypes.sizeof(v.InstanceInfo)), (32, 48, 64))
        self.assertEqual((v.VK_SHADER_STAGE_COMPUTE_BIT, v.VK_SHADER_STAGE_MESH_BIT_EXT), (0x20, 0x80))

    def test_a_32_bit_python_gets_an_error_not_a_crash(self):
        import json
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import vk_subgroup_info as v
        with mock.patch.object(v.sys, "maxsize", 2**31 - 1), mock.patch.object(v, "load_vulkan") as load, \
                mock.patch("builtins.print") as pr:
            self.assertEqual(v.main(), 1)
        load.assert_not_called()
        self.assertIn("64-bit", json.loads(pr.call_args[0][0])["error"])

    def test_script_runs_isolated_and_prints_one_json_line(self):
        import json
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import vk_subgroup_info as v
        try:
            v.load_vulkan()
        except RuntimeError:
            self.skipTest("no Vulkan loader on this machine")
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "vk_subgroup_info.py")],
                           capture_output=True, text=True, timeout=60)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(r.returncode, 1 if "error" in info else 0)


class TranslateFlagTest(unittest.TestCase):
    """translate() passes --wave64-emulate to every shard, and only when asked."""

    def shard_args(self, emulate):
        started = []

        class FakeProc:
            def __init__(self, args):
                started.append(args)
                self.stdout = None

            def poll(self):
                return 0

            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as spirv, \
                mock.patch.object(make_game.subprocess, "Popen", side_effect=FakeProc), \
                mock.patch.object(make_game, "py") as py, \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.translate("disasm", spirv, 3, emulate=emulate)
        self.assertEqual(py.call_count, 1)                    # the --merge step
        self.assertNotIn("--wave64-emulate", py.call_args.args)
        return started

    def test_emulated_shards_get_the_flag(self):
        shards = self.shard_args(True)
        self.assertEqual(len(shards), 3)
        for args in shards:
            self.assertEqual(args.count("--wave64-emulate"), 1, args)

    def test_native_shards_do_not(self):
        for args in self.shard_args(False):
            self.assertNotIn("--wave64-emulate", args)


class ArgumentTest(unittest.TestCase):
    def test_bad_mode_is_rejected(self):
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "make_game.py"), "--wave64-emulate", "maybe"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("invalid choice", p.stderr)

    def test_help_lists_the_modes(self):
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "make_game.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--wave64-emulate {auto,on,off}", p.stdout)


if __name__ == "__main__":
    unittest.main()
