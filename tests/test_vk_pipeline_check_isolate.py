"""vk_pipeline_check --isolate: a module that kills the driver is recorded as failed and the pass goes on.

No game data: five tiny compute shaders are assembled here. The crash is simulated with the tool's test
hook VK_PIPELINE_CHECK_SIMULATE_CRASH just before the pipeline of the named module, so no crashing
driver is needed. Skips when vk_pipeline_check isn't built or there is no usable Vulkan device.
Pin a GPU with VK_LOADER_DRIVERS_SELECT / LEONIDA_VK_DEVICE as for the other
Vulkan tools. Set LEONIDA_VK_PIPELINE_CHECK to use a binary elsewhere.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from spirv_asm import assemble  # noqa: E402

NAMES = ["m_alpha", "m_bravo", "m_charlie", "m_delta", "m_echo"]
CRASHER = "m_charlie"
WINDOWS = os.name == "nt"
FPE_EXIT = 0xC000008E if WINDOWS else -8
FPE_LABEL = "driver exited with status 0xc000008e" if WINDOWS else "driver crashed: SIGFPE"
SEGV_LABEL = "driver exited with status 0xc0000005" if WINDOWS else "driver crashed: SIGSEGV"

ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {size} 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


def find_binary():
    env = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if env:
        return Path(env) if Path(env).is_file() else None
    build = ROOT / "runtime" / "build"
    paths = ((build / "vk_pipeline_check.exe", build / "RelWithDebInfo" / "vk_pipeline_check.exe") if WINDOWS else
             (build / "linux" / "vk_pipeline_check", build / "vk_pipeline_check"))
    for p in paths:
        if p.is_file():
            return p
    return None


class IsolateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (set LEONIDA_VK_PIPELINE_CHECK to its path)")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.dir = Path(cls.tmp.name) / "spirv with spaces"
        cls.dir.mkdir()
        index = {}
        for i, name in enumerate(NAMES):
            (cls.dir / (name + ".spv")).write_bytes(assemble(ASM.format(size=1 << i), target_env="vulkan1.1"))
            index[name] = {"stage": "cs", "subgroup_size": None, "features": []}
        (cls.dir / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
        probe = cls.run_tool([], cache=Path(cls.tmp.name) / "probe.cache")
        if probe.returncode != 0 or "cs pipelines: 5 ok, 0 failed" not in probe.stdout:
            raise unittest.SkipTest("no usable Vulkan device (exit %d): %s" % (probe.returncode, probe.stderr.strip()[-300:]))

    @classmethod
    def run_tool(cls, args, crash=None, cache=None, *, device=None):
        env = dict(os.environ)
        env.pop("VK_PIPELINE_CHECK_SIMULATE_CRASH", None)
        if crash:
            env["VK_PIPELINE_CHECK_SIMULATE_CRASH"] = crash
        if device:
            env["LEONIDA_VK_DEVICE"] = device
        # Windows paths ending in a backslash must survive the parent's command-line quoting too.
        directory = str(cls.dir) + (os.sep if WINDOWS else "")
        cmd = [str(cls.exe), directory] + ([str(cache)] if cache else []) + ["--jobs", "2"] + args
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

    def cache(self, name):
        return Path(self.tmp.name) / name

    def test_normal_run_passes(self):
        r = self.run_tool([], cache=self.cache("normal.cache"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs: 5 modules ok, 0 failed", r.stdout)
        self.assertIn("cs pipelines: 5 ok, 0 failed, 0 skipped", r.stdout)

    def test_isolate_without_a_crash_matches_normal_counts(self):
        r = self.run_tool(["--isolate", "--isolate-batch", "2"], cache=self.cache("isolated clean.cache"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs: 5 modules ok, 0 failed", r.stdout)
        self.assertIn("cs pipelines: 5 ok, 0 failed, 0 skipped", r.stdout)

    def test_oversized_batch_keeps_the_requested_slice(self):
        r = self.run_tool(["--isolate", "--start", "2", "--count", "2",
                           "--isolate-batch", str(2 * sys.maxsize + 1)], cache=self.cache("iso_slice.cache"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs: 2 modules ok, 0 failed", r.stdout)
        self.assertIn("cs pipelines: 2 ok, 0 failed, 0 skipped", r.stdout)
        self.assertNotIn("CRASH", r.stdout)
        self.assertNotIn("ISO-", r.stdout)

    def test_isolate_checks_only_the_selected_indices(self):
        r = self.run_tool(["--isolate", "--isolate-batch", "1", "--start", "3", "--count", "2"], crash=CRASHER)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs: 2 modules ok, 0 failed", r.stdout)
        self.assertIn("cs pipelines: 2 ok, 0 failed, 0 skipped", r.stdout)

    def test_failed_child_setup_is_not_attributed_to_a_shader(self):
        r = self.run_tool(["--isolate"], device="leonida-test-device-that-does-not-exist")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("no Vulkan physical device matches", r.stderr)
        self.assertIn("child process failed during setup", r.stderr)
        self.assertNotIn("MODULE CRASH", r.stdout)

    def test_without_isolate_the_driver_crash_ends_the_process(self):
        r = self.run_tool([], crash=CRASHER, cache=self.cache("dies.cache"))
        self.assertEqual(r.returncode, FPE_EXIT, r.stdout + r.stderr)

    def test_isolate_records_the_crashed_module_and_goes_on(self):
        cache = self.cache("iso.cache")
        for batch in ("5", "2", "1"):   # one batch bisected down, several batches, one module at a time
            with self.subTest(batch=batch):
                r = self.run_tool(["--isolate", "--isolate-batch", batch], crash=CRASHER + ":SIGFPE", cache=cache)
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertGreater(r.returncode, 0, "the parent itself must not die: " + r.stdout + r.stderr)
                self.assertIn(f"MODULE CRASH {CRASHER} (cs, {FPE_LABEL})", r.stdout)
                self.assertEqual(r.stdout.count("MODULE CRASH"), 1, r.stdout)
                self.assertIn("cs: 4 modules ok, 0 failed", r.stdout)
                self.assertIn("cs pipelines: 4 ok, 1 failed, 0 skipped", r.stdout)
                self.assertIn("cs crashed the driver: 1", r.stdout)
                for name in NAMES:
                    if name != CRASHER:
                        self.assertNotIn("FAIL " + name, r.stdout)
                self.assertTrue(cache.is_file(), "the shared pipeline cache must survive the crash")

    def test_other_signal(self):
        r = self.run_tool(["--isolate"], crash=CRASHER + ":SEGV")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(f"MODULE CRASH {CRASHER} (cs, {SEGV_LABEL})", r.stdout)

    def test_isolate_rejects_stats(self):
        r = self.run_tool(["--isolate", "--stats", str(self.cache("s.csv"))])
        self.assertEqual(r.returncode, 2)

    def test_isolate_updates_the_inflight_snapshot(self):
        snapshot = self.cache("inflight.txt")
        snapshot.write_text("stale\tmodule\n", encoding="utf-8")
        args = ["--isolate", "--start", str(NAMES.index(CRASHER)), "--count", "1",
                "--large-module-bytes", "1", "--large-parallel", "1", "--inflight-file", str(snapshot)]

        failed = self.run_tool(args, crash=CRASHER)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        self.assertIn(f"MODULE CRASH {CRASHER}", failed.stdout)
        self.assertEqual(snapshot.read_text(encoding="utf-8"), f"{CRASHER}\tpipeline\n")

        passed = self.run_tool(args)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertEqual(snapshot.read_bytes(), b"")


def main():
    """unittest.main(), except that a file whose tests all skipped succeeds: unittest exits 5 for an empty run, and
    CI runs each file as `python -I tests/test_*.py || exit 1`, so a runner without vk_pipeline_check or on Windows
    would fail the job here. A file that has no tests at all (not even skips) is still a failure."""
    result = unittest.main(exit=False).result
    if result.failures or result.errors or result.unexpectedSuccesses:
        return 1
    if result.testsRun == 0 and not result.skipped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
