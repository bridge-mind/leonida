"""scripts/build_runtime.ps1's cl.exe fallback builds every GPU test harness the Python tools need (no build run).

The fallback is what runs when Visual Studio has no "C++ CMake tools" component. Without these programs,
check.py / run_gpu_tests.py stop with "missing vk_compute_test", and so do the fuzzers.

    python -I tests/test_build_runtime_fallback.py
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runtime" / "test" / "shaders"), str(ROOT / "tools")]
import check  # noqa: E402
import image_fuzz  # noqa: E402
import run_gpu_tests  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_runtime.ps1"


def fallback_body(text: str) -> str:
    """The cl.exe branch: from the line-starting `} else {` after the CMake test to the end of its here-string."""
    start = re.search(r"^\} else \{", text, re.M).start()
    return text[start:text.index('"@', text.index('$body = @"', start))]


class FallbackBuildsTheHarnesses(unittest.TestCase):
    def setUp(self):
        self.body = fallback_body(SCRIPT.read_text(encoding="utf-8"))
        self.programs = set(re.findall(r"/Fe:([A-Za-z0-9_]+)\.exe", self.body))

    def test_the_harnesses_the_gpu_tools_require_are_built(self):
        needed = set(run_gpu_tests.REQUIRED_TEST_PROGRAMS) | set(check.TEST_PROGRAMS) | {image_fuzz.HARNESS}
        self.assertEqual(needed - self.programs, set(), "missing from build_runtime.ps1's cl.exe fallback")

    def test_every_harness_the_gpu_suite_starts_is_built(self):
        """A harness run_gpu_tests.py or check.py starts by name, but the fallback did not build, fails its test
        with "could not start" (ps_ssharp: vk_sampler_test.exe, WinError 2)."""
        named = set()
        for path in (ROOT / "scripts" / "run_gpu_tests.py", ROOT / "runtime" / "test" / "shaders" / "check.py"):
            named |= set(re.findall(r'"(vk_[a-z0-9_]+)"', path.read_text(encoding="utf-8")))
        harnesses = {n for n in named if (ROOT / "runtime" / "test" / f"{n}.cpp").is_file()}
        self.assertIn("vk_sampler_test", harnesses)
        self.assertEqual(harnesses - self.programs, set(), "missing from build_runtime.ps1's cl.exe fallback")

    def test_the_launcher_compiles_every_source_cmake_gives_gta6rt(self):
        """BUILD-680: main.cpp calls rt::fs::configure, but the fallback left out shims/file_system.cpp (LNK2019).
        Every source of CMake's add_executable(gta6rt ...) must be in the fallback's launcher list ($files)."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        block = re.search(r"add_executable\(gta6rt\s+(.*?)\)", cmake, re.S).group(1)
        wanted = {item.replace("/", "\\") for item in block.split() if item.endswith(".cpp")}
        files = re.search(r"\$files = (.*?) \| ForEach-Object", self.body).group(1)
        listed = set(re.findall(r'"([^"]+\.cpp)"', files))
        self.assertTrue(wanted)
        self.assertEqual(wanted - listed, set(), "in CMake's gta6rt sources but not in the cl.exe fallback")

    def test_every_test_source_it_compiles_exists(self):
        sources = re.findall(r'"\$src\\test\\([A-Za-z0-9_]+\.cpp)"', self.body)
        self.assertTrue(sources)
        for name in sources:
            with self.subTest(source=name):
                self.assertTrue((ROOT / "runtime" / "test" / name).is_file())

    def test_the_pipeline_checker_gets_the_stack_reserve_cmake_gives_it(self):
        """BUILD-1457: a driver compiles on vk_pipeline_check's threads, and AMD's Windows compiler overflows the
        1 MB PE default. CMake links the checker with /STACK:<LEONIDA_PIPELINE_CHECK_STACK>; the fallback has to pass
        the same number on the checker's own cl line (a /link option applies to that one program)."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        reserve = re.search(r"set\(LEONIDA_PIPELINE_CHECK_STACK (\d+)\)", cmake).group(1)
        self.assertGreaterEqual(int(reserve), 8 * 1024 * 1024)
        self.assertIn("target_link_options(vk_pipeline_check PRIVATE /STACK:${LEONIDA_PIPELINE_CHECK_STACK})", cmake)
        lines = [line for line in self.body.splitlines() if "/Fe:vk_pipeline_check.exe" in line]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], rf"/Fe:vk_pipeline_check\.exe /link (\S+ )*/STACK:{reserve}( |$)")

    def test_the_loader_gets_the_stack_reserve_cmake_gives_it(self):
        """BUILD-1457, the loader's side: a game thread with the default stack size gets the reserve in gta6rt.exe's
        header, and the driver compiles a pipeline's shaders on the thread that creates it. CMake links gta6rt with
        /STACK:<GTA6RT_STACK>; the fallback has to pass the same number, and it is the number run_entry gives the
        entry thread, so every game thread that asks for no size has the same room."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        reserve = re.search(r"set\(GTA6RT_STACK (\d+)\)", cmake).group(1)
        self.assertGreaterEqual(int(reserve), 8 * 1024 * 1024)
        self.assertRegex(cmake, r"target_link_options\(gta6rt PRIVATE [^)]*/STACK:\$\{GTA6RT_STACK\}\)")
        lines = [line for line in self.body.splitlines() if "/Fe:gta6rt.exe" in line]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], rf"/Fe:gta6rt\.exe /link (\S+ )*/STACK:{reserve}( |$)")
        shim = (ROOT / "runtime" / "shims" / "process_shim.h").read_text(encoding="utf-8")
        megabytes = re.search(r"run_entry\(void\* entry, std::size_t stack_bytes = (\d+)u << 20\);", shim).group(1)
        self.assertEqual(int(megabytes) << 20, int(reserve))

    def test_build_cmd_sets_utf8_codepage_for_ninja_showincludes(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("chcp 65001 >nul", text)
        self.assertLess(text.index("chcp 65001 >nul"), text.index('call `"$vcvars`"'))

    @unittest.skipUnless(sys.platform == "win32", "requires Windows PowerShell")
    def test_path_fallback_prefers_vs_and_skips_msys_cmake(self):
        text = SCRIPT.read_text(encoding="utf-8")
        # Extract the helper functions and the two Find-BuildTool assignments directly from scripts/build_runtime.ps1.
        helpers_start = text.index("function Invoke-ToolProbe")
        helpers_end = text.index("$root = Split-Path")
        lookup_start = text.index("$cmakeDir = Join-Path")
        lookup_end = text.index("$cfg = if")
        snippet = text[helpers_start:helpers_end] + "\n" + text[lookup_start:lookup_end]

        def run_probe(vs_dir: Path, path_dirs: list[Path]) -> tuple[str, str]:
            ps = (
                "$ErrorActionPreference = 'Stop'\n"
                "$vs = $env:LEONIDA_VS_DIR\n"
                + snippet
                + "\nWrite-Output \"CMAKE=$cmake\"\nWrite-Output \"NINJA=$ninja\"\n"
            )
            env = dict(os.environ, PATH=";".join(str(p) for p in path_dirs), LEONIDA_VS_DIR=str(vs_dir))
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                env=env, capture_output=True, text=True, check=True, timeout=30,
            )
            found = {}
            for line in res.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    found[k.strip()] = v.strip()
            return found.get("CMAKE", ""), found.get("NINJA", "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vs_empty = root / "vs_empty"
            msys_dir = root / "msys_bin"
            native_cmake_dir = root / "native_cmake_bin"
            native_ninja_dir = root / "native_ninja_bin"
            for d in (vs_empty, msys_dir, native_cmake_dir, native_ninja_dir):
                d.mkdir()

            # MSYS/devkitPro-like cmake: lacks Visual Studio generators in --help.
            msys_cmake = msys_dir / "cmake.cmd"
            msys_cmake.write_text("@echo off\r\necho * Ninja\r\necho   Unix Makefiles\r\nexit /b 0\r\n", encoding="ascii")

            # Native Windows cmake: lists both Visual Studio and Ninja generators.
            native_cmake = native_cmake_dir / "cmake.cmd"
            native_cmake.write_text(
                "@echo off\r\necho   Visual Studio 18 2026\r\necho * Ninja\r\nexit /b 0\r\n",
                encoding="ascii",
            )
            native_ninja = native_ninja_dir / "ninja.cmd"
            native_ninja.write_text("@echo off\r\necho 1.12.1\r\nexit /b 0\r\n", encoding="ascii")

            # 1) No VS CMake tools, MSYS cmake comes first on PATH -> must skip MSYS and pick native PATH cmake + ninja.
            cmake, ninja = run_probe(vs_empty, [msys_dir, native_cmake_dir, native_ninja_dir])
            self.assertEqual(Path(cmake).resolve(), native_cmake.resolve())
            self.assertEqual(Path(ninja).resolve(), native_ninja.resolve())

            # 2) Only MSYS cmake on PATH -> must reject it so build_runtime.ps1 falls back to cl.exe.
            cmake, ninja = run_probe(vs_empty, [msys_dir])
            self.assertEqual(cmake, "")
            self.assertEqual(ninja, "")

            # 3) Native cmake on PATH without ninja -> ninja stays empty so CMake branch does not run.
            cmake, ninja = run_probe(vs_empty, [native_cmake_dir])
            self.assertEqual(Path(cmake).resolve(), native_cmake.resolve())
            self.assertEqual(ninja, "")

            # 4) VS bundled CMake + Ninja present -> must prefer VS over PATH.
            vs_full = root / "vs_full"
            vs_cmake = vs_full / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
            vs_ninja = vs_full / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
            vs_cmake.parent.mkdir(parents=True)
            vs_ninja.parent.mkdir(parents=True)
            vs_cmake.write_bytes(b"")
            vs_ninja.write_bytes(b"")
            cmake, ninja = run_probe(vs_full, [native_cmake_dir, native_ninja_dir])
            self.assertEqual(Path(cmake).resolve(), vs_cmake.resolve())
            self.assertEqual(Path(ninja).resolve(), vs_ninja.resolve())

if __name__ == "__main__":
    unittest.main()
