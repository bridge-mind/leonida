"""Shader stage of make_game.py: resuming a stopped build, Windows-safe carve names, an empty unpack (#126).

Standard library only; no game data: the tools are mocked or fed hand-made inputs.
"""
import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import make_game  # noqa: E402
import shader_carve  # noqa: E402

UNPACK = os.path.join(ROOT, "tools", "fssm_unpack.py")


class CarveFileNameTest(unittest.TestCase):
    def names(self, *shader_names):
        seen, used = {}, set()
        return [shader_carve.file_name(n, seen, used) for n in shader_names]

    def test_repeated_name_gets_a_suffix(self):
        self.assertEqual(self.names("ps_a", "ps_a", "ps_a"), ["ps_a.dxbc", "ps_a~1.dxbc", "ps_a~2.dxbc"])

    def test_names_differing_only_in_case_do_not_share_a_file(self):
        out = self.names("PS_Water", "ps_water")
        self.assertEqual(len({n.lower() for n in out}), 2, out)

    def test_names_that_sanitize_to_the_same_file_do_not_share_it(self):
        out = self.names("a:b", "a_b", "a?b")
        self.assertEqual(len({n.lower() for n in out}), 3, out)

    def test_suffix_never_lands_on_a_real_name(self):
        out = self.names("x~1", "x", "x")
        self.assertEqual(len({n.lower() for n in out}), 3, out)

    def test_windows_device_names_are_prefixed(self):
        for dev in ("CON", "nul", "Aux", "PRN", "COM1", "lpt9", "con.vs"):
            name = self.names(dev)[0]
            self.assertNotIn(name.split(".")[0].upper(), shader_carve.RESERVED, name)
        self.assertEqual(self.names("CONSOLE")[0], "CONSOLE.dxbc")


class EmptyUnpackTest(unittest.TestCase):
    def test_nothing_unpacked_fails_clearly_and_writes_no_index(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "junk.dxbc"), "wb") as f:
                f.write(b"DXBC" + bytes(60))     # no parts: nothing to unpack
            r = subprocess.run([sys.executable, "-I", UNPACK, d], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "fssm_index.json")))

    def test_empty_folder_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-I", UNPACK, d], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stderr)


class ShaderStageResumeTest(unittest.TestCase):
    """Which steps shaders() runs for a given work folder. The tools are mocked; each mock writes what the real
    tool writes last, so the next run sees the same files a real one would."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = self.tmp.name
        self.dxbc = os.path.join(self.work, "shaders", "dxbc")
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        self.ran = []
        self.real_llvm_mc_signature = make_game.llvm_mc_signature
        version_probe = mock.patch.object(make_game, "llvm_mc_signature", return_value="test llvm-mc 21")
        self.llvm_mc_signature = version_probe.start()
        self.addCleanup(version_probe.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(*parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if p == os.path.join(self.dxbc, "index.json"):
                json.dump([{"file": "s0.dxbc"}], f)
            elif p == os.path.join(self.dxbc, "fssm_index.json"):
                json.dump([{"name": "s0"}], f)
            elif p.endswith(".s"):
                f.write("synthetic listing\n")

    def fake_py(self, tool, *args):
        self.ran.append(tool)
        if tool == "shader_carve.py":
            self.touch(self.dxbc, "index.json")
        elif tool == "fssm_unpack.py":
            self.assertEqual(args[-2:], ("--carve-index", os.path.join(self.dxbc, "index.json")))
            self.touch(self.dxbc, "fssm_index.json")
        elif tool == "shader_disasm.py":
            self.assertEqual(args[1:3], ("--unpack-index", os.path.join(self.dxbc, "fssm_index.json")))
            self.touch(self.dxbc, "disasm", "s0.s")

    def fake_translate(self, disasm, spirv, jobs, emulate=False, fallbacks=()):
        self.ran.append("translate (emulated)" if emulate else "translate")
        self.touch(spirv, "index.json")

    def run_stage(self, emulate=None, jobs=2):
        self.ran = []
        kw = {} if emulate is None else {"emulate": emulate}
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", self.fake_translate), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.shaders(os.path.join(self.work, "game"), self.work, jobs, **kw)
        return self.ran

    def test_fresh_folder_runs_every_step_then_nothing(self):
        self.assertEqual(self.run_stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate"])
        self.assertEqual(self.run_stage(), [])

    def test_llvm_version_change_at_the_same_path_refreshes_only_the_dependent_steps(self):
        wrapper = os.path.join(self.work, "fake llvm-mc.py")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write("import os, sys\n"
                    "assert sys.argv[1:] == ['--version']\n"
                    "print('LLVM version ' + os.environ['TEST_LLVM_VERSION'])\n")
        command = f'"{sys.executable}" "{wrapper}"'
        with mock.patch.object(make_game, "llvm_mc_signature", self.real_llvm_mc_signature):
            with mock.patch.dict(os.environ, {"LLVM_MC": command, "TEST_LLVM_VERSION": "23.1.5"}):
                self.run_stage()
                self.assertEqual(self.run_stage(), [])
                self.touch(self.dxbc, "disasm", "old.s")
                self.touch(self.dxbc, "disasm", "notes.txt")
                self.touch(self.spirv, make_game.SHARD_PLAN)
            with mock.patch.dict(os.environ, {"LLVM_MC": command, "TEST_LLVM_VERSION": "21.1.0"}):
                self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
                self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm", "old.s")))
                self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm", "notes.txt")))
                self.assertFalse(os.path.exists(os.path.join(self.spirv, make_game.SHARD_PLAN)))
                self.assertEqual(self.run_stage(), [])

    def test_llvm_change_leaves_no_completion_markers_if_disassembly_stops(self):
        self.run_stage()
        self.touch(self.spirv, make_game.SHARD_PLAN)
        self.llvm_mc_signature.return_value = "test llvm-mc 23"
        with mock.patch.object(make_game, "py", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                make_game.shaders(os.path.join(self.work, "game"), self.work, 2)
        self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm.done")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, make_game.SHARD_PLAN)))
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertEqual(self.run_stage(), [])

    def test_version_probe_fails_with_a_clear_error_for_a_failed_command(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="distribution not found")
        with mock.patch.object(make_game.subprocess, "run", return_value=failed), \
                mock.patch.object(make_game, "mc_command", return_value=["wsl", "-e", "llvm-mc"]), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as exc:
                self.real_llvm_mc_signature()
        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Check LLVM_MC: distribution not found", stderr.getvalue())

    def test_unpack_stopped_after_creating_fssm_folder_runs_again(self):
        self.touch(self.dxbc, "index.json")
        os.makedirs(os.path.join(self.dxbc, "fssm"))       # what a stopped fssm_unpack.py leaves
        self.assertEqual(self.run_stage(), ["fssm_unpack.py", "shader_disasm.py", "translate"])

    def test_disasm_stopped_after_many_files_runs_again(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        for i in range(1200):                               # past the old 1000-file threshold
            self.touch(self.dxbc, "disasm", f"s{i}.s")
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])

    def test_disassembly_resume_keeps_completed_listings_for_the_same_llvm(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        listing = os.path.join(self.dxbc, "disasm", "s0.s")

        def stop(tool, *args):
            self.assertEqual(tool, "shader_disasm.py")
            self.assertNotIn("--skip-existing", args)
            self.touch(listing)
            raise KeyboardInterrupt

        for change_llvm in (False, True):
            with self.subTest(change_llvm=change_llvm):
                if change_llvm:
                    os.remove(os.path.join(self.dxbc, "disasm.done"))
                with mock.patch.object(self, "fake_py", side_effect=stop):
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_stage()
                if change_llvm:
                    self.llvm_mc_signature.return_value = "different llvm-mc"
                disassemble = self.fake_py

                def check_reuse(tool, *args):
                    self.assertEqual("--skip-existing" in args, not change_llvm)
                    self.assertEqual(os.path.isfile(listing), not change_llvm)
                    disassemble(tool, *args)

                with mock.patch.object(self, "fake_py", side_effect=check_reuse):
                    self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
                self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm.done")))
                self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm.started")))

    def test_finished_disassembly_repairs_missing_recorded_listing_before_translation(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        with open(os.path.join(self.dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"file": name + ".dxbc"} for name in ("s0", "s1", "unsupported")], f)
        with open(os.path.join(self.dxbc, "fssm_index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"name": name} for name in ("s0", "s1", "unsupported")], f)

        original = self.fake_py

        def disassemble(tool, *args):
            original(tool, *args)
            self.touch(self.dxbc, "disasm", "s1.s")

        with mock.patch.object(self, "fake_py", side_effect=disassemble):
            self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertEqual(self.run_stage(), [])
        os.remove(os.path.join(self.dxbc, "disasm", "s1.s"))
        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))

        with mock.patch.object(make_game, "fail", side_effect=SystemExit(1)) as fail:
            with self.assertRaises(SystemExit):
                self.run_stage()
        fail.assert_called_once_with("Shader disassembly did not restore all expected listings")
        self.assertFalse(os.path.isfile(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.isfile(os.path.join(self.dxbc, "disasm.done")))

        def repair(tool, *args):
            self.assertIn("--skip-existing", args)
            self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm", "s0.s")))
            disassemble(tool, *args)

        with mock.patch.object(self, "fake_py", side_effect=repair):
            self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertTrue(make_game.shader_outputs_reusable(self.work, False, False))

    def test_finished_disassembly_repairs_a_zero_byte_listing(self):
        self.assertEqual(self.run_stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate"])
        listing = os.path.join(self.dxbc, "disasm", "s0.s")
        self.assertGreater(os.path.getsize(listing), 0)
        open(listing, "wb").close()

        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertGreater(os.path.getsize(listing), 0)
        self.assertEqual(self.run_stage(), [])

    def test_deleting_every_listing_and_the_translation_disassembles_again(self):
        # README.md, "Low disk space": deleting the .s files with spirv/index.json must not leave the stage
        # looking finished with zero shaders.
        self.assertEqual(self.run_stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate"])
        disasm = os.path.join(self.dxbc, "disasm")
        for name in os.listdir(disasm):
            if name.endswith(".s"):
                os.remove(os.path.join(disasm, name))
        os.remove(os.path.join(self.spirv, "index.json"))

        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertTrue(os.path.isfile(os.path.join(disasm, "s0.s")))
        self.assertTrue(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), [])

    def test_translation_stopped_after_many_files_runs_again(self):
        self.run_stage()
        os.remove(os.path.join(self.spirv, "index.json"))   # stopped before --merge
        for i in range(1200):
            self.touch(self.spirv, f"s{i}.spv")
        self.assertEqual(self.run_stage(), ["translate"])

    def test_failed_disasm_leaves_no_marker(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")

        def failing_py(tool, *args):
            raise SystemExit(1)
        with mock.patch.object(make_game, "py", failing_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.shaders(os.path.join(self.work, "game"), self.work, 2)
        self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm.done")))

    def test_rerunning_a_step_reruns_the_steps_after_it(self):
        self.run_stage()
        os.remove(os.path.join(self.dxbc, "fssm_index.json"))
        self.assertEqual(self.run_stage(), ["fssm_unpack.py", "shader_disasm.py", "translate"])

    def test_old_unpack_index_with_a_noncurrent_shader_is_refreshed(self):
        self.run_stage()
        with open(os.path.join(self.dxbc, "fssm_index.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "s0"}, {"name": "removed"}], f)
        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), ["fssm_unpack.py", "shader_disasm.py", "translate"])
        self.assertEqual(self.run_stage(), [])

    def test_old_disassembly_with_a_noncurrent_shader_is_refreshed(self):
        self.run_stage()
        self.touch(self.dxbc, "disasm", "removed.s")
        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm", "removed.s")))
        self.assertEqual(self.run_stage(), [])

    def test_legacy_folder_without_llvm_version_refreshes_disassembly_once(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        self.touch(self.dxbc, "disasm", "s0.s")
        self.touch(self.spirv, "index.json")                # translated by a build from before disasm.done
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm.done")))
        self.assertEqual(self.run_stage(), [])

    def test_legacy_completion_marker_with_a_missing_listing_disassembles_everything_again(self):
        # A disasm.done from before the listings were recorded holds only the llvm-mc signature: with a listing
        # missing it can't tell which ones are current, so the whole disassembly runs again, not a resume.
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        self.touch(self.spirv, "index.json")
        with open(os.path.join(self.dxbc, "disasm.done"), "w", encoding="utf-8", newline="\n") as f:
            f.write(make_game.llvm_mc_signature() + "\n")
        os.makedirs(os.path.join(self.dxbc, "disasm"))
        self.assertFalse(make_game.shader_outputs_reusable(self.work, False, False))

        original = self.fake_py

        def disassemble(tool, *args):
            self.assertNotIn("--skip-existing", args)
            original(tool, *args)

        with mock.patch.object(self, "fake_py", side_effect=disassemble):
            self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertTrue(make_game.shader_outputs_reusable(self.work, False, False))
        self.assertEqual(self.run_stage(), [])

    def test_old_disassembly_layout_is_rebuilt_in_the_current_location(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        self.touch(self.work, "shaders", "disasm", "old.s")  # disasm next to the carve folder, the old layout
        self.touch(self.spirv, "index.json")
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])
        self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm", "s0.s")))
        self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm", "old.s")))

    # SHADER-11: the translation depends on --wave64-emulate, so a change of mode translates again.
    def mode(self):
        with open(os.path.join(self.spirv, "wave64_emulate.txt"), encoding="utf-8") as f:
            return f.read().strip()

    def test_the_mode_is_recorded_beside_the_translation(self):
        self.run_stage(emulate=True)
        self.assertEqual(self.mode(), "on")
        self.run_stage(emulate=False)
        self.assertEqual(self.mode(), "off")

    def test_turning_emulation_on_translates_again_once(self):
        self.run_stage(emulate=False)
        self.assertEqual(self.run_stage(emulate=True), ["translate (emulated)"])
        self.assertEqual(self.run_stage(emulate=True), [])

    def test_turning_emulation_off_translates_again_once(self):
        self.run_stage(emulate=True)
        self.assertEqual(self.run_stage(emulate=False), ["translate"])
        self.assertEqual(self.run_stage(emulate=False), [])

    def test_a_folder_from_before_the_mode_file_counts_as_not_emulated(self):
        self.run_stage(emulate=False)
        os.remove(os.path.join(self.spirv, "wave64_emulate.txt"))
        self.assertEqual(self.run_stage(emulate=False), [])  # nothing translated, so still no mode file
        self.assertEqual(self.run_stage(emulate=True), ["translate (emulated)"])

    # A kill between the merge (index.json appears) and the end of shaders() must not leave a finished folder
    # whose mode is unknown or wrong.
    def kill_after_merge(self, emulate):
        def translate_then_die(disasm, spirv, jobs, emulate=False, fallbacks=()):
            self.touch(spirv, "index.json")                 # rdna2spirv.py --merge finished ...
            raise KeyboardInterrupt                         # ... and the PC died right after
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", translate_then_die), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(KeyboardInterrupt):
                make_game.shaders(os.path.join(self.work, "game"), self.work, 2, emulate=emulate)

    def test_a_kill_right_after_the_merge_keeps_the_finished_emulated_translation(self):
        self.kill_after_merge(emulate=True)
        self.assertEqual(self.run_stage(emulate=True), [])      # hours of translation are not redone
        self.assertEqual(self.mode(), "on")

    def test_a_kill_right_after_an_emulated_merge_is_not_mistaken_for_a_native_translation(self):
        self.kill_after_merge(emulate=True)
        self.assertEqual(self.run_stage(emulate=False), ["translate"])   # off was asked for: native, not "off"

    def test_a_kill_right_after_a_native_merge_counts_as_native(self):
        self.kill_after_merge(emulate=False)
        self.assertEqual(self.run_stage(emulate=False), [])
        self.assertEqual(self.run_stage(emulate=True), ["translate (emulated)"])

    # `auto` can answer differently from one run to the next (a GPU asleep or hidden, vulkaninfo missing): an
    # emulated translation is exact on every GPU, so auto keeps it instead of translating the corpus again.
    def run_auto(self, emulate):
        self.ran = []
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", self.fake_translate), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.shaders(os.path.join(self.work, "game"), self.work, 2, emulate=emulate, keep_emulated=True)
        return self.ran

    def test_auto_keeps_an_emulated_translation_when_this_run_would_not_emulate(self):
        self.run_stage(emulate=True)
        self.assertEqual(self.run_auto(emulate=False), [])
        self.assertEqual(self.mode(), "on")

    def test_auto_still_translates_again_when_a_native_translation_needs_emulation(self):
        self.run_stage(emulate=False)
        self.assertEqual(self.run_auto(emulate=True), ["translate (emulated)"])

    def test_an_explicit_off_still_translates_an_emulated_folder_again(self):
        self.run_stage(emulate=True)
        self.assertEqual(self.run_stage(emulate=False), ["translate"])

    def test_a_mode_change_does_not_redo_the_steps_before_translation(self):
        self.run_stage(emulate=False)
        self.assertEqual(self.run_stage(emulate=True), ["translate (emulated)"])
        self.assertTrue(os.path.isfile(os.path.join(self.dxbc, "disasm.done")))

    # The translation also depends on the translator's own files: one made from other files, or from unknown ones, is
    # translated again once, and nothing before it is redone.
    def sources_file(self):
        return os.path.join(self.spirv, "translation_sources.sha256")

    def record_sources(self, digest):
        with open(self.sources_file(), "w", encoding="utf-8") as f:
            f.write(digest + "\n")

    def copy_tools(self):
        tools = os.path.join(self.work, "tools")
        os.makedirs(tools)
        for name in make_game.TRANSLATION_SOURCES + ("shader_disasm.py",):
            shutil.copy(os.path.join(make_game.TOOLS, name), tools)
        patch = mock.patch.object(make_game, "TOOLS", tools)
        patch.start()
        self.addCleanup(patch.stop)
        return tools

    def test_an_unreadable_sources_file_translates_again(self):
        self.run_stage()
        with open(self.sources_file(), "wb") as f:
            f.write(b"\xff\xfe\x00")
        self.assertEqual(self.run_stage(), ["translate"])
        self.assertEqual(self.run_stage(), [])

    def test_editing_any_translation_source_translates_again_once(self):
        tools = self.copy_tools()
        self.run_stage()
        for name in make_game.TRANSLATION_SOURCES:
            with open(os.path.join(tools, name), "ab") as f:
                f.write(b"\n")
            self.assertEqual(self.run_stage(), ["translate"], name)
            self.assertEqual(self.run_stage(), [], name)

    def test_editing_another_tool_or_changing_jobs_keeps_the_translation(self):
        tools = self.copy_tools()
        self.run_stage()
        with open(os.path.join(tools, "shader_disasm.py"), "ab") as f:
            f.write(b"\n")
        self.assertEqual(self.run_stage(), [])
        self.assertEqual(self.run_stage(jobs=7), [])

    def test_an_invalid_mode_file_refreshes_the_translation(self):
        self.run_stage(emulate=True)
        with open(os.path.join(self.spirv, "wave64_emulate.txt"), "wb") as f:
            f.write(b"\xff")
        self.assertEqual(self.run_stage(emulate=False), ["translate"])
        self.assertEqual(self.mode(), "off")

    def test_auto_keeps_emulating_when_translating_again_for_new_sources(self):
        self.run_stage(emulate=True)
        self.record_sources("0" * 64)
        self.assertEqual(self.run_auto(emulate=False), ["translate (emulated)"])
        self.assertEqual(self.mode(), "on")
        self.assertEqual(self.run_auto(emulate=False), [])

    def test_auto_keeps_emulating_through_a_failed_rebuild_for_new_sources(self):
        self.run_stage(emulate=True)
        self.record_sources("0" * 64)

        def failing_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
            self.assertTrue(emulate)
            raise SystemExit(1)
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", failing_translate), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.shaders(os.path.join(self.work, "game"), self.work, 2, emulate=False, keep_emulated=True)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertEqual(self.run_auto(emulate=False), ["translate (emulated)"])
        self.assertEqual(self.mode(), "on")

    def test_an_explicit_off_translates_natively_for_new_sources(self):
        self.run_stage(emulate=True)
        self.record_sources("0" * 64)
        self.assertEqual(self.run_stage(emulate=False), ["translate"])
        self.assertEqual(self.mode(), "off")

    def test_a_kill_right_after_a_merge_for_new_sources_keeps_the_translation(self):
        self.run_stage()
        self.record_sources("0" * 64)
        self.kill_after_merge(emulate=False)
        self.assertEqual(self.run_stage(), [])                  # hours of translation are not redone


class TranslationSourcesTest(unittest.TestCase):
    def test_the_list_covers_every_local_module_the_translator_imports(self):
        seen, todo = set(), ["rdna2spirv"]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            with open(os.path.join(make_game.TOOLS, name + ".py"), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):                     # function-level imports too
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                else:
                    continue
                todo += [m.split(".")[0] for m in mods
                         if os.path.isfile(os.path.join(make_game.TOOLS, m.split(".")[0] + ".py"))]
        self.assertEqual({n + ".py" for n in seen},
                         {n for n in make_game.TRANSLATION_SOURCES if n.endswith(".py")})


class MergePublishesLastTest(unittest.TestCase):
    """rdna2spirv.py --merge writes the stubs before index.json, which tells make_game.py the translation finished."""

    def merge(self, out):
        tool = os.path.join(make_game.TOOLS, "rdna2spirv.py")
        return subprocess.run([sys.executable, "-I", tool, out, "--out", out, "--merge"],
                              capture_output=True, text=True, errors="replace")

    def test_a_failed_stub_write_leaves_no_index_and_a_retry_publishes_it(self):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "index.part0.json"), "w", encoding="utf-8") as f:
                json.dump({"s0": {"stage": "ps", "inputs": [0]}}, f)
            blocker = os.path.join(out, "stubs")
            open(blocker, "w").close()                      # a plain file where the stubs folder goes
            r = self.merge(out)
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertFalse(os.path.exists(os.path.join(out, "index.json")))
            os.remove(blocker)
            r = self.merge(out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out, "index.json")))
            self.assertTrue(os.path.isfile(os.path.join(out, "stubs", "vs_0.spv")))


if __name__ == "__main__":
    unittest.main()
