"""make_game.py's FSPX step (fspx_shaders) and the index key the runtime finds FSPX programs by (fspx_code_sha256).

    python -I tests/test_fspx_shaders_stage.py

Zero game data: the "DLL" is synthetic bytes around hand-built containers (tests/test_fspx_unpack.py builds them), and
the stage tests mock the tools the way tests/test_shader_stage_resume.py does.
"""
import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import make_game  # noqa: E402
import fspx_unpack  # noqa: E402
import shader_meta  # noqa: E402
from test_fspx_unpack import kind0, make_dxbc, make_fspx, program, psv0  # noqa: E402

UNPACK = os.path.join(ROOT, "tools", "fspx_unpack.py")


def container(code, state=None, **fspx):
    payload = make_fspx([(0, kind0() if state is None else state), (1, b"\1" * 20), (6, code), (7, b"\7" * 24)],
                        **fspx)
    return make_dxbc([(b"PSV0", psv0()), (b"HASH", b"\x5a" * 20), (b"ACCC", payload)])


def dll_like(*containers):
    """Containers between unrelated bytes, as a DLL holds them."""
    return b"MZ" + b"\0" * 126 + b"".join(c + b"\xcc" * 12 for c in containers)


def runtime_key(bytecode):
    """What runtime/src/shader_library.h's compute_program() hashes: the kind-6 bytes of a container's ACCC payload,
    or a bare program as it is."""
    if bytecode[:4] == b"DXBC":
        status, entries = fspx_unpack.parse_fspx(fspx_unpack.dxbc_parts(bytecode).get(b"ACCC", b""))
        if status:
            return None
        off, size = entries[6]
        bytecode = fspx_unpack.dxbc_parts(bytecode)[b"ACCC"][off:off + size]
    return hashlib.sha256(bytecode).hexdigest()


class IndexKey(unittest.TestCase):
    """fspx_code_sha256 is the SHA-256 of the kind-6 bytes fspx_unpack.py writes, the key the runtime computes."""

    def test_key_from_the_unpacked_program_matches_both_runtime_inputs(self):
        codes = [program(), program() + b"\0\0\x80\xbf"]       # two different kind-6 programs
        with tempfile.TemporaryDirectory() as t:
            dll = os.path.join(t, "upscaler.dll")
            boxes = [container(c) for c in codes]
            with open(dll, "wb") as f:
                f.write(dll_like(*boxes))
            out = os.path.join(t, "out")
            r = subprocess.run([sys.executable, "-I", UNPACK, dll, "--out", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            os.makedirs(os.path.join(out, "disasm"))
            for n, (code, box) in enumerate(zip(codes, boxes)):
                name = f"upscaler_{n:02d}"
                disasm = os.path.join(out, "disasm", name + ".s")
                key = shader_meta.fspx_code_sha256_file(disasm, name)
                self.assertEqual(key, hashlib.sha256(code).hexdigest())
                self.assertEqual(key, runtime_key(box))           # slot 62: the ACCC container
                self.assertEqual(key, runtime_key(code))          # slot 67: the bare kind-6 program
                meta = shader_meta.collect(disasm, name, [[]], [[]])
                self.assertEqual(meta.get("fspx_code_sha256"), key)
                self.assertNotIn("fssm_sha256", meta)

    def test_same_program_with_two_dispatch_states_gets_one_code_key_and_two_state_keys(self):
        """The review case on #1911: two containers with the same kind 6 and kind-0 entries for 32x1x1 and 64x1x1."""
        code = program()
        states = [kind0(threads=(32, 1, 1)), kind0(threads=(64, 1, 1))]
        with tempfile.TemporaryDirectory() as t:
            dll = os.path.join(t, "pair.dll")
            with open(dll, "wb") as f:
                f.write(dll_like(*(container(code, state) for state in states)))
            out = os.path.join(t, "out")
            r = subprocess.run([sys.executable, "-I", UNPACK, dll, "--out", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            metas = [shader_meta.collect(os.path.join(out, "disasm", f"pair_{n:02d}.s"), f"pair_{n:02d}", [[]], [[]])
                     for n in range(2)]
            self.assertEqual({m["fspx_code_sha256"] for m in metas}, {hashlib.sha256(code).hexdigest()})
            self.assertEqual([m["fspx_state_sha256"] for m in metas], [hashlib.sha256(s).hexdigest() for s in states])

    def test_an_fssm_blob_keeps_its_key_and_gets_no_fspx_key(self):
        with tempfile.TemporaryDirectory() as t:
            for sub, data in (("fssm", b"FSSM blob"), ("fspx", program())):
                os.makedirs(os.path.join(t, sub))
                ext = ".fssm" if sub == "fssm" else ".code"
                with open(os.path.join(t, sub, "s" + ext), "wb") as f:
                    f.write(data)
            disasm = os.path.join(t, "disasm", "s.s")
            self.assertIsNone(shader_meta.fspx_code_sha256_file(disasm, "s"))
            meta = shader_meta.collect(disasm, "s", [[]], [[]])
            self.assertEqual(meta["fssm_sha256"], hashlib.sha256(b"FSSM blob").hexdigest())
            self.assertNotIn("fspx_code_sha256", meta)

    def test_no_program_no_key(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(shader_meta.fspx_code_sha256_file(os.path.join(t, "disasm", "x.s"), "x"))
            self.assertNotIn("fspx_code_sha256", shader_meta.collect(os.path.join(t, "disasm", "x.s"), "x", [[]], [[]]))

    def test_malformed_directory_is_not_unpacked(self):
        """A kind-6 entry past the payload is skipped, as the DLL and the runtime skip it: no program, no key."""
        code = program()
        bad = bytearray(container(code))
        accc = bytes(bad).find(b"FSPX")
        for k in range(4):                                     # the kind-6 entry's size, made too large
            at = accc + 16 + 16 * k
            if struct.unpack_from("<I", bad, at)[0] == 6:
                struct.pack_into("<I", bad, at + 8, 0xFFFFFF00)
        self.assertIsNone(runtime_key(bytes(bad)))
        with tempfile.TemporaryDirectory() as t:
            dll = os.path.join(t, "bad.dll")
            with open(dll, "wb") as f:
                f.write(dll_like(bytes(bad)))
            r = subprocess.run([sys.executable, "-I", UNPACK, dll, "--out", os.path.join(t, "out")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("FSPX status 5", r.stderr)
            self.assertFalse(os.listdir(os.path.join(t, "out", "fspx")))


class FspxStage(unittest.TestCase):
    """Which tools fspx_shaders() runs. Each mock writes what the real tool writes last."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.dll = os.path.join(self.game, make_game.FSPX_DLL)
        self.write_dll(program())
        self.unpacked = os.path.join(self.work, "shaders", "fspx")
        self.spirv = os.path.join(self.work, "shaders", "spirv", "fspx")
        self.calls = []
        # A translation with programs publishes a non-empty index. Default the mock to one so the reuse tests keep
        # their meaning; a test that needs the empty case sets self.translations = {} itself.
        self.translations = {"upscaler_00": ("c" * 64, "s" * 64, b"synthetic module")}

    def write_dll(self, code):
        with open(self.dll, "wb") as f:
            f.write(dll_like(container(code)))

    def fake_py(self, tool, *args):
        self.calls.append((tool, args))
        if tool == "fspx_unpack.py":
            os.makedirs(os.path.join(self.unpacked, "disasm"), exist_ok=True)
            open(os.path.join(self.unpacked, "disasm", "upscaler.s"), "w").close()
        elif tool == "rdna2spirv.py":
            os.makedirs(self.spirv, exist_ok=True)
            open(os.path.join(self.spirv, "upscaler.spv"), "w").close()
            index = {}
            for name, (code, state, module) in getattr(self, "translations", {}).items():
                index[name] = {"stage": "cs", "fspx_code_sha256": code, "fspx_state_sha256": state}
                with open(os.path.join(self.spirv, name + ".spv"), "wb") as f:
                    f.write(module)
            with open(os.path.join(self.spirv, "index.json"), "w") as f:
                json.dump(index, f)

    def run_step(self, signature="llvm-mc 23", emulate=False, fallbacks=()):
        self.calls = []
        with mock.patch.object(make_game, "py", self.fake_py), mock.patch.object(make_game, "log", lambda *a: None):
            make_game.fspx_shaders(self.game, self.work, signature, emulate=emulate, fallbacks=fallbacks)
        return [tool for tool, _ in self.calls]

    def test_runs_once_then_nothing(self):
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        (_, unpack), (_, translate) = self.calls
        self.assertEqual(unpack, (self.dll, "--out", self.unpacked, "--disasm"))
        self.assertEqual(translate, (os.path.join(self.unpacked, "disasm"), "--out", self.spirv, "--no-asm"))
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "fspx_build.json")))
        self.assertEqual(self.run_step(), [])

    def test_empty_index_is_retranslated_every_run(self):
        """A translation that produced no modules publishes `{}`: that is not a finished FSPX stage, and
        the build marker must not latch it as done forever (the port would silently ship with zero FSPX programs)."""
        self.translations = {}
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {})
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "fspx_build.json")))
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])

    def test_empty_index_drops_stale_modules_from_the_previous_run(self):
        self.translations = {"upscaler_00": ("c" * 64, "s" * 64, b"stale module")}
        self.run_step()
        stale = os.path.join(self.spirv, "upscaler_00.spv")
        self.assertTrue(os.path.isfile(stale))
        # Simulate a translation that produced no modules over an existing cache: the marker still matches and the
        # old module is on disk.
        with open(os.path.join(self.spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
        self.translations = {}
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertFalse(os.path.exists(stale))
        with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {})

    def test_failed_disassembly_leaves_no_marker_and_is_retried(self):
        mc = os.path.join(self.tmp.name, "fake_mc.py")
        with open(mc, "w", encoding="utf-8", newline="\n") as f:
            f.write("import sys\nsys.stderr.write('error: disassembly stopped\\n')\nsys.exit(23)\n")
        real_py = make_game.py

        def run_tool(tool, *args):
            if tool == "fspx_unpack.py":
                self.calls.append((tool, args))
                real_py(tool, *args)
            else:
                self.fake_py(tool, *args)

        env = dict(LLVM_MC=f'"{sys.executable}" "{mc}"')
        with mock.patch.dict(os.environ, env), mock.patch.object(make_game, "py", run_tool), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                make_game.fspx_shaders(self.game, self.work, "llvm-mc 23")
            self.assertEqual(error.exception.code, 1)
            self.assertEqual([tool for tool, _ in self.calls], ["fspx_unpack.py"])
            self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
            self.assertFalse(os.path.exists(os.path.join(self.spirv, "fspx_build.json")))
            self.assertEqual(os.listdir(os.path.join(self.unpacked, "disasm")), [])
            with open(mc, "w", encoding="utf-8", newline="\n") as f:
                f.write("print('\\ts_endpgm')\n")
            self.calls = []
            make_game.fspx_shaders(self.game, self.work, "llvm-mc 23")
            self.assertEqual([tool for tool, _ in self.calls], ["fspx_unpack.py", "rdna2spirv.py"])
            self.assertTrue(os.path.isfile(os.path.join(self.spirv, "index.json")))
            self.assertTrue(os.path.isfile(os.path.join(self.spirv, "fspx_build.json")))
            self.calls = []
            make_game.fspx_shaders(self.game, self.work, "llvm-mc 23")
            self.assertEqual(self.calls, [])

    def test_missing_cached_module_is_rebuilt(self) -> None:
        self.translations = {"upscaler_00": ("c" * 64, "s" * 64, b"synthetic module")}
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertEqual(self.run_step(), [])
        module = os.path.join(self.spirv, "upscaler_00.spv")
        os.remove(module)

        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        with open(module, "rb") as f:
            self.assertEqual(f.read(), b"synthetic module")
        self.assertEqual(self.run_step(), [])

    def test_no_dll_no_step(self):
        os.remove(self.dll)
        self.assertEqual(self.run_step(), [])
        self.assertFalse(os.path.exists(self.spirv))

    def test_each_input_change_runs_it_again(self):
        self.run_step()
        self.assertEqual(self.run_step(signature="llvm-mc 24"), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertIn("--wave64-emulate", self.calls[1][1])
        flags = ("--no-float-controls",)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
                         ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertIn("--no-float-controls", self.calls[1][1])
        self.write_dll(program() + b"\0\0\x80\xbf")
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
                         ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), [])
        with mock.patch.object(make_game, "translation_sources_digest", return_value="another translator"):
            self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags),
                             ["fspx_unpack.py", "rdna2spirv.py"])

    def test_same_keys_with_different_modules_fail_the_step(self):
        """Same kind 6 and same kind 0 must translate to one module; otherwise the runtime could not choose."""
        self.translations = {"upscaler_00": ("c" * 64, "s" * 64, b"module A"),
                             "upscaler_01": ("c" * 64, "s" * 64, b"module B")}
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.run_step()
        self.assertIn("upscaler_00", err.getvalue())
        self.assertIn("upscaler_01", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))   # nothing for the runtime to load
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "fspx_build.json")))

    def test_same_code_key_is_allowed_with_other_state_or_the_same_module(self):
        self.translations = {"upscaler_00": ("c" * 64, "s" * 64, b"module A"),
                             "upscaler_01": ("c" * 64, "t" * 64, b"module B"),     # Vonsek's case: other kind 0
                             "upscaler_02": ("c" * 64, "s" * 64, b"module A")}     # a duplicate of 00
        self.assertEqual(self.run_step(), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "fspx_build.json")))

    def test_a_stopped_translation_leaves_no_marker_and_no_old_modules(self):
        self.run_step()
        open(os.path.join(self.spirv, "notes.txt"), "w").close()

        def stop(tool, *args):
            if tool == "rdna2spirv.py":
                raise KeyboardInterrupt
            self.fake_py(tool, *args)
        with mock.patch.object(make_game, "py", stop), mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(KeyboardInterrupt):
                make_game.fspx_shaders(self.game, self.work, "llvm-mc 24")
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "fspx_build.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "upscaler.spv")))   # from the run before
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "notes.txt")))
        self.assertEqual(self.run_step(signature="llvm-mc 24"), ["fspx_unpack.py", "rdna2spirv.py"])
        self.assertEqual(self.run_step(signature="llvm-mc 24"), [])


class ShadersStageWithTheDll(unittest.TestCase):
    """The whole shaders() stage with the DLL in the game folder: the FSPX step comes after the game's shaders and
    never makes them run again."""

    def test_fspx_runs_after_the_corpus_and_alone_when_only_its_inputs_change(self):
        with tempfile.TemporaryDirectory() as t:
            work, game = os.path.join(t, "work"), os.path.join(t, "game")
            os.makedirs(game)
            dll = os.path.join(game, make_game.FSPX_DLL)
            with open(dll, "wb") as f:
                f.write(dll_like(container(program())))
            ran = []
            dxbc = os.path.join(work, "shaders", "dxbc")

            def touch(*parts):
                os.makedirs(os.path.dirname(os.path.join(*parts)), exist_ok=True)
                open(os.path.join(*parts), "w").close()

            def fake_py(tool, *args):
                ran.append(tool)
                if tool == "shader_carve.py":
                    touch(dxbc, "index.json")
                    with open(os.path.join(dxbc, "index.json"), "w") as f:
                        f.write('[{"file": "shader.dxbc"}]')
                elif tool == "fssm_unpack.py":
                    touch(dxbc, "fssm_index.json")
                    with open(os.path.join(dxbc, "fssm_index.json"), "w") as f:
                        f.write('[{"name": "shader"}]')
                elif tool == "shader_disasm.py":
                    os.makedirs(os.path.join(dxbc, "disasm"), exist_ok=True)
                    # make_game checks that every unpacked shader got a non-empty listing
                    with open(os.path.join(dxbc, "disasm", "shader.s"), "w") as f:
                        f.write("s_endpgm\n")
                elif tool == "rdna2spirv.py":
                    out = args[args.index("--out") + 1]
                    touch(out, "index.json")
                    # The FSPX translation must publish a real program, not `{}`: an empty index is an empty step and
                    # makes the second stage() call retranslate. An entry with no fspx_code_sha256 is skipped by
                    # fspx_key_clash, so this is enough to model one translated module.
                    with open(os.path.join(out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                        f.write('{"upscaler_00": {}}')
                    with open(os.path.join(out, "upscaler_00.spv"), "wb") as f:
                        f.write(b"synthetic module")

            def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
                ran.append("translate")
                touch(spirv, "index.json")

            def stage():
                ran.clear()
                with mock.patch.object(make_game, "py", fake_py), \
                        mock.patch.object(make_game, "need", lambda *a: None), \
                        mock.patch.object(make_game, "translate", fake_translate), \
                        mock.patch.object(make_game, "llvm_mc_signature", return_value="llvm-mc 23"), \
                        mock.patch.object(make_game, "log", lambda *a: None):
                    make_game.shaders(game, work, 2)
                return list(ran)

            self.assertEqual(stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate",
                                       "fspx_unpack.py", "rdna2spirv.py"])
            self.assertEqual(stage(), [])
            with open(dll, "wb") as f:
                f.write(dll_like(container(program() + b"\0\0\x80\xbf")))
            self.assertEqual(stage(), ["fspx_unpack.py", "rdna2spirv.py"])
            self.assertTrue(os.path.isfile(os.path.join(work, "shaders", "spirv", "index.json")))
            self.assertTrue(os.path.isfile(os.path.join(work, "shaders", "spirv", "fspx", "index.json")))


class FspxCacheComplete(unittest.TestCase):
    """Pin fspx_cache_complete(), the predicate that decides whether the FSPX outputs can be reused."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spirv = os.path.join(self.tmp.name, "shaders", "spirv", "fspx")
        os.makedirs(self.spirv)

    def write_index(self, text):
        with open(os.path.join(self.spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def test_empty_index_is_not_complete(self):
        self.write_index("{}")
        self.assertFalse(make_game.fspx_cache_complete(self.spirv))

    def test_missing_index_is_not_complete(self):
        self.assertFalse(make_game.fspx_cache_complete(self.spirv))

    def test_malformed_index_is_not_complete(self):
        self.write_index("{not json")
        self.assertFalse(make_game.fspx_cache_complete(self.spirv))

    def test_non_mapping_indexes_are_not_complete(self):
        for text in ("[]", "null"):
            with self.subTest(text=text):
                self.write_index(text)
                self.assertFalse(make_game.fspx_cache_complete(self.spirv))

    def test_one_entry_with_its_module_is_complete(self):
        self.write_index('{"upscaler_00": {}}')
        with open(os.path.join(self.spirv, "upscaler_00.spv"), "wb") as f:
            f.write(b"synthetic module")
        self.assertTrue(make_game.fspx_cache_complete(self.spirv))

    def test_one_entry_without_its_module_is_not_complete(self):
        self.write_index('{"upscaler_00": {}}')
        self.assertFalse(make_game.fspx_cache_complete(self.spirv))


if __name__ == "__main__":
    unittest.main()
