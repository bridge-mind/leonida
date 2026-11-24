"""make_game.py: an empty translated shader index is not kept, and a later stage refuses one.

Standard library only; no GPU, no Vulkan SDK, no game data: the shard processes and rdna2spirv.py are mocked.

    python -I tests/test_make_game_empty_translation.py
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Done:
    def poll(self):
        return 0

    def wait(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


class TranslateEmptyIndexTest(unittest.TestCase):
    """translate(): a merge that published nothing is an error and must not leave a marker behind."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.disasm = os.path.join(tmp.name, "dxbc", "disasm")
        self.spirv = os.path.join(tmp.name, "spirv")
        os.makedirs(self.disasm)

    def translate_with_index(self, text):
        def fake_py(tool, *args):
            if "--merge" in args:
                write(os.path.join(self.spirv, "index.json"), text)

        with mock.patch.object(make_game, "log", lambda *a: None), \
                mock.patch.object(make_game, "py", side_effect=fake_py), \
                mock.patch.object(make_game.subprocess, "Popen", lambda *a, **k: Done()):
            make_game.translate(self.disasm, self.spirv, 1)

    def test_an_empty_index_is_removed_with_the_plan_and_the_retry_fails(self):
        for _ in range(2):
            with self.assertRaises(SystemExit) as cm:
                self.translate_with_index("{}")
            self.assertEqual(cm.exception.code, 1)
            self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
            self.assertFalse(os.path.exists(os.path.join(self.spirv, make_game.SHARD_PLAN)))

    def test_an_unreadable_index_is_removed(self):
        with self.assertRaises(SystemExit) as cm:
            self.translate_with_index("not json")
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))

    def test_a_non_empty_index_is_kept(self):
        self.translate_with_index(json.dumps({"shader_0": {"stage": "ps"}}))
        with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"shader_0": {"stage": "ps"}})


class TranslateLockedIndexTest(unittest.TestCase):
    """translate(): an index that cannot be read or removed (on Windows, a file an antivirus scan or the indexer
    holds) ends the stage with its own message naming the file once, not with a PermissionError traceback."""

    setUp = TranslateEmptyIndexTest.setUp
    translate_with_index = TranslateEmptyIndexTest.translate_with_index

    def locked(self, path):
        return PermissionError(13, "Permission denied", path)

    def failure(self, text, open_locked=False, shard_code=0):
        index = os.path.join(self.spirv, "index.json")
        real_remove, real_open = os.remove, open

        def remove(path, *a, **k):
            if os.path.abspath(path) == os.path.abspath(index):
                raise self.locked(path)
            return real_remove(path, *a, **k)

        def opener(path, mode="r", *a, **k):
            if open_locked and mode == "r" and os.path.abspath(path) == os.path.abspath(index):
                raise self.locked(path)
            return real_open(path, mode, *a, **k)

        class Exited(Done):
            def wait(self):
                return shard_code

        err = io.StringIO()
        with mock.patch.object(make_game.os, "remove", remove), \
                mock.patch("builtins.open", opener), \
                mock.patch.object(make_game.subprocess, "Popen", lambda *a, **k: Exited()), \
                mock.patch("sys.stderr", err), \
                self.assertRaises(SystemExit) as cm:
            if shard_code:
                write(index, text)
                with mock.patch.object(make_game, "log", lambda *a: None):
                    make_game.translate(self.disasm, self.spirv, 1)
            else:
                self.translate_with_index(text)
        self.assertEqual(cm.exception.code, 1)
        message = err.getvalue()
        self.assertIn(f"{index} could not be removed (Permission denied): delete it before running", message)
        self.assertNotIn(repr(index), message)
        return message

    def test_an_unopenable_index_that_cannot_be_removed(self):
        message = self.failure('{"shader_0": {}}', open_locked=True)
        index = os.path.join(self.spirv, "index.json")
        self.assertIn(f"--merge wrote an unreadable {index}: Permission denied.", message)

    def test_an_unparsable_index_that_cannot_be_removed(self):
        self.assertIn("--merge wrote an unreadable", self.failure("not json"))

    def test_an_empty_index_that_cannot_be_removed(self):
        self.assertIn("translated 0 shaders", self.failure("{}"))

    def test_a_stale_index_after_a_failed_shard_that_cannot_be_removed(self):
        self.assertIn("rdna2spirv.py: shard 0 exited with 1.", self.failure('{"old": {}}', shard_code=1))


class RequireTranslatedShadersTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(self.spirv)

    def call(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            return make_game.require_translated_shaders(self.work), err.getvalue()

    def test_missing_index_names_the_stage(self):
        with self.assertRaises(SystemExit) as cm:
            self.call()
        self.assertEqual(cm.exception.code, 1)

    def test_an_empty_index_is_refused(self):
        write(os.path.join(self.spirv, "index.json"), "{}")
        with self.assertRaises(SystemExit) as cm:
            self.call()
        self.assertEqual(cm.exception.code, 1)

    def test_an_unreadable_index_is_refused(self):
        write(os.path.join(self.spirv, "index.json"), "not json")
        with self.assertRaises(SystemExit) as cm:
            self.call()
        self.assertEqual(cm.exception.code, 1)

    def test_a_non_empty_index_is_accepted(self):
        write(os.path.join(self.spirv, "index.json"), '{"shader_0": {}}')
        self.assertEqual(self.call()[0], self.spirv)


class ShadersStageTest(unittest.TestCase):
    """shaders(): a saved `{}` from a merge that produced nothing is translated again, and its plan goes with it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.dxbc = os.path.join(self.work, "shaders", "dxbc")
        self.spirv = os.path.join(self.work, "shaders", "spirv")

    def index(self):
        return os.path.join(self.spirv, "index.json")

    def plan(self):
        return os.path.join(self.spirv, make_game.SHARD_PLAN)

    def run_stage(self):
        ran = []

        def fake_py(tool, *args):
            if tool == "shader_carve.py":
                write(os.path.join(self.dxbc, "index.json"), json.dumps([{"file": "shader.dxbc"}]))
            elif tool == "fssm_unpack.py":
                write(os.path.join(self.dxbc, "fssm_index.json"), json.dumps([{"name": "shader"}]))
            elif tool == "shader_disasm.py":
                write(os.path.join(self.dxbc, "disasm", "shader.s"), "synthetic listing\n")

        def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
            ran.append("translate")
            write(os.path.join(self.spirv, "index.json"), json.dumps({"shader_0": {"stage": "ps"}}))

        with mock.patch.object(make_game, "py", fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", fake_translate), \
                mock.patch.object(make_game, "llvm_mc_signature", return_value="test llvm-mc 21"), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.shaders(os.path.join(self.work, "game"), self.work, 2)
        return ran

    def test_an_empty_saved_index_is_translated_again_without_its_plan(self):
        self.assertEqual(self.run_stage(), ["translate"])
        write(self.index(), "{}")
        write(self.plan(), "{}")
        self.assertEqual(self.run_stage(), ["translate"])
        self.assertFalse(os.path.exists(self.plan()))

    def test_a_saved_index_that_names_a_shader_is_kept(self):
        self.run_stage()
        self.assertEqual(self.run_stage(), [])
        with open(self.index(), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"shader_0": {"stage": "ps"}})


if __name__ == "__main__":
    unittest.main()
