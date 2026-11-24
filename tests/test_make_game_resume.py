"""make_game.py: a resumed shaders stage keeps the translation shards that already finished.

Standard library only; no GPU, no Vulkan SDK, no game data: the shard processes and the merge are mocked. A fake
shard writes its marker and then index.part<i>.json the way rdna2spirv.py --shard does.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class ResumeShardsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.spirv = tmp.name

    def part(self, i):
        return os.path.join(self.spirv, f"index.part{i}.json")

    def tag(self, i):
        with open(self.part(i), encoding="utf-8") as f:
            return json.load(f)["tag"]

    def translate(self, jobs, emulate=False, digest="d1", tag="a", crash=()):
        """Run translate() with fake shards; returns the shard indices it started. Shards in `crash` exit with 1
        and write no part, like a shard that was stopped or failed."""
        started = []
        spirv = self.spirv

        class FakeProc:
            def __init__(self, args):
                self.i = int(args[args.index("--shard") + 1].split("/")[0])
                self.n = int(args[args.index("--shard") + 1].split("/")[1])
                started.append(self.i)

            def wait(self):
                if self.i in crash:
                    return 1
                with open(os.path.join(spirv, f"shard{self.i}.json"), "w", encoding="utf-8") as f:
                    json.dump({"index": self.i, "count": self.n}, f)
                with open(os.path.join(spirv, f"index.part{self.i}.json"), "w", encoding="utf-8") as f:
                    json.dump({"tag": tag, "n": self.n}, f)
                return 0
        with mock.patch.object(make_game.subprocess, "Popen", side_effect=FakeProc), \
                mock.patch.object(make_game, "py") as py, \
                mock.patch.object(make_game, "translator_digest", return_value=digest), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.translate("disasm", self.spirv, jobs, emulate=emulate)
        self.assertEqual(py.call_count, 1)
        self.assertIn("--merge", py.call_args.args)
        return started

    def test_a_fresh_run_starts_every_shard_and_records_its_plan(self):
        self.assertEqual(self.translate(3), [0, 1, 2])
        with open(os.path.join(self.spirv, make_game.SHARD_PLAN), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"jobs": 3, "wave64_emulate": False, "fallbacks": [],
                                            "translator": "d1"})

    def test_a_stop_during_the_merge_only_merges_again(self):
        self.translate(3, tag="a")              # the merge is mocked, so no index.json: as if stopped there
        self.assertEqual(self.translate(3, tag="b"), [])
        self.assertEqual([self.tag(i) for i in range(3)], ["a", "a", "a"])

    def test_a_stop_part_way_runs_only_the_missing_shards(self):
        self.translate(4, tag="a")
        os.remove(self.part(1))
        os.remove(self.part(3))
        self.assertEqual(self.translate(4, tag="b"), [1, 3])
        self.assertEqual([self.tag(i) for i in range(4)], ["a", "b", "a", "b"])

    def test_a_failed_shard_is_retried_alone(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.translate(3, tag="a", crash={1})
        self.assertFalse(os.path.exists(self.part(1)))
        self.assertEqual(self.translate(3, tag="b"), [1])

    def test_a_shard_with_a_missing_marker_is_retried_alone(self) -> None:
        self.translate(3, tag="a")
        os.remove(os.path.join(self.spirv, "shard1.json"))
        self.assertEqual(self.translate(3, tag="b"), [1])
        self.assertEqual([self.tag(i) for i in range(3)], ["a", "b", "a"])

    def test_another_jobs_count_translates_everything_again(self):
        self.translate(3, tag="a")
        self.assertEqual(self.translate(4, tag="b"), [0, 1, 2, 3])
        self.assertEqual([self.tag(i) for i in range(4)], ["b"] * 4)

    def test_fewer_jobs_leave_no_part_of_the_old_run_behind(self):
        self.translate(4, tag="a")
        self.assertEqual(self.translate(2, tag="b"), [0, 1])
        self.assertEqual(sorted(f for f in os.listdir(self.spirv) if f.startswith("index.part")),
                         ["index.part0.json", "index.part1.json"])

    def test_another_wave64_mode_translates_everything_again(self):
        self.translate(2, emulate=False, tag="a")
        self.assertEqual(self.translate(2, emulate=True, tag="b"), [0, 1])
        self.assertEqual([self.tag(i) for i in range(2)], ["b", "b"])

    def test_another_translator_source_translates_everything_again(self):
        self.translate(2, digest="d1", tag="a")
        self.assertEqual(self.translate(2, digest="d2", tag="b"), [0, 1])
        self.assertEqual([self.tag(i) for i in range(2)], ["b", "b"])

    def test_parts_without_a_plan_are_not_trusted(self):
        # A folder from before the plan existed: its parts can come from any --jobs.
        for i in range(2):
            with open(self.part(i), "w", encoding="utf-8") as f:
                json.dump({"tag": "old"}, f)
        self.assertEqual(self.translate(2, tag="b"), [0, 1])

    def test_an_unreadable_plan_is_not_trusted(self):
        self.translate(2, tag="a")
        with open(os.path.join(self.spirv, make_game.SHARD_PLAN), "w", encoding="utf-8") as f:
            f.write("{")
        self.assertEqual(self.translate(2, tag="b"), [0, 1])


class EarlierStepDropsThePlanTest(unittest.TestCase):
    """A disassembly that runs again makes every finished shard stale, even with the same --jobs and mode."""

    def test_disassembling_again_drops_the_plan(self):
        with tempfile.TemporaryDirectory() as work:
            dxbc = os.path.join(work, "shaders", "dxbc")
            spirv = os.path.join(work, "shaders", "spirv")
            os.makedirs(dxbc)
            os.makedirs(spirv)
            for name, entries in (("index.json", [{"file": "shader.dxbc"}]),
                                  ("fssm_index.json", [{"name": "shader"}])):
                with open(os.path.join(dxbc, name), "w", encoding="utf-8") as f:
                    json.dump(entries, f)
            plan = os.path.join(spirv, make_game.SHARD_PLAN)
            with open(plan, "w", encoding="utf-8") as f:
                json.dump({"jobs": 2, "wave64_emulate": False, "translator": "d1"}, f)
            seen = {}

            def fake_translate(disasm, out, jobs, emulate=False, fallbacks=()):
                seen["plan"] = os.path.exists(plan)
            def fake_disassemble(*args):
                disasm = os.path.join(dxbc, "disasm")
                os.makedirs(disasm)
                with open(os.path.join(disasm, "shader.s"), "w", encoding="utf-8", newline="\n") as f:
                    f.write("synthetic listing\n")

            with mock.patch.object(make_game, "py", side_effect=fake_disassemble), mock.patch.object(make_game, "need"), \
                    mock.patch.object(make_game, "llvm_mc_signature", return_value="test llvm-mc 21"), \
                    mock.patch.object(make_game, "translate", side_effect=fake_translate), \
                    mock.patch.object(make_game, "log", lambda *a: None):
                make_game.shaders("game", work, 2)
            self.assertEqual(seen, {"plan": False})


class TranslatorDigestTest(unittest.TestCase):
    def test_any_file_in_tools_changes_it_and_folders_do_not(self):
        with tempfile.TemporaryDirectory() as tools:
            for name, text in (("rdna2spirv.py", "a"), ("gfx10_formats.json", "{}")):
                with open(os.path.join(tools, name), "w", encoding="utf-8") as f:
                    f.write(text)
            with mock.patch.object(make_game, "TOOLS", tools):
                first = make_game.translator_digest()
                os.makedirs(os.path.join(tools, "__pycache__"))
                self.assertEqual(make_game.translator_digest(), first)
                with open(os.path.join(tools, "gfx10_formats.json"), "w", encoding="utf-8") as f:
                    f.write("{ }")
                self.assertNotEqual(make_game.translator_digest(), first)

    def test_the_real_tools_folder_gives_a_sha256(self):
        digest = make_game.translator_digest()
        self.assertEqual(len(digest), 64)
        int(digest, 16)


if __name__ == "__main__":
    unittest.main()
