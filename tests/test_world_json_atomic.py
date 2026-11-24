"""world.json publishing: private staging file, sibling files and overlapping writes; no game data.

    python -I tests/test_world_json_atomic.py -v
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import world_build


DOC = {"version": 1, "chunks": ["c_0_0.glb"]}


class WorldJsonAtomicTests(unittest.TestCase):
    def test_success_preserves_an_existing_tmp_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "world.json"
            sibling = Path(str(target) + ".tmp")
            target.write_text('{"old": true}\n', encoding="utf-8")
            sibling.write_bytes(b"unrelated file")

            world_build._write_json_atomic(str(target), DOC)

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), DOC)
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_failed_replace_keeps_destination_and_sibling_and_removes_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "world.json"
            sibling = Path(str(target) + ".tmp")
            target.write_text('{"old": true}\n', encoding="utf-8")
            sibling.write_bytes(b"unrelated file")

            with mock.patch.object(world_build.os, "replace", side_effect=PermissionError("busy")):
                with self.assertRaisesRegex(PermissionError, "busy"):
                    world_build._write_json_atomic(str(target), DOC)

            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_each_write_stages_through_its_own_file_in_the_destination_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "world.json")
            sources = []
            real_replace = os.replace

            def recording_replace(source, destination):
                sources.append(source)
                real_replace(source, destination)

            with mock.patch.object(world_build.os, "replace", side_effect=recording_replace):
                world_build._write_json_atomic(target, DOC)
                world_build._write_json_atomic(target, {"version": 2})

            self.assertEqual(len(set(sources)), 2, "two writes shared a staging file")
            for source in sources:
                self.assertNotEqual(source, target + ".tmp")
                self.assertEqual(os.path.dirname(os.path.abspath(source)), os.path.abspath(directory))
            self.assertEqual(json.loads(Path(target).read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(os.listdir(directory), ["world.json"])

    def test_publish_world_replaces_world_json_with_the_retrying_replace(self):
        # The viewer can hold world.json open during a rebuild; on Windows the chunks already go through
        # glb_writer._replace, which retries a sharing violation, and world.json must too.
        with tempfile.TemporaryDirectory() as directory:
            staging = os.path.join(directory, "staging")
            os.makedirs(staging)
            world_path = os.path.join(directory, "world.json")
            calls = []
            glb_mod = mock.Mock()
            glb_mod._replace.side_effect = lambda source, destination: (calls.append(destination),
                                                                          os.replace(source, destination))
            world_build._publish_world(staging, os.path.join(directory, "chunks"), [], world_path, DOC, glb_mod)

            self.assertIn(world_path, calls)
            self.assertEqual(json.loads(Path(world_path).read_text(encoding="utf-8")), DOC)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_published_file_keeps_the_replaced_files_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "world.json"
            target.write_text("{}\n", encoding="utf-8")
            os.chmod(target, 0o640)

            world_build._write_json_atomic(str(target), DOC)

            self.assertEqual(os.stat(target).st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main()
