"""Texture pools in make_game.assets() must not collide on their basename (no game files).

    python -I tests/test_make_game_assets_pools.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_game  # noqa: E402
import ttpf_assets  # noqa: E402


class AssetsPoolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = os.path.join(self.tmp.name, "game")
        self.work = os.path.join(self.tmp.name, "work")
        self.out = os.path.join(self.work, "assets")
        # Mark the other assets stages done so only the texture-pool loop runs.
        for rel in make_game.MODEL_ARCHIVES:
            stem = os.path.splitext(os.path.basename(rel))[0]
            for mode in ("models", "textures"):
                self.stub(os.path.join(self.out, stem, mode, "manifest.json"))
                with open(os.path.join(self.out, stem, mode, make_game.DECODER_RECORD), "w") as f:
                    f.write(make_game.decoder_digest())
        self.stub(os.path.join(self.out, "audio", "audio_manifest.json"))
        self.stub(os.path.join(self.out, "audio", "ogg.done"))
        self.stub(os.path.join(self.out, "audio", "xma.done"))

    def stub(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w"):
            pass

    def pool(self, rel, size=100):
        path = os.path.join(self.game, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        return path

    def run_assets(self, decoder=None):
        """Run assets() with a fake exporter; return (source, pool directory) per ttpf call. With decoder, that is
        the decoder digest for the run (the model and texture stubs get its record, so only the pools can differ)."""
        calls = []
        digest = decoder or make_game.decoder_digest()
        for rel in make_game.MODEL_ARCHIVES:
            stem = os.path.splitext(os.path.basename(rel))[0]
            for mode in ("models", "textures"):
                with open(os.path.join(self.out, stem, mode, make_game.DECODER_RECORD), "w") as f:
                    f.write(digest)

        def fake_run(cmd, *a, **k):
            if cmd[1] == "ttpf":
                src, dest = cmd[2], cmd[3]
                pool_dir = os.path.join(dest, os.path.splitext(os.path.basename(src))[0])
                os.makedirs(pool_dir, exist_ok=True)
                with open(os.path.join(pool_dir, "manifest.json"), "w") as f:
                    json.dump({"src": src, "Count": 0, "Textures": []}, f)
                calls.append((src, pool_dir))

        with mock.patch.object(make_game, "rpf9dump", lambda w: "rpf9dump"), \
             mock.patch.object(make_game, "run", fake_run), \
             mock.patch.object(make_game.shutil, "which", lambda n: None), \
             mock.patch.object(make_game, "log", lambda *a: None), \
             mock.patch.object(make_game, "decoder_digest", lambda: digest), \
             mock.patch.object(ttpf_assets, "complete_pool", lambda *a: (0, 0)):
            make_game.assets(self.game, self.work, False, 1)
        return calls

    def test_same_basename_in_two_folders_exports_both(self):
        a = self.pool("data/ttdb/A/pool.ttpf")
        b = self.pool("data/ttdb/B/pool.ttpf", 101)
        calls = self.run_assets()
        self.assertEqual([src for src, _ in calls], [a, b])
        dirs = [d for _, d in calls]
        self.assertEqual(len(set(dirs)), 2)
        for src, d in calls:
            self.assertEqual(os.path.basename(d), "pool")
            self.assertTrue(os.path.isfile(os.path.join(d, "manifest.json")), src)
        # A resumed build finds both manifests and must not export either pool again.
        self.assertEqual(self.run_assets(), [])

    def test_unique_pool_keeps_its_existing_output_folder(self):
        p = self.pool("data/ttdb/pool.ttpf")
        self.assertEqual(self.run_assets(), [(p, os.path.join(self.out, "ttpf", "pool"))])

    def test_same_basename_in_different_packs_stays_separate(self):
        a = self.pool("pack/a/data/ttdb/pool.ttpf")
        b = self.pool("pack/b/data/ttdb/pool.ttpf", 101)
        self.assertEqual([d for _, d in self.run_assets()], [
            os.path.join(self.out, "ttpf_pack", "a", "pool"),
            os.path.join(self.out, "ttpf_pack", "b", "pool"),
        ])

    def test_collision_is_scoped_to_one_destination_namespace(self):
        loose = self.pool("data/ttdb/pool.ttpf")
        packed = self.pool("pack/a/data/ttdb/pool.ttpf", 101)
        self.assertEqual([d for _, d in self.run_assets()], [
            os.path.join(self.out, "ttpf", "pool"),
            os.path.join(self.out, "ttpf_pack", "a", "pool"),
        ])
        self.assertEqual(len({loose, packed}), 2)

    def test_another_decoder_exports_the_pool_again(self):
        p = self.pool("data/ttdb/pool.ttpf")
        pool_dir = os.path.join(self.out, "ttpf", "pool")
        self.assertEqual(self.run_assets("d1"), [(p, pool_dir)])
        with open(os.path.join(pool_dir, make_game.DECODER_RECORD), encoding="utf-8") as f:
            self.assertEqual(f.read(), "d1\n")
        self.assertEqual(self.run_assets("d1"), [])
        # rpf9dump keeps the DDS it finds, so the old decoder's files must be gone before the new export.
        self.stub(os.path.join(pool_dir, "000000_old.dds"))
        self.assertEqual(self.run_assets("d2"), [(p, pool_dir)])
        self.assertFalse(os.path.exists(os.path.join(pool_dir, "000000_old.dds")))
        with open(os.path.join(pool_dir, make_game.DECODER_RECORD), encoding="utf-8") as f:
            self.assertEqual(f.read(), "d2\n")
        self.assertEqual(self.run_assets("d2"), [])

    def test_a_pool_without_the_record_exports_again(self):
        p = self.pool("data/ttdb/pool.ttpf")
        pool_dir = os.path.join(self.out, "ttpf", "pool")
        self.run_assets("d1")
        os.remove(os.path.join(pool_dir, make_game.DECODER_RECORD))
        self.assertEqual(self.run_assets("d1"), [(p, pool_dir)])

    def test_a_colliding_pool_keeps_its_own_record(self):
        a = self.pool("data/ttdb/A/pool.ttpf")
        b = self.pool("data/ttdb/B/pool.ttpf", 101)
        dirs = [d for _, d in self.run_assets("d1")]
        self.assertEqual(len(set(dirs)), 2)
        for d in dirs:
            self.assertTrue(os.path.isfile(os.path.join(d, make_game.DECODER_RECORD)), d)
        self.assertEqual([src for src, _ in self.run_assets("d2")], [a, b])

    def test_a_broken_calibration_profile_stops_the_build_with_one_error_line(self):
        # The profile is hand-written, so a typo must name the file in one ERROR line, not raise a JSONDecodeError.
        self.pool("data/ttdb/pool.ttpf")
        with open(os.path.join(self.work, "bcpack-calibration.json"), "w", encoding="utf-8") as f:
            f.write("{")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            self.run_assets()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertIn("bcpack-calibration.json", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
