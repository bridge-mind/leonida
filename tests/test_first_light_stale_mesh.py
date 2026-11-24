"""first_light: a preparation run without an OBJ must not leave (and then draw) the mesh.bin of an earlier run.

The carve, translation and draw are stubbed; the folders hold placeholder bytes only. No game data, no GPU.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import first_light  # noqa: E402


class StaleMesh(unittest.TestCase):
    def test_run_without_obj_drops_old_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, out, game = (os.path.join(tmp, n) for n in ("work", "out", "game"))
            os.makedirs(os.path.join(work, "first_light", "spirv"))
            os.makedirs(os.path.join(work, "shaders", "dxbc"))
            os.makedirs(out)
            os.makedirs(game)
            picked = [{"stage": "vs", "file": "a.dxbc", "name": "a"}, {"stage": "ps", "file": "b.dxbc", "name": "b"}]
            for e in picked:
                stem = os.path.splitext(e["file"])[0]
                for path in (os.path.join(work, "first_light", "spirv", stem + ".spv"),
                             os.path.join(work, "shaders", "dxbc", e["file"])):
                    with open(path, "wb") as f:
                        f.write(b"x")
            mesh = os.path.join(out, "mesh.bin")
            with open(mesh, "wb") as f:
                f.write(b"old mesh")
            seen = []
            saved = (first_light.ensure_carve, first_light.choose, first_light.translate, first_light.run_draw)
            first_light.ensure_carve = lambda g, w: "index"
            first_light.choose = lambda i, v, p: picked
            first_light.translate = lambda w, p: {"vs": {}, "ps": {}}
            first_light.run_draw = lambda a, m, o: seen.append(os.path.exists(mesh))
            try:
                first_light.main(["--game", game, "--work", work, "--out", out, "--no-fit", "--run"])
            finally:
                (first_light.ensure_carve, first_light.choose, first_light.translate, first_light.run_draw) = saved
            self.assertEqual(seen, [False])


if __name__ == "__main__":
    unittest.main()
