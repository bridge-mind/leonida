"""First Light must not render geometry left by an earlier preparation run."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import first_light


class FirstLightMeshDrawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.work / "out"
        self.out.mkdir()
        self.mesh = self.out / "mesh.bin"
        self.mesh.write_bytes(b"previous mesh placeholder")
        dxbc = self.work / "shaders" / "dxbc"
        spirv = self.work / "first_light" / "spirv"
        dxbc.mkdir(parents=True)
        spirv.mkdir(parents=True)
        picks = []
        for stage in ("vs", "ps"):
            (dxbc / f"{stage}.dxbc").touch()
            (spirv / f"{stage}.spv").touch()
            picks.append({"stage": stage, "name": stage, "file": f"{stage}.dxbc"})
        self.enterContext(patch.multiple(
            first_light,
            ensure_carve=lambda *args: "stub-index",
            choose=lambda *args: picks,
            translate=lambda *args: {"vs": {}, "ps": {}},
            find_harness=lambda *args: "stub-harness",
        ))
        self.draw = self.enterContext(patch.object(first_light.subprocess, "run", side_effect=self.write_readback))

    @staticmethod
    def write_readback(arguments: list[str], *, env: dict[str, str], check: bool) -> None:
        if not Path(env["VKH_DRAW_MESH"]).is_file():
            raise AssertionError("the harness received no mesh")
        pixels = (0.0, 0.0, 0.0, 1.0) * 2 + (-1.0,) * 8
        Path(arguments[3]).write_bytes(struct.pack("<16f", *pixels))

    def prepare(self, *args: str) -> int:
        return first_light.main([
            "--game", str(self.work), "--work", str(self.work), "--out", str(self.out),
            "--no-fit", "--width", "2", "--height", "2", *args,
        ])

    def test_run_without_obj_never_draws_the_previous_mesh(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires an OBJ with faces"):
            self.prepare("--run")
        self.draw.assert_not_called()
        self.assertFalse((self.out / "image.raw").exists())

    def test_prepare_without_obj_is_allowed_but_cannot_be_replayed(self) -> None:
        self.assertEqual(self.prepare(), 0)
        self.draw.assert_not_called()
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["vertex_count"], manifest["index_count"]), (0, 0))
        self.assertFalse(self.mesh.is_file())   # a run without an OBJ removes the previous run's mesh.bin (95ebfb8e9)
        with self.assertRaisesRegex(SystemExit, "requires an OBJ with faces"):
            first_light.main(["--check", "--out", str(self.out)])
        self.draw.assert_not_called()

    def test_check_can_replay_a_manifest_for_prepared_geometry(self) -> None:
        vertices = [(0.0, 0.0, 0.5, 0.0, 0.0), (1.0, 0.0, 0.5, 1.0, 0.0), (0.0, 1.0, 0.5, 0.0, 1.0)]
        first_light.write_mesh(self.mesh, vertices, [0, 1, 2])
        (self.out / "manifest.json").write_text(
            '{"width": 2, "height": 2, "vertex_count": 3, "index_count": 3, "files": {"mesh": "mesh.bin"}}',
            encoding="utf-8", newline="\n",
        )
        self.assertEqual(first_light.main(["--check", "--out", str(self.out)]), 0)
        self.draw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
