"""Unit and caller integration tests for tools/shader_stage_manifest.py.

    py -3.13 -I tests/test_shader_stage_manifest.py -v

Zero game data: all tests use synthetic temporary manifests.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_game  # noqa: E402
from shader_stage_manifest import shader_stems  # noqa: E402

FSSM_UNPACK = os.path.join(ROOT, "tools", "fssm_unpack.py")
SHADER_DISASM = os.path.join(ROOT, "tools", "shader_disasm.py")


class ShaderStemsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, name: str, data) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def write_bytes(self, name: str, raw: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(raw)
        return path

    def test_valid_carve_index_with_suffix_and_duplicates(self):
        path = self.write_json("index.json", [
            {"file": "ps_water.dxbc"},
            {"file": "vs-main_01.dxbc"},
            {"file": "shader.v1.dxbc"},
            {"file": "ps_water.dxbc"},
        ])
        self.assertEqual(
            shader_stems(path, "file", ".dxbc"),
            {"ps_water", "vs-main_01", "shader.v1"},
        )

    def test_valid_unpack_index_without_suffix(self):
        path = self.write_json("fssm_index.json", [
            {"name": "cs_bloom_0"},
            {"name": "gs-pass.v2"},
            {"name": "cs_bloom_0"},
        ])
        self.assertEqual(
            shader_stems(path, "name"),
            {"cs_bloom_0", "gs-pass.v2"},
        )

    def test_empty_list_returns_empty_set(self):
        path = self.write_json("empty.json", [])
        self.assertEqual(shader_stems(path, "file", ".dxbc"), set())
        self.assertEqual(shader_stems(path, "name"), set())

    def test_rejects_traversal_stems_after_suffix_stripping(self):
        for bad in ("..dxbc", "...dxbc", ".dxbc"):
            with self.subTest(value=bad):
                path = self.write_json("bad_suffix.json", [{"file": bad}])
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "file", ".dxbc")
                self.assertIn(path, str(ctx.exception))

    def test_rejects_bare_dot_stems_without_suffix(self):
        for bad in (".", ".."):
            with self.subTest(value=bad):
                path = self.write_json("bad_dots.json", [{"name": bad}])
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "name")
                self.assertIn(path, str(ctx.exception))

    def test_rejects_separators_drive_prefixes_and_nul_bytes(self):
        bad_values = [
            "../a.dxbc",
            "sub/a.dxbc",
            "sub\\a.dxbc",
            "C:a.dxbc",
            "C:\\a.dxbc",
            "a\x00b.dxbc",
            "a\nb.dxbc",
        ]
        for bad in bad_values:
            with self.subTest(value=bad):
                path = self.write_json("bad_path.json", [{"file": bad}])
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "file", ".dxbc")
                self.assertIn(path, str(ctx.exception))

    def test_rejects_wrong_suffix(self):
        path = self.write_json("wrong_suffix.json", [{"file": "ps_water.fssm"}])
        with self.assertRaises(ValueError) as ctx:
            shader_stems(path, "file", ".dxbc")
        self.assertIn(path, str(ctx.exception))

    def test_rejects_missing_or_invalid_field_values(self):
        cases = [
            [{"other": "ps_water.dxbc"}],
            [{"file": None}],
            [{"file": 123}],
            [{"file": ""}],
            ["ps_water.dxbc"],
            [123],
            [None],
        ]
        for idx, rows in enumerate(cases):
            with self.subTest(case=idx):
                path = self.write_json(f"bad_row_{idx}.json", rows)
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "file", ".dxbc")
                self.assertIn(path, str(ctx.exception))

    def test_rejects_non_list_root(self):
        for idx, root in enumerate(({}, "x", 42, None, True)):
            with self.subTest(root=root):
                path = self.write_json(f"bad_root_{idx}.json", root)
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "file", ".dxbc")
                self.assertIn(path, str(ctx.exception))

    def test_malformed_json_raises_value_error_with_path(self):
        for idx, raw in enumerate((b"", b"{bad", b"[{\"file\": \"a.dxbc\"")):
            with self.subTest(case=idx):
                path = self.write_bytes(f"malformed_{idx}.json", raw)
                with self.assertRaises(ValueError) as ctx:
                    shader_stems(path, "file", ".dxbc")
                self.assertIn(path, str(ctx.exception))

    def test_invalid_utf8_raises_value_error_with_path(self):
        path = self.write_bytes("bad_utf8.json", b"\xff\xfe\xfd")
        with self.assertRaises(ValueError) as ctx:
            shader_stems(path, "file", ".dxbc")
        self.assertIn(path, str(ctx.exception))

    def test_index_saved_with_a_utf8_bom_is_read(self):
        # Windows PowerShell 5.1 (Set-Content, Out-File) and old Notepad write JSON with a UTF-8 BOM.
        carve = self.write_bytes(
            "index.json",
            b"\xef\xbb\xbf" + json.dumps([{"file": "ps_water.dxbc"}]).encode("utf-8"),
        )
        unpack = self.write_bytes(
            "fssm_index.json",
            b"\xef\xbb\xbf" + json.dumps([{"name": "cs_bloom_0"}]).encode("utf-8"),
        )
        self.assertEqual(shader_stems(carve, "file", ".dxbc"), {"ps_water"})
        self.assertEqual(shader_stems(unpack, "name"), {"cs_bloom_0"})


class CallerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_game_unpacked_shader_stems_rejects_traversal_and_bad_json(self):
        dxbc = os.path.join(self.dir, "dxbc")
        os.makedirs(dxbc)
        with open(os.path.join(dxbc, "fssm_index.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": "."}], f)
        with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": "..dxbc"}], f)
        self.assertIsNone(make_game.unpacked_shader_stems(dxbc))

        with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": "...dxbc"}], f)
        with open(os.path.join(dxbc, "fssm_index.json"), "w", encoding="utf-8") as f:
            json.dump([{"name": ".."}], f)
        self.assertIsNone(make_game.unpacked_shader_stems(dxbc))

        with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8") as f:
            f.write("{truncated")
        self.assertIsNone(make_game.unpacked_shader_stems(dxbc))

    def test_fssm_unpack_carve_index_rejects_traversal_and_malformed_json(self):
        src = os.path.join(self.dir, "carved")
        os.makedirs(src)
        for name, content, expected_msg in (
            ("dot.json", json.dumps([{"file": "..dxbc"}]), "expected a shader basename"),
            ("dotdot.json", json.dumps([{"file": "...dxbc"}]), "expected a shader basename"),
            ("bad.json", "{not json", "invalid JSON"),
        ):
            with self.subTest(case=name):
                idx = os.path.join(self.dir, name)
                with open(idx, "w", encoding="utf-8") as f:
                    f.write(content)
                r = subprocess.run(
                    [sys.executable, "-I", FSSM_UNPACK, src, "--carve-index", idx],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(idx, r.stderr)
                self.assertIn(expected_msg, r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_shader_disasm_unpack_index_rejects_traversal_and_malformed_json(self):
        fssm_dir = os.path.join(self.dir, "fssm")
        os.makedirs(fssm_dir)
        for name, content, expected_msg in (
            ("dot.json", json.dumps([{"name": "."}]), "expected a shader basename"),
            ("dotdot.json", json.dumps([{"name": ".."}]), "expected a shader basename"),
            ("bad.json", "[{\"name\":", "invalid JSON"),
        ):
            with self.subTest(case=name):
                idx = os.path.join(self.dir, name)
                with open(idx, "w", encoding="utf-8") as f:
                    f.write(content)
                r = subprocess.run(
                    [sys.executable, "-I", SHADER_DISASM, fssm_dir, "--unpack-index", idx],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(idx, r.stderr)
                self.assertIn(expected_msg, r.stderr)
                self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
