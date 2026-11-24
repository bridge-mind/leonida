import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "carve_all.ps1"
INDEX = {"summary": {"files": 1, "bink2_files": 0, "span": 10, "compressed_bytes": 5, "unread_bytes": 5,
                     "inflated_bytes": 5}, "files": []}


@unittest.skipUnless(os.name == "nt", "PowerShell research script")
class CarveAllGamePathTest(unittest.TestCase):
    def test_invalid_cached_summary_does_not_replace_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            game.mkdir()
            (game / "sample.rpf").write_bytes(b"x")
            cached = out / "sample"
            cached.mkdir(parents=True)
            index = cached / "index.json"
            coverage = out / "coverage.json"
            summary = INDEX["summary"]
            invalid_indexes = {
                "invalid JSON": "{",
                "missing summary": "{}",
                "negative span": json.dumps({"summary": {**summary, "span": -1}}),
                "bytes in empty span": json.dumps({"summary": {**summary, "span": 0}}),
                "non-numeric span": json.dumps({"summary": {**summary, "span": "ten"}}),
                "missing count": json.dumps({"summary": {k: v for k, v in summary.items() if k != "files"}}),
            }
            for case, contents in invalid_indexes.items():
                with self.subTest(case=case):
                    index.write_text(contents, encoding="utf-8")
                    coverage.write_bytes(b"[]\n")
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("Invalid carve index", result.stderr)
                    self.assertIn("index.json", result.stderr)
                    self.assertEqual(coverage.read_bytes(), b"[]\n")

    def test_empty_cached_data_area_reports_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            game.mkdir()
            (game / "empty.rpf").write_bytes(b"x")
            cached = out / "empty"
            cached.mkdir(parents=True)
            summary = {key: 0 for key in INDEX["summary"]}
            (cached / "index.json").write_text(json.dumps({"summary": summary, "files": []}), encoding="utf-8")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            row = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(row, {"archive": "empty.rpf", "files": 0, "bink2": 0, "span": 0,
                                   "readable": 0, "unread": 0, "inflated": 0})

    def test_inconsistent_cached_summary_does_not_replace_coverage(self):
        summary = INDEX["summary"]
        cases = {
            "compressed past span": ({"compressed_bytes": 11, "unread_bytes": 0}, "compressed_bytes"),
            "unread past span": ({"compressed_bytes": 0, "unread_bytes": 11}, "unread_bytes"),
            "combined bytes past span": ({"compressed_bytes": 7, "unread_bytes": 4}, "unread_bytes"),
            "bink past file count": ({"bink2_files": 2}, "bink2_files"),
            "int64 byte sum past span by one": (
                {"span": 2**63 - 1, "compressed_bytes": 2**62, "unread_bytes": 2**62}, "unread_bytes"),
            "int64 bink past file count by one": (
                {"files": 2**62, "bink2_files": 2**62 + 1}, "bink2_files"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            game.mkdir()
            (game / "sample.rpf").write_bytes(b"x")
            cache = out / "sample"
            cache.mkdir(parents=True)
            index, coverage = cache / "index.json", out / "coverage.json"
            previous = b"previous complete coverage\x00\xff"
            for label, (changes, field) in cases.items():
                with self.subTest(case=label):
                    index.write_text(json.dumps({"summary": {**summary, **changes}, "files": []}),
                                     encoding="utf-8", newline="\n")
                    coverage.write_bytes(previous)
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("Invalid carve index", result.stderr)
                    self.assertIn(field, result.stderr)
                    self.assertEqual(coverage.read_bytes(), previous)

    def test_consistent_summary_boundaries_keep_the_expected_counts(self):
        cases = {
            "exact span": {**INDEX["summary"], "bink2_files": 1},
            "int64 exact span": {**INDEX["summary"], "span": 2**63 - 1,
                                 "compressed_bytes": 2**62, "unread_bytes": 2**62 - 1},
            "only padding": {"files": 0, "bink2_files": 0, "span": 10, "compressed_bytes": 0,
                             "unread_bytes": 0, "inflated_bytes": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            game.mkdir()
            (game / "sample.rpf").write_bytes(b"x")
            cache = out / "sample"
            cache.mkdir(parents=True)
            for label, summary in cases.items():
                with self.subTest(case=label):
                    (cache / "index.json").write_text(json.dumps({"summary": summary, "files": []}),
                                                      encoding="utf-8", newline="\n")
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(result.stderr, "")
                    row = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
                    self.assertEqual((row["files"], row["bink2"], row["span"], row["inflated"]),
                                     tuple(summary[key] for key in ("files", "bink2_files", "span", "inflated_bytes")))
                    self.assertEqual(row["readable"], round(summary["compressed_bytes"] / summary["span"], 4))
                    self.assertEqual(row["unread"], round(summary["unread_bytes"] / summary["span"], 4))

    def test_failed_carver_stops_without_replacing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game, out, tools = root / "game", root / "out", root / "tools"
            for folder in (game, out, tools):
                folder.mkdir()
            (game / "first.rpf").write_bytes(b"x")
            (game / "second.rpf").write_bytes(b"xx")
            (tools / "python.cmd").write_bytes(b"@exit /b 7\r\n")
            coverage = out / "coverage.json"
            coverage.write_bytes(b"[]\n")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)],
                env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Carving first.rpf failed (exit 7)", result.stderr)
            self.assertNotIn("second.rpf", result.stdout + result.stderr)
            self.assertEqual(coverage.read_bytes(), b"[]\n")

    def test_relative_and_trailing_separator_game_paths_find_existing_carves(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            (game / "pack" / "p1").mkdir(parents=True)
            (game / "common.rpf").write_bytes(b"x")
            (game / "pack" / "p1" / "p1.rpf").write_bytes(b"xy")
            # Both archives are already carved, so no carve should start.
            for name in ("common", "pack_p1_p1"):
                (out / name).mkdir(parents=True)
                (out / name / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            args = ["game", ".\\game", str(game) + "\\"]
            short_game = ctypes.create_unicode_buffer(32768)
            short_path_length = ctypes.windll.kernel32.GetShortPathNameW(str(game), short_game, len(short_game))
            if short_path_length and short_game.value.casefold() != str(game).casefold():
                args.append(short_game.value + "\\")
            for arg in args:
                with self.subTest(game=arg):
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", arg, "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("carving", result.stdout)
                    rows = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
                    self.assertEqual([r["archive"] for r in rows], ["common.rpf", "pack\\p1\\p1.rpf"])
                    self.assertEqual(sorted(p.name for p in out.iterdir()),
                                     ["common", "coverage.json", "pack_p1_p1"])

    def test_game_path_with_83_short_name_component_finds_existing_carves(self):
        # Resolve-Path keeps an 8.3 component (e.g. RUNNER~1) but Get-ChildItem emits canonical
        # FullNames. A trailing separator plus such a component used to cut the rel path at the
        # wrong offset, re-carving archives that were already present.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp, "LongDirectoryNameForCarveAllShortPath")
            game, out = base / "game", base / "out"
            (game / "pack" / "p1").mkdir(parents=True)
            (game / "common.rpf").write_bytes(b"x")
            (game / "pack" / "p1" / "p1.rpf").write_bytes(b"xy")
            for name in ("common", "pack_p1_p1"):
                (out / name).mkdir(parents=True)
                (out / name / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            buf = ctypes.create_unicode_buffer(1024)
            get_short = ctypes.windll.kernel32.GetShortPathNameW
            get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
            get_short.restype = ctypes.c_uint
            if not get_short(str(game), buf, len(buf)):
                self.skipTest("GetShortPathNameW failed")
            short_game = buf.value + "\\"
            if short_game == str(game) + "\\":
                self.skipTest("8.3 short names are unavailable on this volume")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", short_game, "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("carving", result.stdout)
            rows = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
            self.assertEqual([r["archive"] for r in rows], ["common.rpf", "pack\\p1\\p1.rpf"])

    def test_colliding_archive_names_do_not_share_a_cached_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            (game / "pack").mkdir(parents=True)
            (game / "pack" / "scene.rpf").write_bytes(b"x")
            (game / "PACK_scene.rpf").write_bytes(b"xx")
            cached = out / "pack_scene"
            cached.mkdir(parents=True)
            (cached / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            coverage = out / "coverage.json"
            coverage.write_bytes(b"[]\n")

            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("pack\\scene.rpf", result.stderr)
            self.assertIn("PACK_scene.rpf", result.stderr)
            self.assertNotIn("carving", result.stdout)
            self.assertEqual(coverage.read_bytes(), b"[]\n")


if __name__ == "__main__":
    unittest.main()
