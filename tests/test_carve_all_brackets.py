"""carve_all.ps1 with a game folder whose name holds [ ] (PowerShell wildcard characters)."""
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
class CarveAllBracketPathTest(unittest.TestCase):
    def test_bracketed_game_folder_finds_its_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "GTA VI [1.0.2.1]"), Path(tmp, "carves [1]")
            game.mkdir()
            (game / "common.rpf").write_bytes(b"x")
            (out / "common").mkdir(parents=True)
            (out / "common" / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = (out / "coverage.json").read_text(encoding="utf-8-sig").strip()
            rows = json.loads(text) if text else []
            rows = rows if isinstance(rows, list) else [rows]
            self.assertEqual([r["archive"] for r in rows], ["common.rpf"])


if __name__ == "__main__":
    unittest.main()
