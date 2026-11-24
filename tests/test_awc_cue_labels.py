import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "awc_cue_labels.py"


class LabelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.sheet = Path(temp.name) / "cue_regions.json"
        self.output = Path(temp.name) / "labels"

    def convert(self, regions):
        self.sheet.write_text(json.dumps({"sample_rate": 48000, "regions": regions}),
                              encoding="utf-8", newline="\n")
        return subprocess.run([sys.executable, "-I", TOOL, self.sheet, self.output],
                              capture_output=True, text=True)

    def test_channel_roundtrip(self):
        rows = [("00000001", 1, 48001), ("00000002", 4294967294, 4294967295),
                ("00000001", 7, 48001), ("00000001", 7, 48001)]
        result = self.convert([dict(file=channel + ".ogg", name_hash="1234ABCD",
                                    start_sample=start, end_sample=end) for channel, start, end in rows])
        self.assertEqual(result.returncode, 0, result.stderr)
        recovered = {}
        for track in self.output.iterdir():
            previous = 0
            for line in track.read_text(encoding="utf-8").splitlines():
                start, end, label = line.split("\t")
                cue, index = label.split(" #")
                self.assertEqual(cue, "1234ABCD")
                ordinal = int(index)
                self.assertGreater(ordinal, previous)
                self.assertNotIn(ordinal, recovered)
                previous = ordinal
                recovered[ordinal] = (track.stem, round(float(start) * 48000), round(float(end) * 48000))
        self.assertEqual(sorted(recovered.items()), list(enumerate(rows, 1)))

    def test_reject_invalid_channel_before_publishing(self):
        row = dict(file="00000001.ogg", name_hash="00000000", start_sample=0, end_sample=1)
        result = self.convert([row, dict(row, file="../00000001.ogg")])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
