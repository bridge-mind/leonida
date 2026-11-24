"""Synthetic InnerCoverage metadata census; no game files or GPU required.

Run: python -I tests/test_inner_coverage_metadata.py
"""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from feature_scan import inner_coverage_metadata, scan_dxbc


def signature(values):
    return struct.pack("<II", len(values), 8) + b"".join(
        struct.pack("<8I", 0, 0, 0, value, 3, i, 0xF0F, 0)
        for i, value in enumerate(values))


def container(parts):
    size = 32 + len(parts) * 4
    offsets, body = [], b""
    for tag, data in parts.items():
        offsets.append(size + len(body))
        body += tag.encode("ascii") + struct.pack("<I", len(data)) + data
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, size + len(body), len(parts))
            + b"".join(struct.pack("<I", off) for off in offsets) + body)


class Metadata(unittest.TestCase):
    def test_independent_observations(self):
        for mask, feature in ((0, "clear"), (0x400, "set"), (0x100000400, "set")):
            for values, state in (([], "absent"), ([1, 9], "absent"), ([1, 70], "present")):
                with self.subTest(mask=mask, values=values):
                    self.assertEqual(inner_coverage_metadata({"SFI0": struct.pack("<Q", mask),
                                                              "ISG1": signature(values)}),
                                     (feature, state))

    def test_missing_and_malformed_are_unknown(self):
        for data in (b"", bytes(7)):
            self.assertEqual(inner_coverage_metadata({"SFI0": data}), ("unknown", "unknown"))
        for sig in (bytes(7), struct.pack("<II", 1, 8), struct.pack("<II", 0, 0),
                    struct.pack("<II", 0, 9), struct.pack("<II", 0xFFFFFFFF, 8)):
            with self.subTest(sig=sig):
                self.assertEqual(inner_coverage_metadata({"SFI0": struct.pack("<Q", 0x400),
                                                          "ISG1": sig}), ("set", "unknown"))

    def test_census_and_cli_keep_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            shaders = Path(tmp) / "shaders"
            dxbc = shaders / "dxbc"
            dxbc.mkdir(parents=True)
            index = []
            inputs = [
                ("ps", {"SFI0": struct.pack("<Q", 0x400), "ISG1": signature([])}),
                ("ps", {"SFI0": struct.pack("<Q", 0x400)}),
                ("ps", {}),
                ("vs", {"SFI0": struct.pack("<Q", 0), "ISG1": signature([70])}),
            ]
            for i, (stage, parts) in enumerate(inputs):
                name = f"private_fixture_{i}"
                filename = name + ".dxbc"
                (dxbc / filename).write_bytes(container(parts))
                index.append({"name": name, "file": filename, "stage": stage})
            index.append({"name": "private_missing", "file": "missing.dxbc", "stage": "ps"})
            (dxbc / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
            expected = {"ps": {"sfi0_set__isg1_absent": 1, "sfi0_set__isg1_unknown": 1,
                               "sfi0_unknown__isg1_unknown": 1},
                        "vs": {"sfi0_clear__isg1_present": 1}}
            result = scan_dxbc(str(dxbc), index)
            self.assertEqual(result["inner_coverage_metadata"], expected)
            self.assertEqual(result["unreadable"], 1)
            output = Path(tmp) / "counts.json"
            run = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "feature_scan.py"),
                                  str(shaders), "--out", str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("ISG1 absence does not disprove use", run.stdout)
            encoded = output.read_text(encoding="utf-8")
            self.assertNotIn("private_", run.stdout + encoded)
            self.assertEqual(json.loads(encoded)["dxbc"]["inner_coverage_metadata"], expected)


if __name__ == "__main__":
    unittest.main()
