"""feature_scan.scan_disasm gives the same report whatever order the file system lists the listings in.

    python -I tests/test_feature_scan_order.py

The report is written as JSON with its keys in the order they were first seen, so a scan that follows
`os.listdir` writes a different file on NTFS (name order) and on ext4 (hash order) for the same folder. The two
listings below are hand-written (no game data) and the listing order is forced both ways.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import feature_scan as F  # noqa: E402

LISTINGS = {
    "a_ps": "\tv_mfma_f32_16x16x4f32 v[0:3], v1, v2, v[0:3]\n\ts_endpgm\n",
    "b_cs": "\tv_wmma_f16_16x16x16_f16 v[0:3], v1, v2, v[0:3]\n\tds_gws_init v1 gds\n\ts_endpgm\n",
}
STAGE_OF = {"a_ps": "ps", "b_cs": "cs"}


def report(folder, order):
    real = os.listdir
    with mock.patch.object(os, "listdir", lambda path: order(real(path))):
        return json.dumps(F.scan_disasm(folder, STAGE_OF), indent=1)


class ScanOrderTest(unittest.TestCase):
    def test_report_does_not_follow_the_listing_order(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in LISTINGS.items():
                with open(os.path.join(folder, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            forward = report(folder, sorted)
            backward = report(folder, lambda names: sorted(names, reverse=True))
            self.assertEqual(forward, backward)
            parsed = json.loads(forward)
            self.assertEqual(parsed["files"], 2)
            self.assertEqual(list(parsed["wave_mma_ops"]), sorted(parsed["wave_mma_ops"]))
            self.assertEqual(parsed["wave_mma_shaders"], {"cs": 1, "ps": 1})


if __name__ == "__main__":
    unittest.main()
