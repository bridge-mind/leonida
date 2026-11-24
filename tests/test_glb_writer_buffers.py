"""Test external buffers in synthetic binary glTF containers."""
from __future__ import annotations

import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_writer


def _external_buffer_glb(view_length: int) -> bytes:
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "sample.bin", "byteLength": 4}],
        "bufferViews": [{"buffer": 0, "byteLength": view_length}],
    }
    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    return (
        struct.pack("<4sII", b"glTF", 2, 20 + len(json_bytes))
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
    )


class TestReadGlbSummaryBuffers(unittest.TestCase):
    def test_external_buffer_view_without_bin_chunk(self):
        summary = glb_writer.read_glb_summary(_external_buffer_glb(view_length=4))
        self.assertEqual(summary["bin_bytes"], 0)
        self.assertEqual(summary["json"]["bufferViews"][0]["byteLength"], 4)

    def test_external_buffer_view_exceeding_declared_length_rejected(self):
        with self.assertRaisesRegex(glb_writer.GlbError, r"bufferViews\[0\] exceeds buffer 0 \(0\+5 > 4\)"):
            glb_writer.read_glb_summary(_external_buffer_glb(view_length=5))


if __name__ == "__main__":
    unittest.main()
