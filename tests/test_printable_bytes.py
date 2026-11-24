"""The text test the index tools share: tools/story_index.py, tools/anim_index.py and tools/restofworld_index.py.

    python -I tests/test_printable_bytes.py

A stream is called text when more than 90% of its first 512 bytes are printable ASCII, a tab or a line ending. The
C0 control bytes other than those three (0x0B, 0x0C, 0x0E..0x1F) are not text. Every byte here is built by the
test; no game file is used.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_index  # noqa: E402
import restofworld_index  # noqa: E402
import story_index  # noqa: E402

TOOLS = (story_index, anim_index, restofworld_index)

# No tool has a signature for these: what is left is the text test and the raw fallback.
CONTROLS = bytes(range(0x0E, 0x20)) * 28                    # 504 bytes, none of them printable
TEXT = b"key\tvalue\r\nname\tsynthetic entry\r\n" * 16     # tabs and CRLF count as text


def signature(tool, data):
    return tool.classify(data, 0, len(data))["signature"]


class PrintableBytes(unittest.TestCase):
    def test_control_bytes_are_not_text(self):
        for tool in TOOLS:
            with self.subTest(tool=tool.__name__):
                self.assertFalse(tool._printable(CONTROLS))
                self.assertEqual(signature(tool, CONTROLS), "raw:0e0f1011")

    def test_each_remaining_control_byte_is_not_printable(self):
        for tool in TOOLS:
            for byte in (0x00, 0x08, 0x0B, 0x0C, 0x0E, 0x1B, 0x1F, 0x7F, 0x80, 0xFF):
                with self.subTest(tool=tool.__name__, byte=hex(byte)):
                    self.assertFalse(tool._printable(bytes([byte]) * 64))

    def test_ascii_with_tabs_and_line_endings_is_text(self):
        for tool in TOOLS:
            with self.subTest(tool=tool.__name__):
                self.assertTrue(tool._printable(TEXT))
                self.assertEqual(signature(tool, TEXT), "text")
                for byte in (0x09, 0x0A, 0x0D, 0x20, 0x7E):
                    self.assertTrue(tool._printable(bytes([byte]) * 64), hex(byte))

    def test_the_ninety_percent_bound_is_unchanged(self):
        for tool in TOOLS:
            with self.subTest(tool=tool.__name__):
                self.assertTrue(tool._printable(b"a" * 93 + b"\x01" * 7))      # 93% text
                self.assertFalse(tool._printable(b"a" * 90 + b"\x01" * 10))    # exactly 90% is not "more than"
                self.assertFalse(tool._printable(b"a" * 63))                   # under 64 bytes is never text


if __name__ == "__main__":
    unittest.main()
