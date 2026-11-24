"""tools/find_hash_lookup.py ranks the function a model hash is passed to (standard library only, no game data).

    python -I tests/test_find_hash_lookup.py

The images are synthetic PE files: one executable section holding `mov ecx, <hash>` followed by calls, with a
"lookup" called from many places and a decoy called once.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "find_hash_lookup.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import find_hash_lookup as fhl  # noqa: E402

TEXT_RVA, TEXT_RAW = 0x1000, 0x400


def pe(code):
    """A minimal PE32+ file whose only section, .text (executable), holds `code`."""
    header = bytearray(TEXT_RAW)
    header[0:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 0x40)
    header[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", header, 0x44, 0x8664, 1, 0, 0, 0, 0xF0, 0x22)
    first = 0x44 + 20 + 0xF0
    struct.pack_into("<8sIIIIIIHHI", header, first, b".text", len(code), TEXT_RVA, len(code), TEXT_RAW,
                     0, 0, 0, 0, 0x60000020)
    return bytes(header) + code


def call(at, target):
    """E8 rel32 placed at offset `at` of .text, calling .text offset `target`."""
    return b"\xE8" + struct.pack("<i", target - (at + 5))


def image():
    lucia, mandy = fhl.joaat("player_story000_00"), fhl.joaat("cs_mandy")
    code = bytearray(b"\xCC" * 0x400)
    lookup, decoy = 0x300, 0x380
    code[lookup] = code[decoy] = 0xC3
    # mov ecx, lucia ; call lookup   (at 0x00)
    code[0x00:0x05] = b"\xB9" + struct.pack("<I", lucia)
    code[0x05:0x0A] = call(0x05, lookup)
    # mov ecx, mandy ; nop ; call lookup   (at 0x20)
    code[0x20:0x25] = b"\xB9" + struct.pack("<I", mandy)
    code[0x25] = 0x90
    code[0x26:0x2B] = call(0x26, lookup)
    # mov ecx, lucia ; call decoy   (at 0x40): led by one hash, called once
    code[0x40:0x45] = b"\xB9" + struct.pack("<I", lucia)
    code[0x45:0x4A] = call(0x45, decoy)
    for k in range(20):                          # other callers of the lookup
        at = 0x100 + 8 * k
        code[at:at + 5] = call(at, lookup)
    return pe(bytes(code)), TEXT_RVA + lookup, TEXT_RVA + decoy


class FindHashLookupTest(unittest.TestCase):
    def test_joaat_matches_the_measured_model_ids(self):
        self.assertEqual(fhl.joaat("player_story000_00"), 0xE472B687)
        self.assertEqual(fhl.joaat("CS_Mandy"), 0xD7C0CDFF)
        self.assertEqual(fhl.parse_hash("0xD7C0CDFF"), 0xD7C0CDFF)
        with self.assertRaises(ValueError):
            fhl.parse_hash("0x123456789")

    def test_the_lookup_ranks_first(self):
        data, lookup, decoy = image()
        report = fhl.rank(data, [("player_story000_00", fhl.joaat("player_story000_00")),
                                 ("cs_mandy", fhl.joaat("cs_mandy"))])
        top = report["candidates"][0]
        self.assertEqual(int(top["rva"], 16), lookup)
        self.assertEqual(top["led_by"], ["cs_mandy", "player_story000_00"])
        self.assertEqual(top["direct_callers"], 22)
        self.assertEqual(int(report["candidates"][1]["rva"], 16), decoy)
        self.assertEqual({h["name"]: h["occurrences"] for h in report["hashes"]},
                         {"player_story000_00": 2, "cs_mandy": 1})

    def test_an_encrypted_file_is_refused_clearly(self):
        with self.assertRaisesRegex(ValueError, "encrypted"):
            fhl.rank(b"\x6f\x1b\xbc\xae" * 64, [("x", 1)])

    def test_cli_writes_the_report_and_fails_cleanly(self):
        data, lookup, _ = image()
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "image.exe")
            with open(exe, "wb") as f:
                f.write(data)
            out = os.path.join(tmp, "report.json")
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"0x{lookup:X}", r.stdout)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(int(json.load(f)["candidates"][0]["rva"], 16), lookup)
            bad = os.path.join(tmp, "packed.exe")
            with open(bad, "wb") as f:
                f.write(b"\x00" * 64)
            r = subprocess.run([sys.executable, "-I", TOOL, bad], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("ERROR", r.stderr)

    def test_json_writes_nothing_over_the_image_and_never_crashes(self):
        """--json naming the scanned image must be refused before it truncates the image (the game executable),
        and an unwritable path is one error line, not a traceback."""
        data, _, _ = image()
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "image.exe")
            with open(exe, "wb") as f:
                f.write(data)
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(exe, "rb") as f:
                self.assertEqual(f.read(), data, "the scanned image must be untouched")
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", tmp], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is a folder", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            missing = os.path.join(tmp, "no-such-folder", "report.json")
            r = subprocess.run([sys.executable, "-I", TOOL, exe, "--json", missing], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("ERROR: cannot write", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
