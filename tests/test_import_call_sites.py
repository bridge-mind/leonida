"""Unit tests for tools/import_call_sites.py: thunks and the direct calls to them, on an image built here byte by byte.

The image has two executable-code layouts the tool must tell apart: calls (E8) and tail jumps (E9) to the thunk of an
import, and an import nothing calls. It needs no game files and no third-party packages.

    python -I tests/test_import_call_sites.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import import_call_sites  # noqa: E402
import pe_inspect  # noqa: E402

IMAGE_BASE = 0x140000000
TEXT_RVA = 0x1000
IDATA_RVA = 0x2000

# .text layout (RVAs): three thunks, then the callers.
THUNK_FOO = TEXT_RVA + 0x00
THUNK_BAR = TEXT_RVA + 0x10
THUNK_ORD = TEXT_RVA + 0x20
# .idata layout: descriptor table, lookup table, address table, hint/name entries, module name.
IAT_FOO = IDATA_RVA + 0x60
IAT_BAR = IAT_FOO + 8
IAT_ORD = IAT_FOO + 16
OTHER_IAT_RVA = IDATA_RVA + 0xE0


def rel32(from_end, target):
    return struct.pack("<i", target - from_end)


def build_image(callers):
    """A PE32+ with one import descriptor (xmem.dll: XMemFoo, XMemBar, #7) and the given `.text` callers."""
    buf = bytearray(0x3000)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, 2, 0, 0, 0, 0xF0, 0x22)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<Q", buf, opt + 24, IMAGE_BASE)
    struct.pack_into("<I", buf, opt + 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", buf, opt + 36, 0x1000)  # FileAlignment
    struct.pack_into("<I", buf, opt + 56, 0x3000)  # SizeOfImage
    struct.pack_into("<I", buf, opt + 60, 0x1000)  # SizeOfHeaders
    struct.pack_into("<I", buf, opt + 108, 16)  # NumberOfRvaAndSizes
    struct.pack_into("<II", buf, opt + 112 + 8, IDATA_RVA, 40)  # import directory
    table = opt + 0xF0
    for i, (name, rva, chars) in enumerate(((b".text", TEXT_RVA, 0x60000020), (b".idata", IDATA_RVA, 0xC0000040))):
        at = table + i * 40
        buf[at:at + len(name)] = name
        struct.pack_into("<IIII", buf, at + 8, 0x1000, rva, 0x1000, rva)
        struct.pack_into("<I", buf, at + 36, chars)

    # .idata: one descriptor, its terminator, the lookup and address tables (same entries), names, the module name.
    lookup = IDATA_RVA + 0x40
    names = IDATA_RVA + 0xA0
    struct.pack_into("<IIIII", buf, IDATA_RVA, lookup, 0, 0, IDATA_RVA + 0xD0, IAT_FOO)
    entries = (names, names + 0x10, (1 << 63) | 7)
    for i, value in enumerate(entries):
        struct.pack_into("<Q", buf, lookup + i * 8, value)
        struct.pack_into("<Q", buf, IAT_FOO + i * 8, value)
    buf[names + 2:names + 2 + 8] = b"XMemFoo\0"
    buf[names + 0x10 + 2:names + 0x10 + 2 + 8] = b"XMemBar\0"
    buf[IDATA_RVA + 0xD0:IDATA_RVA + 0xD0 + 9] = b"xmem.dll\0"

    # .text: the thunks are `jmp qword [rip+disp32]` to the address-table slots.
    for thunk, slot in ((THUNK_FOO, IAT_FOO), (THUNK_BAR, IAT_BAR), (THUNK_ORD, IAT_ORD)):
        buf[thunk:thunk + 2] = b"\xff\x25"
        buf[thunk + 2:thunk + 6] = rel32(thunk + 6, slot)
    for at, opcode, target in callers:
        buf[at] = opcode
        buf[at + 1:at + 5] = rel32(at + 5, target)
    return bytes(buf)


class ImportCallSitesTest(unittest.TestCase):
    def setUp(self):
        self.callers = [
            (TEXT_RVA + 0x100, 0xE8, THUNK_FOO),
            (TEXT_RVA + 0x110, 0xE8, THUNK_FOO),
            (TEXT_RVA + 0x120, 0xE9, THUNK_FOO),  # a tail jump
            (TEXT_RVA + 0x130, 0xE8, THUNK_ORD),
            (TEXT_RVA + 0x140, 0xE8, TEXT_RVA + 0x200),  # a call to something that is not a thunk
        ]
        self.data = build_image(self.callers)

    def test_counts_calls_and_jumps_per_import(self):
        out = import_call_sites.call_sites(self.data)["imports"]
        self.assertEqual(sorted(out), ["xmem.dll!#7", "xmem.dll!XMemBar", "xmem.dll!XMemFoo"])
        foo = out["xmem.dll!XMemFoo"]
        self.assertEqual(foo["thunks"], [THUNK_FOO])
        self.assertEqual(foo["call"], [TEXT_RVA + 0x100, TEXT_RVA + 0x110])
        self.assertEqual(foo["jmp"], [TEXT_RVA + 0x120])

    def test_an_import_nothing_calls_has_a_thunk_and_no_sites(self):
        bar = import_call_sites.call_sites(self.data)["imports"]["xmem.dll!XMemBar"]
        self.assertEqual((bar["thunks"], bar["call"], bar["jmp"]), ([THUNK_BAR], [], []))

    def test_an_ordinal_import_is_named_by_its_number(self):
        out = import_call_sites.call_sites(self.data, wanted={"xmem.dll!#7"})["imports"]
        self.assertEqual(list(out), ["xmem.dll!#7"])
        self.assertEqual(out["xmem.dll!#7"]["call"], [TEXT_RVA + 0x130])

    def test_wanted_names_and_dlls_select_imports(self):
        out = import_call_sites.call_sites(self.data, wanted={"XMEM.DLL!xmemfoo"})["imports"]
        self.assertEqual(list(out), ["xmem.dll!XMemFoo"])
        self.assertEqual(len(import_call_sites.call_sites(self.data, dlls={"XMem.dll"})["imports"]), 3)
        self.assertEqual(import_call_sites.call_sites(self.data, dlls={"other.dll"})["imports"], {})

    def test_a_call_is_not_counted_for_a_different_thunk(self):
        data = build_image([(TEXT_RVA + 0x100, 0xE8, THUNK_BAR)])
        out = import_call_sites.call_sites(data)["imports"]
        self.assertEqual(out["xmem.dll!XMemFoo"]["call"], [])
        self.assertEqual(out["xmem.dll!XMemBar"]["call"], [TEXT_RVA + 0x100])

    def test_a_32_bit_image_is_refused(self):
        data = bytearray(self.data)
        struct.pack_into("<H", data, 0x58, 0x10B)
        with self.assertRaises(pe_inspect.PeFormatError):
            import_call_sites.call_sites(bytes(data))

    def test_cli_prints_counts_and_json_with_hex_rvas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "image.exe")
            with open(path, "wb") as f:
                f.write(self.data)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(import_call_sites.main([path, "--import", "xmem.dll!XMemFoo"]), 0)
            self.assertIn("xmem.dll!XMemFoo", out.getvalue())
            self.assertIn("1 import(s), 3 call site(s)", out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(import_call_sites.main([path, "--dll", "xmem.dll", "--json"]), 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["image_base"], hex(IMAGE_BASE))
            self.assertEqual(report["imports"]["xmem.dll!XMemFoo"]["call"], [hex(TEXT_RVA + 0x100), hex(TEXT_RVA + 0x110)])

    def test_cli_reports_an_unreadable_or_non_pe_file_with_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not_a_pe.bin")
            with open(path, "wb") as f:
                f.write(b"not a pe")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(import_call_sites.main([path]), 2)
            self.assertTrue(err.getvalue().strip())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(import_call_sites.main([os.path.join(tmp, "missing.exe")]), 2)


if __name__ == "__main__":
    unittest.main()
