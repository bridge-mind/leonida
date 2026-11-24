"""tools/d3d12_iid_scan.py on a hand-written header and hand-built PE images (no game files, no SDK needed)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import d3d12_iid_scan as scan  # noqa: E402

QUEUE = uuid.UUID("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
RESOURCE = uuid.UUID("696442be-a72e-4059-bc79-5b5c98040fad")

HEADER = f"""
    MIDL_INTERFACE("{QUEUE}")
    ID3D12CommandQueue : public ID3D12Pageable
    {{ }};
    MIDL_INTERFACE("{RESOURCE}")
    ID3D12Resource : public ID3D12Pageable
    {{ }};
    MIDL_INTERFACE("00000000-0000-0000-c000-000000000046")
    IUnknown
    {{ }};
"""


def pe32plus(imports, payload=b""):
    """A minimal PE32+ image: one section holding `payload` and an import directory.

    imports: {dll_name: [function name or ordinal int, ...]}.
    """
    sect_rva, sect_raw = 0x1000, 0x200
    body = bytearray(payload)

    def put(blob, align=8):
        while len(body) % align:
            body.append(0)
        at = len(body)
        body.extend(blob)
        return sect_rva + at

    descriptors = []
    entries = imports.items() if isinstance(imports, dict) else imports
    for dll, funcs in entries:
        thunks = []
        for f in funcs:
            if isinstance(f, int):
                thunks.append((1 << 63) | f)
            else:
                thunks.append(put(struct.pack("<H", 0) + f.encode() + b"\0", 2))
        lookup = put(b"".join(struct.pack("<Q", t) for t in thunks) + b"\0" * 8)
        iat = put(b"".join(struct.pack("<Q", t) for t in thunks) + b"\0" * 8)
        name = put(dll.encode() + b"\0", 2)
        descriptors.append((lookup, name, iat))
    imp = put(b"".join(struct.pack("<IIIII", lk, 0, 0, nm, ia) for lk, nm, ia in descriptors) + b"\0" * 20, 4)

    pe = 0x80
    opt_size = 240
    hdr = bytearray(sect_raw)
    hdr[0:2] = b"MZ"
    struct.pack_into("<I", hdr, 0x3C, pe)
    hdr[pe:pe + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", hdr, pe + 4, 0x8664, 1, 0, 0, 0, opt_size, 0x2022)
    opt = pe + 24
    struct.pack_into("<H", hdr, opt, 0x20B)
    struct.pack_into("<I", hdr, opt + 108, 16)          # NumberOfRvaAndSizes
    struct.pack_into("<II", hdr, opt + 112 + 8, imp, 20 * (len(descriptors) + 1))
    sec = opt + opt_size
    hdr[sec:sec + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", hdr, sec + 8, len(body), sect_rva, len(body), sect_raw)
    return bytes(hdr) + bytes(body)


class IidScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.header = os.path.join(self.tmp.name, "d3d12.h")
        with open(self.header, "w", encoding="utf-8") as f:
            f.write(HEADER)

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_header_reads_only_d3d12_interfaces(self):
        iids = scan.read_iids(self.header)
        self.assertEqual(iids, {"ID3D12CommandQueue": QUEUE, "ID3D12Resource": RESOURCE})

    def test_finds_little_endian_guid_bytes_only(self):
        iids = scan.read_iids(self.header)
        # The big-endian (string-order) bytes of the queue IID must not count: memory holds bytes_le.
        path = self.image("a.dll", pe32plus({}, b"pad" + RESOURCE.bytes_le + QUEUE.bytes + b"pad"))
        r = scan.scan(path, iids)
        self.assertEqual(r["found"], ["ID3D12Resource"])
        self.assertEqual(r["focus"], {"ID3D12CommandQueue": False})
        self.assertIsNone(r["d3d12_xs_imports"])
        self.assertEqual(r["possible_iid_imports"], [])

    def test_lists_d3d12_xs_imports_and_flags_guid_ordinals(self):
        iids = scan.read_iids(self.header)
        path = self.image("b.dll", pe32plus({"KERNEL32.dll": ["Sleep"],
                                              "d3d12_xs.dll": ["D3D12SerializeVersionedRootSignature", 80,
                                                               "IID_ID3D12Fence", 7]}))
        r = scan.scan(path, iids)
        self.assertEqual(r["d3d12_xs_imports"], ["D3D12SerializeVersionedRootSignature", "#80", "IID_ID3D12Fence", "#7"])
        # Ordinal 80 is in the GUID export range (23-115); ordinal 7 is a function export.
        self.assertEqual(r["possible_iid_imports"], ["#80", "IID_ID3D12Fence"])

    def test_function_only_import_is_not_flagged(self):
        iids = scan.read_iids(self.header)
        path = self.image("c.dll", pe32plus({"d3d12_xs.dll": ["D3D12CreateRootSignatureDeserializer"]}))
        r = scan.scan(path, iids)
        self.assertEqual(r["d3d12_xs_imports"], ["D3D12CreateRootSignatureDeserializer"])
        self.assertEqual(r["possible_iid_imports"], [])

    def test_collects_imports_across_multiple_d3d12_xs_descriptors(self):
        iids = scan.read_iids(self.header)
        path = self.image("multi.dll", pe32plus([
            ("d3d12_xs.dll", ["D3D12CreateRootSignatureDeserializer"]),
            ("KERNEL32.dll", ["Sleep"]),
            ("D3D12_XS.DLL", [80, "IID_ID3D12Fence"]),
        ]))
        r = scan.scan(path, iids)
        self.assertEqual(
            r["d3d12_xs_imports"],
            ["D3D12CreateRootSignatureDeserializer", "#80", "IID_ID3D12Fence"],
        )
        self.assertEqual(r["possible_iid_imports"], ["#80", "IID_ID3D12Fence"])

    def test_rva_in_virtual_tail_past_raw_data_does_not_read_next_section(self):
        # Section 1 (.rdata) has VirtualSize > SizeOfRawData; an import descriptor's Name RVA
        # falls in the unbacked virtual tail (va + rsize), which must not alias section 2's raw bytes.
        sect1_rva, sect1_raw, sect1_rsize, sect1_vsize = 0x1000, 0x200, 0x200, 0x1000
        sect2_rva, sect2_raw = 0x2000, 0x400
        # Place one import descriptor at sect1_rva pointing Name RVA into sect1's virtual tail (0x1200)
        # and Lookup/IAT at 0x1040 (single ordinal #80 + null terminator).
        sect1 = bytearray(sect1_rsize)
        struct.pack_into("<IIIII", sect1, 0, 0x1040, 0, 0, sect1_rva + sect1_rsize, 0x1040)
        struct.pack_into("<QQ", sect1, 0x40, (1 << 63) | 80, 0)
        # Section 2 begins at file offset sect1_raw + sect1_rsize (0x400) with "d3d12_xs.dll\0".
        sect2 = bytearray(b"d3d12_xs.dll\0")

        pe, opt_size = 0x80, 240
        hdr = bytearray(sect1_raw)
        hdr[0:2] = b"MZ"
        struct.pack_into("<I", hdr, 0x3C, pe)
        hdr[pe:pe + 4] = b"PE\0\0"
        struct.pack_into("<HHIIIHH", hdr, pe + 4, 0x8664, 2, 0, 0, 0, opt_size, 0x2022)
        opt = pe + 24
        struct.pack_into("<H", hdr, opt, 0x20B)
        struct.pack_into("<I", hdr, opt + 108, 16)
        struct.pack_into("<II", hdr, opt + 112 + 8, sect1_rva, 40)
        sec1 = opt + opt_size
        hdr[sec1:sec1 + 8] = b".rdata\0\0"
        struct.pack_into("<IIII", hdr, sec1 + 8, sect1_vsize, sect1_rva, sect1_rsize, sect1_raw)
        sec2 = sec1 + 40
        hdr[sec2:sec2 + 8] = b".data\0\0\0"
        struct.pack_into("<IIII", hdr, sec2 + 8, len(sect2), sect2_rva, len(sect2), sect2_raw)

        self.assertIsNone(scan.d3d12_xs_imports(bytes(hdr) + bytes(sect1) + bytes(sect2)))

    def test_zero_virtual_size_section_falls_back_to_raw_size(self):
        pe = bytearray(pe32plus({"d3d12_xs.dll": ["D3D12CreateRootSignatureDeserializer", 80]}))
        # Section 0 header starts at pe(0x80) + 24 + opt_size(240) = 0x188; VirtualSize is at +8 (0x190).
        struct.pack_into("<I", pe, 0x190, 0)
        self.assertEqual(
            scan.d3d12_xs_imports(bytes(pe)),
            ["D3D12CreateRootSignatureDeserializer", "#80"],
        )

    def test_truncated_file_rva_past_eof_keeps_prior_imports(self):
        # Two d3d12_xs.dll descriptors in one section whose header declares SizeOfRawData = VirtualSize = 0x400,
        # but the file is truncated at sect_raw + 0x100. Descriptor 1's thunks live before EOF; descriptor 2's
        # lookup RVA maps to sect_raw + 0x100 (exactly len(data), inside the declared section bounds), so offset()
        # must return None rather than letting struct.unpack_from raise and discard descriptor 1's imports.
        sect_rva, sect_raw, declared_size = 0x1000, 0x200, 0x400
        body = bytearray(0x100)
        # Descriptor table at +0x00 (2 descriptors + null terminator = 60 bytes):
        #   name "d3d12_xs.dll\0" at +0x40, descriptor 1 thunks at +0x60 (ordinal #80 + 0),
        #   descriptor 2 lookup RVA at sect_rva + 0x100 (exact EOF boundary, inside declared_size).
        body[0x40:0x40 + 13] = b"d3d12_xs.dll\0"
        struct.pack_into("<QQ", body, 0x60, (1 << 63) | 80, 0)
        struct.pack_into("<IIIII", body, 0, sect_rva + 0x60, 0, 0, sect_rva + 0x40, sect_rva + 0x60)
        struct.pack_into("<IIIII", body, 20, sect_rva + 0x100, 0, 0, sect_rva + 0x40, sect_rva + 0x100)

        pe, opt_size = 0x80, 240
        hdr = bytearray(sect_raw)
        hdr[0:2] = b"MZ"
        struct.pack_into("<I", hdr, 0x3C, pe)
        hdr[pe:pe + 4] = b"PE\0\0"
        struct.pack_into("<HHIIIHH", hdr, pe + 4, 0x8664, 1, 0, 0, 0, opt_size, 0x2022)
        opt = pe + 24
        struct.pack_into("<H", hdr, opt, 0x20B)
        struct.pack_into("<I", hdr, opt + 108, 16)
        struct.pack_into("<II", hdr, opt + 112 + 8, sect_rva, 60)
        sec = opt + opt_size
        hdr[sec:sec + 8] = b".rdata\0\0"
        struct.pack_into("<IIII", hdr, sec + 8, declared_size, sect_rva, declared_size, sect_raw)

        self.assertEqual(scan.d3d12_xs_imports(bytes(hdr) + bytes(body)), ["#80"])

    def test_not_a_pe_still_scans_bytes(self):
        iids = scan.read_iids(self.header)
        r = scan.scan(self.image("blob.bin", QUEUE.bytes_le), iids)
        self.assertEqual(r["found"], ["ID3D12CommandQueue"])
        self.assertIsNone(r["d3d12_xs_imports"])

    def test_cli_prints_names_and_writes_json(self):
        dll = self.image("d.dll", pe32plus({"d3d12_xs.dll": [30]}, RESOURCE.bytes_le))
        out = os.path.join(self.tmp.name, "out.json")
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), dll,
                            "--header", self.header, "--json", out], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("found:  ID3D12Resource", p.stdout)
        self.assertIn("absent from the queue/allocator/list/fence set: ID3D12CommandQueue", p.stdout)
        self.assertIn("WARNING", p.stdout)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["iids"], 2)
        self.assertEqual(data["images"][0]["possible_iid_imports"], ["#30"])

    def test_cli_unreadable_image_fails_cleanly(self):
        good = self.image("h.dll", b"MZ")
        report = os.path.join(self.tmp.name, "report.json")
        for bad in (os.path.join(self.tmp.name, "missing.dll"), self.tmp.name):   # missing, and a folder
            with self.subTest(bad=bad):
                p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), good, bad,
                                    "--header", self.header, "--json", report], capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn(f"ERROR: cannot read {bad}", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertFalse(os.path.exists(report))

    def test_cli_missing_header_fails_cleanly(self):
        dll = self.image("e.dll", b"MZ")
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), dll,
                            "--header", os.path.join(self.tmp.name, "missing.h")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("ERROR", p.stderr)

    def test_cli_json_over_a_missing_directory_fails_cleanly(self):
        dll = self.image("i.dll", b"MZ")
        target = os.path.join(self.tmp.name, "missing_dir", "out.json")
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), dll,
                            "--header", self.header, "--json", target], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("ERROR: cannot write", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertFalse(os.path.exists(target))

    def test_cli_json_over_an_input_image_is_refused(self):
        dll = self.image("f.dll", pe32plus({}, QUEUE.bytes_le))
        with open(dll, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), dll,
                            "--header", self.header, "--json", dll], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite", p.stderr)
        with open(dll, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_cli_json_over_the_header_is_refused(self):
        dll = self.image("g.dll", b"MZ")
        with open(self.header, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_iid_scan.py"), dll,
                            "--header", self.header, "--json", self.header], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite", p.stderr)
        with open(self.header, "rb") as f:
            self.assertEqual(f.read(), before)


if __name__ == "__main__":
    unittest.main()
