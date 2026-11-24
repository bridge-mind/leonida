"""tools/d3d12_guid_sites.py on hand-built PE32+ images (no game files, no SDK needed)."""
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

import d3d12_guid_sites as sites  # noqa: E402

QUEUE = uuid.UUID("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
RESOURCE = uuid.UUID("696442be-a72e-4059-bc79-5b5c98040fad")
UNKNOWN = uuid.UUID("12345678-4242-4343-8444-555555555555")
KNOWN = {"ID3D12CommandQueue": QUEUE, "ID3D12Resource": RESOURCE}


def build_rdata(rdata: bytes, imports=None):
    """The .rdata body, its import directory RVA and its IAT slot RVAs; pe32plus places it at 0x2000.

    imports: {dll_name: [function names, ...]} placed inside the body after `rdata`.
    """
    body = bytearray(rdata)
    imports = imports or {}
    rdata_rva = 0x2000
    iat_slots = []

    def put(blob, align=8):
        while len(body) % align:
            body.append(0)
        at = len(body)
        body.extend(blob)
        return rdata_rva + at

    imp = 0
    descriptors = []
    for dll, funcs in imports.items():
        thunks = [put(struct.pack("<H", 0) + f.encode() + b"\0", 2) for f in funcs]
        lookup = put(b"".join(struct.pack("<Q", t) for t in thunks) + b"\0" * 8)
        iat = put(b"".join(struct.pack("<Q", t) for t in thunks) + b"\0" * 8)
        name = put(dll.encode() + b"\0", 2)
        descriptors.append((lookup, name, iat))
        iat_slots.extend(iat + 8 * i for i in range(len(funcs)))
    if descriptors:
        imp = put(b"".join(struct.pack("<IIIII", lk, 0, 0, nm, ia) for lk, nm, ia in descriptors)
                  + b"\0" * 20, 4)
    return bytes(body), imp, iat_slots


def pe32plus(text: bytes, rdata: bytes, imports=None):
    """The finished two-section PE32+ image: (bytes, {import name: IAT slot RVA})."""
    body, imp, iat_slots = build_rdata(rdata, imports)
    names = [f for funcs in (imports or {}).values() for f in funcs]
    text_rva, text_raw = 0x1000, 0x400
    rdata_rva, rdata_raw = 0x2000, text_raw + ((len(text) + 0x1FF) // 0x200) * 0x200

    pe = 0x80
    opt_size = 240
    hdr = bytearray(rdata_raw)
    hdr[0:2] = b"MZ"
    struct.pack_into("<I", hdr, 0x3C, pe)
    hdr[pe:pe + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", hdr, pe + 4, 0x8664, 2, 0, 0, 0, opt_size, 0x2022)
    opt = pe + 24
    struct.pack_into("<H", hdr, opt, 0x20B)  # PE32+
    struct.pack_into("<Q", hdr, opt + 24, 0x180000000)  # ImageBase
    struct.pack_into("<I", hdr, opt + 108, 16)  # NumberOfRvaAndSizes
    if imp:
        struct.pack_into("<II", hdr, opt + 112 + 8, imp, 20 * (len(imports) + 1))
    sec = opt + opt_size
    hdr[sec:sec + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", hdr, sec + 8, len(text), text_rva, len(text), text_raw)
    struct.pack_into("<I", hdr, sec + 36, 0x60000020)  # code, execute, read
    hdr[sec + 40:sec + 48] = b".rdata\0\0"
    struct.pack_into("<IIII", hdr, sec + 48, len(body), rdata_rva, len(body), rdata_raw)
    struct.pack_into("<I", hdr, sec + 76, 0x40000040)  # initialized, read
    out = bytearray(hdr[:text_raw])
    out[text_raw:text_raw + len(text)] = text
    pad = (len(out) + 0x1FF) // 0x200 * 0x200
    out.extend(b"\0" * (pad - len(out)))
    out[rdata_raw:rdata_raw + len(body)] = body
    return bytes(out), dict(zip(names, iat_slots))


def lea(target_rva: int, insn_rva: int, rex=b"\x48", opcode=b"\x8d", reg=b"\x15"):
    """A RIP-relative `lea rdx, [rip+disp32]` at `insn_rva` pointing at `target_rva` (7 bytes)."""
    return rex + opcode + reg + struct.pack("<i", target_rva - (insn_rva + 7))


class GuidSites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_known_and_unknown_guids_at_call_sites(self):
        rdata = bytearray(b"\0" * 64)
        rdata[0:16] = QUEUE.bytes_le       # rva +0
        rdata[16:32] = UNKNOWN.bytes_le    # rva +16
        rdata[32:48] = RESOURCE.bytes_le   # rva +32
        text = lea(0x2000, 0x1000) + lea(0x2010, 0x1007) + lea(0x2020, 0x100E) + b"\x48\x89\xc1"
        image, _ = pe32plus(text, bytes(rdata))
        r = sites.scan(self.write("t1.dll", image), KNOWN)
        values = {c["value"]: c for c in r["guid_candidates"]}
        self.assertEqual(values[str(QUEUE)]["known"], ["ID3D12CommandQueue"])
        self.assertEqual(values[str(RESOURCE)]["known"], ["ID3D12Resource"])
        self.assertIsNone(values[str(UNKNOWN)]["known"])
        self.assertEqual(values[str(QUEUE)]["refs"], 1)
        self.assertEqual(r["lea_count"], 3)

    def test_lea_into_code_is_not_a_candidate(self):
        text = lea(0x1000, 0x1000) + b"\x48\x8b\x41\x08"  # points into .text itself
        image, _ = pe32plus(text, QUEUE.bytes_le)
        r = sites.scan(self.write("t2.dll", image), KNOWN)
        self.assertEqual(r["guid_candidates"], [])

    def test_dxbc_tagged_blob_is_not_a_candidate(self):
        blob = bytearray(b"DXBC" + bytes(range(12)))
        blob[7] = 0x42  # version 4
        blob[8] = 0x8a  # variant 10xx
        text = lea(0x2000, 0x1000)
        image, _ = pe32plus(text, bytes(blob))
        r = sites.scan(self.write("t3.dll", image), KNOWN)
        self.assertEqual(r["guid_candidates"], [])

    def test_iat_guid_load_is_named(self):
        rdata = QUEUE.bytes_le + b"\0" * (64 - 16)
        imports = {"d3d12_xs.dll": ["IID_Fake"]}
        _, slots = pe32plus(b"\x90" * 7, rdata, imports=imports)  # layout probe: same sizes, same slots
        text = lea(slots["IID_Fake"], 0x1000, rex=b"\x48", opcode=b"\x8b")  # mov rdx, [rip+iat slot]
        image, _ = pe32plus(text, rdata, imports=imports)
        r = sites.scan(self.write("t4.dll", image), KNOWN)
        self.assertEqual(len(r["iat_guid_loads"]), 1)
        self.assertEqual(r["iat_guid_loads"][0]["dll"], "d3d12_xs.dll")
        self.assertEqual(r["iat_guid_loads"][0]["name"], "IID_Fake")
        self.assertEqual(r["guid_candidates"], [])  # nothing leas the queue GUID in this image

    def test_queue_desc_window(self):
        rdata = (struct.pack("<IIII", 2, 0, 0, 0)      # a COMPUTE-shaped window at +0
                 + struct.pack("<IIII", 6, 0, 0, 0)    # a DMA-shaped window at +16
                 + b"\0" * 16                          # an all-zero window: padding
                 + UNKNOWN.bytes_le)
        text = lea(0x2000, 0x1000) + lea(0x2010, 0x1007) + lea(0x2030, 0x100E)  # GUID lands at +48
        image, _ = pe32plus(text, rdata)
        r = sites.scan(self.write("t5.dll", image), KNOWN)
        types = {d["type"] for d in r["queue_descs"]}
        self.assertEqual(types, {2, 6})
        self.assertEqual(len(r["queue_descs"]), 2)
        self.assertEqual(r["guid_candidates"][0]["value"], str(UNKNOWN))

    def test_guid_shaped_rejects_non_guids(self):
        self.assertFalse(sites.guid_shaped(b"\0" * 16))
        self.assertFalse(sites.guid_shaped(b"\x01" * 16))  # constant fill
        version_bad = bytearray(UNKNOWN.bytes_le)
        version_bad[7] = 0x90  # version 9
        self.assertFalse(sites.guid_shaped(bytes(version_bad)))
        variant_bad = bytearray(UNKNOWN.bytes_le)
        variant_bad[8] = 0x0a  # variant 00xx
        self.assertFalse(sites.guid_shaped(bytes(variant_bad)))
        self.assertTrue(sites.guid_shaped(UNKNOWN.bytes_le))

    def test_cli(self):
        rdata = QUEUE.bytes_le + UNKNOWN.bytes_le
        text = lea(0x2000, 0x1000) + lea(0x2010, 0x1007)
        image, _ = pe32plus(text, rdata)
        path = self.write("t6.dll", image)
        out = os.path.join(self.tmp.name, "out.json")
        header = os.path.join(self.tmp.name, "none.h")
        result = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "d3d12_guid_sites.py"), path,
             "--header", header, "--known", f"ID3D12CommandQueue={QUEUE}", "--json", out],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(UNKNOWN), result.stdout)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual({c["value"] for c in data["images"][0]["guid_candidates"]}, {str(QUEUE), str(UNKNOWN)})

    def test_cli_cannot_overwrite_an_input_with_its_json_report(self):
        image, _ = pe32plus(b"\x90", QUEUE.bytes_le)
        script = os.path.join(ROOT, "tools", "d3d12_guid_sites.py")
        header_text = b"// Synthetic IID header\n"
        for input_name in ("second.dll", "d3d12.h"):
            with self.subTest(input=input_name):
                first = self.write("first.dll", image)
                second = self.write("second.dll", image)
                header = self.write("d3d12.h", header_text)
                output = os.path.join(self.tmp.name, input_name)
                result = subprocess.run(
                    [sys.executable, "-I", script, first, second, "--header", header, "--json", output],
                    capture_output=True, text=True, cwd=ROOT,
                )
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("would overwrite", result.stderr)
                with open(second, "rb") as f:
                    self.assertEqual(f.read(), image)
                with open(header, "rb") as f:
                    self.assertEqual(f.read(), header_text)


if __name__ == "__main__":
    unittest.main()
