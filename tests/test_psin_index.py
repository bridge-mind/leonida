"""Synthetic tests for tools/psin_index.py (no game data)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psin_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import psin_index as P  # noqa: E402


def make_section(tag, payload=b""):
    size = 8 + len(payload)
    return tag.encode("latin-1") + struct.pack(">I", size) + payload


def make_psin(sections, version=22, table_bytes=0, first=None):
    body = b"".join(make_section(tag, payload) for tag, payload in sections)
    first = 16 if first is None else first
    header = bytearray(max(first, 16))
    header[0:4] = b"PSIN"
    struct.pack_into(">I", header, 4, first)
    struct.pack_into("<I", header, 8, version)
    struct.pack_into("<I", header, 12, table_bytes)
    return bytes(header) + body


class TestParse(unittest.TestCase):
    def test_complete_record(self):
        blob = make_psin([
            ("LSCH", b"\x01\x02\x03"),
            ("LENM", b"\x00" * 8),
            ("LGID", b""),
            ("LRSC", b"0DRP\x7a\x00\x00\x00"),
            ("CHKS", b"\x00" * 12),
        ])
        got = P.parse(blob)
        self.assertTrue(got["complete"])
        self.assertEqual(got["version"], 22)
        self.assertEqual([s["tag"] for s in got["sections"]],
                         ["LSCH", "LENM", "LGID", "LRSC", "CHKS"])
        self.assertEqual(got["sections"][2]["size"], 8)
        self.assertEqual(got["sections"][0]["payload"], 3)
        self.assertEqual(got["rsc_version"], 122)
        self.assertEqual(got["checksum_size"], 0)
        self.assertEqual(got["checksum"], "00000000")

    def test_rsc_and_checksum_fields(self):
        blob = make_psin([
            ("LRSC", b"0DRP\x7a\x00\x00\x00"),
            ("CHKS", struct.pack(">I", 5392) + b"\x99\xfe\x66\x9f"),
        ])
        got = P.parse(blob)
        self.assertTrue(got["complete"])
        self.assertEqual(got["rsc_version"], 122)
        self.assertEqual(got["sections"][0]["rsc_magic"], "0DRP")
        self.assertEqual(got["checksum_size"], 5392)
        self.assertEqual(got["checksum"], "99fe669f")

    def test_not_psin(self):
        self.assertIsNone(P.parse(b"NOPE" + b"\x00" * 20))

    def test_first_section_outside_file(self):
        blob = make_psin([("LSCH", b"")], first=16)
        blob = blob[:4] + struct.pack(">I", 9999) + blob[8:]
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("outside", got["error"])

    def test_first_section_must_exist(self):
        for padding in (0, 1, 7):
            with self.subTest(padding=padding):
                blob = make_psin([], first=32, table_bytes=16) + b"\0" * padding
                got = P.parse(blob)
                self.assertFalse(got["complete"])
                self.assertEqual(got["sections"], [])
                self.assertIn("first section header", got["error"])

    def test_empty_first_section_payload_is_valid(self):
        got = P.parse(make_psin([("LSCH", b"")], first=32, table_bytes=16))
        self.assertTrue(got["complete"])
        self.assertEqual(got["sections"][0]["size"], 8)

    def test_truncated_section(self):
        blob = make_psin([("LSCH", b"\x00" * 8)])
        got = P.parse(blob[:-2])
        self.assertFalse(got["complete"])

    def test_zero_size_section(self):
        blob = make_psin([("LSCH", b"")])
        blob = blob[:20] + struct.pack(">I", 0) + blob[24:]
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("size 0", got["error"])

    def test_trailing_bytes(self):
        blob = make_psin([("LSCH", b"")]) + b"junk"
        got = P.parse(blob)
        self.assertFalse(got["complete"])
        self.assertIn("ended at", got["error"])


def make_lsch(structs):
    """structs: [(hash, signature, size, align_log2, [(hash, offset, type, sub, index, ref), ...]), ...]"""
    heads = b"".join(struct.pack("<IIBBHI8sIHH", h, sig, 0, align, 0, 0, b"\xaa" * 8, size, 0, len(members))
                     for h, sig, size, align, members in structs)
    body = b"".join(struct.pack("<IIBBHI", *m) for *_, members in structs for m in members)
    return heads + body


def make_lenm(enums):
    """enums: [(hash, signature, [(value, hash), ...]), ...]"""
    heads = b"".join(struct.pack("<II8sII", h, sig, b"\xee" * 8, len(values), 0) for h, sig, values in enums)
    body = b"".join(struct.pack("<qQ", v, vh) for *_, values in enums for v, vh in values)
    return heads + body


def joaat(text):
    """Case-sensitive joaat. The fixtures hash invented names, so they carry no game hashes."""
    h = 0
    for c in text.encode():
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


SCHEMA = [
    (joaat("Outer"), 0x11111111, 0x20, 3, [
        (0x100, 0, 5, 0, 0, joaat("Inner")),           # element descriptor of the array below
        (joaat("items"), 0x00, 0x5C, 0, 0, 0),          # array of Inner, element at index 0
        (joaat("kind"), 0x10, 0x62, 0, 0, joaat("Kind")),
        (joaat("scale"), 0x14, 0x21, 0, 0, 0),
    ]),
    (joaat("Inner"), 0x22222222, 0x8, 2, [
        (joaat("a"), 0, 0x21, 0, 0, 0),
        (joaat("b"), 4, 0x21, 0, 0, 0),
    ]),
]
ENUMS = [
    (joaat("Kind"), 0x33333333, [(0, joaat("KIND_A")), (1, joaat("KIND_B")), (-1, joaat("KIND_NONE"))]),
    (joaat("Flags"), 0x44444444, [(1, joaat("F_ONE")), (4, joaat("F_FOUR"))]),
]


def make_lsns(blocks):
    """An LSNS payload: block count, (kind, bytes) table, then the blocks (filler records)."""
    table = struct.pack("<I", len(blocks)) + b"".join(struct.pack("<II", k, len(d)) for k, d in blocks)
    return table + b"".join(d for _k, d in blocks)


class TestLsns(unittest.TestCase):
    def test_blocks_fill_the_section(self):
        payload = make_lsns([(0x101, bytes(range(32))), (0x207, b"\xab" * 48)])
        self.assertEqual(P.decode_lsns(payload), [(0x101, 32), (0x207, 48)])
        record = P.parse(make_psin([("LSCH", b""), ("LSNS", payload), ("LGID", b"")]))
        self.assertTrue(record["complete"])
        self.assertEqual((record["lsns_blocks"], record["lsns_records"], record["lsns_error"]), (2, 5, ""))
        section = [s for s in record["sections"] if s["tag"] == "LSNS"][0]
        self.assertEqual(section["blocks"], [{"kind": "0x101", "bytes": 32}, {"kind": "0x207", "bytes": 48}])

    def test_empty_section(self):
        record = P.parse(make_psin([("LSNS", struct.pack("<I", 0)), ("LGID", b"")]))
        self.assertEqual((record["lsns_blocks"], record["lsns_records"]), (0, 0))
        self.assertEqual(P.summarize([record])["lsns_empty"], 1)

    def test_errors(self):
        cases = {
            "shorter than its block count": b"\x01",
            "runs past the section": struct.pack("<I", 3) + bytes(8),
            "not a multiple of 16": make_lsns([(0x101, bytes(20))]),
            "do not fill the section": make_lsns([(0x101, bytes(16))]) + bytes(16),
        }
        for message, payload in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    P.decode_lsns(payload)
                record = P.parse(make_psin([("LSNS", payload)]))
                self.assertIn(message, record["lsns_error"])
                self.assertIsNone(record["lsns_blocks"])
                self.assertTrue(record["complete"])        # a bad LSNS does not break the section walk

    def test_summary_kinds(self):
        records = [P.parse(make_psin([("LSNS", make_lsns([(0x101, bytes(16)), (0x11c, bytes(32))]))])),
                   P.parse(make_psin([("LSNS", make_lsns([(0x101, bytes(16))]))]))]
        summary = P.summarize(records)
        self.assertEqual(summary["lsns_kinds"], {"0x101": 2, "0x11c": 1})
        self.assertEqual((summary["lsns_decoded"], summary["lsns_records"], summary["lsns_errors"]), (2, 4, 0))


class TestSchema(unittest.TestCase):
    def test_lsch_layout(self):
        got = P.decode_lsch(make_lsch(SCHEMA))
        self.assertEqual([s["hash"] for s in got], [joaat("Outer"), joaat("Inner")])
        outer = got[0]
        self.assertEqual((outer["signature"], outer["size"], outer["align_log2"]), (0x11111111, 0x20, 3))
        self.assertEqual(len(outer["members"]), 4)
        self.assertEqual(outer["members"][1], {"hash": joaat("items"), "offset": 0, "type": 0x5C, "sub": 0,
                                               "index": 0, "ref": 0})
        self.assertEqual(outer["members"][2]["ref"], joaat("Kind"))
        self.assertEqual([m["offset"] for m in got[1]["members"]], [0, 4])
        self.assertEqual(P.check_schema(got), "")

    def test_lenm_layout(self):
        got = P.decode_lenm(make_lenm(ENUMS))
        self.assertEqual([e["hash"] for e in got], [joaat("Kind"), joaat("Flags")])
        self.assertEqual([v["value"] for v in got[0]["values"]], [0, 1, -1])
        self.assertEqual(got[1]["values"][1], {"value": 4, "hash": joaat("F_FOUR")})
        self.assertEqual(got[1]["signature"], 0x44444444)

    def test_empty_payloads(self):
        self.assertEqual(P.decode_lsch(b""), [])
        self.assertEqual(P.decode_lenm(b""), [])

    def test_struct_with_no_members(self):
        got = P.decode_lsch(make_lsch([(joaat("Empty"), 1, 0, 0, [])]))
        self.assertEqual(got[0]["members"], [])

    def test_count_that_fits_nothing(self):
        with self.assertRaises(ValueError):
            P.decode_lsch(make_lsch(SCHEMA)[:-16])
        with self.assertRaises(ValueError):
            P.decode_lenm(make_lenm(ENUMS) + b"\0" * 8)

    def test_check_schema_flags_bad_array_index(self):
        bad = [(joaat("Bad"), 0, 0x10, 3, [(joaat("items"), 0, 0x5C, 0, 0, 0)])]
        self.assertIn("not an element descriptor", P.check_schema(P.decode_lsch(make_lsch(bad))))

    def test_check_schema_flags_member_past_size(self):
        bad = [(joaat("Bad"), 0, 0x8, 2, [(joaat("x"), 8, 0x21, 0, 0, 0)])]
        self.assertIn("past size", P.check_schema(P.decode_lsch(make_lsch(bad))))

    def test_parse_counts_and_schema(self):
        blob = make_psin([("LSCH", make_lsch(SCHEMA)), ("LENM", make_lenm(ENUMS)), ("LGID", b"")])
        got = P.parse(blob, schema=True)
        self.assertTrue(got["complete"])
        self.assertEqual((got["structs"], got["members"], got["enums"], got["enum_values"]), (2, 6, 2, 5))
        self.assertEqual(got["schema_error"], "")
        self.assertEqual(got["schema"]["structs"][0]["hash"], f"{joaat('Outer'):08x}")
        self.assertEqual(got["schema"]["enums"][0]["values"][2], {"value": -1, "hash": f"{joaat('KIND_NONE'):08x}"})
        self.assertNotIn("schema", P.parse(blob))

    def test_undecodable_lsch_is_reported_not_fatal(self):
        got = P.parse(make_psin([("LSCH", b"\x01\x02\x03"), ("LENM", b"")]))
        self.assertTrue(got["complete"])
        self.assertIsNone(got["structs"])
        self.assertIn("LSCH", got["schema_error"])
        self.assertEqual(got["enums"], 0)

    def test_lenm_error_before_valid_lsch_is_preserved(self):
        got = P.parse(make_psin([("LENM", b"\x01\x02\x03"), ("LSCH", make_lsch(SCHEMA))]), schema=True)
        self.assertTrue(got["complete"])
        self.assertEqual(got["structs"], 2)
        self.assertIsNone(got["enums"])
        self.assertIn("LENM: no enum count", got["schema_error"])
        self.assertEqual(P.summarize([got])["schema_errors"], 1)
        self.assertTrue(got["schema"]["lsch"])
        self.assertFalse(got["schema"]["lenm"])
        self.assertIn("LENM: no enum count", got["schema"]["reason"])

    def test_both_lsch_and_lenm_errors_are_preserved(self):
        for sections in (
            [("LSCH", b"\x01\x02\x03"), ("LENM", b"\x01\x02\x03")],
            [("LENM", b"\x01\x02\x03"), ("LSCH", b"\x01\x02\x03")],
        ):
            with self.subTest(order=[t for t, _ in sections]):
                got = P.parse(make_psin(sections), schema=True)
                self.assertIn("LSCH: no struct count", got["schema_error"])
                self.assertIn("LENM: no enum count", got["schema_error"])
                self.assertIn("LSCH: no struct count", got["schema"]["reason"])
                self.assertIn("LENM: no enum count", got["schema"]["reason"])

    def test_lsch_check_schema_error_does_not_mask_lenm_decode_error(self):
        bad_lsch = make_lsch([(joaat("Bad"), 0, 0x8, 2, [(joaat("x"), 8, 0x21, 0, 0, 0)])])
        got = P.parse(make_psin([("LSCH", bad_lsch), ("LENM", b"\x01\x02\x03")]), schema=True)
        self.assertIn("past size", got["schema_error"])
        self.assertIn("LENM: no enum count", got["schema_error"])
        self.assertTrue(got["schema"]["lsch"])
        self.assertFalse(got["schema"]["lenm"])
        self.assertEqual(got["schema"]["reason"], "LENM: no enum count fills 3 bytes")

    def test_first_lsch_or_lenm_error_is_not_overwritten_by_duplicate_section(self):
        got = P.parse(make_psin([("LSCH", b"\x01\x02\x03"), ("LSCH", make_lsch(SCHEMA)), ("LENM", b"")]), schema=True)
        self.assertIsNone(got["structs"])
        self.assertIn("LSCH: no struct count", got["schema_error"])
        self.assertFalse(got["schema"]["lsch"])
        self.assertIn("LSCH: no struct count", got["schema"]["reason"])



class TestCli(unittest.TestCase):
    def test_schema_on_truncated_psin(self):
        # Found by @kapitche on #776: --schema read r["schema"] on a record parse() had returned early.
        good = make_psin([("LSCH", make_lsch(SCHEMA)), ("LENM", make_lenm(ENUMS))])
        cases = {
            "00000001.bin": good[:4] + struct.pack(">I", 9999) + good[8:],   # first section past EOF
            "00000002.bin": make_psin([], first=32, table_bytes=16),          # no first section header
            "00000003.bin": good[:-4],                                        # LENM cut short
            "00000004.bin": good,
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, blob in cases.items():
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(blob)
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--schema"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            files = {r["file"]: r for r in json.loads(p.stdout)["files"]}
            self.assertEqual(len(files), 4)
            for r in files.values():   # one shape for every record
                self.assertEqual(set(r["schema"]), {"structs", "enums", "lsch", "lenm", "reason"})
            first = files["00000001.bin"]["schema"]
            self.assertEqual((first["structs"], first["enums"], first["lsch"], first["lenm"]), ([], [], False, False))
            self.assertIn("outside the file", first["reason"])
            self.assertIn("first section header", files["00000002.bin"]["schema"]["reason"])
            cut = files["00000003.bin"]["schema"]
            self.assertTrue(cut["lsch"])
            self.assertFalse(cut["lenm"])
            self.assertEqual(len(cut["structs"]), 2)
            self.assertIn("no whole LENM section", cut["reason"])
            self.assertEqual(files["00000004.bin"]["schema"]["reason"], "")
            t = subprocess.run([sys.executable, "-I", TOOL, tmp, "--schema"], capture_output=True, text=True)
            self.assertEqual(t.returncode, 0, t.stderr)
            self.assertNotIn("Traceback", t.stderr)
            self.assertIn("schema: no whole LSCH section", t.stdout)

    def test_schema_reason_for_missing_and_bad_sections(self):
        got = P.parse(make_psin([("LGID", b"")]), schema=True)["schema"]
        self.assertEqual(got["reason"], "no LSCH section; no LENM section")
        got = P.parse(make_psin([("LSCH", b"\x01\x02\x03"), ("LENM", b"")]), schema=True)["schema"]
        self.assertFalse(got["lsch"])
        self.assertTrue(got["lenm"])
        self.assertIn("LSCH: no struct count", got["reason"])

    def test_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(make_psin([("LSCH", make_lsch(SCHEMA)), ("LENM", make_lenm(ENUMS))]))
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--schema"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            s = doc["summary"]
            self.assertEqual((s["lsch_decoded"], s["lenm_decoded"], s["structs"], s["members"],
                              s["enums"], s["enum_values"], s["schema_errors"]), (1, 1, 2, 6, 2, 5, 0))
            self.assertEqual(len(doc["files"][0]["schema"]["structs"][0]["members"]), 4)
            t = subprocess.run([sys.executable, "-I", TOOL, tmp, "--schema"], capture_output=True, text=True)
            self.assertEqual(t.returncode, 0, t.stderr)
            self.assertIn(f"struct {joaat('Outer'):08x}", t.stdout)
            self.assertIn("structs=2 members=6", t.stdout)


    def test_json_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = make_psin([("LSCH", b""), ("CHKS", b"\x00" * 4)])
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(good)
            with open(os.path.join(tmp, "00000002.bin"), "wb") as f:
                f.write(b"not a psin")
            with open(os.path.join(tmp, "00000003.bin"), "wb") as f:
                f.write(make_psin([]))
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            self.assertEqual(doc["summary"]["files"], 2)
            self.assertEqual(doc["summary"]["complete"], 1)
            self.assertEqual(doc["summary"]["incomplete"], 1)
            self.assertEqual(doc["summary"]["section_tags"], {"CHKS": 1, "LSCH": 1})

    def test_limit_below_one_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "00000001.bin")
            with open(path, "wb") as f:
                f.write(make_psin([("LSCH", b"")]))
            for value in ("0", "-1"):
                p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--limit", value],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertIn("must be at least 1", p.stderr)

    def test_missing_source_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            for missing in (os.path.join(tmp, "nope"), os.path.join(tmp, "nope.bin")):
                p = subprocess.run([sys.executable, "-I", TOOL, missing, "--json"],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertIn(missing, p.stderr)
                self.assertEqual(p.stdout, "")

    def test_unlistable_folder_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A folder that cannot be listed (no permission); patched so the test runs the same on every OS.
            with unittest.mock.patch("os.listdir", side_effect=PermissionError(13, "Permission denied")):
                code, err = run_main([tmp])
            self.assertEqual(code, 2)
            self.assertIn("Permission denied", err)


def run_main(argv):
    import io
    from contextlib import redirect_stderr, redirect_stdout
    err = io.StringIO()
    with unittest.mock.patch.object(sys, "argv", ["psin_index.py", *argv]), \
            redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = P.main()
    return code, err.getvalue()


if __name__ == "__main__":
    unittest.main()
