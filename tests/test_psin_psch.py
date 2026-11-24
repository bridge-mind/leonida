"""psin_psch.py must decode synthetic PMAP/PSCH/PSIG/STNS PSIN containers without game data.

    python -I tests/test_psin_psch.py

Every fixture is synthetic: minimal 'pppppppp' data blocks, big-endian PMAP and
PSCH struct/enum tables, PSIG signatures, STDN/STNS blocks and CHKS.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "psin_psch.py")
sys.path.insert(0, TOOLS)

import psin_psch  # noqa: E402


def _sec(tag: str, payload: bytes) -> bytes:
    return tag.encode("ascii") + struct.pack(">I", 8 + len(payload)) + payload


def build_synthetic_psch_psin(*, bad_array_desc: bool = False, bad_pmap_size: bool = False,
                              with_stdn: bool = True) -> bytes:
    """Build a synthetic complete PMAP/PSCH/PSIG/STNS PSIN container."""
    # Blocks at +0x10:
    #   block 1: struct 0x11111111, byte size 32
    #   block 2: anonymous scalar block type 0x1, length 16
    b1_len = 36 if bad_pmap_size else 32
    b2_len = 16
    data_area = bytearray(16 + b1_len + b2_len)
    data_area[0:4] = b"PSIN"
    struct.pack_into(">I", data_area, 4, len(data_area))  # first_section
    data_area[8:16] = b"pppppppp"

    # PMAP
    pmap_pay = struct.pack(">IHHH6s", 1, 0, 0, 2, b"pppppp")
    pmap_pay += struct.pack(">4I", 0x11111111, 16, 0, b1_len)
    pmap_pay += struct.pack(">4I", 0x00000001, 16 + b1_len, 0, b2_len)

    # PSCH (3 entries: Outer 0x11111111, Inner 0x22222222, Kind 0x33333333)
    outer_members = [
        (0x100, 0x0C, 0, 0, 0x22222222),              # [0] array elem descriptor (Inner)
        (0xAA01, 0x0D, 10, 0, 9 if bad_array_desc else 0),  # [1] array pointing to [0]
        (0xAA02, 0x0E, 1, 16, 0x33333333),            # [2] enum Kind
        (0x100, 0x0B, 0, 0, 0),                       # [3] map val elem descriptor
        (0x100, 0x07, 0, 0, 0),                       # [4] map key elem descriptor
        (0xAA03, 0x10, 1, 20, (3 << 16) | 4),         # [5] map (key=[4], val=[3])
    ]
    inner_members = [(0xBB01, 0x07, 0, 0, 0)]
    enum_values = [(0xCC01, 0, 1), (0xCC02, 0, -2)]

    e0 = struct.pack(">BBHII", 0, 12, len(outer_members), 32, 0) + b"".join(
        struct.pack(">IBBHI", *m) for m in outer_members
    )
    e1 = struct.pack(">BBHII", 0, 12, len(inner_members), 16, 0) + b"".join(
        struct.pack(">IBBHI", *m) for m in inner_members
    )
    e2 = struct.pack(">BBH", 1, 0, len(enum_values)) + b"".join(
        struct.pack(">IIq", *v) for v in enum_values
    )

    dir_end = 12 + 3 * 8
    o0 = dir_end
    o1 = o0 + len(e0)
    o2 = o1 + len(e1)
    psch_pay = (
        struct.pack(">I", 3)
        + struct.pack(">II", 0x11111111, o0)
        + struct.pack(">II", 0x22222222, o1)
        + struct.pack(">II", 0x33333333, o2)
        + e0 + e1 + e2
    )
    psig_pay = struct.pack(">3I", 0x01020304, 0x05060708, 0x090A0B0C)
    stns_rec = bytes(range(16))
    stns_pay = struct.pack("<3I", 1, 0x101, 16) + stns_rec + b"\x00\x00\x00\x00"

    sections = [
        _sec("PMAP", pmap_pay),
        _sec("PSCH", psch_pay),
        _sec("PSIG", psig_pay),
    ]
    if with_stdn:
        sections.append(_sec("STDN", stns_rec))
    sections.append(_sec("STNS", stns_pay))
    total = len(data_area) + sum(len(s) for s in sections) + 20
    sections.append(_sec("CHKS", struct.pack(">II", total, 0xCAFEBABE) + b"zppp"))
    return bytes(data_area) + b"".join(sections)


class PsinPschTests(unittest.TestCase):
    def test_parse_validates_pmap_psch_psig_stns_and_cross_checks(self):
        blob = build_synthetic_psch_psin()
        self.assertTrue(psin_psch.is_psch_container(blob))
        rep = psin_psch.parse_psch_container(blob)
        self.assertTrue(rep["valid"], rep["errors"])
        self.assertEqual(rep["pmap_blocks"], 2)
        self.assertEqual(rep["pmap_struct_blocks"], 1)
        self.assertEqual(rep["pmap_anon_kinds"], {"0x1": 1})
        self.assertTrue(rep["pmap_exact_end"])
        self.assertEqual((rep["structs"], rep["members"], rep["enums"], rep["enum_values"]), (2, 7, 1, 2))
        self.assertEqual(rep["psig_count"], 3)
        self.assertEqual((rep["stns_blocks"], rep["stns_records"], rep["stns_tail_pad"]), (1, 1, 4))
        self.assertTrue(rep["stdn_match"])

        # Empty (count=0) or 2-block STNS sections pad to a 16-byte boundary with 12 zero bytes.
        stns_zero = psin_psch.decode_stns(struct.pack("<I12x", 0))
        self.assertEqual((stns_zero["blocks"], stns_zero["records"], stns_zero["tail_pad"]), ([], 0, 12))
        with self.assertRaisesRegex(ValueError, "padding"):
            psin_psch.decode_stns(struct.pack("<I", 0) + b"\x01" * 12)

    def test_detects_bad_array_descriptor_and_pmap_size_mismatch(self):
        bad_arr = psin_psch.parse_psch_container(build_synthetic_psch_psin(bad_array_desc=True))
        self.assertFalse(bad_arr["valid"])
        self.assertTrue(any("bad descriptor" in e for e in bad_arr["errors"]))

        bad_pmap = psin_psch.parse_psch_container(build_synthetic_psch_psin(bad_pmap_size=True))
        self.assertFalse(bad_pmap["valid"])
        self.assertTrue(any("not divisible" in e for e in bad_pmap["errors"]))

    def test_cli_scans_carve_dir_writes_lf_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            # Mix one LSCH-style PSIN (must be ignored) and one PMAP/PSCH PSIN
            lsch_psin = bytearray(16)
            lsch_psin[0:4] = b"PSIN"
            struct.pack_into(">I", lsch_psin, 4, 16)
            lsch_blob = bytes(lsch_psin) + _sec("LSCH", b"") + _sec("LENM", b"") + _sec("CHKS", b"\x00" * 12)
            psch_blob = build_synthetic_psch_psin(with_stdn=False)

            rows = []
            for off, data in ((0x1000, lsch_blob), (0x2000, psch_blob)):
                with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
                    f.write(data)
                rows.append({"off": off, "stored": len(data), "size": len(data), "head": data[:16].hex()})
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

            out_json = os.path.join(tmp, "out", "psch.json")
            p = subprocess.run([sys.executable, "-I", TOOL, carve, "--json", out_json],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(out_json, "rb") as f:
                raw = f.read()
            self.assertIn(b"\n", raw)
            self.assertNotIn(b"\r\n", raw)
            rep = json.loads(raw.decode("utf-8"))
            self.assertEqual(rep["containers"], 1)
            self.assertEqual(rep["valid_containers"], 1)

            # Collision refusal
            p2 = subprocess.run([sys.executable, "-I", TOOL, carve, "--json", os.path.join(carve, "00002000".rjust(12, "0") + ".bin")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p2.returncode, 2)
            self.assertIn("overwrite", p2.stderr)


if __name__ == "__main__":
    unittest.main()
