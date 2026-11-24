"""The PAL stage-descriptor reader decodes tag, ids, stride and sentinel of a synthetic FSSM header.

All blobs here are built by hand; no shader from the game is used.

    python -I tests/test_pal_stages.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "pal_stages.py")
sys.path.insert(0, TOOLS)

import pal_stages  # noqa: E402

TAG = 0x000000C000020000
SENTINEL64 = 0xDEADBEEFDEADBEEF


def blob(descriptors=(), code_off=0x210, sentinel_at=(), lone_at=(), size=0x300, magic=b"FSSM"):
    """Synthetic FSSM header with one 0x88-byte record per (api_id, hardware_id) pair.

    `sentinel_at` lists record-relative offsets written with a 0xDEADBEEFDEADBEEF qword and
    `lone_at` lists record-relative offsets written with a single 0xDEADBEEF word.
    """
    data = bytearray(size)
    data[0:4] = magic
    struct.pack_into("<H", data, 4, 0x101)
    struct.pack_into("<Q", data, 0xC0, code_off)
    for index, (api_id, hardware_id) in enumerate(descriptors):
        base = 0x100 + 0x88 * index
        struct.pack_into("<Q", data, base, TAG | (api_id << 8) | hardware_id)
        for offset in sentinel_at:
            struct.pack_into("<Q", data, base + offset, SENTINEL64)
        for offset in lone_at:
            struct.pack_into("<I", data, base + offset, 0xDEADBEEF)
    return bytes(data)


def make_xbch(fssm):
    compressed = zlib.compress(fssm)
    return b"XBCH" + struct.pack("<IIIII", 0, len(compressed), 0, len(fssm), 0) + compressed


def make_dxbc(payloads):
    """A DXBC container whose DXIL chunk holds an XDAT directory of tag-0x0D payloads."""
    table_len = 12 + 12 * len(payloads)
    entries, body = [], bytearray()
    for payload in payloads:
        entries.append(struct.pack("<III", 0x0D, table_len + len(body), len(payload)))
        body.extend(payload)
    dxil = b"XDAT" + struct.pack("<II", len(payloads), 8) + b"".join(entries) + bytes(body)
    chunks = [(b"DXIL", dxil)]
    head = 32 + 4 * len(chunks)
    offsets, parts = [], bytearray()
    for fourcc, payload in chunks:
        offsets.append(head + len(parts))
        parts.extend(fourcc + struct.pack("<I", len(payload)) + payload)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(parts), len(chunks))
    return header + b"".join(struct.pack("<I", off) for off in offsets) + bytes(parts)


class TestDescriptors(unittest.TestCase):
    def test_pixel_record(self):
        records = pal_stages.stage_descriptors(blob([(1, 5)]))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["offset"], 0x100)
        self.assertEqual(record["stride"], 0x88)
        self.assertEqual(record["tag"], 0x000000C000020105)
        self.assertEqual((record["api_id"], record["api_stage"]), (1, "ps"))
        self.assertEqual((record["hardware_id"], record["hardware_stage"]), (5, "ps"))

    def test_geometry_records_have_the_documented_stride(self):
        records = pal_stages.stage_descriptors(blob([(2, 3), (2, 4)]))
        self.assertEqual([record["offset"] for record in records], [0x100, 0x188])
        self.assertEqual([record["api_stage"] for record in records], ["gs", "gs"])
        self.assertEqual([record["hardware_stage"] for record in records], ["gs", "vs"])

    def test_mesh_record_uses_the_ngg_hardware_id(self):
        record = pal_stages.stage_descriptors(blob([(6, 7)]))[0]
        self.assertEqual((record["api_id"], record["api_stage"]), (6, "ms"))
        self.assertEqual((record["hardware_id"], record["hardware_stage"]), (7, "ngg"))

    def test_compute_record(self):
        record = pal_stages.stage_descriptors(blob([(3, 6)]))[0]
        self.assertEqual((record["api_stage"], record["hardware_stage"]), ("cs", "cs"))

    def test_export_shader_variant_of_a_vertex_stage(self):
        record = pal_stages.stage_descriptors(blob([(0, 2)]))[0]
        self.assertEqual((record["api_stage"], record["hardware_stage"]), ("vs", "es"))

    def test_unknown_ids_keep_the_record(self):
        record = pal_stages.stage_descriptors(blob([(9, 9)]))[0]
        self.assertEqual((record["api_id"], record["api_stage"]), (9, None))
        self.assertEqual((record["hardware_id"], record["hardware_stage"]), (9, None))

    def test_table_ends_at_a_zero_qword(self):
        self.assertEqual(len(pal_stages.stage_descriptors(blob([(1, 5)]))), 1)

    def test_table_ends_at_a_foreign_qword(self):
        data = bytearray(blob([(1, 5)]))
        struct.pack_into("<Q", data, 0x188, 0x00000006010B0105)
        self.assertEqual(len(pal_stages.stage_descriptors(bytes(data))), 1)

    def test_code_offset_bounds_the_table(self):
        # A second record at 0x188 starts where the code section begins, so it is not a record.
        records = pal_stages.stage_descriptors(blob([(2, 3), (2, 4)], code_off=0x188))
        self.assertEqual(len(records), 1)

    def test_bad_magic(self):
        self.assertEqual(pal_stages.stage_descriptors(blob([(1, 5)], magic=b"SSMF")), [])

    def test_stage_helpers(self):
        data = blob([(3, 6)])
        self.assertEqual(pal_stages.api_stage(data), "cs")
        self.assertEqual(pal_stages.hardware_stage(data), "cs")
        self.assertIsNone(pal_stages.api_stage(blob(magic=b"SSMF")))

    def test_paired_sentinel_fields(self):
        record = pal_stages.stage_descriptors(blob([(1, 5)], sentinel_at=(0x30, 0x70, 0x78)))[0]
        self.assertEqual(record["sentinel_words"], 3)
        self.assertEqual(record["lone_sentinel_words"], 0)

    def test_lone_sentinel_word_is_counted_separately(self):
        record = pal_stages.stage_descriptors(
            blob([(1, 5)], sentinel_at=(0x30,), lone_at=(0x3C,)))[0]
        self.assertEqual(record["sentinel_words"], 1)
        self.assertEqual(record["lone_sentinel_words"], 1)


class TestContainers(unittest.TestCase):
    def test_every_tag_0d_payload_is_read(self):
        container = make_dxbc([make_xbch(blob([(0, 2)])), make_xbch(blob([(0, 4)]))])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "shader.dxbc")
            with open(path, "wb") as f:
                f.write(container)
            rows = pal_stages.blob_rows(folder)
        self.assertEqual([row["payload"] for row in rows], ["xdat0", "xdat1"])
        self.assertEqual([pal_stages.api_stage(row["blob"]) for row in rows], ["vs", "vs"])
        self.assertEqual([pal_stages.hardware_stage(row["blob"]) for row in rows], ["es", "vs"])

    def test_directory_prefers_a_bare_fssm_over_a_same_stem_container(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "shader.fssm"), "wb") as f:
                f.write(blob([(1, 5)]))
            with open(os.path.join(folder, "shader.dxbc"), "wb") as f:
                f.write(make_dxbc([make_xbch(blob([(3, 6)]))]))
            rows = pal_stages.blob_rows(folder)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload"], "fssm")


class TestCli(unittest.TestCase):
    def test_json_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "shader.fssm"), "wb") as f:
                f.write(blob([(2, 3), (2, 4)], sentinel_at=(0x30,)))
            result = subprocess.run([sys.executable, "-I", TOOL, folder, "--json"],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["tally"]["descriptor_counts"], {"2": 1})
        self.assertEqual(document["tally"]["sentinel_blobs"], 1)
        self.assertEqual(document["tally"]["lone_sentinel_words"], 0)
        records = document["blobs"][0]["descriptors"]
        self.assertEqual([record["hardware_stage"] for record in records], ["gs", "vs"])
        self.assertEqual([record["offset"] for record in records], [0x100, 0x188])

    def test_container_is_read_and_limit_counts_files(self):
        container = make_dxbc([make_xbch(blob([(6, 7)]))])
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "a.dxbc"), "wb") as f:
                f.write(container)
            with open(os.path.join(folder, "b.fssm"), "wb") as f:
                f.write(blob([(1, 5)]))
            result = subprocess.run([sys.executable, "-I", TOOL, folder, "--json", "--limit", "1"],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["tally"]["blobs"], 1)
        self.assertEqual(document["blobs"][0]["payload"], "xdat0")
        self.assertEqual(document["blobs"][0]["hardware_stage"], "ngg")

    def test_limit_below_one_is_refused(self):
        """--limit 0 used to select no file and report an empty run as '0 blobs, errors 0' with exit 0;
        a negative value cut files off the end of the listing. Both are a usage error now."""
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "shader.fssm"), "wb") as f:
                f.write(blob([(2, 3)]))
            for value in ("0", "-1"):
                result = subprocess.run([sys.executable, "-I", TOOL, folder, "--limit", value],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, f"--limit {value}: {result.stdout}")
                self.assertIn("--limit", result.stderr)
                self.assertIn("at least 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
