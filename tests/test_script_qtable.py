"""script_qtable.py must decode synthetic 0x71717171 object-tree containers without game data.

    python -I tests/test_script_qtable.py

Every fixture is synthetic: 40-byte header + u64 offset table + 16-byte-aligned
struct, wrapper and array records linked by 0x71717171-tagged pointers.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "script_qtable.py")
sys.path.insert(0, TOOLS)

import script_qtable  # noqa: E402

Q_PTR_HIGH = 0x71717171 << 32


def _pad16(buf: bytes) -> bytes:
    rem = len(buf) % 16
    return buf if rem == 0 else buf + b"\x00" * (16 - rem)


def build_synthetic_qtable(*, pad_to_bytes: int = 0) -> bytes:
    """Build a valid 7-record, 2-tree synthetic object-tree payload."""
    # Preorder record layout for 7 records (odd count => zero gap after 0x28 + 7*8):
    #   Tree 0 (root 0, hash 0x11111111):
    #     [0] 32B wrapper struct (type 0xAAAA0001, flags 0xFFFF0000) -> ptr array [1]
    #     [1] pointer array (count=2, stride=8, flags=0x10000, elem=0xBBBB0001) -> [2], [3] (64B)
    #     [2] 48B struct (type 0xAAAA0002, leaf or 0 ptrs)
    #     [3] 64B struct (type 0xAAAA0003)
    #   Tree 1 (root 4, hash 0x22222222):
    #     [4] 48B struct (type 0xAAAA0004) -> [5] and [6]
    #     [5] value array (count=3, stride=12, flags=0, elem=0xBBBB0002) -> align16(48+36)=96B
    #     [6] 32B leaf struct (type 0xAAAA0005)
    sizes = [32, 64, 48, 64, 48, 96, 32]
    if pad_to_bytes:
        # Enlarge leaf value array [5] so align16 stays exact.
        other_bytes = 40 + len(sizes) * 8 + sum(sizes) - sizes[5]
        cnt5 = max(3, (pad_to_bytes - other_bytes - 48) // 12)
        sizes[5] = ((48 + cnt5 * 12) + 15) & ~15
    else:
        cnt5 = 3

    table_end = 40 + len(sizes) * 8  # 96 = 0x60, 16-byte aligned because 7 is odd
    offsets = []
    cur = table_end
    for sz in sizes:
        offsets.append(cur)
        cur += sz

    def qptr(target_idx: int) -> int:
        return Q_PTR_HIGH | (offsets[target_idx] + 8)

    recs = [
        # [0] 32B wrapper -> [1]
        struct.pack("<4I2Q", 0x11111111, 0xFFFF0000, 0xAAAA0001, 0, qptr(1), 0),
        # [1] ptr array count=2, stride=8, flags=0x10000 -> 48 + 16 = 64B
        struct.pack("<12I2Q", 0, 0, 0x14, 0, 2, 2, 8, 0x10000, 0, 0, 0xBBBB0001, 0, qptr(2), qptr(3)),
        # [2] 48B struct (leaf)
        struct.pack("<4I", 0, 0, 0xAAAA0002, 0) + b"\x01" * 32,
        # [3] 64B struct (leaf)
        struct.pack("<4I", 0, 0, 0xAAAA0003, 0) + b"\x02" * 48,
        # [4] 48B struct (root 0x22222222) -> qptr(5) at +0x10 and qptr(6) at +0x20
        struct.pack("<4IQ2Q", 0x22222222, 0, 0xAAAA0004, 0, qptr(5), 0, qptr(6)) + bytes(8),
        # [5] value array (cnt5, stride=12, flags=0, elem=0xBBBB0002)
        _pad16(struct.pack("<12I", 0, 0, 0x14, 0, cnt5, cnt5, 12, 0, 0, 0, 0xBBBB0002, 0) + b"\x03" * (cnt5 * 12)),
        # [6] 32B leaf struct
        struct.pack("<4I2Q", 0, 0, 0xAAAA0005, 0, 42, 99),
    ]
    for i, r in enumerate(recs):
        assert len(r) == sizes[i], f"rec {i} len {len(r)} != {sizes[i]}"

    header = struct.pack("<10I", 0, 0, 0, 0, 0, len(offsets), 0, 0, 0, 0)
    offset_table = struct.pack(f"<{len(offsets)}Q", *offsets)
    return header + offset_table + b"".join(recs)


class ScriptQtableTests(unittest.TestCase):
    def test_parse_validates_synthetic_forest_and_invariants(self):
        blob = build_synthetic_qtable()
        rep = script_qtable.parse_qtable(blob)
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["record_count"], 7)
        self.assertEqual(rep["root_count"], 2)
        self.assertEqual(rep["distinct_root_hashes"], 2)
        self.assertTrue(rep["root_bijection_holds"])
        self.assertTrue(rep["contiguous_preorder"])
        self.assertEqual(rep["total_pointers"], 5)
        self.assertEqual(rep["resolved_pointers"], 5)
        self.assertEqual(rep["unresolved_pointers"], 0)
        self.assertEqual(rep["array_records"], 2)
        self.assertEqual(rep["array_size_exact"], 2)
        self.assertEqual(rep["array_pointer_records"], 1)
        self.assertEqual(rep["array_value_records"], 1)
        self.assertEqual(rep["struct_records"], 5)
        self.assertEqual(rep["struct_wrappers"], 1)
        self.assertEqual(rep["array_strides"], {"8": 1, "12": 1})
        self.assertEqual(rep["struct_sizes"], {"32": 2, "48": 2, "64": 1})

    def test_load_joins_split_64k_streams_from_carve_dir(self):
        big = build_synthetic_qtable(pad_to_bytes=70_000)
        self.assertGreater(len(big), 65536)
        chunk0 = big[:65536]
        chunk1 = big[65536:]

        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            rows = [
                {"off": 0x1000, "stored": 500, "size": len(b"PSIN" + bytes(120)), "head": (b"PSIN" + bytes(12)).hex()},
                {"off": 0x2000, "stored": 10000, "size": len(chunk0), "head": chunk0[:16].hex()},
                {"off": 0x5000, "stored": 2000, "size": len(chunk1), "head": chunk1[:16].hex()},
                {"off": 0x9000, "stored": 400, "size": 128, "head": (b"PSIN" + bytes(12)).hex()},
            ]
            with open(os.path.join(carve, f"{rows[0]['off']:012x}.bin"), "wb") as f:
                f.write(b"PSIN" + bytes(124))
            with open(os.path.join(carve, f"{rows[1]['off']:012x}.bin"), "wb") as f:
                f.write(chunk0)
            with open(os.path.join(carve, f"{rows[2]['off']:012x}.bin"), "wb") as f:
                f.write(chunk1)
            with open(os.path.join(carve, f"{rows[3]['off']:012x}.bin"), "wb") as f:
                f.write(b"PSIN" + bytes(124))
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

            out_json = os.path.join(tmp, "out", "qtable.json")
            p = subprocess.run(
                [sys.executable, "-I", TOOL, carve, "--root-hash", "0x11111111", "--type-hash", "0xBBBB0002", "--json", out_json],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(out_json, "rb") as f:
                raw = f.read()
            self.assertIn(b"\n", raw)
            self.assertNotIn(b"\r\n", raw)
            rep = json.loads(raw.decode("utf-8"))
            self.assertEqual(rep["location"]["offset_hex"], "0x2000")
            self.assertEqual(rep["location"]["chunks"], 2)
            self.assertEqual(rep["root_query"]["matches"], 1)
            self.assertEqual(rep["root_query"]["trees"][0]["nodes"], 4)
            self.assertEqual(rep["type_query"]["array_matches"], 1)
            self.assertEqual(rep["type_query"]["distinct_trees"], 1)

    def test_detects_corrupt_pointer_and_refuses_input_overwrite(self):
        bad = bytearray(build_synthetic_qtable())
        # Corrupt the last record's u64 at +0x10 into a dangling 0x71717171 pointer
        struct.pack_into("<Q", bad, len(bad) - 16, Q_PTR_HIGH | 0xDEADBEEF)
        rep = script_qtable.parse_qtable(bytes(bad))
        self.assertFalse(rep["valid"])
        self.assertEqual(rep["unresolved_pointers"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "qtable.bin")
            with open(bin_path, "wb") as f:
                f.write(build_synthetic_qtable())
            p = subprocess.run(
                [sys.executable, "-I", TOOL, bin_path, "--json", bin_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 2)
            self.assertIn("overwrite", p.stderr)

            bad_file = os.path.join(tmp, "not_qtable.bin")
            with open(bad_file, "wb") as f:
                f.write(b"NOTQ" * 16)
            p2 = subprocess.run(
                [sys.executable, "-I", TOOL, bad_file],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p2.returncode, 1)

            p3 = subprocess.run(
                [sys.executable, "-I", TOOL, bin_path, "--root-hash", "nothex"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p3.returncode, 2)
            self.assertIn("invalid hex", p3.stderr)

    def test_missing_input_is_one_error_line_and_exit_2(self):
        # A target the tool cannot open is a diagnostic, not a short census: the other tools exit 2 here.
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no_such_container.bin")
            p = subprocess.run(
                [sys.executable, "-I", TOOL, missing],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            lines = [line for line in p.stderr.splitlines() if line.strip()]
            self.assertEqual(len(lines), 1, p.stderr)
            self.assertEqual(lines[0], f"ERROR: {missing}: does not exist")
            self.assertEqual(p.stdout, "")

    def test_json_it_cannot_write_is_one_error_line_and_exit_2(self):
        # The scan succeeds and then the report cannot be written; that is not a payload verdict (exit 1).
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "qtable.bin")
            with open(bin_path, "wb") as f:
                f.write(build_synthetic_qtable())
            blocker = os.path.join(tmp, "blocker")
            with open(blocker, "wb") as f:
                f.write(b"")
            p = subprocess.run(
                [sys.executable, "-I", TOOL, bin_path, "--json", os.path.join(blocker, "out.json")],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            lines = [line for line in p.stderr.splitlines() if line.strip()]
            self.assertEqual(len(lines), 1, p.stderr)
            self.assertTrue(lines[0].startswith("ERROR: cannot write "), lines[0])
            self.assertFalse(os.path.exists(os.path.join(blocker, "out.json")))


        exact = build_synthetic_qtable(pad_to_bytes=65536)
        self.assertEqual(len(exact), 65536)
        opaque = b"\x01\x02\x03\x04" * 4

        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            rows = [
                {"off": 0x2000, "stored": 9000, "size": len(exact), "head": exact[:16].hex()},
                {"off": 0x8000, "stored": 16, "size": len(opaque), "head": opaque[:16].hex()},
            ]
            with open(os.path.join(carve, f"{rows[0]['off']:012x}.bin"), "wb") as f:
                f.write(exact)
            with open(os.path.join(carve, f"{rows[1]['off']:012x}.bin"), "wb") as f:
                f.write(opaque)
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

            blob, meta = script_qtable.load_qtable_payload(carve)
            self.assertEqual(meta["chunks"], 1)
            self.assertEqual(meta["inflated"], 65536)
            self.assertEqual(meta["source_type"], "directory")
            self.assertTrue(script_qtable.parse_qtable(blob)["valid"])

            raw_dir = os.path.join(tmp, "raw_bins")
            os.makedirs(raw_dir)
            with open(os.path.join(raw_dir, "000000002000.bin"), "wb") as f:
                f.write(exact)
            with open(os.path.join(raw_dir, "000000008000.bin"), "wb") as f:
                f.write(opaque)
            blob2, meta2 = script_qtable.load_qtable_payload(raw_dir)
            self.assertEqual(meta2["source_type"], "directory")
            self.assertEqual(meta2["chunks"], 1)
            self.assertEqual(len(blob2), 65536)

    def test_array_marker_mismatched_capacity_and_truncated_header_fail_validation(self):
        # Record [1] is the 64B array at offsets[1]; bump capacity from 2 to 3.
        bad_cap = bytearray(build_synthetic_qtable())
        rec1_off = struct.unpack_from("<Q", bad_cap, 40 + 8)[0]
        struct.pack_into("<I", bad_cap, rec1_off + 20, 3)
        rep = script_qtable.parse_qtable(bytes(bad_cap))
        self.assertEqual(rep["array_records"], 2)
        self.assertEqual(rep["array_count_eq_capacity"], 1)
        self.assertFalse(rep["valid"])

        # Set w2=0x14, w3=0 on 32B struct record [6] (shorter than 48B array header).
        short_arr = bytearray(build_synthetic_qtable())
        rec6_off = struct.unpack_from("<Q", short_arr, 40 + 6 * 8)[0]
        struct.pack_into("<2I", short_arr, rec6_off + 8, script_qtable.ARRAY_MARKER, 0)
        rep2 = script_qtable.parse_qtable(bytes(short_arr))
        self.assertEqual(rep2["array_records"], 3)
        self.assertEqual(rep2["array_header_complete"], 2)
        self.assertFalse(rep2["valid"])

    def test_contiguous_preorder_requires_strict_dfs_and_rejects_short_offset_table(self):
        def build_4rec(r0_ptrs: list[int], r1_ptrs: list[int]) -> bytes:
            table_end = 40 + 4 * 8  # 72; align16(72) == 80
            first_off = 80
            offsets = [first_off + i * 32 for i in range(4)]

            def pack_rec(rkey: int, thash: int, ptrs: list[int]) -> bytes:
                w = [0, 0]
                for idx, tgt in enumerate(ptrs):
                    w[idx] = Q_PTR_HIGH | (offsets[tgt] + 8)
                return struct.pack("<4I2Q", rkey, 0, thash, 0, w[0], w[1])

            recs = b"".join([
                pack_rec(0x11111111, 0xAAAA0001, r0_ptrs),
                pack_rec(0, 0xAAAA0002, r1_ptrs),
                pack_rec(0, 0xAAAA0003, []),
                pack_rec(0, 0xAAAA0004, []),
            ])
            header = struct.pack("<10I", 0, 0, 0, 0, 0, 4, 0, 0, 0, 0)
            return header + struct.pack("<4Q", *offsets) + bytes(first_off - table_end) + recs

        ok_rep = script_qtable.parse_qtable(build_4rec([1, 3], [2]))
        self.assertTrue(ok_rep["contiguous_preorder"])
        self.assertTrue(ok_rep["valid"])

        bad_interleaved = script_qtable.parse_qtable(build_4rec([1, 2], [3]))
        self.assertFalse(bad_interleaved["contiguous_preorder"])
        self.assertFalse(bad_interleaved["valid"])

        bad_sibling_order = script_qtable.parse_qtable(build_4rec([3, 1], [2]))
        self.assertFalse(bad_sibling_order["contiguous_preorder"])
        self.assertFalse(bad_sibling_order["valid"])

        # Header claiming 11 records in 64B payload must raise ValueError, not struct.error.
        trunc = bytearray(64)
        struct.pack_into("<I", trunc, 0x14, 11)
        struct.pack_into("<Q", trunc, 40, 128)
        with self.assertRaisesRegex(ValueError, "shorter than"):
            script_qtable.parse_qtable(bytes(trunc))

    def test_all_root_forward_chain_runs_linear_time(self):
        n = 2000
        table_end = 40 + n * 8
        first_off = (table_end + 15) & ~15
        offsets = [first_off + i * 16 for i in range(n)]
        header = struct.pack("<10I", 0, 0, 0, 0, 0, n, 0, 0, 0, 0)
        offset_table = struct.pack(f"<{n}Q", *offsets)
        recs = bytearray()
        for i in range(n):
            word = (Q_PTR_HIGH | (offsets[i + 1] + 8)) if i + 1 < n else 0
            recs += struct.pack("<2IQ", i + 1, 0, word)
        blob = header + offset_table + bytes(first_off - table_end) + bytes(recs)

        t0 = time.monotonic()
        rep = script_qtable.parse_qtable(blob)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 1.0)
        self.assertFalse(rep["valid"])
        self.assertEqual(rep["cross_tree_edges"], n - 1)


if __name__ == "__main__":
    unittest.main()
