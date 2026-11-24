"""Unit tests for tools/bcpack_tables_bptc.py covering public BPTC
specification tables, BC7 signature configuration mappings, and runtime
history loader validation.

All tests operate strictly on public D3D11/Khronos BPTC specification tables
and synthetic temporary inputs; zero game data or proprietary assets are used.

    python -I tests/test_bcpack_tables_bptc.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import bcpack_tables_bptc as bptc  # noqa: E402


def _make_bc6h_section() -> dict:
    """Build a synthetic sparse BC6H history section with valid shapes."""
    eh0 = [[100, 200, 300, 400], [500, 600, 700, 800], [900, 1000, 1100, 1200]]
    ih0 = list(range(16))
    return {
        "endpointHistory": {"0": eh0, "255": eh0},
        "indexHistory": {"0": ih0, "255": list(reversed(ih0))},
        "endpointClass": {"0": 3, "255": 1},
        "indexClass": {"0": 2, "255": 3},
    }


def _make_bc7_section() -> dict:
    """Build a synthetic sparse BC7 history section with nested and flat rows."""
    ep_nested = [[[c + e * 4 + sub * 8 for c in range(4)] for e in range(2)] for sub in range(3)]
    ix_nested = [[p + st * 16 for p in range(16)] for st in range(2)]
    return {
        "historyEndpoints": {"5": ep_nested},
        "historyEndpointBits": {"5": [8] * 24},
        "historyEndpointConfirmedBits": {"5": [8] * 24},
        "historyIndices": {"5": ix_nested},
        "historyIndexBits": {"5": [4] * 16 + [2] * 16},
        "historyIndexConfirmedBits": {"5": [4] * 16 + [0] * 16},
        "historyEndpointLastResort": [[5, 1, 0, 2], [9, 17]],
    }


class TestPartitionAndAnchorTables(unittest.TestCase):
    """Verifies public D3D11/Khronos 2-subset and 3-subset partition and anchor tables."""

    def test_bc7_part2_packed_structure_and_invariants(self) -> None:
        self.assertIsInstance(bptc.BC7_PART2_PACKED, tuple)
        self.assertEqual(len(bptc.BC7_PART2_PACKED), 64)
        self.assertEqual(len(set(bptc.BC7_PART2_PACKED)), 64)

        for part, mask in enumerate(bptc.BC7_PART2_PACKED):
            self.assertIsInstance(mask, int)
            self.assertTrue(0 <= mask <= 0xFFFF, f"part2[{part}]={mask:#x} out of 16-bit range")
            subsets = [(mask >> p) & 1 for p in range(16)]
            self.assertEqual(subsets[0], 0, f"part2[{part}] pixel 0 must be subset 0")
            self.assertEqual(set(subsets), {0, 1}, f"part2[{part}] must use both subsets 0 and 1")

        # Canonical BPTC partition masks: vertical half split (0) and horizontal half split (13).
        self.assertEqual(bptc.BC7_PART2_PACKED[0], 0xCCCC)
        self.assertEqual(bptc.BC7_PART2_PACKED[13], 0xFF00)

    def test_bc7_part3_packed_structure_and_invariants(self) -> None:
        self.assertIsInstance(bptc.BC7_PART3_PACKED, tuple)
        self.assertEqual(len(bptc.BC7_PART3_PACKED), 64)
        self.assertEqual(len(set(bptc.BC7_PART3_PACKED)), 64)

        for part, mask in enumerate(bptc.BC7_PART3_PACKED):
            self.assertIsInstance(mask, int)
            self.assertTrue(0 <= mask <= 0xFFFFFFFF, f"part3[{part}]={mask:#x} out of 32-bit range")
            subsets = [(mask >> (2 * p)) & 3 for p in range(16)]
            for p, sub in enumerate(subsets):
                self.assertIn(sub, (0, 1, 2), f"part3[{part}] pixel {p} has invalid subset {sub}")
            self.assertEqual(subsets[0], 0, f"part3[{part}] pixel 0 must be subset 0")
            self.assertEqual(set(subsets), {0, 1, 2}, f"part3[{part}] must use all 3 subsets")

    def test_bc7_anchor2_invariants_and_bc6h_prefix(self) -> None:
        self.assertIsInstance(bptc.BC7_ANCHOR2, tuple)
        self.assertEqual(len(bptc.BC7_ANCHOR2), 64)

        for part, (mask, anchor) in enumerate(zip(bptc.BC7_PART2_PACKED, bptc.BC7_ANCHOR2)):
            self.assertIsInstance(anchor, int)
            self.assertTrue(1 <= anchor <= 15, f"anchor2[{part}]={anchor} outside 1..15")
            self.assertEqual((mask >> anchor) & 1, 1, f"anchor2[{part}]={anchor} not in subset 1")

        # BC6H two-region partitions reuse the first 32 entries of BC7_ANCHOR2.
        for part in range(32):
            anchor = bptc.BC7_ANCHOR2[part]
            self.assertNotEqual(anchor, 0)
            self.assertEqual((bptc.BC7_PART2_PACKED[part] >> anchor) & 1, 1)

    def test_bc7_anchor3_invariants_and_distinct_subsets(self) -> None:
        self.assertIsInstance(bptc.BC7_ANCHOR3A, tuple)
        self.assertIsInstance(bptc.BC7_ANCHOR3B, tuple)
        self.assertEqual(len(bptc.BC7_ANCHOR3A), 64)
        self.assertEqual(len(bptc.BC7_ANCHOR3B), 64)

        for part, (mask, a3a, a3b) in enumerate(
            zip(bptc.BC7_PART3_PACKED, bptc.BC7_ANCHOR3A, bptc.BC7_ANCHOR3B)
        ):
            self.assertIsInstance(a3a, int)
            self.assertIsInstance(a3b, int)
            self.assertTrue(1 <= a3a <= 15, f"anchor3a[{part}]={a3a} outside 1..15")
            self.assertTrue(1 <= a3b <= 15, f"anchor3b[{part}]={a3b} outside 1..15")
            self.assertNotEqual(a3a, a3b, f"part3[{part}] anchors collide at {a3a}")

            sub0 = mask & 3
            sub_a = (mask >> (2 * a3a)) & 3
            sub_b = (mask >> (2 * a3b)) & 3
            self.assertEqual((sub0, sub_a, sub_b), (0, 1, 2), f"part3[{part}] anchor subsets mismatch")


class TestBc7ConfigAndStaticTables(unittest.TestCase):
    """Verifies BC7_CONFIG permutation, BC7_CONFIG_INFERRED, and _bc7_static()."""

    def test_mode2_head_is_descending_subset_of_64(self) -> None:
        self.assertIsInstance(bptc._BC7_MODE2_HEAD, tuple)
        self.assertEqual(len(bptc._BC7_MODE2_HEAD), 22)
        self.assertEqual(len(set(bptc._BC7_MODE2_HEAD)), 22)
        self.assertTrue(all(0 <= p < 64 for p in bptc._BC7_MODE2_HEAD))
        self.assertEqual(
            list(bptc._BC7_MODE2_HEAD),
            sorted(bptc._BC7_MODE2_HEAD, reverse=True),
        )

    def test_bc7_config_bijection_bounds_and_order(self) -> None:
        self.assertEqual(bptc._build_bc7_config(), bptc.BC7_CONFIG)
        self.assertIsInstance(bptc.BC7_CONFIG, tuple)
        self.assertEqual(len(bptc.BC7_CONFIG), 286)
        self.assertEqual(bptc.BC7_CONFIG[285], (255, 255, 0, 0))

        expected_configs = (
            {(0, p, 0, 0) for p in range(16)}
            | {(1, p, 0, 0) for p in range(64)}
            | {(2, p, 0, 0) for p in range(64)}
            | {(3, p, 0, 0) for p in range(64)}
            | {(4, 0, rot, isel) for rot in range(4) for isel in range(2)}
            | {(5, 0, rot, 0) for rot in range(4)}
            | {(6, 0, 0, 0)}
            | {(7, p, 0, 0) for p in range(64)}
        )
        self.assertEqual(len(expected_configs), 285)
        self.assertEqual(len(set(bptc.BC7_CONFIG[:-1])), 285)
        self.assertEqual(set(bptc.BC7_CONFIG[:-1]), expected_configs)

        for rank, entry in enumerate(bptc.BC7_CONFIG[:-1]):
            mode, part, rot, isel = entry
            self.assertTrue(0 <= mode <= 7, f"rank {rank} invalid mode {mode}")
            max_part = 16 if mode == 0 else (64 if mode in (1, 2, 3, 7) else 1)
            max_rot = 4 if mode in (4, 5) else 1
            max_isel = 2 if mode == 4 else 1
            self.assertTrue(0 <= part < max_part, f"rank {rank} invalid part {part} for mode {mode}")
            self.assertTrue(0 <= rot < max_rot, f"rank {rank} invalid rot {rot} for mode {mode}")
            self.assertTrue(0 <= isel < max_isel, f"rank {rank} invalid isel {isel} for mode {mode}")

        # Verify section ordering of BC7_CONFIG.
        self.assertEqual(bptc.BC7_CONFIG[0:64], tuple((7, 63 - r, 0, 0) for r in range(64)))
        self.assertEqual(bptc.BC7_CONFIG[64], (6, 0, 0, 0))
        self.assertEqual(bptc.BC7_CONFIG[65:69], tuple((5, 0, 3 - r, 0) for r in range(4)))
        self.assertEqual(
            bptc.BC7_CONFIG[69:77],
            tuple((4, 0, (7 - r) >> 1, (7 - r) & 1) for r in range(8)),
        )
        self.assertEqual(bptc.BC7_CONFIG[77:141], tuple((3, 63 - r, 0, 0) for r in range(64)))
        self.assertEqual(bptc.BC7_CONFIG[141:163], tuple((2, p, 0, 0) for p in bptc._BC7_MODE2_HEAD))
        self.assertEqual(bptc.BC7_CONFIG[163:227], tuple((1, 63 - r, 0, 0) for r in range(64)))
        self.assertEqual(
            bptc.BC7_CONFIG[227:241],
            tuple((0, p, 0, 0) for p in (*range(15, 7, -1), *range(5, -1, -1))),
        )
        mode2_tail = bptc.BC7_CONFIG[241:283]
        self.assertEqual(len(mode2_tail), 42)
        self.assertTrue(all(e[0] == 2 and e[2] == 0 and e[3] == 0 for e in mode2_tail))
        tail_parts = [e[1] for e in mode2_tail]
        self.assertEqual(tail_parts, sorted(tail_parts, reverse=True))
        self.assertEqual(set(tail_parts) | set(bptc._BC7_MODE2_HEAD), set(range(64)))
        self.assertEqual(bptc.BC7_CONFIG[283:285], ((0, 7, 0, 0), (0, 6, 0, 0)))

    def test_bc7_config_inferred_ranks(self) -> None:
        self.assertIsInstance(bptc.BC7_CONFIG_INFERRED, frozenset)
        self.assertEqual(
            bptc.BC7_CONFIG_INFERRED,
            frozenset((149, 151, 152, 153, 246, 248, 251, 252, 253, 254)),
        )
        for rank in bptc.BC7_CONFIG_INFERRED:
            self.assertTrue(0 <= rank < 285)
            self.assertEqual(bptc.BC7_CONFIG[rank][0], 2)

    def test_bc7_static_tables_and_rank_of_grid(self) -> None:
        static = bptc._bc7_static()
        self.assertIs(bptc._bc7_static(), static)
        self.assertEqual(
            set(static.keys()),
            {"partition2", "partition3", "anchor2", "anchor3a", "anchor3b", "config", "inferred", "rank_of"},
        )
        self.assertIs(static["anchor2"], bptc.BC7_ANCHOR2)
        self.assertIs(static["anchor3a"], bptc.BC7_ANCHOR3A)
        self.assertIs(static["anchor3b"], bptc.BC7_ANCHOR3B)
        self.assertIs(static["config"], bptc.BC7_CONFIG)
        self.assertIs(static["inferred"], bptc.BC7_CONFIG_INFERRED)

        self.assertEqual(len(static["partition2"]), 64)
        for mask, row in zip(bptc.BC7_PART2_PACKED, static["partition2"]):
            self.assertEqual(row, tuple((mask >> p) & 1 for p in range(16)))

        self.assertEqual(len(static["partition3"]), 64)
        for mask, row in zip(bptc.BC7_PART3_PACKED, static["partition3"]):
            self.assertEqual(row, tuple((mask >> (2 * p)) & 3 for p in range(16)))

        rank_of = static["rank_of"]
        self.assertIsInstance(rank_of, tuple)
        self.assertEqual(len(rank_of), 512)
        assigned = [r for r in rank_of if r >= 0]
        self.assertEqual(len(assigned), len(bptc.BC7_CONFIG))
        self.assertEqual(set(assigned), set(range(len(bptc.BC7_CONFIG))))
        self.assertEqual(rank_of.count(-1), 512 - len(bptc.BC7_CONFIG))

        # Rank 0 is (0, 0), rank 1 is (0, 1), rank 2 is (1, 0) via ((a+1)*(b+1), a).
        self.assertEqual(rank_of[0 * 16 + 0], 0)
        self.assertEqual(rank_of[0 * 16 + 1], 1)
        self.assertEqual(rank_of[1 * 16 + 0], 2)

        cells = sorted(
            ((a, b) for a in range(32) for b in range(16)),
            key=lambda ab: ((ab[0] + 1) * (ab[1] + 1), ab[0]),
        )
        for rank, (a, b) in enumerate(cells):
            expected_rank = rank if rank < len(bptc.BC7_CONFIG) else -1
            self.assertEqual(rank_of[a * 16 + b], expected_rank)


class TestHistoryHelpers(unittest.TestCase):
    """Verifies internal runtime history parsing, validation, and file caching helpers."""

    def test_iter_entries_valid_and_invalid(self) -> None:
        self.assertEqual(list(bptc._iter_entries(None)), [])
        self.assertEqual(list(bptc._iter_entries([10, 20, 30], count=3)), [(0, 10), (1, 20), (2, 30)])
        self.assertEqual(
            list(bptc._iter_entries({"0": "a", 255: "b"})),
            [(0, "a"), (255, "b")],
        )

        with self.assertRaises(ValueError):
            list(bptc._iter_entries([0] * 255))
        with self.assertRaises(ValueError):
            list(bptc._iter_entries([0] * 257))
        with self.assertRaises(ValueError):
            list(bptc._iter_entries({"-1": 1}))
        with self.assertRaises(ValueError):
            list(bptc._iter_entries({"256": 1}))
        with self.assertRaises(ValueError):
            list(bptc._iter_entries({"not_an_int": 1}))
        for bad_obj in ("string", 42, (1, 2, 3)):
            with self.assertRaises(ValueError):
                list(bptc._iter_entries(bad_obj))

    def test_flatten_bc7_entry_valid_and_invalid(self) -> None:
        nested24 = [[[c + e * 4 + sub * 8 for c in range(4)] for e in range(2)] for sub in range(3)]
        flat24 = tuple(range(24))
        self.assertEqual(bptc._flatten_bc7_entry(nested24, 24), flat24)
        self.assertEqual(bptc._flatten_bc7_entry([str(i) for i in range(24)], 24), flat24)

        nested32 = [[p + st * 16 for p in range(16)] for st in range(2)]
        self.assertEqual(bptc._flatten_bc7_entry(nested32, 32), tuple(range(32)))

        for bad_val, expected_len in (
            ([0] * 23, 24),
            ([0] * 25, 24),
            ([], 24),
            ([[[0] * 3 for _ in range(2)] for _ in range(3)], 24),
            ([0] * 31, 32),
            ([0] * 33, 32),
        ):
            with self.assertRaises(ValueError):
                bptc._flatten_bc7_entry(bad_val, expected_len)

        with self.assertRaises(ValueError):
            bptc._flatten_bc7_entry(["bad"] + [0] * 23, 24)
        with self.assertRaises(TypeError):
            bptc._flatten_bc7_entry([None] + [0] * 23, 24)

    def test_read_history_section_sources_errors_and_mtime_cache(self) -> None:
        self.assertIsNone(bptc._read_history_section(None, "bc6h", "endpointHistory"))
        self.assertEqual(
            bptc._read_history_section({"bc6h": {"a": 1}}, "bc6h", "endpointHistory"),
            {"a": 1},
        )
        direct = {"endpointHistory": {}}
        self.assertIs(bptc._read_history_section(direct, "bc6h", "endpointHistory"), direct)
        self.assertIsNone(bptc._read_history_section({"bc7": {}}, "bc6h", "endpointHistory"))

        with tempfile.TemporaryDirectory() as tmp:
            missing_file = os.path.join(tmp, "does_not_exist.json")
            with self.assertRaises(FileNotFoundError):
                bptc._read_history_section(missing_file, "bc6h", "endpointHistory")

            bad_json = os.path.join(tmp, "bad.json")
            with open(bad_json, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            with self.assertRaises(json.JSONDecodeError):
                bptc._read_history_section(bad_json, "bc6h", "endpointHistory")

            for idx, raw_root in enumerate(("[]", "123", '"str"', "null", "true")):
                non_obj = os.path.join(tmp, f"non_obj_{idx}.json")
                with open(non_obj, "w", encoding="utf-8") as fh:
                    fh.write(raw_root)
                with self.assertRaises(ValueError):
                    bptc._read_history_section(non_obj, "bc6h", "endpointHistory")

            empty_dir = os.path.join(tmp, "empty_dir")
            os.mkdir(empty_dir)
            self.assertIsNone(bptc._read_history_section(Path(empty_dir), "bc6h", "endpointHistory"))

            # Verify _FILE_CACHE reloads when mtime advances.
            hist_path = os.path.join(tmp, "cached.json")
            with open(hist_path, "w", encoding="utf-8") as fh:
                json.dump({"bc6h": {"endpointClass": {"0": 1}}}, fh)
            first = bptc._read_history_section(Path(hist_path), "bc6h", "endpointHistory")
            self.assertEqual(first, {"endpointClass": {"0": 1}})

            st = os.stat(hist_path)
            with open(hist_path, "w", encoding="utf-8") as fh:
                json.dump({"bc6h": {"endpointClass": {"0": 3}}}, fh)
            os.utime(hist_path, (st.st_atime, st.st_mtime + 10.0))
            second = bptc._read_history_section(hist_path, "bc6h", "endpointHistory")
            self.assertEqual(second, {"endpointClass": {"0": 3}})


class TestLoadBc6h(unittest.TestCase):
    """Verifies load_bc6h() defaults, source resolution, and input validation."""

    def test_legacy_confirmation_depth_defaults(self) -> None:
        section = _make_bc6h_section()
        eh, ih, ec, ic, ep_cb, ix_cb = bptc.load_bc6h(section, with_confirmed_bits=True)
        self.assertEqual((eh, ih, ec, ic), bptc.load_bc6h(section))
        self.assertEqual(ep_cb[0], ((16, 10, 6, 6),) * 3)
        self.assertEqual(ep_cb[255], ((0, 0, 0, 0),) * 3)
        self.assertEqual(ix_cb[255], (4,) * 16)
        self.assertEqual(ix_cb[0], (0,) * 16)
        empty = bptc.load_bc6h(with_confirmed_bits=True)
        self.assertEqual(empty[4], empty[0])
        self.assertEqual(empty[5], empty[1])

    def test_explicit_sparse_and_full_confirmation_depths(self) -> None:
        section = _make_bc6h_section()
        section["endpointConfirmedBits"] = {"0": [[16, 4, 0, 0]] * 3}
        section["indexConfirmedBits"] = {"0": [3] * 16}
        *_, ep_cb, ix_cb = bptc.load_bc6h(section, with_confirmed_bits=True)
        self.assertEqual(ep_cb[0], ((16, 4, 0, 0),) * 3)
        # Explicit sparse tables do not inherit legacy class-3 precision.
        self.assertEqual(ix_cb[255], (0,) * 16)
        self.assertEqual(ix_cb[0], (3,) * 16)
        section["endpointConfirmedBits"] = [list(map(list, row)) for row in ep_cb]
        section["indexConfirmedBits"] = [list(row) for row in ix_cb]
        self.assertEqual(bptc.load_bc6h(section, with_confirmed_bits=True)[4:], (ep_cb, ix_cb))

    def test_invalid_confirmation_depths(self) -> None:
        bad_rows = (
            ("endpointConfirmedBits", [[0] * 4] * 2),
            ("endpointConfirmedBits", [[0] * 3] * 3),
            ("endpointConfirmedBits", [[17, 0, 0, 0]] * 3),
            ("endpointConfirmedBits", [[0, 11, 0, 0]] * 3),
            ("endpointConfirmedBits", [[0, 0, 7, 0]] * 3),
            ("endpointConfirmedBits", [[0, 0, 0, -1]] * 3),
            ("indexConfirmedBits", [0] * 15),
            ("indexConfirmedBits", [5] * 16),
            ("indexConfirmedBits", [-1] * 16),
        )
        for key, row in bad_rows:
            with self.subTest(key=key, row=row):
                section = _make_bc6h_section()
                section[key] = {"0": row}
                with self.assertRaises(ValueError):
                    bptc.load_bc6h(section)

    def test_defaults_when_history_none_or_absent(self) -> None:
        eh, ih, ec, ic = bptc.load_bc6h()
        self.assertEqual(len(eh), 256)
        self.assertEqual(len(ih), 256)
        self.assertEqual(ec, (0,) * 256)
        self.assertEqual(ic, (0,) * 256)
        self.assertTrue(all(row == ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)) for row in eh))
        self.assertTrue(all(row == (0,) * 16 for row in ih))

        self.assertEqual(bptc.load_bc6h(None), (eh, ih, ec, ic))
        self.assertEqual(bptc.load_bc6h({"bc7": _make_bc7_section()}), (eh, ih, ec, ic))

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bptc.load_bc6h(tmp), (eh, ih, ec, ic))
            only_bc7 = os.path.join(tmp, "only_bc7.json")
            with open(only_bc7, "w", encoding="utf-8") as fh:
                json.dump({"bc7": _make_bc7_section()}, fh)
            self.assertEqual(bptc.load_bc6h(only_bc7), (eh, ih, ec, ic))

    def test_load_from_dict_file_and_directory_formats(self) -> None:
        sec = _make_bc6h_section()
        eh_d, ih_d, ec_d, ic_d = bptc.load_bc6h({"bc6h": sec})
        self.assertEqual(eh_d[0], ((100, 200, 300, 400), (500, 600, 700, 800), (900, 1000, 1100, 1200)))
        self.assertEqual(eh_d[1], ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
        self.assertEqual(ih_d[0], tuple(range(16)))
        self.assertEqual(ih_d[255], tuple(reversed(range(16))))
        self.assertEqual((ec_d[0], ec_d[1], ec_d[255]), (3, 0, 1))
        self.assertEqual((ic_d[0], ic_d[1], ic_d[255]), (2, 0, 3))

        # Direct dict (without "bc6h" wrapper) and full 256-entry list format.
        list_sec = {
            "endpointHistory": [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for _ in range(256)],
            "indexHistory": [[i % 16] * 16 for i in range(256)],
            "endpointClass": [3] * 256,
            "indexClass": [2] * 256,
        }
        list_sec["endpointHistory"][42] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        eh_l, ih_l, ec_l, ic_l = bptc.load_bc6h(list_sec)
        self.assertEqual(eh_l[42], ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)))
        self.assertEqual(ih_l[17], (1,) * 16)
        self.assertEqual(ec_l, (3,) * 256)
        self.assertEqual(ic_l, (2,) * 256)

        with tempfile.TemporaryDirectory() as tmp:
            combined = os.path.join(tmp, "combined.json")
            with open(combined, "w", encoding="utf-8") as fh:
                json.dump({"bc6h": sec, "bc7": _make_bc7_section()}, fh)
            self.assertEqual(bptc.load_bc6h(combined), (eh_d, ih_d, ec_d, ic_d))
            self.assertEqual(bptc.load_bc6h(Path(combined)), (eh_d, ih_d, ec_d, ic_d))

            hist_dir = os.path.join(tmp, "hist_dir")
            os.mkdir(hist_dir)
            with open(os.path.join(hist_dir, "bc6h.json"), "w", encoding="utf-8") as fh:
                json.dump(sec, fh)
            self.assertEqual(bptc.load_bc6h(hist_dir), (eh_d, ih_d, ec_d, ic_d))
            self.assertEqual(bptc.load_bc6h(Path(hist_dir)), (eh_d, ih_d, ec_d, ic_d))

    def test_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                bptc.load_bc6h(os.path.join(tmp, "missing.json"))

            bad_json = os.path.join(tmp, "corrupt.json")
            with open(bad_json, "w", encoding="utf-8") as fh:
                fh.write("[1, 2")
            with self.assertRaises(json.JSONDecodeError):
                bptc.load_bc6h(bad_json)

            non_dict = os.path.join(tmp, "array.json")
            with open(non_dict, "w", encoding="utf-8") as fh:
                json.dump([1, 2, 3], fh)
            with self.assertRaises(ValueError):
                bptc.load_bc6h(non_dict)

        bad_sections = [
            {"endpointHistory": [ [[0] * 4] * 3 ] * 255},
            {"endpointHistory": [ [[0] * 4] * 3 ] * 257},
            {"endpointHistory": {"-1": [[0] * 4] * 3}},
            {"endpointHistory": {"256": [[0] * 4] * 3}},
            {"endpointHistory": {"bad": [[0] * 4] * 3}},
            {"endpointHistory": "not_a_table"},
            {"endpointHistory": {"0": [[0] * 4] * 2}},
            {"endpointHistory": {"0": [[0] * 4] * 4}},
            {"endpointHistory": {"0": [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]]}},
            {"endpointHistory": {"0": [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 3]]}},
            {"indexHistory": [[0] * 16] * 255},
            {"indexHistory": {"256": [0] * 16}},
            {"indexHistory": {"0": [0] * 15}},
            {"indexHistory": {"0": [0] * 17}},
            {"endpointClass": [0] * 255},
            {"endpointClass": {"-1": 1}},
            {"endpointClass": 123},
            {"indexClass": [0] * 257},
            {"indexClass": {"256": 1}},
            {"endpointHistory": {"0": [["bad", 0, 0, 0], [0] * 4, [0] * 4]}},
            {"indexHistory": {"0": ["bad"] + [0] * 15}},
            {"endpointClass": {"0": "bad"}},
            {"indexClass": {"0": "bad"}},
        ]
        for bad_sec in bad_sections:
            with self.subTest(bad_sec=bad_sec):
                with self.assertRaises((ValueError, TypeError)):
                    bptc.load_bc6h({"bc6h": bad_sec})


class TestLoadBc7(unittest.TestCase):
    """Verifies load_bc7() defaults, mid-range fill, last-resort parsing, and validation."""

    def test_defaults_when_history_none_or_absent(self) -> None:
        t = bptc.load_bc7()
        self.assertEqual(
            set(t.keys()),
            {
                "partition2", "partition3", "anchor2", "anchor3a", "anchor3b",
                "config", "inferred", "rank_of",
                "history_endpoints", "history_endpoint_bits", "history_endpoint_confirmed_bits",
                "history_indices", "history_index_bits", "history_index_confirmed_bits",
                "last_resort",
            },
        )
        self.assertEqual(t["last_resort"], frozenset())
        for key in ("history_endpoints", "history_endpoint_bits", "history_endpoint_confirmed_bits"):
            self.assertEqual(len(t[key]), 256)
            self.assertTrue(all(row == (0,) * 24 for row in t[key]))
        for key in ("history_indices", "history_index_bits", "history_index_confirmed_bits"):
            self.assertEqual(len(t[key]), 256)
            self.assertTrue(all(row == (0,) * 32 for row in t[key]))

        self.assertEqual(bptc.load_bc7(None), t)
        self.assertEqual(bptc.load_bc7({"bc6h": _make_bc6h_section()}), t)

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bptc.load_bc7(tmp), t)
            only_bc6h = os.path.join(tmp, "only_bc6h.json")
            with open(only_bc6h, "w", encoding="utf-8") as fh:
                json.dump(_make_bc6h_section(), fh)
            self.assertEqual(bptc.load_bc7(only_bc6h), t)

    def test_load_from_dict_file_and_directory_formats(self) -> None:
        sec = _make_bc7_section()
        t_dict = bptc.load_bc7({"bc7": sec})
        self.assertEqual(t_dict["history_endpoints"][5], tuple(range(24)))
        self.assertEqual(t_dict["history_endpoints"][0], (0,) * 24)
        self.assertEqual(t_dict["history_endpoint_bits"][5], (8,) * 24)
        self.assertEqual(t_dict["history_endpoint_confirmed_bits"][5], (8,) * 24)
        self.assertEqual(t_dict["history_indices"][5], tuple(range(32)))
        self.assertEqual(t_dict["history_index_bits"][5], (4,) * 16 + (2,) * 16)
        self.assertEqual(t_dict["history_index_confirmed_bits"][5], (4,) * 16 + (0,) * 16)
        # 4-tuple [5, 1, 0, 2] -> (5, 1*8 + 0*4 + 2) = (5, 10)
        self.assertEqual(t_dict["last_resort"], frozenset({(5, 10), (9, 17)}))

        # Direct section dict and full 256-entry list format.
        list_sec = {
            "historyEndpoints": [[i % 256] * 24 for i in range(256)],
            "historyEndpointBits": [[8] * 24 for _ in range(256)],
            "historyEndpointConfirmedBits": [[8] * 24 for _ in range(256)],
            "historyIndices": [[i % 16] * 32 for i in range(256)],
            "historyIndexBits": [[4] * 32 for _ in range(256)],
            "historyIndexConfirmedBits": [[4] * 32 for _ in range(256)],
            "historyEndpointLastResort": [[0, 0, 0, 0], [0, 0], [255, 2, 1, 3]],
        }
        t_list = bptc.load_bc7(list_sec)
        self.assertEqual(t_list["history_endpoints"][200], (200,) * 24)
        self.assertEqual(t_list["history_indices"][3], (3,) * 32)
        self.assertEqual(t_list["last_resort"], frozenset({(0, 0), (255, 23)}))

        with tempfile.TemporaryDirectory() as tmp:
            combined = os.path.join(tmp, "combined.json")
            with open(combined, "w", encoding="utf-8") as fh:
                json.dump({"bc6h": _make_bc6h_section(), "bc7": sec}, fh)
            self.assertEqual(bptc.load_bc7(combined), t_dict)
            self.assertEqual(bptc.load_bc7(Path(combined)), t_dict)

            hist_dir = os.path.join(tmp, "hist_dir")
            os.mkdir(hist_dir)
            with open(os.path.join(hist_dir, "bc6h.json"), "w", encoding="utf-8") as fh:
                json.dump(_make_bc6h_section(), fh)
            with open(os.path.join(hist_dir, "bc7.json"), "w", encoding="utf-8") as fh:
                json.dump(sec, fh)
            self.assertEqual(bptc.load_bc7(hist_dir), t_dict)
            self.assertEqual(bptc.load_bc7(Path(hist_dir)), t_dict)
            self.assertEqual(bptc.load_bc6h(hist_dir)[2][0], 3)

    def test_midrange_fill_for_partial_endpoint_depths(self) -> None:
        # Test depths 0..8 across distinct byte patterns; 0 < d < 8 gets mid-range filled.
        test_vals = [0x00, 0xFF, 0xA5, 0x5A, 0x81, 0x7E, 0x3C, 0xC3, 0x99] + [0xAB] * 15
        depths = [0, 1, 2, 3, 4, 5, 6, 7, 8] + [9] * 15
        sec = {
            "historyEndpoints": {"12": test_vals},
            "historyEndpointBits": {"12": depths},
        }
        row = bptc.load_bc7({"bc7": sec})["history_endpoints"][12]
        for j, (v, d) in enumerate(zip(test_vals, depths)):
            if 0 < d < 8:
                expected = ((v >> (8 - d)) << (8 - d)) | (1 << (7 - d))
                self.assertEqual(row[j], expected, f"slot {j} depth {d}")
                # Top d bits preserved, bit 7-d set, lower bits cleared.
                self.assertEqual(row[j] >> (8 - d), v >> (8 - d))
                self.assertEqual((row[j] >> (7 - d)) & 1, 1)
                if d < 7:
                    self.assertEqual(row[j] & ((1 << (7 - d)) - 1), 0)
            else:
                self.assertEqual(row[j], v, f"slot {j} depth {d} should remain unchanged")

    def test_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                bptc.load_bc7(os.path.join(tmp, "missing.json"))

            bad_json = os.path.join(tmp, "corrupt.json")
            with open(bad_json, "w", encoding="utf-8") as fh:
                fh.write("{bad")
            with self.assertRaises(json.JSONDecodeError):
                bptc.load_bc7(bad_json)

            non_dict = os.path.join(tmp, "non_dict.json")
            with open(non_dict, "w", encoding="utf-8") as fh:
                json.dump("just a string", fh)
            with self.assertRaises(ValueError):
                bptc.load_bc7(non_dict)

        bad_ep_row = [0] * 23
        bad_ix_row = [0] * 31
        bad_sections = [
            {"historyEndpoints": [[0] * 24] * 255},
            {"historyEndpoints": [[0] * 24] * 257},
            {"historyEndpoints": {"-1": [0] * 24}},
            {"historyEndpoints": {"256": [0] * 24}},
            {"historyEndpoints": {"abc": [0] * 24}},
            {"historyEndpoints": "bad"},
            {"historyEndpoints": {"0": bad_ep_row}},
            {"historyEndpoints": {"0": [0] * 25}},
            {"historyEndpoints": {"0": [[[0] * 3 for _ in range(2)] for _ in range(3)]}},
            {"historyEndpointBits": {"0": bad_ep_row}},
            {"historyEndpointConfirmedBits": {"0": [0] * 25}},
            {"historyIndices": [[0] * 32] * 255},
            {"historyIndices": {"256": [0] * 32}},
            {"historyIndices": {"0": bad_ix_row}},
            {"historyIndices": {"0": [0] * 33}},
            {"historyIndexBits": {"0": bad_ix_row}},
            {"historyIndexConfirmedBits": {"0": [0] * 33}},
            {"historyEndpoints": {"0": ["x"] + [0] * 23}},
            {"historyIndices": {"0": ["x"] + [0] * 31}},
            {"historyEndpointLastResort": [[]]},
            {"historyEndpointLastResort": [[1]]},
            {"historyEndpointLastResort": [[1, 2, 3]]},
            {"historyEndpointLastResort": [[1, 2, 3, 4, 5]]},
            {"historyEndpointLastResort": [["bad", 0]]},
        ]
        for bad_sec in bad_sections:
            with self.subTest(bad_sec=bad_sec):
                with self.assertRaises((ValueError, TypeError)):
                    bptc.load_bc7({"bc7": bad_sec})


if __name__ == "__main__":
    unittest.main()
