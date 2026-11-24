"""Unit tests for tools/bcpack_tables.py covering BC encoding lookup tables,
format descriptors, and mode bit validation.

All tests operate strictly on codec-level lookup tables and synthetic inputs;
zero game data or proprietary assets are used.

    python -I tests/test_bcpack_tables.py
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import bcpack  # noqa: E402
import bcpack_tables as tables  # noqa: E402


# --- Lookup helpers and validation utilities ------------------------------

def decode_hdr0(hdr0: int, default_w: int) -> tuple[int, int, int, int, int]:
    """Decode and validate BCPack fragment header byte 0 (hdr0).

    Bit layout:
      bits 6-7: z (width reduction from default, 0..3)
      bit 5:    h3 (length header size: 0 = 1 byte, 1 = 2 bytes)
      bit 4:    flush (flush rANS stream flag)
      bits 0-3: q (exponential probability model index, 0..15)

    Returns (z, h3, flush, q, effective_w).
    Raises ValueError on invalid bit patterns or unsupported (effective_w, q).
    """
    if not (0 <= hdr0 <= 0xFF):
        raise ValueError(f"hdr0 byte out of range 0..255: {hdr0}")
    z = (hdr0 >> 6) & 3
    h3 = (hdr0 >> 5) & 1
    flush = (hdr0 >> 4) & 1
    q = hdr0 & 15
    effective_w = default_w - z
    if effective_w < 1 or effective_w > bcpack.MAX_WIDTH:
        raise ValueError(f"effective symbol width {effective_w} out of range 1..{bcpack.MAX_WIDTH}")
    if (effective_w, q) not in tables.PROB_TABLES:
        raise ValueError(f"unsupported probability model ({effective_w}, {q})")
    return z, h3, flush, q, effective_w


def decode_fragment_length(h3: int, hdr1: int, hdr2: int = 0) -> int:
    """Compute payload length n from fragment header fields."""
    if not (0 <= hdr1 <= 0xFF):
        raise ValueError(f"hdr1 out of range: {hdr1}")
    if h3 == 0:
        return hdr1 + 1
    if not (0 <= hdr2 <= 0xFF):
        raise ValueError(f"hdr2 out of range: {hdr2}")
    return hdr1 + 256 * hdr2 + 257


def lookup_history_entry(
    history_confirmed: dict[int, tuple[int, ...]],
    history_provisional: dict[int, tuple[int, ...]],
    index: int,
    *,
    allow_provisional: bool = False,
) -> tuple[int, ...]:
    """Look up a 1-based initial endpoint history entry (1..255)."""
    if not (1 <= index <= 255):
        raise ValueError(f"history index out of range 1..255: {index}")
    if index in history_confirmed:
        return history_confirmed[index]
    if index in history_provisional:
        if not allow_provisional:
            raise bcpack.BcpackProvisionalError(
                f"unconfirmed built-in history entry {index} (pass allow_provisional=True)"
            )
        return history_provisional[index]
    raise KeyError(f"unknown history entry {index}")


def pack_rgb565(r: int, g: int, b: int) -> int:
    """Pack 5-bit R, 6-bit G, 5-bit B into a 16-bit unsigned integer."""
    if not (0 <= r < 32 and 0 <= g < 64 and 0 <= b < 32):
        raise ValueError(f"RGB565 channel out of bounds: r={r}, g={g}, b={b}")
    return (r << 11) | (g << 5) | b


def unpack_rgb565(val: int) -> tuple[int, int, int]:
    """Unpack 16-bit unsigned integer into 5-bit R, 6-bit G, 5-bit B."""
    return (val >> 11) & 0x1F, (val >> 5) & 0x3F, val & 0x1F


def quadrant_to_indices(quad_byte: int) -> tuple[int, int, int, int]:
    """Extract 4 2-bit quadrant indices (TL, TR, BL, BR) from an 8-bit quadrant byte."""
    if not (0 <= quad_byte <= 0xFF):
        raise ValueError(f"quadrant byte out of bounds: {quad_byte}")
    return (
        quad_byte & 0x03,
        (quad_byte >> 2) & 0x03,
        (quad_byte >> 4) & 0x03,
        (quad_byte >> 6) & 0x03,
    )


def indices_to_quadrant(tl: int, tr: int, bl: int, br: int) -> int:
    """Pack 4 2-bit quadrant indices (TL, TR, BL, BR) into an 8-bit quadrant byte."""
    for name, v in [("tl", tl), ("tr", tr), ("bl", bl), ("br", br)]:
        if not (0 <= v <= 3):
            raise ValueError(f"index {name}={v} out of 2-bit range 0..3")
    return tl | (tr << 2) | (bl << 4) | (br << 6)


# --- Unit Test Suites -----------------------------------------------------

class TestProbTablesStructure(unittest.TestCase):
    """Verifies rANS banded exponential probability lookup tables in PROB_TABLES."""

    def test_total_table_count(self) -> None:
        # Exactly 10 symbol widths (1..10) * 16 models (q=0..15) = 160 tables
        self.assertEqual(len(tables.PROB_TABLES), 160)

    def test_key_space_completeness(self) -> None:
        expected_keys = {(w, q) for w in range(1, 11) for q in range(16)}
        self.assertEqual(set(tables.PROB_TABLES.keys()), expected_keys)

    def test_frequency_tuple_structure_and_types(self) -> None:
        for (w, q), freqs in tables.PROB_TABLES.items():
            self.assertIsInstance(freqs, tuple, f"freqs for ({w}, {q}) should be tuple")
            # Width w has band 0 (symbol 0) plus w non-zero bands: total w + 1 entries
            self.assertEqual(len(freqs), w + 1, f"len mismatch for ({w}, {q})")
            for p, f in enumerate(freqs):
                self.assertIsInstance(f, int, f"band {p} in ({w}, {q}) is not int")
                self.assertGreater(f, 0, f"band {p} in ({w}, {q}) must be strictly positive")

    def test_weight_sum_normalization_to_65536(self) -> None:
        # Each band p covers 1 symbol (p=0) or 2^(p-1) symbols (p>=1).
        # Total cumulative frequency weight must sum to exactly 65536 (2^16).
        for (w, q), freqs in tables.PROB_TABLES.items():
            acc = sum(f * (1 if p == 0 else (1 << (p - 1))) for p, f in enumerate(freqs))
            self.assertEqual(acc, 65536, f"weight sum for ({w}, {q}) is {acc} != 65536")

    def test_uniform_model_q0(self) -> None:
        # At q=0 (uniform distribution), every symbol has identical frequency 65536 >> w
        for w in range(1, 11):
            freqs = tables.PROB_TABLES[(w, 0)]
            expected_freq = 65536 >> w
            for p, f in enumerate(freqs):
                self.assertEqual(f, expected_freq, f"band {p} of ({w}, 0) is not uniform")

    def test_q_monotonicity_symbol0_concentration(self) -> None:
        # Higher q concentrates greater mass on symbol 0 (band 0 frequency is non-decreasing)
        for w in range(1, 11):
            f0_prev = 0
            for q in range(16):
                f0_cur = tables.PROB_TABLES[(w, q)][0]
                self.assertGreaterEqual(
                    f0_cur,
                    f0_prev,
                    f"freq[0] decreased at w={w}, q={q} ({f0_cur} < {f0_prev})",
                )
                f0_prev = f0_cur

    def test_cumulative_base_monotonicity_and_partition(self) -> None:
        # Verify cumulative base intervals strictly partition [0, 65536)
        for (w, q), freqs in tables.PROB_TABLES.items():
            base: list[int] = []
            acc = 0
            for p, f in enumerate(freqs):
                base.append(acc)
                acc += f * (1 if p == 0 else (1 << (p - 1)))
            self.assertEqual(base[0], 0)
            self.assertEqual(acc, 65536)
            for p in range(1, len(base)):
                self.assertGreater(base[p], base[p - 1], f"non-increasing base in ({w}, {q})")

    def test_model_symbol_lookup_boundaries(self) -> None:
        # Test cumulative frequency cf mapping to valid symbol range [0, 2^w - 1]
        for w in (1, 4, 8, 10):
            for q in (0, 7, 15):
                m = bcpack._get_model(w, q)
                self.assertIsNotNone(m)
                # cf = 0 maps to symbol 0
                self.assertEqual(m.lookup(0), 0)
                # cf = 65535 maps to maximum symbol (1 << w) - 1
                self.assertEqual(m.lookup(65535), (1 << w) - 1)
                # Monotonicity test across intermediate sample points
                samples = [0, 1, 100, 1000, 16384, 32768, 49152, 65534, 65535]
                prev_sym = 0
                for cf in samples:
                    sym = m.lookup(cf)
                    self.assertGreaterEqual(sym, prev_sym)
                    self.assertLess(sym, 1 << w)
                    prev_sym = sym


class TestBC1Tables(unittest.TestCase):
    """Verifies BC1 2x2 quadrant lane ranking and initial endpoint history."""

    def test_bc1_quadrant_type_and_length(self) -> None:
        self.assertIsInstance(tables.BC1_QUADRANT, bytes)
        self.assertEqual(len(tables.BC1_QUADRANT), 256)

    def test_bc1_quadrant_permutation_of_byte_values(self) -> None:
        # BC1_QUADRANT must be an exact 256-entry permutation of all bytes 0x00..0xFF
        self.assertEqual(len(set(tables.BC1_QUADRANT)), 256)
        self.assertEqual(set(tables.BC1_QUADRANT), set(range(256)))

    def test_bc1_quadrant_bidirectional_lookup(self) -> None:
        inv_quad = {b: i for i, b in enumerate(tables.BC1_QUADRANT)}
        self.assertEqual(len(inv_quad), 256)
        for sym in range(256):
            quad_byte = tables.BC1_QUADRANT[sym]
            self.assertEqual(inv_quad[quad_byte], sym)

    def test_bc1_quadrant_subfield_indices(self) -> None:
        for sym, quad_byte in enumerate(tables.BC1_QUADRANT):
            tl, tr, bl, br = quadrant_to_indices(quad_byte)
            self.assertTrue(0 <= tl <= 3 and 0 <= tr <= 3 and 0 <= bl <= 3 and 0 <= br <= 3)
            repacked = indices_to_quadrant(tl, tr, bl, br)
            self.assertEqual(repacked, quad_byte)

    def test_bc1_quadrant_canonical_anchors(self) -> None:
        # Uniform quadrant patterns: 0xaa (2,2,2,2), 0xff (3,3,3,3), 0x00 (0,0,0,0), 0x55 (1,1,1,1)
        self.assertEqual(tables.BC1_QUADRANT[0], 0xAA)
        self.assertEqual(tables.BC1_QUADRANT[1], 0xFF)
        self.assertEqual(tables.BC1_QUADRANT[2], 0x00)
        self.assertEqual(tables.BC1_QUADRANT[3], 0x55)
        self.assertEqual(quadrant_to_indices(tables.BC1_QUADRANT[0]), (2, 2, 2, 2))
        self.assertEqual(quadrant_to_indices(tables.BC1_QUADRANT[1]), (3, 3, 3, 3))
        self.assertEqual(quadrant_to_indices(tables.BC1_QUADRANT[2]), (0, 0, 0, 0))
        self.assertEqual(quadrant_to_indices(tables.BC1_QUADRANT[3]), (1, 1, 1, 1))

    def test_bc1_history_confirmed_counts_and_keys(self) -> None:
        self.assertEqual(len(tables.BC1_HISTORY_CONFIRMED), 252)
        self.assertNotIn(0, tables.BC1_HISTORY_CONFIRMED)
        for k in tables.BC1_HISTORY_CONFIRMED:
            self.assertTrue(1 <= k <= 255, f"key {k} out of range 1..255")

    def test_bc1_history_confirmed_bit_widths(self) -> None:
        # Each entry is (r0, g0, b0, r1, g1, b1) in RGB565 bit widths
        for k, ep in tables.BC1_HISTORY_CONFIRMED.items():
            self.assertEqual(len(ep), 6, f"entry {k} len != 6")
            r0, g0, b0, r1, g1, b1 = ep
            self.assertTrue(0 <= r0 < 32 and 0 <= r1 < 32, f"red out of range at key {k}")
            self.assertTrue(0 <= g0 < 64 and 0 <= g1 < 64, f"green out of range at key {k}")
            self.assertTrue(0 <= b0 < 32 and 0 <= b1 < 32, f"blue out of range at key {k}")

    def test_bc1_history_provisional_entries(self) -> None:
        self.assertEqual(len(tables.BC1_HISTORY_PROVISIONAL), 3)
        expected_provisional_keys = {6, 71, 122}
        self.assertEqual(set(tables.BC1_HISTORY_PROVISIONAL.keys()), expected_provisional_keys)
        # Confirmed and provisional must be strictly disjoint
        self.assertTrue(
            set(tables.BC1_HISTORY_CONFIRMED).isdisjoint(set(tables.BC1_HISTORY_PROVISIONAL))
        )
        # Together they cover all 255 non-zero history entries
        all_bc1_keys = set(tables.BC1_HISTORY_CONFIRMED) | set(tables.BC1_HISTORY_PROVISIONAL)
        self.assertEqual(all_bc1_keys, set(range(1, 256)))

    def test_bc1_history_rgb565_pack_unpack_roundtrip(self) -> None:
        for k, (r0, g0, b0, r1, g1, b1) in tables.BC1_HISTORY_CONFIRMED.items():
            packed0 = pack_rgb565(r0, g0, b0)
            packed1 = pack_rgb565(r1, g1, b1)
            self.assertTrue(0 <= packed0 <= 0xFFFF and 0 <= packed1 <= 0xFFFF)
            self.assertEqual(unpack_rgb565(packed0), (r0, g0, b0))
            self.assertEqual(unpack_rgb565(packed1), (r1, g1, b1))


class TestBC3Tables(unittest.TestCase):
    """Verifies BC3 alpha pair ranking, colour quadrant table, and endpoint history."""

    def test_bc3_alpha_pair_structure_and_permutation(self) -> None:
        self.assertIsInstance(tables.BC3_ALPHA_PAIR, tuple)
        self.assertEqual(len(tables.BC3_ALPHA_PAIR), 64)
        # Exactly 64 pairs covering all combinations of 3-bit indices (0..7, 0..7)
        all_expected_pairs = {(i, j) for i in range(8) for j in range(8)}
        self.assertEqual(set(tables.BC3_ALPHA_PAIR), all_expected_pairs)
        self.assertEqual(len(set(tables.BC3_ALPHA_PAIR)), 64)

    def test_bc3_alpha_pair_bidirectional_lookup(self) -> None:
        inv_pair = {p: i for i, p in enumerate(tables.BC3_ALPHA_PAIR)}
        self.assertEqual(len(inv_pair), 64)
        for i, pair in enumerate(tables.BC3_ALPHA_PAIR):
            self.assertEqual(inv_pair[pair], i)

    def test_bc3_colour_quadrant_structure_and_permutation(self) -> None:
        self.assertIsInstance(tables.BC3_COLOUR_QUADRANT, bytes)
        self.assertEqual(len(tables.BC3_COLOUR_QUADRANT), 256)
        self.assertEqual(len(set(tables.BC3_COLOUR_QUADRANT)), 256)
        self.assertEqual(set(tables.BC3_COLOUR_QUADRANT), set(range(256)))

    def test_bc3_colour_quadrant_bidirectional_lookup(self) -> None:
        inv_quad = {b: i for i, b in enumerate(tables.BC3_COLOUR_QUADRANT)}
        self.assertEqual(len(inv_quad), 256)
        for sym in range(256):
            self.assertEqual(inv_quad[tables.BC3_COLOUR_QUADRANT[sym]], sym)

    def test_bc3_history_structure_and_bounds(self) -> None:
        # BC3 confirmed history is empty; provisional has 35 entries
        self.assertEqual(len(tables.BC3_HISTORY_CONFIRMED), 0)
        self.assertEqual(len(tables.BC3_HISTORY_PROVISIONAL), 35)
        for k, ep in tables.BC3_HISTORY_PROVISIONAL.items():
            self.assertTrue(1 <= k <= 255, f"key {k} out of range 1..255")
            self.assertEqual(len(ep), 8, f"BC3 history entry {k} must have 8 channels")
            a0, a1, r0, g0, b0, r1, g1, b1 = ep
            self.assertTrue(0 <= a0 < 256 and 0 <= a1 < 256, f"alpha out of range at key {k}")
            self.assertTrue(0 <= r0 < 32 and 0 <= r1 < 32, f"red out of range at key {k}")
            self.assertTrue(0 <= g0 < 64 and 0 <= g1 < 64, f"green out of range at key {k}")
            self.assertTrue(0 <= b0 < 32 and 0 <= b1 < 32, f"blue out of range at key {k}")


class TestBC4AndBC5Tables(unittest.TestCase):
    """Verifies BC4 and BC5 3-bit pair ranking tables and endpoint history."""

    def test_bc4_pair_structure_and_permutation(self) -> None:
        self.assertIsInstance(tables.BC4_PAIR, tuple)
        self.assertEqual(len(tables.BC4_PAIR), 64)
        all_expected_pairs = {(i, j) for i in range(8) for j in range(8)}
        self.assertEqual(set(tables.BC4_PAIR), all_expected_pairs)
        self.assertEqual(len(set(tables.BC4_PAIR)), 64)

    def test_bc4_pair_bidirectional_lookup(self) -> None:
        inv_pair = {p: i for i, p in enumerate(tables.BC4_PAIR)}
        self.assertEqual(len(inv_pair), 64)
        for i, pair in enumerate(tables.BC4_PAIR):
            self.assertEqual(inv_pair[pair], i)

    def test_bc4_history_structure_and_bounds(self) -> None:
        self.assertEqual(len(tables.BC4_HISTORY_CONFIRMED), 2)
        self.assertEqual(len(tables.BC4_HISTORY_PROVISIONAL), 170)
        self.assertTrue(set(tables.BC4_HISTORY_CONFIRMED).isdisjoint(set(tables.BC4_HISTORY_PROVISIONAL)))
        for k, ep in {**tables.BC4_HISTORY_CONFIRMED, **tables.BC4_HISTORY_PROVISIONAL}.items():
            self.assertTrue(1 <= k <= 255, f"key {k} out of range 1..255")
            self.assertEqual(len(ep), 2, f"BC4 history entry {k} must have 2 channels")
            a0, a1 = ep
            self.assertTrue(0 <= a0 < 256 and 0 <= a1 < 256, f"endpoint out of range at key {k}")

    def test_bc5_pair_structure_and_permutation(self) -> None:
        self.assertIsInstance(tables.BC5_PAIR, tuple)
        self.assertEqual(len(tables.BC5_PAIR), 64)
        all_expected_pairs = {(i, j) for i in range(8) for j in range(8)}
        self.assertEqual(set(tables.BC5_PAIR), all_expected_pairs)
        self.assertEqual(len(set(tables.BC5_PAIR)), 64)

    def test_bc5_pair_bidirectional_lookup(self) -> None:
        inv_pair = {p: i for i, p in enumerate(tables.BC5_PAIR)}
        self.assertEqual(len(inv_pair), 64)
        for i, pair in enumerate(tables.BC5_PAIR):
            self.assertEqual(inv_pair[pair], i)

    def test_bc5_history_structure_and_bounds(self) -> None:
        self.assertEqual(len(tables.BC5_HISTORY_CONFIRMED), 5)
        self.assertEqual(set(tables.BC5_HISTORY_CONFIRMED.keys()), {105, 182, 183, 222, 255})
        self.assertEqual(len(tables.BC5_HISTORY_PROVISIONAL), 209)
        self.assertTrue(
            set(tables.BC5_HISTORY_CONFIRMED).isdisjoint(set(tables.BC5_HISTORY_PROVISIONAL))
        )
        for history in (tables.BC5_HISTORY_CONFIRMED, tables.BC5_HISTORY_PROVISIONAL):
            for k, ep in history.items():
                self.assertTrue(1 <= k <= 255, f"key {k} out of range 1..255")
                self.assertEqual(len(ep), 4, f"BC5 history entry {k} must have 4 channels")
                r0, r1, g0, g1 = ep
                self.assertTrue(
                    0 <= r0 < 256 and 0 <= r1 < 256 and 0 <= g0 < 256 and 0 <= g1 < 256,
                    f"channel out of range at key {k}",
                )


class TestFormatDescriptorsAndModeValidation(unittest.TestCase):
    """Verifies BCPack codec format descriptors, mode bits, and history lookup helpers."""

    def test_codec_names_and_block_bytes(self) -> None:
        self.assertEqual(bcpack.CODEC_NAMES[1], "BC1")
        self.assertEqual(bcpack.CODEC_NAMES[3], "BC3")
        self.assertEqual(bcpack.CODEC_NAMES[4], "BC4")
        self.assertEqual(bcpack.CODEC_NAMES[5], "BC5")
        self.assertEqual(bcpack.CODEC_NAMES[6], "BC6H")
        self.assertEqual(bcpack.CODEC_NAMES[7], "BC7")

        self.assertEqual(bcpack.BLOCK_BYTES[1], 8)
        self.assertEqual(bcpack.BLOCK_BYTES[3], 16)
        self.assertEqual(bcpack.BLOCK_BYTES[4], 8)
        self.assertEqual(bcpack.BLOCK_BYTES[5], 16)
        self.assertEqual(bcpack.BLOCK_BYTES[6], 16)
        self.assertEqual(bcpack.BLOCK_BYTES[7], 16)

    def test_lane_widths_match_prob_tables_domain(self) -> None:
        # Every default lane width must allow all valid reductions z in 0..3
        # such that effective width remains in [1, 10]
        for codec, widths in bcpack.LANE_WIDTHS.items():
            expected_lane_count = 16 if codec in (6, 7) else 12
            self.assertEqual(
                len(widths),
                expected_lane_count,
                f"codec {codec} should have {expected_lane_count} lanes",
            )
            for lane_idx, default_w in enumerate(widths):
                self.assertTrue(
                    4 <= default_w <= bcpack.MAX_WIDTH,
                    f"lane width {default_w} at codec {codec} lane {lane_idx} out of range",
                )
                for z in range(4):
                    eff_w = default_w - z
                    self.assertTrue(1 <= eff_w <= bcpack.MAX_WIDTH)
                    for q in range(16):
                        self.assertIn((eff_w, q), tables.PROB_TABLES)

    def test_mode_bit_validation_valid_cases(self) -> None:
        # Test synthetic hdr0 bitfield decoding
        # hdr0: z=1 (bits 6-7 = 01 -> 0x40), h3=1 (bit 5 = 1 -> 0x20), flush=1 (bit 4 = 1 -> 0x10), q=5 (0x05)
        # byte value: 0x40 | 0x20 | 0x10 | 0x05 = 0x75
        z, h3, flush, q, eff_w = decode_hdr0(0x75, default_w=8)
        self.assertEqual(z, 1)
        self.assertEqual(h3, 1)
        self.assertEqual(flush, 1)
        self.assertEqual(q, 5)
        self.assertEqual(eff_w, 7)

    def test_mode_bit_validation_rejection_rules(self) -> None:
        # Out-of-byte range
        with self.assertRaises(ValueError):
            decode_hdr0(-1, default_w=8)
        with self.assertRaises(ValueError):
            decode_hdr0(256, default_w=8)

        # Reduction z makes effective_w < 1: e.g. default_w = 2, z = 3 -> effective_w = -1
        # z=3 is 0b11000000 = 0xC0
        with self.assertRaises(ValueError):
            decode_hdr0(0xC0, default_w=2)

    def test_fragment_length_descriptor(self) -> None:
        # Short header (h3 = 0): n = hdr1 + 1 (1..256 bytes)
        self.assertEqual(decode_fragment_length(h3=0, hdr1=0), 1)
        self.assertEqual(decode_fragment_length(h3=0, hdr1=255), 256)

        # Extended header (h3 = 1): n = hdr1 + 256*hdr2 + 257 (257..65792 bytes)
        self.assertEqual(decode_fragment_length(h3=1, hdr1=0, hdr2=0), 257)
        self.assertEqual(decode_fragment_length(h3=1, hdr1=255, hdr2=255), 65792)

        # Invalid field values
        with self.assertRaises(ValueError):
            decode_fragment_length(h3=0, hdr1=-1)
        with self.assertRaises(ValueError):
            decode_fragment_length(h3=1, hdr1=0, hdr2=300)

    def test_history_lookup_helper_strict_vs_provisional(self) -> None:
        # BC1: confirmed entry 1 must succeed in both strict and provisional mode
        entry_conf = lookup_history_entry(
            tables.BC1_HISTORY_CONFIRMED,
            tables.BC1_HISTORY_PROVISIONAL,
            1,
            allow_provisional=False,
        )
        self.assertEqual(entry_conf, (7, 29, 6, 8, 30, 7))

        entry_conf_prov = lookup_history_entry(
            tables.BC1_HISTORY_CONFIRMED,
            tables.BC1_HISTORY_PROVISIONAL,
            1,
            allow_provisional=True,
        )
        self.assertEqual(entry_conf_prov, (7, 29, 6, 8, 30, 7))

        # BC1: provisional entry 6 must fail in strict mode and succeed in provisional mode
        with self.assertRaises(bcpack.BcpackProvisionalError):
            lookup_history_entry(
                tables.BC1_HISTORY_CONFIRMED,
                tables.BC1_HISTORY_PROVISIONAL,
                6,
                allow_provisional=False,
            )

        entry_prov = lookup_history_entry(
            tables.BC1_HISTORY_CONFIRMED,
            tables.BC1_HISTORY_PROVISIONAL,
            6,
            allow_provisional=True,
        )
        self.assertEqual(entry_prov, (12, 24, 11, 7, 14, 6))

        # BC5: confirmed entry 182 vs provisional entry 3
        bc5_conf = lookup_history_entry(
            tables.BC5_HISTORY_CONFIRMED,
            tables.BC5_HISTORY_PROVISIONAL,
            182,
            allow_provisional=False,
        )
        self.assertEqual(bc5_conf, (133, 126, 124, 55))

        with self.assertRaises(bcpack.BcpackProvisionalError):
            lookup_history_entry(
                tables.BC5_HISTORY_CONFIRMED,
                tables.BC5_HISTORY_PROVISIONAL,
                3,
                allow_provisional=False,
            )

        bc5_prov = lookup_history_entry(
            tables.BC5_HISTORY_CONFIRMED,
            tables.BC5_HISTORY_PROVISIONAL,
            3,
            allow_provisional=True,
        )
        self.assertEqual(bc5_prov, (130, 218, 45, 192))

        # Key 0 or out of bounds (1..255) must raise ValueError
        with self.assertRaises(ValueError):
            lookup_history_entry(
                tables.BC1_HISTORY_CONFIRMED,
                tables.BC1_HISTORY_PROVISIONAL,
                0,
                allow_provisional=True,
            )
        with self.assertRaises(ValueError):
            lookup_history_entry(
                tables.BC1_HISTORY_CONFIRMED,
                tables.BC1_HISTORY_PROVISIONAL,
                256,
                allow_provisional=True,
            )


if __name__ == "__main__":
    unittest.main()
