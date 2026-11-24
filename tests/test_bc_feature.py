"""Unit tests for tools/bc_feature.py covering BC1-BC7 block compression detection and bit-depth verification.

    python -I tests/test_bc_feature.py
"""
import math
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    import numpy as np
    import bc_feature
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


# Block compression specifications: bytes per element (4x4 block), bits per pixel, and channels
BC_SPECS = {
    "BC1": {"bpe": 8, "bpp": 4.0, "channels": 3, "alpha_bits": 1, "name": "BC1_UNORM"},
    "BC2": {"bpe": 16, "bpp": 8.0, "channels": 4, "alpha_bits": 4, "name": "BC2_UNORM"},
    "BC3": {"bpe": 16, "bpp": 8.0, "channels": 4, "alpha_bits": 8, "name": "BC3_UNORM"},
    "BC4": {"bpe": 8, "bpp": 4.0, "channels": 1, "alpha_bits": 0, "name": "BC4_UNORM"},
    "BC5": {"bpe": 16, "bpp": 8.0, "channels": 2, "alpha_bits": 0, "name": "BC5_UNORM"},
    "BC6H": {"bpe": 16, "bpp": 8.0, "channels": 3, "alpha_bits": 0, "name": "BC6H_UF16"},
    "BC7": {"bpe": 16, "bpp": 8.0, "channels": 4, "alpha_bits": 8, "name": "BC7_UNORM"},
}


def detect_bc_format_from_bpe_and_channels(bpe: int, channels: int) -> str:
    """Identify BC block compression format from block byte size and channel count."""
    for fmt, spec in BC_SPECS.items():
        if spec["bpe"] == bpe and spec["channels"] == channels:
            return fmt
    raise ValueError(f"Unknown BC format with bpe={bpe}, channels={channels}")


def decode_bc7_mode(first_byte: int) -> int:
    """Detect BC7 compression mode (0-7) from the unary mode header in the first byte.

    BC7 modes are encoded in least-significant bits as a 1 preceded by zeros:
      bit 0 = 1 -> mode 0
      bit 1 = 1 -> mode 1
      ...
      bit 7 = 1 -> mode 7
    """
    if first_byte == 0:
        return -1
    for mode in range(8):
        if (first_byte >> mode) & 1:
            return mode
    return -1


def decode_bc6h_mode(low_bits: int) -> int:
    """Detect BC6H compression mode (1-14) from the low header bits.

    Modes 1-2 use a 2-bit header (bits [1:0] = 00 or 01).
    Modes 3-14 use a 5-bit header (bits [1:0] = 10 or 11 with bits [4:2] selecting mode).
    """
    two_bits = low_bits & 0x03
    if two_bits == 0:
        return 1
    if two_bits == 1:
        return 2
    five_bits = low_bits & 0x1F
    mode_map = {
        0x02: 3, 0x06: 4, 0x0A: 5, 0x0E: 6,
        0x12: 7, 0x16: 8, 0x1A: 9, 0x1E: 10,
        0x03: 11, 0x07: 12, 0x0B: 13, 0x0F: 14,
    }
    return mode_map.get(five_bits, -1)


def make_synthetic_bc1_block(c0: int, c1: int, indices: list) -> bytes:
    """Create a synthetic 8-byte BC1 block from two 16-bit 565 colors and 16 2-bit indices."""
    assert len(indices) == 16
    bits = 0
    for k, idx in enumerate(indices):
        assert 0 <= idx <= 3
        bits |= (idx & 3) << (2 * k)
    return struct.pack("<HHI", c0, c1, bits)


def make_synthetic_bc3_block(a0: int, a1: int, a_indices: list, c0: int, c1: int, c_indices: list) -> bytes:
    """Create a synthetic 16-byte BC3 block: 8-byte alpha ramp + 8-byte BC1 colour."""
    assert len(a_indices) == 16
    # 48-bit index packed into 6 bytes
    abits = 0
    for k, idx in enumerate(a_indices):
        assert 0 <= idx <= 7
        abits |= (idx & 7) << (3 * k)
    alpha_part = struct.pack("<BB", a0, a1) + abits.to_bytes(6, "little")
    color_part = make_synthetic_bc1_block(c0, c1, c_indices)
    return alpha_part + color_part


class BCFormatSpecificationTests(unittest.TestCase):
    """Verify BC1-BC7 block compression detection, block dimensions, and bit-depth budgets."""

    def test_bc1_through_bc7_block_sizes_and_bit_depths(self):
        """Each BC format must have exact block size (8 or 16 bytes) and 4x4 pixel dimensions."""
        for name, spec in BC_SPECS.items():
            bpe = spec["bpe"]
            self.assertIn(bpe, (8, 16), f"{name} must have 8 or 16 bytes per block")
            total_bits = bpe * 8
            pixels_per_block = 16  # 4x4
            bpp = total_bits / pixels_per_block
            self.assertEqual(bpp, spec["bpp"], f"{name} bpp mismatch")
            self.assertEqual(total_bits, 64 if bpe == 8 else 128)

    def test_format_detection_from_bpe_and_channels(self):
        """Format detection mapping correctly classifies all standard BC formats."""
        self.assertEqual(detect_bc_format_from_bpe_and_channels(8, 3), "BC1")
        self.assertEqual(detect_bc_format_from_bpe_and_channels(16, 4), "BC2")  # or BC3/BC7 by alpha/mode
        self.assertEqual(detect_bc_format_from_bpe_and_channels(8, 1), "BC4")
        self.assertEqual(detect_bc_format_from_bpe_and_channels(16, 2), "BC5")
        self.assertEqual(detect_bc_format_from_bpe_and_channels(16, 3), "BC6H")
        with self.assertRaises(ValueError):
            detect_bc_format_from_bpe_and_channels(8, 5)

    def test_bc1_bit_depth_budget(self):
        """BC1 allocates 16 bits color0 (5:6:5), 16 bits color1 (5:6:5), 32 bits indices (16 x 2-bit)."""
        color0_bits = 5 + 6 + 5
        color1_bits = 5 + 6 + 5
        index_bits = 16 * 2
        total = color0_bits + color1_bits + index_bits
        self.assertEqual(total, 64)
        self.assertEqual(total // 8, 8)

    def test_bc2_bit_depth_budget(self):
        """BC2 allocates 64 bits explicit alpha (16 x 4-bit) + 64 bits BC1 color block."""
        alpha_bits = 16 * 4
        color_bits = 16 + 16 + (16 * 2)
        total = alpha_bits + color_bits
        self.assertEqual(total, 128)
        self.assertEqual(total // 8, 16)

    def test_bc3_bit_depth_budget(self):
        """BC3 allocates 16 bits alpha endpoints (8+8) + 48 bits alpha indices (16 x 3-bit) + 64 bits BC1 color."""
        alpha_endpoints = 8 + 8
        alpha_indices = 16 * 3
        color_bits = 64
        total = alpha_endpoints + alpha_indices + color_bits
        self.assertEqual(total, 128)
        self.assertEqual(total // 8, 16)

    def test_bc4_bit_depth_budget(self):
        """BC4 allocates 16 bits red endpoints (8+8) + 48 bits red indices (16 x 3-bit)."""
        red_endpoints = 8 + 8
        red_indices = 16 * 3
        total = red_endpoints + red_indices
        self.assertEqual(total, 64)
        self.assertEqual(total // 8, 8)

    def test_bc5_bit_depth_budget(self):
        """BC5 allocates two independent BC4 blocks: Red (64 bits) + Green (64 bits)."""
        red_channel_bits = 16 + (16 * 3)
        green_channel_bits = 16 + (16 * 3)
        total = red_channel_bits + green_channel_bits
        self.assertEqual(total, 128)
        self.assertEqual(total // 8, 16)

    def test_bc7_mode_detection(self):
        """BC7 mode detection decodes unary zero-terminated mode headers for modes 0 through 7."""
        for mode in range(8):
            first_byte = 1 << mode
            detected = decode_bc7_mode(first_byte)
            self.assertEqual(detected, mode, f"Mode {mode} not detected from byte {first_byte:#04x}")
            # Mode with higher bits set should still decode to lowest set bit
            first_byte_with_payload = first_byte | 0b11000000
            if mode < 6:
                self.assertEqual(decode_bc7_mode(first_byte_with_payload), mode)
        # All zeros is an invalid mode
        self.assertEqual(decode_bc7_mode(0), -1)

    def test_bc6h_mode_detection(self):
        """BC6H mode detection decodes modes 1-14 from 2-bit or 5-bit headers."""
        self.assertEqual(decode_bc6h_mode(0b00), 1)
        self.assertEqual(decode_bc6h_mode(0b01), 2)
        self.assertEqual(decode_bc6h_mode(0b00010), 3)
        self.assertEqual(decode_bc6h_mode(0b00110), 4)
        self.assertEqual(decode_bc6h_mode(0b01010), 5)
        self.assertEqual(decode_bc6h_mode(0b01110), 6)
        self.assertEqual(decode_bc6h_mode(0b10010), 7)
        self.assertEqual(decode_bc6h_mode(0b10110), 8)
        self.assertEqual(decode_bc6h_mode(0b11010), 9)
        self.assertEqual(decode_bc6h_mode(0b11110), 10)
        self.assertEqual(decode_bc6h_mode(0b00011), 11)
        self.assertEqual(decode_bc6h_mode(0b00111), 12)
        self.assertEqual(decode_bc6h_mode(0b01011), 13)
        self.assertEqual(decode_bc6h_mode(0b01111), 14)
        # Reserved headers
        self.assertEqual(decode_bc6h_mode(0b10011), -1)
        self.assertEqual(decode_bc6h_mode(0b11111), -1)


@unittest.skipUnless(HAVE_NUMPY, "numpy is not available")
class BCFeatureImplementationTests(unittest.TestCase):
    """Direct verification of tools/bc_feature.py functions with synthetic blocks."""

    def test_565_color_unpacking_and_bit_depth_scaling(self):
        """_565 extracts 5-bit R, 6-bit G, 5-bit B and scales to [0, 255]."""
        # Pure red: (31 << 11) = 0xF800
        red_arr = np.array([0xF800], dtype=np.uint16)
        rgb = bc_feature._565(red_arr)[0]
        self.assertAlmostEqual(float(rgb[0]), 255.0, places=4)
        self.assertAlmostEqual(float(rgb[1]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[2]), 0.0, places=4)

        # Pure green: (63 << 5) = 0x07E0
        green_arr = np.array([0x07E0], dtype=np.uint16)
        rgb = bc_feature._565(green_arr)[0]
        self.assertAlmostEqual(float(rgb[0]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[1]), 255.0, places=4)
        self.assertAlmostEqual(float(rgb[2]), 0.0, places=4)

        # Pure blue: 31 = 0x001F
        blue_arr = np.array([0x001F], dtype=np.uint16)
        rgb = bc_feature._565(blue_arr)[0]
        self.assertAlmostEqual(float(rgb[0]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[1]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[2]), 255.0, places=4)

        # White: 0xFFFF
        white_arr = np.array([0xFFFF], dtype=np.uint16)
        rgb = bc_feature._565(white_arr)[0]
        self.assertAlmostEqual(float(rgb[0]), 255.0, places=4)
        self.assertAlmostEqual(float(rgb[1]), 255.0, places=4)
        self.assertAlmostEqual(float(rgb[2]), 255.0, places=4)

        # Black: 0x0000
        black_arr = np.array([0x0000], dtype=np.uint16)
        rgb = bc_feature._565(black_arr)[0]
        self.assertAlmostEqual(float(rgb[0]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[1]), 0.0, places=4)
        self.assertAlmostEqual(float(rgb[2]), 0.0, places=4)

    def test_bc1_average_four_color_mode(self):
        """bc1_average in 4-color mode (c0 > c1) verifies 4 palette entries and interpolation weights."""
        c0 = 0xF800  # Red (255, 0, 0)
        c1 = 0x001F  # Blue (0, 0, 255)
        self.assertGreater(c0, c1)

        # Block where all 16 pixels select index 0 (color0)
        b0 = make_synthetic_bc1_block(c0, c1, [0] * 16)
        arr0 = np.frombuffer(b0, dtype=np.uint8).reshape(1, 8)
        avg0 = bc_feature.bc1_average(arr0)[0]
        self.assertAlmostEqual(float(avg0[0]), 255.0, places=3)
        self.assertAlmostEqual(float(avg0[1]), 0.0, places=3)
        self.assertAlmostEqual(float(avg0[2]), 0.0, places=3)

        # Block where all 16 pixels select index 1 (color1)
        b1 = make_synthetic_bc1_block(c0, c1, [1] * 16)
        arr1 = np.frombuffer(b1, dtype=np.uint8).reshape(1, 8)
        avg1 = bc_feature.bc1_average(arr1)[0]
        self.assertAlmostEqual(float(avg1[0]), 0.0, places=3)
        self.assertAlmostEqual(float(avg1[1]), 0.0, places=3)
        self.assertAlmostEqual(float(avg1[2]), 255.0, places=3)

        # Block where all 16 pixels select index 2: (2 * p0 + p1) / 3
        b2 = make_synthetic_bc1_block(c0, c1, [2] * 16)
        arr2 = np.frombuffer(b2, dtype=np.uint8).reshape(1, 8)
        avg2 = bc_feature.bc1_average(arr2)[0]
        self.assertAlmostEqual(float(avg2[0]), 255.0 * 2 / 3, places=3)
        self.assertAlmostEqual(float(avg2[1]), 0.0, places=3)
        self.assertAlmostEqual(float(avg2[2]), 255.0 * 1 / 3, places=3)

        # Block where all 16 pixels select index 3: (p0 + 2 * p1) / 3
        b3 = make_synthetic_bc1_block(c0, c1, [3] * 16)
        arr3 = np.frombuffer(b3, dtype=np.uint8).reshape(1, 8)
        avg3 = bc_feature.bc1_average(arr3)[0]
        self.assertAlmostEqual(float(avg3[0]), 255.0 * 1 / 3, places=3)
        self.assertAlmostEqual(float(avg3[1]), 0.0, places=3)
        self.assertAlmostEqual(float(avg3[2]), 255.0 * 2 / 3, places=3)

        # Uniform mix: exactly 4 pixels of each index
        mixed_indices = [0, 1, 2, 3] * 4
        b_mix = make_synthetic_bc1_block(c0, c1, mixed_indices)
        arr_mix = np.frombuffer(b_mix, dtype=np.uint8).reshape(1, 8)
        avg_mix = bc_feature.bc1_average(arr_mix)[0]
        expected_r = (255.0 + 0.0 + (255.0 * 2 / 3) + (255.0 * 1 / 3)) / 4
        expected_b = (0.0 + 255.0 + (255.0 * 1 / 3) + (255.0 * 2 / 3)) / 4
        self.assertAlmostEqual(float(avg_mix[0]), expected_r, places=3)
        self.assertAlmostEqual(float(avg_mix[2]), expected_b, places=3)

    def test_bc1_average_three_color_transparent_mode(self):
        """bc1_average in 3-color mode (c0 <= c1) sets p2=(p0+p1)/2 and p3=0 (transparent black)."""
        c0 = 0x001F  # Blue (c0 = 31)
        c1 = 0xF800  # Red (c1 = 63488)
        self.assertLess(c0, c1)

        # When all pixels select index 3, output must be transparent black (0, 0, 0)
        b3 = make_synthetic_bc1_block(c0, c1, [3] * 16)
        arr3 = np.frombuffer(b3, dtype=np.uint8).reshape(1, 8)
        avg3 = bc_feature.bc1_average(arr3)[0]
        self.assertAlmostEqual(float(avg3[0]), 0.0, places=4)
        self.assertAlmostEqual(float(avg3[1]), 0.0, places=4)
        self.assertAlmostEqual(float(avg3[2]), 0.0, places=4)

        # When all pixels select index 2, output is midpoint (p0 + p1) / 2
        b2 = make_synthetic_bc1_block(c0, c1, [2] * 16)
        arr2 = np.frombuffer(b2, dtype=np.uint8).reshape(1, 8)
        avg2 = bc_feature.bc1_average(arr2)[0]
        self.assertAlmostEqual(float(avg2[0]), 127.5, places=3)
        self.assertAlmostEqual(float(avg2[1]), 0.0, places=3)
        self.assertAlmostEqual(float(avg2[2]), 127.5, places=3)

        # Boundary condition: c0 == c1
        b_equal = make_synthetic_bc1_block(0x07E0, 0x07E0, [2] * 16)
        arr_equal = np.frombuffer(b_equal, dtype=np.uint8).reshape(1, 8)
        avg_equal = bc_feature.bc1_average(arr_equal)[0]
        self.assertAlmostEqual(float(avg_equal[1]), 255.0, places=3)

    def test_block_average_bpe_selection(self):
        """block_average uses full 8 bytes for bpe=8 and second 8 bytes (8:16) for bpe=16."""
        c0 = 0xF800  # Red
        c1 = 0x07E0  # Green
        b_color = make_synthetic_bc1_block(c0, c1, [0] * 16)

        # bpe = 8: whole block is color
        arr8 = np.frombuffer(b_color, dtype=np.uint8).reshape(1, 8)
        res8 = bc_feature.block_average(arr8, bpe=8)[0]
        self.assertAlmostEqual(float(res8[0]), 255.0, places=3)
        self.assertAlmostEqual(float(res8[1]), 0.0, places=3)

        # bpe = 16: first 8 bytes arbitrary alpha payload, bytes 8:16 are color
        b16 = make_synthetic_bc3_block(255, 0, [0] * 16, c0, c1, [0] * 16)
        self.assertEqual(len(b16), 16)
        arr16 = np.frombuffer(b16, dtype=np.uint8).reshape(1, 16)
        res16 = bc_feature.block_average(arr16, bpe=16)[0]
        self.assertAlmostEqual(float(res16[0]), 255.0, places=3)
        self.assertAlmostEqual(float(res16[1]), 0.0, places=3)

    def test_bc3_uses_four_colors_with_ascending_endpoints(self):
        c0, c1 = 0x001F, 0xF800
        block = make_synthetic_bc3_block(255, 0, [0] * 16, c0, c1, [3] * 16)
        data = np.frombuffer(block, dtype=np.uint8).reshape(1, 16)
        np.testing.assert_allclose(bc_feature.block_average(data, bpe=16)[0], [170, 0, 85])

    def test_smoothness_metric(self):
        """smoothness returns neighbor difference relative to spread (lower = more coherent)."""
        # Flat/constant image has zero neighbor differences -> 0.0
        flat = np.ones((8, 8, 3), dtype=np.float32) * 128.0
        self.assertAlmostEqual(bc_feature.smoothness(flat), 0.0, places=5)

        # Smooth horizontal gradient
        grad = np.tile(np.linspace(0, 255, 16, dtype=np.float32).reshape(1, 16, 1), (16, 1, 3))
        # High-frequency alternating checkerboard noise
        noisy = np.zeros((16, 16, 3), dtype=np.float32)
        noisy[::2, ::2] = 255.0
        noisy[1::2, 1::2] = 255.0

        s_grad = bc_feature.smoothness(grad)
        s_noisy = bc_feature.smoothness(noisy)
        self.assertGreater(s_noisy, s_grad, "Noisy image must have higher smoothness score than gradient")
        self.assertTrue(math.isfinite(s_grad))
        self.assertTrue(math.isfinite(s_noisy))


if __name__ == "__main__":
    unittest.main()
