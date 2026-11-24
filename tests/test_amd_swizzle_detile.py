"""amd_swizzle.offsets / detile give the same bytes as the per-bit parity reference they replaced.

    python -I tests/test_amd_swizzle_detile.py
    python -m pytest tests/test_amd_swizzle_detile.py

The equations are hand-built (no addrlib header, no game data): an S layout (interleaved x/y bits), a D_X layout
(pipe bits XORed with coordinate bits above the in-block ones) and an R_X layout (every address bit XORed with
high coordinate bits), for every element size and for the 256 B / 4 KB / 64 KB block sizes.
"""
import os
import random
import sys
import unittest

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    import numpy as np
    import amd_swizzle as A
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    np = A = None


def reference_offsets(eq, bpe, wb, hb, block_bytes=65536):
    """The pre-optimisation implementation: per address bit, parity over the whole grid."""
    y, x = np.mgrid[0:hb, 0:wb].astype(np.int64)
    off = np.zeros_like(x)
    for bit, (mx, my) in enumerate(eq):
        if mx or my:
            off |= (A._parity(x & mx) ^ A._parity(y & my)) << bit
    xb = max((mx.bit_length() for mx, _ in eq if mx), default=0)
    yb = max((my.bit_length() for _, my in eq if my), default=0)
    elems = block_bytes // bpe
    lb = elems.bit_length() - 1
    bw = 1 << ((lb + 1) // 2)
    bh = elems // bw
    blocks_x = (wb + bw - 1) // bw
    block = (y // bh) * blocks_x + (x // bw)
    return off + block * block_bytes, (bw, bh, xb, yb)


def reference_detile(data, eq, bpe, wb, hb, block_bytes=65536):
    off, _ = reference_offsets(eq, bpe, wb, hb, block_bytes)
    idx = off // bpe
    raw = np.frombuffer(data[: len(data) // bpe * bpe], np.uint8).reshape(-1, bpe)
    if idx.max() >= len(raw):
        return None
    return raw[idx.reshape(-1)]


def make_equation(mode, bpe, bits):
    """One (x mask, y mask) per address bit; the bits below the element size address bytes inside an element."""
    low = bpe.bit_length() - 1
    eq, xi, yi = [], 0, 0
    for bit in range(bits):
        if bit < low:
            eq.append((0, 0))
        elif (bit - low) % 2 == 0:
            eq.append((1 << xi, 0))
            xi += 1
        else:
            eq.append((0, 1 << yi))
            yi += 1
    if mode == "D_X":
        for bit in range(low, min(bits, low + 6)):
            mx, my = eq[bit]
            eq[bit] = (mx ^ (1 << (xi + bit % 3)), my ^ (1 << (yi + bit % 2)))
    elif mode == "R_X":
        for bit in range(low, bits):
            mx, my = eq[bit]
            eq[bit] = (mx ^ (1 << (xi + bit % 4)) ^ (1 << (xi + 1 + bit % 3)), my ^ (1 << (yi + bit % 5)) ^ (1 << (yi + 2)))
    return eq


def random_data(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# Narrow, tall, single-row/column, and sizes that are not multiples of any block extent.
SIZES = [(1, 1), (1, 70), (70, 1), (3, 5), (37, 5), (5, 37), (64, 64), (100, 37), (129, 65), (130, 3), (257, 513)]
BLOCKS = {256: 8, 4096: 12, 65536: 16}


@unittest.skipIf(np is None, "numpy is not installed")
class DetileMatchesReference(unittest.TestCase):
    def check(self, eq, bpe, wb, hb, block_bytes, data=None):
        label = (bpe, wb, hb, block_bytes)
        want_off, want_meta = reference_offsets(eq, bpe, wb, hb, block_bytes)
        got_off, got_meta = A.offsets(eq, bpe, wb, hb, block_bytes)
        self.assertEqual(got_meta, want_meta, label)
        self.assertEqual(got_off.shape, (hb, wb), label)
        self.assertEqual(got_off.dtype, np.int64, label)
        np.testing.assert_array_equal(got_off, want_off, str(label))
        if data is None:
            data = random_data(int(want_off.max()) + bpe + 3, wb * 7 + hb)
        want = reference_detile(data, eq, bpe, wb, hb, block_bytes)
        got = A.detile(data, eq, bpe, wb, hb, block_bytes)
        if want is None:
            self.assertIsNone(got, label)
        else:
            self.assertEqual(got.shape, want.shape, label)
            self.assertEqual(got.dtype, np.uint8, label)
            np.testing.assert_array_equal(got, want, str(label))
        return want

    def test_modes_bpe_sizes_and_block_sizes(self):
        for block_bytes, bits in BLOCKS.items():
            for bpe in (1, 2, 4, 8, 16):
                if bpe > block_bytes:
                    continue
                for mode in ("S", "D_X", "R_X"):
                    eq = make_equation(mode, bpe, bits)
                    for wb, hb in SIZES:
                        with self.subTest(block=block_bytes, bpe=bpe, mode=mode, size=(wb, hb)):
                            self.check(eq, bpe, wb, hb, block_bytes)

    def test_realistic_texture_sizes(self):
        # BC1 / BC7 are 4x4-pixel elements; RGBA8 is one pixel per element.
        for bpe, pixels_per_element in ((8, 4), (16, 4), (4, 1)):
            for px in (512, 1024):
                n = px // pixels_per_element
                for mode in ("S", "D_X", "R_X"):
                    with self.subTest(bpe=bpe, px=px, mode=mode):
                        self.check(make_equation(mode, bpe, 16), bpe, n, n, 65536)

    def test_random_equations(self):
        rng = random.Random(20260807)
        for _ in range(60):
            bpe = rng.choice((1, 2, 4, 8, 16))
            bits = rng.choice((8, 12, 16))
            block_bytes = 1 << bits
            if bpe > block_bytes:
                continue
            eq = [(rng.getrandbits(16) if rng.random() < 0.8 else 0, rng.getrandbits(16) if rng.random() < 0.8 else 0)
                  for _ in range(bits)]
            wb, hb = rng.randint(1, 200), rng.randint(1, 200)
            with self.subTest(bpe=bpe, bits=bits, size=(wb, hb)):
                self.check(eq, bpe, wb, hb, block_bytes)

    def test_too_short_data_returns_none(self):
        eq = make_equation("R_X", 4, 16)
        off, _ = A.offsets(eq, 4, 100, 37)
        need = int(off.max()) // 4 * 4 + 4
        for size in (0, 3, need - 4, need - 1):
            with self.subTest(size=size):
                self.assertIsNone(self.check(eq, 4, 100, 37, 65536, data=bytes(size)))
        self.assertIsNotNone(self.check(eq, 4, 100, 37, 65536, data=bytes(need)))

    def test_trailing_partial_element_is_ignored(self):
        eq = make_equation("D_X", 8, 16)
        off, _ = A.offsets(eq, 8, 9, 9)
        data = random_data(int(off.max()) + 8 + 5, 5)
        for cut in range(0, 8):
            self.check(eq, 8, 9, 9, 65536, data=data[: len(data) - cut])

    def test_accepts_bytearray_and_memoryview(self):
        eq = make_equation("R_X", 16, 16)
        data = random_data(65536 * 3, 9)
        want = A.detile(data, eq, 16, 70, 33)
        for wrapper in (bytearray, memoryview):
            np.testing.assert_array_equal(A.detile(wrapper(data), eq, 16, 70, 33), want)

    def test_element_size_not_a_power_of_two_or_not_dividing_the_block(self):
        eq = make_equation("S", 4, 12)
        for bpe, block_bytes in ((3, 4096), (3, 65536), (6, 4096), (12, 65536), (5, 256)):
            for wb, hb in ((1, 1), (7, 9), (40, 33)):
                with self.subTest(bpe=bpe, block=block_bytes, size=(wb, hb)):
                    self.check(eq, bpe, wb, hb, block_bytes, data=random_data(1 << 21, bpe))

    def test_round_trip_tiles_then_detiles(self):
        for block_bytes, bits in ((4096, 12), (65536, 16)):
            for bpe in (1, 2, 4, 8, 16):
                eq = make_equation("S", bpe, bits)  # an S layout is a bijection within each block
                wb, hb = 97, 53
                linear = np.random.default_rng(bpe).integers(0, 256, (hb * wb, bpe), dtype=np.uint8)
                off, _ = A.offsets(eq, bpe, wb, hb, block_bytes)
                self.assertEqual(len(np.unique(off)), off.size)
                tiled = np.zeros((int(off.max()) // bpe + 1, bpe), np.uint8)
                tiled[off.reshape(-1) // bpe] = linear
                got = A.detile(tiled.tobytes(), eq, bpe, wb, hb, block_bytes)
                np.testing.assert_array_equal(got, linear)


if __name__ == "__main__":
    unittest.main()
