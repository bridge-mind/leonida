"""The Rust port of the lane model accepts exactly the streams Python accepts, with the same symbols.

Builds BCPack chunks with the synthetic encoder of `test_bcpack.py` (no game
files), some of them corrupted or cut short, and decodes each one both ways.
Skipped unless `tools/bcpack_calibration` has a release build:

    cargo build --release --locked --offline --manifest-path tools/bcpack_calibration/Cargo.toml
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, HERE)

import bcpack  # noqa: E402
import test_bcpack as synth  # noqa: E402


def _python(raw, codec, size):
    try:
        lanes, used = bcpack._decode_lanes_raw(memoryview(raw), bcpack.LANE_WIDTHS[codec], size // 8)
    except (bcpack.BcpackError, ZeroDivisionError, IndexError):
        return None
    return None if any(memoryview(raw)[used:]) else (lanes, used)


def _cases(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        codec = rng.choice(sorted(bcpack.LANE_WIDTHS))
        size = rng.choice([1, 2, 7, 64]) * bcpack.BLOCK_BYTES[codec]
        lanes = []
        for w in bcpack.LANE_WIDTHS[codec]:
            rate = rng.random() + 0.05
            lanes.append([min((1 << w) - 1, int(rng.expovariate(rate))) if rng.random() < 0.9
                          else rng.randrange(1 << w) for _ in range(size // 8)])
        q = rng.randrange(16) if rng.random() < 0.7 else 0
        raw = bytearray(synth.encode_lanes(lanes, codec, q=q))
        kind = rng.random()
        if kind < 0.25:
            for _ in range(rng.randint(1, 3)):
                raw[rng.randrange(len(raw))] = rng.randrange(256)
        elif kind < 0.35:
            del raw[rng.randrange(len(raw)):]
        elif kind < 0.4:
            raw += bytes([rng.randrange(1, 256)])
        yield codec, size, bytes(raw), lanes if kind >= 0.4 else None


@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeLaneDecoderTest(unittest.TestCase):
    def test_native_accepts_exactly_what_python_accepts(self):
        accepted = refused = 0
        for codec, size, raw, lanes in _cases(20261123, 160):
            with self.subTest(codec=codec, size=size, raw=raw.hex()[:32]):
                fast = bcpack._decode_lanes_native(raw, bcpack.LANE_WIDTHS[codec], size // 8)
                slow = _python(raw, codec, size)
                self.assertEqual(fast, slow)
                if lanes is not None and slow is not None:
                    self.assertEqual(slow[0], lanes)
                accepted += fast is not None
                refused += fast is None
        self.assertGreater(accepted, 40)
        self.assertGreater(refused, 20)

    def test_uniform_model_and_virtual_zeros(self):
        # q = 0 fragments end on a zero byte the encoder leaves out; the model
        # shifts it back in (docs/textures.md, flush = 1). third_party/bcpack-rs,
        # which follows the hardware RTL, refuses these.
        widths = bcpack.LANE_WIDTHS[4]
        for sym in (0, 1, 3, 63):
            lanes = [[min(sym, (1 << w) - 1)] * 16 for w in widths]
            raw = synth.encode_lanes(lanes, 4, q=0)
            fast = bcpack._decode_lanes_native(raw, widths, 16)
            self.assertIsNotNone(fast)
            self.assertEqual(fast, _python(raw, 4, 128))
            self.assertEqual(fast[0], lanes)

    def test_decode_lanes_falls_back_and_raises_like_python(self):
        for codec, size, raw, _ in _cases(7, 60):
            with self.subTest(codec=codec, size=size):
                slow = _python(raw, codec, size)
                if slow is None:
                    with self.assertRaises(bcpack.BcpackError):
                        bcpack.decode_lanes(raw, codec, size)
                else:
                    self.assertEqual(bcpack.decode_lanes(raw, codec, size), slow)

    def test_switch_off(self):
        saved = list(bcpack._NATIVE), os.environ.get("LEONIDA_BCPACK_NATIVE")
        try:
            bcpack._NATIVE.clear()
            os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
            self.assertIsNone(bcpack._native_unpack())
        finally:
            bcpack._NATIVE[:] = saved[0]
            if saved[1] is None:
                os.environ.pop("LEONIDA_BCPACK_NATIVE", None)
            else:
                os.environ["LEONIDA_BCPACK_NATIVE"] = saved[1]


def _python_chunk(raw, codec, size, provisional, history=None):
    """`decode_with_info` with both native paths switched off, or the exception it raises."""
    saved = bcpack._decode_blocks_native, bcpack._decode_lanes_native
    bcpack._decode_blocks_native = lambda *a: None
    bcpack._decode_lanes_native = lambda *a: None
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional, history=history)
    except Exception as exc:  # noqa: BLE001 - the comparison includes the error
        return type(exc), str(exc)
    finally:
        bcpack._decode_blocks_native, bcpack._decode_lanes_native = saved


def _chunk(raw, codec, size, provisional, history=None):
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional, history=history)
    except Exception as exc:  # noqa: BLE001
        return type(exc), str(exc)


def _split_distance(d):
    """The two 4-bit lane-0 symbols of a BC3/BC5 block that carry history distance `d` (0..255)."""
    code = d << 1 if d <= 127 else ((256 - d) << 1) - 1
    return (sum(((code >> 2 * k) & 1) << k for k in range(4)),
            sum(((code >> 2 * k + 1) & 1) << k for k in range(4)))


def _zig(v):
    return v << 1 if v >= 0 else ((-v) << 1) - 1


def _random_two_element_lanes(rng, codec, n, entries, builtin):
    """Random lanes of `n` BC3 or BC5 blocks (two symbols per block).

    With `builtin`, some distances reach into the built-in history (the entries in `entries` and wild ones).
    """
    widths = bcpack.LANE_WIDTHS[codec]
    lanes = [[rng.randrange(1 << w) if rng.random() < 0.5 else min((1 << w) - 1, int(rng.expovariate(0.3)))
              for _ in range(2 * n)] for w in widths]
    lanes[1] = [0] * (2 * n) if rng.random() < 0.9 else [rng.randrange(4) for _ in range(2 * n)]
    for i in range(n):
        r = rng.random()
        d = 0
        if builtin and r >= 0.9:
            d = rng.randint(1, 255)
        elif builtin and r >= 0.7 and entries:
            d = 256 + i - rng.choice(entries)
            d = d if 1 <= d <= 255 else 0
        elif r >= 0.3 and i:
            d = rng.randint(1, min(i, 20))
        lanes[0][2 * i], lanes[0][2 * i + 1] = _split_distance(d)
    return lanes


def _encode_two_element_blocks(codec, blocks, ref_dists):
    """A codec 3 or 5 chunk for raw 16-byte blocks, each predicted from the block `ref_dists[i]` back (0: none)."""
    inv_pair = {v: i for i, v in enumerate(bcpack._tables.BC3_ALPHA_PAIR if codec == 3 else bcpack._tables.BC5_PAIR)}
    inv_quad = {v: i for i, v in enumerate(bcpack._tables.BC3_COLOUR_QUADRANT)}
    mods = (256, 256, 32, 64, 32, 32, 64, 32) if codec == 3 else (256,) * 4
    none = (128, 128, 0, 0, 0, 0, 0, 0) if codec == 3 else (128,) * 4
    n = len(blocks) // 16
    lanes = [[] for _ in bcpack.LANE_WIDTHS[codec]]

    def pairs(index_bytes):  # six bytes of 3-bit indices -> row symbols of the left and right element
        bits = int.from_bytes(index_bytes, "little")
        out = []
        for r in range(4):
            row = bits >> 12 * r
            out.append((inv_pair[(row & 7, (row >> 3) & 7)], inv_pair[((row >> 6) & 7, (row >> 9) & 7)]))
        return out

    hist = []
    for i in range(n):
        b = blocks[i * 16:i * 16 + 16]
        if codec == 3:
            c0, c1 = int.from_bytes(b[8:10], "little"), int.from_bytes(b[10:12], "little")
            e = (b[0], b[1], c0 >> 11, (c0 >> 5) & 63, c0 & 31, c1 >> 11, (c1 >> 5) & 63, c1 & 31)
        else:
            e = (b[0], b[1], b[8], b[9])
        d = ref_dists[i]
        base = none if d == 0 else hist[i - d]
        hist.append(e)
        a_sym, b_sym = _split_distance(d)
        lanes[0] += [a_sym, b_sym]
        lanes[1] += [0, 0]
        delta = []
        for k in range(len(e)):
            v = (e[k] - base[k]) % mods[k]
            delta.append(_zig(v - mods[k] if v >= mods[k] // 2 else v))
        if codec == 3:
            lanes[2] += [delta[0], delta[1]]
            for lane, (left, right) in zip((7, 8, 9), ((2, 5), (3, 6), (4, 7))):
                lanes[lane] += [delta[left], delta[right]]
        else:
            lanes[2] += [delta[0], delta[1]]
            lanes[3] += [delta[2], delta[3]]
        if codec == 3:
            rows = pairs(b[2:8])
            for r in range(4):
                lanes[3 + r] += list(rows[r])
            q = [b[12] & 15 | (b[13] & 15) << 4, b[12] >> 4 | (b[13] >> 4) << 4,
                 b[14] & 15 | (b[15] & 15) << 4, b[14] >> 4 | (b[15] >> 4) << 4]
            lanes[10] += [inv_quad[q[0]], inv_quad[q[1]]]
            lanes[11] += [inv_quad[q[2]], inv_quad[q[3]]]
        else:
            for ch in range(2):
                rows = pairs(b[8 * ch + 2:8 * ch + 8])
                for r in range(4):
                    lanes[4 + 4 * ch + r] += list(rows[r])
    return synth.encode_lanes(lanes, codec)



@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeBlockRebuildTest(unittest.TestCase):
    """BC1, BC3, BC4 and BC5 chunks give the same bytes, provisional entries and errors with the native rebuild."""

    def test_bc1_bc4_match_python(self):
        rng = random.Random(4242)
        native = 0
        for _ in range(120):
            codec = rng.choice([1, 4])
            n = rng.choice([1, 3, 64, 300])
            widths = bcpack.LANE_WIDTHS[codec]
            lanes = [[rng.randrange(1 << w) if rng.random() < 0.5 else min((1 << w) - 1, int(rng.expovariate(0.3)))
                      for _ in range(n)] for w in widths]
            lanes[1] = [0] * n if rng.random() < 0.9 else [rng.randrange(4) for _ in range(n)]
            for i in range(n):
                r = rng.random()
                d = 0 if r < 0.3 else rng.randint(1, max(1, min(i, 20))) if i and r < 0.9 else rng.randint(1, 255)
                lanes[0][i] = (d << 1) if d <= 127 else ((256 - d) << 1) - 1
            raw = bytearray(synth.encode_lanes(lanes, codec, q=rng.randrange(16)))
            if rng.random() < 0.15:
                raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
            raw = bytes(raw)
            provisional = rng.random() < 0.4
            with self.subTest(codec=codec, n=n, provisional=provisional):
                self.assertEqual(_chunk(raw, codec, n * 8, provisional),
                                 _python_chunk(raw, codec, n * 8, provisional))
            native += bcpack._decode_blocks_native(raw, codec, n * 8, None) is not None
        self.assertGreater(native, 20, "the native rebuild should take many of these chunks")

    def test_bc1_round_trip(self):
        rng = random.Random(5)
        blocks = bytes(rng.randrange(256) for _ in range(8 * 256))
        raw = synth.encode_bc1_blocks(blocks)
        self.assertEqual(bcpack._decode_blocks_native(raw, 1, len(blocks), None), blocks)


    def test_bc3_bc5_match_python(self):
        rng = random.Random(3535)
        native = {3: 0, 5: 0}
        seen = {3: 0, 5: 0}
        for _ in range(240):
            codec = rng.choice([3, 5])
            n = rng.choice([1, 3, 64, 300])
            entries = sorted(set(getattr(bcpack._tables, f"BC{codec}_HISTORY_CONFIRMED"))
                             | set(getattr(bcpack._tables, f"BC{codec}_HISTORY_PROVISIONAL")))
            lanes = _random_two_element_lanes(rng, codec, n, entries, rng.random() < 0.5)
            raw = bytearray(synth.encode_lanes(lanes, codec, q=rng.randrange(16)))
            if rng.random() < 0.15:
                raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
            elif rng.random() < 0.05:
                del raw[rng.randrange(len(raw)):]
            raw = bytes(raw)
            provisional = rng.random() < 0.4
            with self.subTest(codec=codec, n=n, provisional=provisional):
                got = _chunk(raw, codec, n * 16, provisional)
                self.assertEqual(got, _python_chunk(raw, codec, n * 16, provisional))
            fast = bcpack._decode_blocks_native(raw, codec, n * 16, None)
            seen[codec] += 1
            native[codec] += fast is not None
            if fast is not None:  # accepted natively: Python accepts it too, in strict mode, with the same bytes
                self.assertEqual(_python_chunk(raw, codec, n * 16, False), (fast, set()))
        for codec in (3, 5):
            self.assertGreater(native[codec], 15, f"the native rebuild should take many BC{codec} chunks")
            self.assertGreater(seen[codec] - native[codec], 15, f"BC{codec} should have refused chunks too")

    def test_bc3_bc5_round_trip(self):
        rng = random.Random(77)
        for codec in (3, 5):
            for n in (1, 2, 17, 256, 600):
                blocks = bytes(rng.randrange(256) for _ in range(16 * n))
                dists = [0 if i == 0 or rng.random() < 0.3 else rng.randint(1, min(i, 40)) for i in range(n)]
                raw = _encode_two_element_blocks(codec, blocks, dists)
                with self.subTest(codec=codec, n=n):
                    self.assertEqual(bcpack._decode_blocks_native(raw, codec, len(blocks), None), blocks)
                    self.assertEqual(bcpack.decode(raw, codec, len(blocks)), blocks)
                    self.assertEqual(_python_chunk(raw, codec, len(blocks), False), (blocks, set()))

    def test_bc3_bc5_refuse_what_python_decides(self):
        rng = random.Random(8)
        for codec in (3, 5):
            blocks = bytes(rng.randrange(256) for _ in range(16 * 40))
            raw = _encode_two_element_blocks(codec, blocks, [0] * 40)
            # a non-zero lane 1 is Python's error to raise
            lanes = _random_two_element_lanes(rng, codec, 8, [], False)
            lanes[1] = [0] * 15 + [1]
            bad = synth.encode_lanes(lanes, codec)
            self.assertIsNone(bcpack._decode_blocks_native(bad, codec, 8 * 16, None))
            for provisional in (False, True):
                got = _chunk(bad, codec, 8 * 16, provisional)
                self.assertEqual(got, _python_chunk(bad, codec, 8 * 16, provisional))
                self.assertIs(got[0], bcpack.BcpackError)
                self.assertIn("lane 1 is non-zero", got[1])
            # sizes that are not a whole number of blocks
            self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 16 * 40 + 8, None))
            self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 0, None))
            # a corrupted or cut-off chunk
            for broken in (raw[:len(raw) // 2], raw + b"\x01", bytes(len(raw))):
                self.assertIsNone(bcpack._decode_blocks_native(broken, codec, len(blocks), None))
                for provisional in (False, True):
                    self.assertEqual(_chunk(broken, codec, len(blocks), provisional),
                                     _python_chunk(broken, codec, len(blocks), provisional))

    def test_bc3_bc5_history_entries(self):
        def one_block(codec, entry):
            lanes = [[0, 0] for _ in bcpack.LANE_WIDTHS[codec]]
            lanes[0] = list(_split_distance(256 - entry))  # block 0 reaches back to built-in entry `entry`
            return synth.encode_lanes(lanes, codec)

        def table(codec, kind):
            return set(getattr(bcpack._tables, f"BC{codec}_HISTORY_{kind}"))

        confirmed = sorted(table(5, "CONFIRMED"))[0]
        loose = {c: sorted(table(c, "PROVISIONAL") - table(c, "CONFIRMED"))[0] for c in (3, 5)}
        unknown = {c: next(e for e in range(1, 256) if e not in table(c, "CONFIRMED") | table(c, "PROVISIONAL"))
                   for c in (3, 5)}
        # confirmed entry: the native path takes it, nothing is provisional
        raw = one_block(5, confirmed)
        fast = bcpack._decode_blocks_native(raw, 5, 16, None)
        self.assertIsNotNone(fast)
        self.assertEqual(_python_chunk(raw, 5, 16, False), (fast, set()))
        for codec in (3, 5):
            for label, entry in (("unconfirmed", loose[codec]), ("unknown", unknown[codec])):
                raw = one_block(codec, entry)
                with self.subTest(codec=codec, kind=label):
                    self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 16, None))
                    strict = _chunk(raw, codec, 16, False)
                    self.assertEqual(strict, _python_chunk(raw, codec, 16, False))
                    self.assertIs(strict[0], bcpack.BcpackProvisionalError)
                    loosely = _chunk(raw, codec, 16, True)
                    self.assertEqual(loosely, _python_chunk(raw, codec, 16, True))
                    self.assertTrue(loosely[1], "provisional mode records the entry it used")

    def test_bc3_bc5_external_history(self):
        rng = random.Random(1212)
        for codec, mods in ((3, (256, 256, 32, 64, 32, 32, 64, 32)), (5, (256,) * 4)):
            for replace in (False, True):
                history = {f"bc{codec}": {"confirmed": {"9": [rng.randrange(m) for m in mods],
                                                        "200": [7 % m for m in mods]},
                                          "replace": replace}}
                accepted = 0
                for _ in range(20):
                    n = rng.choice([1, 5, 40])
                    lanes = _random_two_element_lanes(rng, codec, n, [9, 200, 105, 2], True)
                    lanes[1] = [0] * (2 * n)
                    raw = synth.encode_lanes(lanes, codec, q=rng.randrange(16))
                    for provisional in (False, True):
                        with self.subTest(codec=codec, replace=replace, n=n, provisional=provisional):
                            self.assertEqual(_chunk(raw, codec, n * 16, provisional, history),
                                             _python_chunk(raw, codec, n * 16, provisional, history))
                            self.assertEqual(_chunk(raw, codec, n * 16, provisional),
                                             _python_chunk(raw, codec, n * 16, provisional))
                    accepted += bcpack._decode_blocks_native(raw, codec, n * 16, history) is not None
                self.assertGreater(accepted, 0)
        # a history the loader rejects is raised after the lanes, as in Python: a broken chunk still gives BcpackError
        broken = {"bc3": {"confirmed": {"9": [999] * 8}}}
        raw = synth.encode_lanes(_random_two_element_lanes(rng, 3, 4, [], False), 3)
        for chunk in (raw, raw[:5]):
            self.assertEqual(_chunk(chunk, 3, 64, False, broken), _python_chunk(chunk, 3, 64, False, broken))
        self.assertIsNone(bcpack._decode_blocks_native(raw, 3, 64, broken))

    def test_bc3_bc5_negative_controls(self):
        """The comparison notices a wrong lookup table, and the switch turns the native rebuild off."""
        rng = random.Random(99)
        swaps = (
            (3, lambda t: t[:128] + t[129:130] + t[128:129] + t[130:]),  # two colour quadrant rows
            (3, lambda t: t[2:4] + t[:2] + t[4:]),  # two alpha pair rows
            (5, lambda t: t[2:4] + t[:2] + t[4:]),  # two pair rows
        )
        for codec, swap in swaps:
            blocks = bytes(rng.randrange(256) for _ in range(16 * 64))
            raw = _encode_two_element_blocks(codec, blocks, [0] * 64)
            spec = bcpack._NATIVE_BLOCKS[codec]
            saved = dict(bcpack._NATIVE_BUILTIN)
            try:
                bcpack._NATIVE_BLOCKS[codec] = (lambda lut=spec[0]: swap(lut()),) + spec[1:]
                bcpack._NATIVE_BUILTIN.pop(codec, None)
                wrong = bcpack._decode_blocks_native(raw, codec, len(blocks), None)
            finally:
                bcpack._NATIVE_BLOCKS[codec] = spec
                bcpack._NATIVE_BUILTIN.clear()
                bcpack._NATIVE_BUILTIN.update(saved)
            self.assertIsNotNone(wrong)
            self.assertNotEqual(wrong, blocks, "a swapped lookup row must show up in the output")
            self.assertEqual(bcpack._decode_blocks_native(raw, codec, len(blocks), None), blocks)
        saved = list(bcpack._NATIVE), os.environ.get("LEONIDA_BCPACK_NATIVE")
        try:
            bcpack._NATIVE.clear()
            os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
            for codec in (3, 5):
                raw = _encode_two_element_blocks(codec, blocks, [0] * 64)
                self.assertIsNone(bcpack._decode_blocks_native(raw, codec, len(blocks), None))
                self.assertEqual(bcpack.decode(raw, codec, len(blocks)), blocks)
        finally:
            bcpack._NATIVE[:] = saved[0]
            if saved[1] is None:
                os.environ.pop("LEONIDA_BCPACK_NATIVE", None)
            else:
                os.environ["LEONIDA_BCPACK_NATIVE"] = saved[1]


if __name__ == "__main__":
    unittest.main()
