"""Tests for tools/bcpack.py, bcpack_tables.py and bcpack_tables_bptc.py.

Builds genuine BCPack rANS streams from scratch with a self-contained test
encoder — no game files or texture bytes are used.

    python -I tests/test_bcpack.py
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bcpack  # noqa: E402
import bcpack_tables as tables  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402


# --- Synthetic BCPack rANS encoder (test helper only) ----------------------

def _zig(v):
    """Inverse of bcpack._unzig for signed integer deltas."""
    return (-2 * v - 1) if v < 0 else (2 * v)


def _deint_n(v, n):
    """Inverse of bcpack._inter: split 2n-bit integer into two n-bit halves."""
    a = b = 0
    for k in range(n):
        a |= ((v >> (2 * k)) & 1) << k
        b |= ((v >> (2 * k + 1)) & 1) << k
    return a, b


def _deint4(v):
    """Inverse of bcpack._inter8: split 8-bit integer into two 4-bit nibbles."""
    return _deint_n(v, 4)


def _pack_ix_pair(d0, d1, y):
    """Inverse of the BC6H/BC7 row index de-interleave for pixels (2k, 2k+1) on row y."""
    return bcpack._inter(d1, d0, 4) if y < 3 else bcpack._inter(d0, d1, 4)


def _encode_rans_syms(syms, model, x):
    """Reverse-encode `syms` starting from end state `x`, returning `(new_x, bytes_reversed)`."""
    buf = []
    for s in reversed(syms):
        p = s.bit_length()
        f = model.freq[p]
        c = model.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if c == 0:
            f, c = f - 1, 1
        limit = f << 16
        while x >= limit:
            buf.append(x & 0xFF)
            x >>= 8
        q_div, r_mod = divmod(x, f)
        x = (q_div << 16) + c + r_mod
    return x, buf


MAX_FRAGMENT_LEN = 257 + 0xFFFF  # 65,792 bytes (3-byte header stores n - 257 in u16)


def _make_hdr(default_w, w, q, flush, n):
    z = default_w - w
    if not 0 <= z <= 3:
        raise ValueError(f"width reduction out of range: {z}")
    if not 1 <= n <= MAX_FRAGMENT_LEN:
        raise ValueError(f"fragment length {n} out of range 1..{MAX_FRAGMENT_LEN}")
    if n <= 256:
        return [((z & 3) << 6) | ((1 if flush else 0) << 4) | (q & 15), n - 1]
    n2 = n - 257
    return [
        ((z & 3) << 6) | (1 << 5) | ((1 if flush else 0) << 4) | (q & 15),
        n2 & 0xFF,
        (n2 >> 8) & 0xFF,
    ]


def _encode_fragment(syms, default_w, w, q, *, flush=True, strip_zeros=True):
    """Encode `syms` into one standalone flush fragment (header + rANS payload)."""
    assert flush
    model = bcpack._get_model(w, q)
    assert model is not None
    x, buf = _encode_rans_syms(syms, model, bcpack.RANS_L)
    payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
    if strip_zeros:
        stripped = 0
        while len(payload) > 1 and payload[-1] == 0 and stripped < bcpack.MAX_VIRTUAL_ZEROS:
            payload.pop()
            stripped += 1
    return bytes(_make_hdr(default_w, w, q, True, len(payload)) + payload), 0


def _encode_lane_stream(fragments, default_w):
    """Chain one or more `(syms, w, q, flush)` fragments for a single lane.

    The final fragment must have `flush=True`. For a non-flush fragment followed
    by another fragment, the boundary state is shifted below `RANS_L` so the
    decoder finishes fragment `i` at `st < RANS_L` and renormalizes from fragment `i+1`.
    """
    encoded = [None] * len(fragments)
    x = bcpack.RANS_L
    for idx in range(len(fragments) - 1, -1, -1):
        syms, w, q, flush = fragments[idx]
        model = bcpack._get_model(w, q)
        assert model is not None
        if flush:
            x = bcpack.RANS_L
        x, buf = _encode_rans_syms(syms, model, x)
        prev_flush = True if idx == 0 else fragments[idx - 1][3]
        if prev_flush:
            # This fragment starts from state 0: prepend all 4 initial state bytes.
            payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
            if flush:
                stripped = 0
                while len(payload) > 1 and payload[-1] == 0 and stripped < bcpack.MAX_VIRTUAL_ZEROS:
                    payload.pop()
                    stripped += 1
        else:
            # Previous fragment is non-flush: emit renormalization bytes into this
            # fragment until `x < RANS_L` so the previous fragment hands off `x < RANS_L`.
            while x >= bcpack.RANS_L:
                buf.append(x & 0xFF)
                x >>= 8
            payload = list(reversed(buf))
        encoded[idx] = bytes(_make_hdr(default_w, w, q, flush, len(payload)) + payload)
    return b"".join(encoded)


def interleave_lanes(lane_streams, widths, nsym):
    """Interleave per-lane byte streams according to the BCPack 8-byte/cycle scheduler."""
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = bcpack.RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [nsym <= 0] * nl
    full = [False] * nl
    held = [False] * nl
    vz = [0] * nl
    inbuf = [0] * nl
    sym = [0] * nl
    in_rem = [0] * nl
    p0 = [0] * nl
    p1 = [0] * nl
    model = [None] * nl
    slotfull = [False] * nl
    stall = [False] * nl
    lpos = [0] * nl
    interleaved = bytearray()
    max_cycles = max(1000, nsym * 64 + sum(len(s) for s in lane_streams) * 8)

    for _ in range(max_cycles):
        if all(done[i] and not slotfull[i] for i in range(nl)):
            for i in range(nl):
                assert lpos[i] == len(lane_streams[i]), f"lane {i} left {len(lane_streams[i]) - lpos[i]} bytes"
            return bytes(interleaved)
        left = bcpack.PREFETCH_BUDGET
        for i in range(nl):
            if done[i] or (held[i] and slotfull[i]):
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                b = lane_streams[i][lpos[i]]
                lpos[i] += 1
                interleaved.append(b)
                inbuf[i] = b
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                held[i] = False
        for i in range(nl):
            if done[i] or stall[i]:
                continue
            c = ctrl[i]
            if c == HDR0:
                if not full[i]:
                    continue
                h = inbuf[i]
                q[i] = h & 15
                flush[i] = (h >> 4) & 1
                h3[i] = (h >> 5) & 1
                w[i] = widths[i] - (h >> 6)
                full[i] = False
                ctrl[i] = HDR1
                vz[i] = 0
            elif c == HDR1:
                if not full[i]:
                    continue
                full[i] = False
                model[i] = bcpack._get_model(w[i], q[i])
                if h3[i]:
                    in_rem[i] = inbuf[i]
                    ctrl[i] = HDR2
                else:
                    in_rem[i] = inbuf[i] + 1
                    ctrl[i] = PROC
            elif c == HDR2:
                if not full[i]:
                    continue
                in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257
                ctrl[i] = PROC
                full[i] = False
            else:
                st = p0[i]
                if st < L:
                    if in_rem[i] > 0:
                        if full[i]:
                            st = (st << 8) | inbuf[i]
                            in_rem[i] -= 1
                            full[i] = False
                            if in_rem[i] == 0 and flush[i]:
                                while st < L and st != 0 and vz[i] < bcpack.MAX_VIRTUAL_ZEROS:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        while st < L and st != 0 and vz[i] < bcpack.MAX_VIRTUAL_ZEROS:
                            st <<= 8
                            vz[i] += 1
                    else:
                        p1[i] = st
                        if out_rem[i] > 0:
                            ctrl[i] = HDR0
                        else:
                            done[i] = True
                        continue
                nxt = st > L and out_rem[i] > 0
                if nxt:
                    sym[i] = model[i].lookup(st & 0xFFFF)
                    out_rem[i] -= 1
                held[i] = nxt
                p1[i] = st
                if in_rem[i] == 0:
                    if flush[i]:
                        if st == L:
                            if out_rem[i] > 0:
                                ctrl[i] = HDR0
                                p0[i] = 0
                                p1[i] = 0
                            else:
                                done[i] = True
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
        ready = True
        anyfull = False
        for i in range(nl):
            if slotfull[i]:
                anyfull = True
            elif not (done[i] and not held[i]):
                ready = False
                break
        if ready and anyfull:
            for i in range(nl):
                slotfull[i] = False
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                held[i] = False
    raise AssertionError("interleave_lanes exceeded max_cycles")


def encode_lanes(lanes, codec, *, q=0):
    """Encode per-lane symbol lists into a complete BCPack chunk."""
    widths = bcpack.LANE_WIDTHS[codec]
    nsym = len(lanes[0])
    streams = []
    for i, (syms, dw) in enumerate(zip(lanes, widths)):
        qi = q[i] if isinstance(q, (list, tuple)) else q
        frag, _ = _encode_fragment(syms, dw, dw, qi, flush=True)
        streams.append(frag)
    return interleave_lanes(streams, widths, nsym)


def encode_bc1_blocks(blocks, *, ref_dists=None):
    """Encode raw BC1 blocks (8 bytes each) into a valid codec-1 BCPack chunk."""
    n = len(blocks) // 8
    inv_quad = {v: i for i, v in enumerate(tables.BC1_QUADRANT)}
    mods = (32, 64, 32, 32, 64, 32)
    lanes = [[] for _ in range(12)]
    hist = []
    for i in range(n):
        c0, c1, r0, r1, r2, r3 = struct.unpack_from("<HHBBBB", blocks, i * 8)
        e = ((c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31,
             (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31)
        d = 0 if ref_dists is None else ref_dists[i]
        if d == 0:
            base = (0, 0, 0, 0, 0, 0)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = tables.BC1_HISTORY_CONFIRMED.get(entry) or tables.BC1_HISTORY_PROVISIONAL.get(entry) or (0, 0, 0, 0, 0, 0)
        signed_d = d if d < 128 else d - 256
        lanes[0].append(_zig(signed_d))
        lanes[1].append(0)
        for k in range(6):
            delta = (e[k] - base[k]) % mods[k]
            if delta >= mods[k] // 2:
                delta -= mods[k]
            lanes[2 + k].append(_zig(delta))
        hist.append(e)
        qa = (r0 & 15) | ((r1 & 15) << 4)
        qb = (r0 >> 4) | ((r1 >> 4) << 4)
        qc = (r2 & 15) | ((r3 & 15) << 4)
        qd = (r2 >> 4) | ((r3 >> 4) << 4)
        lanes[8].append(inv_quad[qa])
        lanes[9].append(inv_quad[qb])
        lanes[10].append(inv_quad[qc])
        lanes[11].append(inv_quad[qd])
    return encode_lanes(lanes, 1)


# --- Unit tests ------------------------------------------------------------

class ProbTablesTest(unittest.TestCase):
    def test_all_160_tables_valid_and_invertible(self):
        self.assertEqual(len(tables.PROB_TABLES), 160)
        for w in range(1, 11):
            for q in range(16):
                m = bcpack._get_model(w, q)
                self.assertIsNotNone(m)
                # Verify boundary symbols (0, 1, mid, max) invert cleanly.
                for s in {0, 1, (1 << (w - 1)), (1 << w) - 1}:
                    p = s.bit_length()
                    f = m.freq[p]
                    c = m.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
                    if c == 0:
                        f, c = f - 1, 1
                    x = (256 << 16) + c
                    self.assertEqual(m.lookup(x & 0xFFFF), s)
                    self.assertEqual(m.advance(s, x), 256 * f)


class EntropyLayerTest(unittest.TestCase):
    def test_round_trip_12_and_16_lanes(self):
        for codec in (1, 3, 4, 5, 6, 7):
            widths = bcpack.LANE_WIDTHS[codec]
            nsym = 24
            lanes = [[((i * 13 + j * 7) & ((1 << dw) - 1)) for j in range(nsym)]
                     for i, dw in enumerate(widths)]
            qs = [(i * 3) % 15 for i in range(len(widths))]
            raw = encode_lanes(lanes, codec, q=qs)
            decoded, used = bcpack.decode_lanes(raw + b"\x00\x00\x00", codec, nsym * 8)
            self.assertEqual(used, len(raw))
            self.assertEqual(decoded, lanes)

    def test_multi_fragment_flush_and_non_flush_and_3byte_header(self):
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        # 400 symbols forces a >256-byte fragment (3-byte header, h3=1) on 8-bit lanes.
        nsym = 400
        lanes = [[((i * 29 + j * 17) & ((1 << dw) - 1)) for j in range(nsym)]
                 for i, dw in enumerate(widths)]
        streams = []
        for i, (syms, dw) in enumerate(zip(lanes, widths)):
            if i == 0:
                # Two flush fragments with reduced width on the second half.
                s1 = [x & 0x3F for x in syms[:100]]
                s2 = [x & 0x1F for x in syms[100:]]
                lanes[0] = s1 + s2
                streams.append(_encode_lane_stream([
                    (s1, dw - 2, 2, True),
                    (s2, dw - 3, 5, True),
                ], dw))
            elif i == 2:
                # Chained non-flush -> flush fragments with a model change mid-stream.
                streams.append(_encode_lane_stream([
                    (syms[:150], dw, 1, False),
                    (syms[150:], dw, 4, True),
                ], dw))
            else:
                # Single large flush fragment (3-byte header on 8-bit lanes).
                streams.append(_encode_lane_stream([(syms, dw, 0, True)], dw))
        raw = interleave_lanes(streams, widths, nsym)
        decoded, used = bcpack.decode_lanes(raw, codec, nsym * 8)
        self.assertEqual(used, len(raw))
        self.assertEqual(decoded, lanes)

    def test_fragment_length_boundaries(self):
        # 2-byte header covers 1..256; 3-byte header stores (n - 257) in u16 -> 257..65,792.
        self.assertEqual(len(_make_hdr(8, 8, 0, True, 1)), 2)
        self.assertEqual(_make_hdr(8, 8, 0, True, 256), [0x10, 0xFF])
        self.assertEqual(_make_hdr(8, 8, 0, True, 257), [0x30, 0x00, 0x00])
        self.assertEqual(_make_hdr(8, 8, 0, True, MAX_FRAGMENT_LEN), [0x30, 0xFF, 0xFF])
        with self.assertRaises(ValueError):
            _make_hdr(8, 8, 0, True, 0)
        with self.assertRaises(ValueError):
            _make_hdr(8, 8, 0, True, MAX_FRAGMENT_LEN + 1)

        # Pick the (w=8, q) model whose largest symbol has the lowest frequency so a
        # 65,792-byte payload needs as few symbols as possible.
        dw = 8
        q_heavy = min(
            (q for q in range(16) if bcpack._get_model(dw, q) is not None),
            key=lambda q: bcpack._get_model(dw, q).freq[dw],
        )
        model = bcpack._get_model(dw, q_heavy)
        s_heavy = (1 << dw) - 1

        def build_exact_payload_syms(target_len):
            x = bcpack.RANS_L
            buf_len = 0
            syms_rev = []

            def step(x_in, s):
                p = s.bit_length()
                f = model.freq[p]
                c = model.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
                if c == 0:
                    f, c = f - 1, 1
                limit = f << 16
                emitted = 0
                while x_in >= limit:
                    x_in >>= 8
                    emitted += 1
                q_div, r_mod = divmod(x_in, f)
                return (q_div << 16) + c + r_mod, emitted

            while 4 + buf_len < target_len:
                need = target_len - (4 + buf_len)
                chosen = None
                for cand in (s_heavy, 1, 0) if need >= 2 else (0, 1, s_heavy):
                    nx, em = step(x, cand)
                    if em <= need:
                        chosen = (cand, nx, em)
                        break
                self.assertIsNotNone(chosen, f"could not hit exact payload length {target_len}")
                cand, x, em = chosen
                syms_rev.append(cand)
                buf_len += em
            return list(reversed(syms_rev))

        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        for target_len, exp_hdr_len, exp_tail in (
            (256, 2, bytes([0xFF])),
            (257, 3, bytes([0x00, 0x00])),
            (MAX_FRAGMENT_LEN, 3, bytes([0xFF, 0xFF])),
        ):
            with self.subTest(payload_len=target_len):
                syms0 = build_exact_payload_syms(target_len)
                nsym = len(syms0)
                frag0, _ = _encode_fragment(syms0, dw, dw, q_heavy, flush=True, strip_zeros=False)
                self.assertEqual(len(frag0), exp_hdr_len + target_len)
                self.assertEqual(frag0[1:exp_hdr_len], exp_tail)
                lanes = [syms0] + [[0] * nsym for _ in range(len(widths) - 1)]
                streams = [frag0]
                for w_i in widths[1:]:
                    f_i, _ = _encode_fragment([0] * nsym, w_i, w_i, 0, flush=True)
                    streams.append(f_i)
                raw = interleave_lanes(streams, widths, nsym)
                decoded, used = bcpack.decode_lanes(raw, codec, nsym * 8)
                self.assertEqual(used, len(raw))
                self.assertEqual(decoded[0], syms0)
                self.assertTrue(all(decoded[k] == lanes[k] for k in range(1, len(widths))))

    def test_rejects_corrupt_and_trailing_data(self):
        lanes = [[0] * 8 for _ in range(12)]
        raw = encode_lanes(lanes, 1)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw + b"\x01", 1, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw[:-2], 1, 64)
        flipped = bytearray(raw)
        flipped[-1] ^= 0xFF
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(bytes(flipped), 1, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw, 99, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw, 1, 60)

    def test_unsupported_codec_rebuild_raises(self):
        # Codec 2 (BC2) is unused in the corpus and has no lane/block decoder.
        with self.assertRaises(bcpack.BcpackUnsupportedCodec):
            bcpack.decode(b"", 2, 16)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"", 2, 16)


class BlockRebuildTest(unittest.TestCase):
    def test_bc1_confirmed_and_provisional_history(self):
        # Block 0: d=0 (zero base); Block 1: dynamic ref to block 0 (d=1);
        # Block 2: confirmed built-in preset 1 (256 + 2 - d == 1 => d=257? no, 256+2-255=3, use entry 255 -> d=3).
        blocks = b"".join([
            struct.pack("<HHBBBB", 0xF800, 0x07E0, 0x12, 0x34, 0x56, 0x78),
            struct.pack("<HHBBBB", 0xF81F, 0x001F, 0xAA, 0xBB, 0xCC, 0xDD),
            struct.pack("<HHBBBB", 0xFFFF, 0x0000, 0x00, 0x55, 0xAA, 0xFF),
        ])
        # Block 2 with d=3 -> 256 + 2 - 3 = 255 (confirmed in BC1_HISTORY_CONFIRMED).
        self.assertIn(255, tables.BC1_HISTORY_CONFIRMED)
        raw = encode_bc1_blocks(blocks, ref_dists=[0, 1, 3])
        out, prov = bcpack.decode_with_info(raw, 1, len(blocks))
        self.assertEqual(out, blocks)
        self.assertEqual(prov, set())

        # Reference provisional entry 6 from block 0: 256 + 0 - d = 6 => d = 250.
        self.assertIn(6, tables.BC1_HISTORY_PROVISIONAL)
        raw_prov = encode_bc1_blocks(blocks[:8], ref_dists=[250])
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_prov, 1, 8)
        out_p, prov_p = bcpack.decode_with_info(raw_prov, 1, 8, allow_provisional=True)
        self.assertEqual(out_p, blocks[:8])
        self.assertEqual(prov_p, {"BC1 history 6"})

    def test_bc4_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC4_PAIR)}
        # Build 2 BC4 blocks: block 0 uses d=0 (base 128,128), block 1 refs block 0 (d=1).
        b0_bits = sum(((r + p) & 7) << (3 * (4 * r + p)) for r in range(4) for p in range(4))
        b1_bits = sum(((7 - r - p) & 7) << (3 * (4 * r + p)) for r in range(4) for p in range(4))
        expected = (
            bytes([130, 120]) + b0_bits.to_bytes(6, "little") +
            bytes([140, 110]) + b1_bits.to_bytes(6, "little")
        )
        lanes = [[] for _ in range(12)]
        for i, (d, base, (a0, a1), bits) in enumerate([
            (0, (128, 128), (130, 120), b0_bits),
            (1, (130, 120), (140, 110), b1_bits),
        ]):
            lanes[0].append(_zig(d))
            lanes[1].append(0)
            lanes[2].append(_zig(a0 - base[0]))
            lanes[3].append(_zig(a1 - base[1]))
            for r in range(4):
                row = (bits >> (12 * r)) & 0xFFF
                lanes[4 + 2 * r].append(inv_pair[(row & 7, (row >> 3) & 7)])
                lanes[4 + 2 * r + 1].append(inv_pair[((row >> 6) & 7, (row >> 9) & 7)])
        raw = encode_lanes(lanes, 4)
        self.assertEqual(bcpack.decode(raw, 4, len(expected)), expected)

        # Provisional BC4 history entry (entry 25 -> block 0 uses d = 231).
        self.assertIn(25, tables.BC4_HISTORY_PROVISIONAL)
        prov_a0, prov_a1 = tables.BC4_HISTORY_PROVISIONAL[25]
        lanes_p = [[_zig(231 - 256)], [0], [0], [0]] + [[0]] * 8
        raw_p = encode_lanes(lanes_p, 4)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_p, 4, 8)
        out_p, info_p = bcpack.decode_with_info(raw_p, 4, 8, allow_provisional=True)
        self.assertEqual((out_p[0], out_p[1]), (prov_a0, prov_a1))
        self.assertEqual(info_p, {"BC4 history 25"})

    def test_bc5_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC5_PAIR)}
        # Block 0 uses confirmed BC5 entry if present, plus block 1 dynamic ref.
        confirmed_entry = next(iter(sorted(tables.BC5_HISTORY_CONFIRMED)))
        base0 = tables.BC5_HISTORY_CONFIRMED[confirmed_entry]
        d0 = 256 - confirmed_entry  # block 0: 256 + 0 - d0 == confirmed_entry
        e0 = ((base0[0] + 2) & 255, (base0[1] - 3) & 255, (base0[2] + 1) & 255, (base0[3] - 1) & 255)
        e1 = ((e0[0] + 5) & 255, (e0[1] - 4) & 255, (e0[2] + 3) & 255, (e0[3] + 2) & 255)
        bits = [
            (0x0123456789AB, 0xFEDCBA987654),
            (0x112233445566, 0x778899AABBCC),
        ]
        expected = bytearray()
        for e, (br, bg) in zip((e0, e1), bits):
            expected += bytes([e[0], e[1]]) + br.to_bytes(6, "little")
            expected += bytes([e[2], e[3]]) + bg.to_bytes(6, "little")
        lanes = [[] for _ in range(12)]
        for d, base, e, (br, bg) in ((d0, base0, e0, bits[0]), (1, e0, e1, bits[1])):
            sd = d if d < 128 else d - 256
            da, db = _deint4(_zig(sd))
            lanes[0].extend([da, db])
            lanes[1].extend([0, 0])
            for lane_idx, (c_a, c_b, b_a, b_b) in ((2, (e[0], e[1], base[0], base[1])),
                                                   (3, (e[2], e[3], base[2], base[3]))):
                d_a = (c_a - b_a + 128) % 256 - 128
                d_b = (c_b - b_b + 128) % 256 - 128
                lanes[lane_idx].extend([_zig(d_a), _zig(d_b)])
            for ch, bval in enumerate((br, bg)):
                for r in range(4):
                    row = (bval >> (12 * r)) & 0xFFF
                    lanes[4 + 4 * ch + r].extend([
                        inv_pair[(row & 7, (row >> 3) & 7)],
                        inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
                    ])
        raw = encode_lanes(lanes, 5)
        out, prov = bcpack.decode_with_info(raw, 5, len(expected))
        self.assertEqual(out, bytes(expected))
        self.assertEqual(prov, set())

    def test_bc3_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC3_ALPHA_PAIR)}
        inv_quad = {v: i for i, v in enumerate(tables.BC3_COLOUR_QUADRANT)}
        # Block 0 with d=0: base is (128, 128, 0, 0, 0, 0, 0, 0).
        a0, a1 = 135, 120
        abits = 0x0123456789AB
        c0 = (5 << 11) | (10 << 5) | 3
        c1 = (2 << 11) | (6 << 5) | 1
        r0, r1, r2, r3 = 0x12, 0x34, 0x56, 0x78
        expected = bytes([a0, a1]) + abits.to_bytes(6, "little") + struct.pack("<HHBBBB", c0, c1, r0, r1, r2, r3)
        lanes = [[] for _ in range(12)]
        lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[2].extend([_zig(a0 - 128), _zig(a1 - 128)])
        for r in range(4):
            row = (abits >> (12 * r)) & 0xFFF
            lanes[3 + r].extend([
                inv_pair[(row & 7, (row >> 3) & 7)],
                inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
            ])
        lanes[7].extend([_zig(5), _zig(2)])
        lanes[8].extend([_zig(10), _zig(6)])
        lanes[9].extend([_zig(3), _zig(1)])
        qa = (r0 & 15) | ((r1 & 15) << 4)
        qb = (r0 >> 4) | ((r1 >> 4) << 4)
        qc = (r2 & 15) | ((r3 & 15) << 4)
        qd = (r2 >> 4) | ((r3 >> 4) << 4)
        lanes[10].extend([inv_quad[qa], inv_quad[qb]])
        lanes[11].extend([inv_quad[qc], inv_quad[qd]])
        raw = encode_lanes(lanes, 3)
        out, prov = bcpack.decode_with_info(raw, 3, len(expected))
        self.assertEqual(out, expected)
        self.assertEqual(prov, set())

    def _signed_mod_delta(self, target, ref, bits):
        mod = 1 << bits
        d = (target - ref) % mod
        if d >= mod // 2:
            d -= mod
        return d

    def _bc7_sig_pair(self, table, mode, part=0, rot=0, isel=0):
        rank = table["config"].index((mode, part, rot, isel))
        idx = table["rank_of"].index(rank)
        return divmod(idx, 16)

    def test_bc6h_round_trip_and_provisional(self):
        # 3-block synthetic BC6H stream exercising:
        #   block 0: mode 1 (2-region, part=5, di=0, de=0)
        #   block 1: mode 13 (1-region, bit 10/11 swap, dynamic ref di=1, de=1)
        #   block 2: mode 14 (1-region, bit 10..15 reverse, dynamic ref di=1, de=1)
        specs = [
            (1, 5, 0, 0,
             [(0x123, 0x05, 0x11, 0x0A), (0x2A5, 0x14, 0x07, 0x19), (0x0F1, 0x0C, 0x13, 0x04)],
             [2, 5, 7, 1, 3, 6, 4, 2, 0, 3, 2, 5, 7, 1, 4, 0]),
            (13, 0, 1, 1,
             [(0xA55, 0x33, 0, 0), (0x5AA, 0x81, 0, 0), (0xC12, 0x47, 0, 0)],
             [5, 12, 9, 3, 14, 7, 1, 8, 11, 4, 6, 15, 2, 10, 13, 0]),
            (14, 0, 1, 1,
             [(0xB419, 0x07, 0, 0), (0x9C23, 0x0B, 0, 0), (0x4D05, 0x02, 0, 0)],
             [3, 14, 8, 6, 1, 12, 9, 4, 15, 2, 7, 10, 5, 13, 0, 11]),
        ]
        # Ensure anchor2[5] has top bit 0 for block 0.
        a2_5 = bptc.BC7_ANCHOR2[5]
        specs[0][5][a2_5] &= 3

        lanes = [[] for _ in range(16)]
        st_e, st_i = [], []
        expected = bytearray()
        for idx, (mode, part, di, de, eps, ixs) in enumerate(specs):
            code = bcpack._BC6H_CODE[mode]
            trans, base, deltas, reg = bcpack._BC6H_INFO[code]
            eref = bcpack._BC6H_NONE_E if de == 0 else st_e[idx - de]
            iref = bcpack._BC6H_NONE_I if di == 0 else st_i[idx - di]
            diff_d = (de - di) % 256
            if diff_d >= 128:
                diff_d -= 256
            l0a, l0b = _deint4(_zig(diff_d))
            l1a, l1b = _deint4(di)
            lanes[0].extend([l0a, l0b])
            lanes[1].extend([l1a, l1b])
            lanes[15].extend([mode - 1, part])

            w_shift = 16 - base
            f_blk = []
            st_blk = []
            for c in range(3):
                dc = deltas[c]
                sh = 6 - dc if dc < 6 else 0
                mD = (1 << dc) - 1
                w, x, y, z = eps[c]
                r0, r1, r2, r3 = eref[c]
                dz0 = self._signed_mod_delta(w, r0 >> w_shift, base)
                dz1 = self._signed_mod_delta(x, r1 >> (10 - dc), dc)
                dz2 = self._signed_mod_delta(y, r2 >> sh, dc) if reg == 2 else 0
                dz3 = self._signed_mod_delta(z, r3 >> sh, dc) if reg == 2 else 0
                a0, b0 = _deint_n(_zig(dz0), 8)
                a1, b1 = _deint_n(_zig(dz1), 5)
                lanes[2 + c].extend([a0, b0])
                lanes[5 + c].extend([a1, b1])
                lanes[8 + c].extend([_zig(dz2), _zig(dz3)])
                x10 = x << (10 - dc)
                if reg == 2:
                    st_blk.append((w << w_shift, x10, y << sh, z << sh))
                    f_blk.append((bcpack._bc6h_stream_w(mode, w), x, y, z))
                else:
                    v = (w & mD) ^ ((1 << (dc - 1)) if trans == 1 else 0)
                    low_y = (v >> (dc - 6)) if dc >= 6 else (v << sh)
                    st_blk.append((w << w_shift, x10, low_y, x10 >> 4))
                    f_blk.append((bcpack._bc6h_stream_w(mode, w), x, 0, 0))
            st_e.append(tuple(st_blk))

            ibits = 3 if reg == 2 else 4
            ishift = 4 - ibits
            for yy in range(4):
                ds = [_zig(self._signed_mod_delta(ixs[yy * 4 + xx], iref[yy * 4 + xx] >> ishift, ibits))
                      for xx in range(4)]
                lanes[11 + yy].extend([
                    _pack_ix_pair(ds[0], ds[1], yy),
                    _pack_ix_pair(ds[2], ds[3], yy),
                ])
            st_i.append(tuple(v << ishift for v in ixs))

            plan, header = bcpack._bc6h_plan(code)
            acc = code
            for c, q, bit_idx, pos in plan:
                if c < 0:
                    if (part >> bit_idx) & 1:
                        acc |= 1 << pos
                elif (f_blk[c][q] >> bit_idx) & 1:
                    acc |= 1 << pos
            second = bptc.BC7_ANCHOR2[part] if reg == 2 else -1
            at = header
            for p in range(16):
                nb = (ibits - 1) if (p == 0 or p == second) else ibits
                acc |= (ixs[p] & ((1 << nb) - 1)) << at
                at += nb
            expected += acc.to_bytes(16, "little")

        raw = encode_lanes(lanes, 6)
        out, prov = bcpack.decode_with_info(raw, 6, len(expected))
        self.assertEqual(out, bytes(expected))
        self.assertEqual(prov, set())

        # Built-in BC6H history reference (entry k=42, di=de=256-42) with no
        # external history table: strict refuses with --history hint; a 3-block
        # chain exceeding _loose_limit also refuses in provisional mode.
        k_fit = 42
        l1a, l1b = _deint4(256 - k_fit)
        eh_val = ((0x2000, 0x100, 0x08, 0x04), (0x4000, 0x200, 0x0C, 0x08), (0x6000, 0x300, 0x10, 0x0C))
        ih_val = (3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12)
        d0 = _zig(self._signed_mod_delta(0, ih_val[0], 4))
        lanes_p = [[0, 0] for _ in range(16)]
        lanes_p[1] = [l1a, l1b]
        lanes_p[15] = [10, 0]  # mode 11 (1-region, 4-bit indices)
        lanes_p[11] = [_pack_ix_pair(d0, 0, 0), 0]
        raw_p = encode_lanes(lanes_p, 6)
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "--history"):
            bcpack.decode(raw_p, 6, 16)

        # 3-block chain off built-in history exceeds loose limit (max(2, n//1024) == 2).
        lanes_3 = [[0] * 6 for _ in range(16)]
        lanes_3[1][0:2] = [l1a, l1b]
        d_prev_a, d_prev_b = _deint4(1)
        lanes_3[1][2:4] = [d_prev_a, d_prev_b]
        lanes_3[1][4:6] = [d_prev_a, d_prev_b]
        lanes_3[15] = [10, 0, 10, 0, 10, 0]
        raw_3 = encode_lanes(lanes_3, 6)
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "--history"):
            bcpack.decode(raw_3, 6, 48, allow_provisional=True)

        # External runtime history JSON: test confirmed (class 3), fitted (class 1),
        # and last-resort (class 2) via combined JSON file and directory layouts.
        k_conf, k_lr = 17, 43
        bc6h_doc = {
            "endpointHistory": {str(k_conf): eh_val, str(k_fit): eh_val, str(k_lr): eh_val},
            "indexHistory": {str(k_conf): list(ih_val), str(k_fit): list(ih_val), str(k_lr): list(ih_val)},
            "endpointClass": {str(k_conf): 3, str(k_fit): 1, str(k_lr): 2},
            "indexClass": {str(k_conf): 3, str(k_fit): 1, str(k_lr): 3},
        }
        with tempfile.TemporaryDirectory() as tmp:
            combined = os.path.join(tmp, "history.json")
            with open(combined, "w", encoding="utf-8") as fh:
                json.dump({"bc6h": bc6h_doc}, fh)
            hist_dir = os.path.join(tmp, "hist_dir")
            os.mkdir(hist_dir)
            with open(os.path.join(hist_dir, "bc6h.json"), "w", encoding="utf-8") as fh:
                json.dump(bc6h_doc, fh)

            # Class 1 (fitted): strict refuses, provisional accepts and tags both.
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_p, 6, 16, history=combined)
            out_p, prov_p = bcpack.decode_with_info(raw_p, 6, 16, allow_provisional=True, history=combined)
            self.assertEqual(len(out_p), 16)
            self.assertIn(f"BC6H endpoint history {k_fit}", prov_p)
            self.assertIn(f"BC6H index history {k_fit}", prov_p)

            # Class 3 (confirmed): strict accepts via directory path with empty prov set.
            c1a, c1b = _deint4(256 - k_conf)
            lanes_c = [[0, 0] for _ in range(16)]
            lanes_c[1] = [c1a, c1b]
            lanes_c[15] = [10, 0]
            lanes_c[11] = [_pack_ix_pair(d0, 0, 0), 0]
            raw_c = encode_lanes(lanes_c, 6)
            out_c, prov_c = bcpack.decode_with_info(raw_c, 6, 16, history=hist_dir)
            self.assertEqual(out_c, out_p)
            self.assertEqual(prov_c, set())

            # Class 2 (last-resort): strict refuses, provisional labels last-resort.
            r1a, r1b = _deint4(256 - k_lr)
            lanes_lr = [[0, 0] for _ in range(16)]
            lanes_lr[1] = [r1a, r1b]
            lanes_lr[15] = [10, 0]
            lanes_lr[11] = [_pack_ix_pair(d0, 0, 0), 0]
            raw_lr = encode_lanes(lanes_lr, 6)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_lr, 6, 16, history=hist_dir)
            _, prov_lr = bcpack.decode_with_info(raw_lr, 6, 16, allow_provisional=True, history=hist_dir)
            self.assertIn(f"BC6H endpoint history {k_lr} (last-resort)", prov_lr)

        # Anchor violation in strict mode (zero built-in refs, anchor 0 = 8) raises BcpackError.
        lanes_bad = [[0, 0] for _ in range(16)]
        lanes_bad[15] = [10, 0]  # mode 11, di=0 => iref[0]=8 => delta 0 leaves anchor bit set!
        raw_bad = encode_lanes(lanes_bad, 6)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode(raw_bad, 6, 16)

    def test_bc7_config_completeness_and_empty_defaults(self):
        # BC7_CONFIG must be a bijection over all 285 valid D3D11 BC7 mode
        # configurations plus the (255, 255, 0, 0) sentinel at rank 285.
        self.assertEqual(len(bptc.BC7_CONFIG), 286)
        self.assertEqual(bptc.BC7_CONFIG[-1], (255, 255, 0, 0))
        expected = (
            {(0, p, 0, 0) for p in range(16)}
            | {(1, p, 0, 0) for p in range(64)}
            | {(2, p, 0, 0) for p in range(64)}
            | {(3, p, 0, 0) for p in range(64)}
            | {(4, 0, rot, isel) for rot in range(4) for isel in range(2)}
            | {(5, 0, rot, 0) for rot in range(4)}
            | {(6, 0, 0, 0)}
            | {(7, p, 0, 0) for p in range(64)}
        )
        self.assertEqual(len(expected), 285)
        self.assertEqual(set(bptc.BC7_CONFIG[:-1]), expected)
        self.assertEqual(len(bptc.BC7_CONFIG_INFERRED), 10)
        for rank in bptc.BC7_CONFIG_INFERRED:
            self.assertTrue(0 <= rank < 285)
            self.assertEqual(bptc.BC7_CONFIG[rank][0], 2)

        # Default history loaders embed no fitted/game-derived history tables.
        _, _, ec, ic = bptc.load_bc6h()
        self.assertEqual(ec, (0,) * 256)
        self.assertEqual(ic, (0,) * 256)
        t7 = bptc.load_bc7()
        self.assertEqual(t7["last_resort"], frozenset())
        self.assertTrue(all(not any(row) for row in t7["history_endpoint_bits"]))
        self.assertTrue(all(not any(row) for row in t7["history_index_bits"]))

    def test_bc7_round_trip_and_provisional(self):
        table = bptc.load_bc7()
        # Block 0: mode 6 (RGBA7777 + per-endpoint p-bits, 4-bit indices), di=0, de=0.
        # Block 1: mode 1 (2 subsets, RGB666 + shared p-bits, 3-bit indices), dynamic di=1, de=1.
        # Block 2: mode 4 (rot=2, isel=1, 2-bit primary + 3-bit secondary indices), di=0, de=0.
        # Block 3: mode 0 all-zero block -> 16 zero bytes.
        s_m6 = self._bc7_sig_pair(table, 6, 0, 0, 0)
        s_m1 = self._bc7_sig_pair(table, 1, 3, 0, 0)
        s_m4 = self._bc7_sig_pair(table, 4, 0, 2, 1)
        s_m0 = self._bc7_sig_pair(table, 0, 0, 0, 0)

        lanes = [[] for _ in range(16)]
        expected = bytearray()

        # --- Block 0: mode 6 ---
        # R (7+1=8 bits, LSB is p-bit), G (7), B (7), A (7).
        ep0 = [((0x65, 0x3A, 0x51, 0x70), (0x92, 0x1F, 0x24, 0x61))]  # R0 p=1, R1 p=0
        ix0 = [3, 10, 15, 2, 7, 12, 4, 9, 14, 1, 6, 11, 5, 8, 13, 0]
        lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[15].extend(list(s_m6))
        for c, ln in enumerate((2, 3, 4, 5)):
            bits = 8 if c == 0 else 7
            lanes[ln].extend([
                _zig(self._signed_mod_delta(ep0[0][0][c], 0, bits)),
                _zig(self._signed_mod_delta(ep0[0][1][c], 0, bits)),
            ])
        for ln in (6, 7, 8, 9, 10):
            lanes[ln].extend([0, 0])
        for y in range(4):
            ds = [_zig(self._signed_mod_delta(ix0[y * 4 + x], 0, 4)) for x in range(4)]
            lanes[11 + y].extend([_pack_ix_pair(ds[0], ds[1], y), _pack_ix_pair(ds[2], ds[3], y)])
        # Pack expected mode-6 block directly from BC7 spec.
        acc = 1 << 6
        pos = 7
        for c in range(4):
            for e in range(2):
                v = ep0[0][e][c] >> 1 if c == 0 else ep0[0][e][c]
                acc |= (v & 0x7F) << pos
                pos += 7
        acc |= (ep0[0][0][0] & 1) << pos
        pos += 1
        acc |= (ep0[0][1][0] & 1) << pos
        pos += 1
        for p in range(16):
            nb = 3 if p == 0 else 4
            acc |= (ix0[p] & ((1 << nb) - 1)) << pos
            pos += nb
        self.assertEqual(pos, 128)
        expected += acc.to_bytes(16, "little")

        # Stored state after block 0 (replicated across subsets 0..2).
        sep0 = [
            ep0[0][0][0], ep0[0][0][1] << 1, ep0[0][0][2] << 1, ep0[0][0][3] << 1,
            ep0[0][1][0], ep0[0][1][1] << 1, ep0[0][1][2] << 1, ep0[0][1][3] << 1,
        ] * 3
        six0 = list(ix0) + [0] * 16

        # --- Block 1: mode 1 (part=3, 2 subsets, shared p-bit, di=1, de=1) ---
        # R has 6+1=7 bits (LSB is shared p-bit per subset!), G/B have 6 bits.
        ep1 = [
            ((0x53, 0x22, 0x15, 0), (0x29, 0x31, 0x0B, 0)),  # sub 0: both R LSB = 1
            ((0x44, 0x12, 0x38, 0), (0x62, 0x05, 0x20, 0)),  # sub 1: both R LSB = 0
        ]
        ix1 = [2, 5, 7, 1, 4, 6, 0, 3, 7, 2, 1, 5, 6, 3, 4, 2]
        a2_3 = table["anchor2"][3]
        ix1[a2_3] &= 3
        l1a, l1b = _deint4(1)
        lanes[0].extend([0, 0])  # de - di = 0 => de = 1
        lanes[1].extend([l1a, l1b])
        lanes[15].extend(list(s_m1))
        for sub, lns in ((0, (2, 3, 4)), (1, (6, 7, 8))):
            for c, ln in enumerate(lns):
                bits = 7 if c == 0 else 6
                sh = 8 - bits
                lanes[ln].extend([
                    _zig(self._signed_mod_delta(ep1[sub][0][c], sep0[sub * 8 + c] >> sh, bits)),
                    _zig(self._signed_mod_delta(ep1[sub][1][c], sep0[sub * 8 + 4 + c] >> sh, bits)),
                ])
        for ln in (5, 9, 10):
            lanes[ln].extend([0, 0])
        for y in range(4):
            ds = [_zig(self._signed_mod_delta(ix1[y * 4 + x], six0[y * 4 + x] >> 1, 3)) for x in range(4)]
            lanes[11 + y].extend([_pack_ix_pair(ds[0], ds[1], y), _pack_ix_pair(ds[2], ds[3], y)])
        acc = (1 << 1) | (3 << 2)
        pos = 8
        for c in range(3):
            for sub in range(2):
                for e in range(2):
                    v = ep1[sub][e][c] >> 1 if c == 0 else ep1[sub][e][c]
                    acc |= (v & 0x3F) << pos
                    pos += 6
        acc |= (ep1[0][0][0] & 1) << pos
        pos += 1
        acc |= (ep1[1][0][0] & 1) << pos
        pos += 1
        p2_3 = table["partition2"][3]
        for p in range(16):
            anchor = p == (0 if p2_3[p] == 0 else a2_3)
            nb = 2 if anchor else 3
            acc |= (ix1[p] & ((1 << nb) - 1)) << pos
            pos += nb
        self.assertEqual(pos, 128)
        expected += acc.to_bytes(16, "little")

        # --- Block 2: mode 4 (rot=2, isel=1, di=0, de=0) ---
        # With rot=2: true channel 1 (G) gets ab=6 bits, true channel 3 (A) gets cb=5 bits.
        ep2 = ((0x15, 0x2B, 0x0C, 0x1A), (0x09, 0x34, 0x1E, 0x07))
        ix2_s1 = [2, 7, 5, 1, 6, 3, 4, 0, 7, 2, 1, 5, 3, 6, 0, 4]  # 3-bit secondary (lanes 11..14)
        ix2_s2 = [1, 3, 2, 0, 2, 1, 3, 0, 3, 2, 1, 0, 1, 2, 3, 0]  # 2-bit primary (lanes 6,7,8,10)
        lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[15].extend(list(s_m4))
        for c, (ln, bits) in enumerate(((2, 5), (3, 6), (4, 5), (5, 5))):
            lanes[ln].extend([
                _zig(self._signed_mod_delta(ep2[0][c], 0, bits)),
                _zig(self._signed_mod_delta(ep2[1][c], 0, bits)),
            ])
        lanes[9].extend([0, 0])
        for y, ln2 in enumerate((6, 7, 8, 10)):
            ds1 = [_zig(self._signed_mod_delta(ix2_s1[y * 4 + x], 0, 3)) for x in range(4)]
            lanes[11 + y].extend([_pack_ix_pair(ds1[0], ds1[1], y), _pack_ix_pair(ds1[2], ds1[3], y)])
            ds2 = [_zig(self._signed_mod_delta(ix2_s2[y * 4 + x], 0, 2)) for x in range(4)]
            lanes[ln2].extend([_pack_ix_pair(ds2[0], ds2[1], y), _pack_ix_pair(ds2[2], ds2[3], y)])
        acc = (1 << 4) | (2 << 5) | (1 << 7)
        pos = 8
        for c in range(3):
            tc = 3 if c == 1 else c
            for e in range(2):
                acc |= (ep2[e][tc] & 0x1F) << pos
                pos += 5
        for e in range(2):
            acc |= (ep2[e][1] & 0x3F) << pos
            pos += 6
        for p in range(16):
            nb = 1 if p == 0 else 2
            acc |= (ix2_s2[p] & ((1 << nb) - 1)) << pos
            pos += nb
        for p in range(16):
            nb = 2 if p == 0 else 3
            acc |= (ix2_s1[p] & ((1 << nb) - 1)) << pos
            pos += nb
        self.assertEqual(pos, 128)
        expected += acc.to_bytes(16, "little")

        # --- Block 3: mode 0 all-zero block ---
        lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[15].extend(list(s_m0))
        for ln in range(2, 15):
            lanes[ln].extend([0, 0])
        expected += b"\x00" * 16

        raw = encode_lanes(lanes, 7)
        out, prov = bcpack.decode_with_info(raw, 7, len(expected))
        self.assertEqual(out, bytes(expected))
        self.assertEqual(prov, set())

        # Inferred BC7 signature: strict refuses, provisional accepts and tags it.
        inf_rank = min(bptc.BC7_CONFIG_INFERRED)
        inf_a, inf_b = divmod(table["rank_of"].index(inf_rank), 16)
        lanes_inf = [[0, 0] for _ in range(16)]
        lanes_inf[15] = [inf_a, inf_b]
        raw_inf = encode_lanes(lanes_inf, 7)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_inf, 7, 16)
        out_inf, prov_inf = bcpack.decode_with_info(raw_inf, 7, 16, allow_provisional=True)
        self.assertEqual(len(out_inf), 16)
        self.assertIn(f"BC7 signature {inf_rank}", prov_inf)

        # Built-in BC7 history ref without external history: strict refuses with --history hint.
        def _bc7_hist_chunk(k, nblk=1):
            de = 256 - k
            diff = de if de < 128 else de - 256
            l0a, l0b = _deint4(_zig(diff))
            ls = [[0] * (2 * nblk) for _ in range(16)]
            ls[0][0:2] = [l0a, l0b]
            d1a, d1b = _deint4(1)
            for bi in range(1, nblk):
                ls[1][2 * bi:2 * bi + 2] = [d1a, d1b]
            for bi in range(nblk):
                ls[15][2 * bi:2 * bi + 2] = list(s_m6)
            return encode_lanes(ls, 7)

        k_conf, k_prov, k_lr = 11, 19, 27
        raw_h1 = _bc7_hist_chunk(k_conf, nblk=1)
        raw_h3 = _bc7_hist_chunk(k_conf, nblk=3)
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "--history"):
            bcpack.decode(raw_h1, 7, 16)
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "--history"):
            bcpack.decode(raw_h3, 7, 48, allow_provisional=True)

        # External runtime BC7 history: confirmed, provisional, and partial+last-resort.
        ep_entry = [0x65, 0x3A << 1, 0x51 << 1, 0x70 << 1, 0x92, 0x1F << 1, 0x24 << 1, 0x61 << 1] * 3
        bc7_doc = {
            "historyEndpoints": {str(k_conf): ep_entry, str(k_prov): ep_entry, str(k_lr): ep_entry},
            "historyEndpointBits": {
                str(k_conf): [8] * 24,
                str(k_prov): [8] * 24,
                str(k_lr): [6] * 24,
            },
            "historyEndpointConfirmedBits": {
                str(k_conf): [8] * 24,
                str(k_prov): [0] * 24,
                str(k_lr): [0] * 24,
            },
            "historyEndpointLastResort": [[k_lr, 0, 0, 0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            hist_dir = os.path.join(tmp, "hist_dir")
            os.mkdir(hist_dir)
            with open(os.path.join(hist_dir, "bc7.json"), "w", encoding="utf-8") as fh:
                json.dump(bc7_doc, fh)

            # Confirmed entry: strict accepts and matches exact mode-6 block bytes.
            out_c, prov_c = bcpack.decode_with_info(raw_h1, 7, 16, history=hist_dir)
            self.assertEqual(prov_c, set())
            # Zero-delta mode-6 block from ep_entry and zero indices.
            exp_h = 1 << 6
            pos = 7
            for c in range(4):
                for e in range(2):
                    v = ep_entry[e * 4 + c] >> 1
                    exp_h |= (v & 0x7F) << pos
                    pos += 7
            exp_h |= (ep_entry[0] & 1) << pos
            exp_h |= (ep_entry[4] & 1) << (pos + 1)
            self.assertEqual(out_c, exp_h.to_bytes(16, "little"))

            # Unconfirmed entry: strict refuses, provisional accepts via in-memory dict too.
            raw_prov = _bc7_hist_chunk(k_prov, nblk=1)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_prov, 7, 16, history=bc7_doc)
            out_pr, prov_pr = bcpack.decode_with_info(
                raw_prov, 7, 16, allow_provisional=True, history=bc7_doc
            )
            self.assertEqual(out_pr, out_c)
            self.assertIn(f"BC7 endpoint history {k_prov}", prov_pr)

            # Partial-depth (top 6 bits) + last-resort entry: strict refuses, provisional labels both.
            raw_lr = _bc7_hist_chunk(k_lr, nblk=1)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_lr, 7, 16, history=hist_dir)
            out_lr, prov_lr = bcpack.decode_with_info(
                raw_lr, 7, 16, allow_provisional=True, history=hist_dir
            )
            self.assertEqual(len(out_lr), 16)
            self.assertIn(f"BC7 endpoint history {k_lr} (top 6 bits known)", prov_lr)
            self.assertIn(f"BC7 endpoint history {k_lr} (last-resort)", prov_lr)

        # Mode-1 shared p-bit mismatch in strict mode raises BcpackError.
        lanes_pbit = [[0, 0] for _ in range(16)]
        lanes_pbit[15] = list(s_m1)
        lanes_pbit[2] = [_zig(1), _zig(0)]  # R0 LSB=1, R1 LSB=0 => shared p-bit conflict
        raw_pbit = encode_lanes(lanes_pbit, 7)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode(raw_pbit, 7, 16)


class DocumentedExampleTest(unittest.TestCase):
    """The table and the BC7 example in docs/textures.md ("From a chunk entry to bcpack.decode")."""

    BC7_BLOCK = bytes.fromhex("40 31 72 04 a5 68 19 be 02 34 56 78 9a bc de f0")
    BC7_PACKED = bytes.fromhex(
        "101010101010101004040303030304040101010101010140"
        "000000000000024002027546281800000101df5f4b7c0341"
        "020202411010101004040403101010104004100103030304"
        "4001080001010104000000080000000041020994e4db27a4"
        "410511fe6b020005"
    )

    def test_bc7_fixed_stream_strict(self):
        out, prov = bcpack.decode_with_info(self.BC7_PACKED, 7, 16)
        self.assertEqual(out, self.BC7_BLOCK)
        self.assertFalse(prov)
        # The hex in the doc is what the test encoder writes for that block.
        self.assertEqual(encode_bc7_mode6_blocks(self.BC7_BLOCK), self.BC7_PACKED)

    def test_lane_widths_and_symbols_per_block(self):
        widths = {
            1: (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8),
            3: (4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8),
            4: (8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
            5: (4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
            6: (4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5),
            7: (4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5),
        }
        self.assertEqual(bcpack.LANE_WIDTHS, widths)
        for codec in widths:
            # Chunk-entry tag = codec << 2: 0x04, 0x0c, 0x10, 0x14, 0x18, 0x1c.
            self.assertEqual(codec << 2, (0x04, 0x0C, 0x10, 0x14, 0x18, 0x1C)[sorted(widths).index(codec)])
            self.assertEqual(bcpack.BLOCK_BYTES[codec] // bcpack.ELEMENT_BYTES, 1 if codec in (1, 4) else 2)


def encode_bc7_mode6_blocks(blocks, *, first_ep_history=None):
    """Encode raw BC7 mode-6 blocks (16 bytes each) into a codec-7 BCPack chunk.

    When `first_ep_history` is `(entry_index, ep_row_24)`, block 0 predicts its
    endpoints from built-in history `entry_index` instead of zero.
    """
    table = bptc.load_bc7()
    rank = table["config"].index((6, 0, 0, 0))
    s15a, s15b = divmod(table["rank_of"].index(rank), 16)
    n = len(blocks) // 16
    lanes = [[] for _ in range(16)]
    for i in range(n):
        acc = int.from_bytes(blocks[i * 16:i * 16 + 16], "little")
        assert (acc & 0x7F) == (1 << 6), "expected mode-6 BC7 block"
        pos = 7
        ch = [[0, 0] for _ in range(4)]
        for c in range(4):
            for e in range(2):
                ch[c][e] = (acc >> pos) & 0x7F
                pos += 7
        p0 = (acc >> pos) & 1
        p1 = (acc >> (pos + 1)) & 1
        pos += 2
        ch[0][0] = (ch[0][0] << 1) | p0
        ch[0][1] = (ch[0][1] << 1) | p1
        ixs = []
        for p in range(16):
            nb = 3 if p == 0 else 4
            ixs.append((acc >> pos) & ((1 << nb) - 1))
            pos += nb
        if i == 0 and first_ep_history is not None:
            k, rep = first_ep_history
            de = 256 - k
            diff = de if de < 128 else de - 256
            l0a, l0b = _deint4(_zig(diff))
            lanes[0].extend([l0a, l0b])
        else:
            rep = (0,) * 24
            lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[15].extend([s15a, s15b])
        for c, ln in enumerate((2, 3, 4, 5)):
            mod = 256 if c == 0 else 128
            sh = 0 if c == 0 else 1
            for e in range(2):
                d = (ch[c][e] - (rep[e * 4 + c] >> sh)) % mod
                if d >= mod // 2:
                    d -= mod
                lanes[ln].append(_zig(d))
        for ln in (6, 7, 8, 9, 10):
            lanes[ln].extend([0, 0])
        for y in range(4):
            ds = []
            for x in range(4):
                d = ixs[y * 4 + x]
                if d >= 8:
                    d -= 16
                ds.append(_zig(d))
            lanes[11 + y].extend([_pack_ix_pair(ds[0], ds[1], y), _pack_ix_pair(ds[2], ds[3], y)])
    return encode_lanes(lanes, 7)


class UnknownBlockCountTest(unittest.TestCase):
    """The provisional gate counts undetermined blocks. A BC6H or BC7 block whose endpoint and index both come
    from unknown history entries is one block, not two."""

    def test_a_block_with_both_roots_unknown_counts_once(self):
        self.assertEqual(bcpack._unknown_blocks([255, 255, 7, -1], [255, 3, 255, -1], {255}, {255}), 3)
        self.assertEqual(bcpack._unknown_blocks([-1, -1], [-1, -1], {255}, {255}), 0)

    def test_bc7_gate_reports_distinct_blocks(self):
        n = 4
        lanes = [[] for _ in range(16)]
        for i in range(n):
            di = i + 1   # both lanes reach back past block 0: endpoint and index history entry 256 + i - di = 255
            lanes[0].extend(_deint4(_zig(0)))
            lanes[1].extend(_deint4(di))
            for ln in range(2, 16):
                lanes[ln].extend([0, 0])
        from unittest import mock
        with mock.patch.object(bcpack, "_bc7_check", return_value=[("endpoint", 255), ("index", 255)]):
            with self.assertRaisesRegex(bcpack.BcpackProvisionalError, f" {n}/{n} blocks"):
                bcpack._bc7_blocks(lanes, n, set())


if __name__ == "__main__":
    unittest.main()
