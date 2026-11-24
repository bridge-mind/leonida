"""Tests for tools/bcpack_history_calib.py and external --history support.

All rANS BCPack streams, DDS references, and RDS5/ETPF pools are constructed
synthetically in memory — no game files or game-derived tables are used.

    python -I tests/test_bcpack_history_calib.py
"""
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bcpack  # noqa: E402
import bcpack_history_calib as calib  # noqa: E402
import bcpack_tables as tables  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402
import ttpf_mips as ttpf  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic rANS encoder and container builders (test helpers only)
# ---------------------------------------------------------------------------

def _zig(v):
    return (-2 * v - 1) if v < 0 else (2 * v)


def _signed_mod_delta(target, ref, bits):
    mod = 1 << bits
    d = (target - ref) % mod
    if d >= mod // 2:
        d -= mod
    return d


def _signed_mod_delta_m(target, ref, mod):
    d = (target - ref) % mod
    if d >= mod // 2:
        d -= mod
    return d


def _deint_n(v, n):
    a = b = 0
    for k in range(n):
        a |= ((v >> (2 * k)) & 1) << k
        b |= ((v >> (2 * k + 1)) & 1) << k
    return a, b


def _deint4(v):
    return _deint_n(v, 4)


def _pack_ix_pair(d0, d1, y):
    return bcpack._inter(d1, d0, 4) if y < 3 else bcpack._inter(d0, d1, 4)


def _encode_rans_syms(syms, model, x):
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


def _make_hdr(default_w, w, q, flush, n):
    z = default_w - w
    if n <= 256:
        return [((z & 3) << 6) | ((1 if flush else 0) << 4) | (q & 15), n - 1]
    n2 = n - 257
    return [
        ((z & 3) << 6) | (1 << 5) | ((1 if flush else 0) << 4) | (q & 15),
        n2 & 0xFF,
        (n2 >> 8) & 0xFF,
    ]


def _encode_fragment(syms, default_w, w, q):
    model = bcpack._get_model(w, q)
    x, buf = _encode_rans_syms(syms, model, bcpack.RANS_L)
    payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
    stripped = 0
    while len(payload) > 1 and payload[-1] == 0 and stripped < bcpack.MAX_VIRTUAL_ZEROS:
        payload.pop()
        stripped += 1
    return bytes(_make_hdr(default_w, w, q, True, len(payload)) + payload)


def _interleave_lanes(lane_streams, widths, nsym):
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
    widths = bcpack.LANE_WIDTHS[codec]
    nsym = len(lanes[0])
    streams = [_encode_fragment(syms, dw, dw, q) for syms, dw in zip(lanes, widths)]
    return _interleave_lanes(streams, widths, nsym)


def _write_dx10_dds(path, dxgi, width, height, mip_blobs):
    hdr = bytearray(148)
    hdr[0:4] = b"DDS "
    struct.pack_into("<IIIIIII", hdr, 4, 124, 0x1007, height, width, 0, 0, len(mip_blobs))
    struct.pack_into("<II", hdr, 76, 32, 0x4)
    hdr[84:88] = b"DX10"
    struct.pack_into("<IIIII", hdr, 128, dxgi, 3, 0, 1, 0)
    with open(path, "wb") as fh:
        fh.write(hdr)
        for m in mip_blobs:
            fh.write(m)


def _write_custom_rds5_pool(path, records, chunk_specs):
    """Write an RDS5/ETPF pool from `(codec, csize, payload_bytes)` chunk specs."""
    count = len(chunk_specs) + 2
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    area = len(records) * 24
    metadata = 16 + area + len(records) * 32
    padded = ttpf.align(metadata, 4096)

    source = etpf + padded
    virt = padded
    entries = []
    packed_chunks = []
    for codec, csize, payload in chunk_specs:
        assert csize % 4096 == 0
        pad = (-len(payload)) % 256
        entries.append((source // 256) | (codec << 26) | ((virt // 4096) << 30) | (pad << 52))
        packed_chunks.append((source, payload))
        source += len(payload) + pad
        virt += csize

    vsize = virt
    entries.append((source // 256) | ((vsize // 4096) << 30))

    blob = bytearray(source)
    blob[0:4] = b"RDS5"
    struct.pack_into("<I", blob, 4, count)
    struct.pack_into("<Q", blob, 8, vsize | (22 << 56))
    struct.pack_into("<I", blob, 16, (count + 31) // 32)
    for i, e in enumerate(entries):
        struct.pack_into("<Q", blob, 24 + i * 8, e)
    blob[etpf:etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, len(records), area)
    for i, rec in enumerate(records):
        blob[etpf + 16 + i * 16:etpf + 16 + (i + 1) * 16] = rec
        struct.pack_into("<Q", blob, etpf + 16 + len(records) * 16 + i * 8, 0x1000 + i)
    for src, payload in packed_chunks:
        blob[src:src + len(payload)] = payload
    with open(path, "wb") as fh:
        fh.write(blob)
    return padded


def _pack_etpf(width, height, packed_mips, non_tail, fmt, base, mode=4):
    a = (width - 1) | ((height - 1) << 14) | ((packed_mips & 0xF) << 28)
    b = non_tail | (fmt << 3) | (0 << 10) | (1 << 21) | (mode << 23)
    c = (base // 4096) | (0x103 << 21)
    return struct.pack("<IIII", a, b, c, 0)


def _encode_bc7_mode6_tiled(tiled_bytes, *, secret_entry=None, secret_ep=None, secret_ix=None, ref_count=4):
    """Encode a 4096-byte-multiple buffer of Mode-6 BC7 blocks into codec-7 BCPack."""
    tab = bptc.load_bc7()
    rank = tab["config"].index((6, 0, 0, 0))
    s15a, s15b = divmod(tab["rank_of"].index(rank), 16)
    nblocks = len(tiled_bytes) // 16
    lanes = [[] for _ in range(16)]
    eps_state = []
    ixs_state = []
    for i in range(nblocks):
        acc = int.from_bytes(tiled_bytes[i * 16:(i + 1) * 16], "little")
        assert (acc & 0x7F) == (1 << 6)
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

        if secret_entry is not None and i < ref_count:
            d = 256 + i - secret_entry
            di = d if secret_ix is not None else 0
            de = d if secret_ep is not None else 0
            rep = secret_ep if secret_ep is not None else (0,) * 24
            rix = secret_ix if secret_ix is not None else (0,) * 32
        elif i > 0 and secret_entry is not None and i < ref_count + 4:
            di = 1 if secret_ix is not None else 0
            de = 1
            rep = eps_state[i - 1]
            rix = ixs_state[i - 1] if secret_ix is not None else (0,) * 32
        else:
            di = 0
            de = 0
            rep = (0,) * 24
            rix = (0,) * 32

        diff_d = _signed_mod_delta_m(de, di, 256)
        lanes[0].extend(_deint4(_zig(diff_d)))
        lanes[1].extend(_deint4(di))
        lanes[15].extend([s15a, s15b])
        sep = [0] * 24
        for c, ln in enumerate((2, 3, 4, 5)):
            bits = 8 if c == 0 else 7
            sh = 8 - bits
            lanes[ln].extend([
                _zig(_signed_mod_delta(ch[c][0], rep[c] >> sh, bits)),
                _zig(_signed_mod_delta(ch[c][1], rep[4 + c] >> sh, bits)),
            ])
            sep[c] = ch[c][0] << sh
            sep[4 + c] = ch[c][1] << sh
        eps_state.append(sep * 3)
        for ln in (6, 7, 8, 9, 10):
            lanes[ln].extend([0, 0])
        for y in range(4):
            ds = [_zig(_signed_mod_delta(ixs[y * 4 + x], rix[y * 4 + x], 4)) for x in range(4)]
            lanes[11 + y].extend([_pack_ix_pair(ds[0], ds[1], y), _pack_ix_pair(ds[2], ds[3], y)])
        ixs_state.append(list(ixs) + [0] * 16)
    return encode_lanes(lanes, 7)


def _encode_bc6h_mode14_tiled(tiled_bytes, *, secret_entry=None, secret_eh=None, secret_ih=None, ref_count=4):
    """Encode Mode-14 (code 0x0F, 16-bit base, 4-bit delta, 1-region) BC6H blocks into codec-6 BCPack."""
    nblocks = len(tiled_bytes) // 16
    lanes = [[] for _ in range(16)]
    for i in range(nblocks):
        parsed = calib.unpack_bc6h_block(tiled_bytes, i * 16)
        assert parsed is not None and parsed[0] == 14
        _mode, part, f, ixs = parsed
        if secret_entry is not None and i < ref_count:
            d = 256 + i - secret_entry
            di = d if secret_ih is not None else 0
            de = d if secret_eh is not None else 0
            eref = secret_eh if secret_eh is not None else bcpack._BC6H_NONE_E
            iref = secret_ih if secret_ih is not None else bcpack._BC6H_NONE_I
        else:
            di = 0
            de = 0
            eref = bcpack._BC6H_NONE_E
            iref = bcpack._BC6H_NONE_I
        diff_d = _signed_mod_delta_m(de, di, 256)
        lanes[0].extend(_deint4(_zig(diff_d)))
        lanes[1].extend(_deint4(di))
        lanes[15].extend([13, part])
        for c in range(3):
            dz0 = _signed_mod_delta(f[c][0], eref[c][0], 16)
            dz1 = _signed_mod_delta(f[c][1], eref[c][1] >> 6, 4)
            lanes[2 + c].extend(_deint_n(_zig(dz0), 8))
            lanes[5 + c].extend(_deint_n(_zig(dz1), 5))
            lanes[8 + c].extend([0, 0])
        for yy in range(4):
            ds = [_zig(_signed_mod_delta(ixs[yy * 4 + xx], iref[yy * 4 + xx], 4)) for xx in range(4)]
            lanes[11 + yy].extend([_pack_ix_pair(ds[0], ds[1], yy), _pack_ix_pair(ds[2], ds[3], yy)])
    return encode_lanes(lanes, 6)


# ---------------------------------------------------------------------------
# Unit test cases
# ---------------------------------------------------------------------------

class PrefixDecoderAndCliTest(unittest.TestCase):
    def test_standalone_help_under_isolated_python(self):
        script = os.path.join(ROOT, "tools", "bcpack_history_calib.py")
        proc = subprocess.run(
            [sys.executable, "-I", script, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--ref-dds", proc.stdout)
        self.assertIn("--verify", proc.stdout)

    def test_decode_lanes_prefix_matches_full_decode(self):
        for codec in (1, 4, 6, 7):
            widths = bcpack.LANE_WIDTHS[codec]
            nsym = 40
            lanes = [[((i * 19 + j * 11) & ((1 << dw) - 1)) for j in range(nsym)]
                     for i, dw in enumerate(widths)]
            raw = encode_lanes(lanes, codec)
            for k in (1, 7, 16, nsym):
                got = calib.decode_lanes_prefix(raw, codec, nsym * 8, k)
                self.assertEqual(got, [row[:k] for row in lanes])

    def test_cli_clean_errors_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_cases = [
                [],
                ["--out", os.path.join(tmp, "out")],
                ["--verify"],
                ["--codec", "bogus", "--verify", tmp],
                ["--limit", "-1", "--verify", tmp],
                ["--max-blocks", "0", "--verify", tmp],
                [os.path.join(tmp, "missing.ttpf"), "--out", os.path.join(tmp, "out")],
                ["--ref-dds", "bad_spec", "--out", os.path.join(tmp, "out")],
                ["--verify", os.path.join(tmp, "nonexistent.json")],
            ]
            for argv in bad_cases:
                err = io.StringIO()
                with redirect_stderr(err):
                    rc = calib.main(argv)
                self.assertEqual(rc, 2, f"argv={argv!r} returned {rc}")
                msg = err.getvalue()
                self.assertIn("ERROR:", msg)
                self.assertNotIn("Traceback", msg)


class ReferenceInversionTest(unittest.TestCase):
    def test_ref_pairs_inverts_all_six_codecs_for_strict_decode(self):
        # 1. BC1 custom entry k=3 (unreferenced in default tables)
        k_bc1 = 0
        sec_bc1 = (29, 45, 11, 7, 18, 25)
        c0 = (sec_bc1[0] << 11) | (sec_bc1[1] << 5) | sec_bc1[2]
        c1 = (sec_bc1[3] << 11) | (sec_bc1[4] << 5) | sec_bc1[5]
        blk_bc1 = struct.pack("<HHBBBB", c0, c1, 0x00, 0x55, 0xAA, 0xFF)
        inv_q1 = {v: i for i, v in enumerate(tables.BC1_QUADRANT)}
        l_bc1 = [[0] for _ in range(12)]
        l_bc1[0] = [_zig(0 - k_bc1)]  # d = 256 - 0 -> wait: d must be in 1..255 => use k_bc1 = 3 => d = 253 -> signed -3
        k_bc1 = 3
        # Pick an entry not in BC1_HISTORY_CONFIRMED (entry 0 is unconfirmed: d = 256 wraps to 0 on block 0, so use block 1 with d = 255 -> 256 + 1 - 255 = 2, wait: check unused entry or --replace!)
        # Entry 0 from block 1: d = 257 invalid; instead entry 6 is provisional (not confirmed)!
        k_bc1 = 6
        self.assertNotIn(k_bc1, tables.BC1_HISTORY_CONFIRMED)
        l_bc1 = [[0] for _ in range(12)]
        l_bc1[0] = [_zig((256 - k_bc1) - 256)]
        l_bc1[8] = [inv_q1[0x05]]
        l_bc1[9] = [inv_q1[0x50]]
        l_bc1[10] = [inv_q1[0xAF]]
        l_bc1[11] = [inv_q1[0xFA]]
        # Recompute exact BC1 block with zero deltas from sec_bc1 and those quadrants:
        qa, qb, qc, qd = 0x05, 0x50, 0xAF, 0xFA
        blk_bc1 = struct.pack(
            "<HHBBBB", c0, c1,
            (qa & 15) | ((qb & 15) << 4), (qa >> 4) | ((qb >> 4) << 4),
            (qc & 15) | ((qd & 15) << 4), (qc >> 4) | ((qd >> 4) << 4),
        )
        raw_bc1 = encode_lanes(l_bc1, 1)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_bc1, 1, 8, allow_provisional=False)

        # 2. BC3 custom entry k=12
        k_bc3 = 12
        self.assertNotIn(k_bc3, tables.BC3_HISTORY_CONFIRMED)
        sec_bc3 = (210, 40, 20, 50, 10, 5, 15, 25)
        l_bc3 = [[0, 0] for _ in range(12)]
        l_bc3[0] = list(_deint4(_zig(-k_bc3)))
        raw_bc3 = encode_lanes(l_bc3, 3)
        p0_3 = tables.BC3_ALPHA_PAIR[0]
        abits_3 = 0
        for r in range(4):
            abits_3 |= (p0_3[0] | (p0_3[1] << 3) | (p0_3[0] << 6) | (p0_3[1] << 9)) << (12 * r)
        q0_3 = tables.BC3_COLOUR_QUADRANT[0]
        bc3_c0 = (sec_bc3[2] << 11) | (sec_bc3[3] << 5) | sec_bc3[4]
        bc3_c1 = (sec_bc3[5] << 11) | (sec_bc3[6] << 5) | sec_bc3[7]
        blk_bc3 = (
            bytes([sec_bc3[0], sec_bc3[1]])
            + abits_3.to_bytes(6, "little")
            + struct.pack(
                "<HHBBBB", bc3_c0, bc3_c1,
                (q0_3 & 15) | ((q0_3 & 15) << 4), (q0_3 >> 4) | ((q0_3 >> 4) << 4),
                (q0_3 & 15) | ((q0_3 & 15) << 4), (q0_3 >> 4) | ((q0_3 >> 4) << 4),
            )
        )
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_bc3, 3, 16, allow_provisional=False)

        # 3. BC4 custom entry k=20
        k_bc4 = 20
        self.assertNotIn(k_bc4, tables.BC4_HISTORY_CONFIRMED)
        sec_bc4 = (195, 62)
        l_bc4 = [[0] for _ in range(12)]
        l_bc4[0] = [_zig(-k_bc4)]
        raw_bc4 = encode_lanes(l_bc4, 4)
        p0_4 = tables.BC4_PAIR[0]
        abits_4 = 0
        for r in range(4):
            abits_4 |= (p0_4[0] | (p0_4[1] << 3) | (p0_4[0] << 6) | (p0_4[1] << 9)) << (12 * r)
        blk_bc4 = bytes(sec_bc4) + abits_4.to_bytes(6, "little")

        # 4. BC5 custom entry k=30
        k_bc5 = 30
        self.assertNotIn(k_bc5, tables.BC5_HISTORY_CONFIRMED)
        sec_bc5 = (180, 90, 220, 35)
        l_bc5 = [[0, 0] for _ in range(12)]
        l_bc5[0] = list(_deint4(_zig(-k_bc5)))
        raw_bc5 = encode_lanes(l_bc5, 5)
        p0_5 = tables.BC5_PAIR[0]
        abits_5 = 0
        for r in range(4):
            abits_5 |= (p0_5[0] | (p0_5[1] << 3) | (p0_5[0] << 6) | (p0_5[1] << 9)) << (12 * r)
        blk_bc5 = (
            bytes([sec_bc5[0], sec_bc5[1]]) + abits_5.to_bytes(6, "little")
            + bytes([sec_bc5[2], sec_bc5[3]]) + abits_5.to_bytes(6, "little")
        )

        # 5. BC6H custom entry k=42 (Mode 14 block)
        k_bc6 = 42
        sec_eh6 = ((0x3456, 0x240, 0x20, 0x20), (0x789A, 0x180, 0x20, 0x20), (0x1234, 0x2C0, 0x20, 0x20))
        sec_ih6 = (2, 9, 14, 5, 3, 11, 7, 1, 12, 4, 8, 15, 6, 10, 13, 0)
        # Pack Mode 14 (code 0x0F) block matching zero deltas off sec_eh6/sec_ih6
        f6 = [[bcpack._bc6h_stream_w(14, sec_eh6[c][0]), sec_eh6[c][1] >> 6, 0, 0] for c in range(3)]
        plan6, hdr6 = bcpack._bc6h_plan(0x0F)
        acc6 = 0x0F
        for c, q, bit_idx, pos in plan6:
            if c >= 0 and ((f6[c][q] >> bit_idx) & 1):
                acc6 |= 1 << pos
        at6 = hdr6
        for p in range(16):
            nb = 3 if p == 0 else 4
            acc6 |= (sec_ih6[p] & ((1 << nb) - 1)) << at6
            at6 += nb
        blk_bc6 = acc6.to_bytes(16, "little")
        raw_bc6 = _encode_bc6h_mode14_tiled(
            blk_bc6, secret_entry=k_bc6, secret_eh=sec_eh6, secret_ih=sec_ih6, ref_count=1
        )
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_bc6, 6, 16)

        # 6. BC7 custom entry k=19 exercised by BOTH Mode 6 AND Mode 4 (rot=2, isel=1)
        k_bc7 = 19
        tab7 = bptc.load_bc7()
        # Mode 4 block at block 0 referencing entry k_bc7 for both endpoints and dual indices:
        s_m4 = divmod(tab7["rank_of"].index(tab7["config"].index((4, 0, 2, 1))), 16)
        ep_m4 = ((0x15, 0x2B, 0x0C, 0x1A), (0x09, 0x34, 0x1E, 0x07))
        s1_m4 = [2, 7, 5, 1, 6, 3, 4, 0, 7, 2, 1, 5, 3, 6, 0, 4]  # 3-bit rows1
        s2_m4 = [1, 3, 2, 0, 2, 1, 3, 0, 3, 2, 1, 0, 1, 2, 3, 0]  # 2-bit rows2
        l_bc7 = [[0, 0] for _ in range(16)]
        l_bc7[1] = list(_deint4(256 - k_bc7))  # di = de = 256 - k_bc7
        l_bc7[15] = list(s_m4)
        raw_bc7 = encode_lanes(l_bc7, 7)
        acc7 = (1 << 4) | (2 << 5) | (1 << 7)
        pos7 = 8
        for c in range(3):
            tc = 3 if c == 1 else c
            for e in range(2):
                acc7 |= (ep_m4[e][tc] & 0x1F) << pos7
                pos7 += 5
        for e in range(2):
            acc7 |= (ep_m4[e][1] & 0x3F) << pos7
            pos7 += 6
        for p in range(16):
            nb = 1 if p == 0 else 2
            acc7 |= (s2_m4[p] & ((1 << nb) - 1)) << pos7
            pos7 += nb
        for p in range(16):
            nb = 2 if p == 0 else 3
            acc7 |= (s1_m4[p] & ((1 << nb) - 1)) << pos7
            pos7 += nb
        blk_bc7 = acc7.to_bytes(16, "little")
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_bc7, 7, 16)

        with tempfile.TemporaryDirectory() as tmp:
            pairs_json = os.path.join(tmp, "pairs.json")
            with open(pairs_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pairs": [
                            {"codec": "bc1", "chunk_hex": raw_bc1.hex(), "ref_hex": blk_bc1.hex()},
                            {"codec": "bc3", "chunk_hex": raw_bc3.hex(), "ref_hex": blk_bc3.hex()},
                            {"codec": "bc4", "chunk_hex": raw_bc4.hex(), "ref_hex": blk_bc4.hex()},
                            {"codec": "bc5", "chunk_hex": raw_bc5.hex(), "ref_hex": blk_bc5.hex()},
                            {"codec": "bc6h", "chunk_hex": raw_bc6.hex(), "ref_hex": blk_bc6.hex()},
                            {"codec": "bc7", "chunk_hex": raw_bc7.hex(), "ref_hex": blk_bc7.hex()},
                        ]
                    },
                    fh,
                )

            out_dir = os.path.join(tmp, "hist_dir")
            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = calib.main(["--ref-pairs", pairs_json, "--codec", "all", "--out", out_dir, "--check", "--json"])
            self.assertEqual(rc, 0)
            rep = json.loads(out_buf.getvalue())
            self.assertEqual(rep["files_written"], 6)
            self.assertTrue(rep["verify"]["schema_ok"])

            # Every single chunk must now decode in STRICT mode and match byte-for-byte!
            for cid, raw, ref in (
                (1, raw_bc1, blk_bc1),
                (3, raw_bc3, blk_bc3),
                (4, raw_bc4, blk_bc4),
                (5, raw_bc5, blk_bc5),
                (6, raw_bc6, blk_bc6),
                (7, raw_bc7, blk_bc7),
            ):
                dec, prov = bcpack.decode_with_info(raw, cid, len(ref), allow_provisional=False, history=out_dir)
                self.assertEqual(dec, ref, f"codec {cid} mismatch")
                self.assertEqual(prov, set(), f"codec {cid} had provisional tags {prov}")

            # Also verify combined single-file JSON output and deterministic byte-identical re-run
            combined = os.path.join(tmp, "combined.json")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(calib.main(["--ref-pairs", pairs_json, "--codec", "all", "--out", combined]), 0)
            with open(combined, "rb") as fh:
                first_bytes = fh.read()
            self.assertNotIn(b"\r\n", first_bytes)
            combined2 = os.path.join(tmp, "combined2.json")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(calib.main(["--ref-pairs", pairs_json, "--codec", "all", "--out", combined2]), 0)
            with open(combined2, "rb") as fh:
                self.assertEqual(first_bytes, fh.read())
            self.assertEqual(bcpack.decode(raw_bc7, 7, 16, history=combined), blk_bc7)

            # Verify --replace on BC1 drops default built-in entries not in the JSON
            rep_json = os.path.join(tmp, "bc1_replace.json")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    calib.main(["--ref-pairs", pairs_json, "--codec", "bc1", "--replace", "--out", rep_json]),
                    0,
                )
            self.assertEqual(bcpack.decode(raw_bc1, 1, 8, history=rep_json), blk_bc1)

    def test_reference_dds_does_not_confirm_tiled_padding(self):
        # Only one BC6H block is in this 4x4 reference. The other 255 blocks
        # of its 4 KiB tile are unknown, not exact zero-valued references.
        lanes = [[0] * 512 for _ in range(16)]
        lanes[1] = [15] * 512  # distance 255 -> initial entries 1..255
        raw = encode_lanes(lanes, 6)
        record = _pack_etpf(4, 4, 1, 0, 95, 4096, mode=2)
        calibrator = calib.Bc6hCalibrator()
        with tempfile.TemporaryDirectory() as temp:
            pool = os.path.join(temp, "partial.ttpf")
            reference = os.path.join(temp, "partial.dds")
            _write_custom_rds5_pool(pool, [record], [(6, 4096, raw)])
            _write_dx10_dds(reference, 95, 4, 4, [bytes(16)])
            calib.apply_ref_dds(pool, 0, reference, {6: calibrator})
        self.assertEqual(sum(calibrator.ref_ep_seen), 1)
        self.assertEqual(sum(calibrator.ref_ix_seen), 1)
        self.assertTrue(calibrator.ref_ep_seen[1])

    def test_reference_dds_masks_scalar_and_bc7_padding(self):
        for codec, dxgi in ((1, 71), (7, 98)):
            with self.subTest(codec=codec):
                if codec == 1:
                    lanes = [[0] * 512 for _ in bcpack.LANE_WIDTHS[codec]]
                    lanes[0] = [1] * 512  # -1 -> distance 255
                    block = bytes(8)
                    calibrator = calib.BcSimpleCalibrator(codec)
                else:
                    lanes = [[0] * 512 for _ in bcpack.LANE_WIDTHS[codec]]
                    lanes[1] = [15] * 512
                    tab = bptc.load_bc7()
                    rank = tab["config"].index((6, 0, 0, 0))
                    aa, bb = divmod(tab["rank_of"].index(rank), 16)
                    lanes[15] = [aa, bb] * 256
                    block = bytes([1 << 6]) + bytes(15)
                    calibrator = calib.Bc7Calibrator()
                raw = encode_lanes(lanes, codec)
                record = _pack_etpf(4, 4, 1, 0, dxgi, 4096, mode=2)
                with tempfile.TemporaryDirectory() as temp:
                    pool = os.path.join(temp, "partial.ttpf")
                    reference = os.path.join(temp, "partial.dds")
                    _write_custom_rds5_pool(pool, [record], [(codec, 4096, raw)])
                    _write_dx10_dds(reference, dxgi, 4, 4, [block])
                    calib.apply_ref_dds(pool, 0, reference, {codec: calibrator})
                self.assertEqual(calibrator.ref_blocks, 1)
                if codec == 1:
                    self.assertEqual(set(calibrator.confirmed), {1})

    def test_reference_dds_preserves_both_mips_in_one_tail_tile(self) -> None:
        width = height = 16
        fmt, codec = 71, 1
        layout = ttpf.surface_layout(width, height, fmt, 2, ttpf.tile_bytes(4))
        first, second = layout["mips"]
        self.assertTrue(first["in_tail"] and second["in_tail"])
        self.assertEqual(first["offset"], second["offset"])
        mips = [bytes.fromhex("00 f8 00 00 00 00 00 00") * 16,
                bytes.fromhex("e0 07 00 00 00 00 00 00") * 4]
        region = first["region"]
        raw = encode_lanes([[0] * (region // 8) for _ in bcpack.LANE_WIDTHS[codec]], codec)
        record = _pack_etpf(width, height, 2, 0, fmt, 4096)

        with tempfile.TemporaryDirectory() as temp:
            pool_path = os.path.join(temp, "tail.ttpf")
            dds_path = os.path.join(temp, "tail.dds")
            _write_custom_rds5_pool(pool_path, [record], [(codec, region, raw)])
            _write_dx10_dds(dds_path, fmt, width, height, mips)
            calibrator = Mock()
            calib.apply_ref_dds(pool_path, 0, dds_path, {codec: calibrator})

        self.assertEqual(calibrator.add_reference_chunk.call_count, 1)
        _, reference, _ = calibrator.add_reference_chunk.call_args.args
        self.assertEqual([ttpf.detile(layout, mip, reference) for mip in (first, second)], mips)
        valid = calibrator.add_reference_chunk.call_args.kwargs["valid_blocks"]
        expected = [False] * 256
        for mip in (first, second):
            for y in range(mip["blocks_h"]):
                for x in range(mip["blocks_w"]):
                    index = calib.tiled_offset(layout, mip, x, y) // 8
                    if index < 256:
                        expected[index] = True
        self.assertEqual(valid, expected)

    def test_ref_dds_with_synthetic_ttpf_pool_and_ttpf_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            width, height, mips_n, fmt = 64, 64, 2, 98  # DXGI_FORMAT_BC7_UNORM
            lay = ttpf.surface_layout(width, height, fmt, mips_n, ttpf.tile_bytes(4))
            # Build linear Mode-6 BC7 blocks for mip 0 (16x16 blocks = 4096B) and mip 1 (8x8 blocks in 4096B tail)
            mip_linear = []
            for m, mip in enumerate(lay["mips"]):
                buf = bytearray()
                scale = 1 if m == 0 else 2
                half = 0 if m == 0 else 4
                for by in range(mip["blocks_h"]):
                    for bx in range(mip["blocks_w"]):
                        r = min(254, (40 + bx * 8 * scale + half) & 0xFE)
                        g = min(254, (60 + by * 8 * scale + half) & 0xFE)
                        b = 120
                        # Mode 6 solid block
                        acc = (1 << 6)
                        pos = 7
                        for v in (r >> 1, r >> 1, g >> 1, g >> 1, b >> 1, b >> 1, 127, 127):
                            acc |= (v & 0x7F) << pos
                            pos += 7
                        acc |= (r & 1) << pos
                        acc |= (r & 1) << (pos + 1)
                        buf += acc.to_bytes(16, "little")
                mip_linear.append(bytes(buf))

            dds_path = os.path.join(tmp, "ref.dds")
            _write_dx10_dds(dds_path, fmt, width, height, mip_linear)

            # Retile mip 1 (tail at offset 0, 4096B) as zlib, mip 0 (at offset 4096, 4096B) as BCPack codec 7
            tiled_tail = calib.retile_mip(mip_linear[1], lay, lay["mips"][1])
            tiled_top = calib.retile_mip(mip_linear[0], lay, lay["mips"][0])
            k_sec = 77
            sec_ep = tuple([0x54, 0x88, 0xAA, 0xFE, 0x54, 0x88, 0xAA, 0xFE] * 3)
            sec_ix = tuple([2, 4, 6, 8] * 4 + [0] * 16)
            raw_top = _encode_bc7_mode6_tiled(
                tiled_top, secret_entry=k_sec, secret_ep=sec_ep, secret_ix=sec_ix, ref_count=6
            )

            pool_path = os.path.join(tmp, "test.ttpf")
            # Calculate padded base offset for 1 record and 2 chunks
            base = ttpf.align(16 + 24 + 32, 4096)
            rec = _pack_etpf(width, height, 1, 1, fmt, base, mode=4)
            _write_custom_rds5_pool(
                pool_path,
                [rec],
                [
                    (8, lay["mips"][1]["region"], zlib.compress(tiled_tail, 9)),
                    (7, lay["mips"][0]["region"], raw_top),
                ],
            )

            # Without --history, ttpf_mips check refuses even with --allow-provisional (6 unknown blocks > 2)
            p = ttpf.Pool(pool_path)
            try:
                out_s = io.StringIO()
                with redirect_stdout(out_s):
                    self.assertEqual(ttpf.check(p, 0, allow_provisional=True), 2)
            finally:
                p.close()

            hist_dir = os.path.join(tmp, "hist")
            spec = f"{pool_path}:0={dds_path}"
            out_s = io.StringIO()
            with redirect_stdout(out_s):
                rc = calib.main(["--ref-dds", spec, "--codec", "bc7", "--out", hist_dir, "--check"])
            self.assertEqual(rc, 0)

            # With --history (STRICT mode, allow_provisional=False!), ttpf_mips check & export succeed cleanly!
            exp_dds = os.path.join(tmp, "exported.dds")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ttpf.main(["check", pool_path, "0", "--history", hist_dir]), 0)
            self.assertEqual(ttpf.main(["export", pool_path, "0", exp_dds, "--history", hist_dir]), 0)
            _, _, _, exp_mips = calib.read_dds(exp_dds)
            self.assertEqual(exp_mips, mip_linear)


class Bc6hConfirmationDepthTest(unittest.TestCase):
    """Exact references confirm only the leading bits their modes consume."""

    @staticmethod
    def lanes(modes, *, endpoints=True, indices=False, chain=False):
        lanes = [[] for _ in range(16)]
        for i, mode in enumerate(modes):
            block = [[0, 0] for _ in range(16)]
            de = (1 if chain else i + 1) if endpoints else 0
            di = (1 if chain else i + 1) if indices else 0
            block[0] = list(_deint4(_zig(de - di)))
            block[1] = list(_deint4(di))
            block[15] = [mode - 1, 0]
            reg = bcpack._BC6H_INFO[bcpack._BC6H_CODE[mode]][3]
            delta = 0 if di else _zig(-4 if reg == 2 else -8)
            for y in range(4):
                block[11 + y] = [_pack_ix_pair(delta, delta, y)] * 2
            for lane, symbols in zip(lanes, block):
                lane.extend(symbols)
        return lanes

    @staticmethod
    def truth():
        return {
            "endpointHistory": {"255": [[32768, 544, 0, 0] for _ in range(3)]},
            "endpointClass": {"255": 3},
            "indexHistory": {"255": [1] * 16},
            "indexClass": {"255": 3},
        }

    def calibrate(self, lanes):
        raw = encode_lanes(lanes, 6)
        reference = bcpack.decode(raw, 6, len(lanes[0]) * 8, history=self.truth())
        cal = calib.Bc6hCalibrator()
        cal.add_reference_chunk(lanes, reference, len(lanes[0]) // 2)
        self.assertEqual(cal.conflicts, 0)
        doc, _stats = cal.build_document()
        self.assertEqual(bcpack.decode(raw, 6, len(reference), history=doc), reference)
        return doc

    def test_endpoint_reference_does_not_confirm_unobserved_fields(self):
        doc = self.calibrate(self.lanes([14]))
        raw = encode_lanes(self.lanes([1]), 6)
        expected = bcpack.decode(raw, 6, 16, history=self.truth())
        actual, labels = bcpack.decode_with_info(raw, 6, 16, history=doc, allow_provisional=True)
        self.assertNotEqual(actual, expected)
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "unconfirmed.*endpoint.*255"):
            bcpack.decode(raw, 6, 16, history=doc)
        self.assertTrue(any("endpoint history 255" in label for label in labels))
        self.assertEqual(doc["endpointConfirmedBits"][255], [[16, 4, 0, 0]] * 3)

    def test_index_reference_does_not_confirm_unobserved_low_bit(self):
        doc = self.calibrate(self.lanes([1], endpoints=False, indices=True))
        raw = encode_lanes(self.lanes([14], endpoints=False, indices=True), 6)
        self.assertNotEqual(
            bcpack.decode(raw, 6, 16, history=doc, allow_provisional=True),
            bcpack.decode(raw, 6, 16, history=self.truth()),
        )
        with self.assertRaisesRegex(bcpack.BcpackProvisionalError, "unconfirmed.*index.*255"):
            bcpack.decode(raw, 6, 16, history=doc)
        _actual, labels = bcpack.decode_with_info(raw, 6, 16, history=doc, allow_provisional=True)
        self.assertTrue(any("index history 255" in label for label in labels))
        self.assertEqual(doc["indexConfirmedBits"][255], [3] * 16)

    def test_endpoint_chain_uses_reconstructed_state_after_truncation(self):
        doc = self.calibrate(self.lanes([14]))
        lanes = self.lanes([14, 1], chain=True)
        raw = encode_lanes(lanes, 6)
        self.assertEqual(
            bcpack.decode(raw, 6, 32, history=doc),
            bcpack.decode(raw, 6, 32, history=self.truth()),
        )
        # A later mode cannot make an earlier incomplete history read safe.
        raw = encode_lanes(self.lanes([1, 14], chain=True), 6)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw, 6, 32, history=doc)

    def test_every_channel_and_field_must_have_sufficient_depth(self):
        raw = encode_lanes(self.lanes([1]), 6)
        for channel in range(3):
            for field, needed in enumerate((10, 5, 5, 5)):
                doc = self.truth()
                bits = [[10, 5, 5, 5] for _ in range(3)]
                doc["endpointConfirmedBits"] = {"255": bits}
                bcpack.decode(raw, 6, 16, history=doc)
                bits[channel][field] = needed - 1
                with self.subTest(channel=channel, field=field):
                    with self.assertRaises(bcpack.BcpackProvisionalError):
                        bcpack.decode(raw, 6, 16, history=doc)

    def test_every_index_pixel_must_have_sufficient_depth(self):
        raw = encode_lanes(self.lanes([14], endpoints=False, indices=True), 6)
        for pixel in range(16):
            doc = self.truth()
            bits = [4] * 16
            doc["indexConfirmedBits"] = {"255": bits}
            bits[pixel] = 3
            with self.subTest(pixel=pixel):
                with self.assertRaises(bcpack.BcpackProvisionalError):
                    bcpack.decode(raw, 6, 16, history=doc)

    def test_all_reference_modes_keep_their_own_blocks_strict(self):
        for mode in range(1, 15):
            with self.subTest(mode=mode):
                self.calibrate(self.lanes([mode], indices=True))

    def test_reference_can_serve_a_different_lower_precision_mode(self):
        # Mode 2 observes 7/6/6/6 bits; mode 10 needs only 6/6/6/6.
        doc = self.calibrate(self.lanes([2], indices=True))
        raw = encode_lanes(self.lanes([10], indices=True), 6)
        self.assertEqual(
            bcpack.decode(raw, 6, 16, history=doc),
            bcpack.decode(raw, 6, 16, history=self.truth()),
        )
        # Mode 1 needs ten base bits, more than that reference observed.
        raw = encode_lanes(self.lanes([1], indices=True), 6)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw, 6, 16, history=doc)

    def test_explicit_depth_does_not_upgrade_provisional_classes(self):
        raw = encode_lanes(self.lanes([1], indices=True), 6)
        for kind in ("endpoint", "index"):
            doc = self.truth()
            doc["endpointConfirmedBits"] = {"255": [[16, 10, 6, 6]] * 3}
            doc["indexConfirmedBits"] = {"255": [4] * 16}
            doc[kind + "Class"]["255"] = 1
            with self.subTest(kind=kind):
                with self.assertRaises(bcpack.BcpackProvisionalError):
                    bcpack.decode(raw, 6, 16, history=doc)

    def test_partial_index_chain_uses_reconstructed_four_bit_state(self):
        doc = self.calibrate(self.lanes([1], endpoints=False, indices=True))
        lanes = self.lanes([1, 14], endpoints=False, indices=True, chain=True)
        raw = encode_lanes(lanes, 6)
        self.assertEqual(
            bcpack.decode(raw, 6, 32, history=doc),
            bcpack.decode(raw, 6, 32, history=self.truth()),
        )

    def test_confirmation_depth_survives_json_and_schema_validation(self):
        doc = self.calibrate(self.lanes([14]))
        raw = encode_lanes(self.lanes([1]), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bc6h.json")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"bc6h": doc}, fh)
            self.assertIn("bc6h", calib.validate_history_schema(path, [6]))
            for history in (path, tmp):
                with self.subTest(history=history):
                    with self.assertRaises(bcpack.BcpackProvisionalError):
                        bcpack.decode(raw, 6, 16, history=history)
            doc["endpointConfirmedBits"][255][0][0] = 17
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"bc6h": doc}, fh)
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(calib.main(["--verify", path, "--codec", "bc6h"]), 2)
            self.assertIn("bit depth out of bounds", err.getvalue())


class PoolSelfCalibrationTest(unittest.TestCase):
    def test_mixed_bc7_and_bc6h_pool_self_calibration_unlocks_provisional_check_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            width, height, mips_n = 64, 64, 2

            # --- Mixed BC7 pool (mip 0 = BCPack codec 7 with unknown entry k=91, mip 1 = zlib) ---
            lay7 = ttpf.surface_layout(width, height, 98, mips_n, ttpf.tile_bytes(4))
            lin7 = []
            for m, mip in enumerate(lay7["mips"]):
                buf = bytearray()
                scale = 1 if m == 0 else 2
                half = 0 if m == 0 else 3
                for by in range(mip["blocks_h"]):
                    for bx in range(mip["blocks_w"]):
                        r = min(254, (32 + bx * 6 * scale + half) & 0xFE)
                        g = min(254, (80 + by * 6 * scale + half) & 0xFE)
                        b = 140
                        acc = 1 << 6
                        pos = 7
                        for v in (r >> 1, r >> 1, g >> 1, g >> 1, b >> 1, b >> 1, 127, 127):
                            acc |= (v & 0x7F) << pos
                            pos += 7
                        acc |= (r & 1) << pos
                        acc |= (r & 1) << (pos + 1)
                        buf += acc.to_bytes(16, "little")
                lin7.append(bytes(buf))
            t7_tail = calib.retile_mip(lin7[1], lay7, lay7["mips"][1])
            t7_top = calib.retile_mip(lin7[0], lay7, lay7["mips"][0])
            sec7_ep = tuple([0x40, 0x60, 0x90, 0xFE, 0x40, 0x60, 0x90, 0xFE] * 3)
            sec7_ix = tuple([0] * 32)
            raw7_top = _encode_bc7_mode6_tiled(
                t7_top, secret_entry=91, secret_ep=sec7_ep, secret_ix=sec7_ix, ref_count=8
            )
            pool7 = os.path.join(tmp, "mixed_bc7.ttpf")
            base = ttpf.align(16 + 24 + 32, 4096)
            _write_custom_rds5_pool(
                pool7,
                [_pack_etpf(width, height, 1, 1, 98, base, mode=4)],
                [(8, lay7["mips"][1]["region"], zlib.compress(t7_tail, 9)), (7, lay7["mips"][0]["region"], raw7_top)],
            )

            # --- Mixed BC6H pool (mip 0 = BCPack codec 6 with unknown entry k=55, mip 1 = zlib) ---
            lay6 = ttpf.surface_layout(width, height, 95, mips_n, ttpf.tile_bytes(4))
            lin6 = []
            for m, mip in enumerate(lay6["mips"]):
                buf = bytearray()
                scale = 1 if m == 0 else 2
                half6 = 0 if m == 0 else 8
                for by in range(mip["blocks_h"]):
                    for bx in range(mip["blocks_w"]):
                        # Unquantized 16-bit unsigned endpoints in Mode 14 (code 0x0F, D3D11 order)
                        rw = 0x3800 + bx * 16 * scale + half6
                        gw = 0x3A00 + by * 16 * scale + half6
                        bw = 0x3C00
                        f6 = [[wv, 0, 0, 0] for wv in (rw, gw, bw)]
                        plan6, _ = bcpack._bc6h_plan(0x0F)
                        acc6 = 0x0F
                        for c, q, bit_idx, pos in plan6:
                            if c >= 0 and ((f6[c][q] >> bit_idx) & 1):
                                acc6 |= 1 << pos
                        buf += acc6.to_bytes(16, "little")
                lin6.append(bytes(buf))
            t6_tail = calib.retile_mip(lin6[1], lay6, lay6["mips"][1])
            t6_top = calib.retile_mip(lin6[0], lay6, lay6["mips"][0])
            sec6_eh = ((0x1800, 0x200, 0x20, 0x20), (0x2800, 0x200, 0x20, 0x20), (0x3000, 0x200, 0x20, 0x20))
            sec6_ih = (0,) * 16
            raw6_top = _encode_bc6h_mode14_tiled(
                t6_top, secret_entry=55, secret_eh=sec6_eh, secret_ih=sec6_ih, ref_count=8
            )
            pool6 = os.path.join(tmp, "mixed_bc6h.ttpf")
            _write_custom_rds5_pool(
                pool6,
                [_pack_etpf(width, height, 1, 1, 95, base, mode=4)],
                [(8, lay6["mips"][1]["region"], zlib.compress(t6_tail, 9)), (6, lay6["mips"][0]["region"], raw6_top)],
            )

            # Both pools must refuse without --history even with --allow-provisional (8+ unknown blocks > 2)
            with redirect_stdout(io.StringIO()):
                self.assertNotEqual(ttpf.main(["check", pool7, "0", "--allow-provisional"]), 0)
                self.assertNotEqual(ttpf.main(["check", pool6, "0", "--allow-provisional"]), 0)

            # Run self-calibration purely from the synthetic pools (zero reference DDS!)
            hist_dir = os.path.join(tmp, "self_hist")
            out_s = io.StringIO()
            with redirect_stdout(out_s):
                rc = calib.main([tmp, "--out", hist_dir, "--codec", "bc6h,bc7", "--limit", "5", "--check", "--json"])
            self.assertEqual(rc, 0)
            rep = json.loads(out_s.getvalue())
            self.assertEqual(rep["verify"]["pool_check"]["bc7"]["provisional_ok"], 1)
            self.assertEqual(rep["verify"]["pool_check"]["bc6h"]["provisional_ok"], 1)

            # Strict still refuses (self-calibration produces provisional, not falsely confirmed, entries)
            with redirect_stdout(io.StringIO()):
                self.assertNotEqual(ttpf.main(["check", pool7, "0", "--history", hist_dir]), 0)
                self.assertNotEqual(ttpf.main(["check", pool6, "0", "--history", hist_dir]), 0)

            # Provisional + --history passes check AND exports full 2-mip DDS chains for both BC7 and BC6H!
            chk7_out = io.StringIO()
            with redirect_stdout(chk7_out):
                self.assertEqual(
                    ttpf.main(["check", pool7, "0", "--history", hist_dir, "--allow-provisional"]),
                    0,
                )
            chk7_json = json.loads(chk7_out.getvalue())
            self.assertLess(chk7_json["max_error"], 2.0)

            chk6_out = io.StringIO()
            with redirect_stdout(chk6_out):
                self.assertEqual(
                    ttpf.main(["check", pool6, "0", "--history", hist_dir, "--allow-provisional"]),
                    0,
                )
            chk6_json = json.loads(chk6_out.getvalue())
            self.assertLess(chk6_json["max_error"], 2.0)

            dds7 = os.path.join(tmp, "out7.dds")
            dds6 = os.path.join(tmp, "out6.dds")
            self.assertEqual(
                ttpf.main(["export", pool7, "0", dds7, "--history", hist_dir, "--allow-provisional"]),
                0,
            )
            self.assertEqual(
                ttpf.main(["export", pool6, "0", dds6, "--history", hist_dir, "--allow-provisional"]),
                0,
            )
            self.assertEqual(len(calib.read_dds(dds7)[3]), 2)
            self.assertEqual(len(calib.read_dds(dds6)[3]), 2)

            # Corrupting schema values causes --verify to fail cleanly with exit code 2
            bad_bc7 = os.path.join(tmp, "bad_bc7.json")
            with open(os.path.join(hist_dir, "bc7.json"), "r", encoding="utf-8") as fh:
                bad_doc = json.load(fh)
            bad_doc["historyEndpointBits"][0][0] = 99
            with open(bad_bc7, "w", encoding="utf-8") as fh:
                json.dump(bad_doc, fh)
            err_s = io.StringIO()
            with redirect_stderr(err_s):
                self.assertEqual(calib.main(["--verify", bad_bc7, "--codec", "bc7"]), 2)
            self.assertIn("ERROR:", err_s.getvalue())


if __name__ == "__main__":
    unittest.main()
