"""Repackage the Opus streams of AWC audio banks into standard Ogg Opus files (lossless: no re-encoding).

    python -I awc_opus.py <bank.awc | dir of .awc> <out dir> [--limit N] [--jobs N] [--pre-skip N]

AWC layout (little-endian "ADAT"), as read by rpf9-explorer's OpusPreview.cpp: a stream table of u32
(id[28:0] | tag count[31:29]), then 8-byte tags (type[63:56], size[55:28], offset[27:0]). Tag 0x55 = data,
0x76 = format (u32 samples, u16 rate at +8, u32 per-channel Opus frame count at +12, codec at +28, channels at
+29). Codec 0x0B = Opus at 48 kHz: data is 2048-byte blocks of [u16 length][packet] ..., a zero length pads the
rest of the block. Codec 0 = PCM16 (written as .wav). Codec 2 = 32-bit float PCM (float .wav).

Opus streams carry the encoder's lookahead as priming samples: the packets start with 312 samples of encoder
delay (libopus at 48 kHz) that RFC 7845 says the decoder drops using OpusHead's pre-skip. The AWC format block
does not store it, so the exporter writes OPUS_PRE_SKIP (overridable with --pre-skip) into OpusHead and makes the
trimmed final granule `pre_skip + length`. The format block's +0 count is interleaved across channels and already
in 48 kHz frames even when its rate field is 32 kHz; +0x0C is the per-channel frame count (equal to +0 for mono
and half of it for stereo in all 7,451 measured streams).

Streamed music (tag 0x48 descriptor: u32 blocks, u32 chunk size, u32 channels, then 16 bytes per channel: id,
samples, rate at +10, codec at +12) is split into one Ogg Opus per descriptor channel: each chunk starts with a
24-byte record per channel (subblock count at +4, repeated-prefix samples to skip at +8), the channel data are
its run of 2048-byte subblocks after the 2048-aligned header, and the first `skip` samples of a chunk repeat the
previous chunk's tail and are dropped (whole packets). The descriptor's `channels` is the number of channel
groups, not necessarily the packet channel count: an Opus packet's TOC stereo bit gives the real channel count,
and the descriptor's `samples` counts interleaved samples across those packet channels.

Output: <out>/<bank>/<stream id>.ogg (or .wav). Directory input preserves each bank's relative path.

For 16-bit PCM copies of the Ogg Opus files, run tools/audio_wav.py on the output (make_game.py does it with --wav).
Usage: `python -I tools/awc_opus.py --help` lists the options.
"""
import argparse
import functools
import json
from collections import Counter
import os
import struct
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import new_file_mode, published_mode  # noqa: E402
from pool_workers import process_workers  # noqa: E402
from rsc8_audio import decode_marker_records, marker_records, pair_marker_regions, parse_adat_header_bytes
from report_paths import path_key  # noqa: E402

# ---- Ogg ----
_CRC = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) if _r & 0x80000000 else (_r << 1)
    _CRC.append(_r & 0xFFFFFFFF)


def ogg_crc(data: bytes) -> int:
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFFFFFF) ^ _CRC[((c >> 24) ^ b) & 0xFF]
    return c


def ogg_page(serial: int, seq: int, granule: int, packets: list, flags: int) -> bytes:
    lacing = bytearray()
    for p in packets:
        n = len(p)
        while n >= 255:
            lacing.append(255)
            n -= 255
        lacing.append(n)
    if len(lacing) > 255:
        raise ValueError("packet needs more than 255 Ogg segments")
    hdr = struct.pack("<4sBBqIIIB", b"OggS", 0, flags, granule, serial, seq, 0, len(lacing)) + bytes(lacing)
    body = b"".join(packets)
    page = bytearray(hdr + body)
    struct.pack_into("<I", page, 22, ogg_crc(bytes(page)))
    return bytes(page)


def opus_samples(packet: bytes) -> int:
    """Samples at 48 kHz in one Opus packet (RFC 6716 TOC)."""
    toc = packet[0]
    config = toc >> 3
    if config < 12:
        frame = [480, 960, 1920, 2880][config & 3]          # SILK 10/20/40/60 ms
    elif config < 16:
        frame = [480, 960][config & 1]                        # hybrid 10/20 ms
    else:
        frame = [120, 240, 480, 960][config & 3]              # CELT 2.5/5/10/20 ms
    c = toc & 3
    if c == 0:
        frames = 1
    elif c in (1, 2):
        frames = 2
    else:
        frames = packet[1] & 0x3F if len(packet) > 1 else 0
    return frame * frames


# libopus lookahead at 48 kHz: the first 312 decoded samples of every packet stream are
# encoder priming, not audio. AWC does not store it, so this is the default the header declares.
OPUS_PRE_SKIP = 312


def write_ogg_opus(path: str, packets: list, channels: int, serial: int, length: int = 0,
                   pre_skip: int = OPUS_PRE_SKIP) -> int:
    """length: the stream's declared per-channel sample count at 48 kHz; the EOS page's granule position is
    trimmed to pre_skip + length (Ogg Opus granules count the pre-skip too), so decoders drop the encoder priming
    at the start and stop at the exact end instead of the padded last frame.

    The EOS page is emitted at the packet that first reaches pre_skip + length; any later packets are encoder
    padding and are dropped. Clamping only the final page would make its granule smaller than the previous page's
    when the trim point falls before that page starts, which is invalid Ogg and makes decoders keep the excess.
    """
    if not 0 <= pre_skip <= 0xFFFF:
        raise ValueError(f"pre-skip must be 0..65535, got {pre_skip}")
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    total = sum(opus_samples(p) for p in packets)
    if pre_skip > total:
        pre_skip = total          # a granule below the pre-skip makes the stream undecodable
    head = b"OpusHead" + struct.pack("<BBHIhB", 1, channels, pre_skip, 48000, 0, 0)
    tags = b"OpusTags" + struct.pack("<I", 8) + b"leonida " + struct.pack("<I", 0)
    out = [ogg_page(serial, 0, 0, [head], 0x02), ogg_page(serial, 1, 0, [tags], 0x00)]
    seq, granule, pending, size = 2, 0, [], 0
    target = pre_skip + length if length > 0 else None
    written = 0
    if not packets:
        out.append(ogg_page(serial, 2, pre_skip, [], 0x04))
        write_file(path, b"".join(out))
        return pre_skip
    for i, p in enumerate(packets):
        pending.append(p)
        granule += opus_samples(p)
        size += len(p)
        reach = target is not None and granule >= target
        last = i == len(packets) - 1
        if reach or size > 4000 or len(pending) >= 50 or last:
            g = target if reach else granule
            out.append(ogg_page(serial, seq, g, pending, 0x04 if (reach or last) else 0x00))
            written = g
            seq += 1
            pending, size = [], 0
            if reach or last:
                break
    write_file(path, b"".join(out))
    return written


@functools.cache
def folder_file_mode(directory: str) -> int:
    return new_file_mode(directory)


def write_file(path: str, data: bytes) -> None:
    """Publish one buffer using the same staged writer as multipart WAV output."""
    _write_file_parts(path, (data,))


def _write_file_parts(path: str, parts: tuple[bytes | memoryview, ...]) -> None:
    """Write under a private temporary name and rename into place: a stopped run or a full disk never leaves a
    short file under the final name, where a later run would keep it. The staging file is created in the
    destination directory (the same volume), so concurrent writers each publish a complete file (last wins)
    instead of clobbering one shared ".part" sibling."""
    out_path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(out_path))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=parent, prefix=".awc-", suffix=".tmp", delete=False) as f:
            tmp_path = f.name
            for part in parts:
                f.write(part)
        # NamedTemporaryFile creates 0600; give the staged file the mode of the file it replaces, or a new file's.
        os.chmod(tmp_path, published_mode(out_path) if os.path.exists(out_path) else folder_file_mode(parent))
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path is not None:
            try:
                os.chmod(tmp_path, 0o600)  # it may already carry a read-only mode, which Windows won't unlink
            except OSError:
                pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def write_wav(path: str, pcm: bytes | memoryview, rate: int, channels: int, float32: bool = False) -> None:
    # A typed or multidimensional view's len() counts elements, not bytes. Flatten the contiguous buffer
    # without copying before sizing the RIFF/data chunks and the float sample count.
    pcm = memoryview(pcm).cast("B")
    bits = 32 if float32 else 16
    block_align = channels * (bits // 8)
    fmt = struct.pack("<HHIIHH", 3 if float32 else 1, channels, rate, rate * block_align, block_align, bits)
    fact = b""
    if float32:
        fmt += struct.pack("<H", 0)  # WAVEFORMATEX.cbSize
        fact = struct.pack("<4sII", b"fact", 4, len(pcm) // block_align)
    header = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + fact + b"data" + struct.pack("<I", len(pcm))
    header = b"RIFF" + struct.pack("<I", len(header) + len(pcm)) + header
    _write_file_parts(path, (header, pcm))


# ---- AWC ----
def parse_awc(d: bytes):
    if d[:4] != b"ADAT":
        raise ValueError("not a little-endian AWC")
    flags, n, header_size = struct.unpack_from("<III", d, 4)
    p = 16 + (4 if flags & 0x400000 else 0)
    index_at = p if flags & 0x10000 else -1
    if index_at >= 0:
        p += n * 2
    if flags & 0x200000:
        p += (n + 1) * 4
    info = p
    p += n * 4
    streams, music = [], []
    tags = 0
    for i in range(n):
        v = struct.unpack_from("<I", d, info + 4 * i)[0]
        sid, ntags = v & 0x1FFFFFFF, v >> 29
        if index_at >= 0:
            index, = struct.unpack_from("<H", d, index_at + 2 * i)
            if index != tags:
                raise ValueError(f"AWC index entry {i} does not match the cumulative tag count")
        tags += ntags
        data = fmt = mus = None
        for _ in range(ntags):
            t = struct.unpack_from("<Q", d, p)[0]
            p += 8
            typ, size, at = t >> 56, (t >> 28) & 0x0FFFFFFF, t & 0x0FFFFFFF
            if typ in (0x55, 0x76, 0x48) and at + size > len(d):
                raise ValueError(f"AWC tag 0x{typ:02x} extends outside the bank")
            if typ == 0x55:
                data = (at, size)
            elif typ == 0x76:
                fmt = (at, size)
            elif typ == 0x48:
                mus = (at, size)
        if mus and data:
            music.append((mus, data))
        if data and fmt and fmt[1] >= 32:
            samples, = struct.unpack_from("<I", d, fmt[0])
            rate, = struct.unpack_from("<H", d, fmt[0] + 8)
            codec, channels = d[fmt[0] + 28], d[fmt[0] + 29]
            frames = (struct.unpack_from("<I", d, fmt[0] + 12)[0]
                      if codec == 0x0B and fmt[1] >= 16 else 0)
            streams.append({"id": sid, "samples": samples, "rate": rate, "codec": codec, "channels": channels,
                            "frames": frames, "data": data})
    return streams, music


def repeated_prefix_packets(packets: list[bytes], previous: list[bytes], samples: int) -> int | None:
    """Count whole leading packets that repeat the previous chunk's tail, or reject a bad skip."""
    for count, packet in enumerate(packets, 1):
        duration = opus_samples(packet)
        if not 0 < duration <= samples:
            return None
        samples -= duration
        if samples == 0:
            return count if packets[:count] == previous[-count:] else None
    return None


def music_channels(d: bytes, mus, data, collect_packets: bool = True):
    """[(channel id, samples, rate, codec, [packets], packet channels)] for one streamed-music descriptor.

    `samples` is the per-channel frame count. The descriptor stores the interleaved sample count, so it is
    divided by the packet channel count when the Opus packets are stereo. `packet channels` is read from the
    Opus TOC stereo bit (bit 2) and defaults to 1 when no packet is present.

    collect_packets=False keeps the packet lists empty for metadata-only callers (awc_manifest.py).
    The chunk size is the stride between blocks. The last block may be shorter, but each channel table and
    subblock run must fit in its own block. A declared repeat must match whole packets from the same channel's
    previous data. Channel rates outside 1000..192000 and non-Opus codecs are rejected as AwcRecovery.cs does.
    Reject malformed descriptors without hiding other bank streams."""
    (m_at, m_size), (d_at, d_size) = mus, data
    if m_size < 12:
        return []
    blocks, chunk, channels = struct.unpack_from("<III", d, m_at)
    if not (1 <= channels <= 32 and 1 <= blocks and chunk >= 2048 and m_size == 12 + 16 * channels):
        return []
    # Every block must start inside the data tag. The last block can be shorter than the chunk stride.
    if (blocks - 1) * chunk >= d_size:
        return []
    info = []
    for c in range(channels):
        base = m_at + 12 + 16 * c
        cid = struct.unpack_from("<I", d, base)[0] & 0x1FFFFFFF
        samples, = struct.unpack_from("<I", d, base + 4)
        rate, = struct.unpack_from("<H", d, base + 10)
        codec = d[base + 12]
        # The field is u16 (max 65535), so the 192000 half of the reference bound cannot fire here.
        if not (1000 <= rate <= 192000) or codec != 0x0B:
            return []
        info.append([cid, samples, rate, codec, [], 0])
    previous = [[] for _ in range(channels)]
    for b in range(blocks):
        offset = b * chunk
        at = d_at + offset
        block_size = min(chunk, d_size - offset)
        if channels * 24 > block_size:
            return []
        recs = [struct.unpack_from("<II", d, at + 24 * c + 4) for c in range(channels)]   # (count, skip)
        total = sum(r[0] for r in recs)
        header = (channels * 24 + total * 4 + 2047) & ~2047
        if header + total * 2048 > block_size:
            return []
        before = 0
        for c, (count, skip) in enumerate(recs):
            start = at + header + before * 2048
            before += count
            packets = opus_packets(d, start, count * 2048)
            repeated = repeated_prefix_packets(packets, previous[c], skip) if skip else 0
            if repeated is None:
                return []
            if packets:
                previous[c] = packets
            if packets and info[c][5] < 2:
                info[c][5] = 2 if any(p[0] & 0x04 for p in packets) else 1
            if collect_packets:
                info[c][4].extend(packets[repeated:] if repeated else packets)
    for x in info:
        if x[3] == 0x0B and x[5] > 1:
            x[1] //= x[5]
    return [tuple(x) for x in info]


def opus_packets(d: bytes, at: int, size: int):
    out = []
    for b in range(at, at + size, 2048):
        p = b
        end = b + 2048
        while p + 2 <= end:
            n, = struct.unpack_from("<H", d, p)
            if n == 0 or p + 2 + n > end:
                break
            out.append(bytes(d[p + 2:p + 2 + n]))
            p += 2 + n
    return out


def export_stems(ids):
    """The file stem for each stream, in stream-table order: `<id>` (8 hex digits) for the first stream with an
    id and `<id>~2`, `<id>~3`, ... for later ones. A bank can hold two streams with the same 29-bit id; naming
    them apart keeps the second from overwriting the first. awc_manifest notes the same stems."""
    seen, out = {}, []
    for sid in ids:
        seen[sid] = seen.get(sid, 0) + 1
        out.append(f"{sid:08X}" if seen[sid] == 1 else f"{sid:08X}~{seen[sid]}")
    return out


def bank_output_dir(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(out_dir, name)


def music_cue_regions(data, frame_limits):
    """Join a unique marker stream-table entry to a unique exported music channel in this bank."""
    if not frame_limits:
        return []
    bank = parse_adat_header_bytes(data, 0, len(data))
    if bank is None:
        return []
    entries = bank["entries"]
    occurrences = Counter(sid for sid, _ in entries)
    rows = []
    for sid, tags in entries:
        if sid not in frame_limits or occurrences[sid] != 1:
            continue
        for kind, offset, size in tags:
            if kind != 0xBD:
                continue
            blob = data[offset:offset + size]
            if not decode_marker_records(blob)["offsets_ascending"]:
                continue
            for region in pair_marker_regions(marker_records(blob))["regions"]:
                if region["end"] <= frame_limits[sid]:
                    rows.append({"file": f"{sid:08X}.ogg", "name_hash": f"{region['name']:08X}",
                                 "start_sample": region["start"], "end_sample": region["end"]})
    return rows


def write_cue_sheet(sheet, rows):
    payload = {"sample_rate": 48000, "regions": rows}
    write_file(sheet, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def export_bank(path: str, out_dir: str, pre_skip: int = OPUS_PRE_SKIP, cue_sheet: bool = False):
    od = bank_output_dir(path, out_dir)
    if cue_sheet:
        sheet = os.path.join(od, "cue_regions.json")
        if os.path.exists(sheet):
            # A skipped bank or interrupted channel write must not leave last run's regions current.
            write_cue_sheet(sheet, [])
    with open(path, "rb") as f:
        d = f.read()
    streams, music = parse_awc(d)
    os.makedirs(od, exist_ok=True)
    counts = {"opus": 0, "pcm": 0, "other": 0, "music": 0}
    channel_counts = Counter()
    frame_limits = {}
    for mus, data in music:
        for cid, samples, rate, codec, pk, packet_channels in music_channels(d, mus, data):
            if cue_sheet:
                channel_counts[cid] += 1
            if pk:
                n48 = samples * 48000 // rate if rate else 0
                granule = write_ogg_opus(os.path.join(od, f"{cid:08X}.ogg"), pk, packet_channels, cid, n48,
                                         pre_skip=pre_skip)
                if cue_sheet and rate == 48000 and samples > 0:
                    frame_limits[cid] = min(samples, max(0, granule - pre_skip))
                counts["music"] += 1
    for s, stem in zip(streams, export_stems([s["id"] for s in streams])):
        at, size = s["data"]
        if s["codec"] == 0x0B and size % 2048 == 0 and s["channels"] in (1, 2):
            pk = opus_packets(d, at, size)
            if pk:
                # +0x0C is the per-channel Opus frame count: equal to the +0 count for mono and
                # half of it for stereo in all 7,451 measured streams. Use it only when it agrees
                # with the interleaved +0 count; a synthetic or malformed block that disagrees
                # must not silently truncate the real audio.
                frames = s["frames"]
                if not frames or frames * s["channels"] != s["samples"]:
                    frames = s["samples"] // max(1, s["channels"])
                write_ogg_opus(os.path.join(od, stem + ".ogg"), pk, s["channels"], s["id"], frames,
                               pre_skip=pre_skip)
                counts["opus"] += 1
                continue
        if s["codec"] == 2 and s["channels"] in (1, 2) and size == s["samples"] * s["channels"] * 4:
            # codec 2: 32-bit float PCM (4 bytes per sample; values can exceed +-1, e.g. impulse responses)
            write_wav(os.path.join(od, stem + ".wav"), memoryview(d)[at:at + size], s["rate"], s["channels"], True)
            counts["pcm"] += 1
            continue
        if s["codec"] == 0 and s["channels"] in (1, 2) and size == s["samples"] * s["channels"] * 2:
            write_wav(os.path.join(od, stem + ".wav"), memoryview(d)[at:at + size], s["rate"], s["channels"])
            counts["pcm"] += 1
            continue
        counts["other"] += 1
    if cue_sheet:
        for stream in streams:
            frame_limits.pop(stream["id"], None)
        verified = {cid: limit for cid, limit in frame_limits.items() if channel_counts[cid] == 1}
        rows = music_cue_regions(d, verified)
        if rows:
            write_cue_sheet(sheet, rows)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--cue-sheet", action="store_true", help="write paired music cues beside their Ogg files")
    ap.add_argument("--limit", type=int, help="maximum number of banks to export (default: all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="banks to export at once (default: half the CPU count)")
    ap.add_argument("--pre-skip", type=int, default=OPUS_PRE_SKIP,
                    help="Opus encoder lookahead in 48 kHz samples written to OpusHead (default: 312)")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 0:
        ap.error("--limit must be non-negative")
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if not 0 <= a.pre_skip <= 0xFFFF:
        ap.error("--pre-skip must be between 0 and 65535")
    if not os.path.exists(a.src):
        ap.error(f"{a.src} does not exist")
    if os.path.isfile(a.out):
        ap.error(f"{a.out} is a file, not a folder")
    if os.path.isdir(a.src):
        banks = []

        def scan_failed(error):
            raise error

        try:
            for root, _, files in os.walk(a.src, onerror=scan_failed):
                output_parent = os.path.join(a.out, os.path.relpath(root, a.src))
                banks.extend((os.path.join(root, name), output_parent)
                             for name in files if name.lower().endswith(".awc"))
        except OSError as error:
            print(f"ERROR: cannot list {error.filename or a.src}: {error.strerror or error}", file=sys.stderr)
            return 1
        banks.sort()
    else:
        banks = [(a.src, a.out)]
    if a.limit is not None:
        banks = banks[:a.limit]
    destinations = {}
    for bank, parent in banks:
        target = bank_output_dir(bank, parent)
        key = path_key(target)
        if key in destinations:
            ap.error(f"{destinations[key]} and {bank} would both export to {target}")
        destinations[key] = bank
    tot = {"opus": 0, "pcm": 0, "other": 0, "music": 0}
    jobs = max(1, a.jobs)

    def add(c):
        for k in tot:
            tot[k] += c[k]

    failed = []

    def skipped(b, e):
        # A bank that does not parse is skipped for good: running again gives the same result. A read or write
        # error (full disk, file in use) is not: it fails the run, so make_game.py runs this step again.
        if isinstance(e, OSError):
            failed.append((b, e))
            print(f"  FAILED {os.path.basename(b)}: {e}")
        else:
            print(f"  skip {os.path.basename(b)}: {e}")

    if jobs == 1:
        for i, (b, output_parent) in enumerate(banks):
            try:
                add(export_bank(b, output_parent, a.pre_skip, a.cue_sheet))
            except (ValueError, struct.error, IndexError, OSError) as e:
                skipped(b, e)
                continue
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(banks)} banks: {tot}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=process_workers(jobs)) as ex:
            futures = {ex.submit(export_bank, b, output_parent, a.pre_skip, a.cue_sheet): b for b, output_parent in banks}
            for i, fut in enumerate(as_completed(futures), 1):
                b = futures[fut]
                try:
                    add(fut.result())
                except (ValueError, struct.error, IndexError, OSError) as e:
                    skipped(b, e)
                if i % 200 == 0:
                    print(f"  {i}/{len(banks)} banks: {tot}", flush=True)
    print(f"{len(banks)} banks: {tot['opus']} Ogg Opus, {tot['music']} music channels (Ogg Opus), {tot['pcm']} WAV, "
          f"{tot['other']} not handled (XMA/other)")
    if failed:
        b, e = failed[0]
        print(f"ERROR: {len(failed)} bank(s) could not be read or written, first {os.path.basename(b)}: {e}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
