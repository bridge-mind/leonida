"""Count the marker events (tag 0xBD) of AWC banks: layout checks, type hashes and counts, never audio or names.

    python -I awc_markers.py <archive.rpf | bank.awc | dir of .awc> [--out FILE.json]

A 0xBD tag is a headerless array of 16-byte events (docs/audio_config.md; the same records as
rsc8_audio.decode_marker_records, which checks one tag, where this tool reports per type and per kind of bank):

    +0   u32  type    (a 32-bit name hash)
    +4   f32  value   (a number for most types; a few types store a second u32 name hash here)
    +8   u32  time    (48 kHz frames from the start of the stream)
    +12  u32  flags

An archive is walked for `ADAT` banks with rsc8_audio.parse_adat_at_file, the same header parser and the same bank
set as rsc8_audio.scan_rpf, but each parsed bank is jumped over, so only the headers, the gaps between banks and the
marker tags are read (under a minute for the 28.5 GB audio.rpf on an NVMe drive). A directory is scanned
recursively for *.awc. An unreadable bank or directory fails the scan without publishing partial counts.

Banks are reported in two groups: `streamed_music` (the bank has a 0x48 descriptor) and `other`. A chunk is compared
with the length of its own stream in 48 kHz frames: for streamed music the descriptor channel with the stream's id
(interleaved samples divided by the packet channel count, read from the TOC stereo bit of the channel's first
packet), otherwise the Opus format tag (samples / channels). Streams with neither have no length and are counted.

The report holds counts, sizes, 32-bit hashes and numeric ranges only.
Usage: `python -I tools/awc_markers.py --help` lists the options.
"""
import argparse
import json
import os
import statistics
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsc8_audio import parse_adat_at_file  # noqa: E402

TAG_MUSIC, TAG_DATA, TAG_FORMAT, TAG_MARKER = 0x48, 0x55, 0x76, 0xBD
EVENT = struct.Struct("<IIII")      # type, value (the bits of an f32), time, flags
RAW_VALUES = 8                      # a type with at most this many distinct values lists their bits
WINDOW = 1 << 20                    # bytes searched at a time for the next bank


def events(chunk: bytes) -> list:
    """[(type, value bits, time, flags)] for the whole 16-byte events of one 0xBD chunk."""
    return list(EVENT.iter_unpack(chunk[:len(chunk) - len(chunk) % EVENT.size]))


def iter_banks(fh, size: int):
    """Every `ADAT` bank of an open file in offset order; bytes inside a parsed bank are not searched.

    The file is searched a window at a time, and a window is kept while the next search starts inside it, so a
    run of small banks is read about once and not once per bank.
    """
    base, buf, pos = 0, b"", 0
    while pos < size:
        if not (buf and base <= pos < base + WINDOW):
            fh.seek(pos)
            base, buf = pos, fh.read(WINDOW + 3)    # +3: a magic that starts in this window and ends in the next
        bank, at = None, buf.find(b"ADAT", pos - base)
        while 0 <= at < WINDOW:
            bank = parse_adat_at_file(fh, base + at, size)
            if bank is not None:
                break
            at = buf.find(b"ADAT", at + 4)
        if bank is None:
            pos = base + WINDOW
            continue
        yield bank
        pos = max(bank["end"], bank["offset"] + 4)


def _read(fh, at: int, size: int) -> bytes:
    fh.seek(at)
    return fh.read(size)


def music_frames(fh, base: int, mus, data) -> dict:
    """{channel id: 48 kHz frames per channel} for one 0x48 descriptor, or {} when it does not parse."""
    desc = _read(fh, base + mus[0], mus[1])
    if len(desc) < 12:
        return {}
    channels = struct.unpack_from("<I", desc, 8)[0]
    if not 1 <= channels <= 32 or len(desc) != 12 + 16 * channels:
        return {}
    table = _read(fh, base + data[0], min(data[1], 24 * channels))
    if len(table) < 24 * channels:
        return {}
    counts = [struct.unpack_from("<I", table, 24 * c + 4)[0] for c in range(channels)]
    first = (24 * channels + 4 * sum(counts) + 2047) & ~2047      # the first channel's first 2048-byte subblock
    out, before = {}, 0
    for c in range(channels):
        cid, samples = struct.unpack_from("<II", desc, 12 + 16 * c)
        packet = _read(fh, base + data[0] + first + before * 2048, 3) if counts[c] else b""
        before += counts[c]
        stereo = len(packet) == 3 and packet[:2] != b"\0\0" and packet[2] & 0x04
        out[cid & 0x1FFFFFFF] = samples // 2 if stereo else samples
    return out


def bank_chunks(fh, bank: dict):
    """(is streamed music, [(event list, chunk size, stream frames or None)]) for one parsed bank."""
    base, frames, music, chunks = bank["offset"], {}, False, []
    for _sid, tags in bank["entries"]:
        tm = {t: (at, size) for t, at, size in tags}
        if TAG_MUSIC in tm:
            music = True
            if TAG_DATA in tm:
                frames.update(music_frames(fh, base, tm[TAG_MUSIC], tm[TAG_DATA]))
    for sid, tags in bank["entries"]:
        tm = {t: (at, size) for t, at, size in tags}
        if TAG_MARKER not in tm:
            continue
        length = frames.get(sid)
        if length is None and TAG_FORMAT in tm and tm[TAG_FORMAT][1] >= 30:
            fmt = _read(fh, base + tm[TAG_FORMAT][0], 30)
            if len(fmt) == 30 and fmt[28] == 0x0B and fmt[29]:
                length = struct.unpack_from("<I", fmt, 0)[0] // fmt[29]
        chunk = _read(fh, base + tm[TAG_MARKER][0], tm[TAG_MARKER][1])
        chunks.append((events(chunk), len(chunk), length))
    return music, chunks


def _new_group() -> dict:
    return {"banks": 0, "banks_with_chunk": 0, "chunks": 0, "chunks_not_multiple_of_16": 0, "events": 0,
            "per_chunk": [], "chunks_time_decreases": 0, "chunks_with_length": 0, "chunks_past_end": 0,
            "flags": {}, "types": {}}


def _add(group: dict, evs: list, size: int, length) -> None:
    group["chunks"] += 1
    group["chunks_not_multiple_of_16"] += size % EVENT.size != 0
    group["events"] += len(evs)
    group["per_chunk"].append(len(evs))
    times = [e[2] for e in evs]
    group["chunks_time_decreases"] += any(b < a for a, b in zip(times, times[1:]))
    if length is not None:
        group["chunks_with_length"] += 1
        group["chunks_past_end"] += bool(times) and max(times) > length
    here = {}
    for typ, value, time, flags in evs:
        group["flags"][flags] = group["flags"].get(flags, 0) + 1
        t = group["types"].setdefault(typ, {"events": 0, "chunks": 0, "chunks_with_one": 0, "max_per_chunk": 0,
                                            "at_time_0": 0, "values": set()})
        t["events"] += 1
        t["at_time_0"] += time == 0
        t["values"].add(value)
        here[typ] = here.get(typ, 0) + 1
    for typ, n in here.items():
        t = group["types"][typ]
        t["chunks"] += 1
        t["chunks_with_one"] += n == 1
        t["max_per_chunk"] = max(t["max_per_chunk"], n)


def _finish(group: dict) -> dict:
    per = group.pop("per_chunk")
    group["events_per_chunk"] = {"min": min(per), "median": statistics.median(per), "max": max(per)} if per else None
    group["flags"] = {f"0x{k:08X}": v for k, v in sorted(group["flags"].items())}
    types = {}
    for typ, t in sorted(group["types"].items(), key=lambda kv: (-kv[1]["events"], kv[0])):
        bits = sorted(t.pop("values"))
        values = [struct.unpack("<f", struct.pack("<I", b))[0] for b in bits]
        finite = [v for v in values if v == v and abs(v) != float("inf")]
        t["distinct_values"] = len(bits)
        t["value_min"], t["value_max"] = (min(finite), max(finite)) if finite else (None, None)
        t["values_all_integers"] = len(finite) == len(values) and all(v == int(v) for v in finite)
        if len(bits) <= RAW_VALUES:     # few enough to be an enumeration: a name hash shows as a meaningless float
            t["value_bits"] = [f"0x{b:08X}" for b in bits]
        types[f"0x{typ:08X}"] = t
    group["types"] = types
    return group


def scan(src: str) -> dict:
    """The report for an archive, one bank or a directory of banks."""
    if os.path.isdir(src):
        def scan_failed(error):
            raise error

        paths = sorted(os.path.join(root, f) for root, _dirs, files in os.walk(src, onerror=scan_failed)
                       for f in files if f.lower().endswith(".awc"))
    else:
        paths = [src]
    groups = {"streamed_music": _new_group(), "other": _new_group()}
    banks = streams = 0
    for path in paths:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            for bank in iter_banks(fh, size):
                banks += 1
                streams += bank["count"]
                music, chunks = bank_chunks(fh, bank)
                group = groups["streamed_music" if music else "other"]
                group["banks"] += 1
                group["banks_with_chunk"] += bool(chunks)
                for evs, chunk_size, length in chunks:
                    _add(group, evs, chunk_size, length)
    return {"banks": banks, "streams": streams, **{name: _finish(g) for name, g in groups.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count AWC marker events (tag 0xBD); hashes and counts only.")
    ap.add_argument("src", help="an RPF9 archive, an .awc bank, or a directory scanned recursively for *.awc")
    ap.add_argument("--out", help="write the full report (every type hash) to this JSON file")
    a = ap.parse_args(argv)
    if not os.path.exists(a.src):
        print(f"awc_markers: error: {a.src}: no such file or directory", file=sys.stderr)
        return 2
    try:
        report = scan(a.src)
    except OSError as error:
        print(f"awc_markers: error: cannot read {error.filename or a.src}: {error.strerror or error}", file=sys.stderr)
        return 1
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print(f"{report['banks']} banks, {report['streams']} streams")
    for name in ("streamed_music", "other"):
        g = report[name]
        print(f"{name}: {g['banks']} banks, {g['banks_with_chunk']} with a 0xBD chunk; {g['chunks']} chunks, "
              f"{g['events']} events, {len(g['types'])} types; size not a multiple of 16: "
              f"{g['chunks_not_multiple_of_16']}; time decreases: {g['chunks_time_decreases']}; "
              f"past the stream end: {g['chunks_past_end']} of {g['chunks_with_length']} with a length; "
              f"flags: {g['flags']}")
        for typ, t in list(g["types"].items())[:20]:
            print(f"  {typ}  events {t['events']:>7}  chunks {t['chunks']:>7}  one per chunk {t['chunks_with_one']:>7}"
                  f"  max per chunk {t['max_per_chunk']:>5}  at time 0 {t['at_time_0']:>7}"
                  f"  distinct values {t['distinct_values']:>5}  "
                  + (f"{t['value_min']:g} .. {t['value_max']:g}" if t["values_all_integers"]
                     else " ".join(t.get("value_bits", ["not integers"]))))
        if len(g["types"]) > 20:
            print(f"  ... {len(g['types']) - 20} more types (use --out)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
