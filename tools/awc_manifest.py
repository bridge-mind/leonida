"""Write a readable manifest of the streams in AWC audio banks, without decoding or exporting audio.

    python -I awc_manifest.py <bank.awc | dir of .awc> [--out FILE] [--format csv|json]

The AWC layout is read by tools/awc_opus.py (parse_awc + music_channels), so this manifest and the
exporter agree on stream ids, codecs, sample rates and streamed-music descriptors. A directory is
scanned recursively for *.awc. CSV is the default on stdout; --out writes a file and a .json
extension selects JSON unless --format says otherwise. A report cannot replace an input bank, including one
found during a directory scan. Unreadable banks are reported on stderr, skipped,
and excluded from bank_count, but a single named file that holds no bank is an error (exit 2) instead of an
empty manifest, since then the path or the format is wrong. The bank column gives the path relative to the
scanned directory or the basename for a single file.

CSV columns: bank, stream, codec, codec_name, channels, sample_rate, declared_samples, duration_s,
streamed_music, data_offset, data_size, bank_size, notes. JSON is an object with bank_count,
stream_count, streamed_music_channel_count and a streams list of the same fields (booleans and
numbers keep their types). duration_s is the decoded duration: a plain Opus stream counts 48 kHz
frames interleaved across channels, so it is `declared_samples / channels / 48000` regardless of
the tag's rate field; PCM and XMA use `declared_samples / sample_rate`. notes flags values that
look inconsistent (zero/odd rate, data-size mismatch, implausible duration).
A streamed-music descriptor contributes one row per channel group; `channels` is the packet channel count read
from the Opus TOC stereo bit, and `declared_samples`/`duration_s` are per channel (the descriptor stores the
interleaved count). `data_offset`/`size` is the shared data tag.
Usage: `python -I tools/awc_manifest.py --help` lists the options.
"""
import argparse
import csv
import json
import os
import struct
import sys
import tempfile

# -I keeps the script directory off sys.path, so add it explicitly before importing the parser.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import awc_opus  # noqa: E402
from file_mode import published_mode  # noqa: E402
from report_paths import same_report_path  # noqa: E402

FIELDS = ("bank", "stream", "codec", "codec_name", "channels", "sample_rate", "declared_samples",
          "duration_s", "streamed_music", "data_offset", "data_size", "bank_size", "notes")
CODEC_NAMES = {0x00: "PCM16", 0x02: "PCM float32", 0x05: "XMA2", 0x0B: "Opus"}


def _seconds(codec: int, channels: int, rate: int, samples: int, streamed: bool):
    """Decoded duration of one stream, or None when the header has no usable timing.

    A plain Opus format tag counts 48 kHz frames interleaved across the packet channels: over
    1,598,151 measured streams, packet samples - declared is 1920..2879 for every rate, which is
    only possible if the count is already 48 kHz frames (and halved for stereo). PCM and XMA
    count per-channel samples at the tag's rate. Streamed-music descriptors are already divided
    to a per-channel count by awc_opus.music_channels()."""
    if codec == 0x0B and not streamed:
        return samples / channels / 48000 if channels > 0 else None
    return samples / rate if rate > 0 else None


def _notes(codec: int, channels: int, rate: int, samples: int, data, bank_size: int,
           streamed: bool = False) -> str:
    """Human-readable warnings for metadata that does not look self-consistent."""
    offset, size = data
    out = []
    if rate <= 0:
        out.append("sample rate is zero")
    elif rate < 1000 or rate > 192000:
        out.append(f"sample rate {rate} Hz is outside 1-192 kHz")
    if samples == 0:
        out.append("no samples declared")
    if size == 0:
        out.append("data size is zero")
    if offset < 0 or size < 0 or offset + size > bank_size:
        out.append("data range is outside the bank")
    seconds = _seconds(codec, channels, rate, samples, streamed)
    if seconds is not None:
        if samples and seconds < 0.001:
            out.append(f"duration {seconds:.6f} s is under 1 ms")
        if seconds > 24 * 3600:
            out.append(f"duration {seconds:.1f} s is over 24 h")
    if codec == 0x0B and rate != 48000:
        out.append("Opus decodes at 48 kHz")
    if codec == 0:
        expected = samples * channels * 2
        if size != expected:
            out.append(f"PCM16 data is {size} bytes, header implies {expected}")
    elif codec == 2:
        expected = samples * channels * 4
        if size != expected:
            out.append(f"float32 data is {size} bytes, header implies {expected}")
    return "; ".join(out)


def _row(bank: str, sid: int, codec: int, channels: int, rate: int, samples: int, data,
         bank_size: int, streamed: bool) -> dict:
    duration = _seconds(codec, channels, rate, samples, streamed)
    return {
        "bank": bank,
        "stream": f"{sid:08X}",
        "codec": f"0x{codec:02X}",
        "codec_name": CODEC_NAMES.get(codec, "unknown"),
        "channels": channels,
        "sample_rate": rate,
        "declared_samples": samples,
        "duration_s": None if duration is None else round(duration, 6),
        "streamed_music": streamed,
        "data_offset": data[0],
        "data_size": data[1],
        "bank_size": bank_size,
        "notes": _notes(codec, channels, rate, samples, data, bank_size, streamed),
    }


def _bank_rows(path: str, bank: str) -> list:
    with open(path, "rb") as f:
        d = f.read()
    parsed = awc_opus.parse_awc(d)
    if isinstance(parsed, tuple) and len(parsed) == 2:       # PR #19: (streams, music)
        streams, music = parsed
    else:                                                    # older parser: streams + attribute
        streams = parsed
        music = getattr(awc_opus.parse_awc, "music", [])
    bank_size, rows = len(d), []
    # A bank can hold two streams with the same id: list both, and note the name awc_opus exports the later
    # one under, instead of letting it replace the first row.
    for s, stem in zip(streams, awc_opus.export_stems([s["id"] for s in streams])):
        row = _row(bank, s["id"], s["codec"], s["channels"], s["rate"], s["samples"], s["data"], bank_size,
                   False)
        if stem != row["stream"]:
            note = f"same id as an earlier stream in this bank; awc_opus exports it as {stem}"
            row["notes"] = "; ".join(x for x in (row["notes"], note) if x)
        rows.append(row)
    for mus, data in music or []:
        try:
            channels = awc_opus.music_channels(d, mus, data, collect_packets=False)
        except (ValueError, struct.error, IndexError) as e:
            print(f"awc_manifest: warning: {bank}: music descriptor: {e}", file=sys.stderr)
            continue
        if not channels:
            print(f"awc_manifest: warning: {bank}: ignored malformed streamed-music descriptor",
                  file=sys.stderr)
            continue
        for cid, samples, rate, codec, _packets, packet_channels in channels:
            row = _row(bank, cid, codec, packet_channels, rate, samples, data, bank_size, True)
            same = next((i for i, r in enumerate(rows) if r["stream"] == row["stream"]), None)
            if same is None:
                rows.append(row)
            else:   # as before: a streamed-music channel replaces a row with its id
                rows[same] = row
    return rows


def collect(src: str, *, out: str | None = None):
    """(parsed bank count, rows, streamed-music row count); one warning per skipped bank.

    Malformed banks are reported on stderr and excluded from the bank count. I/O errors propagate so an
    incomplete scan cannot replace a previous manifest. If out is given, refuse a report that would replace
    any input bank before reading even the first bank (including malformed ones).
    """
    recursive = os.path.isdir(src)
    if recursive:
        def scan_failed(error):
            raise error

        banks = []
        for root, dirs, files in os.walk(src, onerror=scan_failed):
            dirs.sort()
            banks += [os.path.join(root, f) for f in files if f.lower().endswith(".awc")]
        banks.sort(key=lambda p: (os.path.basename(p).lower(), p.lower()))
    else:
        banks = [src]
    if out:
        for path in banks:
            if same_report_path(out, path):
                raise ValueError(f"--out would overwrite the input {path}")
    rows = []
    parsed = 0
    for path in banks:
        try:
            bank = os.path.relpath(path, src).replace(os.sep, "/") if recursive else os.path.basename(path)
            rows += _bank_rows(path, bank)
            parsed += 1
        except (ValueError, struct.error, IndexError) as e:
            print(f"awc_manifest: warning: skipping {path}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: (r["bank"].lower(), int(r["stream"], 16)))
    return parsed, rows, sum(1 for r in rows if r["streamed_music"])


def write_manifest(rows: list, banks: int, streamed: int, fmt: str, out: str) -> None:
    """Write to stdout, or to a temporary file renamed over `out` once it is complete, so a failed
    write keeps the previous manifest instead of truncating it."""
    doc = {"bank_count": banks, "stream_count": len(rows),
           "streamed_music_channel_count": streamed, "streams": rows}
    tmp = None
    if out:
        folder = os.path.dirname(os.path.abspath(out))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".awc_manifest-", suffix=".tmp")
        f = os.fdopen(fd, "w", encoding="utf-8", newline="")
    else:
        f = sys.stdout
    try:
        if fmt == "json":
            json.dump(doc, f, indent=2)
            f.write("\n")
        else:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                row = dict(r)
                row["duration_s"] = "" if r["duration_s"] is None else f"{r['duration_s']:.6f}"
                row["streamed_music"] = "yes" if r["streamed_music"] else "no"
                w.writerow(row)
        if tmp:
            f.close()
            os.chmod(tmp, published_mode(out))   # mkstemp creates 0600
            os.replace(tmp, out)
            tmp = None
    finally:
        if f is not sys.stdout:
            f.close()
        if tmp:
            # The staging file may have inherited a read-only destination's mode; make it deletable
            # first, or Windows strands it beside the manifest when the replace fails.
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write a readable manifest of AWC bank streams; never exports audio.")
    ap.add_argument("src", help=".awc file, or a directory scanned recursively for *.awc")
    ap.add_argument("--out", help="write the manifest to this file instead of stdout")
    ap.add_argument("--format", choices=("csv", "json"),
                    help="output format (default: csv, or json when --out ends in .json)")
    a = ap.parse_args(argv)
    if not os.path.exists(a.src):
        print(f"awc_manifest: error: {a.src}: no such file or directory", file=sys.stderr)
        return 2
    fmt = a.format or ("json" if a.out and a.out.lower().endswith(".json") else "csv")
    try:
        banks, rows, streamed = collect(a.src, out=a.out)
    except ValueError as e:
        print(f"awc_manifest: error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"awc_manifest: error: cannot read {e.filename or a.src}: {e.strerror or e}", file=sys.stderr)
        return 1
    if not banks and not os.path.isdir(a.src):
        # A named file that parsed as no bank is a wrong path or format, not an empty result; a directory scan
        # keeps reporting 0 banks because scanning a folder with no bank in it is a valid answer.
        print(f"awc_manifest: error: {a.src}: holds no AWC bank", file=sys.stderr)
        return 2
    try:
        write_manifest(rows, banks, streamed, fmt, a.out)
    except OSError as e:
        print(f"awc_manifest: error: cannot write {a.out or 'stdout'}: {e}", file=sys.stderr)
        return 1
    print(f"{banks} banks, {len(rows)} streams, {streamed} streamed-music channels", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
