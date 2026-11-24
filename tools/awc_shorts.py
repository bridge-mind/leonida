"""Export short streamed-music channels (station idents, stings, DJ links) from AWC banks.

    python -I tools/awc_shorts.py <bank.awc | directory> <out dir> [--min-seconds 5] [--max-seconds 60] [--jobs N]

Uses `tools/awc_opus.py` for the AWC parse and packet collection. A streamed-music channel is selected
when its decoded duration is inside the requested range. The descriptor's sample count is interleaved
across the Opus packet channels, so the packet channel count is read from the TOC stereo bit and the
per-channel frame count is `samples / packet_channels`. The output is
`<out>/<bank>/<stream id>.ogg` plus `<out>/shorts.csv`. Later selected channels with the same ID use
`<stream id>~2.ogg`, `<stream id>~3.ogg`, and so on. Directory scans preserve the input subdirectories.

This tool never re-encodes: the selected Opus packets are repacked into Ogg Opus exactly as stored.
It does not transcribe or classify the clips; that is a separate analysis step.
Usage: `python -I tools/awc_shorts.py --help` lists the options.
"""
import argparse
import csv
import os
import struct
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# -I keeps the script directory off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import awc_opus as awc  # noqa: E402
from file_mode import published_mode  # noqa: E402
from pool_workers import process_workers  # noqa: E402
from report_paths import path_key  # noqa: E402

FIELDS = ("bank", "stream", "duration_s", "channels", "packets", "file")


def short_channels(d: bytes, mus, data, min_seconds: float, max_seconds: float):
    """[(channel id, frames, rate, packets, packet channels, duration_s)] inside the range."""
    selected = []
    for item in awc.music_channels(d, mus, data):
        cid, samples, rate, codec, packets = item[:5]
        if codec != 0x0B or not packets or rate <= 0:
            continue
        if len(item) > 5:
            # Newer awc_opus returns the per-channel frame count and packet channels.
            packet_channels = item[5]
            frames = samples
        else:
            packet_channels = (2 if any(packet[0] & 0x04 for packet in packets) else 1)
            frames = samples // packet_channels if packet_channels > 1 else samples
        duration = frames / rate
        if min_seconds <= duration <= max_seconds:
            selected.append((cid, frames, rate, packets, packet_channels, duration))
    return selected


def export_bank(path: str, out_dir: str, min_seconds: float, max_seconds: float, *, bank=None):
    """Export the selected channels of one bank; returns CSV rows."""
    with open(path, "rb") as f:
        d = f.read()
    if bank is None:
        bank = os.path.splitext(os.path.basename(path))[0]
    od = os.path.join(out_dir, bank)
    os.makedirs(od, exist_ok=True)
    _streams, music = awc.parse_awc(d)
    selected = [
        channel
        for mus, data in music
        for channel in short_channels(d, mus, data, min_seconds, max_seconds)
    ]
    stems = awc.export_stems(channel[0] for channel in selected)
    rows = []
    for (cid, frames, rate, packets, channel_count, duration), stem in zip(selected, stems):
        op = os.path.join(od, stem + ".ogg")
        awc.write_ogg_opus(op, packets, channel_count, cid, frames * 48000 // rate)
        rows.append({
            "bank": bank,
            "stream": f"{cid:08X}",
            "duration_s": round(duration, 6),
            "channels": channel_count,
            "packets": len(packets),
            "file": os.path.relpath(op, out_dir).replace(os.sep, "/"),
        })
    return rows


def _banks(src: str):
    if os.path.isdir(src):
        found = []

        def scan_failed(error):
            raise error

        for root, _dirs, files in os.walk(src, onerror=scan_failed):
            found += [os.path.join(root, name) for name in files if name.lower().endswith(".awc")]
        return [(path, os.path.splitext(os.path.relpath(path, src))[0].replace(os.sep, "/"))
                for path in sorted(found)]
    return [(src, os.path.splitext(os.path.basename(src))[0])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help=".awc file or directory scanned recursively for *.awc")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--min-seconds", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    if a.jobs < 1:
        print("awc_shorts: error: --jobs must be at least 1", file=sys.stderr)
        return 2
    if not 0 <= a.min_seconds <= a.max_seconds:
        print("awc_shorts: error: require 0 <= --min-seconds <= --max-seconds", file=sys.stderr)
        return 2
    if not os.path.exists(a.src):
        print(f"awc_shorts: error: source does not exist: {a.src}", file=sys.stderr)
        return 2
    try:
        banks = _banks(a.src)
    except OSError as e:
        print(f"awc_shorts: cannot list {e.filename or a.src}: {e.strerror or e}", file=sys.stderr)
        return 2
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        print(f"awc_shorts: cannot create {a.out}: {e}", file=sys.stderr)
        return 1
    destinations = {}
    for path, bank in banks:
        target = os.path.join(a.out, bank)
        key = path_key(target)
        if key in destinations:
            print(f"awc_shorts: error: {destinations[key]} and {path} would both export to {target}", file=sys.stderr)
            return 2
        destinations[key] = path
    rows = []
    io_failed = False

    def report_failure(bank, error):
        nonlocal io_failed
        if isinstance(error, OSError):
            io_failed = True
            print(f"  FAILED {bank}: {error}", file=sys.stderr)
        else:
            print(f"  skip {bank}: {error}", file=sys.stderr)

    if a.jobs == 1:
        for i, (path, bank) in enumerate(banks, 1):
            try:
                rows += export_bank(path, a.out, a.min_seconds, a.max_seconds, bank=bank)
            except (ValueError, struct.error, OSError, IndexError) as e:
                report_failure(bank, e)
            if i % 200 == 0:
                print(f"  {i}/{len(banks)} banks, {len(rows)} clips", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=process_workers(a.jobs)) as ex:
            futures = {ex.submit(export_bank, path, a.out, a.min_seconds, a.max_seconds, bank=bank): bank
                       for path, bank in banks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rows += fut.result()
                except (ValueError, struct.error, OSError, IndexError) as e:
                    report_failure(futures[fut], e)
                if i % 200 == 0:
                    print(f"  {i}/{len(banks)} banks, {len(rows)} clips", flush=True)
    if io_failed:
        return 1
    rows.sort(key=lambda r: (r["bank"].lower(), r["stream"]))
    catalog = os.path.join(a.out, "shorts.csv")
    try:
        with tempfile.TemporaryDirectory(dir=a.out, prefix=".shorts-") as staging:
            staged = os.path.join(staging, "shorts.csv")
            with open(staged, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            os.chmod(staged, published_mode(catalog))
            os.replace(staged, catalog)
    except OSError as e:
        print(f"awc_shorts: cannot write {catalog}: {e}", file=sys.stderr)
        return 1
    print(f"{len(banks)} banks: {len(rows)} short streamed-music channels -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
