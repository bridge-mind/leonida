"""Index Bink 2 streams in an RPF archive or a directory without extracting them.

A Bink 2 stream starts with ``KB2`` and a lowercase revision letter. The first
44 (0x2C) bytes are the fixed header; audio-track metadata and the frame index
follow it, and their size depends on the revision and the track count. This
tool reads only the fixed header plus the audio descriptors, so a malformed
stream cannot make it read past the stored size.

Usage:
    python -I bink_index.py <movies.rpf | dir> [--out index.csv|.json]
                            [--format csv|json] [--probe]

With ``--probe`` each stream is written to a temporary file and passed to
``ffprobe``; the extra columns say only whether that ffprobe build can decode
the stream. A missing ffprobe leaves the base columns unchanged.
A single named file that is neither a Bink 2 stream nor an RPF archive is an
error (exit 2), not an empty index; a directory scan still reports 0 streams
when nothing matches.

A single named file has to start with ``KB2`` or an RPF header; anything else
(an empty file, a text file, a carved chunk) cannot hold a stream and is one
error line with exit 2, which is how a wrong path shows up.

In an RPF9 archive the last ``u32 +12`` bytes of the file are a block of their own
(``docs/rpf.md``), so when the gap after the last stream reaches EOF and is at least
that long, those bytes are counted as ``rpf_trailer`` and only the rest of the gap is
split into zero padding and unread bytes.
Usage: `python -I tools/bink_index.py --help` lists the options.
"""
import argparse
import csv
import json
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

SIG = re.compile(rb"KB2[a-z]")
HDR = 0x2C
MAX_FRAMES = 1_000_000
MAX_WIDTH = 7680
MAX_HEIGHT = 4800
MAX_AUDIO_TRACKS = 32
# Bink Audio flags in the per-track descriptor.
AUD_16BITS = 0x4000
AUD_STEREO = 0x2000
AUD_USEDCT = 0x1000
AUD_KNOWN = AUD_16BITS | AUD_STEREO | AUD_USEDCT

BASE_FIELDS = [
    "source", "offset", "size", "revision", "frames", "width", "height",
    "fps", "fps_num", "fps_den", "duration_s", "audio_tracks",
    "audio_rates", "audio_flags", "audio_codecs", "audio_channels",
]
PROBE_FIELDS = ["decodes", "codec", "probe_duration_s"]


def _audio_layout(data, base, count, end):
    """Decode one candidate audio layout, or return None if it is not plausible."""
    need = base + 12 * count
    if need > end:
        return None
    rates, flags = [], []
    for i in range(count):
        rate, flag = struct.unpack_from("<HH", data, base + 4 * count + 4 * i)
        if not (8000 <= rate <= 192000) or flag & ~AUD_KNOWN:
            return None
        rates.append(rate)
        flags.append(flag)
    return rates, flags


def parse_header(data, offset):
    """Return the indexed fields of a Bink 2 stream, or None if it is not one."""
    if offset < 0 or offset + HDR > len(data):
        return None
    tag = bytes(data[offset:offset + 4])
    if tag[:3] != b"KB2" or not 0x61 <= tag[3] <= 0x7A:
        return None
    (size8, frames, largest, frames2, width, height,
     fps_num, fps_den, _extra, audio) = struct.unpack_from("<10I", data, offset + 4)
    stored = size8 + 8
    if size8 < 0x24 or stored < HDR + 4 or offset + stored > len(data):
        return None
    if not 1 <= frames <= MAX_FRAMES or frames2 != frames:
        return None
    if largest > size8:
        return None
    if not (16 <= width <= MAX_WIDTH and 16 <= height <= MAX_HEIGHT):
        return None
    if fps_den == 0 or not fps_den <= fps_num <= 240 * fps_den:
        return None
    fps = fps_num / fps_den
    if audio > MAX_AUDIO_TRACKS:
        return None

    revision = chr(tag[3])
    rates, flags = [], []
    if audio:
        # Revisions i/j/k/n have one opaque word at 0x2C before the audio
        # descriptors; older revisions start them at 0x2C. Try the expected
        # layout first, then the other, and accept only plausible descriptors.
        bases = [offset + 0x30, offset + 0x2C] if revision in "ijkn" else [offset + 0x2C]
        for base in bases:
            got = _audio_layout(data, base, audio, offset + stored)
            if got:
                rates, flags = got
                break

    codecs = ["binkaudio_dct" if f & AUD_USEDCT else "binkaudio_rdft" for f in flags]
    channels = [2 if f & AUD_STEREO else 1 for f in flags]
    return {
        "source": None,
        "offset": offset,
        "size": stored,
        "revision": revision,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "fps_num": fps_num,
        "fps_den": fps_den,
        "duration_s": round(frames * fps_den / fps_num, 3),
        "audio_tracks": audio,
        "audio_rates": rates,
        "audio_flags": [f"0x{f:04x}" for f in flags],
        "audio_codecs": codecs,
        "audio_channels": channels,
    }


def _gap(data, start, end):
    """Split an unread range into all-zero padding and other bytes."""
    if end <= start:
        return 0, 0
    for pos in range(start, end, 1 << 20):
        chunk = data[pos:min(pos + (1 << 20), end)]
        if chunk.count(0) != len(chunk):
            return 0, end - start
    return end - start, 0


def scan_file(path, source):
    """Index every Bink 2 stream in one file. Returns (rows, stats)."""
    rows = []
    size = os.path.getsize(path)
    if size == 0:
        return rows, {"files_scanned": 1, "streams": 0, "bytes": 0, "zero_padding": 0,
                      "rpf_trailer": 0, "unread_bytes": 0, "span": 0}
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
            start = trailer_size = 0
            if size >= 16 and bytes(data[1:4]) == b"FPR":
                hint, trailer_size = struct.unpack_from("<I4xI", data, 4)
                if HDR <= hint < size:
                    start = hint
                # +12 as the trailer size is measured on RPF9 only, where it is always a multiple of 16
                # (docs/rpf.md, "The trailer at EOF"); any other value leaves the tail to the zero/unread split.
                if bytes(data[0:1]) != b"9" or trailer_size % 16:
                    trailer_size = 0
            pos = start
            last = start
            zero = unread = 0
            while True:
                match = SIG.search(data, pos)
                if not match:
                    break
                row = parse_header(data, match.start())
                if row is None:
                    pos = match.start() + 1
                    continue
                z, u = _gap(data, last, match.start())
                zero += z
                unread += u
                row["source"] = source
                row["_path"] = path
                rows.append(row)
                last = match.start() + row["size"]
                pos = last
            trailer = trailer_size if 0 < trailer_size <= len(data) - last else 0
            z, u = _gap(data, last, len(data) - trailer)
            zero += z
            unread += u
            span = len(data) - start
    stats = {"files_scanned": 1, "streams": len(rows),
             "bytes": sum(r["size"] for r in rows), "zero_padding": zero,
             "rpf_trailer": trailer, "unread_bytes": unread, "span": span}
    return rows, stats


def scan_target(target, skip_report=None):
    """Index a single container or every file below a directory.

    ``skip_report``, when given, is one existing file path (the report the
    caller is about to overwrite) that must be left out of the scan so that a
    repeat invocation does not count the previous report's bytes.
    """
    rows, total = [], {"files_scanned": 0, "streams": 0, "bytes": 0,
                       "zero_padding": 0, "rpf_trailer": 0, "unread_bytes": 0, "span": 0}
    if os.path.isdir(target):
        base = target
        paths = []

        def scan_failed(error):
            raise error

        for root, _dirs, files in os.walk(target, onerror=scan_failed):
            for name in files:
                path = os.path.join(root, name)
                if skip_report is not None and same_output_path(path, skip_report):
                    continue
                # "/" in the source and the sort key, so the report is the same on every platform
                paths.append((os.path.relpath(path, base).replace(os.sep, "/"), path))
        for source, path in sorted(paths):
            got, stats = scan_file(path, source)
            rows.extend(got)
            for key in total:
                total[key] += stats[key]
    else:
        rows, total = scan_file(target, os.path.basename(target))
    return rows, total


def _temp_stream(path, offset, size):
    """Write one stream to a temporary file; return (name, cleanup)."""
    fd, name = tempfile.mkstemp(prefix="bink_index_", suffix=".bk2")
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as fh:
            fh.seek(offset)
            left = size
            while left:
                chunk = fh.read(min(left, 1 << 20))
                if not chunk:
                    break
                out.write(chunk)
                left -= len(chunk)
    except BaseException:
        os.unlink(name)
        raise
    return name, lambda: os.unlink(name)


def probe_row(row, ffprobe):
    """Run ffprobe on one stream and add decodes/codec/duration."""
    cleanup = None
    try:
        if row["offset"] == 0 and row["size"] == os.path.getsize(row["_path"]):
            name = row["_path"]
        else:
            name, cleanup = _temp_stream(row["_path"], row["offset"], row["size"])
        # Force the Bink demuxer: generic probing can misidentify a frame payload
        # as MPEG-4. The Bink 2 video codec itself is not implemented by ffmpeg,
        # so "yes" is expected only on a build that adds it.
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-f", "bink", "-show_entries",
             "stream=codec_name,codec_type:format=duration", "-of", "json", name],
            capture_output=True, text=True, timeout=60,
        )
        proc.check_returncode()
        info = json.loads(proc.stdout)
        if not isinstance(info, dict):
            raise ValueError("expected an ffprobe JSON object")
        streams, container = info.get("streams", []), info.get("format", {})
        if (not isinstance(streams, list) or any(not isinstance(s, dict) for s in streams)
                or not isinstance(container, dict)):
            raise ValueError("invalid ffprobe stream or format metadata")
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        codec = (video or {}).get("codec_name") or ""
        duration = container.get("duration", "")
        if not isinstance(codec, str) or not isinstance(duration, str):
            raise ValueError("invalid ffprobe codec name or duration")
        row["decodes"] = "yes" if codec else "no"
        row["codec"] = codec
        row["probe_duration_s"] = duration
    except (OSError, subprocess.SubprocessError, ValueError):
        row["decodes"] = "no"
        row["codec"] = ""
        row["probe_duration_s"] = ""
    finally:
        if cleanup:
            cleanup()
    return row


def _flat(row, fields):
    out = {}
    for key in fields:
        value = row.get(key, "")
        if isinstance(value, list):
            value = ";".join(str(v) for v in value)
        out[key] = value
    return out


def write_csv(path, rows, probe):
    fields = BASE_FIELDS + (PROBE_FIELDS if probe else [])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_flat(row, fields))


def write_json(path, target, rows, summary, probe):
    fields = BASE_FIELDS + (PROBE_FIELDS if probe else [])
    streams = [_flat(row, fields) for row in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"target": target, "summary": summary, "streams": streams},
                  fh, indent=1)


def is_rpf_archive(path):
    """True when *path* starts with an RPF header (``?FPR``)."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
    except OSError:
        return False
    return len(header) >= 16 and header[1:4] == b"FPR"


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def target_problem(path):
    """None when a single named file is something this tool indexes, else a one-line reason.

    The scan only searches the bytes for `KB2` headers, so a wrong path or a mislabelled file came back as a
    successful scan of 0 streams. An RPF archive may hold no stream and is a real answer.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    if not head:
        return "empty file"
    if head[1:4] != b"FPR" and not SIG.match(head):
        return "not a Bink 2 stream (KB2) or an RPF archive"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="an .rpf archive, a .bk2 file, or a directory")
    ap.add_argument("--out", help="index.csv or index.json; default is stdout")
    ap.add_argument("--format", choices=["csv", "json"], help="inferred from --out if omitted")
    ap.add_argument("--probe", action="store_true", help="run ffprobe on each stream")
    a = ap.parse_args(argv)
    if a.out and same_output_path(a.target, a.out):
        ap.error("--out must not overwrite the source")
    if a.out and os.path.isdir(a.out):
        ap.error(f"--out is a folder, not a report file: {a.out}")
    if a.out and os.path.isdir(a.target):
        manifest = os.path.join(a.target, "index.json")
        # Reserved even before the carve writes it: other tools read <target>/index.json as the carve manifest.
        if same_output_path(a.out, manifest):
            ap.error(f"--out would overwrite the carve manifest: {a.out}")
        if os.path.isfile(a.out):
            for root, _dirs, names in os.walk(a.target):
                for name in names:
                    if name.lower().endswith((".json", ".csv")):
                        continue
                    if same_output_path(os.path.join(root, name), a.out):
                        ap.error(f"--out would overwrite an input file: {a.out}")
    if not os.path.exists(a.target):
        print(f"bink_index: error: {a.target}: no such file or directory", file=sys.stderr)
        return 2
    if not os.path.isdir(a.target):
        try:
            problem = target_problem(a.target)
        except OSError as error:
            print(f"bink_index: error: cannot read {a.target}: {error.strerror or error}", file=sys.stderr)
            return 2
        if problem:
            print(f"bink_index: error: {a.target}: {problem}", file=sys.stderr)
            return 2

    skip_report = None
    if a.out and os.path.isfile(a.out):
        skip_report = a.out
    try:
        rows, summary = scan_target(a.target, skip_report)
    except OSError as error:
        path = error.filename or a.target
        print(f"bink_index: error: cannot read {path}: {error.strerror or error}", file=sys.stderr)
        return 2
    if not rows and not os.path.isdir(a.target) and not is_rpf_archive(a.target):
        # A named file with no Bink 2 stream and no RPF header is a wrong path or
        # format, not an empty index; a directory scan still reports 0 streams.
        print(f"bink_index: error: {a.target}: not a Bink 2 stream or an RPF archive", file=sys.stderr)
        return 2
    probe = False
    if a.probe:
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            probe = True
            for row in rows:
                probe_row(row, ffprobe)
        else:
            print("ffprobe not found; --probe skipped", file=sys.stderr)

    fmt = a.format or ("json" if a.out and a.out.lower().endswith(".json") else "csv")
    if a.out:
        try:
            if fmt == "json":
                write_json(a.out, a.target, rows, summary, probe)
            else:
                write_csv(a.out, rows, probe)
        except OSError as error:
            print(f"bink_index: error: cannot write {a.out}: {error}", file=sys.stderr)
            return 2
        print(f"{len(rows)} Bink 2 streams, {summary['bytes']} bytes, "
              f"{summary['unread_bytes']} unread, {summary['zero_padding']} zero padding, "
              f"{summary['rpf_trailer']} RPF trailer")
    else:
        if fmt == "json":
            fields = BASE_FIELDS + (PROBE_FIELDS if probe else [])
            print(json.dumps({"target": a.target, "summary": summary,
                              "streams": [_flat(r, fields) for r in rows]}, indent=1))
        else:
            fields = BASE_FIELDS + (PROBE_FIELDS if probe else [])
            # stdout is a text stream that already turns "\n" into the platform's line end;
            # csv's default "\r\n" would come out as "\r\r\n" on Windows
            writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(_flat(row, fields))
        print(f"{len(rows)} Bink 2 streams, {summary['bytes']} bytes, "
              f"{summary['unread_bytes']} unread, {summary['zero_padding']} zero padding, "
              f"{summary['rpf_trailer']} RPF trailer",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
