"""Decode Ogg Opus audio to 16-bit PCM WAV with ffmpeg.

    python -I audio_wav.py <src dir | .ogg/.opus file> [--out DIR] [--jobs N] [--ffmpeg PATH]

Every .ogg/.opus file under the source is decoded to <out>/<relative path>.wav (the name keeps its directory,
the extension becomes .wav; files that would share a name, such as x.ogg and x.opus, keep their extension:
x.ogg.wav, x.opus.wav). Existing .wav files are skipped, so a stopped run continues where it left off, and
.wav inputs are never decoded. With no --out, a source directory writes to <src>/wav and a single file writes
next to itself. Resuming assumes the same set of sources: an existing plain-named WAV is trusted by name. Only an
existing name shaped like a kept-extension one (x.ogg.wav, x.opus~2.wav) has its audio compared; if it differs,
that file is left alone and the input moves on to the next ~N name.

The output is pcm_s16le at the stream's own sample rate and channel count (Opus is always 48 kHz). A single
stream is trimmed to its Ogg Opus end granule minus pre-skip, because ffmpeg can keep samples beyond its declared
end. For chained streams, ffmpeg handles each link's timing. ffmpeg must be on PATH or given with
--ffmpeg; without it the tool exits with an explanation instead of a traceback.
The decode queue holds at most one outstanding job per worker, even for a large input tree.

For PCM copies of the AWC banks' Opus streams, first repackage the banks with awc_opus.py, then run this tool on
its output (make_game.py does both, the second only with --wav).
Usage: `python -I tools/audio_wav.py --help` lists the options.
"""
import argparse
import concurrent.futures
from contextlib import ExitStack
import errno
from functools import cache
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO
import unicodedata

EXTS = (".ogg", ".opus")


def error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def find_ffmpeg(explicit: str):
    """The ffmpeg executable to run, or None (with the reason already printed)."""
    if explicit:
        candidate = os.path.expanduser(explicit)
        if os.path.dirname(candidate):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        else:
            found = shutil.which(candidate)
            if found:
                return found
        error(f"ffmpeg not found: {explicit}. Give --ffmpeg the path to ffmpeg.exe, or install ffmpeg and put it "
              "on PATH: https://ffmpeg.org/download.html")
        return None
    found = shutil.which("ffmpeg")
    if not found:
        error("ffmpeg was not found on PATH (it decodes the Ogg Opus files to WAV). Install it from "
              "https://ffmpeg.org/download.html and reopen the terminal, or pass --ffmpeg PATH.")
    return found


def _read_opus_id(stream: BinaryIO) -> tuple[bytes, bytes] | None:
    """Read the Opus ID packet and serial, leaving the stream at the next page."""
    header = stream.read(27)
    if len(header) != 27 or header[:6] != b"OggS\x00\x02":
        return None
    lacing = stream.read(header[26])
    # An Opus identification packet completes on the BOS page and is its only packet (RFC 7845).
    if len(lacing) != header[26] or not lacing or lacing[-1] == 255 or any(v != 255 for v in lacing[:-1]):
        return None
    size = sum(lacing)
    packet = stream.read(size)
    if len(packet) != size or size < 19 or not packet.startswith(b"OpusHead"):
        return None
    return packet, header[14:18]


def looks_like_opus(path: str) -> bool:
    """True if the first Ogg page carries an OpusHead packet (so .ogg files that are Vorbis are left alone)."""
    try:
        with open(path, "rb") as f:
            return _read_opus_id(f) is not None
    except OSError:
        return True         # let ffmpeg report the real problem


def ogg_output_samples(path: str) -> int:
    """End granule minus pre-skip for a single unchained Opus stream, or 0 to use ffmpeg's timing."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            identification = _read_opus_id(f)
            if identification is None:
                return 0
            head, serial = identification
            if head[8] >= 16 or head[9] == 0:
                return 0
            pre_skip, = struct.unpack_from("<H", head, 10)
            # Walk page headers forward so same-serial chained or multiplexed streams are rejected.
            offset = f.tell()
            while offset < size:
                f.seek(offset)
                hdr = f.read(27)
                if len(hdr) != 27 or hdr[:4] != b"OggS" or hdr[4] != 0 or hdr[14:18] != serial:
                    return 0
                htype = hdr[5]
                if htype not in (0x00, 0x01, 0x04, 0x05):
                    return 0
                nseg = hdr[26]
                lacing = f.read(nseg)
                if len(lacing) != nseg:
                    return 0
                offset += 27 + nseg + sum(lacing)
                if offset > size:
                    return 0
                if htype & 0x04:
                    if offset != size:
                        return 0
                    granule, = struct.unpack_from("<q", hdr, 6)
                    return granule - pre_skip if granule > pre_skip else 0
            return 0
    except (OSError, struct.error):
        return 0


def collect(src: str):
    """([(input, relative output stem)], number of .ogg/.opus files that are not Ogg Opus)."""
    def scan_failed(exc: OSError) -> None:
        raise exc

    files, not_opus = [], 0
    for root, dirs, names in os.walk(src, onerror=scan_failed):
        dirs.sort()
        for name in sorted(names):
            if not name.lower().endswith(EXTS):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src)
            if looks_like_opus(path):
                files.append((path, os.path.splitext(rel)[0]))
            else:
                not_opus += 1
    return files, not_opus


def output_names(items, out=None):
    """Give every input its own output stem, and count how many had to keep their extension.

    x.ogg and x.opus (or x.OPUS, which is the same name on Windows) both became x.wav: two jobs then decoded into
    the same x.wav.part.wav at once and one result was lost, and with --jobs 1 the second silently replaced the
    first. Inputs that share a stem now keep their extension (x.ogg.wav, x.opus.wav); a name that still clashes
    gets ~2, ~3. With out supplied, resolve directory links on the destination before checking names. Keep the
    original relative paths in the result so existing WAVs are still found and skipped."""
    @cache
    def output_directory(relative):
        return os.path.normcase(os.path.realpath(os.path.join(out, relative)))

    def destination(stem):
        if out is None:
            return os.path.normcase(stem)
        directory, name = os.path.split(stem)
        return output_directory(directory), os.path.normcase(name)

    count = {}
    for _, rel in items:
        key = destination(rel)
        count[key] = count.get(key, 0) + 1
    if len(count) == len(items):
        return list(items), 0
    named, used, renamed = [], set(), 0
    for path, rel in items:
        name = rel
        if count[destination(rel)] > 1:
            name = rel + os.path.splitext(path)[1]
            renamed += 1
        base, n = name, 2
        key = destination(name)
        while key in used:
            name, n = f"{base}~{n}", n + 1
            key = destination(name)
        used.add(key)
        named.append((path, name))
    return named, renamed


def fold_name(text):
    # A broad, cheap filter. The output volume decides whether two candidate names are really aliases.
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", os.path.normcase(text)).upper().casefold())


def existing_entry(path, st):
    if st.st_nlink == 1:
        return st.st_dev, st.st_ino
    # Two hard links have the same inode but can occupy separate directory entries, even on a read-only volume.
    directory, leaf = os.path.split(path)
    with os.scandir(directory) as entries:
        matches = [e.name for e in entries if fold_name(e.name) == fold_name(leaf) and e.inode() == st.st_ino]
    if leaf in matches or len(matches) == 1:
        return st.st_dev, os.stat(directory).st_ino, leaf if leaf in matches else matches[0]
    return None


def native_collisions(group, named, target, unprobed):
    """Ask the destination volume which suspect names identify one file, without touching the real targets."""
    stems = {i: target(named[i][1]) for i in group}
    collided = set()

    def record(indices, tokens):
        owners, missing = {}, []
        for i, token in zip(indices, tokens):
            if token is None:
                missing.append(i)
            else:
                first = owners.setdefault(token, i)
                if first != i:
                    collided.update((first, i))
        return missing

    def existing_tokens(indices, suffix):
        entries = []
        for i in indices:
            path = stems[i] + suffix
            try:
                st = os.lstat(path)
            except OSError as e:
                if e.errno not in (errno.ENOENT, errno.ENAMETOOLONG):
                    raise
                entries.append(None)
                continue
            entry = existing_entry(path, st)
            if not st.st_ino or entry is None:
                return []  # an ambiguous hard link cannot partition this group
            entries.append(entry)
        return entries

    # Existing and missing spellings for a common suffix cannot be aliases. Classify the existing entries
    # together, then apply another common witness to those still missing; never compare incompatible identities.
    suffixes = [".wav"] + sorted({os.path.splitext(named[i][0])[1] + ".wav" for i in group})
    remaining = group
    while len(remaining) > 1:
        for suffix in suffixes:
            entries = existing_tokens(remaining, suffix)
            if any(entries):
                remaining = record(remaining, entries)
                break
        else:
            break
    if len(remaining) < 2:
        return collided

    probes, tokens = {}, []
    try:
        with ExitStack() as stack:
            for i in remaining:
                stem = stems[i]
                parent = os.path.dirname(stem)
                while not os.path.isdir(parent) and os.path.dirname(parent) != parent:
                    parent = os.path.dirname(parent)
                anchor = os.stat(parent)
                volume_dir = anchor.st_dev, anchor.st_ino
                if volume_dir not in probes:
                    probes[volume_dir] = stack.enter_context(tempfile.TemporaryDirectory(dir=parent))
                probe = os.path.join(probes[volume_dir], os.path.relpath(stem, parent) + ".wav")
                os.makedirs(os.path.dirname(probe), exist_ok=True)
                with open(probe, "ab"):
                    pass
                st = os.stat(probe)
                if not st.st_ino:
                    raise OSError(f"the filesystem did not report a file identity for {stem}")
                tokens.append((st.st_dev, st.st_ino))
    except OSError as e:
        if e.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        unprobed.update(remaining)       # only an existing PCM-verified repair may be reused
    else:
        record(remaining, tokens)
    return collided


def repair_base(stem):
    """The kept-extension base of a possible repair, including names that already have a ~N counter."""
    base, sep, counter = stem.rpartition("~")
    if sep and counter.isascii() and counter.isdigit():
        stem = base
    return stem if stem.lower().endswith(EXTS) else None


def plan_outputs(items, src, out, ffmpeg):
    """Apply the existing naming rules first. Reserve their results before moving any residual native aliases."""
    @cache
    def parent(directory):
        return os.path.realpath(os.path.join(out, directory))

    def target(stem):
        directory, name = os.path.split(stem)
        return os.path.join(parent(directory), name)

    folded_parent = cache(lambda directory: fold_name(parent(directory)))

    def key(stem):
        directory, name = os.path.split(stem)
        return folded_parent(directory), fold_name(name)

    def buckets(pairs):
        reserved, groups, guarded = {}, {}, []
        for i, (_, stem) in enumerate(pairs):
            bucket = key(stem)
            first = reserved.setdefault(bucket, i)
            if i != first:
                groups.setdefault(bucket, [first]).append(i)
            base = repair_base(stem)
            if base is not None:
                guarded.append((i, base))
        return reserved, groups, guarded

    reserved, groups, guarded = buckets(items)
    if not groups and not guarded:
        return items, 0, 0
    named, renamed, repaired, unprobed = list(items), 0, set(), set()
    if groups:
        named, renamed = output_names(items, out)
        if renamed:
            # Legacy assignments, including their counters, take precedence over all new names.
            reserved, groups, guarded = buckets(named)
        for group in groups.values():
            repaired.update(native_collisions(group, named, target, unprobed))
        repaired.update(unprobed)
    counters, digests, dest_digests = {}, {}, {}

    def holds_own_audio(i, stem):
        """True if stem.wav is free or already holds input i's decoded audio."""
        path = named[i][0]
        dst = os.path.join(out, stem + ".wav")
        if not os.path.lexists(dst):
            return True
        if i not in digests:
            if not digests:
                print("Checking audio in existing kept-extension WAVs...", flush=True)
            elif len(digests) % 100 == 0:
                print(f"  Checked {len(digests)} input recordings...", flush=True)
            digests[i] = audio_digest(ffmpeg, path, source=True)
        if digests[i] and os.path.isfile(dst):
            if dst not in dest_digests:
                dest_digests[dst] = audio_digest(ffmpeg, dst, source=False)
            if dest_digests[dst] == digests[i]:
                return True
        print(f"  Keeping {dst}: its audio could not be verified for {path}.", file=sys.stderr)
        return False

    def allocate(i, base):
        """Give input i the first unreserved name of base, base~2, ... that is free or holds its audio."""
        base_key = key(base)
        first = counters.get(base_key, 2)
        stem, n = (base, 2) if first == 2 else (f"{base}~{first}", first + 1)
        skipped_unreserved = False
        while True:
            candidate = key(stem)
            if candidate not in reserved:
                if i in unprobed and not os.path.lexists(os.path.join(out, stem + ".wav")):
                    raise OSError(f"cannot distinguish names in a read-only directory and verify {stem}.wav")
                if holds_own_audio(i, stem):
                    break
                skipped_unreserved = True
            stem, n = f"{base}~{n}", n + 1
        if not skipped_unreserved:
            counters[base_key] = n
        reserved[candidate] = i
        named[i] = (named[i][0], stem)

    for i in sorted(repaired):
        directory, leaf = os.path.split(os.path.relpath(named[i][0], src))
        base, ext = os.path.splitext(leaf)
        # Keep room for the source extension, a counter and ".wav" on a 255-byte filesystem.
        while len(os.fsencode(base)) > 230:
            base = base[:-1]
        allocate(i, os.path.join(directory, base + ext))
    # An earlier run may have left a repair under this shape for different audio. Plain names stay name-only.
    moved = 0
    for i, base in guarded:
        if i not in repaired and not holds_own_audio(i, named[i][1]):
            allocate(i, base)
            moved += 1
    return named, renamed, len(repaired) + moved


def audio_digest(ffmpeg, path, *, source):
    """Hash a metadata-free WAV stream, retaining its PCM, channel layout and sample rate."""
    cmd = [ffmpeg, "-v", "error"]
    if not source:
        # Preserve an unspecified layout (e.g. four mapping-family-255 channels) when reading a prior WAV.
        cmd += ["-f", "wav", "-guess_layout_max", "0"]
    cmd += ["-i", path]
    if source:
        samples = ogg_output_samples(path)
        if samples:
            cmd += ["-af", f"atrim=end_sample={samples}"]
    cmd += ["-map_metadata", "-1", "-fflags", "+bitexact", "-c:a", "pcm_s16le" if source else "copy", "-f", "wav", "-"]
    # No stdin for ffmpeg here or in decode(): it reads stdin for key commands, which stops a run started in the
    # background of a terminal and swallows input piped to the caller (such as a shell loop's file list).
    with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        header = proc.stdout.read(12)
        h = hashlib.sha256(header)
        for block in iter(lambda: proc.stdout.read(1024 * 1024), b""):
            h.update(block)
        valid = header[:4] in (b"RIFF", b"RF64") and header[8:] == b"WAVE"
        return h.digest() if proc.wait() == 0 and valid else None


def decode(ffmpeg: str, src: str, dst: str):
    """Decode under the destination directory and rename the finished WAV into place."""
    try:
        out_dir = os.path.dirname(dst)
        os.makedirs(out_dir, exist_ok=True)
        td_kwargs = {"dir": out_dir}
        if sys.version_info >= (3, 10):
            td_kwargs["ignore_cleanup_errors"] = True
        with tempfile.TemporaryDirectory(**td_kwargs) as tmp_dir:
            tmp = os.path.join(tmp_dir, "decoded.wav")
            cmd = [ffmpeg, "-v", "error", "-y", "-i", src]
            samples = ogg_output_samples(src)
            if samples:
                cmd += ["-af", f"atrim=end_sample={samples}"]
            cmd += ["-c:a", "pcm_s16le", tmp]
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace")
            if r.returncode != 0:
                return False, (r.stderr.strip() or f"ffmpeg exited with {r.returncode}")
            os.replace(tmp, dst)
        return True, ""
    except OSError as e:
        return False, str(e)


def _decode_jobs(pool, ffmpeg, todo, workers):
    """Yield (relative name, completed future) without retaining a future for every input."""
    jobs = iter(todo)
    pending = {}
    while True:
        while len(pending) < workers:
            job = next(jobs, None)
            if job is None:
                break
            path, rel, dst = job
            pending[pool.submit(decode, ffmpeg, path, dst)] = rel
        if not pending:
            return
        ready, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in ready:
            yield pending.pop(future), future


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Decode Ogg Opus audio to 16-bit PCM WAV with ffmpeg.")
    ap.add_argument("src", help="a directory to search recursively, or one .ogg/.opus file")
    ap.add_argument("--out", help="output directory (default: <src>/wav for a directory, next to a single file)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="ffmpeg processes to run at once (default: half the CPU count)")
    ap.add_argument("--ffmpeg", help="ffmpeg executable (default: ffmpeg on PATH)")
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")

    src = os.path.abspath(a.src)
    if os.path.isfile(src):
        if not src.lower().endswith(EXTS):
            error(f"{a.src} is not an .ogg/.opus file (already-WAV files are not decoded).")
            return 2
        if not looks_like_opus(src):
            error(f"{a.src} is not Ogg Opus.")
            return 2
        items = [(src, os.path.splitext(os.path.basename(src))[0])]
        not_opus = 0
        default_out = os.path.dirname(src)
    elif os.path.isdir(src):
        try:
            items, not_opus = collect(src)
        except OSError as exc:
            error(f"cannot scan {exc.filename or src}: {exc.strerror or exc}")
            return 2
        default_out = os.path.join(src, "wav")
    else:
        error(f"source not found: {a.src}")
        return 2

    if a.out and os.path.isfile(a.out):
        ap.error(f"{a.out} is a file, not a folder")

    out = os.path.abspath(a.out or default_out)
    print(f"{len(items)} Ogg Opus file(s) under {src} -> {out}")
    if not items:
        print(f"Nothing to do: no Ogg Opus files found"
              + (f" ({not_opus} .ogg/.opus file(s) are not Ogg Opus)" if not_opus else "") + ".")
        return 0

    ffmpeg = find_ffmpeg(a.ffmpeg)
    if not ffmpeg:
        return 2

    try:
        items, renamed, repaired = plan_outputs(items, src, out, ffmpeg)
    except OSError as e:
        error(f"could not safely distinguish WAV output names under {out}: {e}")
        return 2
    if renamed:
        print(f"{renamed} file(s) share a name with another input (e.g. x.ogg and x.opus) and keep their extension "
              f"in the WAV name (x.ogg.wav)")
    if repaired:
        print(f"{repaired} file(s) needed distinct names on the output filesystem or beside an unverified WAV.")

    t0 = time.time()
    written = skipped = 0
    todo = []
    failed = 0
    failure_samples: list[tuple[str, str]] = []

    def record_failure(rel: str, message: str) -> None:
        nonlocal failed
        failed += 1
        if len(failure_samples) < 10:
            first = next((line.strip() for line in message.splitlines() if line.strip()), "unknown error")
            failure_samples.append((rel, first))

    for path, rel in items:
        dst = os.path.join(out, rel + ".wav")
        if os.path.isfile(dst):
            skipped += 1
        elif os.path.exists(dst):
            record_failure(rel, f"output path is not a file: {dst}")
        else:
            todo.append((path, rel, dst))

    done = 0
    workers = min(a.jobs, len(todo)) if todo else 1
    if todo:
        print(f"Decoding {len(todo)} file(s) with {workers} job(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, future in _decode_jobs(pool, ffmpeg, todo, workers):
            try:
                ok, message = future.result()
            except Exception as e:                  # never lose the summary to an unexpected worker error
                ok, message = False, f"{type(e).__name__}: {e}"
            done += 1
            if ok:
                written += 1
            else:
                record_failure(rel, message)
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} decoded", flush=True)

    seconds = time.time() - t0
    summary = f"{len(items)} Ogg Opus file(s): {written} WAV written, {skipped} already present"
    if failed:
        summary += f", {failed} failed"
    print(summary + f" ({seconds:.1f}s)")
    if not_opus:
        print(f"  {not_opus} .ogg/.opus file(s) skipped: not Ogg Opus")
    for rel, first in failure_samples:
        print(f"  FAILED {rel}: {first}")
    if failed > len(failure_samples):
        print(f"  ... and {failed - len(failure_samples)} more failures")
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
