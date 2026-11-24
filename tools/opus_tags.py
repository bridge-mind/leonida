"""Write stream names, dialogue labels and subtitles into the Ogg Opus files that awc_opus.py exports.

awc_opus.py writes <out>/<bank>/<stream id>.ogg with an empty OpusTags header. This tool fills that header in place,
from two local work products:

``--names``     awc_names.py CSV (source, bank, stream_id, name, rule). A file is matched by its stream id (the file
                name; awc_opus.py's ``<id>~2.ogg`` for a second stream with the same id gets the same name) and its
                bank (the folder path below the Opus folder; if that has no match, the path with its leading folders
                removed one at a time, so the Opus folder may hold several archives' outputs).
``--dialogue``  text_tables.py dialogue CSV (wave, key, label, speaker, text_<lang> ...). A named stream whose name is
                a dialogue wave gets that line's label, speaker and subtitles.

Comments written (RFC 7845 section 5.2; empty values are left out):

    TITLE=<stream name>
    LABEL=<dialogue line label>          SPEAKER=<speaker code from the label>
    SUBTITLE=<English text, or the first language with text>
    SUBTITLE_<LANG>=<text>               one per language column with text

Subtitle markup (``~n~``, ``~sl:...~``, ``~COLOR_...~`` and other ``~...~`` tokens) becomes a space. Only the OpusTags
page (page 1, which holds that packet alone) is replaced, so the audio pages are copied byte for byte and the page
sequence numbers do not change. Streams without a name are left untouched, and so is a stream whose id its bank names
twice (counted as ``ambiguous name``): the archive stores only the low 29 bits of a name hash, so two names can claim
one id, and writing either would put another stream's name and, through --dialogue, its subtitle in the file. Running
the tool again replaces the tags it wrote. A tags packet larger than one Ogg page (65,024 bytes) is skipped and counted.
Files with a bad page checksum or truncated page are also skipped and left untouched; all pages are checked before any replacement is written.

Usage:
    python -I opus_tags.py <Opus folder> --names names.csv [--dialogue dialogue.csv] [--source LABEL] [--dry-run]

Standard library only. The CSVs and the tagged files hold game names and text: keep them in your local work folder.
"""
import argparse
import collections
import csv
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from awc_opus import ogg_crc, ogg_page, write_file  # noqa: E402

VENDOR = b"leonida "
MARKUP = re.compile(r"~[^~]*~")
SPACES = re.compile(r"\s+")
STREAM_FILE = re.compile(r"^([0-9A-Fa-f]{8})(?:~\d+)?\.ogg$")   # <id>~2 ...: a later stream with the same id
MAX_PACKET = 255 * 255 - 1          # lacing values of one page: 254 full segments and a final short one


def clean(text: str) -> str:
    return SPACES.sub(" ", MARKUP.sub(" ", text)).strip()


def tags_packet(comments) -> bytes:
    """An OpusTags packet: magic, vendor string, then the comments as KEY=value (RFC 7845 section 5.2)."""
    items = [f"{k}={v}".encode("utf-8") for k, v in comments if v]
    out = b"OpusTags" + struct.pack("<I", len(VENDOR)) + VENDOR + struct.pack("<I", len(items))
    return out + b"".join(struct.pack("<I", len(i)) + i for i in items)


def pages(data: bytes):
    """(start, end) of each Ogg page. Raises ValueError on anything that is not a page."""
    at = 0
    while at < len(data):
        if data[at:at + 4] != b"OggS" or at + 27 > len(data):
            raise ValueError(f"no Ogg page at byte {at}")
        segments = data[at + 26]
        end = at + 27 + segments + sum(data[at + 27:at + 27 + segments])
        if end > len(data):
            raise ValueError(f"page at byte {at} runs past the end of the file")
        yield at, end
        at = end


def checked_pages(data: bytes):
    """Validate every page checksum before its bytes are used or copied to the output."""
    for start, end in pages(data):
        page = bytearray(data[start:end])  # bounded by one Ogg page, not the whole stream
        expected, = struct.unpack_from("<I", page, 22)
        page[22:26] = b"\0" * 4
        if ogg_crc(page) != expected:
            raise ValueError(f"Ogg checksum mismatch at byte {start}")
        yield start, end


def retag(data: bytes, packet: bytes) -> bytes:
    """`data` with its OpusTags page replaced; refuse corrupt or truncated input."""
    if len(packet) > MAX_PACKET:
        raise ValueError("tags do not fit in one page")
    it = checked_pages(data)
    try:
        _, (tags_at, tags_end) = next(it), next(it)
    except StopIteration:
        raise ValueError("fewer than two Ogg pages") from None
    body = tags_at + 27 + data[tags_at + 26]
    if data[body:body + 8] != b"OpusTags":
        raise ValueError("page 1 does not start an OpusTags packet")
    segments = data[tags_at + 27:tags_at + 27 + data[tags_at + 26]]
    if not segments or segments[-1] == 255:
        raise ValueError("the OpusTags packet continues past page 1")
    if any(segment != 255 for segment in segments[:-1]):
        raise ValueError("the OpusTags page contains another packet")
    for _ in it:  # also check the audio tail before publishing any replacement
        pass
    serial, seq = struct.unpack_from("<II", data, tags_at + 14)
    if seq != 1:
        raise ValueError("the OpusTags page is not page 1")
    return data[:tags_at] + ogg_page(serial, 1, 0, [packet], 0x00) + data[tags_end:]


def load_names(path, source=None):
    """({bank label: {29-bit stream id: name}}, {bank label: {ids a second name claims}}) from an awc_names.py CSV.

    The archive stores only the low 29 bits of a name hash, so two names can claim one id: two sources over the same
    archive (--source is optional) or a genuine collision. The last row used to win, which wrote an arbitrary name -
    and, through --dialogue, another stream's subtitle - into the file. An id that two different names claim is left
    out of the map and returned instead, so it is reported and not tagged.
    """
    out = collections.defaultdict(dict)
    ambiguous = collections.defaultdict(set)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if source and row.get("source") != source:
                continue
            if row["bank"] is None or row["stream_id"] is None or row["name"] is None:   # a row cut short, e.g. by an interrupted write
                raise ValueError(f"{path}: line {reader.line_num} is missing bank, stream_id or name")
            bank = row["bank"].replace("\\", "/")
            sid = int(row["stream_id"], 16) & 0x1FFFFFFF
            if sid in out[bank]:
                if out[bank][sid] != row["name"]:
                    ambiguous[bank].add(sid)
                continue
            out[bank][sid] = row["name"]
    for bank, ids in ambiguous.items():
        for sid in ids:
            del out[bank][sid]
    return dict(out), {bank: ids for bank, ids in ambiguous.items()}


def load_dialogue(path):
    """{wave: row} and language codes; refuse repeated waves whose rendered tags disagree."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        langs = [f[5:] for f in reader.fieldnames or [] if f.startswith("text_")]
        out = {}
        for row in reader:
            if row["wave"] is None:
                raise ValueError(f"{path}: line {reader.line_num} is missing wave")
            wave = row["wave"].lower()
            if wave in out:
                previous = tags_packet(comments(wave, out[wave], langs))
                if previous != tags_packet(comments(wave, row, langs)):
                    raise ValueError(f"{path}:{reader.line_num}: conflicting dialogue rows for wave {wave!r}")
            else:
                out[wave] = row
        return out, langs


def comments(name, line, langs):
    out = [("TITLE", name)]
    if line:
        texts = [(lang, clean(line.get(f"text_{lang}") or "")) for lang in langs]
        texts = [(lang, t) for lang, t in texts if t]
        main = dict(texts).get("en") or (texts[0][1] if texts else "")
        out += [("LABEL", line.get("label", "")), ("SPEAKER", line.get("speaker", "")), ("SUBTITLE", main)]
        out += [(f"SUBTITLE_{lang.upper()}", t) for lang, t in texts]
    return out


def bank_label(names, rel_dir):
    """The names-CSV label of the bank in `rel_dir`, trying shorter paths from the left when the full path is
    unknown; None when no label matches. Each path is tried as is, then with ".awc": awc_names.py labels a bank read
    from an audio_manifest.json with its archive path, extension included, while awc_opus.py's folder has none. A
    label as is wins."""
    parts = rel_dir.split("/")
    for i in range(len(parts)):
        path = "/".join(parts[i:])
        for label in (path, path + ".awc"):
            if names.get(label) is not None:
                return label
    return None

def bank_names(names, rel_dir):
    """The stream names of the bank in `rel_dir`, trying shorter paths from the left when the full path is unknown."""
    label = bank_label(names, rel_dir)
    return None if label is None else names[label]


def tag_folder(root, names, dialogue=None, langs=(), dry_run=False, ambiguous=None):
    def scan_failed(error):
        raise error

    stats = collections.Counter()
    dialogue = dialogue or {}
    ambiguous = ambiguous or {}
    for folder, _dirs, files in os.walk(root, onerror=scan_failed):
        rel = os.path.relpath(folder, root).replace(os.sep, "/")
        streams, label = None, None
        for fname in sorted(files):
            m = STREAM_FILE.match(fname)
            if not m:
                continue
            stats["ogg files"] += 1
            if streams is None:
                label = bank_label(names, rel) if rel != "." else None
                streams = names[label] if label is not None else {}
            sid = int(m.group(1), 16) & 0x1FFFFFFF
            if sid in ambiguous.get(label, ()):
                stats["ambiguous name"] += 1
                continue
            name = streams.get(sid)
            if not name:
                stats["without a name"] += 1
                continue
            line = dialogue.get(name.lower())
            fields = comments(name, line, langs)
            packet = tags_packet(fields)
            path = os.path.join(folder, fname)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                new = retag(data, packet)
            except ValueError as e:
                stats["skipped: " + str(e).split(" at byte")[0]] += 1
                continue
            stats["tagged"] += 1
            if line:
                stats["with a dialogue line"] += 1
                if dict(fields).get("SUBTITLE"):
                    stats["with a subtitle"] += 1
            if not dry_run and new != data:
                write_file(path, new)
    return dict(sorted(stats.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("opus", help="folder written by awc_opus.py")
    ap.add_argument("--names", required=True, help="awc_names.py CSV")
    ap.add_argument("--dialogue", help="text_tables.py dialogue CSV")
    ap.add_argument("--source", help="only use names rows with this source label")
    ap.add_argument("--dry-run", action="store_true", help="count what would be tagged; write nothing")
    a = ap.parse_args(argv)
    for path in [a.opus, a.names] + ([a.dialogue] if a.dialogue else []):
        if not os.path.exists(path):
            print(f"opus_tags: error: {path}: no such file or folder", file=sys.stderr)
            return 2
    if not os.path.isdir(a.opus):
        print(f"opus_tags: error: {a.opus}: not a folder", file=sys.stderr)
        return 2
    try:
        names, ambiguous = load_names(a.names, a.source)
        dialogue, langs = load_dialogue(a.dialogue) if a.dialogue else ({}, [])
    except (OSError, KeyError, ValueError) as e:
        print(f"opus_tags: error: cannot read a CSV: {e}", file=sys.stderr)
        return 2
    try:
        stats = tag_folder(a.opus, names, dialogue, langs, a.dry_run, ambiguous)
    except OSError as e:
        print(f"opus_tags: error: {e}", file=sys.stderr)
        return 2
    for k, v in stats.items():
        print(f"{k:32} {v:,}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
