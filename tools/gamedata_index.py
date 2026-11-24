"""Index a carved RPF9 archive by payload type and XML root.

RPF9 stores files as one or more zlib streams, usually 64 KiB of inflated data per stream. The table of
contents is encrypted, so a carve has offsets and sizes but no names. This tool turns that stream list into
a compact payload index:

    python -I gamedata_index.py <carve_dir> [--csv FILE] [--json FILE]
    python -I gamedata_index.py <archive.rpf> [--csv FILE] [--json FILE] [--carve-dir DIR]

<carve_dir> is an rpf9_carve.py output directory (index.json plus <offset>.bin files). For an .rpf, the
carver is run with --index-only in a temporary directory and the streams are inflated on demand, so no
carved files are written. Pass --carve-dir to reuse an existing index.json.

The rows contain offsets, sizes, a coarse type and an XML root element (a schema/class name). They do not
contain element values, entry names or payload bytes, so the CSV/JSON can be reviewed without copying the
game data. Non-XML payloads follow the 64 KiB block rule; a file whose inflated size is an exact multiple of
64 KiB can merge with the next stream. XML documents close on their root element, so their grouping is exact.
The tool is standard library only and never writes outside the requested output files.
Usage: `python -I tools/gamedata_index.py --help` lists the options.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import json
import mmap
import os
import re
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CARVER = os.path.join(ROOT, "rpf9_carve.py")
BOM = b"\xef\xbb\xbf"
XML_DECL = re.compile(rb"<\?xml[^>]*\?>")
XML_OPEN = re.compile(rb"<([A-Za-z_][\w.:-]*)")
XML_CLOSE_TAIL = re.compile(rb"</([A-Za-z_][\w.:-]*)>\s*$")
# Tag-like text inside these cannot open or close an element.
XML_NON_TAG = re.compile(rb"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>", re.S)
MAGIC = {b"MHIU": "MHIU", b"PAMH": "PAMH", b"SPS1": "SPS1", b"RBF0": "RBF0", b"PSIN": "PSIN"}
# A resource whose page directory starts the stream carries "0DRP" at +16 (docs/animation.md, "0DRP page marker").
# Its first 4 KiB can be over 95% printable, so without this check such a stream reads as text, and as XML when it
# holds a "<": data.rpf has 337 of them and no XML at all (docs/gamedata.md, "Relationship to data.rpf").
PAGE_MARKER, PAGE_MARKER_AT = b"0DRP", 16


def printable(data: bytes) -> float:
    sample = data[:4096]
    if not sample:
        return 0.0
    good = sum(1 for c in sample if c in (9, 10, 13) or 32 <= c < 127)
    return good / len(sample)


def _utf8_text(data: bytes) -> bool:
    """True when the leading 4 KiB decodes as printable UTF-8 text (tolerating a split multibyte boundary)."""
    sample = data[:4096]
    if not sample or b"\x00" in sample:
        return False
    start = 0
    while start < min(3, len(sample)) and 0x80 <= sample[start] <= 0xBF:
        start += 1
    sample = sample[start:]
    if not sample:
        return False
    try:
        text = codecs.getincrementaldecoder("utf-8")("strict").decode(sample, final=False)
    except UnicodeDecodeError:
        return False
    if not text:
        return False
    good = sum(1 for ch in text if ch in "\t\r\n" or (ord(ch) >= 0x20 and ord(ch) != 0x7F and ch.isprintable()))
    return good / len(text) >= 0.95


def has_xml_decl(data: bytes) -> bool:
    # xml_root() drops the BOM before looking for the declaration; do the same here, or a BOM followed by
    # whitespace makes the two disagree and the stream is grouped by the 64 KiB binary rule instead.
    text = data[3:] if data.startswith(BOM) else data
    return bool(XML_DECL.match(text) or XML_DECL.match(text.lstrip(b" \t\r\n")))


def xml_root(data: bytes) -> str:
    """First element name after the XML declaration/comments/PIs, or ""."""
    text = data[3:] if data.startswith(BOM) else data
    text = text.lstrip(b" \t\r\n")
    while True:
        m = XML_DECL.match(text)
        if m:
            text = text[m.end():].lstrip(b" \t\r\n")
            continue
        if text.startswith(b"<!--"):
            end = text.find(b"-->")
            text = text[end + 3:].lstrip(b" \t\r\n") if end >= 0 else b""
            continue
        if text.startswith(b"<?"):
            end = text.find(b"?>")
            text = text[end + 2:].lstrip(b" \t\r\n") if end >= 0 else b""
            continue
        break
    m = XML_OPEN.match(text)
    return m.group(1).decode("ascii", "replace") if m else ""


def tail_root(data: bytes) -> str:
    m = XML_CLOSE_TAIL.search(data[-512:])
    return m.group(1).decode("ascii", "replace") if m else ""


def xml_complete(root: str, data: bytes) -> bool:
    """True when every <root> in data is closed, not merely when a </root> occurs somewhere in the tail.

    A nested element with the same name as the root used to satisfy the old substring test, so a truncated
    document was reported complete and its continuation streams were not absorbed into the payload.
    """
    if not root:
        return False
    text = XML_NON_TAG.sub(b"", data)
    name = re.escape(root.encode("ascii", "replace"))
    # A quoted attribute value may contain '>'; consume quoted strings as one unit or a self-closing
    # tag with such a value is mistaken for an unclosed element.
    opens = re.findall(rb"<" + name + rb"(?=[\s/>])(?:\"[^\"]*\"|'[^']*'|[^'\">])*>", text)
    closes = re.findall(rb"</" + name + rb"\s*>", text)
    self_closed = sum(1 for tag in opens if tag.rstrip().endswith(b"/>"))
    if len(opens) - self_closed > len(closes):
        return False
    return bool(closes or self_closed)


def _self_closing_root(root: str, data: bytes) -> bool:
    text = data[3:] if data.startswith(BOM) else data
    text = XML_NON_TAG.sub(b"", text).strip()
    name = re.escape(root.encode("ascii", "replace"))
    return bool(re.fullmatch(rb"<" + name + rb"(?=[\s/])(?:\"[^\"]*\"|'[^']*'|[^'\">])*/>", text))


def standalone_xml(data: bytes) -> bool:
    """A complete document in one stream, with or without a declaration."""
    root = xml_root(data)
    return bool(root) and xml_complete(root, data) and (tail_root(data) == root or _self_closing_root(root, data))


def stream_kind(data: bytes, row: dict) -> str:
    if row.get("type") == "bink2" or data.startswith(b"KB2"):
        return "bink2"
    if has_xml_decl(data):
        return "decl"
    if data.startswith(b"RBF0"):
        return "rbf0"
    if data.startswith(b"PSIN"):
        return "psin"
    for magic in MAGIC:
        if data.startswith(magic):
            return "other"
    if data[PAGE_MARKER_AT:PAGE_MARKER_AT + len(PAGE_MARKER)] == PAGE_MARKER:
        return "other"
    if printable(data) > 0.95:
        return "text"
    return "binary"


def magic_name(data: bytes) -> str:
    for magic, name in MAGIC.items():
        if data.startswith(magic):
            return name
    if data[PAGE_MARKER_AT:PAGE_MARKER_AT + len(PAGE_MARKER)] == PAGE_MARKER:
        return PAGE_MARKER.decode()
    return ""


CARVED_STREAM = re.compile(r"[0-9a-f]{12}\.(?:bin|bk2)")


def carved_stream(address: str, directory: str):
    """The carved stream a report at `address` would overwrite in `directory`, or None.

    rpf9_carve.py writes one `<offset:012x>.bin` (or `.bk2`) per stream beside its index.json, so it is an input,
    not a place for a report; guarding only index.json left `--json`/`--csv` free to replace a carved stream.
    """
    if not CARVED_STREAM.fullmatch(os.path.basename(address)):
        return None
    candidate = os.path.join(directory, os.path.basename(address))
    return candidate if os.path.isfile(candidate) and same_report_path(address, candidate) else None


class DirectorySource:
    """Reads .bin/.bk2 files next to an rpf9_carve.py index.json."""

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "index.json"), encoding="utf-8-sig") as f:
            self.index = json.load(f)
        self.rows = sorted(self.index["files"], key=lambda r: r["off"])

    def read(self, row: dict) -> bytes:
        ext = ".bk2" if row.get("type") == "bink2" else ".bin"
        path = os.path.join(self.directory, f'{row["off"]:012x}{ext}')
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return bytes.fromhex(row.get("head", ""))

    @property
    def summary(self) -> dict:
        return self.index.get("summary", {})


class ArchiveSource:
    """Reads streams on demand from the archive, using an index-only carve."""

    def __init__(self, archive: str, index_dir: str | None = None):
        self.archive = archive
        self._tmp = None
        if index_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="gamedata-index-")
            index_dir = self._tmp.name
            r = subprocess.run([sys.executable, "-I", CARVER, archive, index_dir, "--index-only"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError((r.stdout + r.stderr).strip() or "rpf9_carve.py failed")
        with open(os.path.join(index_dir, "index.json"), encoding="utf-8-sig") as f:
            self.index = json.load(f)
        self.rows = sorted(self.index["files"], key=lambda r: r["off"])
        self._fh = open(archive, "rb")
        self._map = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, row: dict) -> bytes:
        if row.get("type") == "bink2":
            return self._map[row["off"]:row["off"] + min(row["stored"], 64)]
        try:
            d = zlib.decompressobj()
            out = d.decompress(self._map[row["off"]:row["off"] + row["stored"]])
            return out + d.flush()
        except zlib.error:
            return bytes.fromhex(row.get("head", ""))

    @property
    def summary(self) -> dict:
        return self.index.get("summary", {})

    def close(self) -> None:
        self._map.close()
        self._fh.close()
        if self._tmp is not None:
            self._tmp.cleanup()


def build_rows(rows: list, source) -> list:
    """Group streams into payloads; XML documents are closed by their root element."""
    out = []
    n = len(rows)
    i = 0
    while i < n:
        data = source.read(rows[i])
        kind = stream_kind(data, rows[i])

        if kind == "decl":
            root = xml_root(data)
            group = [i]
            blob = data
            j = i + 1
            while j < n and not xml_complete(root, blob):
                nxt = source.read(rows[j])
                nk = stream_kind(nxt, rows[j])
                if nk in ("decl", "bink2", "rbf0", "psin", "other", "binary"):
                    break
                blob += nxt
                group.append(j)
                j += 1
            out.append(row_for("xml", rows, group, root=root, header=True,
                               complete=xml_complete(root, blob)))
            i = j
            continue

        if kind == "other" or kind == "bink2":
            group = [i]
            detail = magic_name(data)
            j = i + 1
            # A resource's first stream may be any whole number of 64 KiB blocks (docs/page_directory.md: 128 and
            # 192 KiB first streams continue in the next one). Only these headed kinds take that rule: a binary,
            # text or XML run has no header proving a continuation, so it keeps the exact 64 KiB rule below.
            while j < n and rows[j - 1]["size"] and rows[j - 1]["size"] % 65536 == 0:
                nxt = source.read(rows[j])
                nk = stream_kind(nxt, rows[j])
                if nk in ("decl", "bink2", "rbf0", "psin", "other"):
                    break
                group.append(j)
                j += 1
            typ = "bink2" if kind == "bink2" else "other"
            out.append(row_for(typ, rows, group, detail=detail))
            i = j
            continue

        if kind in ("rbf0", "psin", "text", "binary"):
            group = [i]
            chunks = [data] if kind == "text" and b"<" in data else None
            start_root = xml_root(data) if chunks is not None else ""
            j = i + 1
            if not (chunks is not None and standalone_xml(data)):
                while j < n and rows[j - 1]["size"] == 65536:
                    nxt = source.read(rows[j])
                    if standalone_xml(nxt):
                        break
                    nk = stream_kind(nxt, rows[j])
                    if nk in ("decl", "bink2", "rbf0", "psin", "other") or (
                        chunks is not None and nk == "binary" and not _utf8_text(nxt)
                    ):
                        break
                    if chunks is not None:
                        chunks.append(nxt)
                    group.append(j)
                    j += 1
                    if chunks is not None and start_root and tail_root(nxt) == start_root and xml_complete(start_root, b"".join(chunks)):
                        break
            if chunks is not None:
                blob = b"".join(chunks)
                root = tail_root(blob) or (xml_root(blob) if standalone_xml(blob) else "")
                out.append(row_for("xml", rows, group, root=root, header=False, complete=bool(root)))
            else:
                out.append(row_for(kind, rows, group))
            i = j
            continue

        # An XML continuation stream with no declaration: a tail-only document if it closes a root.
        group = [i]
        blob = data
        j = i + 1
        while j < n and rows[j - 1]["size"] == 65536:
            nxt = source.read(rows[j])
            if stream_kind(nxt, rows[j]) in ("decl", "bink2", "rbf0", "psin", "other", "binary"):
                break
            blob += nxt
            group.append(j)
            j += 1
        root = tail_root(blob)
        out.append(row_for("xml", rows, group, root=root, header=False,
                           complete=bool(root) and xml_complete(root, blob)))
        i = j
    return out


def row_for(typ: str, rows: list, group: list, root: str = "", detail: str = "",
            header: bool = False, complete: bool = True) -> dict:
    first = rows[group[0]]
    return {
        "offset": first["off"],
        "offset_hex": f'{first["off"]:x}',
        "stored": sum(rows[x]["stored"] for x in group),
        "size": sum(rows[x]["size"] for x in group),
        "type": typ,
        "root": root,
        "detail": detail,
        "chunks": len(group),
        "complete": complete,
        "header": header,
    }


def summarize(rows: list, source_summary: dict, target: str) -> dict:
    by_type = defaultdict(lambda: {"files": 0, "streams": 0, "stored": 0, "inflated": 0})
    by_root = defaultdict(lambda: {"files": 0, "inflated": 0})
    for r in rows:
        t = by_type[r["type"]]
        t["files"] += 1
        t["streams"] += r["chunks"]
        t["stored"] += r["stored"]
        t["inflated"] += r["size"]
        if r["root"]:
            by_root[r["root"]]["files"] += 1
            by_root[r["root"]]["inflated"] += r["size"]
    xml = [r for r in rows if r["type"] == "xml"]
    streams = source_summary.get("files", 0)
    if isinstance(streams, list):
        streams = len(streams)
    return {
        "target": target,
        "streams": streams,
        "stored": sum(r["stored"] for r in rows),
        "inflated": sum(r["size"] for r in rows),
        "zero_padding": source_summary.get("zero_padding"),
        "unread_bytes": source_summary.get("unread_bytes"),
        "types": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["inflated"])},
        "xml_documents": sum(1 for r in xml if r["complete"]),
        "xml_with_header": sum(1 for r in xml if r["header"]),
        "xml_tail_only": sum(1 for r in xml if r["complete"] and not r["header"]),
        "xml_incomplete": sum(1 for r in xml if not r["complete"]),
        "roots": {k: dict(v) for k, v in sorted(by_root.items(), key=lambda kv: (-kv[1]["files"], kv[0]))},
    }


def _make_parent(path: str) -> None:
    """Create the folder an output path lives in: the scan is over by the time we write, so a missing
    folder used to end the run in a FileNotFoundError and lose the result."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, rows: list) -> None:
    _make_parent(path)
    fields = ["offset", "offset_hex", "stored", "size", "type", "root", "detail", "chunks",
              "complete", "header"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="carve directory or .rpf archive")
    ap.add_argument("--carve-dir", help="existing rpf9_carve.py index.json directory for an .rpf")
    ap.add_argument("--csv", help="write one row per payload to this CSV")
    ap.add_argument("--json", help="write the summary and rows to this JSON")
    a = ap.parse_args()
    if a.csv and a.json and same_report_path(a.csv, a.json):
        ap.error("--csv and --json must use different paths")

    if not os.path.exists(a.target):
        print(f"ERROR: {a.target} does not exist", file=sys.stderr)
        return 2
    protected = [a.target] if os.path.isfile(a.target) else [os.path.join(a.target, "index.json")]
    if a.carve_dir:
        protected.append(os.path.join(a.carve_dir, "index.json"))
    for flag, path in (("--json", a.json), ("--csv", a.csv)):
        if path:
            if os.path.isdir(path):
                ap.error(f"{flag} is a directory: {path}")
            for input_path in protected:
                if same_report_path(path, input_path):
                    ap.error(f"{flag} would overwrite the input {input_path}")
            for directory in (a.target, a.carve_dir):
                stream = directory and os.path.isdir(directory) and carved_stream(path, directory)
                if stream:
                    ap.error(f"{flag} would overwrite the carved stream {stream}")
    source = None
    try:
        if os.path.isdir(a.target):
            source = DirectorySource(a.target)
        else:
            source = ArchiveSource(a.target, a.carve_dir)
        rows = build_rows(source.rows, source)
        summary = summarize(rows, source.summary, a.target)
    except RuntimeError as e:          # rpf9_carve.py refused the archive; its own message says why
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"ERROR: {e.filename} is missing. Give a carve folder from rpf9_carve.py or an .rpf archive.",
              file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        where = getattr(e, "filename", None) or a.target
        print(f"ERROR: cannot read {where}: {getattr(e, 'strerror', None) or e!r}", file=sys.stderr)
        return 2
    finally:
        if isinstance(source, ArchiveSource):
            source.close()

    if a.csv:
        try:
            write_csv(a.csv, rows)
        except OSError as e:
            print(f"ERROR: cannot write {a.csv}: {e.strerror or e}", file=sys.stderr)
            return 2
    if a.json:
        try:
            _make_parent(a.json)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": summary, "rows": rows}, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2

    print(f'{summary["streams"]} streams, {summary["stored"]:,} stored, '
          f'{summary["inflated"]:,} inflated')
    for name, t in summary["types"].items():
        print(f'  {name:8} {t["files"]:4} payloads {t["streams"]:5} streams '
              f'{t["stored"]:10,} stored {t["inflated"]:12,} inflated')
    print(f'XML: {summary["xml_documents"]} complete documents '
          f'({summary["xml_with_header"]} with a declaration, {summary["xml_tail_only"]} tail-only), '
          f'{summary["xml_incomplete"]} incomplete, {len(summary["roots"])} roots')
    if a.csv:
        print("wrote", a.csv)
    if a.json:
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
