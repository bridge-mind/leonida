"""Read local content-definition XML (ASSET-149), without resolving references.

Accepts XML files or rpf9_carve directories. The CLI prints aggregate counts only.
The Python API returns ordered records containing local game data: never publish
those records. Archive offsets identify recovered spans, not named RPF entries.
Usage: `python -I tools/content_definitions.py --help` lists the options.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOTS = {
    "CDataFileMgr__ContentChangeSetDef": "change_set",
    "CDataFileMgr__ContentDef": "content",
    "SSetupData": "setup",
}


class ContentError(ValueError):
    """Invalid content XML; messages never include input values or paths."""


class _OtherRoot(ContentError):
    """Well-formed XML outside the supported content schemas."""


@dataclass(frozen=True)
class Record:
    tag: str
    attributes: tuple[tuple[str, str], ...]
    text: str
    tail: str
    children: tuple[Record, ...]

    def select(self, *path: str) -> tuple[Record, ...]:
        """Select ordered direct-child paths; duplicates are not collapsed."""
        nodes = (self,)
        for tag in path:
            nodes = tuple(child for node in nodes for child in node.children if child.tag == tag)
        return nodes

    def scalar(self, tag: str) -> str | None:
        """One text/value field, untyped; absent is None, explicitly empty is ''."""
        fields = self.select(tag)
        if not fields:
            return None
        if len(fields) != 1 or fields[0].children:
            raise ContentError("ambiguous scalar field")
        field = fields[0]
        text = field.text.strip()
        attrs = dict(field.attributes)
        if "value" in attrs:
            if text:
                raise ContentError("scalar has both text and value attribute")
            return attrs["value"]
        return text


@dataclass(frozen=True)
class FileEntry:
    filename: str | None
    file_type: str | None
    record: Record


@dataclass(frozen=True)
class ContentDocument:
    kind: str
    root: Record

    @property
    def files(self) -> tuple[FileEntry, ...]:
        if self.kind != "change_set":
            return ()
        return tuple(FileEntry(item.scalar("filename"), item.scalar("fileType"), item)
                     for item in self.root.select("fileList", "Item"))

    @property
    def included_changesets(self) -> tuple[Record, ...]:
        return self.root.select("includedChangesets", "Item") if self.kind == "content" else ()

    @property
    def included_sets(self) -> tuple[Record, ...]:
        return self.root.select("includedSets", "Item") if self.kind == "content" else ()

    @property
    def change_set_groups(self) -> tuple[Record, ...]:
        return self.root.select("contentChangeSetGroups", "Item") if self.kind == "setup" else ()


class _Builder(ET.TreeBuilder):
    def __init__(self):
        super().__init__()
        self.depth = 0

    def start(self, tag, attrs):
        self.depth += 1
        if self.depth > 128:
            raise ContentError("XML nesting exceeds 128 elements")
        return super().start(tag, attrs)

    def end(self, tag):
        self.depth -= 1
        return super().end(tag)

    def doctype(self, name, pubid, system):
        raise ContentError("DTD declarations are unsupported")


def parse_content(chunks, *, max_bytes: int = 64 << 20) -> ContentDocument:
    """Parse one document from byte chunks. No partially parsed result escapes.

    Preserves all elements, attributes, text and tails (after XML normalization),
    including unknown fields. Comments/PIs are not configuration fields. This
    reader retains a whole document, with a configurable byte limit, not a whole
    archive. No path, macro, hash or include is resolved and no numeric conversion
    is inferred. Namespaced roots are not silently treated as the known schema.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    parser = ET.XMLParser(target=_Builder())
    total = 0
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("content input must contain byte chunks")
            total += len(chunk)
            if total > max_bytes:
                raise ContentError("XML exceeds byte limit")
            parser.feed(chunk)
        element = parser.close()
    except ContentError:
        raise
    except (ET.ParseError, LookupError, ValueError):
        raise ContentError("malformed or unsupported XML") from None
    if element.tag not in ROOTS:
        raise _OtherRoot("unsupported content root")

    def record(el):
        return Record(el.tag, tuple(el.attrib.items()), el.text or "", el.tail or "",
                      tuple(record(child) for child in el))

    result = ContentDocument(ROOTS[element.tag], record(element))
    # Validate the known scalar fields exposed by the file-entry convenience API.
    result.files
    for item in result.included_changesets:
        item.scalar("loadPath")
    return result


def file_chunks(stream):
    while chunk := stream.read(65536):
        yield chunk


def carve_documents(directory):
    """Yield (offset, stored size, document) for XML payloads in an existing carve.

    Existing grouping determines spans; it does not recover encrypted TOC names.
    Strict reads prevent missing .bin files falling back to index header bytes.
    Non-content XML yields None; malformed content raises ContentError.
    """
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import gamedata_index as gi

    class Source(gi.DirectorySource):
        def read(self, row):
            ext = ".bk2" if row.get("type") == "bink2" else ".bin"
            with open(os.path.join(self.directory, f'{row["off"]:012x}{ext}'), "rb") as stream:
                # Grouping only needs the Bink signature, not the video payload.
                return stream.read(64) if ext == ".bk2" else stream.read()

    source = Source(directory)
    positions = {r["off"]: i for i, r in enumerate(source.rows)}
    for row in gi.build_rows(source.rows, source):
        if row["type"] != "xml":
            continue
        if row["root"] not in ROOTS:
            yield row["offset"], row["stored"], None
            continue
        first = positions[row["offset"]]
        group = source.rows[first:first + row["chunks"]]
        yield row["offset"], row["stored"], parse_content(source.read(r) for r in group)


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", nargs="+", help="local XML files or carve directories")
    args = cli.parse_args()
    counts = dict(documents=0, change_sets=0, contents=0, setups=0, files=0,
                  included_changesets=0, included_sets=0, groups=0,
                  other_xml=0, failed_inputs=0)

    def add(doc):
        if doc is None:
            counts["other_xml"] += 1
            return
        counts["documents"] += 1
        counts[{"change_set": "change_sets", "content": "contents", "setup": "setups"}[doc.kind]] += 1
        counts["files"] += len(doc.files)
        counts["included_changesets"] += len(doc.included_changesets)
        counts["included_sets"] += len(doc.included_sets)
        counts["groups"] += len(doc.change_set_groups)

    status = 0
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"ERROR: {path}: does not exist", file=sys.stderr)
            counts["failed_inputs"] += 1
            status = 2
            continue
        try:
            if os.path.isdir(path):
                for _, _, doc in carve_documents(path):
                    add(doc)
            else:
                with open(path, "rb") as stream:
                    try:
                        doc = parse_content(file_chunks(stream))
                    except _OtherRoot:
                        doc = None
                    add(doc)
        except (OSError, ContentError, ValueError, KeyError, TypeError):
            counts["failed_inputs"] += 1
            status = max(status, 1)
    print(json.dumps(counts, sort_keys=True))
    return status


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
