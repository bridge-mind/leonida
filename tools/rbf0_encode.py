"""Encode an RBF0 tree document back to bytes, and check that decoding and encoding give the original resource.

`rbf0_tree.parse_resource` turns one RBF0 resource into a tree document (docs/rbf0.md section 9). `encode(doc)`
writes that tree back as an RBF0 record stream with the grammar of docs/rbf0.md section 3: the `FB FF 01 00`
prologue and 8-byte values for the wide layout, no prologue and 4-byte values for the narrow one; each descriptor
name at its first use, in index order; an element's three header words `0, 0, attribute count`; its attributes, then
its content in tree order; text and strings with the bytes and terminating NUL the tree records.

The encoder fails closed. When the tree cannot hold the original bytes it raises EncodeError with a short reason
instead of writing something else:
    status       the parse was not complete (bad, truncated, or bytes after the root closed)
    utf8         a descriptor name, string or text had invalid UTF-8, or an edit contains an unpaired surrogate
    nan          a float or vec3 component was NaN, whose bits the tree renders as text (infinities are exact)
    order        a value counted as an attribute came after the element's content started, or an element short of
                 its attribute count has a value in its content, which would decode as an attribute
    descriptor   a descriptor index outside the table, or a first use out of order
    length       a string or text no longer has its recorded byte length
    value        a value the record cannot hold (an integer out of range, a float that overflows the layout, an
                 unknown type, more attribute records than the element's attribute count)

Usage:
    python -I rbf0_encode.py --check <carve-dir | resource> [--limit N]
        Decode every RBF0 resource (joining 64 KiB continuations as rbf0_tree.py does), encode it, and compare the
        result with the original bytes. Prints counts only: exact, different, and refused by reason. Exit 1 when a
        resource encodes to different bytes or was refused: only a run where every resource is exact exits 0.
    python -I rbf0_encode.py <rbf0_tree.json> <new_out_dir>
        Write each resource of an rbf0_tree.py JSON report (possibly edited) as <offset>.rbf. The directory must be
        new or empty. Exit 1 if any resource is refused; valid resources are still written.

Everything written is game data: keep it in your work folder, out of git, issues and pull requests.
"""
import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rbf0_tree  # noqa: E402

MAGIC = rbf0_tree.MAGIC
PROLOGUE = rbf0_tree.PROLOGUE
REASONS = ("status", "utf8", "nan", "order", "descriptor", "length", "value")


class EncodeError(ValueError):
    """The tree cannot be written back byte for byte; `reason` is one of REASONS."""

    def __init__(self, reason, message):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class _Writer:
    def __init__(self, doc):
        self.wide = doc.get("layout") == "wide-64"
        self.out = bytearray(MAGIC + (PROLOGUE if self.wide else b""))
        self.names = [d.get("name") for d in doc.get("descriptors") or []]
        self.bad_names = {d["index"] for d in doc.get("descriptors") or [] if d.get("utf8_valid") is False}
        self.emitted = 0

    def record(self, node, kind):
        index = node.get("descriptor")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 0xFD:
            raise EncodeError("descriptor", f"descriptor index {index!r} cannot be written")
        if index > self.emitted:
            raise EncodeError("descriptor", f"descriptor {index} is used before descriptor {self.emitted}")
        self.out += bytes((index, kind))
        name = node.get("name", "")
        if index == self.emitted:
            if index in self.bad_names:
                raise EncodeError("utf8", f"descriptor {index} name had invalid UTF-8")
            if index < len(self.names) and self.names[index] not in (None, name):
                raise EncodeError("descriptor", f"descriptor {index} is named both {self.names[index]!r} and {name!r}")
            if index == len(self.names):
                self.names.append(name)
            else:
                self.names[index] = name
            raw = name.encode("utf-8")
            if len(raw) > 0xFFFF:
                raise EncodeError("length", f"descriptor {index} name is {len(raw)} bytes")
            self.out += struct.pack("<H", len(raw)) + raw
            self.emitted += 1
        elif index < len(self.names) and self.names[index] not in (None, name):
            raise EncodeError("descriptor", f"descriptor {index} is named both {self.names[index]!r} and {name!r}")

    def float_bits(self, value, special):
        if special is not None:
            if special == "nan":
                raise EncodeError("nan", "a NaN float's bits are not in the tree")
            value = {"inf": math.inf, "-inf": -math.inf}.get(special)
            if value is None:
                raise EncodeError("value", f"float spelling {special!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EncodeError("value", f"float value {value!r}")
        fval = float(value)
        if math.isnan(fval):
            raise EncodeError("nan", "a NaN float's bits are not in the tree")
        try:
            return struct.pack("<d" if self.wide else "<f", fval)
        except OverflowError:
            raise EncodeError("value", f"float value {value!r} overflows {'double' if self.wide else 'float32'}") from None

    def value(self, node):
        kind = node.get("type")
        value = node.get("value")
        if kind == "int":
            limit = 1 << (64 if self.wide else 32)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
                raise EncodeError("value", f"integer {value!r} does not fit {'8' if self.wide else '4'} bytes")
            self.record(node, 0x10)
            self.out += struct.pack("<Q" if self.wide else "<I", value)
        elif kind == "bool":
            if not isinstance(value, bool):
                raise EncodeError("value", f"bool value {value!r}")
            self.record(node, 0x20 if value else 0x30)
        elif kind == "float":
            bits = self.float_bits(value, node.get("float_special"))
            self.record(node, 0x40)
            self.out += bits
        elif kind == "vec3":
            specials = node.get("float_special") or [None, None, None]
            if not isinstance(value, list) or len(value) != 3 or len(specials) != 3:
                raise EncodeError("value", f"vec3 value {value!r}")
            bits = b"".join(self.float_bits(v, s if v is None else None) for v, s in zip(value, specials))
            self.record(node, 0x50)
            self.out += bits
        elif kind == "string":
            if node.get("utf8_valid") is False:
                raise EncodeError("utf8", "a string had invalid UTF-8")
            if not isinstance(value, str):
                raise EncodeError("value", f"string value {value!r}")
            raw = value.encode("utf-8")
            if len(raw) > 0xFFFF or ("byte_length" in node and len(raw) != node["byte_length"]):
                raise EncodeError("length", f"string of {len(raw)} bytes, recorded {node.get('byte_length')}")
            self.record(node, 0x60)
            self.out += struct.pack("<H", len(raw)) + raw
        else:
            raise EncodeError("value", f"value type {kind!r}")

    def text(self, node):
        if node.get("utf8_valid") is False:
            raise EncodeError("utf8", "a text record had invalid UTF-8")
        text = node.get("text", "")
        if not isinstance(text, str):
            raise EncodeError("value", f"text value {text!r}")
        raw = text.encode("utf-8") + (b"\x00" if node.get("nul_terminated") else b"")
        if len(raw) > 0xFFFFFFFF or ("length" in node and len(raw) != node["length"]):
            raise EncodeError("length", f"text of {len(raw)} bytes, recorded {node.get('length')}")
        self.out += b"\xfd\xff" + struct.pack("<I", len(raw)) + raw

    def element(self, node):
        if node.get("attribute_order_violation"):
            raise EncodeError("order", f"element {node.get('name')!r} has a value attribute after its content")
        attrs = node.get("attrs_declared", len(node.get("attributes", [])))
        if not isinstance(attrs, int) or isinstance(attrs, bool) or not 0 <= attrs <= 0xFFFF:
            raise EncodeError("value", f"attribute count {attrs!r}")
        # The header word says how many of the following value records are attributes. One record more would decode
        # as the element's first content value; with fewer, the next content value would decode as an attribute.
        listed = len(node.get("attributes", []))
        if listed > attrs:
            raise EncodeError("value", f"element {node.get('name')!r} has {listed} attribute records, count {attrs}")
        if listed < attrs and any(c.get("kind") == "value" for c in node.get("content", [])):
            raise EncodeError("order", f"element {node.get('name')!r} is short of its {attrs} attributes and has a "
                                       "value in its content")
        self.record(node, 0x00)
        self.out += struct.pack("<HHH", 0, 0, attrs)
        for child in node.get("attributes", []):
            self.value(child)
        for child in node.get("content", []):
            kind = child.get("kind")
            if kind == "element":
                self.element(child)
            elif kind == "value":
                self.value(child)
            elif kind == "text":
                self.text(child)
            else:
                raise EncodeError("value", f"content node kind {kind!r}")
        self.out += b"\xff\xff"


def encode(doc):
    """The RBF0 bytes of one rbf0_tree document. Raises EncodeError when they would not be the original's."""
    if not isinstance(doc, dict):
        raise EncodeError("value", f"resource document {doc!r}")
    if doc.get("status") != "ok":
        raise EncodeError("status", f"the parse status is {doc.get('status')!r}")
    if doc.get("layout") not in ("wide-64", "narrow-32"):
        raise EncodeError("value", f"layout {doc.get('layout')!r}")
    tree = doc.get("tree")
    if not isinstance(tree, dict) or tree.get("kind") != "element":
        raise EncodeError("status", "no root element")
    w = _Writer(doc)
    try:
        w.element(tree)
    except UnicodeEncodeError:
        # JSON accepts escaped lone surrogates, but RBF0 strings must be UTF-8.
        # Refuse the resource through the same path as invalid decoded UTF-8.
        raise EncodeError("utf8", "a descriptor name, string or text contains an unpaired Unicode surrogate") from None
    return bytes(w.out)


def resources(source, limit=None):
    """Yield (doc, original bytes) for every RBF0 resource in a carve directory or a single file. The bytes are
    the resource's chunks joined as rbf0_tree.decode_source joined them; check() compares their length with the
    size the document records."""
    docs, _summary = rbf0_tree.decode_source(source, limit=limit)
    rows, read = rbf0_tree._rows(source)
    by_name = {}
    for i, row in enumerate(rows):
        if row["name"] in by_name:      # a document names its first chunk; two rows of one name cannot be told apart
            raise rbf0_tree.SelectionError(f"two carved streams are named {row['name']!r}")
        by_name[row["name"]] = i
    for doc in docs:
        i = by_name[doc["source"]]
        yield doc, b"".join(read(rows[j]) for j in range(i, i + doc["chunks"]))


def check(source, limit=None):
    """Counts for the round trip over `source`: total, exact, different and refused by reason."""
    counts = {"resources": 0, "exact": 0, "different": 0, "refused": 0}
    reasons = dict.fromkeys(REASONS, 0)
    layouts = {}
    for doc, blob in resources(source, limit):
        counts["resources"] += 1
        layout = layouts.setdefault(doc["layout"], {"resources": 0, "exact": 0})
        layout["resources"] += 1
        try:
            encoded = encode(doc)
        except EncodeError as e:
            counts["refused"] += 1
            reasons[e.reason] += 1
            continue
        if encoded == blob and doc.get("size") == len(blob):
            counts["exact"] += 1
            layout["exact"] += 1
        else:
            counts["different"] += 1
    return {**counts, "refused_by_reason": {k: v for k, v in reasons.items() if v}, "by_layout": layouts}


def write_all(report, out_dir):
    """Encode every resource of an rbf0_tree JSON report into out_dir. Returns (written, refused)."""
    try:
        with open(report, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:  # ValueError: bad JSON or non-UTF-8 bytes
        raise SystemExit(f"{report}: cannot read report: {e}") from None
    if not isinstance(data, dict) or data.get("tool") != "rbf0_tree":
        raise SystemExit(f"{report}: not an rbf0_tree.py JSON report")
    items = data.get("resources", [])
    if not isinstance(items, list):
        raise SystemExit(f"{report}: 'resources' must be a list")
    try:
        if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
            raise SystemExit(f"refusing to write into an existing non-empty path: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"{out_dir}: cannot prepare output directory: {e.strerror or e}") from None
    written, refused = 0, []
    for i, doc in enumerate(items):
        raw_offset = doc.get("offset", i) if isinstance(doc, dict) else i
        try:
            blob = encode(doc)
        except EncodeError as e:
            refused.append((raw_offset, str(e)))
            continue
        file_offset = raw_offset if isinstance(raw_offset, int) and not isinstance(raw_offset, bool) and raw_offset >= 0 else i
        name = f"{file_offset:08x}.rbf"
        if os.path.exists(os.path.join(out_dir, name)):
            name = f"{file_offset:08x}-{i:04d}.rbf"
        target = os.path.join(out_dir, name)
        try:
            with open(target, "xb") as f:
                f.write(blob)
        except OSError as e:
            raise SystemExit(f"{target}: cannot write: {e.strerror or e}") from None
        written += 1
    return written, refused


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", metavar="SOURCE", help="round-trip every resource of a carve directory or file")
    ap.add_argument("--limit", type=int, help="with --check, stop after N resources")
    ap.add_argument("report", nargs="?", help="an rbf0_tree.py JSON report to encode")
    ap.add_argument("out", nargs="?", help="a new or empty output directory")
    a = ap.parse_args(argv)
    if a.check:
        if a.report or a.out:
            ap.error("--check takes no report or output directory")
        if a.limit is not None and a.limit < 1:
            ap.error("--limit must be at least 1")
        if not os.path.exists(a.check):
            ap.error(f"{a.check}: no such file or directory")
        try:
            r = check(a.check, a.limit)
        except rbf0_tree.SelectionError as e:
            ap.error(str(e))
        print(f"{r['resources']} resource(s): {r['exact']} exact, {r['different']} different, {r['refused']} refused")
        for reason, n in r["refused_by_reason"].items():
            print(f"  refused ({reason}): {n}")
        for layout, c in sorted(r["by_layout"].items()):
            print(f"  {layout}: {c['exact']} of {c['resources']} exact")
        return 1 if r["different"] or r["refused"] else 0
    if not a.report or not a.out:
        ap.error("give --check SOURCE, or a report and an output directory")
    if a.limit is not None:
        ap.error("--limit only applies to --check")
    if not os.path.isfile(a.report):
        ap.error(f"{a.report}: no such file")
    written, refused = write_all(a.report, a.out)
    print(f"{written} resource(s) written, {len(refused)} refused")
    for offset, why in refused:
        print(f"  0x{offset:x}: {why}" if isinstance(offset, int) else f"  {offset}: {why}")
    return 1 if refused else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
