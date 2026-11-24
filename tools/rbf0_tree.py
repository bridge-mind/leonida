"""Decode RBF0 resources into strict JSON trees, with an optional CSV node table and simple XML rendering.

RBF0 is RAGE's binary form of an XML data file: one record stream with an inline descriptor table. This tool reads
inflated RBF0 resources from a carve directory or a single resource file, joins the 64 KiB continuation streams of
one logical resource, and builds the element tree. It reports unknown descriptors, unknown record types, truncated
payloads and structural violations as errors instead of guessing a value.

The output contains names and payloads taken from the archive, so treat it as game data: keep it out of git and do
not attach it to an issue or a pull request. Format facts and aggregate counts are in docs/rbf0.md.

Usage:
    python -I tools/rbf0_tree.py <carve-dir> <out.json> [--csv PATH] [--xml DIR] [--root NAME]... [--offset 0x...] [--limit N]

The input directory holds inflated zlib streams or complete RBF0 resources. A file name that starts with a
hexadecimal archive offset is used to order and label the resource, as in the rbf0_index.py carve layout.
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import struct
import sys
import tempfile
from contextlib import suppress
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rbf0_index  # noqa: E402
from file_mode import published_mode  # noqa: E402

MAGIC = rbf0_index.MAGIC
PROLOGUE = rbf0_index.PROLOGUE
MAX_CHUNK = rbf0_index.MAX_CHUNK
MAX_CHUNKS = rbf0_index.MAX_CHUNKS
MAX_BYTES = rbf0_index.MAX_BYTES
RbfError = rbf0_index.RbfError
TYPE_NAMES = {0x00: "open", 0x10: "int", 0x20: "true", 0x30: "false", 0x40: "float",
              0x50: "vec3", 0x60: "string"}
VALUE_TYPES = {0x10: "int", 0x20: "bool", 0x30: "bool", 0x40: "float", 0x50: "vec3", 0x60: "string"}
XML_BAD = re.compile(r"[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")
CSV_COLUMNS = ["resource", "root", "status", "parent", "slot", "sibling", "child_index", "depth", "kind",
               "name", "descriptor", "offset", "end", "attrs", "children", "content_count", "type",
               "length", "nul", "utf8", "value", "source"]


def _utf8(raw):
    """Return (text, valid). Invalid bytes become U+FFFD and are counted, never silently accepted."""
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", "replace"), False


def _render(value):
    """A stable textual form for XML and for JSON consumers that want one scalar spelling."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    return str(value)


class TreeError(RbfError):
    """A malformed or truncated RBF0 record stream."""


class SelectionError(ValueError):
    """An invalid resource offset or limit."""


class Parser:
    """One RBF0 resource. `parse()` always returns a document, with a partial tree on failure."""

    def __init__(self, data, source="", offset=0, chunks=1):
        self.data = data
        self.source = source
        self.offset = offset
        self.chunks = chunks
        self.wide = data[4:8] == PROLOGUE
        self.p = 8 if self.wide else 4
        self.scalar = 8 if self.wide else 4
        self.descs = []
        self.stack = []
        self.root = None
        self.status = "ok"
        self.error = ""
        self.error_offset = None
        self.counts = collections.Counter()
        self.descriptor_names = set()
        self.attribute_mismatches = 0
        self.attribute_order_violations = 0
        self.text_after_child = 0
        self.text_before_child = 0
        self.value_after_child = 0
        self.invalid_utf8 = 0
        self.max_depth = 0

    def take(self, n, what):
        if self.p + n > len(self.data):
            raise TreeError(f"{what} truncated", self.p, truncated=True)
        start = self.p
        self.p += n
        return self.data[start:self.p]

    def _descriptor(self, index, kind, at):
        if index == len(self.descs):
            n = struct.unpack("<H", self.take(2, "descriptor name length"))[0]
            raw = self.take(n, "descriptor name")
            name, valid = _utf8(raw)
            if not valid:
                self.invalid_utf8 += 1
            self.descs.append({
                "index": index,
                "name": name,
                "first_type": TYPE_NAMES.get(kind, f"0x{kind:02x}"),
                "offset": at,
                "name_offset": self.p - n,
                "uses": 0,
                "types": {},
                "utf8_valid": valid,
            })
            self.descriptor_names.add(name)
        elif index > len(self.descs):
            raise TreeError(f"descriptor index {index} of {len(self.descs)}", at)
        desc = self.descs[index]
        desc["uses"] += 1
        key = TYPE_NAMES.get(kind, f"0x{kind:02x}")
        desc["types"][key] = desc["types"].get(key, 0) + 1
        return desc

    def _attach(self, frame, node):
        node["depth"] = frame["depth"] + 1
        if frame["node"]["kind"] != "element":
            raise TreeError("record outside an element", node["offset"])
        if node["kind"] == "value" and frame["attrs_seen"] < frame["attrs_declared"]:
            if frame["content_started"]:
                frame["node"]["attribute_order_violation"] = True
                self.attribute_order_violations += 1
            frame["node"]["attributes"].append(node)
            frame["node"]["attrs_actual"] += 1
            frame["attrs_seen"] += 1
            return
        if node["kind"] in ("element", "text") and frame["attrs_seen"] < frame["attrs_declared"]:
            frame["node"]["attribute_order_violation"] = True
            self.attribute_order_violations += 1
        frame["content_started"] = True
        frame["node"]["content"].append(node)
        frame["node"]["content_count"] += 1
        if node["kind"] == "element":
            frame["node"]["children"] += 1
            if frame["text_seen"]:
                frame["node"]["text_before_child"] = True
                self.text_before_child += 1
            frame["child_seen"] = True
        elif node["kind"] == "value":
            if frame["child_seen"]:
                frame["node"]["value_after_child"] = True
                self.value_after_child += 1
        elif node["kind"] == "text":
            if frame["child_seen"]:
                frame["node"]["text_after_child"] = True
                self.text_after_child += 1
            if frame["text_seen"]:
                frame["node"]["multiple_text"] = True
            frame["text_seen"] = True

    def parse(self):
        data = self.data
        if len(data) < 4 or data[:4] != MAGIC:
            raise TreeError("not an RBF0 stream", 0)
        try:
            while self.p < len(data):
                self.record_offset = self.p
                at = self.p
                index = self.take(1, "record index")[0]
                kind = self.take(1, "record type")[0]
                if index == 0xFF:
                    if kind != 0xFF:
                        raise TreeError("close tag is not FF FF", at)
                    if not self.stack:
                        raise TreeError("close without an open element", at)
                    frame = self.stack.pop()
                    frame["node"]["end"] = self.p
                    if frame["attrs_seen"] < frame["attrs_declared"]:
                        frame["node"]["attribute_mismatch"] = True
                        self.attribute_mismatches += 1
                    if not self.stack:
                        self.root = frame["node"]
                        break
                    continue
                if index == 0xFD:
                    if kind != 0xFF:
                        raise TreeError("text tag is not FD FF", at)
                    n = struct.unpack("<I", self.take(4, "text length"))[0]
                    raw = self.take(n, "text")
                    payload_at = self.p - n
                    body = raw[:-1] if raw.endswith(b"\x00") else raw
                    text, valid = _utf8(body)
                    if not valid:
                        self.invalid_utf8 += 1
                    node = {
                        "kind": "text",
                        "offset": at,
                        "payload_offset": payload_at,
                        "length": n,
                        "nul_terminated": raw.endswith(b"\x00"),
                        "text": text,
                        "utf8_valid": valid,
                    }
                    if not self.stack:
                        raise TreeError("text outside an element", at)
                    self._attach(self.stack[-1], node)
                    self.counts["texts"] += 1
                    continue
                desc = self._descriptor(index, kind, at)
                name = desc["name"]
                if kind == 0x00:
                    raw = self.take(6, "element header")
                    payload_at = self.p - 6
                    word0, word1, attrs = struct.unpack("<HHH", raw)
                    if word0 or word1:
                        raise TreeError(f"element header word is not zero ({word0}, {word1})", at)
                    node = {
                        "kind": "element",
                        "name": name,
                        "descriptor": index,
                        "offset": at,
                        "header_offset": payload_at,
                        "attrs_declared": attrs,
                        "attrs_actual": 0,
                        "depth": len(self.stack) + 1,
                        "children": 0,
                        "content_count": 0,
                        "attributes": [],
                        "content": [],
                        "end": None,
                    }
                    self.counts["elements"] += 1
                    self.max_depth = max(self.max_depth, len(self.stack) + 1)
                    if self.stack:
                        self._attach(self.stack[-1], node)
                    else:
                        if self.root is not None:
                            raise TreeError("second root element", at)
                        self.root = node
                    self.stack.append({
                        "node": node,
                        "attrs_declared": attrs,
                        "attrs_seen": 0,
                        "content_started": False,
                        "child_seen": False,
                        "text_seen": False,
                        "depth": len(self.stack) + 1,
                    })
                    continue
                if kind not in VALUE_TYPES:
                    raise TreeError(f"unknown record type {kind:#04x}", at)
                node = self._value(kind, name, index, at)
                if not self.stack:
                    raise TreeError("value outside an element", at)
                self._attach(self.stack[-1], node)
                self.counts["values"] += 1
        except TreeError as exc:
            if not self.error:
                self.status = "truncated" if exc.truncated else "bad"
                self.error = str(exc)
                self.error_offset = exc.offset
        if self.status == "ok":
            if self.stack:
                self.status = "truncated"
                self.error = f"{len(self.stack)} unclosed element(s) at 0x{self.p:x}"
                self.error_offset = self.p
            elif self.root is None:
                self.status = "bad"
                self.error = "no root element"
                self.error_offset = self.p
            elif self.root.get("end") != len(data):
                self.status = "trailing"
                self.error = f"{len(data) - self.root['end']} bytes after the root closed"
                self.error_offset = self.root["end"]
        return {
            "source": self.source,
            "offset": self.offset,
            "size": len(data),
            "chunks": self.chunks,
            "layout": "wide-64" if self.wide else "narrow-32",
            "prologue": PROLOGUE.hex() if self.wide else "",
            "root_name": self.root["name"] if self.root else "",
            "status": self.status,
            "error": self.error,
            "error_offset": self.error_offset,
            "end": self.root["end"] if self.root and self.root.get("end") is not None else self.p,
            "descriptors": self.descs,
            "descriptor_count": len(self.descs),
            "distinct_descriptor_names": len(self.descriptor_names),
            "counts": dict(self.counts),
            "attribute_mismatches": self.attribute_mismatches,
            "attribute_order_violations": self.attribute_order_violations,
            "text_after_child": self.text_after_child,
            "text_before_child": self.text_before_child,
            "value_after_child": self.value_after_child,
            "invalid_utf8": self.invalid_utf8,
            "max_depth": self.max_depth,
            "tree": self.root,
        }

    def _value(self, kind, name, index, at):
        payload_at = self.p
        extra = {}
        if kind == 0x10:
            raw = self.take(self.scalar, "integer")
            value = struct.unpack("<Q" if self.wide else "<I", raw)[0]
            self.counts["ints"] += 1
        elif kind == 0x20 or kind == 0x30:
            value = kind == 0x20
            self.counts["bools"] += 1
        elif kind == 0x40:
            raw = self.take(self.scalar, "float")
            value = struct.unpack("<d" if self.wide else "<f", raw)[0]
            self.counts["floats"] += 1
        elif kind == 0x50:
            raw = self.take(3 * self.scalar, "vector")
            value = list(struct.unpack("<ddd" if self.wide else "<fff", raw))
            if any(not math.isfinite(v) for v in value):
                extra["float_special"] = [_render(v) for v in value]
                value = [v if math.isfinite(v) else None for v in value]
            self.counts["vec3"] += 1
        else:  # 0x60
            n = struct.unpack("<H", self.take(2, "string length"))[0]
            raw = self.take(n, "string")
            payload_at = self.p - n
            # The measured string form is length-prefixed, not NUL-terminated. Keep every byte even if the last
            # one is zero; only the text record (whose u32 length includes a terminator in this corpus) is trimmed.
            value, valid = _utf8(raw)
            if not valid:
                self.invalid_utf8 += 1
            extra = {"byte_length": n, "nul_terminated": raw.endswith(b"\x00"), "utf8_valid": valid}
            self.counts["strings"] += 1
        if isinstance(value, float) and not math.isfinite(value):
            extra["float_special"] = _render(value)
            value = None
        return {
            "kind": "value",
            "name": name,
            "descriptor": index,
            "type": VALUE_TYPES[kind],
            "offset": at,
            "payload_offset": payload_at,
            "size": self.p - payload_at,
            "value": value,
            **extra,
        }


def parse_resource(data, source="", offset=0, chunks=1):
    """Parse one complete RBF0 resource into a tree document. A non-RBF0 magic raises; a malformed stream
    returns a partial document with status bad or truncated."""
    return Parser(data, source, offset, chunks).parse()


def _rows(source):
    """Return (rows, read) for a single resource or a directory of inflated streams/resources."""
    if os.path.isdir(source):
        entries = []
        unnamed = []
        for name in sorted(os.listdir(source)):
            full = os.path.join(source, name)
            # rpf9_carve.py's index.json and any .csv/.json report sit next to the streams; they are not payloads.
            if not os.path.isfile(full) or name.lower().endswith((".json", ".csv")):
                continue
            with open(full, "rb") as f:
                head = f.read(8)
            match = re.match(r"([0-9a-fA-F]{6,})", name)
            if match:
                entries.append((int(match.group(1), 16), name, full, head))
            else:
                unnamed.append((name, full, head))
        next_offset = max((e[0] for e in entries), default=-1) + 1
        for name, full, head in unnamed:
            entries.append((next_offset, name, full, head))
            next_offset += 1
        entries.sort(key=lambda e: e[0])
        rows = [{"offset": off, "stored": os.path.getsize(full), "size": os.path.getsize(full),
                 "head": head, "name": name} for off, name, full, head in entries]

        def read(row):
            with open(os.path.join(source, row["name"]), "rb") as f:
                return f.read()
        return rows, read
    with open(source, "rb") as f:
        data = f.read()
    row = {"offset": 0, "stored": len(data), "size": len(data), "head": data[:8],
           "name": os.path.basename(source)}
    return [row], lambda _row: data


def decode_source(source, roots=None, offset=None, limit=None, progress=None):
    """Decode RBF0 resources, joining 64 KiB continuations, and return (docs, summary).

    An explicit offset must be nonnegative and an explicit limit must be positive.
    """
    if limit is not None and limit < 1:
        raise SelectionError("--limit must be at least 1")
    if offset is not None and offset < 0:
        raise SelectionError("--offset must be non-negative")
    rows, read = _rows(source)
    starts = [i for i, row in enumerate(rows) if row["head"][:4] == MAGIC]
    if not starts:
        raise SystemExit(f"no RBF0 resource in {source}")
    if offset is not None:
        starts = [i for i in starts if rows[i]["offset"] == offset]
        if not starts:
            raise SystemExit(f"no RBF0 resource at offset 0x{offset:x}")
    wanted = set(roots or ())
    docs = []
    for i in starts:
        blob = read(rows[i])
        chunks = 1
        doc = parse_resource(blob, rows[i]["name"], rows[i]["offset"], chunks)
        while (doc["status"] == "truncated" and chunks < MAX_CHUNKS and len(blob) < MAX_BYTES
               and rows[i + chunks - 1]["size"] == MAX_CHUNK and i + chunks < len(rows)):
            nxt = rows[i + chunks]
            if nxt["size"] > MAX_CHUNK or nxt["head"][:4] in (MAGIC, b"PSIN"):
                break
            blob += read(nxt)
            chunks += 1
            doc = parse_resource(blob, rows[i]["name"], rows[i]["offset"], chunks)
            if doc["status"] != "truncated":
                break
        if wanted and doc["root_name"] not in wanted:
            continue
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
        if progress:
            progress(len(docs), len(starts))
    summary = _summary(docs)
    summary["source"] = os.path.abspath(source)
    return docs, summary


def _summary(docs):
    roots = collections.Counter(d["root_name"] for d in docs)
    layouts = collections.Counter(d["layout"] for d in docs)
    status = collections.Counter(d["status"] for d in docs)
    counts = collections.Counter()
    for d in docs:
        counts.update(d["counts"])
    by_depth = collections.Counter()
    children_hist = collections.Counter()
    content_hist = collections.Counter()
    attr_hist = collections.Counter()
    nodes = 0
    tree_roots = 0
    leaf_elements = empty_elements = text_only_elements = 0
    max_children = 0
    for d in docs:
        root = d["tree"]
        if root is None:
            continue
        tree_roots += 1
        stack = [root]
        while stack:
            node = stack.pop()
            nodes += 1
            if node["kind"] != "element":
                continue
            by_depth[node["depth"]] += 1
            attr_hist[node["attrs_declared"]] += 1
            child_count = node.get("children", sum(1 for c in node["content"] if c["kind"] == "element"))
            children_hist[child_count] += 1
            content_hist[node.get("content_count", len(node["content"]))] += 1
            max_children = max(max_children, child_count)
            if child_count == 0:
                leaf_elements += 1
            content = node["content"]
            if not content:
                empty_elements += 1
            elif all(c["kind"] == "text" for c in content):
                text_only_elements += 1
            for child in content:
                stack.append(child)
            for attribute in node["attributes"]:
                stack.append(attribute)
    tree_elements = sum(children_hist.values())
    edges = tree_elements - tree_roots
    return {
        "resources": len(docs),
        "ok": status.get("ok", 0),
        "bad": status.get("bad", 0),
        "truncated": status.get("truncated", 0),
        "trailing": status.get("trailing", 0),
        "layouts": dict(layouts),
        "roots": dict(roots),
        "descriptors": sum(d["descriptor_count"] for d in docs),
        "distinct_descriptor_names": len({x["name"] for d in docs for x in d["descriptors"]}),
        "elements": counts.get("elements", 0),
        "values": counts.get("values", 0),
        "texts": counts.get("texts", 0),
        "strings": counts.get("strings", 0),
        "ints": counts.get("ints", 0),
        "floats": counts.get("floats", 0),
        "bools": counts.get("bools", 0),
        "vec3": counts.get("vec3", 0),
        "nodes": nodes,
        "max_depth": max((d["max_depth"] for d in docs), default=0),
        "elements_by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "children_histogram": {str(k): v for k, v in sorted(children_hist.items())},
        "content_length_histogram": {str(k): v for k, v in sorted(content_hist.items())},
        "attribute_count_histogram": {str(k): v for k, v in sorted(attr_hist.items())},
        "leaf_elements": leaf_elements,
        "empty_elements": empty_elements,
        "text_only_elements": text_only_elements,
        "parent_child_edges": edges,
        "max_children": max_children,
        "mean_children": round(edges / tree_elements, 4) if tree_elements else 0.0,
        "attribute_mismatches": sum(d["attribute_mismatches"] for d in docs),
        "attribute_order_violations": sum(d["attribute_order_violations"] for d in docs),
        "text_after_child": sum(d["text_after_child"] for d in docs),
        "value_after_child": sum(d["value_after_child"] for d in docs),
        "invalid_utf8": sum(d["invalid_utf8"] for d in docs),
        "status": dict(status),
    }


def _xml_safe(text):
    return XML_BAD.sub("\ufffd", text)


def _xml_node(parent, node, tag=None):
    if node["kind"] == "element":
        attrs = {"name": _xml_safe(node["name"]), "descriptor": str(node["descriptor"]),
                 "offset": f"0x{node['offset']:x}", "attrs": str(node["attrs_declared"])}
        if node.get("end") is not None:
            attrs["end"] = f"0x{node['end']:x}"
        el = ET.SubElement(parent, "node", attrs)
        for value in node["attributes"]:
            _xml_node(el, value, "attr")
        for child in node["content"]:
            _xml_node(el, child)
        return el
    tag = tag or ("text" if node["kind"] == "text" else "value")
    attrs = {"offset": f"0x{node['offset']:x}"}
    if node["kind"] == "value":
        attrs.update({"name": _xml_safe(node["name"]), "descriptor": str(node["descriptor"]),
                      "type": node["type"]})
        if node.get("float_special") is not None:
            text = _render(node["float_special"])
        else:
            text = _render(node["value"])
    else:
        attrs["length"] = str(node["length"])
        text = node["text"]
    if node.get("utf8_valid") is False:
        attrs["utf8"] = "invalid"
    el = ET.SubElement(parent, tag, attrs)
    el.text = _xml_safe(text)
    return el


def write_xml(directory, docs):
    """Write one generic XML file per resource. Element names are fixed, so any descriptor name is safe."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    used = set()
    for i, doc in enumerate(docs):
        root = ET.Element("rbf0", {
            "root": _xml_safe(doc["root_name"]), "layout": doc["layout"],
            "offset": f"0x{doc['offset']:x}",
            "size": str(doc["size"]), "chunks": str(doc["chunks"]),
            "descriptors": str(doc["descriptor_count"]), "status": doc["status"]})
        if doc["error"]:
            root.set("error", _xml_safe(doc["error"]))
        if doc["tree"] is not None:
            _xml_node(root, doc["tree"])
        ET.indent(root, space=" ")
        name = f"{doc['offset']:08x}.xml"
        if name in used:
            name = f"{doc['offset']:08x}-{i:04d}.xml"
        used.add(name)
        path = os.path.join(directory, name)
        # XML readers normalize literal carriage returns. Escape them after serializing ordinary XML text.
        xml = ET.tostring(root, encoding="utf-8", xml_declaration=True).replace(b"\r", b"&#13;")
        with open(path, "wb") as f:
            f.write(xml)
        paths.append(path)
    return paths


def write_csv(path, docs):
    """Write one row per tree node. Parent offsets and sibling slots make the flat table reconstructable.

    The rows carry names and payloads, so the file is game data and must stay local.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for doc in docs:
            root = doc["tree"]
            if root is None:
                continue
            resource = f"0x{doc['offset']:x}"
            stack = [(root, "", "", "", "")]
            while stack:
                node, parent, slot, sibling, child_index = stack.pop()
                if node["kind"] == "element":
                    end = f"0x{node['end']:x}" if node.get("end") is not None else ""
                    writer.writerow([
                        resource, doc["root_name"], doc["status"], parent, slot, sibling, child_index,
                        node["depth"], node["kind"], node["name"], node["descriptor"],
                        f"0x{node['offset']:x}", end, node["attrs_declared"], node["children"],
                        node["content_count"], "", "", "", "", "", doc["source"]])
                    content = node["content"]
                    child_indices = []
                    next_child = 0
                    for c in content:
                        if c["kind"] == "element":
                            child_indices.append(str(next_child))
                            next_child += 1
                        else:
                            child_indices.append("")
                    for i in range(len(content) - 1, -1, -1):
                        stack.append((content[i], f"0x{node['offset']:x}", "content", str(i),
                                      child_indices[i]))
                    for i in range(len(node["attributes"]) - 1, -1, -1):
                        stack.append((node["attributes"][i], f"0x{node['offset']:x}", "attribute",
                                      str(i), ""))
                elif node["kind"] == "value":
                    if node.get("float_special") is not None:
                        value = _render(node["float_special"])
                    else:
                        value = _render(node["value"])
                    writer.writerow([
                        resource, doc["root_name"], doc["status"], parent, slot, sibling, child_index,
                        node["depth"], node["kind"], node["name"], node["descriptor"],
                        f"0x{node['offset']:x}", "", "", "", "", node["type"],
                        node.get("byte_length", ""),
                        "" if "nul_terminated" not in node else int(node["nul_terminated"]),
                        "" if "utf8_valid" not in node else int(node["utf8_valid"]), value, doc["source"]])
                else:
                    writer.writerow([
                        resource, doc["root_name"], doc["status"], parent, slot, sibling, child_index,
                        node["depth"], node["kind"], "", "", f"0x{node['offset']:x}", "",
                        "", "", "", "text", node["length"],
                        int(node["nul_terminated"]), int(node["utf8_valid"]), node["text"], doc["source"]])


def write_json(path, docs, summary):
    """Replace the report only after the full JSON has been written, keeping its permission bits."""
    directory = os.path.dirname(os.path.abspath(path))
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=directory,
                                         prefix=".rbf0_tree-", suffix=".tmp", delete=False) as f:
            temp = f.name
            json.dump({"tool": "rbf0_tree", "format": 1, "summary": summary, "resources": docs},
                      f, indent=1, ensure_ascii=False, allow_nan=False)
        os.chmod(temp, published_mode(path))   # NamedTemporaryFile creates 0600
        os.replace(temp, path)
    except BaseException:
        if temp is not None:
            with suppress(OSError):
                # The staging file may have inherited a read-only destination's mode; make it deletable
                # first, or Windows strands it beside the report when the replace fails.
                os.chmod(temp, 0o600)
                os.unlink(temp)
        raise


def same_output_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="a carve directory, a complete resource, or a single RBF0 stream")
    ap.add_argument("out", help="JSON output path")
    ap.add_argument("--csv", metavar="PATH", help="also write one flat CSV row per tree node (game data)")
    ap.add_argument("--xml", metavar="DIR", help="also write one generic XML file per resource into DIR")
    ap.add_argument("--root", action="append", metavar="NAME", help="keep only this root schema (repeatable)")
    ap.add_argument("--offset", type=lambda s: int(s, 0), help="decode only the resource at this archive offset")
    ap.add_argument("--limit", type=int, help="stop after this many selected resources")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    # The reports are written after the decode: an out or --csv naming the source used to replace it, and the
    # same file for both left only the CSV.
    if args.csv and same_output_path(args.out, args.csv):
        ap.error("out and --csv must be different files")
    if os.path.isfile(args.source):
        for flag, path in (("out", args.out), ("--csv", args.csv)):
            if path and same_output_path(path, args.source):
                ap.error(f"{flag} would overwrite the source {args.source}")
    elif os.path.isdir(args.source):
        # A carve folder: out or --csv naming one of its streams used to replace that stream, and --xml on the
        # folder itself put the XML files among the streams, where the next run read them as resources.
        try:
            streams = [full for full in (os.path.join(args.source, n) for n in os.listdir(args.source))
                       if os.path.isfile(full) and not full.lower().endswith((".json", ".csv"))]
        except OSError:
            streams = []   # decode_source reports the unreadable folder
        for flag, path in (("out", args.out), ("--csv", args.csv)):
            if path and os.path.exists(path) and any(same_output_path(path, s) for s in streams):
                ap.error(f"{flag} would overwrite the stream {path} in the source folder")
        if args.xml and same_output_path(args.xml, args.source):
            ap.error("--xml must not be the source folder: its XML files would be read as streams")

    def progress(n, total):
        if not args.quiet and (n % 100 == 0 or n == total):
            print(f"[{n}/{total}] resources", flush=True)

    path = args.source
    try:
        docs, summary = decode_source(path, roots=args.root, offset=args.offset,
                                      limit=args.limit, progress=progress)
        path = args.out
        write_json(path, docs, summary)
        if args.csv:
            path = args.csv
            write_csv(path, docs)
        paths = []
        if args.xml:
            path = args.xml
            paths = write_xml(path, docs)
    except SelectionError as error:
        ap.error(str(error))
    except RecursionError:
        print(f"{path}: tree exceeds the export depth limit", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"{path}: {error.strerror or error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"{summary['resources']} resources: {summary['ok']} ok, {summary['bad']} bad, "
              f"{summary['truncated']} truncated, {summary['trailing']} trailing")
        print(f"{summary['elements']} elements, {summary['texts']} texts, {summary['strings']} strings, "
              f"{summary['descriptors']} descriptors -> {args.out}")
        print(f"tree: {summary['parent_child_edges']} parent-child edges, {summary['leaf_elements']} leaves, "
              f"{summary['empty_elements']} empty, max depth {summary['max_depth']}, "
              f"max children {summary['max_children']}")
        if args.csv:
            print(f"node CSV -> {args.csv}")
        if paths:
            print(f"{len(paths)} XML files -> {args.xml}")
    return 0 if summary["ok"] == summary["resources"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
