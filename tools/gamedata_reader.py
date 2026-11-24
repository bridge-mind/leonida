"""Stream the game-data XML of a carved RPF9 archive into Python values, dispatching on root tag and ``type``.

The XML in ``common.rpf`` is RAGE parser metadata (docs/gamedata.md): 118 root tags and 5,293 element tags, written
with a handful of conventions. This reader turns any of it into plain values without a schema per tag, and lets a
caller plug in a schema where one is known:

- ``<x value="1.5"/>``: a scalar (bool for ``true``/``false``, int for decimal or ``0x`` hex, float, else the text);
- ``<v x=".." y=".." [z=".." [w=".."]]/>``: a vector, as a tuple of floats;
- ``<a content="short_array|float_array|float16_array|vector2_array|vec3v_array|matrix33|matrix34">..</a>``: the
  whitespace-separated numbers of the body (pairs, triples or matrix rows where the content says so);
- ``<Item type="CSomeClass">..</Item>``: a polymorphic object, ``{"$type": "CSomeClass", ...}``; ``type="NULL"`` is
  ``None``;
- ``<x ref="name"/>``: ``{"$ref": "name"}``;
- children that are all ``<Item>``: a list; children that all carry ``key``: a dict by key; repeated tags: a list;
- an element with only text: the text.

Dispatch: ``register_root(tag)`` registers a handler for a whole document's root element, ``register_type(name)``
one for every element with ``type="name"`` (both called as ``handler(element, reader)``; ``reader.convert`` gives the
generic value of a sub-element). Everything else takes the generic conversion. The documents are read with
``iterparse`` and each top-level field is converted and dropped as soon as it ends, so memory is bounded by the largest
field, not the document.

    python -I tools/gamedata_reader.py <carve_dir or .xml> [...] [--root TAG] [--out DIR]

prints counts only: documents read, roots, the elements each convention covered, ``type`` classes seen, and
documents that are not well-formed XML (tail-only or truncated carves), with the reason. ``--out DIR`` writes one JSON
file per document: those files hold the game's data and stay on the local machine (CONTRIBUTING.md). Standard
library only.
Usage: `python -I tools/gamedata_reader.py --help` lists the options.
"""
import argparse
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOM = b"\xef\xbb\xbf"
VECTOR_KEYS = ("x", "y", "z", "w")
BASE_CONVENTION_KEYS = frozenset(("type", "content", "value", "ref", "key"))
CONVENTION_KEYS = BASE_CONVENTION_KEYS | frozenset(VECTOR_KEYS)
GROUPS = {"vector2_array": 2, "vec3v_array": 3, "matrix33": 3, "matrix34": 3}
INT = re.compile(r"[-+]?\d+")
HEX = re.compile(r"0x[0-9A-Fa-f]+")
FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?f?|[-+]?(inf|nan)", re.I)

ROOT_HANDLERS = {}
TYPE_HANDLERS = {}


def _is_vector(attrib):
    """True when ``attrib`` carries ``x`` and every present ``x``/``y``/``z``/``w`` attribute is a float."""
    return "x" in attrib and all(FLOAT.fullmatch(attrib[k]) for k in VECTOR_KEYS if k in attrib)


def _uses_vector_keys(attrib):
    """True when ``attrib`` resolves via the vector convention rather than another attribute convention."""
    return (attrib.get("type") != "NULL"
            and "content" not in attrib
            and "value" not in attrib
            and _is_vector(attrib))


def register_root(tag):
    """Decorator: ``handler(element, reader)`` for every document whose root element is ``tag``."""
    def wrap(fn):
        ROOT_HANDLERS[tag] = fn
        return fn
    return wrap


def register_type(name):
    """Decorator: ``handler(element, reader)`` for every element with ``type="name"``."""
    def wrap(fn):
        TYPE_HANDLERS[name] = fn
        return fn
    return wrap


def parse_scalar(text):
    """A ``value`` attribute or array token: bool, int (decimal or 0x hex), float, else the text unchanged."""
    if text in ("true", "false"):
        return text == "true"
    if HEX.fullmatch(text):
        return int(text, 16)
    if INT.fullmatch(text):
        return int(text)
    if FLOAT.fullmatch(text):
        return _float(text)
    return text


def _float(text):
    """A float with an optional C-style ``f`` suffix ("1.5f"); "inf" keeps its f."""
    try:
        return float(text)
    except ValueError:
        return float(text[:-1])


def parse_content(kind, text):
    """The numbers of a ``content=`` body, grouped into tuples for vector arrays and matrices."""
    nums = [parse_scalar(t) for t in (text or "").split()]
    width = GROUPS.get(kind)
    if not width:
        return nums
    if len(nums) % width:
        raise ValueError(f"{kind}: {len(nums)} numbers is not a multiple of {width}")
    return [tuple(nums[i:i + width]) for i in range(0, len(nums), width)]


class Reader:
    """Converts elements, counting which convention each one took (``stats``) and the ``type`` classes seen."""

    def __init__(self, root_handlers=None, type_handlers=None):
        self.root_handlers = ROOT_HANDLERS if root_handlers is None else root_handlers
        self.type_handlers = TYPE_HANDLERS if type_handlers is None else type_handlers
        self.stats = Counter()
        self.types = Counter()
        self._active_type_elements = set()

    def convert(self, el, use_type_handler=True):
        """The generic value of element ``el`` (a registered ``type`` handler first, unless told not to). Attributes
        no convention uses (``name``, ``userFlags``, ...) are kept as ``"@name"`` entries."""
        in_handler = id(el) in self._active_type_elements
        value = self._convert(el, use_type_handler)
        return self.with_attributes(el.attrib, value, count_stats=not in_handler)

    def with_attributes(self, attrib, value, count_stats=True):
        consumed = CONVENTION_KEYS if _uses_vector_keys(attrib) else BASE_CONVENTION_KEYS
        extra = {f"@{k}": parse_scalar(v) for k, v in attrib.items() if k not in consumed}
        if not extra:
            return value
        if count_stats:
            self.stats["extra attributes"] += 1
        if isinstance(value, dict):
            return {**value, **extra}
        return {"$value": value, **extra} if value not in ("", None) else extra

    def _convert(self, el, use_type_handler=True):
        a = el.attrib
        t = a.get("type")
        if t == "NULL":
            self.stats["null"] += 1
            return None
        if t is not None:
            if id(el) not in self._active_type_elements:
                self.types[t] += 1
            handler = self.type_handlers.get(t) if use_type_handler else None
            if handler:
                self.stats["type handler"] += 1
                self._active_type_elements.add(id(el))
                try:
                    return handler(el, self)
                finally:
                    self._active_type_elements.discard(id(el))
        if "content" in a:
            self.stats["array"] += 1
            return parse_content(a["content"], el.text)
        if "value" in a:
            self.stats["scalar"] += 1
            return parse_scalar(a["value"])
        if _is_vector(a):
            self.stats["vector"] += 1
            return tuple(_float(a[k]) for k in VECTOR_KEYS if k in a)
        if "ref" in a:
            self.stats["reference"] += 1
            return {"$ref": a["ref"]}
        kids = list(el)
        if not kids:
            self.stats["text"] += 1
            return (el.text or "").strip()
        return self._typed_object(t, self.children(kids))

    def _typed_object(self, t, value):
        """Wrap an assembled child container in a polymorphic ``$type`` dict when ``t`` is set."""
        if t is None:
            return value
        self.stats["object"] += 1
        if not isinstance(value, dict):
            value = {"$items": value}
        return {"$type": t, **value}

    def _streams_children(self, root):
        """True when top-level children can be converted and dropped piecemeal instead of keeping the whole root."""
        if root.tag in self.root_handlers:
            return False
        a = root.attrib
        t = a.get("type")
        if t == "NULL" or (t is not None and t in self.type_handlers):
            return False
        return not ("content" in a or "value" in a or _is_vector(a) or "ref" in a)

    def children(self, kids):
        """Child elements as a list (all ``Item``), a dict by ``key`` (all keyed) or a dict by tag."""
        return self.assemble([(k.tag, k.attrib.get("key"), self.convert(k)) for k in kids])

    def assemble(self, parts):
        """``(tag, key or None, value)`` of each child, in order -> list, dict by key or dict by tag (a tag that occurs
        more than once becomes the list of its values)."""
        if all(tag == "Item" for tag, _, _ in parts):
            if all(key is not None for _, key, _ in parts):
                self.stats["map"] += 1
                return {key: value for _, key, value in parts}
            self.stats["list"] += 1
            return [value for _, _, value in parts]
        self.stats["struct"] += 1
        out, repeated = {}, set()
        for tag, _, value in parts:
            if tag in out:
                if tag not in repeated:
                    out[tag], repeated = [out[tag]], repeated | {tag}
                out[tag].append(value)
            else:
                out[tag] = value
        return out

    def read(self, data):
        """One document (bytes) -> ``(root tag, root type, value)``. Fields are converted as their end tag arrives
        and then dropped from the tree. Raises ``ET.ParseError`` for a document that is not well-formed."""
        stream = io.BytesIO(data[len(BOM):] if data.startswith(BOM) else data)
        depth, root, parts, stream_kids = 0, None, [], False
        for event, el in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                depth += 1
                if root is None:
                    root = el
                    stream_kids = self._streams_children(root)
                continue
            depth -= 1
            if depth == 1 and stream_kids:
                # a top-level field is complete: convert it now and drop its sub-tree and root reference
                parts.append((el.tag, el.attrib.get("key"), self.convert(el)))
                el.clear()
                del root[:]
        handler = self.root_handlers.get(root.tag)
        if handler:
            self.stats["root handler"] += 1
            return root.tag, root.attrib.get("type"), handler(root, self)
        if not parts:
            return root.tag, root.attrib.get("type"), self.convert(root)
        attrib = dict(root.attrib)
        value = self.assemble(parts)
        t = attrib.get("type")
        if t is not None:
            self.types[t] += 1
            value = self._typed_object(t, value)
        return root.tag, t, self.with_attributes(attrib, value)


def documents(targets):
    """XML payloads of carve directories (index.json) or .xml files, as gamedata_xml.load_documents yields them."""
    import gamedata_xml as gx
    return gx.load_documents(targets)


def run(targets, root_filter=None, out_dir=None, reader=None):
    """Read every document; returns a summary of counts (no names of entries, no values)."""
    reader = reader or Reader()
    roots, failed, read = Counter(), Counter(), 0
    written_names = set()
    for n, doc in enumerate(documents(targets)):
        hint = doc.get("root_hint")
        if root_filter and hint != root_filter:
            continue
        try:
            tag, _, value = reader.read(doc["data"])
        except (ET.ParseError, LookupError, ValueError) as e:
            # Unsupported encodings and malformed content arrays fail outside ET.ParseError too.
            failed[str(e).split(":")[0]] += 1
            continue
        read += 1
        roots[tag] += 1
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            name = doc.get("offset_hex") or doc.get("name") or str(n)
            stem = f"{tag}_{name}"
            filename = f"{stem}.json"
            suffix = 2
            # Another input can have the same basename or carve offset. Keep existing names on ordinary reruns.
            while filename.casefold() in written_names:
                filename = f"{stem}_{suffix}.json"
                suffix += 1
            written_names.add(filename.casefold())
            with open(os.path.join(out_dir, filename), "w", encoding="utf-8", newline="\n") as f:
                json.dump(value, f, indent=1, default=str)
    return {"documents": read, "roots": len(roots), "per_root_documents": dict(roots),
            "not_well_formed": sum(failed.values()), "parse_errors": dict(failed),
            "elements_by_convention": dict(reader.stats), "type_classes": len(reader.types),
            "typed_elements": sum(reader.types.values()),
            "registered": {"roots": len(reader.root_handlers), "types": len(reader.type_handlers)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+", help="rpf9_carve.py directories (with index.json) or .xml files")
    ap.add_argument("--root", help="only documents with this root tag")
    ap.add_argument("--out", help="write one JSON file per document here (game data: keep it local)")
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out) and not os.path.isdir(a.out):
        print(f"ERROR: --out {a.out} exists and is not a directory", file=sys.stderr)
        return 2
    try:
        s = run(a.targets, a.root, a.out)
    except FileNotFoundError as e:
        print(f"ERROR: {e} does not exist", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        # an unreadable file, or a carve folder whose index.json is not JSON or not the rpf9_carve.py shape
        where = getattr(e, "filename", None) or ", ".join(a.targets)
        print(f"ERROR: cannot read {where}: {getattr(e, 'strerror', None) or e!r}", file=sys.stderr)
        return 2
    print(f"{s['documents']} documents read, {s['roots']} roots, {s['not_well_formed']} not well-formed "
          f"{s['parse_errors'] or ''}".rstrip())
    print("elements by convention:", ", ".join(f"{k} {v}" for k, v in sorted(s["elements_by_convention"].items())))
    print(f"type classes: {s['type_classes']} ({s['typed_elements']} typed elements); registered handlers: "
          f"{s['registered']['roots']} roots, {s['registered']['types']} types")
    if a.out:
        print(f"wrote {s['documents']} JSON files to {a.out} (game data: keep them local)")
    if not s["documents"] and s["not_well_formed"]:
        # Nothing parsed. A corpus with some truncated tails still reads the rest and stays 0; a run that read
        # nothing at all cannot tell "no documents here" from "every document was broken", so the exit code says it.
        print(f"ERROR: no document could be read; {s['not_well_formed']} not well-formed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
