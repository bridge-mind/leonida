"""Resolve 0x7a placement name hashes against locally recovered model and map names.

The map resources in ``levels.rpf`` store names as hashes. This tool builds a
hash -> name dictionary from names that are already recovered locally (model
manifests, level-model scan paths, name lists and ``common.rpf`` XML) and
resolves the hash-only fields without guessing:

* ``archetypeName`` (``0xA023A02C``) is the 32-bit joaat of the model or
  archetype name, lower-cased.
* ``CMapData.name`` (``0xE6E17F14``) and ``CMapData.parent`` (``0xA99C4651``)
  use the same 32-bit joaat.
* The per-archetype 64-bit key (entity field ``0xC8A7B90A`` at +0x20) and the
  per-map 64-bit key (``CMapData`` field ``0x8BFB6095`` at +0x10) are the low
  64 bits of MurmurHash3 x64 128 (seed 0) over the exact UTF-8 name bytes.
  They confirm a 32-bit match and reject a same-hash collision.
* Entity field ``0x1B60404D`` at +0x08 is a per-entity 64-bit id, not a name
  key. Do not resolve it as an archetype hash.

The dictionary and the resolution reports contain game-derived names. Keep
them in the local work folder; only aggregate counts belong in git, issues or
pull requests. Tests use synthetic names only.

Usage:
    python -I tools/world_names.py build --manifest <models> --scan <scan.json> \
        --names <names.txt> --rpf9-names <rpf9_names.json> --xml <common-concat.txt> --out <names.json>
    python -I tools/world_names.py resolve --dict <names.json> \
        --maps <maps.json> --archetypes <archetypes.json> --out <report.json>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tempfile

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# 0x7a member hashes (case-sensitive joaat of the engine field names).
ARCHETYPE_NAME_FIELD = 0xA023A02C
MAP_NAME_FIELD = 0xE6E17F14
MAP_PARENT_FIELD = 0xA99C4651
ENTITY_ID_FIELD = 0x1B60404D
ARCHETYPE_KEY_FIELD = 0xC8A7B90A
MAP_KEY_FIELD = 0x8BFB6095

# Derived names are not recovered strings. They are useful LOD spellings whose
# 32-bit and 64-bit hashes both matched in the measured sample.
DERIVED_SUFFIXES = ("_lod", "_slod")

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def joaat(text: str, *, case_sensitive: bool = False) -> int:
    """Jenkins one-at-a-time joaat over the engine's name bytes.

    The engine lower-cases the name for the normal 32-bit hash, and hashes it as 8-bit bytes: the default
    form is ``joaat`` of the lowercased Latin-1 bytes, the rule ``tools/hashdict.py`` (the pinned name
    dictionary) and ``tools/awc_names.py`` use, so a name hashes alike in every tool. A character outside
    Latin-1 cannot occur in a name and becomes ``?``. Encoding to UTF-8 first and lower-casing the bytes
    afterwards gave a different hash for any non-ASCII name, since bytes.lower() only folds ASCII.

    Pass ``case_sensitive=True`` for the field/struct-name variant, which keeps the exact bytes (UTF-8).
    Every field name measured so far is ASCII, where both encodings agree.
    """
    if case_sensitive:
        data = text.encode("utf-8")
    else:
        data = text.lower().encode("latin-1", "replace")
    h = 0
    for c in data:
        h = (h + c) & MASK32
        h = (h + (h << 10)) & MASK32
        h ^= h >> 6
    h = (h + (h << 3)) & MASK32
    h ^= h >> 11
    return (h + (h << 15)) & MASK32


def joaat_hex(text: str, *, case_sensitive: bool = False) -> str:
    return f"{joaat(text, case_sensitive=case_sensitive):08x}"


def _rotl64(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & MASK64


def _fmix64(value: int) -> int:
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & MASK64
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & MASK64
    value ^= value >> 33
    return value


def murmur3_x64_128(data: bytes, seed: int = 0) -> tuple[int, int]:
    """Reference MurmurHash3 x64 128-bit implementation (two 64-bit words)."""
    c1, c2 = 0x87C37B91114253D5, 0x4CF5AD432745937F
    h1 = h2 = seed & MASK64
    nblocks = len(data) // 16
    for i in range(nblocks):
        k1 = int.from_bytes(data[i * 16:i * 16 + 8], "little")
        k2 = int.from_bytes(data[i * 16 + 8:i * 16 + 16], "little")
        k1 = (k1 * c1) & MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & MASK64
        h1 ^= k1
        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & MASK64
        h1 = (h1 * 5 + 0x52DCE729) & MASK64
        k2 = (k2 * c2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & MASK64
        h2 ^= k2
        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & MASK64
        h2 = (h2 * 5 + 0x38495AB5) & MASK64

    tail = data[nblocks * 16:]
    k1 = k2 = 0
    for i, byte in enumerate(tail):
        if i < 8:
            k1 |= byte << (8 * i)
        else:
            k2 |= byte << (8 * (i - 8))
    if tail:
        if len(tail) > 8:
            k2 = (k2 * c2) & MASK64
            k2 = _rotl64(k2, 33)
            k2 = (k2 * c1) & MASK64
            h2 ^= k2
        k1 = (k1 * c1) & MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & MASK64
        h1 ^= k1

    h1 ^= len(data)
    h2 ^= len(data)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return h1, h2


def hash64(name: str) -> int:
    """Validated 64-bit key: low half of MurmurHash3 x64 128(seed 0)."""
    return murmur3_x64_128(name.encode("utf-8"))[0]


def format_u64(value: int) -> str:
    return f"{value & MASK64:016x}"


def parse_u64(value) -> int | None:
    """Parse an integer, ``0x`` hex string or 16-hex-digit string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value & MASK64
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower().startswith("0x"):
        return int(text, 16) & MASK64
    if len(text) == 16 and _HEX_RE.match(text):
        return int(text, 16)
    if len(text) <= 8 and _HEX_RE.match(text) and any(c in "abcdefABCDEF" for c in text):
        return int(text, 16)
    return int(text, 10) & MASK64


def _unique_names(table: dict[int, set[str]]) -> set[str]:
    out: set[str] = set()
    for names in table.values():
        out.update(names)
    return out


def _collisions(table: dict[int, set[str]]) -> int:
    return sum(1 for names in table.values() if len(names) > 1)


_XML_ENTITY = re.compile(r"&(?:(amp|lt|gt|quot|apos)|#([0-9]+)|#x([0-9A-Fa-f]+));")
_XML_NAMED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _xml_unescape(text: str) -> str:
    """Decode XML's five named entities and numeric character references; leave anything else as written."""
    def one(m: re.Match) -> str:
        if m[1]:
            return _XML_NAMED[m[1]]
        code = int(m[2]) if m[2] else int(m[3], 16)
        return chr(code) if 0 < code <= 0x10FFFF else m[0]
    return _XML_ENTITY.sub(one, text)


class NameDictionary:
    """Hash -> names, with exact and derived (LOD spelling) tables kept apart."""

    def __init__(self) -> None:
        self.names: dict[int, set[str]] = {}
        self.derived: dict[int, set[str]] = {}
        self.observed: collections.Counter[str] = collections.Counter()

    def add(self, name: str, *, derived: bool = False, source: str = "") -> bool:
        if not isinstance(name, str):
            return False
        name = name.strip()
        if not name or len(name) > 512:
            return False
        if any(ord(ch) < 0x20 for ch in name):
            return False
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:   # a lone surrogate (JSON \ud800 escape, XML &#xD800;) has no UTF-8 form to hash
            return False
        table = self.derived if derived else self.names
        h = joaat(name)
        table.setdefault(h, set()).add(name)
        self.observed[source or ("derived" if derived else "exact")] += 1
        return True

    def add_model_name(self, name: str, *, source: str = "manifest") -> int:
        if not isinstance(name, str) or not name.strip():
            return 0
        if ".#" in name:
            return int(self.add(name.split(".#", 1)[0], source=source))
        return int(self.add(name, source=source))

    def add_manifest(self, path: str, *, source: str = "manifest") -> int:
        with open(path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return 0
        items = data.get("Items")
        if not isinstance(items, list):
            return 0
        added = 0
        for item in items:
            if isinstance(item, dict):
                added += self.add_model_name(item.get("Name", ""), source=source)
        return added

    def add_scan(self, path: str, *, source: str = "scan") -> int:
        with open(path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return 0
        added = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rel = entry.get("Path")
            if not isinstance(rel, str) or not rel:
                continue
            base = os.path.basename(rel.replace("\\", "/"))
            stem = base.split("._", 1)[0]
            added += int(self.add(stem, source=source))
        return added

    def add_rpf9_names(self, path: str, *, source: str = "rpf9_names") -> int:
        """Names from an ``rpf9_names.py`` index (``files[].names``, spelled ``name.#dr``, ``name.#ft`` ...)."""
        with open(path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, list):
            return 0
        added = 0
        for entry in files:
            names = entry.get("names") if isinstance(entry, dict) else None
            for name in names if isinstance(names, list) else ():
                added += self.add_model_name(name, source=source)
        return added

    def add_names_file(self, path: str, *, source: str = "names") -> int:
        added = 0
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if text and not text.startswith("#"):
                    added += int(self.add(text, source=source))
        return added

    def add_xml(self, path: str, *, source: str = "xml") -> int:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read()

        # Comments are dropped, CDATA and processing instructions are kept. The input may be many files joined
        # together, so an unterminated comment ends at the next XML declaration, not at the end of the input.
        text = re.sub(r"(<!\[CDATA\[.*?\]\]>|<\?.*?\?>)|<!--.*?(?:-->|(?=<\?xml)|\Z)",
                      lambda m: m[1] or "", text, flags=re.S)

        added = 0
        for match in re.finditer(r"<filename>(.*?)</filename>", text, re.S):
            raw = _xml_unescape(match.group(1)).strip().replace("\\", "/")
            base = raw.rsplit("/", 1)[-1]
            added += int(self.add(base, source=source + "_filename"))
            if "." in base:
                added += int(self.add(base.rsplit(".", 1)[0], source=source + "_filename"))
        for match in re.finditer(r"<(?:[A-Za-z0-9_]+:)?(?:name|Name|locationName)>(.*?)</", text, re.S):
            raw = match.group(1)
            if "<" in raw:
                continue
            value = _xml_unescape(raw).strip()
            if value and len(value) <= 120:
                added += int(self.add(value, source=source + "_name"))
        return added

    def derive_lod(self, suffixes=DERIVED_SUFFIXES) -> int:
        """Add ``_lod`` / ``_slod`` spellings for exact names that lack them."""
        added = 0
        exact = _unique_names(self.names)
        for name in sorted(exact):
            for suffix in suffixes:
                candidate = name + suffix
                if candidate in exact:
                    continue
                added += int(self.add(candidate, derived=True, source="derived"))
        return added

    def match(self, value: int, key64: int | None = None, *, include_derived: bool = False) -> dict:
        """Resolve one 32-bit hash, optionally confirming it with the 64-bit key."""
        h = value & MASK32
        exact = set(self.names.get(h, ()))
        derived = set(self.derived.get(h, ())) if include_derived else set()
        derived -= exact
        candidates = sorted(exact) + sorted(derived)
        if key64 is not None:
            key = key64 & MASK64
            confirmed = [n for n in candidates if hash64(n) == key]
            rejected = [n for n in candidates if hash64(n) != key]
            if len(confirmed) == 1:
                name = confirmed[0]
                return {"status": "resolved", "name": name, "derived": name in derived,
                        "verified": True, "candidates": candidates, "rejected": rejected}
            if len(confirmed) > 1:
                return {"status": "ambiguous", "name": None, "derived": False,
                        "verified": True, "candidates": confirmed, "rejected": rejected}
            status = "unconfirmed" if candidates else "unresolved"
            return {"status": status, "name": None, "derived": False, "verified": False,
                    "candidates": candidates, "rejected": rejected}
        if len(candidates) == 1:
            name = candidates[0]
            return {"status": "resolved", "name": name, "derived": name in derived,
                    "verified": False, "candidates": candidates, "rejected": []}
        if len(candidates) > 1:
            return {"status": "ambiguous", "name": None, "derived": False,
                    "verified": False, "candidates": candidates, "rejected": []}
        return {"status": "unresolved", "name": None, "derived": False, "verified": False,
                "candidates": [], "rejected": []}

    def counts(self) -> dict:
        exact = _unique_names(self.names)
        derived = _unique_names(self.derived) - exact
        return {
            "exact_names": len(exact),
            "exact_hashes": len(self.names),
            "exact_collisions": _collisions(self.names),
            "derived_names": len(derived),
            "derived_hashes": sum(1 for h in self.derived if h not in self.names),
            "derived_collisions": _collisions(self.derived),
            "sources": dict(sorted(self.observed.items())),
        }

    def to_json(self) -> dict:
        def enc(table):
            return {f"{h:08x}": sorted(names) for h, names in sorted(table.items())}
        return {"format": 1, "counts": self.counts(), "names": enc(self.names),
                "derived": enc(self.derived)}

    @classmethod
    def from_json(cls, data: dict) -> "NameDictionary":
        out = cls()
        if not isinstance(data, dict) or data.get("format") != 1:
            raise ValueError("not a world_names dictionary (format 1)")
        for field, derived, source in (("names", False, "dictionary"),
                                       ("derived", True, "dictionary_derived")):
            table = data.get(field, {})
            if not isinstance(table, dict):
                raise ValueError(f"dictionary {field!r} must be a hash-to-names object")
            for names in table.values():
                if not isinstance(names, list):
                    raise ValueError(f"dictionary {field!r} entries must be lists of names")
                for name in names:
                    if not out.add(name, derived=derived, source=source):
                        raise ValueError(f"dictionary {field!r} has an invalid name")
        return out


def build_dictionary(manifests=(), scans=(), names_files=(), xml_files=(), *,
                     derive_lod: bool = False, rpf9_names=()) -> NameDictionary:
    def scan_failed(error):
        raise error

    dictionary = NameDictionary()
    for path in manifests:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, onerror=scan_failed):
                dirs.sort()
                if "manifest.json" in files and os.path.basename(root).lower() == "models":
                    dictionary.add_manifest(os.path.join(root, "manifest.json"))
        elif os.path.isfile(path):
            dictionary.add_manifest(path)
        else:
            raise FileNotFoundError(path)
    for path in scans:
        dictionary.add_scan(path)
    for path in names_files:
        dictionary.add_names_file(path)
    for path in rpf9_names:
        dictionary.add_rpf9_names(path)
    for path in xml_files:
        dictionary.add_xml(path)
    if derive_lod:
        dictionary.derive_lod()
    return dictionary


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("maps", "resources", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"{path}: expected a map row list or a maps, resources or rows table")


def resolve_maps(rows, dictionary: NameDictionary, *, include_derived: bool = False) -> dict:
    names = set()
    parents = collections.Counter()
    stats = {
        "rows": 0, "name_present": 0, "name_resolved": 0, "name_verified": 0,
        "name_derived": 0, "name_ambiguous": 0, "name_unconfirmed": 0, "name_unresolved": 0,
        "parent_present": 0, "parent_resolved": 0, "parent_ambiguous": 0,
        "parent_unresolved": 0, "parent_links": 0, "parent_links_resolved": 0,
    }
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in (None, "map"):
            continue
        stats["rows"] += 1
        name = parse_u64(row.get("map_name", row.get("name")))
        parent = parse_u64(row.get("parent"))
        key = parse_u64(row.get("map_name64", row.get("name64")))
        if name is not None:
            names.add(name)
            stats["name_present"] += 1
            match = dictionary.match(name, key, include_derived=include_derived)
            status = match["status"]
            if status == "resolved":
                stats["name_resolved"] += 1
                stats["name_verified"] += int(match["verified"])
                stats["name_derived"] += int(match["derived"])
            elif status == "ambiguous":
                stats["name_ambiguous"] += 1
            elif status == "unconfirmed":
                stats["name_unconfirmed"] += 1
            else:
                stats["name_unresolved"] += 1
        if parent:
            parents[parent] += 1
    for parent, count in parents.items():
        stats["parent_present"] += count
        status = dictionary.match(parent, include_derived=include_derived)["status"]
        if status == "resolved":
            stats["parent_resolved"] += count
        elif status == "ambiguous":
            stats["parent_ambiguous"] += count
        else:
            stats["parent_unresolved"] += count
        if parent in names:
            stats["parent_links"] += count
            if status == "resolved":
                stats["parent_links_resolved"] += count
    return stats


def resolve_archetypes(data, dictionary: NameDictionary, *, include_derived: bool = False) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected an archetype report object")
    raw = data.get("archetypes")
    if not isinstance(raw, dict):
        raw = data.get("hashes")
    if not isinstance(raw, dict):
        raise ValueError("expected an archetypes or hashes table")
    keys = data.get("keys") if isinstance(data.get("keys"), dict) else data.get("key64")
    if not isinstance(keys, dict):
        keys = {}
    key_by_hash = {}
    for raw_hash, key in keys.items():
        try:
            key_by_hash[int(str(raw_hash), 16) & MASK32] = parse_u64(key)
        except (TypeError, ValueError):
            continue
    stats = {
        "entities": 0, "unique_hashes": 0,
        "resolved_exact_hashes": 0, "resolved_exact_entities": 0,
        "resolved_derived_hashes": 0, "resolved_derived_entities": 0,
        "verified_hashes": 0, "verified_entities": 0,
        "unverified_hashes": 0, "unverified_entities": 0,
        "unconfirmed_hashes": 0, "unconfirmed_entities": 0,
        "ambiguous_hashes": 0, "ambiguous_entities": 0,
        "unresolved_hashes": 0, "unresolved_entities": 0,
    }
    for raw_hash, raw_count in raw.items():
        try:
            h = int(str(raw_hash), 16) & MASK32
            count = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            continue
        if count <= 0:
            continue
        key = key_by_hash.get(h)
        stats["entities"] += count
        stats["unique_hashes"] += 1
        match = dictionary.match(h, key, include_derived=include_derived)
        status = match["status"]
        if status == "resolved":
            if match["derived"]:
                stats["resolved_derived_hashes"] += 1
                stats["resolved_derived_entities"] += count
            else:
                stats["resolved_exact_hashes"] += 1
                stats["resolved_exact_entities"] += count
            if match["verified"]:
                stats["verified_hashes"] += 1
                stats["verified_entities"] += count
            else:
                stats["unverified_hashes"] += 1
                stats["unverified_entities"] += count
        elif status == "unconfirmed":
            stats["unconfirmed_hashes"] += 1
            stats["unconfirmed_entities"] += count
        elif status == "ambiguous":
            stats["ambiguous_hashes"] += 1
            stats["ambiguous_entities"] += count
        else:
            stats["unresolved_hashes"] += 1
            stats["unresolved_entities"] += count
    return stats


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def _build_input_paths(args):
    for path in (*args.manifest, *args.scan, *args.names, *args.rpf9_names, *args.xml):
        yield path
    if os.path.exists(args.out):
        for path in args.manifest:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    dirs.sort()
                    if "manifest.json" in files and os.path.basename(root).lower() == "models":
                        yield os.path.join(root, "manifest.json")


def _build(args) -> int:
    if not (args.manifest or args.scan or args.names or args.xml or args.rpf9_names):
        print("build: at least one --manifest/--scan/--names/--rpf9-names/--xml source is required",
              file=sys.stderr)
        return 2
    for path in _build_input_paths(args):
        if same_report_path(args.out, path):
            raise ValueError(
                f"build: --out {args.out} would overwrite input {path}")
    dictionary = build_dictionary(args.manifest, args.scan, args.names, args.xml,
                                  derive_lod=args.derive_lod, rpf9_names=args.rpf9_names)
    counts = dictionary.counts()
    if counts["exact_names"] == 0:
        print("build: no names recovered from the given sources", file=sys.stderr)
        return 2
    _write_json(args.out, dictionary.to_json())
    print(json.dumps(counts, indent=1, sort_keys=True))
    return 0


def _resolve(args) -> int:
    if args.out:
        for path in (args.dict, *args.maps, *args.archetypes):
            if same_report_path(args.out, path):
                raise ValueError(
                    f"resolve: --out {args.out} would overwrite input {path}")
    try:
        dictionary = NameDictionary.from_json(_load_json(args.dict))
    except ValueError as exc:
        raise ValueError(f"{args.dict}: {exc}") from None
    report = {"dictionary": dictionary.counts(), "maps": {}, "archetypes": {}}
    if args.maps:
        rows = (row for path in args.maps for row in load_rows(path))
        report["maps"] = resolve_maps(rows, dictionary, include_derived=args.include_derived)
    archetypes = collections.Counter()
    for path in args.archetypes:
        try:
            data = _load_json(path)
            stats = resolve_archetypes(data, dictionary, include_derived=args.include_derived)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
        for key, value in stats.items():
            archetypes[key] += value
    if args.archetypes:
        report["archetypes"] = dict(archetypes)
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def _write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".world-names-") as staging:
        tmp = os.path.join(staging, "report.json")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=1, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a hash -> name dictionary")
    build.add_argument("--manifest", action="append", default=[],
                       help="rpf9dump models/manifest.json or a directory tree of them")
    build.add_argument("--scan", action="append", default=[],
                       help="rpf9dump levels scan JSON with entries[].Path")
    build.add_argument("--names", action="append", default=[],
                       help="text file with one recovered name per line")
    build.add_argument("--rpf9-names", action="append", default=[],
                       help="rpf9_names.py index JSON (files[].names)")
    build.add_argument("--xml", action="append", default=[],
                       help="common.rpf XML or a concatenated XML text")
    build.add_argument("--derive-lod", action="store_true",
                       help="add derived _lod/_slod spellings (verified separately)")
    build.add_argument("--out", required=True, help="dictionary JSON output")
    build.set_defaults(func=_build)

    resolve = sub.add_parser("resolve", help="resolve map and archetype hashes")
    resolve.add_argument("--dict", required=True, help="dictionary JSON from build")
    resolve.add_argument("--maps", action="append", default=[],
                         help="world_index/world_census JSON or a list of map rows")
    resolve.add_argument("--archetypes", action="append", default=[],
                         help="JSON with archetypes{hash: count} and keys{hash: u64}")
    resolve.add_argument("--include-derived", action="store_true",
                         help="count derived _lod/_slod names separately")
    resolve.add_argument("--out", help="report JSON output")
    resolve.set_defaults(func=_resolve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
