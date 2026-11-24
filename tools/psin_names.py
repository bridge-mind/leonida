"""Resolve the name-hash fields of PSIN instance data against a local list of resource names.

`psin_data.py` walks a PSIN container from its root through the `LSCH` schema. Members of type 0x4a, 0x4d and 0x4b
hold a u32 hash and members of type 0x4f a u64 hash. Many of them are the names of other resources (drawables,
fragments, dictionaries, maps). This tool collects every such value reached from the root and looks it up in a
dictionary built from names you already have:

    u32 hash   joaat of the lower-cased name (the rule docs/world_names.md gives for `archetypeName`)
    u64 hash   low 64 bits of MurmurHash3 x64 128, seed 0, over the name's UTF-8 bytes (as listed and lower-cased)

Every run also hashes a salted control: each name with CONTROL_SALT appended, through the same two rules. Control
names are not real names, so their matches show how many of the real matches chance alone would give. The summary
reports both, per field and in total.

Names come from `rpf9_names.py` JSON (the `.#dr`, `.#ft`, `.#dd` suffix is dropped) or from text files with one name
per line. They are game-derived: keep the name lists, `--refs` output and `--json` field tables in your local work
folder. Only aggregate counts belong in git, issues or pull requests.

Usage:
    python -I psin_names.py <PSIN file or carve directory> --names <rpf9_names.json|names.txt> [--names ...]
                            [--json] [--fields N] [--refs OUT.tsv] [--limit N]

`--refs` writes one row per resolved value: file, struct hash, field hash, member type, value and name.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psin_data  # noqa: E402
import psin_index  # noqa: E402
from report_paths import same_report_path  # noqa: E402
from world_names import joaat, murmur3_x64_128  # noqa: E402

CONTROL_SALT = "#zq"
HASH32 = (0x4A, 0x4D, 0x4B)
HASH64 = (0x4F,)
SUFFIX = re.compile(r"\.#[a-z0-9]{2}$")


class HashWalker(psin_data.Walker):
    """A psin_data walker that also records every hash value with the struct and member it belongs to.

    Array elements and map keys and values carry the hash of the member that holds the array or map.
    """

    def __init__(self, container, stats):
        super().__init__(container, stats)
        self.found = []
        self._structs = []
        self._fields = []

    def struct(self, name, at, depth, number=None):
        self._structs.append(name)
        try:
            return super().struct(name, at, depth, number)
        finally:
            self._structs.pop()

    def value(self, m, members, at, depth):
        named = m["hash"] != psin_data.ELEMENT
        if named:
            self._fields.append(m["hash"])
        try:
            t = m["type"]
            if t in psin_data.HASHES and self._structs and self._fields:
                v = struct.unpack_from(psin_data.HASHES[t], self.c.blob, at)[0]
                self.found.append((self._structs[-1], self._fields[-1], t, v))
            return super().value(m, members, at, depth)
        finally:
            if named:
                self._fields.pop()


def hash_values(blob):
    """[(struct hash, field hash, member type, value)] for one PSIN container. Raises ValueError if it is unreadable."""
    container = psin_data.Container(blob)
    walker = HashWalker(container, collections.Counter())
    try:
        walker.run()
    except (struct.error, IndexError) as e:
        raise ValueError(f"walk error: {e}") from None
    return walker.found


def load_names(paths):
    """The set of names in rpf9_names.py JSON files and one-name-per-line text files. A .json file that is not
    JSON, or not the shape rpf9_names.py writes, raises ValueError naming the file (--names is repeatable)."""
    names = set()
    for path in paths:
        if path.endswith(".json"):
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except (ValueError, RecursionError) as e:   # ValueError: not JSON, or not UTF-8
                raise ValueError(f"{path}: not JSON: {e}") from None
            rows = doc.get("files", []) if isinstance(doc, dict) else None
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected rpf9_names.py JSON, an object with a 'files' array")
            for row in rows:
                listed = row.get("names", []) if isinstance(row, dict) else None
                if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
                    raise ValueError(f"{path}: each 'files' row must be an object whose 'names' is an array of "
                                     "strings")
                names.update(SUFFIX.sub("", n) for n in listed)
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                names.update(line.strip() for line in f)
    names.discard("")
    return names


def dictionaries(names, salt=""):
    """({u32 hash: name}, {u64 hash: name}) for the names with `salt` appended."""
    d32, d64 = {}, {}
    for n in sorted(names):
        s = n + salt
        d32.setdefault(joaat(s), n)
        for form in {s, s.lower()}:
            d64.setdefault(murmur3_x64_128(form.encode("utf-8"))[0], n)
    return d32, d64


def scan(src, names, limit=None, refs=None):
    """Resolve every hash value in the PSIN containers under `src`. Returns the summary and the per-field table."""
    real = dictionaries(names)
    control = dictionaries(names, CONTROL_SALT)
    total = collections.Counter(names=len(names))
    fields = collections.defaultdict(collections.Counter)
    distinct = collections.defaultdict(set)
    seen = 0
    for file_name, blob in psin_index.files_in(src):
        if blob[:4] != psin_index.MAGIC:
            continue
        total["PSIN files"] += 1
        seen += 1
        try:
            found = hash_values(blob)
        except ValueError:
            total["skipped"] += 1
            found = None
        if found is not None:
            total["containers"] += 1
            hit = False
            for s, field, t, v in found:
                size = "u64" if t in HASH64 else "u32"
                which = 1 if size == "u64" else 0
                row = fields[(s, field, t)]
                row["values"] += 1
                total[f"{size} values"] += 1
                if v == 0:
                    row["zero"] += 1
                    continue
                distinct[(s, field, t)].add(v)
                name = real[which].get(v)
                if name is not None:
                    hit = True
                    row["resolved"] += 1
                    total[f"{size} resolved"] += 1
                    if refs:
                        refs.write(f"{file_name}\t{s:08x}\t{field:08x}\t{t:02x}\t{v:x}\t{name}\n")
                if v in control[which]:
                    row["control"] += 1
                    total[f"{size} control"] += 1
            total["containers with a resolved name"] += hit
        if limit and seen >= limit:
            break
    table = []
    for key, row in fields.items():
        s, field, t = key
        table.append({"struct": f"{s:08x}", "field": f"{field:08x}", "type": f"{t:02x}",
                      "values": row["values"], "zero": row["zero"], "distinct": len(distinct[key]),
                      "resolved": row["resolved"], "control": row["control"]})
    table.sort(key=lambda r: (-r["resolved"], r["struct"], r["field"]))
    total["fields"] = len(table)
    total["fields with a resolved value"] = sum(1 for r in table if r["resolved"])
    return dict(sorted(total.items())), table


def fields_count(text):
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


def main():
    ap = argparse.ArgumentParser(description="Resolve PSIN name-hash fields against a local name list.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--names", action="append", required=True,
                    help="rpf9_names.py JSON or a text file with one name per line (repeatable)")
    ap.add_argument("--json", action="store_true", help="print the summary and the whole field table as JSON")
    ap.add_argument("--fields", type=fields_count, default=10,
                    help="fields to list in the text output (0 for none)")
    ap.add_argument("--refs", help="write every resolved value as TSV (local only: it holds game names)")
    ap.add_argument("--limit", type=psin_index.count, help="stop after this many PSIN files")
    a = ap.parse_args()
    inputs = [a.src, *a.names]
    for path in inputs:
        if not os.path.exists(path):
            print(f"ERROR: {path}: no such file or folder", file=sys.stderr)
            return 2
    try:
        if a.refs:
            if os.path.isdir(a.refs):
                ap.error("--refs must be a file path, not a directory")
            if any(same_report_path(path, a.refs) for path in inputs):
                ap.error("--refs must not overwrite an input")
            if os.path.isdir(a.src):
                src_real = os.path.realpath(a.src)
                refs_real = os.path.realpath(a.refs)
                try:
                    inside_src = os.path.commonpath([src_real, refs_real]) == src_real
                except ValueError:
                    inside_src = False
                if inside_src:
                    ap.error("--refs must not overwrite an input")
                carve_inputs = [os.path.join(a.src, "index.json"), *psin_index.input_paths(a.src)]
                if any(same_report_path(path, a.refs) for path in carve_inputs):
                    ap.error("--refs must not overwrite an input")
        names = load_names(a.names)
        if a.refs:
            with open(a.refs, "w", encoding="utf-8", newline="\n") as refs:
                refs.write("file\tstruct\tfield\ttype\tvalue\tname\n")
                summary, table = scan(a.src, names, a.limit, refs)
        else:
            summary, table = scan(a.src, names, a.limit)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if a.json:
        json.dump({"summary": summary, "fields": table}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for k, v in summary.items():
        print(f"{k:36} {v:,}")
    if table and a.fields:
        print(f"\n{'struct':8} {'field':8} {'type':4} {'values':>9} {'distinct':>9} {'resolved':>9} {'control':>8}")
        for r in table[:a.fields]:
            print(f"{r['struct']} {r['field']} {r['type']:4} {r['values']:9,} {r['distinct']:9,} "
                  f"{r['resolved']:9,} {r['control']:8,}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
