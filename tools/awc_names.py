"""Name AWC streams by matching dictionary words against their 29-bit joaat stream ids.

An AWC stream table stores ``joaat(lowercase name) & 0x1FFFFFFF`` per stream (docs/audio.md). With over a million
distinct ids in a 2^29 space, about 0.3% of arbitrary words collide by chance, so a hit alone is not trusted. A
name is accepted under one of two rules:

``dialogue``  an exact dictionary word shaped like a dialogue wave, ``<root>au_<4 letters>_<NN>``.
``family``    a name without its trailing ``_<number>`` (the base) that hits ids in the same bank with several
              numbered suffixes (``''``, ``_00`` .. ``_40``, ``_1`` .. ``_9``): at least 3 members, or 2 with adjacent
              numbers. The base must be made of common tokens (each in at least ``--min-token`` dictionary words), and
              at least one member must be a dictionary word itself.

Every run repeats the attack with a salt prepended to each word (``zq``): those names cannot be right, so their
accepted count estimates the false names. Per source, the dialogue rule is kept only if its control is under 2% of
its real count, and the family rule uses the smallest family size whose control is under 2%.

Dictionary sources:

``--words``         a text file with one word per line, or a ``hashdict.py`` JSON (its values are used);
``--conversations`` files or directories whose conversation data pairs a dialogue key with line labels
                    (``<ROOT>AU_<XXXX>\\0<LABEL>\\0``). A key with N labels yields the waves ``<key>_01`` ..
                    ``<key>_<N>``: wave ``_01`` speaks the first label, ``_02`` the second, and so on.
                    In a carve folder a ``.bin`` stream of a whole number of 64 KiB blocks is read together with the
                    streams that continue it (docs/page_directory.md), so a pair across that boundary still counts.

Bank sources (positional; each one is reported and thresholded on its own): ``.awc`` files, directories searched
for ``.awc`` files, or an ``audio_manifest.json`` written by ``rpf9dump audio``. Sources with the same basename
include their parent paths in report labels so they can be selected separately with ``opus_tags.py --source``.

Usage:
    python -I awc_names.py <banks> [<banks> ...] (--words FILE | --conversations PATH) [...]
                           [--min-token N] [--out names.csv|.json]

A JSON inventory from ``awc_manifest.py`` can also supply the bank names and stream IDs.

Standard library only. The output names game files: it is a local work product and must not be committed or
attached to an issue. The tests use synthetic banks and words.
Usage: `python -I tools/awc_names.py --help` lists the options.
"""
import argparse
import collections
import csv
import json
import os
import re
import struct
import sys

M32 = 0xFFFFFFFF
M29 = 0x1FFFFFFF
DIALOGUE = re.compile(r"^[a-z0-9_]+au_[a-z]{4}_\d\d$")
TRAILING_NUMBER = re.compile(r"_\d+$")
TOKEN_SPLIT = re.compile(r"[_\-]")
FAMILY_NUMBER = re.compile(r"_(\d+)$")
SUFFIXES = [""] + [f"_{i:02d}" for i in range(41)] + [f"_{i}" for i in range(1, 10)]
CONVERSATION_PAIR = re.compile(rb"([A-Z0-9_]{2,40}AU_[A-Z]{4})\x00([A-Za-z0-9_]{3,80})\x00")
SALT = "zq"
CONTROL_LIMIT = 0.02
FAMILY_SIZES = range(2, 12)
MIN_TOKEN = 5
MAX_SCAN_BYTES = 256 << 20
# A carved stream that inflates to a whole number of 64 KiB blocks continues in the next one (docs/page_directory.md);
# these heads start a stream of their own instead.
CONTINUATION_BLOCK = 65536
NEW_STREAM_HEADS = (b"PSIN", b"RBF0", b"<?xml", b"\xef\xbb\xbf<?xml")


# ---- joaat ----
def joaat_state(data: bytes, h: int = 0) -> int:
    """The one-at-a-time state after ``data``, before the final avalanche (so suffixes can continue it)."""
    for c in data:
        h = (h + c) & M32
        h = (h + (h << 10)) & M32
        h ^= h >> 6
    return h


def joaat_finish(h: int) -> int:
    h = (h + (h << 3)) & M32
    h ^= h >> 11
    return (h + (h << 15)) & M32


def joaat(s: str) -> int:
    return joaat_finish(joaat_state(s.lower().encode("latin-1", "replace")))


def _enc(s: str) -> bytes:
    return s.encode("latin-1", "replace")


# ---- banks ----
def bank_ids(path: str) -> list:
    """Stream ids from an AWC stream table, reading only the header and the table."""
    with open(path, "rb") as fh:
        head = fh.read(16)
        if len(head) < 16 or head[:4] != b"ADAT":
            raise ValueError(f"{path}: not a little-endian AWC")
        flags, n, header_size = struct.unpack_from("<III", head, 4)
        skip = (4 if flags & 0x400000 else 0) + (n * 2 if flags & 0x10000 else 0) \
            + ((n + 1) * 4 if flags & 0x200000 else 0)
        table_end = 16 + skip + 4 * n
        file_size = os.fstat(fh.fileno()).st_size
        if table_end > file_size:
            raise ValueError(f"{path}: stream table extends past the end of the file")
        if header_size > file_size:
            raise ValueError(f"{path}: declared header extends past the end of the file")
        if header_size < 16 or table_end > header_size:
            raise ValueError(f"{path}: stream table extends past the declared header")
        fh.seek(16 + skip)
        table = fh.read(4 * n)
    if len(table) < 4 * n:
        raise ValueError(f"{path}: stream table extends past the end of the file")
    return [v & M29 for v in struct.unpack(f"<{n}I", table)]


def load_banks(source: str) -> dict:
    """{bank label: [stream ids]} from AWC banks, a Python inventory, or an rpf9dump manifest."""
    if os.path.isfile(source) and source.lower().endswith(".json"):
        with open(source, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict) and "bank_count" in doc and "streams" in doc:
            grouped = collections.defaultdict(list)
            for row in doc["streams"]:
                bank = row["bank"]
                # Match the export folder used by awc_opus, including nested bank paths.
                label = os.path.splitext(bank)[0] if bank.lower().endswith(".awc") else bank
                grouped[label].append(int(row["stream"], 16) & M29)
            return dict(grouped)
        banks = doc.get("Banks", doc) if isinstance(doc, dict) else doc
        if not isinstance(banks, list):
            raise ValueError("expected a list of banks")
        out = {}
        for bank in banks:
            if not isinstance(bank, dict):
                raise ValueError(f"expected each bank to be an object, got {type(bank).__name__}")
            out[bank["Path"]] = [int(s["Id"], 16) & M29 for s in bank.get("Streams") or []]
        return out
    if os.path.isfile(source):
        return {os.path.splitext(os.path.basename(source))[0]: bank_ids(source)}
    out = {}
    for root, _dirs, files in os.walk(source):
        for name in sorted(files):
            if name.lower().endswith(".awc"):
                full = os.path.join(root, name)
                label = os.path.splitext(os.path.relpath(full, source))[0].replace(os.sep, "/")
                out[label] = bank_ids(full)
    return dict(sorted(out.items()))


class Index:
    """id -> bank numbers, for every distinct id of every bank."""

    def __init__(self, banks: dict):
        self.labels = list(banks)
        self.by_id = collections.defaultdict(list)
        for k, label in enumerate(self.labels):
            for i in set(banks[label]):
                self.by_id[i].append(k)
        self.streams = sum(len(set(v)) for v in banks.values())


# ---- dictionary ----
def _files(path: str):
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                yield os.path.join(root, name)
    else:
        yield path


def read_words(path: str) -> list:
    """Words from a text file (one per line), a hashdict.py JSON object or a JSON list of words."""
    with open(path, "rb") as fh:
        head = fh.read(1).lstrip()
    if path.lower().endswith(".json") or head == b"{":
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict):
            return [str(v) for v in doc.values()]
        if isinstance(doc, list):
            return [str(v) for v in doc]
        raise ValueError(f"expected a JSON object of hash -> word or a list of words, got {type(doc).__name__}")
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.strip() for line in fh if line.strip()]


def conversation_waves(paths) -> list:
    """Dialogue wave names from conversation key/label pairs: <key>_01 .. <key>_<labels>."""
    labels = collections.defaultdict(list)
    for path in paths:
        for data in _blobs(path):
            if b"AU_" not in data:
                continue
            for key, label in CONVERSATION_PAIR.findall(data):
                seen = labels[key]
                if label not in seen:
                    seen.append(label)
    return [f"{key.decode().lower()}_{n:02d}" for key, ls in labels.items() for n in range(1, len(ls) + 1)]


def _blobs(path: str):
    """The contents of the files under `path`, a carved stream joined with the streams that continue it.

    rpf9_carve.py writes every zlib stream as its own `<offset>.bin`, and a stream that inflates to a whole number
    of 64 KiB blocks continues in the next one: the next `.bin` of the same folder in name (offset) order, unless
    that one starts a stream of its own. Matched apart, a pair across that boundary would be lost."""
    parts, folder, size = [], None, 0
    for full in _files(path):
        try:
            if os.path.getsize(full) > MAX_SCAN_BYTES:
                data = None
            else:
                with open(full, "rb") as fh:
                    data = fh.read()
        except OSError:
            data = None
        here, carved = os.path.dirname(full), full.lower().endswith(".bin")
        continues = (data is not None and carved and parts and here == folder
                     and len(parts[-1]) and len(parts[-1]) % CONTINUATION_BLOCK == 0
                     and not data.startswith(NEW_STREAM_HEADS) and size + len(data) <= MAX_SCAN_BYTES)
        if not continues:
            if parts:
                yield b"".join(parts)
            parts, size = [], 0
        if data is None:
            continue
        if not carved:
            yield data                       # not a carved stream: read on its own
            continue
        parts.append(data)
        size += len(data)
        folder = here
    if parts:
        yield b"".join(parts)


class Dictionary:
    def __init__(self, words):
        self.words = sorted({w.strip().lower() for w in words if w.strip()})
        self.wordset = set(self.words)
        self.tokens = collections.Counter()
        for w in self.words:
            for t in set(TOKEN_SPLIT.split(w)):
                self.tokens[t] += 1

    def wordy(self, base: str, min_token: int) -> bool:
        if "-" in base or not re.match(r"^[a-z]", base):
            return False
        return all(not t or t.isdigit() or self.tokens[t] >= min_token for t in base.split("_"))

    def bases(self, min_token: int) -> list:
        return sorted(b for b in {TRAILING_NUMBER.sub("", w) for w in self.words} if self.wordy(b, min_token))


# ---- attack ----
def family_ok(names) -> bool:
    """At least 3 members, or 2 with adjacent numbers (_01/_02); chance hits rarely land on adjacent numbers."""
    names = list(names)
    if len(names) >= 3:
        return True
    if len(names) < 2:
        return False
    nums = sorted(int(m.group(1)) for m in (FAMILY_NUMBER.search(n) for n in names) if m)
    return len(nums) == 2 and nums[1] - nums[0] == 1


def _attack_split(dictionary: Dictionary, bases: list, index: Index, salt: str) -> tuple:
    """Separate dialogue and best family hits per (bank number, id) before control filtering."""
    dialogue_hits = {}
    for w in dictionary.words:
        if DIALOGUE.match(w):
            name = salt + w
            h = joaat(name) & M29
            for b in index.by_id.get(h, ()):
                dialogue_hits[(b, h)] = (name, "dialogue", 99)
    suffix_bytes = [_enc(s) for s in SUFFIXES]
    families = collections.defaultdict(dict)     # (bank, base) -> {id: (name, verbatim)}
    by_id = index.by_id
    for base in bases:
        state = joaat_state(_enc(salt + base))
        for s, sb in zip(SUFFIXES, suffix_bytes):
            i = joaat_finish(joaat_state(sb, state)) & M29
            banks = by_id.get(i)
            if banks:
                verbatim = (base + s) in dictionary.wordset
                for b in banks:
                    families[(b, base)][i] = (salt + base + s, verbatim)
    family_hits = {}
    for (b, _base), members in families.items():
        if family_ok(n for n, _v in members.values()) and any(v for _n, v in members.values()):
            size = len(members)
            for i, (name, _v) in members.items():
                prev = family_hits.get((b, i))
                if prev is None or size > prev[2]:
                    family_hits[(b, i)] = (name, "family", size)
    return dialogue_hits, family_hits


def attack(dictionary: Dictionary, bases: list, index: Index, salt: str) -> dict:
    """{(bank number, id): (name, rule, family size)}; ``salt`` != '' is the control run."""
    dialogue_hits, family_hits = _attack_split(dictionary, bases, index, salt)
    return {**family_hits, **dialogue_hits}


def _distinct(found: dict, rule: str, kmin: int = 0) -> int:
    return len({n for n, r, k in found.values() if r == rule and k >= kmin})


def name_source(dictionary: Dictionary, bases: list, index: Index) -> tuple:
    """Accepted names and the per-source statistics, after the salted control picks the thresholds."""
    real_dlg, real_fam = _attack_split(dictionary, bases, index, "")
    ctrl_dlg, ctrl_fam = _attack_split(dictionary, bases, index, SALT)
    keep_dialogue = _distinct(ctrl_dlg, "dialogue") < CONTROL_LIMIT * max(_distinct(real_dlg, "dialogue"), 1)
    kmin = None
    for k in FAMILY_SIZES:
        r, c = _distinct(real_fam, "family", k), _distinct(ctrl_fam, "family", k)
        if r and c < CONTROL_LIMIT * r:
            kmin = k
            break

    def combine(dlg, fam):
        accepted = {}
        if kmin is not None:
            for key, val in fam.items():
                if val[2] >= kmin:
                    accepted[key] = val
        if keep_dialogue:
            accepted.update(dlg)
        return accepted

    real = combine(real_dlg, real_fam)
    control = combine(ctrl_dlg, ctrl_fam)
    stats = {
        "banks": len(index.labels),
        "streams": index.streams,
        "named": len(real),
        "dialogue": sum(v[1] == "dialogue" for v in real.values()),
        "family": sum(v[1] == "family" for v in real.values()),
        "dialogue_rule": keep_dialogue,
        "family_min_size": kmin,
        "control_dialogue": sum(v[1] == "dialogue" for v in control.values()),
        "control_family": sum(v[1] == "family" for v in control.values()),
    }
    return real, stats


def source_labels(sources):
    """Keep basename labels where possible and qualify collisions relative to their shared parent."""
    labels = {src: os.path.basename(os.path.normpath(src)) or src for src in sources}
    groups = collections.defaultdict(list)
    for src, label in labels.items():
        groups[label].append(src)
    for group in groups.values():
        if len(group) < 2:
            continue
        paths = [os.path.abspath(src) for src in group]
        try:
            parent = os.path.commonpath([os.path.dirname(path) for path in paths])
        except ValueError:  # Different Windows drives have no shared parent.
            parent = None
        for src, path in zip(group, paths):
            labels[src] = (os.path.relpath(path, parent) if parent else path).replace(os.sep, "/")
    return labels


def positive_int(text):
    """--min-token: a count of dictionary words a token must appear in, at least 1. 0 or a negative silently
    turned the common-token filter off, since every token passes tokens[t] >= min_token."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("banks", nargs="+", help="an .awc file, a bank directory, or JSON from awc_manifest.py or rpf9dump audio")
    ap.add_argument("--words", action="append", default=[], help="word list or hashdict.py JSON (repeatable)")
    ap.add_argument("--conversations", action="append", default=[],
                    help="file or directory with conversation key/label pairs (repeatable)")
    ap.add_argument("--min-token", type=positive_int, default=MIN_TOKEN,
                    help="dictionary words a family base token must appear in (default 5, at least 1)")
    ap.add_argument("--out", help="names.csv or names.json; default is a summary only")
    a = ap.parse_args(argv)
    if not a.words and not a.conversations:
        ap.error("give at least one --words or --conversations source")
    inputs = a.banks + a.words + a.conversations
    for path in inputs:
        if not os.path.exists(path):
            print(f"awc_names: error: {path}: no such file or directory", file=sys.stderr)
            return 2
    if a.out and os.path.exists(a.out):
        protected = inputs.copy()
        for sources, suffix in ((a.banks, ".awc"), (a.conversations, "")):
            for source in sources:
                if os.path.isdir(source):
                    protected.extend(full for full in _files(source)
                                     if full.lower().endswith(suffix) and os.path.isfile(full))
        if any(os.path.samefile(p, a.out) for p in protected):
            ap.error("--out must not overwrite an input")

    words = []
    for path in a.words:
        try:
            words += read_words(path)
        except (OSError, ValueError) as error:
            print(f"awc_names: error: cannot read {path}: {error}", file=sys.stderr)
            return 2
    waves = conversation_waves(a.conversations)
    dictionary = Dictionary(words + waves)
    bases = dictionary.bases(a.min_token)
    print(f"{len(dictionary.words)} words ({len(waves)} dialogue waves from conversations), "
          f"{len(bases)} family bases", flush=True)

    rows, summary = [], {}
    labels = source_labels(a.banks)
    for source in a.banks:
        try:
            banks = load_banks(source)
        except (OSError, ValueError, KeyError) as error:
            print(f"awc_names: error: {source}: {error}", file=sys.stderr)
            return 2
        index = Index(banks)
        found, stats = name_source(dictionary, bases, index)
        label = labels[source]
        summary[label] = stats
        for (b, i), (name, rule, _k) in sorted(found.items()):
            rows.append({"source": label, "bank": index.labels[b], "stream_id": f"{i:08x}", "name": name,
                         "rule": rule})
        print(f"{label}: {stats['banks']} banks, {stats['streams']} distinct bank/ID pairs, {stats['named']} named "
              f"({stats['dialogue']} dialogue, {stats['family']} family, family size >= {stats['family_min_size']}); "
              f"control {stats['control_dialogue']} dialogue, {stats['control_family']} family", flush=True)
    if a.out:
        try:
            if a.out.lower().endswith(".json"):
                with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                    json.dump({"summary": summary, "streams": rows}, fh, indent=1)
            else:
                with open(a.out, "w", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=["source", "bank", "stream_id", "name", "rule"],
                                       lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
        except OSError as error:
            print(f"awc_names: error: cannot write {a.out}: {error}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
