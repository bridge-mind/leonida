"""Decode the localized string tables and build the dialogue subtitle table.

Text resources are ``0DRP`` logical resources (docs/page_directory.md) whose payload holds a hash table, an index
array and a string blob. A descriptor in the resource locates the three; all pointers are ``0x5xxxxxxx`` resource
pointers whose low 28 bits are offsets into the logical resource (the ``0DRP`` stream joined with its 64 KiB
continuation streams):

```
descriptor   ptr index | u32 | u32 0x10 | ptr | 01 01 00 00 | u32 hash_bytes | ptr hash
             | u32 0x10 | u32 blob_bytes | ptr blob | ... more 16-byte entries
hash entry   16 bytes: u32 joaat(lowercase label), u32 0, u8 1, u8[2], u24 index (at +11), u8
index entry  16 bytes: u64, u64 (blob offset << 20 | flags, blob chunk = flags & 3)
blob         NUL-terminated UTF-8 strings, in one or more chunks
```

From the blob entry on, the descriptor is a run of 16-byte entries (u32 tag, u32 bytes, pointer). A blob of more than
1 MiB is split into several tag-0x10 chunks, and an index entry names its chunk in its low bits (docs/text.md).

Conversation data pairs a dialogue key with its line labels as ``<ROOT>AU_<XXXX>\\0<LABEL>\\0``. A key with N distinct
labels has the audio waves ``<key>_01`` .. ``<key>_<N>`` (the stream names ``awc_names.py`` matches): wave ``_01``
speaks the first label. Joining a label's hash to the string tables gives each wave its subtitle in every language.

The tables carry no language field. Each table's language is guessed from its script (kana, hangul, CJK, Cyrillic)
or, for Latin scripts, from distinctive letters and common words, and is ``xx`` when nothing decides it. Spanish and
Chinese ship two variants each; markers split them into ``es-ES``/``es-419`` and ``zh-Hant``/``zh-Hans`` (the plain
code when a table has no marker).

A table inside a nested RPF9 archive takes its archive's language instead of its own guess. Nested archives are
``9FPR`` headers at 4 KiB-aligned offsets past 0, found in an ``.rpf`` source or in the archive that an
``rpf9_carve.py`` directory's ``index.json`` names. An archive's language is the guess most of its tables share
(``xx`` left out), when that guess covers more than half of them; otherwise each table keeps its own guess. In
``data.rpf`` each of the 13 nested archives that hold string tables is one language (docs/text.md). The tables report
keeps every table's own guess in ``guess`` and its nested archive's offset in ``archive``.

Usage:
    python -I text_tables.py <source> [<source> ...] [--labels FILE ...]
                             [--dialogue dialogue.csv] [--tables tables.csv] [--strings strings.csv]

A source is an ``rpf9_carve.py`` directory, an ``.rpf`` archive (read through ``gamedata_index.py``; nothing is
written), or files and directories of whole resources. ``--labels`` adds label names (word lists or a
``hashdict.py`` JSON) so non-dialogue strings and table groups can be named. Standard library only. The outputs
quote game text: they are local work products and must not be committed or attached to an issue. The tests use
synthetic resources.
"""
import argparse
import bisect
import collections
import csv
import json
import mmap
import os
import re
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from file_mode import published_mode  # noqa: E402
from hashdict import joaat  # noqa: E402

RESOURCE_MAGIC = b"0DRP"
RPF9_MAGIC = b"9FPR"
ARCHIVE_ALIGN = 0x1000
BLOCK = 65536
BLOB_TAG = 0x10                 # descriptor entry tag of a string blob chunk
CHUNK_MASK = 3                  # an index entry's blob chunk, in the low bits of its second word
MAX_RESOURCE_BYTES = 256 << 20
CONVERSATION_PAIR = re.compile(rb"([A-Z0-9_]{2,40}AU_[A-Z]{4})\x00([A-Za-z0-9_]{3,80})\x00")
_PTR = rb"(.{3}[\x50-\x5f])\x00\x00\x00\x00"
DESCRIPTOR = re.compile(_PTR + rb".{4}\x10\x00\x00\x00.{3}[\x50-\x5f]\x00\x00\x00\x00"
                        rb"\x01\x01\x00\x00(.{4})" + _PTR + rb"\x10\x00\x00\x00(.{4})" + _PTR, re.S)
SPEAKER = re.compile(r"_([A-Z]{1,3})_\d+$")
LABEL_WORD = re.compile(r"[A-Za-z0-9_]{3,80}")

STOP_WORDS = {
    "en": "the you and to is what it this that i me my your we not",
    "es": "que el la de no y es lo por un una se mi te",
    "fr": "le les est pas je vous que et un une ce tu",
    "de": "der die und ist nicht ich du das ein es zu",
    "it": "che il non di sono un la ho per mi ti",
    "pt": "que não você é o de um uma eu isso",
    "pl": "nie to jest się co ja ty na tak",
}
STOP_WORDS = {k: set(v.split()) for k, v in STOP_WORDS.items()}
# Letters that occur in one Latin-script language of the set and not (or rarely) in the others.
LETTERS = {"pl": "ąćęłńśźż", "de": "äöüß", "pt": "ãõ", "es": "ñ¿¡", "fr": "œêëîïûù", "it": "ìò"}
LETTER_WEIGHT = 2
# Variant markers. Chinese: characters that differ between Traditional and Simplified script.
ZH_HANT = set("這們說個來時會為過還沒對嗎讓後裡開關動發現樣點車號裝經話見長門問麼")
ZH_HANS = set("这们说个来时会为过还没对吗让后里开关动发现样点车号装经话见长门问么")
# Spanish: second-person plural and vocabulary of Spain against those of Latin America.
ES_ES = set("vosotros vosotras os vuestro vuestra vuestros vuestras habéis hacéis vais sois tenéis queréis estáis "
            "podéis sabéis decís id venid mirad móvil ordenador guay joder hostia".split())
ES_419 = set("ustedes carro celular ahorita chamba güey wey platicar chido neta pinche cuate computadora manejar "
             "órale acá chingar chingada pendejo mande güera".split())
SCRIPTS = (("ja", re.compile(r"[\u3040-\u30ff]")), ("ko", re.compile(r"[\uac00-\ud7af]")),
           ("zh", re.compile(r"[\u4e00-\u9fff]")), ("ru", re.compile(r"[\u0400-\u04ff]")))
LATIN_WORD = re.compile(r"[a-zà-ÿąćęłńóśźż]+")
UNKNOWN = "xx"


# ---- string tables ----
def _ptr(b4: bytes) -> int:
    return struct.unpack("<I", b4)[0] & 0x0FFFFFFF


def blob_chunks(buf: bytes, at: int) -> list:
    """(start, end) of each string blob chunk, reading descriptor entries from ``at`` (the first blob entry).

    A descriptor entry is 16 bytes: u32 tag, u32 bytes, resource pointer. The run ends at the first 16 bytes that are
    not an entry. Every tag-0x10 entry in it is a blob chunk, in order. A chunk that runs past the resource ends the
    list, so the first chunk must fit for the table to be read.
    """
    chunks = []
    while at + 16 <= len(buf):
        tag, size, lo, hi = struct.unpack_from("<IIII", buf, at)
        if hi or lo >> 28 != 5:
            break
        if tag == BLOB_TAG:
            start = lo & 0x0FFFFFFF
            if start + size > len(buf):
                break
            chunks.append((start, start + size))
        at += 16
    return chunks


def string_tables(buf: bytes):
    """Yield {label hash: text} for every string table in one logical resource."""
    for m in DESCRIPTOR.finditer(buf):
        index = _ptr(m.group(1))
        hash_bytes = struct.unpack("<I", m.group(2))[0]
        hashes = _ptr(m.group(3))
        chunks = blob_chunks(buf, m.end() - 16)
        if not (0 < hash_bytes < len(buf) and hashes + hash_bytes <= len(buf) and chunks):
            continue
        out = {}
        for o in range(hashes, hashes + hash_bytes - 15, 16):
            h, zero = struct.unpack_from("<II", buf, o)
            if not h or zero or buf[o + 8] != 1:
                continue
            i = int.from_bytes(buf[o + 11:o + 14], "little")
            at = index + 16 * i + 8
            if at + 8 > len(buf):
                continue
            ref = struct.unpack_from("<Q", buf, at)[0]
            if ref & CHUNK_MASK >= len(chunks):
                continue
            lo, hi = chunks[ref & CHUNK_MASK]
            start = lo + (ref >> 20)
            end = buf.find(b"\0", start, hi)
            if lo <= start <= end:
                out[h] = buf[start:end].decode("utf-8", "replace")
        if out:
            yield out


def language(strings) -> str:
    """Language code of a table: script first, then distinctive letters and common words; "xx" without evidence.

    Spanish and Chinese ship two variants each, told apart by markers: ``es-ES``/``es-419`` and
    ``zh-Hant``/``zh-Hans`` (the plain code when no marker occurs).
    """
    strings = list(strings)[:400]
    sample = " ".join(strings)
    for lang, script in SCRIPTS:
        if script.search(sample):
            if lang == "zh":
                hant = sum(c in ZH_HANT for c in sample)
                hans = sum(c in ZH_HANS for c in sample)
                return "zh-Hant" if hant > hans else "zh-Hans" if hans > hant else "zh"
            return lang
    low = sample.lower()
    words = LATIN_WORD.findall(low)
    score = {k: sum(w in stop for w in words) for k, stop in STOP_WORDS.items()}
    for k, letters in LETTERS.items():
        score[k] += LETTER_WEIGHT * sum(low.count(c) for c in letters)
    best = max(score, key=score.get)
    if score[best] == 0 or list(score.values()).count(score[best]) > 1:
        return UNKNOWN
    if best == "es":
        spain = sum(w in ES_ES for w in words)
        latam = sum(w in ES_419 for w in words)
        return "es-ES" if spain > latam else "es-419" if latam > spain else "es"
    return best


def table_group(labels) -> str:
    """The longest '_'-prefix (up to 3 parts) shared by at least half of the resolved labels."""
    labels = list(labels)
    if len(labels) < 3:
        return ""
    best = ""
    for depth in (1, 2, 3):
        c = collections.Counter("_".join(x.split("_")[:depth]) for x in labels if x.count("_") >= depth)
        if not c:
            break
        prefix, n = c.most_common(1)[0]
        if n * 2 < len(labels):
            break
        best = prefix
    return best


# ---- sources ----
def _is_resource(data: bytes) -> bool:
    return data[0x10:0x14] == RESOURCE_MAGIC


def _chains(rows, read):
    """(offset, bytes) per logical blob: a stream plus the streams that continue a full one.

    A stream is full when its size is a whole number of 64 KiB blocks: a resource's first stream is often 128 or
    192 KiB (docs/page_directory.md), not only 64 KiB, and its descriptors then end in the stream that follows."""
    cur_off, parts, last = None, [], 0
    for row in rows:
        data = read(row)
        continues = (cur_off is not None and last > 0 and last % BLOCK == 0 and not _is_resource(data)
                     and data[:4] != b"PSIN" and data[:4] != b"RBF0")
        if continues and sum(map(len, parts)) + len(data) <= MAX_RESOURCE_BYTES:
            parts.append(data)
        else:
            if cur_off is not None:
                yield cur_off, b"".join(parts)
            cur_off, parts = row["off"], [data]
        last = row["size"]
    if cur_off is not None:
        yield cur_off, b"".join(parts)


def blobs(source: str):
    """(label, offset, bytes) for every logical blob of a source."""
    if os.path.isdir(source) and os.path.isfile(os.path.join(source, "index.json")):
        import gamedata_index
        src = gamedata_index.DirectorySource(source)
        rows = [r for r in src.rows if r.get("type") != "bink2"]
        for off, data in _chains(rows, src.read):
            yield source, off, data
    elif os.path.isfile(source) and source.lower().endswith(".rpf"):
        import gamedata_index
        src = gamedata_index.ArchiveSource(source)
        try:
            rows = [r for r in src.rows if r.get("type") != "bink2"]
            for off, data in _chains(rows, src.read):
                yield source, off, data
        finally:
            src.close()
    else:
        paths = [source]
        if os.path.isdir(source):
            def scan_failed(error: OSError) -> None:
                raise error

            paths = [os.path.join(r, f) for r, _d, fs in os.walk(source, onerror=scan_failed)
                     for f in sorted(fs)]
        for path in paths:
            if os.path.getsize(path) <= MAX_RESOURCE_BYTES:
                with open(path, "rb") as fh:
                    yield path, 0, fh.read()


def nested_archives(path: str) -> list:
    """Offsets of the RPF9 archives nested in an archive file, in order.

    A nested archive is a ``9FPR`` header at a 4 KiB-aligned offset past 0 whose data offset (u32 at +4) is at least
    8 and stays inside the file. Only the header's first 8 bytes are read.
    """
    starts = []
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < 8:
            return starts
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            p = m.find(RPF9_MAGIC, 1)
            while p != -1:
                if p % ARCHIVE_ALIGN == 0 and p + 8 <= size:
                    data_off = struct.unpack_from("<I", m, p + 4)[0]
                    if data_off >= 8 and p + data_off <= size:
                        starts.append(p)
                p = m.find(RPF9_MAGIC, p + 1)
    return starts


def archive_starts(source: str) -> list:
    """Nested-archive offsets for a source whose blob offsets are archive offsets; [] when there is no archive."""
    path = None
    if os.path.isdir(source):
        index = os.path.join(source, "index.json")
        if os.path.isfile(index):
            try:
                with open(index, encoding="utf-8") as fh:
                    named = (json.load(fh).get("summary") or {}).get("archive")
            except (OSError, ValueError, AttributeError):
                named = None
            if isinstance(named, str):
                for candidate in (named, os.path.join(source, named)):
                    if os.path.isfile(candidate):
                        path = candidate
                        break
    elif os.path.isfile(source) and source.lower().endswith(".rpf"):
        path = source
    if path is None:
        return []
    try:
        return nested_archives(path)
    except (OSError, ValueError):
        return []


def archive_of(starts: list, offset: int):
    """The offset of the nested archive holding `offset`, or None before the first one."""
    k = bisect.bisect_right(starts, offset) - 1
    return starts[k] if k >= 0 else None


def read_labels(path: str) -> dict:
    """{hash: label}: a hashdict.py JSON maps hashes already; a text file is hashed word by word."""
    with open(path, "rb") as fh:
        first = fh.read(1).lstrip()
    if path.lower().endswith(".json") or first == b"{":
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a JSON object of hash -> label, got {type(doc).__name__}")
        return {int(h, 16): str(v).upper() for h, v in doc.items()}
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for w in LABEL_WORD.findall(line):
                out.setdefault(joaat(w), w.upper())
    return out


# ---- scan ----
class Scan:
    def __init__(self):
        self.pairs = collections.defaultdict(list)          # key -> labels in order of appearance
        self.found = []                                     # (source, offset, guess, archive, {hash: text})
        self._resolved = None

    def add(self, source, offset, data, archive=None):
        """Read one logical blob. `archive` identifies the nested archive it sits in (None: not in one)."""
        if b"AU_" in data:
            for key, label in CONVERSATION_PAIR.findall(data):
                labels = self.pairs[key.decode()]
                label = label.decode()
                if label not in labels:
                    labels.append(label)
        if _is_resource(data) and b"\x01\x01\x00\x00" in data:
            for table in string_tables(data):
                self.found.append((source, offset, language(table.values()), archive, table))
                self._resolved = None

    def _resolve(self):
        """Languages per table: the archive's majority guess when it has one, else the table's own guess."""
        if self._resolved is None:
            votes = collections.defaultdict(collections.Counter)
            for _s, _o, guess, archive, _t in self.found:
                if archive is not None and guess != UNKNOWN:
                    votes[archive][guess] += 1
            archive_lang = {}
            for archive, counts in votes.items():
                lang, n = counts.most_common(1)[0]
                if 2 * n > sum(counts.values()):
                    archive_lang[archive] = lang
            texts = collections.defaultdict(dict)
            tables = []
            for source, offset, guess, archive, table in self.found:
                lang = archive_lang.get(archive, guess)
                texts[lang].update(table)
                tables.append((source, offset, lang, table))
            self._resolved = {"texts": texts, "tables": tables, "archive_lang": archive_lang}
        return self._resolved

    @property
    def texts(self):
        """{lang: {hash: text}}"""
        return self._resolve()["texts"]

    @property
    def tables(self):
        """[(source, offset, lang, {hash: text})] in reading order."""
        return self._resolve()["tables"]

    @property
    def archive_lang(self):
        """{archive: lang} for every nested archive whose tables agree on a language."""
        return self._resolve()["archive_lang"]

    def languages(self):
        return sorted(self.texts, key=lambda k: (k != "en", k))

    def dialogue_rows(self):
        langs = self.languages()
        for key in sorted(self.pairs):
            for n, label in enumerate(self.pairs[key], 1):
                h = joaat(label)
                speaker = SPEAKER.search(label)
                row = {"wave": f"{key.lower()}_{n:02d}", "key": key, "label": label,
                       "speaker": speaker.group(1) if speaker else ""}
                for lang in langs:
                    row[f"text_{lang}"] = _clean(self.texts[lang].get(h, ""))
                yield row


def _clean(s: str) -> str:
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        n = 0
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def _same_output_path(first: str, second: str) -> bool:
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", help="rpf9_carve.py directory, .rpf archive, or resource files/dirs")
    ap.add_argument("--labels", action="append", default=[], help="label names: word list or hashdict.py JSON")
    ap.add_argument("--dialogue", help="CSV: wave, key, label, speaker, text per language")
    ap.add_argument("--tables", help="CSV: one row per string table (source, offset, lang, group, counts)")
    ap.add_argument("--strings", help="CSV: every string (source, offset, lang, hash, label, text)")
    a = ap.parse_args(argv)
    out_flags = [(flag, path) for flag, path in (("--dialogue", a.dialogue), ("--tables", a.tables),
                                                 ("--strings", a.strings)) if path]
    outs = [path for _, path in out_flags]
    seen = []
    for flag, p in out_flags:
        if any(_same_output_path(p, prev) for prev in seen):
            ap.error("outputs must be distinct files")
        seen.append(p)
        parent = os.path.dirname(os.path.abspath(p))
        if not os.path.isdir(parent):
            ap.error(f"{flag} parent directory does not exist: {parent}")
    for path in a.sources + a.labels:
        if not os.path.exists(path):
            print(f"text_tables: error: {path}: no such file or directory", file=sys.stderr)
            return 2
        if any(os.path.exists(o) and os.path.samefile(path, o) for o in outs):
            ap.error("an output must not overwrite an input")
    for source in a.sources:
        index = os.path.join(source, "index.json")
        if os.path.isdir(source) and os.path.isfile(index) and any(_same_output_path(index, o) for o in outs):
            ap.error(f"an output must not overwrite an input: {index}")

    scan = Scan()
    try:
        for source in a.sources:
            starts = archive_starts(source)
            for label, offset, data in blobs(source):
                nested = archive_of(starts, offset) if starts else None
                scan.add(label, offset, data, None if nested is None else (source, nested))
        names = {}
        for path in a.labels:
            names.update(read_labels(path))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:   # KeyError/TypeError: index.json of the wrong shape
        print(f"text_tables: error: cannot read input: {error}", file=sys.stderr)
        return 2
    for labels in scan.pairs.values():               # dialogue labels name themselves
        for label in labels:
            names[joaat(label)] = label

    lines = sum(len(v) for v in scan.pairs.values())
    langs = scan.languages()
    jobs = []
    if a.dialogue:
        fields = ["wave", "key", "label", "speaker"] + [f"text_{x}" for x in langs]
        jobs.append((a.dialogue, fields, scan.dialogue_rows()))
    if a.tables:
        jobs.append((a.tables, ["source", "offset", "lang", "guess", "archive", "group", "strings", "resolved"],
                     ({"source": s, "offset": o, "lang": lang, "guess": found[2],
                       "archive": "" if found[3] is None else f"{found[3][1]:#x}",
                       "group": table_group(names[h] for h in t if h in names),
                       "strings": len(t), "resolved": sum(h in names for h in t)}
                      for (s, o, lang, t), found in zip(scan.tables, scan.found))))
    if a.strings:
        jobs.append((a.strings, ["source", "offset", "lang", "hash", "label", "text"],
                     ({"source": s, "offset": o, "lang": lang, "hash": f"{h:08x}", "label": names.get(h, ""),
                       "text": _clean(text)}
                      for s, o, lang, t in scan.tables for h, text in t.items())))
    # Stage every report next to its destination, then swap them all in. A write that fails part-way must
    # leave the previous reports intact instead of truncating the ones that came before it.
    staged = []
    originals = {}
    try:
        for dest, fields, rows in jobs:
            parent = os.path.dirname(os.path.abspath(dest))
            fd, tmp = tempfile.mkstemp(dir=parent, prefix=".text_tables-", suffix=".csv")
            os.close(fd)
            staged.append((tmp, dest))
            _write_csv(tmp, fields, rows)
        for _tmp, dest in staged:
            if os.path.isdir(dest):
                raise OSError(f"{dest} is a directory")
        for tmp, dest in staged:
            os.chmod(tmp, published_mode(dest))
        # Move every old report aside before putting any new one in place. On Windows a report held
        # open by a reader cannot be renamed; failing here leaves all the originals untouched.
        for _tmp, dest in staged:
            if os.path.lexists(dest):
                fd, backup = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dest)),
                                              prefix=".text_tables-old-", suffix=".csv")
                os.close(fd)
                os.remove(backup)
                os.replace(dest, backup)
                originals[dest] = backup
            else:
                originals[dest] = None
        for tmp, dest in staged:
            os.replace(tmp, dest)
    except OSError as error:
        for dest, backup in originals.items():
            try:
                if os.path.lexists(dest):
                    os.remove(dest)
                if backup is not None:
                    os.replace(backup, dest)
            except OSError:
                pass
        for tmp, _ in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"text_tables: error: cannot write: {error}", file=sys.stderr)
        return 2
    for backup in originals.values():
        if backup is not None:
            try:
                os.remove(backup)
            except OSError:
                pass
    with_text = collections.Counter()
    for row in scan.dialogue_rows():
        for lang in langs:
            with_text[lang] += bool(row[f"text_{lang}"])
    per_lang = collections.Counter(lang for _s, _o, lang, _t in scan.tables)
    print(f"{len(scan.tables)} string tables ({', '.join(f'{k} {per_lang[k]}' for k in langs)}); "
          f"{len(scan.pairs)} dialogue keys, {lines} lines; lines with text: "
          + ", ".join(f"{k} {with_text[k]}" for k in langs))
    if scan.archive_lang:
        by_archive = sum(found[3] in scan.archive_lang for found in scan.found)
        changed = sum(lang != found[2] for (_s, _o, lang, _t), found in zip(scan.tables, scan.found))
        print(f"language from the nested archive: {by_archive} tables in {len(scan.archive_lang)} archives; "
              f"{changed} differ from their own guess")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
