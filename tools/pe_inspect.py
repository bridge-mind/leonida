"""Inspect a PE image (exe or dll) for porting work.

Usage:
    python -I pe_inspect.py <image> [--game-dir DIR] [--out DIR]

Writes <out>/<file>.json and <out>/<file>.md, where <file> is the image's full file name (game.exe.json,
game.dll.md), so an exe and a dll with the same stem get separate reports. Images with the same file name from
different directories get a numbered suffix (x.exe-2.json), and the name used is printed to stderr. The same file
given twice with the same import search folder is inspected once. Reads the file only; never runs it.

The report also lists run-time load candidates: DLL names the image holds as ASCII or UTF-16 strings without
importing them, where a LoadLibrary* the import table doesn't show would point (docs/runtime_imports.md).

Standard library only: the headers, the section table, the import, delay-import, export, TLS and relocation
directories are parsed with struct, so the tool runs under `python -I` without third-party packages and is covered
by the same standard-library-only CI job as the other tools.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from collections import Counter

SYSTEM32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

MACHINES = {0x8664: "x86-64", 0x14C: "x86", 0xAA64: "ARM64"}
SUBSYSTEMS = {2: "GUI", 3: "Console", 14: "Xbox"}

DIR_EXPORT = 0
DIR_IMPORT = 1
DIR_EXCEPTION = 3
DIR_BASERELOC = 5
DIR_TLS = 9
DIR_DELAY_IMPORT = 13

MAX_ENTRIES = 65536  # a table this long is not a table; stops a malformed image from looping forever


class PeFormatError(ValueError):
    """The file is not a PE image this parser can read."""


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


# An MSVC decorated name can be 4,096 characters (the documented limit of a decorated name); with the 512 this
# was, a long template import became "?" and a DLL with a long name lost its whole entry.
NAME_LIMIT = 4097


def _cstring(data: bytes, offset: int, limit: int = NAME_LIMIT) -> str | None:
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        return None
    return data[offset:end].decode(errors="replace")


class PeImage:
    """The parts of a PE file this tool reports on, read in place from the file bytes."""

    def __init__(self, data: bytes):
        self.data = data
        if data[:2] != b"MZ":
            raise PeFormatError("no MZ header")
        if len(data) < 0x40:
            # The MZ signature is there; the file is just too short to carry the MS-DOS header whose dword at
            # 0x3c holds e_lfanew. Saying "no MZ header" here would blame a signature that is present.
            raise PeFormatError(f"file is {len(data)} bytes, shorter than the 64-byte MS-DOS header")
        self.nt = struct.unpack_from("<I", data, 0x3C)[0]
        if self.nt + 24 > len(data):
            raise PeFormatError(f"file is {len(data)} bytes, too short to hold the PE header at 0x{self.nt:x}")
        if data[self.nt:self.nt + 4] != b"PE\0\0":
            raise PeFormatError("no PE signature")
        (self.machine, self.section_count, self.timestamp, _symbol_table, _symbols, opt_size,
         self.characteristics) = struct.unpack_from("<HHIIIHH", data, self.nt + 4)
        opt = self.nt + 24
        if opt + 2 > len(data):
            raise PeFormatError("truncated optional header")
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic not in (0x10B, 0x20B):
            raise PeFormatError(f"unknown optional header magic 0x{magic:x}")
        # The declared optional header must contain every fixed field we read,
        # and the complete declared header must fit inside the file.
        if opt_size < 72 or opt + opt_size > len(data):
            raise PeFormatError("truncated optional header")
        self.is64 = magic == 0x20B
        self.entry_rva = struct.unpack_from("<I", data, opt + 16)[0]
        self.image_base = struct.unpack_from("<Q" if self.is64 else "<I", data, opt + (24 if self.is64 else 28))[0]
        self.size_of_image = struct.unpack_from("<I", data, opt + 56)[0]
        self.size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
        self.subsystem = struct.unpack_from("<H", data, opt + 68)[0]
        self.dll_characteristics = struct.unpack_from("<H", data, opt + 70)[0]

        # NumberOfRvaAndSizes counts the data directories: a slot past it is not a directory, whatever bytes it
        # holds (runtime/src/pe_loader.cpp reads them the same way). The count is the dword before the table.
        opt_end = min(len(data), opt + opt_size)
        dirs = opt + (112 if self.is64 else 96)
        count = struct.unpack_from("<I", data, dirs - 4)[0] if dirs <= opt_end else 0
        self.directories = []
        for i in range(16):
            at = dirs + i * 8
            present = i < count and at + 8 <= opt_end
            self.directories.append(struct.unpack_from("<II", data, at) if present else (0, 0))

        # A section's data can be virtual only, so lookups past SizeOfRawData have no file bytes to read.
        self.sections = []
        table = opt + opt_size
        for i in range(self.section_count):
            at = table + i * 40
            if at + 40 > len(data):
                raise PeFormatError("truncated section table")
            vsize, rva, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
            chars = struct.unpack_from("<I", data, at + 36)[0]
            self.sections.append({
                "name": data[at:at + 8].rstrip(b"\0").decode(errors="replace"),
                "rva": rva, "vsize": vsize, "rawsize": rawsize, "rawptr": rawptr,
                "exec": bool(chars & 0x20000000), "write": bool(chars & 0x80000000),
            })

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def _range(self, rva: int) -> tuple[int, int] | None:
        """Contiguous file [start, end) backed by headers or the single section holding `rva`."""
        if rva < 0:
            return None
        if rva < self.size_of_headers:
            end = min(self.size_of_headers, len(self.data))
            return (rva, end) if rva < end else None
        for s in self.sections:
            # A section is VirtualSize bytes in memory (SizeOfRawData when that is 0). File bytes past it are not
            # mapped, as in runtime/src/pe_loader.cpp: with SizeOfRawData larger than the span to the next
            # section, max(vsize, rawsize) answered the next section's RVAs from this one's raw tail.
            span = s["vsize"] or s["rawsize"]
            if s["rva"] <= rva < s["rva"] + span:
                delta = rva - s["rva"]
                mapped = min(s["rawsize"], span)
                if delta >= mapped:
                    return None
                at = s["rawptr"] + delta
                end = min(s["rawptr"] + mapped, len(self.data))
                return (at, end) if at < end else None
        return None

    def offset(self, rva: int, size: int = 1) -> int | None:
        """File offset of an RVA span wholly backed by headers or one section."""
        if size <= 0:
            return None
        bounds = self._range(rva)
        if bounds is None:
            return None
        start, end = bounds
        return start if start + size <= end else None

    def section_data(self, section: dict) -> bytes:
        at = section["rawptr"]
        if at >= len(self.data):
            return b""
        return self.data[at:at + min(section["rawsize"], len(self.data) - at)]

    def string_at(self, rva: int) -> str | None:
        bounds = self._range(rva)
        if bounds is None:
            return None
        start, end = bounds
        return _cstring(self.data, start, min(NAME_LIMIT, end - start))

    def word(self, at: int) -> int:
        return struct.unpack_from("<Q" if self.is64 else "<I", self.data, at)[0]

    def thunk_names(self, lookup_rva: int) -> list[str]:
        """Names (or '#ordinal') of one lookup table, which is terminated by a zero entry."""
        ordinal_flag = 1 << (63 if self.is64 else 31)
        out = []
        width = 8 if self.is64 else 4
        for i in range(MAX_ENTRIES):
            at = self.offset(lookup_rva + i * width, width)
            if at is None:
                break
            value = self.word(at)
            if not value:
                break
            if value & ordinal_flag:
                out.append(f"#{value & 0xFFFF}")
                continue
            name = self.string_at(value + 2)  # skip the 2-byte hint
            out.append(name if name is not None else "?")
        return out


def _imports(img: PeImage, kind: str, index: int, delay: bool = False) -> list[dict]:
    rva, size = img.directory(index)
    if not rva or not size:
        return []

    entries = []
    step = 32 if delay else 20
    for i in range(min(size // step, MAX_ENTRIES)):
        base = img.offset(rva + i * step, step)
        if base is None:
            break
        if delay:
            (attributes, dll_rva, _handle, iat_rva, int_rva, _bound, _unload,
             _stamp) = struct.unpack_from("<IIIIIIII", img.data, base)
            if not dll_rva:
                break
            # dlattrRva (0x1) is set on every descriptor Windows writes; without it the fields are virtual
            # addresses, which the linker uses for 32-bit images. Convert them so both kinds read the same here.
            if not attributes & 1:
                dll_rva = dll_rva - img.image_base
                iat_rva = iat_rva - img.image_base
                int_rva = int_rva - img.image_base
        else:
            oft, _stamp, _forwarder, dll_rva, iat_rva = struct.unpack_from("<IIIII", img.data, base)
            int_rva = oft
            if not dll_rva:
                break
        if dll_rva < 0:
            continue
        dll = img.string_at(dll_rva)
        if dll is None:
            continue
        lookup = int_rva if int_rva > 0 else iat_rva
        entries.append({"dll": dll, "kind": kind, "functions": img.thunk_names(lookup) if lookup > 0 else []})
    return entries


def _exports(img: PeImage) -> list[str]:
    rva, size = img.directory(DIR_EXPORT)
    if not rva or size < 40:
        return []
    at = img.offset(rva, 40)
    if at is None:
        start = img.offset(rva)
        if start is not None and start + 40 > len(img.data):
            raise PeFormatError("truncated export directory")
        return []
    (_chars, _stamp, _major, _minor, _name, base, functions, name_count, addr_functions,
     addr_names, addr_ordinals) = struct.unpack_from("<IIHHIIIIIII", img.data, at)

    def has_export_at(index: int) -> bool:
        if index >= functions:
            return False
        # A zero RVA marks an unused slot. An RVA inside the export directory is a valid forwarder.
        slot = img.offset(addr_functions + index * 4, 4)
        return slot is not None and struct.unpack_from("<I", img.data, slot)[0] != 0

    named, names = set(), []
    for i in range(min(name_count, MAX_ENTRIES)):
        name_at = img.offset(addr_names + i * 4, 4)
        ord_at = img.offset(addr_ordinals + i * 2, 2)
        if name_at is None or ord_at is None:
            break
        index = struct.unpack_from("<H", img.data, ord_at)[0]
        if not has_export_at(index):
            continue
        symbol = img.string_at(struct.unpack_from("<I", img.data, name_at)[0])
        names.append(symbol if symbol is not None else "?")
        named.add(index)
    for i in range(min(functions, MAX_ENTRIES)):
        if i not in named and has_export_at(i):
            names.append(f"#{base + i}")
    return names


def _export_module_name(img: PeImage) -> str:
    """The module name the export directory records (its Name field), or "" when there is none."""
    rva, size = img.directory(DIR_EXPORT)
    at = img.offset(rva, 40) if rva and size >= 40 else None
    if at is None:
        return ""
    return img.string_at(struct.unpack_from("<I", img.data, at + 12)[0]) or ""


def _tls_callbacks(img: PeImage) -> int:
    rva, size = img.directory(DIR_TLS)
    header_size = 40 if img.is64 else 24
    callbacks_end = 32 if img.is64 else 16
    if not rva or size < header_size:
        return 0
    at = img.offset(rva, header_size)
    if at is None:
        start = img.offset(rva)
        if start is not None and start + callbacks_end > len(img.data):
            raise PeFormatError("truncated TLS directory")
        return 0
    callbacks_va = struct.unpack_from("<Q" if img.is64 else "<I", img.data, at + (24 if img.is64 else 12))[0]
    if not callbacks_va or callbacks_va < img.image_base:
        return 0
    callbacks = callbacks_va - img.image_base
    count = 0
    width = 8 if img.is64 else 4
    for i in range(MAX_ENTRIES):
        entry = img.offset(callbacks + i * width, width)
        if entry is None or not img.word(entry):
            break
        count += 1
    return count


_apiset_cache: dict[str, bool] = {}


def _apiset_resolves(name: str) -> bool:
    """True if this Windows' API-set schema maps the contract to a host DLL (ntdll query; nothing is loaded)."""
    if name not in _apiset_cache:
        _apiset_cache[name] = _query_apiset(name)
    return _apiset_cache[name]


def _query_apiset(name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", wintypes.LPWSTR)]

        query = ctypes.WinDLL("ntdll").ApiSetQueryApiSetPresence
        query.argtypes = [ctypes.POINTER(UNICODE_STRING), ctypes.POINTER(ctypes.c_ubyte)]
        query.restype = ctypes.c_long
        base = name[:-4] if name.lower().endswith(".dll") else name
        buf = ctypes.create_unicode_buffer(base)
        text = ctypes.cast(buf, wintypes.LPWSTR)
        us = UNICODE_STRING(len(base) * 2, (len(base) + 1) * 2, text)
        present = ctypes.c_ubyte(0)
        status = query(ctypes.byref(us), ctypes.byref(present))
        return status == 0 and bool(present.value)
    except (OSError, AttributeError, ValueError):
        return False


# A DLL file name as an image stores it for LoadLibrary*: ASCII or UTF-16LE, letters, digits, '_', '-' and '.'.
# A UTF-16 name starts after a NUL (or at the start), so the tail of an ASCII string followed by one isn't read as one.
_DLL_ASCII = re.compile(rb"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}\.[dD][lL][lL](?![A-Za-z0-9_])")
_DLL_WIDE = re.compile(rb"(?<![^\x00])(?:[A-Za-z0-9_]\x00)(?:[A-Za-z0-9_.\-]\x00){0,63}"
                       rb"\.\x00[dD]\x00[lL]\x00[lL]\x00(?![A-Za-z0-9_]\x00)")


def dll_strings(data: bytes, imported: set[str], own_name: str | tuple[str, ...] = "") -> list[dict]:
    """DLL names the image holds as strings but doesn't import: candidates for a run-time LoadLibrary*, which the
    import table doesn't show (XCurl.dll and libHttpClient.GDK.dll reach xgameruntime.dll only this way,
    docs/runtime_imports.md). `imported` is the lower-cased names from the import tables; the image's own names
    (its file name and the name in its export directory) are left out too. Each entry is {"name", "encoding":
    "ascii" | "utf-16" | "both"}, sorted by name. A string is not proof of a load, only of where to look."""
    seen: dict[str, set[str]] = {}
    for m in _DLL_ASCII.finditer(data):
        seen.setdefault(m.group().decode("ascii").lower(), set()).add("ascii")
    for m in _DLL_WIDE.finditer(data):
        seen.setdefault(m.group().decode("utf-16-le").lower(), set()).add("utf-16")
    own = (own_name,) if isinstance(own_name, str) else own_name
    skip = imported | {n.lower() for n in own if n}
    return [{"name": n, "encoding": "both" if len(e) == 2 else next(iter(e))}
            for n, e in sorted(seen.items()) if n not in skip]


def classify_dll(name: str, game_dir: str | None) -> str:
    """bundled = ships beside the game, system = present on this Windows, shim = must be reimplemented."""
    if game_dir and os.path.isfile(os.path.join(game_dir, name)):
        return "bundled"
    low = name.lower()
    if low.startswith("api-ms-win-") or low.startswith("ext-ms-win-"):
        # API sets resolve through the OS schema: ask this Windows whether it knows the contract.
        return "system" if _apiset_resolves(name) else "shim"
    if os.path.isfile(os.path.join(SYSTEM32, name)):
        return "system"
    return "shim"


def inspect(path: str, game_dir: str | None) -> dict:
    """The report for one image. A file cut off inside a table raises PeFormatError like any other bad image;
    the struct.error it used to raise escaped main() and stopped a multi-file run at that file."""
    try:
        return _inspect(path, game_dir)
    except struct.error as e:
        raise PeFormatError(f"truncated or corrupt table: {e}") from None


def _inspect(path: str, game_dir: str | None) -> dict:
    with open(path, "rb") as f:
        img = PeImage(f.read())
    sections = []
    for s in img.sections:
        data = img.section_data(s)
        e = entropy(data)
        sections.append({
            "name": s["name"],
            "rva": s["rva"], "vsize": s["vsize"], "rawsize": s["rawsize"],
            "exec": s["exec"],
            "write": s["write"],
            "entropy": round(e, 3),
            "looks_encrypted": e > 7.9 and len(data) > 4096,
        })

    imports = _imports(img, "normal", DIR_IMPORT) + _imports(img, "delay", DIR_DELAY_IMPORT, delay=True)
    for entry in imports:
        entry["where"] = classify_dll(entry["dll"], game_dir)

    exports = _exports(img)
    own_names = (os.path.basename(path), _export_module_name(img))
    loads = dll_strings(img.data, {i["dll"].lower() for i in imports}, own_names)
    for entry in loads:
        entry["where"] = classify_dll(entry["name"], game_dir)
    pdata_rva, pdata_size = img.directory(DIR_EXCEPTION)
    reloc_rva, reloc_size = img.directory(DIR_BASERELOC)

    return {
        "file": os.path.basename(path),
        "size": os.path.getsize(path),
        "machine": MACHINES.get(img.machine, hex(img.machine)),
        "timestamp": img.timestamp,
        "is_dll": bool(img.characteristics & 0x2000),
        "subsystem": SUBSYSTEMS.get(img.subsystem, img.subsystem),
        "image_base": hex(img.image_base),
        "size_of_image": img.size_of_image,
        "entry_rva": hex(img.entry_rva),
        "has_relocs": bool(reloc_rva and reloc_size),
        "dynamic_base": bool(img.dll_characteristics & 0x40),
        "sections": sections,
        "imports": imports,
        "export_count": len(exports),
        "exports": exports,
        "pdata_functions": pdata_size // 12 if pdata_rva and pdata_size >= 12 else 0,
        "tls_callbacks": _tls_callbacks(img),
        "runtime_load_candidates": loads,
    }


def to_markdown(r: dict) -> str:
    out = [f"# {r['file']}", "",
           f"- {r['machine']}, {'DLL' if r['is_dll'] else 'EXE'}, subsystem {r['subsystem']}, {r['size']:,} bytes",
           f"- image base {r['image_base']}, size {r['size_of_image']:,}, entry RVA {r['entry_rva']}",
           f"- relocations: {'yes' if r['has_relocs'] else 'NO (must load at preferred base)'}; "
           f"ASLR flag: {r['dynamic_base']}",
           f"- .pdata functions: {r['pdata_functions']:,}; TLS callbacks: {r['tls_callbacks']}; exports: {r['export_count']}",
           "", "## Sections", "", "| name | rva | vsize | exec | write | entropy | |", "|---|---|---|---|---|---|---|"]
    for s in r["sections"]:
        flag = "**encrypted?**" if s["looks_encrypted"] else ""
        out.append(f"| {s['name']} | {s['rva']:#x} | {s['vsize']:,} | {s['exec']} | {s['write']} | {s['entropy']} | {flag} |")
    out += ["", "## Imports", ""]
    by_where = Counter()
    for imp in r["imports"]:
        by_where[imp["where"]] += len(imp["functions"])
    out.append(", ".join(f"{k}: {v}" for k, v in sorted(by_where.items())) or "none")
    out += ["", "| dll | kind | resolves | functions |", "|---|---|---|---|"]
    for imp in sorted(r["imports"], key=lambda i: (i["where"] != "shim", i["dll"].lower())):
        out.append(f"| {imp['dll']} | {imp['kind']} | {imp['where']} | {len(imp['functions'])} |")
    loads = r.get("runtime_load_candidates", [])
    if loads:
        out += ["", "## Run-time load candidates", "",
                "DLL names held as strings but not imported: a `LoadLibrary*` the import table doesn't show "
                "(a string is where to look, not proof of a load).", "",
                "| dll | encoding | resolves |", "|---|---|---|"]
        out += [f"| {d['name']} | {d['encoding']} | {d['where']} |" for d in loads]
    shims = [i for i in r["imports"] if i["where"] == "shim"]
    if shims:
        out += ["", "## Needs checking or a shim", ""]
        for imp in shims:
            out.append(f"### {imp['dll']} ({imp['where']})")
            out.append("")
            out.append(", ".join(f"`{f}`" for f in imp["functions"]))
            out.append("")
    return "\n".join(out) + "\n"


def same_output_path(first: str, second: str) -> bool:
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="+")
    ap.add_argument("--game-dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "pe"))
    a = ap.parse_args(argv)
    missing = [img for img in a.image if not os.path.exists(img)]
    if missing:
        for img in missing:
            print(f"ERROR: {img}: does not exist", file=sys.stderr)
        return 2
    if os.path.isfile(a.out):
        print(f"cannot create output directory {a.out}: path is a file, not a directory", file=sys.stderr)
        return 1
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        print(f"cannot create output directory {a.out}: {e.strerror or e}", file=sys.stderr)
        return 1
    rc = 0
    seen_paths: set[tuple[str, str]] = set()
    used_names: set[str] = set()
    for img in a.image:
        search_dir = a.game_dir or os.path.dirname(os.path.abspath(img))
        key = (os.path.normcase(os.path.realpath(img)), os.path.normcase(os.path.abspath(search_dir)))
        if key in seen_paths:
            print(f"{img}: given more than once; inspected once", file=sys.stderr)
            continue
        seen_paths.add(key)
        try:
            r = inspect(img, search_dir)
        except (PeFormatError, OSError) as e:
            print(f"{img}: not a PE image ({e})", file=sys.stderr)
            rc = 1
            continue
        # One report per image: the full file name keeps game.exe and game.dll apart, and a numbered suffix keeps
        # a/x.exe and b/x.exe apart. Compared case-insensitively, as Windows file names are.
        name = r["file"]
        n = 1
        while name.lower() in used_names:
            n += 1
            name = f"{r['file']}-{n}"
        used_names.add(name.lower())
        if n > 1:
            print(f"{img}: another input is also named {r['file']}; report written as {name}.json/.md",
                  file=sys.stderr)
        stem = os.path.join(a.out, name)
        json_path = stem + ".json"
        md_path = stem + ".md"
        collision = None
        for other in a.image:
            if same_output_path(other, json_path):
                collision = json_path
                break
            if same_output_path(other, md_path):
                collision = md_path
                break
        if collision:
            print(f"{img}: report {collision} would overwrite input file", file=sys.stderr)
            rc = 1
            continue
        try:
            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(r, f, indent=1)
            with open(md_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(to_markdown(r))
        except OSError as e:
            print(f"{img}: cannot write report ({e.strerror or e})", file=sys.stderr)
            rc = 1
            continue
        enc = [s["name"] for s in r["sections"] if s["looks_encrypted"]]
        shim = sum(len(i["functions"]) for i in r["imports"] if i["where"] == "shim")
        print(f"{r['file']:<40} {r['machine']:<7} subsys={r['subsystem']!s:<7} pdata={r['pdata_functions']:<7} "
              f"shim-imports={shim:<5} encrypted={','.join(enc) or '-'}")
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
