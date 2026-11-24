r"""Which GUID constants does a PE image's code pass at its call sites? (D3D-441, stdlib only)

A console client that queries or creates a D3D12 object names the interface with a GUID: either a
16-byte constant in its own data (IID_PPV_ARGS, QueryInterface), a GUID imported as data, or an
unknown console IID that no PC header lists. tools/d3d12_iid_scan.py answers "are these *known*
IIDs present anywhere in the image"; this tool answers the call-site question: *which* 16-byte
constants does the image's code take the address of, so an unknown GUID (a console-only IID) cannot
hide among them.

It scans executable sections for RIP-relative `lea reg, [rip+disp32]` (the way MSVC materialises
`&GUID` for x64 argument passing), maps each target into the image, and keeps the targets in
non-executable initialized sections whose 16 bytes are GUID-shaped (version nibble 1-5, RFC-4122
variant). It also scans data sections for absolute 8-byte pointers to GUID-shaped constants (a GUID
address stored in a struct), and lists `mov reg, [rip+...]` loads that go through an import slot,
because a GUID imported as data would otherwise be invisible. Every kept candidate is matched
against the IIDs of the installed d3d12.h plus `--known NAME=GUID` values; the rest are printed as
unknown, with the referencing count and the constant's own RVA, for call-site triage.

It also reports data windows referenced by code that parse as a `D3D12_COMMAND_QUEUE_DESC`
{Type, Priority, Flags, NodeMask}: a static queue description with a console list type is where the
DMA type's numeric value would appear if a shipped module created such a queue.

It prints names, counts and GUID values only, no bytes of the image.

Usage:
    python -I tools/d3d12_guid_sites.py <image> [<image> ...] [--header PATH/d3d12.h]
                                        [--known NAME=GUID ...] [--json OUT]

Without --header it takes d3d12.h from the newest Windows SDK under "Windows Kits\\10\\Include".
The --json output must name a file separate from the images and the header.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

MIDL = re.compile(r'MIDL_INTERFACE\("([0-9a-fA-F-]{36})"\)\s*\n\s*(ID3D12\w+)')

# RIP-relative operand: REX.W(le) 0x8D/0x8B, modrm mod=00 rm=101; instruction length 7 (disp32).
_LEA, _MOV = 0x8D, 0x8B
REXES = (0x48, 0x49, 0x4C, 0x4D)  # REX.W with optional R/B; mod=00 rm=101 stays RIP-relative


def default_header() -> str | None:
    root = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                        "Windows Kits", "10", "Include")
    found = glob.glob(os.path.join(root, "*", "um", "d3d12.h"))

    def version(path: str) -> tuple[int, ...]:
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return tuple(int(p) for p in name.split(".") if p.isdigit())

    return max(found, key=version) if found else None


def read_iids(header: str) -> dict[str, uuid.UUID]:
    with open(header, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return {name: uuid.UUID(guid) for guid, name in MIDL.findall(text)}


class Image:
    """A parsed PE32/PE32+ image: sections, the image base and the import directory."""

    def __init__(self, data: bytes):
        self.data = data
        if data[:2] != b"MZ":
            raise ValueError("not an MZ image")
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("not a PE image")
        sections, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        self.pe32plus = struct.unpack_from("<H", data, opt)[0] == 0x20B
        base_off = 24 if self.pe32plus else 28  # PE32+ ImageBase at opt+24, PE32 at opt+28
        self.image_base = struct.unpack_from("<Q" if self.pe32plus else "<I", data, opt + base_off)[0]
        dirs = opt + (112 if self.pe32plus else 96)
        self.dirs = [struct.unpack_from("<II", data, dirs + 8 * i) for i in range(16)]
        self.sections = []
        table = opt + opt_size
        for i in range(sections):
            s = table + 40 * i
            name = data[s:s + 8].rstrip(b"\0").decode("ascii", "replace")
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
            chars = struct.unpack_from("<I", data, s + 36)[0]
            self.sections.append({"name": name, "va": va, "vsize": vsize, "raw": raw, "rsize": rsize,
                                  "chars": chars})

    def rva_to_offset(self, rva: int) -> int | None:
        for s in self.sections:
            if s["va"] <= rva < s["va"] + max(s["vsize"], s["rsize"]):
                off = s["raw"] + rva - s["va"]
                return off if off < len(self.data) else None
        return None

    def read(self, rva: int, size: int) -> bytes | None:
        off = self.rva_to_offset(rva)
        return self.data[off:off + size] if off is not None else None

    def initialized(self) -> list[dict]:
        return [s for s in self.sections if s["rsize"] > 0]

    def section_of(self, rva: int) -> dict | None:
        for s in self.initialized():
            if s["va"] <= rva < s["va"] + max(s["vsize"], s["rsize"]):
                return s
        return None

    def imports(self) -> list[tuple[str, list[str]]]:
        """(dll, imported names or "#ordinal") from the import directory."""
        out: list[tuple[str, list[str]]] = []
        imp_rva = self.dirs[1][0]
        if not imp_rva:
            return out
        pos = self.rva_to_offset(imp_rva)
        width, flag = (8, 1 << 63) if self.pe32plus else (4, 1 << 31)
        while pos is not None and pos + 20 <= len(self.data):
            lookup_rva, _, _, name_rva, iat_rva = struct.unpack_from("<IIIII", self.data, pos)
            if name_rva == 0:
                break
            at = self.rva_to_offset(name_rva)
            if at is None:
                break
            dll = self.data[at:self.data.index(b"\0", at)].decode("ascii", "replace").lower()
            names: list[str] = []
            thunk = self.rva_to_offset(lookup_rva or iat_rva)
            while thunk is not None and thunk + width <= len(self.data):
                value = struct.unpack_from("<Q" if self.pe32plus else "<I", self.data, thunk)[0]
                if value == 0:
                    break
                if value & flag:
                    names.append(f"#{value & 0xFFFF}")
                else:
                    hint = self.rva_to_offset(value & 0x7FFFFFFF)
                    names.append(self.data[hint + 2:self.data.index(b"\0", hint + 2)].decode("ascii", "replace")
                                 if hint is not None else "?")
                thunk += width
            out.append((dll, names))
            pos += 20
        return out

    def iat_slots(self) -> dict[int, tuple[str, str]]:
        """RVA of every IAT slot -> (dll, import name); targets of `mov reg, [rip+...]` are matched here."""
        slots: dict[int, tuple[str, str]] = {}
        imp_rva = self.dirs[1][0]
        if not imp_rva:
            return slots
        pos = self.rva_to_offset(imp_rva)
        width = 8 if self.pe32plus else 4
        while pos is not None and pos + 20 <= len(self.data):
            lookup_rva, _, _, name_rva, iat_rva = struct.unpack_from("<IIIII", self.data, pos)
            if name_rva == 0:
                break
            at = self.rva_to_offset(name_rva)
            if at is None:
                break
            dll = self.data[at:self.data.index(b"\0", at)].decode("ascii", "replace").lower()
            thunk_rva = lookup_rva or iat_rva
            thunk = self.rva_to_offset(thunk_rva)
            i = 0
            while thunk is not None and thunk + width <= len(self.data):
                value = struct.unpack_from("<Q" if self.pe32plus else "<I", self.data, thunk)[0]
                if value == 0:
                    break
                if value & (1 << 63 if self.pe32plus else 1 << 31):
                    name = f"#{value & 0xFFFF}"
                else:
                    hint = self.rva_to_offset(value & 0x7FFFFFFF)
                    name = (self.data[hint + 2:self.data.index(b"\0", hint + 2)].decode("ascii", "replace")
                            if hint is not None else "?")
                slots[thunk_rva + i * width] = (dll, name)
                thunk += width
                i += 1
            if iat_rva and iat_rva != thunk_rva:  # the loader binds the IAT copy; code loads through it
                for j in range(i):
                    slots.setdefault(iat_rva + 8 * j, slots[thunk_rva + 8 * j])
            pos += 20
        return slots


def guid_shaped(blob: bytes) -> bool:
    """A plausible RFC-4122 GUID: version nibble 1-5, variant 10xx, and not all-zero/constant."""
    if len(blob) < 16 or blob[:4] == b"\0\0\0\0":
        return False
    if blob[:4] == b"DXBC":  # an embedded shader container, whose header hashes parse as GUID-shaped
        return False
    version = blob[7] >> 4
    variant = blob[8] >> 6
    if version not in (1, 2, 3, 4, 5) or variant != 0b10:
        return False
    return len(set(blob)) > 6  # random data, not a fill pattern


def rip_refs(img: Image) -> list[tuple[int, int, int]]:
    """(instruction RVA, target RVA, opcode) for every RIP-relative lea/mov in executable sections."""
    out: list[tuple[int, int, int]] = []
    for s in img.initialized():
        if not s["chars"] & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
            continue
        blob = img.data[s["raw"]:s["raw"] + s["rsize"]]
        for i in range(len(blob) - 6):  # a 7-byte RIP-relative instruction must fit entirely
            if blob[i] not in REXES or blob[i + 1] not in (_LEA, _MOV):
                continue
            modrm = blob[i + 2]
            if (modrm & 0xC7) != 0x05:  # mod=00, rm=101
                continue
            disp = struct.unpack_from("<i", blob, i + 3)[0]
            insn_rva = s["va"] + i
            out.append((insn_rva, insn_rva + 7 + disp, blob[i + 1]))
    return out

def movabs_pairs(img: Image) -> list[tuple[int, bytes]]:
    """(first instruction RVA, 16 bytes) of adjacent `movabs reg, imm64` pairs that inline a GUID.

    A compiler that materialises an IID without a constant pool writes its two halves as two 10-byte
    `REX.W B8+r imm64` moves; the concatenation is the GUID a lea would otherwise point at.
    """
    out: list[tuple[int, bytes]] = []
    for s in img.initialized():
        if not s["chars"] & 0x20000000:
            continue
        blob = img.data[s["raw"]:s["raw"] + s["rsize"]]
        imm: list[tuple[int, bytes]] = []
        for i in range(len(blob) - 10):
            if blob[i] == 0x48 and 0xB8 <= blob[i + 1] <= 0xBF:
                imm.append((s["va"] + i, blob[i + 2:i + 10]))
        for (rva1, first), (rva2, second) in zip(imm, imm[1:]):
            if rva2 - rva1 > 32:  # the two halves are written together
                continue
            joined = first + second
            if guid_shaped(joined):
                out.append((rva1, joined))
    return out

def absolute_refs(img: Image) -> list[tuple[int, int]]:
    """(location RVA, target RVA) for 8-byte aligned absolute pointers into the image's own data."""
    out: list[tuple[int, int]] = []
    if not img.pe32plus:
        return out
    lo = img.image_base
    for s in img.initialized():
        blob = img.data[s["raw"]:s["raw"] + s["rsize"]]
        for i in range(0, len(blob) - 8, 8):
            value = struct.unpack_from("<Q", blob, i)[0]
            if lo <= value < lo + 0x10000000:
                target = value - lo
                if any(t["va"] <= target < t["va"] + max(t["vsize"], t["rsize"]) for t in img.sections):
                    out.append((s["va"] + i, target))
    return out


def queue_desc_shaped(blob: bytes) -> bool:
    """A plausible `D3D12_COMMAND_QUEUE_DESC` {Type, Priority, Flags, NodeMask}."""
    if blob == b"\0" * 16:  # a referenced zero block is padding, not a description
        return False
    t, p, flags, node = struct.unpack_from("<IIII", blob)
    if t > 15 or node > 1:  # console list types stay small; a title queues on one node
        return False
    if p not in (0, 1, 2, 0xFFFFFFFF, 0x80000000):  # NORMAL/HIGH/GLOBAL_REALTIME/"-1"/OR-ed
        return False
    return flags <= 0x3  # D3D12_COMMAND_QUEUE_FLAG_NONE/ALLOW_GPU_TIMEOUT/DISABLE_GPU_TIMEOUT


def scan(path: str, known: dict[str, uuid.UUID]) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    img = Image(data)
    by_value = {}
    for name, guid in known.items():
        by_value.setdefault(guid.bytes_le, []).append(name)

    def constant_at(rva: int) -> tuple[bytes, str] | None:
        """The 16 bytes at `rva` when it sits in an initialized, non-executable section (a constant pool)."""
        s = img.section_of(rva)
        if s is None or s["chars"] & 0x20000000:
            return None
        blob = img.read(rva, 16)
        return (blob, s["name"]) if blob is not None else None

    candidates: dict[bytes, dict] = {}
    leas = 0
    iat_loads: list[tuple[int, str, str]] = []
    slots = img.iat_slots()
    for insn_rva, target, opcode in rip_refs(img):
        if opcode == _MOV:
            if target in slots:
                dll, name = slots[target]
                iat_loads.append((insn_rva, dll, name))
            continue
        leas += 1
        at = constant_at(target)
        if at is None or not guid_shaped(at[0]):
            continue
        entry = candidates.setdefault(at[0], {"refs": [], "known": None, "section": at[1], "rva": target})
        entry["refs"].append(insn_rva)
        if at[0] in by_value:
            entry["known"] = by_value[at[0]]
    for loc_rva, target in absolute_refs(img):
        at = constant_at(target)
        if at is None or not guid_shaped(at[0]):
            continue
        entry = candidates.setdefault(at[0], {"refs": [], "known": None, "section": at[1], "rva": target})
        entry["refs"].append(loc_rva)
        if at[0] in by_value:
            entry["known"] = by_value[at[0]]
    for rva, joined in movabs_pairs(img):
        entry = candidates.setdefault(joined, {"refs": [], "known": None, "section": "movabs", "rva": rva})
        entry["refs"].append(rva)
        if joined in by_value:
            entry["known"] = by_value[joined]
    lea_targets = {t for _, t, op in rip_refs(img) if op == _LEA}
    descs = []
    for s in img.initialized():
        if s["chars"] & 0x20000000:  # code sections are not queue descriptions
            continue
        blob = img.data[s["raw"]:s["raw"] + s["rsize"]]
        for i in range(0, len(blob) - 16, 4):
            if s["va"] + i in lea_targets and queue_desc_shaped(blob[i:i + 16]):
                t, p, flags, node = struct.unpack_from("<IIII", blob, i)
                descs.append({"rva": s["va"] + i, "type": t, "priority": p, "flags": flags, "node": node})

    return {
        "image": os.path.basename(path),
        "lea_count": leas,
        "guid_candidates": [
            {"value": str(uuid.UUID(bytes_le=blob)), "known": e["known"], "refs": len(e["refs"]),
             "rva": e["rva"], "section": e["section"]}
            for blob, e in sorted(candidates.items(), key=lambda kv: -len(kv[1]["refs"]))
        ],
        "iat_guid_loads": [{"rva": rva, "dll": dll, "name": name} for rva, dll, name in iat_loads
                           if name.startswith("IID_") or "CLSID" in name or "GUID" in name],
        "queue_descs": descs,
        "imports_of_note": [f"{dll}: {', '.join(names)}" for dll, names in img.imports()
                            if any(n.startswith("IID_") or "GUID" in n or "CLSID" in n for n in names)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--header", help="d3d12.h to read the known IIDs from (default: newest Windows SDK)")
    ap.add_argument("--known", action="append", default=[], metavar="NAME=GUID",
                    help="a known GUID the header does not list, e.g. a console IID (repeatable)")
    ap.add_argument("--json", help="also write the result here")
    args = ap.parse_args(argv)

    header = args.header
    if header is None:
        header = default_header()
    if args.json:
        for source in (*args.images, header):
            if source and same_report_path(args.json, source):
                ap.error(f"--json {args.json} would overwrite the input {source}")

    known: dict[str, uuid.UUID] = {}
    if header and os.path.isfile(header):
        known.update(read_iids(header))
    for pair in args.known:
        name, _, value = pair.partition("=")
        try:
            known[name] = uuid.UUID(value)
        except ValueError:
            print(f"ERROR: --known expects NAME=GUID, got {pair!r}", file=sys.stderr)
            return 2
    print(f"{len(known)} known GUIDs" + (f" from {header}" if header else ""))

    results = []
    for path in args.images:
        if not os.path.isfile(path):
            print(f"ERROR: cannot read {path}: no such file", file=sys.stderr)
            return 2
        try:
            r = scan(path, known)
        except (ValueError, struct.error) as e:
            print(f"== {os.path.basename(path)}: skipped ({e})")
            continue
        results.append(r)
        print(f"== {r['image']}: {r['lea_count']} RIP-relative lea, "
              f"{len(r['guid_candidates'])} GUID-shaped referenced constants, "
              f"{len(r['queue_descs'])} referenced queue-desc-shaped windows")
        for c in r["guid_candidates"]:
            names = ", ".join(c["known"]) if c["known"] else "UNKNOWN"
            print(f"   {c['value']}  x{c['refs']}  at {c['section']} rva 0x{c['rva']:x}  {names}")
        for w in r["iat_guid_loads"]:
            print(f"   IAT GUID load at rva 0x{w['rva']:x}: {w['dll']}!{w['name']}")
        for d in r["queue_descs"]:
            print(f"   queue-desc-shaped window at rva 0x{d['rva']:x}: Type={d['type']} "
                  f"Priority={d['priority']} Flags={d['flags']} NodeMask={d['node']}")
        for line in r["imports_of_note"]:
            print(f"   GUID-named imports: {line}")
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"known": len(known), "images": results}, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
