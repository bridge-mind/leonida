r"""Which D3D12 interfaces can a PE image ask for? (D3D-350, stdlib only)

A client that creates a D3D12 object passes the interface's IID (`IID_PPV_ARGS`), so the 16 GUID bytes are in the
caller's image unless it imports them as data (`d3d12_xs.dll` exports IIDs; see docs/d3d12_xs_surface.md section
2). This tool reads every `MIDL_INTERFACE("...") ID3D12...` declaration from the Windows SDK's d3d12.h and reports
which of those IIDs occur, little-endian as in memory, anywhere in each image. An interface that is absent was
never created or queried by that image through a constant of its own.

docs/d3d12_command_objects.md section 2.1 uses it to show that no DLL in the package can call the device's queue,
allocator, command-list or fence creators (slots 8, 9, 12 and 36). The interfaces that are found act as the
positive control: they match the objects the image is known to create.

It prints interface names and counts only, no bytes of the image.

Usage:
    python -I tools/d3d12_iid_scan.py <image> [<image> ...] [--header PATH/d3d12.h] [--json OUT]

Without --header it takes d3d12.h from the newest Windows SDK under "Windows Kits\10\Include".
It also lists what each image imports from d3d12_xs.dll and warns when one of those imports could be IID data (an
IID_* name, or an ordinal in the GUID range 23-115), because an IID taken from there would not show up here.
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

# The creators whose result interfaces docs/d3d12_command_objects.md section 2.1 asks about.
FOCUS = ("ID3D12CommandQueue", "ID3D12CommandAllocator", "ID3D12CommandList", "ID3D12GraphicsCommandList",
         "ID3D12Fence", "ID3D12Fence1")


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


def d3d12_xs_imports(data: bytes) -> list[str] | None:
    """What the PE image imports from d3d12_xs.dll: names, or "#N" for an import by ordinal (ordinals 23-115 are
    GUID data, docs/d3d12_xs_surface.md section 2). None when the image does not import d3d12_xs.dll at all."""
    try:
        if data[:2] != b"MZ":
            return None
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return None
        sections, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        pe32plus = struct.unpack_from("<H", data, opt)[0] == 0x20B
        dirs = opt + (112 if pe32plus else 96)
        imp_rva = struct.unpack_from("<I", data, dirs + 8)[0]
        table = opt + opt_size

        def offset(rva: int) -> int | None:
            for i in range(sections):
                s = table + 40 * i
                vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
                span = min(vsize, rsize) if vsize else rsize
                if va <= rva < va + span:
                    pos_in_file = raw + rva - va
                    return pos_in_file if pos_in_file < len(data) else None
            return None

        def cstr(at: int) -> bytes:
            return data[at:data.index(b"\0", at)]

        names: list[str] | None = None
        pos = offset(imp_rva) if imp_rva else None
        while pos is not None and pos + 20 <= len(data):
            lookup_rva, _, _, name_rva, iat_rva = struct.unpack_from("<IIIII", data, pos)
            if name_rva == 0:
                break
            at = offset(name_rva)
            if at is not None and cstr(at).lower() == b"d3d12_xs.dll":
                if names is None:
                    names = []
                thunk = offset(lookup_rva or iat_rva)
                width, flag = (8, 1 << 63) if pe32plus else (4, 1 << 31)
                while thunk is not None:
                    value = struct.unpack_from("<Q" if pe32plus else "<I", data, thunk)[0]
                    if value == 0:
                        break
                    if value & flag:
                        names.append(f"#{value & 0xFFFF}")
                    else:
                        hint = offset(value & 0x7FFFFFFF)
                        names.append(cstr(hint + 2).decode("ascii", "replace") if hint is not None else "?")
                    thunk += width
            pos += 20
        return names
    except (struct.error, ValueError):
        return None


def possible_iid_imports(names: list[str] | None) -> list[str]:
    """The d3d12_xs.dll imports that could be IID data: IID_* names and ordinals in the GUID range 23-115."""
    out = []
    for n in names or []:
        if n.startswith("IID_") or (n.startswith("#") and n[1:].isdigit() and 23 <= int(n[1:]) <= 115):
            out.append(n)
    return out


def scan(path: str, iids: dict[str, uuid.UUID]) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    found = sorted(name for name, guid in iids.items() if guid.bytes_le in data)
    imports = d3d12_xs_imports(data)
    return {
        "image": os.path.basename(path),
        "found": found,
        "focus": {name: name in found for name in FOCUS if name in iids},
        "d3d12_xs_imports": imports,
        "possible_iid_imports": possible_iid_imports(imports),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--header", help="d3d12.h to read the IIDs from (default: newest Windows SDK)")
    ap.add_argument("--json", help="also write the result here")
    args = ap.parse_args(argv)
    if args.json:
        # The report is written last, after every input has been read, so an aliased input would
        # silently be replaced by it. Refuse the alias before anything is opened.
        for what, other in (("--header", args.header), *(("an input image", p) for p in args.images)):
            if other and same_report_path(args.json, other):
                print(f"ERROR: --json would overwrite {what}: {other}", file=sys.stderr)
                return 2
    header = args.header or default_header()
    if not header or not os.path.isfile(header):
        print("ERROR: d3d12.h not found; pass --header", file=sys.stderr)
        return 2
    try:
        iids = read_iids(header)
    except OSError as e:
        print(f"ERROR: cannot read {header}: {e.strerror or e}", file=sys.stderr)
        return 2
    if not iids:
        print(f"ERROR: no ID3D12 MIDL_INTERFACE declarations in {header}", file=sys.stderr)
        return 2
    print(f"{len(iids)} ID3D12 IIDs from {header}")
    results = []
    for path in args.images:
        try:
            r = scan(path, iids)
        except OSError as e:  # a missing, locked or folder image used to end the run in a traceback
            print(f"ERROR: cannot read {path}: {e.strerror or e}", file=sys.stderr)
            return 2
        results.append(r)
        absent = [n for n, present in r["focus"].items() if not present]
        print(f"== {r['image']}: {len(r['found'])} found")
        print("  found:  " + (", ".join(r["found"]) or "none"))
        print("  absent from the queue/allocator/list/fence set: " + (", ".join(absent) or "none"))
        if r["d3d12_xs_imports"] is not None:
            print("  imports from d3d12_xs.dll: " + (", ".join(r["d3d12_xs_imports"]) or "none"))
            if r["possible_iid_imports"]:
                print("  WARNING: these d3d12_xs.dll imports may be IID data, so an absent IID proves nothing for "
                      "this image: " + ", ".join(r["possible_iid_imports"]))
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"header": os.path.basename(header), "iids": len(iids), "images": results}, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {args.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
