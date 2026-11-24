"""Unpack the native compute programs in FSPX payloads: the ACCC part of the shader containers that the package's
FidelityFX upscaler DLL (amd_fidelityfx_upscaler_xbox.dll) carries instead of DXIL. Layout and meaning in
docs/fspx.md.

    DXBC container (SFI0 ISG1 OSG1 PSV0 RTS0 HASH ACCC, no DXIL)
      ACCC part = FSPX payload
        u32 'FSPX', u32 version, u32 entry count, u32 total size
        count x {u32 kind, u32 offset, u32 size, u32 0}    offsets from the start of the payload
          kind 0  dispatch state (thread-group size, wave size, COMPUTE_PGM_RSRC1/2)
          kind 1  20 bytes, meaning unknown (optional)
          kind 6  the RDNA2 program
          kind 7  resource words and the container's shader hash (optional)

`parse_fspx` follows the DLL's own reader (RVA 0x1a20), including its status codes: 1 too short, 2 not FSPX,
4 directory longer than the payload, 5 kind 0 or kind 6 missing or empty. An entry outside the payload is skipped,
as the DLL skips it. The DLL does not check the version word.

Usage:
    python -I fspx_unpack.py <file> [<file> ...] --out DIR [--disasm] [--mcpu gfx1030]
<file> is the DLL, a .dxbc container or a bare FSPX payload. Writes, per shader, the same layout the game's shader
corpus uses, so tools/rdna2spirv.py can read it:
    DIR/<name>.dxbc          the container (absent for a bare payload)
    DIR/index.json           [{"file": "<name>.dxbc", "stage": "cs", ...}]
    DIR/fspx/<name>.json     the decoded kind-0 entry, the directory and the code range
    DIR/fspx/<name>.code     the kind-6 program
    DIR/fspx/<name>.state    the kind-0 entry as it is (its SHA-256 is the index's fspx_state_sha256)
    DIR/disasm/<name>.s      with --disasm: tools/shader_disasm.py output for the code after the prologue
"""
import argparse
import json
import os
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import path_key, same_report_path  # noqa: E402

MAGIC = b"FSPX"
DXBC = b"DXBC"
ENDPGM = struct.pack("<I", 0xBF810000)
ENTRY_SIZE = 16
HEADER_SIZE = 16
KINDS = (0, 1, 6, 7)
# The program starts with three 12-byte VOP3 words (opcode 0x180, bits 31..16 = 0xD580) with literals that carry the
# kind-7 hash; llvm-mc rejects them. Measured on all 29 shaders of the 1.0.2.1 DLL (docs/fspx.md).
PROLOGUE_WORDS = (0, 3, 6)
PROLOGUE_SIZE = 36
STAGE_COMPUTE = 5  # PSVRuntimeInfo1 ShaderStage (DirectX): 5 = compute


def inside_folder(path: str, folder: str) -> bool:
    """True when path is folder or below it, by the resolved and normcased name report_paths compares."""
    key, root = path_key(path), path_key(folder)
    try:
        return os.path.commonpath((key, root)) == root
    except ValueError:  # different drives
        return False


def parse_fspx(data: bytes):
    """(status, {kind: (offset, size)}) with the DLL's status codes; 0 means usable."""
    if len(data) < HEADER_SIZE:
        return 1, {}
    if data[:4] != MAGIC:
        return 2, {}
    count, total = struct.unpack_from("<II", data, 8)
    total = min(total, len(data))
    end_dir = HEADER_SIZE + ENTRY_SIZE * count
    if total < end_dir:
        return 4, {}
    entries = {}
    for k in range(count):
        kind, off, size = struct.unpack_from("<III", data, HEADER_SIZE + ENTRY_SIZE * k)
        if off < end_dir or off + size > total:
            continue  # outside the payload: the DLL ignores the entry
        if kind in KINDS:
            entries[kind] = (off, size)
    if not entries.get(0, (0, 0))[1] or not entries.get(6, (0, 0))[1]:
        return 5, entries
    return 0, entries


def dxbc_parts(c: bytes) -> dict:
    """Part tag -> payload of a DXBC container, or {} when the header doesn't fit."""
    if len(c) < 32 or c[:4] != DXBC:
        return {}
    total, nparts = struct.unpack_from("<II", c, 24)
    if total > len(c) or 32 + 4 * nparts > len(c):
        return {}
    out = {}
    for k in range(nparts):
        po = struct.unpack_from("<I", c, 32 + 4 * k)[0]
        if po + 8 > len(c):
            return {}
        size = struct.unpack_from("<I", c, po + 4)[0]
        if po + 8 + size > len(c):
            return {}
        out[c[po:po + 4]] = c[po + 8:po + 8 + size]
    return out


def find_containers(data: bytes):
    """Every DXBC container in `data` that has an ACCC part holding an FSPX payload, as (offset, bytes)."""
    found, i = [], 0
    while True:
        i = data.find(DXBC, i)
        if i < 0 or i + 32 > len(data):
            return found
        total, nparts = struct.unpack_from("<II", data, i + 24)
        if 32 < total <= len(data) - i and 0 < nparts < 64:
            c = data[i:i + total]
            accc = dxbc_parts(c).get(b"ACCC", b"")
            if accc[:4] == MAGIC:
                found.append((i, c))
                i += total
                continue
        i += 4


def decode_kind0(b: bytes) -> dict:
    """The kind-0 words and the fields docs/fspx.md describes. `threads` is measured against PSV0; the rest is the
    reading that fits the code (wave size, RSRC1/RSRC2), see docs/fspx.md for how far each one was checked."""
    w = struct.unpack_from("<7I", b) if len(b) >= 28 else ()
    if not w:
        return {"words": list(struct.unpack_from(f"<{len(b) // 4}I", b))}
    rsrc1, rsrc2 = w[5], w[6]
    wave32 = bool((w[4] >> 20) & 1)
    return {
        "words": list(w),
        "threads": [w[3] & 0xFFFF, w[3] >> 16, w[4] & 0xFFFF],
        "wave32": wave32,
        "rsrc1": rsrc1,
        "rsrc2": rsrc2,
        "vgprs": ((rsrc1 & 63) + 1) * (8 if wave32 else 4),
        "user_sgprs": (rsrc2 >> 1) & 31,
        "tgid_enable": [(rsrc2 >> b) & 1 for b in (7, 8, 9)],
        "tidig_comp_cnt": (rsrc2 >> 11) & 3,
        "lds_bytes": ((rsrc2 >> 15) & 0x1FF) * 512,
    }


def code_range(code: bytes):
    """(start, end) of the instructions in a kind-6 program: after the 36-byte prologue when it is there, through
    the first s_endpgm. end is None when the program has no s_endpgm."""
    words = struct.unpack_from(f"<{len(code) // 4}I", code)
    start = PROLOGUE_SIZE if len(words) > 8 and all(words[k] >> 16 == 0xD580 for k in PROLOGUE_WORDS) else 0
    pos = code.find(ENDPGM, start)
    while pos >= 0 and pos % 4:
        pos = code.find(ENDPGM, pos + 1)
    return start, (pos + 4 if pos >= 0 else None)


def psv_stage(container_parts: dict):
    """PSVRuntimeInfo1 ShaderStage, or None (PSV0 = u32 size of the runtime info, then the info)."""
    psv = container_parts.get(b"PSV0", b"")
    if len(psv) < 4:
        return None
    size = struct.unpack_from("<I", psv, 0)[0]
    return psv[4 + 24] if size > 24 and len(psv) > 4 + 24 else None


def unpack_payload(payload: bytes):
    status, entries = parse_fspx(payload)
    if status:
        return None, status
    blob = {k: payload[o:o + s] for k, (o, s) in entries.items()}
    start, end = code_range(blob[6])
    meta = {
        "version": struct.unpack_from("<I", payload, 4)[0],
        "directory": {str(k): [o, s] for k, (o, s) in sorted(entries.items())},
        "code": {"start": start, "end": end, "prologue": start == PROLOGUE_SIZE},
        **decode_kind0(blob[0]),
    }
    return (meta, blob), 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unpack the FSPX compute programs in FidelityFX shader containers.")
    ap.add_argument("files", nargs="+", help="the DLL, .dxbc containers or bare FSPX payloads")
    ap.add_argument("--out", required=True)
    ap.add_argument("--disasm", action="store_true", help="also write disasm/<name>.s (needs llvm-mc, see LLVM_MC)")
    ap.add_argument("--mcpu", default="gfx1030",
                    help="llvm-mc CPU for --disasm (default gfx1030: the programs use v_dot4 instructions, which "
                         "gfx1010 and gfx1013 do not decode)")
    a = ap.parse_args(argv)
    # An output in the input folder can replace a .dxbc input and, when the input is the folder itself,
    # os.makedirs writes there before open() tries to read the folder as a file. Refuse before either.
    for path in a.files:
        folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        if (same_report_path(a.out, path) or same_report_path(a.out, folder)
                or (os.path.isdir(path) and inside_folder(a.out, folder))):
            print(f"ERROR: --out {a.out} collides with the input folder {folder}; give another output folder",
                  file=sys.stderr)
            return 2
    fspx_dir = os.path.join(a.out, "fspx")
    try:
        os.makedirs(fspx_dir, exist_ok=True)
    except OSError as error:
        print(f"cannot create output directory {fspx_dir}: {error}", file=sys.stderr)
        return 2
    disasm = None
    if a.disasm:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import shader_disasm  # noqa: E402  (resolves LLVM_MC at import)
        disasm = shader_disasm
        disasm_dir = os.path.join(a.out, "disasm")
        try:
            os.makedirs(disasm_dir, exist_ok=True)
        except OSError as error:
            print(f"cannot create output directory {disasm_dir}: {error}", file=sys.stderr)
            return 2
    index, bad = [], 0
    used_names = set()
    for path in a.files:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as error:
            print(f"{path}: cannot read input: {error}", file=sys.stderr)
            bad += 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if data[:4] == MAGIC:
            items = [(0, None, data)]
        else:
            items = [(off, c, dxbc_parts(c)[b"ACCC"]) for off, c in find_containers(data)]
        if not items:
            print(f"{path}: no FSPX payload found", file=sys.stderr)
            bad += 1
            continue
        for n, (off, container, payload) in enumerate(items):
            name = f"{stem}_{n:02d}" if len(items) > 1 else stem
            got, status = unpack_payload(payload)
            if not got:
                print(f"{name}: FSPX status {status}", file=sys.stderr)
                bad += 1
                continue
            base_name = name
            number = 1
            while name.casefold() in used_names:
                number += 1
                name = f"{base_name}_{number}"
            used_names.add(name.casefold())
            meta, blob = got
            stage = psv_stage(dxbc_parts(container)) if container else None
            meta["source"] = {"file": os.path.basename(path), "offset": off}
            entry = {"file": name + ".dxbc", "stage": "cs" if stage in (None, STAGE_COMPUTE) else f"psv{stage}",
                     "source_offset": off}
            if "threads" in meta:
                entry["threads"] = meta["threads"]
            if "wave32" in meta:
                entry["wave32"] = meta["wave32"]
            listing = None
            if disasm and meta["code"]["end"]:
                start, end = meta["code"]["start"], meta["code"]["end"]
                asm, invalid, err = disasm.disasm(blob[6][start:end], a.mcpu)
                if not asm.strip():
                    reason = (err.strip().splitlines() or ["no output"])[0]
                    print(f"{name}: {reason}", file=sys.stderr)
                    bad += 1
                    continue
                entry["invalid_words"] = invalid
                listing = f"; {name}  code 0x{start:x}..0x{end:x} (fspx), {a.mcpu}, {invalid} invalid words\n" + asm
                if err.strip():
                    listing += "\n; --- llvm-mc messages ---\n; " + err.strip().replace("\n", "\n; ") + "\n"
            if container:
                with open(os.path.join(a.out, name + ".dxbc"), "wb") as f:
                    f.write(container)
            with open(os.path.join(a.out, "fspx", name + ".code"), "wb") as f:
                f.write(blob[6])
            with open(os.path.join(a.out, "fspx", name + ".state"), "wb") as f:
                f.write(blob[0])
            with open(os.path.join(a.out, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, indent=1)
            if listing is not None:
                with open(os.path.join(a.out, "disasm", name + ".s"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(listing)
            index.append(entry)
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    print(f"{len(index)} FSPX program(s) unpacked to {a.out}" + (f", {bad} input(s) failed" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
