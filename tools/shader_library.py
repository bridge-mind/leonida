"""Unpack the pipeline library in shaders.rpf: the zlib part of the archive that tools/shader_carve.py does not read.

Behind the 26,503 raw-deflate shader entries, shaders.rpf holds one more file, stored as 64 KiB zlib blocks laid end
to end. It is a library of precompiled pipelines: graphics and compute pipeline records that name their programs per
stage, a table of state blobs and a table of native RDNA2 programs. Layout (docs/shader_library.md):

    0x00  'PS' 00 09, a name ("0x65F41EDD") and a version string ("930.048.020")
    0x48  u64 hash
    0x50  u32 graphics pipelines G, u32 compute pipelines C, u32 blobs B, u32 programs P
    0x60  u32 blob section size, u32 code section size
    0x80  G x 0x70  graphics:  u64 hash, u32 state blob, u32 input-layout blob, 40 bytes, 7 x (u32 program, u32 blob)
          C x 24    compute:   u64 hash, u32 state blob, u32 blob, u32 program, u32 blob
          B x 8     (offset, size) into the blob section
          P x 8     (offset, size) into the code section, 256-byte aligned
          blob section, then code section; the file ends exactly there

A missing stage is program 0xFFFFFFFF. Stage slots: 0 pixel, 1 vertex, 2 mesh (6 is used once and not identified).
A compute state blob has the 28-byte layout of an FSPX kind-0 entry (tools/fspx_unpack.py), and every program starts
with the same 36-byte hash prologue as an FSPX program, so the programs are written in the layout fspx_unpack.py uses
and tools/rdna2spirv.py reads (fspx/<name>.code, fspx/<name>.json, index.json).

Usage:
    python -I shader_library.py <shaders.rpf | library.bin> --out DIR [--stages cs,ps,vs,ms] [--disasm]
                                [--mcpu gfx1013] [--jobs N]
    python -I shader_library.py <shaders.rpf> --save library.bin       # the reassembled file, nothing else

DIR/library.json gets the header, both pipeline tables (stage -> program name) and the per-program stage. Programs
are named lib_<stage>_<index>. The output is game data and must stay in the user's work folder.
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fspx_unpack  # noqa: E402

MAGIC = b"PS\x00\x09"
HEADER = 0x80
GFX_SIZE = 0x70
CS_SIZE = 24
NONE = 0xFFFFFFFF
SLOTS = {0: "ps", 1: "vs", 2: "ms", 6: "slot6"}
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
BLOCK = 1 << 16


class LibraryError(ValueError):
    pass


def header(b: bytes) -> dict:
    if len(b) < HEADER or b[:4] != MAGIC:
        raise LibraryError("not a pipeline library (no 'PS' 00 09 header)")
    g, c, nb, npg, bsize, csize = struct.unpack_from("<6I", b, 0x50)
    return {
        "name": b[4:0x24].split(b"\0")[0].decode("latin-1"),
        "version": b[0x24:0x48].split(b"\0")[0].decode("latin-1"),
        "hash": f"{struct.unpack_from('<Q', b, 0x48)[0]:016x}",
        "graphics": g, "compute": c, "blobs": nb, "programs": npg,
        "blob_bytes": bsize, "code_bytes": csize,
    }


def layout(h: dict) -> dict:
    """Section offsets for a header; 'end' is the size the file must have."""
    gfx = HEADER
    cs = gfx + h["graphics"] * GFX_SIZE
    blobs = cs + h["compute"] * CS_SIZE
    progs = blobs + h["blobs"] * 8
    blob_data = progs + h["programs"] * 8
    code = blob_data + h["blob_bytes"]
    return {"graphics": gfx, "compute": cs, "blob_table": blobs, "program_table": progs, "blob_data": blob_data,
            "code": code, "end": code + h["code_bytes"]}


def find_library(m) -> bytes:
    """Reassemble the library from the archive's zlib blocks: the first stream that inflates to the header starts
    it, and the following streams are appended until the size the header gives."""
    pos, out, need, start = 0, None, None, None
    while pos < len(m):
        mt = ZHDR.search(m, pos)
        if not mt:
            break
        s = mt.start()
        d = zlib.decompressobj()
        inp = m[s:s + 2 * BLOCK]   # a 64 KiB block never deflates to more than this
        try:
            data = d.decompress(inp) + d.flush()
        except zlib.error:
            pos = s + 1
            continue
        if not d.eof:
            pos = s + 1
            continue
        used = len(inp) - len(d.unused_data)
        if out is None:
            if data[:4] == MAGIC and len(data) >= HEADER:
                need, out, start = layout(header(data))["end"], bytearray(data), s
        else:
            out += data
        pos = s + used
        if out is not None and len(out) >= need:
            break
    if out is None:
        raise LibraryError("no pipeline library found")
    if len(out) != need:
        raise LibraryError(f"library at 0x{start:x} is {len(out)} bytes, the header gives {need}")
    return bytes(out)


def table(b: bytes, off: int, count: int):
    return [struct.unpack_from("<II", b, off + 8 * i) for i in range(count)]


def parse(b: bytes) -> dict:
    h = header(b)
    lay = layout(h)
    if lay["end"] != len(b):
        raise LibraryError(f"file is {len(b)} bytes, the header gives {lay['end']}")
    blobs = table(b, lay["blob_table"], h["blobs"])
    progs = table(b, lay["program_table"], h["programs"])
    for name, tab, size in (("blob", blobs, h["blob_bytes"]), ("program", progs, h["code_bytes"])):
        for i, (o, s) in enumerate(tab):
            if o + s > size:
                raise LibraryError(f"{name} {i} ends past its section")
    graphics, compute = [], []
    stage, links = {}, []

    def ref(i, what):
        if i != NONE and not i < (h["programs"] if what == "program" else h["blobs"]):
            raise LibraryError(f"{what} index {i} out of range")
        return None if i == NONE else i

    for i in range(h["graphics"]):
        r = b[lay["graphics"] + i * GFX_SIZE: lay["graphics"] + (i + 1) * GFX_SIZE]
        hsh, state, layout_blob = struct.unpack_from("<QII", r)
        pairs = struct.unpack_from("<14I", r, 56)
        rec = {"hash": f"{hsh:016x}", "state": ref(state, "blob"), "input_layout": ref(layout_blob, "blob"),
               "digest": r[16:56].hex(), "stages": {}}
        for slot in range(7):
            p, blob = ref(pairs[2 * slot], "program"), ref(pairs[2 * slot + 1], "blob")
            if p is None:
                continue
            st = SLOTS.get(slot, f"slot{slot}")
            rec["stages"][st] = {"program": p, "blob": blob}
            stage.setdefault(p, set()).add(st)
            if blob is not None:
                links.append((p, blob))
        graphics.append(rec)
    for i in range(h["compute"]):
        hsh, state, blob1, p, blob2 = struct.unpack_from("<QIIII", b, lay["compute"] + i * CS_SIZE)
        prog_i, layout_i = ref(p, "program"), ref(blob2, "blob")
        compute.append({"hash": f"{hsh:016x}", "state": ref(state, "blob"), "blob": ref(blob1, "blob"),
                        "program": prog_i, "layout": layout_i})
        if prog_i is not None:
            stage.setdefault(prog_i, set()).add("cs")
            if layout_i is not None:
                links.append((prog_i, layout_i))
    digest = {}
    for p, i in links:
        o, size = blobs[i]
        d = stage_digest(b[lay["blob_data"] + o: lay["blob_data"] + o + size])
        if d is not None:
            digest.setdefault(p, set()).add(d)
    return {"header": h, "layout": lay, "blobs": blobs, "programs": progs, "graphics": graphics, "compute": compute,
            "stage": stage, "digest": digest}


STAGE_HEAD = {2: 16, 3: 24}   # resource-layout blob: word 0 kind (2 graphics stage, 3 compute) -> head size


def stage_digest(y: bytes):
    """The 16-byte digest a stage's resource-layout blob ends with, or None. Layout: a head of STAGE_HEAD[kind]
    bytes whose last word is tagged 0x40, 32-byte records tagged 0x41 or 0x42, then a word 0x43 and the digest (the
    shape of an FSPX kind-7 entry, whose digest is its container's HASH part, docs/fspx.md)."""
    head = STAGE_HEAD.get(y[0]) if y else None
    if head is None or len(y) < head or y[head - 4] != 0x40:
        return None
    o = head
    while o + 4 <= len(y):
        tag = struct.unpack_from("<I", y, o)[0]
        if tag in (0x41, 0x42):
            o += 32
            continue
        return y[o + 4:o + 20] if tag == 0x43 and o + 20 <= len(y) else None
    return None


def carve_digests(folder: str) -> dict:
    """digest -> [(shader name, container path)] over a tools/shader_carve.py output folder (its index.json and
    .dxbc files)."""
    import shader_carve
    with open(os.path.join(folder, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    out = {}
    for e in index:
        try:
            with open(os.path.join(folder, e["file"]), "rb") as f:
                c = f.read()
            parts = shader_carve.parts(c)
        except (OSError, ValueError):
            continue
        if "HASH" in parts and parts["HASH"][1] >= 20:
            o = parts["HASH"][0]
            out.setdefault(c[o + 4:o + 20], []).append((e["name"], os.path.join(folder, e["file"])))
    return out


def blob(b: bytes, lib: dict, i: int) -> bytes:
    o, s = lib["blobs"][i]
    base = lib["layout"]["blob_data"]
    return b[base + o: base + o + s]


def program(b: bytes, lib: dict, i: int) -> bytes:
    o, s = lib["programs"][i]
    base = lib["layout"]["code"]
    return b[base + o: base + o + s]


ENDPGM = struct.pack("<I", 0xBF810000)


def code_range(code: bytes):
    """(start, end) of a program's instructions: after the FSPX-style prologue (fspx_unpack.code_range), through the
    last 4-byte-aligned s_endpgm, as for FSSM blobs (tools/shader_disasm.py). Ending at the first one cut off the
    code after it: 579 of the first 3,000 pixel programs have more than one, and branches reach the later code."""
    start, first = fspx_unpack.code_range(code)
    if first is None:
        return start, None
    last = code.rfind(ENDPGM)
    while last >= first and (last - start) % 4:
        last = code.rfind(ENDPGM, 0, last)
    return start, max(first, last + 4)


def program_name(i: int, stages) -> str:
    st = "+".join(sorted(stages)) if stages else "unused"
    return f"lib_{st}_{i:05d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unpack the pipeline library stored in shaders.rpf.")
    ap.add_argument("src", help="shaders.rpf, or a library file written with --save")
    ap.add_argument("--out", help="output folder (library.json, index.json, fspx/, disasm/)")
    ap.add_argument("--save", help="write the reassembled library file here")
    ap.add_argument("--stages", default="cs,ps,vs,ms",
                    help="stages whose programs to write (default cs,ps,vs,ms; 'all' adds the unidentified slot)")
    ap.add_argument("--disasm", action="store_true", help="also write disasm/<name>.s (needs llvm-mc, see LLVM_MC)")
    ap.add_argument("--mcpu", default="gfx1013", help="llvm-mc CPU for --disasm (default gfx1013)")
    ap.add_argument("--carve", help="a tools/shader_carve.py output folder: name each program's source shader by the "
                                    "digest its resource-layout blob ends with (the container's HASH part), and "
                                    "copy that container to <name>.dxbc for the translator's signature reads")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="llvm-mc processes for --disasm (default: half the CPUs)")
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if not a.out and not a.save:
        ap.error("give --out, --save or both")
    with open(a.src, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if head == MAGIC:
            b = f.read()
        else:
            if os.path.getsize(a.src) == 0:
                raise LibraryError("empty input")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                b = find_library(m)
            finally:
                m.close()
    lib = parse(b)
    if a.save:
        with open(a.save, "wb") as f:
            f.write(b)
    if not a.out:
        print(f"library: {len(b)} bytes written to {a.save}")
        return 0
    want = None if a.stages == "all" else set(a.stages.split(","))
    os.makedirs(os.path.join(a.out, "fspx"), exist_ok=True)
    disasm = None
    if a.disasm:
        import shader_disasm  # noqa: E402  (resolves LLVM_MC at import)
        disasm = shader_disasm
        os.makedirs(os.path.join(a.out, "disasm"), exist_ok=True)
    names = {i: program_name(i, lib["stage"].get(i)) for i in range(lib["header"]["programs"])}
    sources = carve_digests(a.carve) if a.carve else None
    cs_state = {c["program"]: c["state"] for c in lib["compute"]}
    index, written, work = [], 0, []
    for i in range(lib["header"]["programs"]):
        stages = lib["stage"].get(i, set())
        st = "+".join(sorted(stages)) if stages else "unused"
        if want is not None and not stages & want:
            continue
        code = program(b, lib, i)
        start, end = code_range(code)
        meta = {"program": i, "stage": st, "offset": lib["programs"][i][0], "size": len(code),
                "code": {"start": start, "end": end, "prologue": start == fspx_unpack.PROLOGUE_SIZE}}
        if i in cs_state:
            meta.update(fspx_unpack.decode_kind0(blob(b, lib, cs_state[i])))
        name = names[i]
        entry = {"file": name + ".dxbc", "stage": st, "program": i}
        digests = sorted(lib["digest"].get(i, ()))
        meta["digest"] = [d.hex() for d in digests]
        if sources is not None:
            found = sorted({src for d in digests for src in sources.get(d, ())})
            entry["source"] = meta["source"] = sorted({n for n, _ in found})
            if found:
                # The source container's signatures and PSV0 describe the stage's D3D12 interface, which the
                # pipeline build keeps; the translator reads them from <name>.dxbc next to its folder.
                with open(found[0][1], "rb") as f:
                    container = f.read()
                with open(os.path.join(a.out, name + ".dxbc"), "wb") as f:
                    f.write(container)
        for k in ("threads", "wave32"):
            if k in meta:
                entry[k] = meta[k]
        with open(os.path.join(a.out, "fspx", name + ".code"), "wb") as f:
            f.write(code)
        with open(os.path.join(a.out, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=1)
        if disasm and end:
            work.append((entry, name, code[start:end], start, end))
        index.append(entry)
        written += 1

    def one(job):
        entry, name, code, start, end = job
        asm, invalid, err = disasm.disasm(code, a.mcpu)
        entry["invalid_words"] = invalid
        with open(os.path.join(a.out, "disasm", name + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  code 0x{start:x}..0x{end:x} (fspx), {a.mcpu}, {invalid} invalid words\n")
            f.write(asm)
            if err.strip():
                f.write("\n; --- llvm-mc messages ---\n; " + err.strip().replace("\n", "\n; ") + "\n")

    if work:
        with ThreadPoolExecutor(a.jobs) as ex:   # each job waits on its own llvm-mc process
            list(ex.map(one, work))

    def named(stages):
        return {k: names[v["program"]] for k, v in stages.items()}

    summary = {
        "header": lib["header"],
        "layout": lib["layout"],
        "graphics": [{"hash": g["hash"], "state": g["state"], "input_layout": g["input_layout"],
                      "stages": named(g["stages"])} for g in lib["graphics"]],
        "compute": [{"hash": c["hash"], "state": c["state"], "program": names.get(c["program"])}
                    for c in lib["compute"]],
    }
    with open(os.path.join(a.out, "library.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    h = lib["header"]
    print(f"library {h['name']} {h['version']}: {h['graphics']} graphics + {h['compute']} compute pipelines, "
          f"{h['programs']} programs; {written} written to {a.out}")
    if sources is not None:
        linked = sum(1 for e in index if e["source"])
        print(f"{linked} of {written} named by a carved shader's HASH digest, {written - linked} not in the carve")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    try:
        sys.exit(main())
    except LibraryError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
