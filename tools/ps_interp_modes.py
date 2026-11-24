"""Decode pixel-shader interpolation modes from a DXBC container and its FSSM sidecar.

The translator declares every pixel-shader interpolant as a perspective-centre input
(tools/rdna2spirv.py Gen.attribute; the PS system-input layout comment says the pair is
"the perspective-center one, the only pair this layout models").  Which inputs are
actually centre/centroid/sample and perspective/linear is not in the FSSM header word set:
a scan of every u16 in the first 0x138 bytes of all 12,101 corpus pixel shaders against the
SPI_PS_INPUT_ENA value reconstructed from the DXBC signature finds no matching word (the
only equalities are constants that coincide for the persp-centre-only cohort).  The mode
lives in the DXBC container that is carved next to the FSSM:

PSV0 chunk (writer: dxc lib/DxilContainer/DxilPipelineStateValidation.{h,cpp},
DxilPSVWriter in lib/DxilContainer/DxilContainerAssembler.cpp):

    u32 runtime-info size, runtime info bytes,
    u32 resource count, [u32 bind-info size, bind records],
    u32 string-table size, strings,
    u32 semantic-index entries, u32 per entry,
    u32 signature-element size (16 on the corpus), then input elements, output elements,
    each element 16 bytes:
        u32  semantic-name offset (0 = system value, no string)
        u32  semantic-index offset
        u8   rows
        u8   start row (register)
        u8   cols [3:0], start col [5:4], allocated [6]
        u8   semantic kind      (DXIL::SemanticKind: 0 Arbitrary, 3 Position, 6 ClipDistance,
                                 7 CullDistance, 12 SampleIndex, 13 IsFrontFace, 14 Coverage)
        u8   component type     (DxilProgramSigCompType: 1 UInt32, 3 Float32, 6 Float16)
        u8   interpolation mode (DXIL::InterpolationMode, "must match D3D_INTERPOLATION_MODE":
                                 1 Constant, 2 Linear, 3 LinearCentroid, 4 LinearNoperspective,
                                 5 LinearNoperspectiveCentroid, 6 LinearSample,
                                 7 LinearNoperspectiveSample)
        u8   dynamic mask [3:0] | output stream [5:4]
        u8   reserved

The mode maps onto the barycentric pair the hardware must enable, named by the
SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR fields shared by both registers
(Mesa src/amd/registers/gfx103.json, mm 165580/0x286CC and 165584/0x286D0):

    mode 2  Linear                    -> PERSP_CENTER_ENA   (bit 1)
    mode 3  LinearCentroid            -> PERSP_CENTROID_ENA (bit 2)
    mode 6  LinearSample              -> PERSP_SAMPLE_ENA   (bit 0)
    mode 4  LinearNoperspective       -> LINEAR_CENTER_ENA  (bit 5)
    mode 5  LinearNoperspectiveCentroid-> LINEAR_CENTROID_ENA(bit 6)
    mode 7  LinearNoperspectiveSample -> LINEAR_SAMPLE_ENA  (bit 4)
    mode 1  Constant                  -> none (flat: P0 coefficient)

D3D forces noperspective on SV_Position, so Position rows carry mode 4 (a few 5) everywhere;
an SV_Position delivered through the parameter cache is interpolated with the *linear* pair,
while one read as a system input arrives in the POS_X/Y/Z/W_FLOAT VGPRs after the barycentric
pair.  The FSSM input table itself (tools/fssm_header.py, 8-byte records from 0x138) carries
the read method, not the mode - corpus-wide the record flags agree with the disassembly as
bit 0 (0x01) read through v_interp_mov_f32, bit 1 (0x02) read through 32-bit v_interp_p1/p2,
bit 3 (0x08) 16-bit parameter (Float16 component type), bit 4 (0x10, FLAG_FP16) read through
the *_f16 interpolation opcodes.

Usage:
    python -I ps_interp_modes.py <dxbc file or dir> [--json] [--limit N]

For a directory, one row per pixel shader (stage from index.json) with the PSV0 input
modes, the FSSM table read-method decode and the v_interp usage from a sibling disasm/
directory, then a census tally.  Output is counts and bit layouts only; no game data.
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fssm_header  # noqa: E402

# DXIL::SemanticKind values that reach the shader as system inputs, not parameter-cache
# interpolation (names from dxc include/dxc/DXIL/DxilConstants.h).
SYSTEM_KINDS = {1: "vertex-id", 2: "instance-id", 12: "sample-index", 13: "front-face",
                14: "coverage", 15: "inner-coverage", 25: "barycentrics", 26: "shading-rate"}

# (kind class, mode) -> (class name, SPI_PS_INPUT_ENA bit).  Mode numbers are
# DXIL::InterpolationMode ("must match D3D_INTERPOLATION_MODE"); bit numbers and names are
# Mesa gfx103.json register type SPI_PS_INPUT_ENA.
MODE_CLASS = {
    1: ("flat", 0),
    2: ("persp_center", 1 << 1),
    3: ("persp_centroid", 1 << 2),
    6: ("persp_sample", 1 << 0),
    4: ("linear_center", 1 << 5),
    5: ("linear_centroid", 1 << 6),
    7: ("linear_sample", 1 << 4),
}

COMPONENT_TYPES = {1: "uint", 2: "sint", 3: "float", 4: "uint16", 5: "sint16", 6: "float16"}

VINTERP = re.compile(r"v_interp_(p1_f32|p2_f32|mov_f32|p1ll_f16|p1lv_f16|p2_f16)\S*\s+\S+,\s*(\S+),"
                     r"\s*attr(\d+)\.([xyzw])")


def chunk_table(data):
    """fourcc -> (offset, size) for a DXBC container, or {} when the header does not parse."""
    if len(data) < 32 or data[:4] != b"DXBC":
        return {}
    count = struct.unpack_from("<I", data, 28)[0]
    if 32 + 4 * count > len(data):
        return {}
    out = {}
    for i in range(count):
        off = struct.unpack_from("<I", data, 32 + 4 * i)[0]
        if off + 8 > len(data):
            continue
        out[data[off:off + 4]] = (off, struct.unpack_from("<I", data, off + 4)[0])
    return out


def psv0_inputs(data):
    """PSV0 input signature elements as dicts (row, cols, kind, ctype, mode, name).

    Raises ValueError at the first structural inconsistency (truncated chunk, element size
    below the documented 16 bytes, element array past the chunk end).
    """
    chunks = chunk_table(data)
    if b"PSV0" not in chunks:
        raise ValueError("no PSV0 chunk")
    off, size = chunks[b"PSV0"]
    if off + 8 + size > len(data):
        raise ValueError("PSV0 past end of container")
    p = data[off + 8:off + 8 + size]
    if len(p) < 4:
        raise ValueError("truncated before runtime-info size")
    rt_size = struct.unpack_from("<I", p, 0)[0]
    pos = 4 + rt_size                                    # runtime-info size
    if pos + 4 > len(p):
        raise ValueError("truncated before resource count")
    resources = struct.unpack_from("<I", p, pos)[0]
    pos += 4
    if resources:
        if pos + 4 > len(p):
            raise ValueError("truncated before bind-info size")
        bind = struct.unpack_from("<I", p, pos)[0]
        pos += 4 + bind * resources
    if pos + 4 > len(p):
        raise ValueError("truncated before string table")
    strings = struct.unpack_from("<I", p, pos)[0]
    pos += 4 + strings
    heap = p[pos - strings:pos] if strings <= pos else b""
    if pos + 4 > len(p):
        raise ValueError("truncated before semantic-index table")
    semantic = struct.unpack_from("<I", p, pos)[0]
    pos += 4 + 4 * semantic
    if pos + 4 > len(p):
        raise ValueError("truncated before element size")
    esz = struct.unpack_from("<I", p, pos)[0]
    pos += 4
    if esz < 16:
        raise ValueError("element size below 16")
    r1 = 4 + 24                                          # PSVRuntimeInfo1 follows the 24-byte v0 part
    if rt_size < 32 or r1 + 8 > len(p):
        raise ValueError("runtime info too short for element counts")
    count = p[r1 + 4]
    if pos + esz * count > len(p):
        raise ValueError("element array past chunk end")
    out = []
    for i in range(count):
        b = p[pos + i * esz:pos + (i + 1) * esz]
        noff = struct.unpack_from("<I", b, 0)[0]
        name = ""
        if 0 < noff < len(heap):
            end = heap.find(b"\0", noff)
            if end > noff:
                name = heap[noff:end].decode("latin-1")
        out.append({
            "rows": b[8],
            "row": b[9],
            "cols": b[10] & 0xF,
            "start_col": (b[10] >> 4) & 0x3,
            "kind": b[11],
            "ctype": b[12],
            "mode": b[13],
            "name": name,
        })
    return out


def isg1_elements(data):
    """ISG1 elements as dicts (register, mask, rw, name); the repo's 32-byte layout."""
    chunks = chunk_table(data)
    if b"ISG1" not in chunks:
        return []
    off, size = chunks[b"ISG1"]
    if off + 8 + size > len(data):
        return []
    p = data[off + 8:off + 8 + size]
    if len(p) < 8:
        return []
    count, first = struct.unpack_from("<II", p, 0)
    if first < 8:
        return []
    out = []
    for i in range(count):
        base = first + 32 * i
        if base + 32 > len(p):
            break
        _, noff, sem_idx, sysv, ctype, reg = struct.unpack_from("<6I", p, base)
        name = ""
        if 8 <= noff < len(p):
            end = p.find(b"\0", noff)
            if end > noff:
                name = p[noff:end].decode("latin-1")
        out.append({"register": reg, "semantic_index": sem_idx, "system_value": sysv,
                    "component_type": ctype, "mask": p[base + 24], "rw": p[base + 25],
                    "name": name})
    return out


def interp_class(kind, mode):
    """(class, SPI_PS_INPUT_ENA bit) of one input row, or ("system", 0) for system inputs."""
    if kind in SYSTEM_KINDS:
        return "system", 0
    if mode not in MODE_CLASS:
        return "other", 0
    return MODE_CLASS[mode]


def read_method(flags):
    """FSSM record flags -> the read method the corpus disassembly agrees with."""
    return {
        "mov": bool(flags & 0x01),
        "p1p2": bool(flags & 0x02),
        "param16": bool(flags & 0x08),
        "f16_ops": bool(flags & 0x10),
    }


def disasm_facts(text):
    """attr -> set of read kinds, plus the I/J operand set of the non-mov interpolations."""
    per = {}
    ij = set()
    for line in text.splitlines():
        m = VINTERP.search(line.split(";")[0])
        if not m:
            continue
        op, src, attr = m.group(1), m.group(2), int(m.group(3))
        kind = "mov" if op == "mov_f32" else ("f16" if op.endswith("_f16") else "p1p2")
        per.setdefault(attr, set()).add(kind)
        if kind != "mov":
            ij.add(src)
    return per, ij


def pair_class(rows, table, usage, ij):
    """The one barycentric pair the code reads, as a class name.

    Every corpus pixel shader that interpolates reads the same single pair (v0/v1, plus one
    shader that moves it to v14), so a shader's pair has one class even though
    SPI_PS_INPUT_ENA itself is not stored anywhere in the FSSM:

    persp_center  - a read arbitrary/clip/cull row with a perspective mode: the pair is
                    PERSP_CENTER (centroid/sample would need their own pair, and no shader
                    reads a second pair)
    linear_*      - SV_Position rows are the only read inputs: the position is interpolated
                    through the parameter cache with the linear pair (D3D forces
                    noperspective on SV_Position; mode 5 asks for the centroid variant).
                    A shader that reads any other row interpolates that row instead and
                    takes SV_Position from the system VGPRs after the pair.
    flat_only     - interpolation reads only Constant rows (P0 semantics): any enabled pair
                    gives the same value
    none          - no v_interp at all (system inputs only)
    """
    read = [r for r in rows if r["rw"]]
    persp = [r for r in read if r["kind"] in (0, 6, 7)
             and interp_class(r["kind"], r["mode"])[0] in ("persp_center", "persp_centroid",
                                                           "persp_sample")]
    positions = [r for r in read if r["kind"] == 3]
    others = [r for r in read if r["kind"] != 3]
    interpolates = any(k - {"mov"} for k in usage.values())
    if interpolates and persp:
        return "persp_center"
    if interpolates and positions and not others:
        # the position rows are the only read inputs, so the interpolation reads them
        # through the parameter cache with the linear pair (D3D forces noperspective on
        # SV_Position; mode 5 asks for the centroid variant).  A shader that also reads
        # other rows interpolates those and takes SV_Position from the system VGPRs.
        mode = max(r["mode"] for r in positions)
        return "linear_centroid" if mode == 5 else "linear_center"
    if interpolates:
        return "flat_only"
    if ij:
        return "raw_bary"
    return "none"


def shader_row(dxbc_path, fssm_path, disasm_path):
    """Decode one pixel shader; returns None when the PSV0 chunk is absent/unparsable."""
    with open(dxbc_path, "rb") as f:
        data = f.read()
    try:
        rows = psv0_inputs(data)
    except ValueError:
        return None
    isg = isg1_elements(data)
    for r in rows:
        rmask = (((1 << r["cols"]) - 1) << r["start_col"]) & 0xF
        span = range(r["row"], r["row"] + max(r.get("rows", 1), 1))
        reg_matches = [e for e in isg if e["register"] in span]
        col_matches = [e for e in reg_matches if not rmask or not e["mask"] or (e["mask"] & rmask)]
        matches = col_matches or reg_matches
        rw_val = 0
        for e in matches:
            rw_val |= (e["rw"] & rmask) if (rmask and e["mask"]) else e["rw"]
        r["rw"] = rw_val
        if not r["name"]:
            for e in matches:
                if e["name"]:
                    r["name"] = e["name"]
                    break
        r["class"], r["ena_bit"] = interp_class(r["kind"], r["mode"])
    table = []
    if fssm_path and os.path.isfile(fssm_path):
        with open(fssm_path, "rb") as f:
            table = fssm_header.pixel_inputs(f.read())
    usage, ij = {}, set()
    if disasm_path and os.path.isfile(disasm_path):
        with open(disasm_path, encoding="utf-8", errors="replace") as f:
            usage, ij = disasm_facts(f.read())
    pair = pair_class(rows, table, usage, ij)
    ena = 0
    for r in rows:
        ena |= r["ena_bit"]
    return {
        "name": os.path.splitext(os.path.basename(dxbc_path))[0],
        "inputs": [{k: r[k] for k in ("row", "cols", "kind", "ctype", "mode", "rw", "class",
                                      "name")} for r in rows],
        "table": [{**read_method(t["flags"]), "mask": t["mask"], "flags": t["flags"]}
                  for t in table],
        "usage": {str(a): sorted(k) for a, k in sorted(usage.items())},
        "ij": sorted(ij),
        "pair_class": pair,
        "ena_bits": ena,
    }


def tally(rows):
    """Census over per-shader rows: counts only."""
    inputs = [i for r in rows for i in r["inputs"]]
    live = [i for i in inputs if i["rw"]]
    modes = Counter((i["kind"], i["mode"]) for i in inputs)
    live_modes = Counter((i["kind"], i["mode"]) for i in live)
    non_pc = [i for i in live if i["class"] in ("persp_centroid", "persp_sample",
                                                "linear_center", "linear_centroid",
                                                "linear_sample")]
    method = Counter()
    for r in rows:
        for t in r["table"]:
            method[(t["mov"], t["p1p2"], t["param16"], t["f16_ops"])] += 1
    flag_use = Counter()
    for r in rows:
        for pos, kinds in r["usage"].items():
            t = r["table"][int(pos)] if int(pos) < len(r["table"]) else None
            if t:
                flag_use[(t["mov"], t["p1p2"], t["f16_ops"], tuple(kinds))] += 1
    flag_vs_usage = {}
    for (mov, p1p2, f16ops, kinds), n in sorted(flag_use.items()):
        key = (f"mov={int(mov)} p1p2={int(p1p2)} f16={int(f16ops)}"
               f" -> {'+'.join(kinds) or 'unread'}")
        flag_vs_usage[key] = flag_vs_usage.get(key, 0) + n
    return {
        "shaders": len(rows),
        "inputs": len(inputs),
        "live_inputs": len(live),
        "mode_by_kind": {f"kind{k} mode{m}": n for (k, m), n in sorted(modes.items())},
        "live_mode_by_kind": {f"kind{k} mode{m}": n for (k, m), n in sorted(live_modes.items())},
        "live_non_persp_center_inputs": len(non_pc),
        "shaders_with_non_persp_center": sum(1 for r in rows
                                             if any(i["class"] in ("persp_centroid", "persp_sample",
                                                                   "linear_center", "linear_centroid",
                                                                   "linear_sample") and i["rw"]
                                                    for i in r["inputs"])),
        "pair_class": dict(Counter(r["pair_class"] for r in rows)),
        "ij_sets": {",".join(ij) or "(none)": n
                    for ij, n in Counter(tuple(r["ij"]) for r in rows).items()},
        "table_records": sum(len(r["table"]) for r in rows),
        "table_flags": {f"mov={a} p1p2={b} p16={c} f16={d}": n
                        for (a, b, c, d), n in sorted(method.items())},
        "flag_vs_usage": flag_vs_usage,
    }


def files_of(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for n in sorted(os.listdir(path)):
        if not n.endswith(".dxbc"):
            continue
        candidate = os.path.join(path, n)
        if os.path.isfile(candidate):
            out.append(candidate)
    return out


def _at_least_one(value):
    """An integer option that must be at least 1: 0 and negatives used to export one shader."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {n})")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode PS interpolation modes (PSV0 + FSSM + disasm).")
    ap.add_argument("path", help="a .dxbc file or a directory of them")
    ap.add_argument("--json", action="store_true", help="print one JSON document")
    ap.add_argument("--limit", type=_at_least_one, default=None, help="stop after this many pixel shaders")
    args = ap.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        return 2
    base = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    stages = fssm_header.stage_index(base)
    try:
        paths = files_of(args.path)
    except OSError as e:
        print(f"ERROR: cannot list {args.path}: {e.strerror or e}", file=sys.stderr)
        return 2
    rows = []
    skipped = Counter()
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stages and stages.get(stem) != "ps":
            continue
        fssm = os.path.join(base, "fssm", stem + ".fssm")
        disasm = os.path.join(base, "disasm", stem + ".s")
        try:
            row = shader_row(path, fssm, disasm)
        except OSError as e:
            print(f"ERROR: cannot read {e.filename or path}: {e.strerror or e}", file=sys.stderr)
            return 2
        if row is None:
            skipped["no psv0"] += 1
            continue
        row["stage"] = stages.get(stem)
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break

    if args.json:
        json.dump({"files": rows, "tally": tally(rows), "skipped": dict(skipped)},
                  sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for r in rows:
        ins = ", ".join(f"{i['name'] or 'kind' + str(i['kind'])}@{i['row']}"
                        f":{COMPONENT_TYPES.get(i['ctype'], i['ctype'])}"
                        f"/mode{i['mode']}{'/rw' if i['rw'] else ''}" for i in r["inputs"])
        print(f"{r['name']}: pair={r['pair_class']} ij={','.join(r['ij']) or '-'}")
        if ins:
            print(f"  inputs: {ins}")
    t = tally(rows)
    print(f"{t['shaders']} shaders, {t['inputs']} inputs ({t['live_inputs']} read), "
          f"{t['table_records']} FSSM records")
    print("pair classes:", ", ".join(f"{k}:{v}" for k, v in sorted(t["pair_class"].items())))
    print("I/J operand sets:", ", ".join(f"{k}:{v}" for k, v in sorted(t["ij_sets"].items())))
    print("modes (kind, mode):", ", ".join(f"{k}:{v}" for k, v in sorted(t["mode_by_kind"].items())))
    print("live modes:", ", ".join(f"{k}:{v}" for k, v in sorted(t["live_mode_by_kind"].items())))
    print(f"live non-perspective-centre inputs: {t['live_non_persp_center_inputs']} "
          f"in {t['shaders_with_non_persp_center']} shaders")
    if skipped:
        print("skipped:", ", ".join(f"{k}:{v}" for k, v in skipped.items()))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
