r"""Census of the root signatures the package carries next to a shader, and the two packing models (stdlib only).

No game shader carries its root signature: all 26,503 containers in `shaders.rpf` have `SFI0`, `ISG1`, `OSG1`, `PSV0`,
`HASH`, `DXIL` and `PRIV`, and none has `RTS0`, so a game shader cannot say which root parameters it was compiled
against (docs/d3d12_vulkan.md section 3.5). The 29 compute containers in `amd_fidelityfx_upscaler_xbox.dll` do: each
carries `RTS0` beside its `ACCC`/`FSPX` program, and `tools/fspx_unpack.py` decodes that program's
`COMPUTE_PGM_RSRC2`, whose `USER_SGPR` field is the number of user SGPRs the console launched it with (docs/fspx.md,
kind 0). Those 29 are the only place in the package where a real signature and the dword count the driver supplied
sit next to each other.

For each signature this reports two spans:

  * `table2`: the layer's model, `rt::rootsig::pack` in `runtime/src/root_signature.{h,cpp}`: a constant takes
    `NumValues` dwords, a root descriptor and a descriptor table take two (a 64-bit console VA);
  * `table1`: one dword per table pointer, which is the model the corpus census in section 3.5 found in the reads
    (pointer SGPRs adjacent at distance 1, and no 64-bit pointer in any PS/VS/CS/MS).

`--json` carries the numbers and the summary prints the spare dwords per model, so a reader can count how many
signatures each model fits. `--disasm DIR` adds the program's own reads, from a `tools/fspx_unpack.py --disasm`
folder: which user SGPRs it reads, which one it zero-extends into an `s_load` base, and which 64-bit pairs it copies
whole. That is where the tables and the root descriptors surface.

What it does not measure: which parameter a shader reads. A read set needs `tools/shader_disasm.py` (LLVM_MC) and
`tools/rdna2spirv.py` on the program, and the game's own shaders have no signature to compare it with.

Usage:
    python -I tools/rootsig_census.py <dir> [--json OUT]
<dir> is a `tools/fspx_unpack.py --out` folder: `<name>.dxbc` per container and `fspx/<name>.json` per program.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
from collections import Counter

KNOWN_VERSIONS = {1: "1.0", 2: "1.1", 3: "1.2"}
TYPE_NAMES = {0: "table", 1: "constants", 2: "cbv", 3: "srv", 4: "uav"}
# D3D12_SHADER_VISIBILITY. ALL is 0, so anything else is a stage-specific parameter; compute has no value of its own
# because every compute parameter is ALL.
VISIBILITY_NAMES = {0: "all", 1: "vertex", 2: "hull", 3: "domain", 4: "geometry", 5: "pixel",
                    6: "amplification", 7: "mesh"}
VISIBILITY_ALL = 0
MAX_PARAMS = 64
MAX_SAMPLERS = 4096


def parts(dxbc):
    """{FourCC tag: payload} from a DXBC container, or {} when the container is malformed (as tools/fssm_unpack.py)."""
    if len(dxbc) < 32 or dxbc[:4] != b"DXBC":
        return {}
    count = struct.unpack_from("<I", dxbc, 28)[0]
    out = {}
    for k in range(count):
        if 32 + 4 * k + 4 > len(dxbc):
            return {}
        off = struct.unpack_from("<I", dxbc, 32 + 4 * k)[0]
        if off + 8 > len(dxbc):
            return {}
        size = struct.unpack_from("<I", dxbc, off + 4)[0]
        end = off + 8 + size
        if end > len(dxbc):
            return {}
        out[dxbc[off:off + 4]] = dxbc[off + 8:end]
    return out


def _u32(payload, off):
    if off < 0 or off + 4 > len(payload):
        raise ValueError(f"read at 0x{off:x} is past the {len(payload)}-byte payload")
    return struct.unpack_from("<I", payload, off)[0]


def parse_rts0(payload):
    """Decode an RTS0 payload. Returns (signature, error); the signature is None on an error.

    Mirrors `rt::rootsig::parse_rts0_payload` (runtime/src/root_signature.cpp), which was measured against the PC
    serializer: header {version, num_params, params_offset, num_samplers, samplers_offset, flags}; parameter entries
    {d3d_type, visibility, payload_offset}; table ranges 20 bytes in 1.0 and 24 in 1.1/1.2, with the flags dword
    before the table offset; static samplers 13 dwords, 14 in 1.2.
    """
    try:
        version, nparams, params_off, nsamplers, samplers_off, flags = (
            _u32(payload, off) for off in (0, 4, 8, 12, 16, 20))
    except ValueError as e:
        return None, f"truncated header: {e}"
    if version not in KNOWN_VERSIONS:
        return None, f"unknown RTS0 version {version} (not 1.0/1.1/1.2)"
    if nparams > MAX_PARAMS:
        return None, f"{nparams} root parameters (more than {MAX_PARAMS})"
    if nsamplers > MAX_SAMPLERS:
        return None, f"{nsamplers} static samplers (absurd)"
    flags_v11 = version >= 2
    try:
        params = []
        for i in range(nparams):
            entry = params_off + 12 * i
            dtype = _u32(payload, entry)
            visibility = _u32(payload, entry + 4)
            offset = _u32(payload, entry + 8)
            if dtype not in TYPE_NAMES:
                return None, f"parameter {i}: unknown type {dtype}"
            p = {"type": TYPE_NAMES[dtype], "visibility": visibility}
            if dtype == 1:
                p["count"] = _u32(payload, offset + 8)          # D3D12_ROOT_CONSTANTS::Num32BitValues
            elif dtype in (2, 3, 4):
                if flags_v11:
                    p["flags"] = _u32(payload, offset + 8)      # the flags dword exists from 1.1 on
            else:
                nranges = _u32(payload, offset)
                ranges_off = _u32(payload, offset + 4)
                if nranges > 1024:
                    return None, f"parameter {i}: {nranges} descriptor ranges (absurd)"
                stride = 24 if flags_v11 else 20
                for j in range(nranges):
                    # Walk every range so a truncated table is an error here, where it is cheap to say so.
                    _u32(payload, ranges_off + j * stride + (20 if flags_v11 else 16))
                p["ranges"] = nranges
            params.append(p)
        samplers = []
        sampler_bytes = (14 if version == 3 else 13) * 4
        for i in range(nsamplers):
            base = samplers_off + i * sampler_bytes
            _u32(payload, base + sampler_bytes - 4)
            samplers.append({"visibility": _u32(payload, base + 48)})
    except ValueError as e:
        return None, f"truncated payload: {e}"
    return {"version": KNOWN_VERSIONS[version], "raw_version": version, "flags": flags,
            "params": params, "static_samplers": samplers}, ""


def span(sig, table_dwords=2, visibility=None):
    """Dwords the block needs: constants take NumValues, a root descriptor two, a table `table_dwords`.

    `visibility` keeps only the parameters of that D3D12_SHADER_VISIBILITY value, which is what a model that omits
    other stages' parameters would pass.
    """
    total = 0
    for p in sig["params"]:
        if visibility is not None and p["visibility"] != visibility:
            continue
        if p["type"] == "constants":
            total += p.get("count", 0)
        elif p["type"] == "table":
            total += table_dwords
        else:
            total += 2
    return total


NO_DEST = ("s_cmp", "s_cbranch", "s_endpgm", "s_waitcnt", "s_clause", "s_inst_prefetch", "s_nop",
           "s_barrier", "s_sleep", "s_setpc", "s_ttracedata", "s_sendmsg", "s_atc")
_REGISTER = re.compile(r"s\[(\d+):(\d+)\]|(?<![\w:])s(\d+)(?![\d\w])")


def _registers(text):
    """(lo, hi) for every register or register range in an operand list."""
    out = []
    for m in _REGISTER.finditer(text):
        if m.group(1) is not None:
            out.append((int(m.group(1)), int(m.group(2))))
        else:
            n = int(m.group(3))
            out.append((n, n))
    return out


def instructions(asm):
    """[(opcode, [operands])] from tools/shader_disasm.py output, comments dropped."""
    out = []
    for line in asm.splitlines():
        code = line.split(";")[0].strip()
        if not code:
            continue
        bits = code.split(None, 1)
        out.append((bits[0], [o.strip() for o in bits[1].split(",")] if len(bits) > 1 else []))
    return out


def read_scan(asm, user_sgprs):
    """The user SGPRs a program reads, the ones it zero-extends into a load base, and the pairs it copies whole.

    A descriptor table is one dword the compiler copies into the low half of a pair, zeroes the high half of and
    passes to `s_load` as the base; a root CBV is a 64-bit pair copied whole into an SRD (docs/d3d12_vulkan.md 3.5).
    Registers at or above `user_sgprs` (from `fspx/<name>.json`) are scratch, not user data.
    """
    ins = instructions(asm)
    read, bases, pairs = set(), set(), set()
    for i, (op, ops) in enumerate(ins):
        for lo, hi in _registers(",".join(ops if op.startswith(NO_DEST) else ops[1:])):
            read.update(r for r in range(lo, hi + 1) if r < user_sgprs)
        if op == "s_mov_b32" and len(ops) >= 2:
            dst, src = _registers(ops[0]), _registers(ops[1])
            if len(dst) == 1 and dst[0][0] == dst[0][1] and len(src) == 1 \
                    and src[0][0] == src[0][1] and src[0][0] < user_sgprs:
                low = dst[0][0]
                window = ins[i + 1:i + 4]
                zeroed = any(o == "s_mov_b32" and len(p) >= 2 and _registers(p[0]) == [(low + 1, low + 1)]
                             and p[1] == "0" for o, p in window)
                loaded = any(o.startswith("s_load") and len(p) >= 2 and _registers(p[1]) == [(low, low + 1)]
                             for o, p in window)
                if zeroed and loaded:
                    bases.add(src[0][0])
        elif op == "s_mov_b64" and len(ops) >= 2:
            src = _registers(ops[1])
            if src and src[0][1] == src[0][0] + 1 and src[0][0] < user_sgprs:
                pairs.add(src[0][0])
    return {"read": sorted(read), "table_bases": sorted(bases), "whole_pairs": sorted(pairs)}


def _read_summary(rows):
    """Counts over the rows that carry a read scan; {} when no row does."""
    scans = [r["reads"] for r in rows if "reads" in r]
    if not scans:
        return {}
    return {
        "programs": len(scans),
        "low_read": sum(1 for s in scans if set(range(4)) & set(s["read"])),
        "table_bases": dict(sorted(Counter(b for s in scans for b in s["table_bases"]).items())),
        "whole_pairs": dict(sorted(Counter(p for s in scans for p in s["whole_pairs"]).items())),
    }


def census(root, disasm=None):
    """Rows for every container in `root`, joined with the program's measured USER_SGPR count.

    With `disasm`, the `--disasm` folder of tools/fspx_unpack.py, each row also carries the program's reads.
    """
    rows, errors = [], []
    for path in sorted(glob.glob(os.path.join(glob.escape(root), "*.dxbc"))):
        name = os.path.basename(path)[:-5]
        with open(path, "rb") as fh:
            chunks = parts(fh.read())
        state_path = os.path.join(root, "fspx", name + ".json")
        user_sgprs = None
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as fh:
                    user_sgprs = json.load(fh).get("user_sgprs")
                if user_sgprs is not None and (type(user_sgprs) is not int or user_sgprs < 0):
                    raise ValueError(f"user_sgprs must be a non-negative integer, got {user_sgprs!r}")
            except (ValueError, AttributeError) as e:
                errors.append(f"{state_path}: {e}")
                continue
        if b"RTS0" not in chunks:
            rows.append({"name": name, "error": "no RTS0 part", "user_sgprs": user_sgprs})
            continue
        sig, error = parse_rts0(chunks[b"RTS0"])
        if sig is None:
            errors.append(f"{name}: {error}")
            rows.append({"name": name, "error": error, "user_sgprs": user_sgprs})
            continue
        row = {"name": name, "version": sig["version"], "flags": sig["flags"],
               "params": sig["params"], "static_samplers": len(sig["static_samplers"]),
               "user_sgprs": user_sgprs,
               "span_table2": span(sig, 2), "span_table1": span(sig, 1),
               "visible_table2": span(sig, 2, VISIBILITY_ALL), "visible_table1": span(sig, 1, VISIBILITY_ALL)}
        if user_sgprs is not None:
            row["spare_table2"] = user_sgprs - row["span_table2"]
            row["spare_table1"] = user_sgprs - row["span_table1"]
            asm_path = os.path.join(disasm, name + ".s") if disasm else None
            if asm_path and os.path.exists(asm_path):
                with open(asm_path, encoding="utf-8") as fh:
                    row["reads"] = read_scan(fh.read(), user_sgprs)
        rows.append(row)
    return rows, errors


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    params = [p for r in ok for p in r["params"]]
    return {
        "containers": len(rows),
        "parsed": len(ok),
        "errors": len(rows) - len(ok),
        "versions": dict(sorted(Counter(r["version"] for r in ok).items())),
        "flags": {hex(k): v for k, v in sorted(Counter(r["flags"] for r in ok).items())},
        "parameters": dict(sorted(Counter(p["type"] for p in params).items())),
        "visibility": {VISIBILITY_NAMES.get(k, str(k)): v for k, v in sorted(Counter(p["visibility"] for p in params).items())},
        "static_samplers": dict(sorted(Counter(r["static_samplers"] for r in ok).items())),
        "user_sgprs": dict(sorted(Counter(r["user_sgprs"] for r in ok if r["user_sgprs"] is not None).items())),
        "span_table2": dict(sorted(Counter(r["span_table2"] for r in ok).items())),
        "span_table1": dict(sorted(Counter(r["span_table1"] for r in ok).items())),
        "spare_table2": dict(sorted(Counter(r["spare_table2"] for r in ok if "spare_table2" in r).items())),
        "spare_table1": dict(sorted(Counter(r["spare_table1"] for r in ok if "spare_table1" in r).items())),
        "read_scan": _read_summary(ok),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="tools/fspx_unpack.py --out folder")
    ap.add_argument("--json", help="also write the rows and the summary here")
    ap.add_argument("--disasm", help="a tools/fspx_unpack.py --disasm folder; adds each program's reads")
    a = ap.parse_args(argv)
    if a.json:
        try:
            if os.path.samefile(a.root, a.json):
                ap.error("--json must not be the input folder")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            pass
    if not os.path.isdir(a.root):
        ap.error(f"not a folder: {a.root}")
    if a.disasm and not os.path.isdir(a.disasm):
        ap.error(f"not a folder: {a.disasm}")
    rows, errors = census(a.root, a.disasm)
    summary = summarize(rows)
    for row in rows:
        if "error" in row:
            print(f"{row['name']}: {row['error']}")
            continue
        types = Counter(p["type"] for p in row["params"])
        print(f"{row['name']}: RTS0 {row['version']}, {len(row['params'])} params {dict(sorted(types.items()))}, "
              f"{row['static_samplers']} static sampler(s), user_sgprs {row['user_sgprs']}, "
              f"span table2 {row['span_table2']} table1 {row['span_table1']}")
    print(f"\n{summary['parsed']}/{summary['containers']} containers parsed; versions {summary['versions']}; "
          f"flags {summary['flags']}")
    print(f"parameters {summary['parameters']}; visibility {summary['visibility']}; "
          f"static samplers {summary['static_samplers']}")
    print(f"user_sgprs {summary['user_sgprs']}")
    print(f"span table2 {summary['span_table2']} table1 {summary['span_table1']}")
    print(f"spare table2 {summary['spare_table2']} table1 {summary['spare_table1']}")
    if summary["read_scan"]:
        rs = summary["read_scan"]
        print(f"reads {rs['programs']} program(s): table bases {rs['table_bases']}, whole pairs {rs['whole_pairs']}, "
              f"{rs['low_read']} read s0-s3")
    for line in errors:
        print(f"error: {line}", file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"rows": rows, "summary": summary}, fh, indent=1, sort_keys=True)
            fh.write("\n")
    return 1 if errors else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
