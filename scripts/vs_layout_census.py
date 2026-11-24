"""SHADER-33 follow-up: which PAL hardware stage every FSSM payload runs on, and whether the VS input layout holds.

    python -I scripts/vs_layout_census.py <work>/shaders/dxbc [--json OUT] [--limit N]

Reads <dxbc>/index.json, every tag-0x0D payload of each container (tools/fssm_unpack.unpack_all) and its PAL
stage descriptors (tools/pal_stages.stage_descriptors), then the code: the .s listing for the container's last
payload (fssm_unpack's default) and, for any earlier payload, a fresh llvm-mc disassembly. Prints counts only:

- payloads by (API id, PAL hardware id) and the carve index's stage for the same stem
- NGG markers per hardware id: `exp prim`, any `s_sendmsg`, `s_sendmsg MSG_GS_ALLOC_REQ` (the NGG
  allocation), `s_sendmsg` with `GS_OP_*` (legacy geometry-stream output) and `v_mbcnt`
- for API-VS payloads (hardware id 4 = the hardware VS; id 2 = the ES variant of a two-payload container):
  the fetch call, the entry VGPRs read before written (scripts/vs_input_survey.read_before_write) and the
  v0/v3 families of scripts/vs_input_survey.survey

The hardware-stage names come from tools/pal_stages.py (an outside project's id order); this census is what
checks them against the code: every hardware-id-7 payload carries the NGG markers and no other stage does.
Standard library only; llvm-mc only for payloads without a listing. Nothing from the game is printed or
written beyond counts.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fssm_unpack  # noqa: E402
import pal_stages  # noqa: E402
import shader_disasm  # noqa: E402
import shader_ir  # noqa: E402
from vs_input_survey import (FETCH_CALL, MBCNT, PARAM_EXPORT, POS0_EXPORT, PRIM_EXPORT, SENDMSG,  # noqa: E402
                             read_before_write)

ALLOC_REQ = re.compile(r"s_sendmsg\s+sendmsg\(MSG_GS_ALLOC_REQ\)")
GS_OP = re.compile(r"s_sendmsg\s+sendmsg\(MSG_GS,")


def markers(text):
    """The code markers that separate the NGG path, legacy GS stream output and the plain VS."""
    return {
        "exp prim": bool(PRIM_EXPORT.search(text)),
        "s_sendmsg": bool(SENDMSG.search(text)),
        "s_sendmsg GS_ALLOC_REQ": bool(ALLOC_REQ.search(text)),
        "s_sendmsg GS_OP": bool(GS_OP.search(text)),
        "v_mbcnt": bool(MBCNT.search(text)),
        "exp pos0": bool(POS0_EXPORT.search(text)),
        "exp param": bool(PARAM_EXPORT.search(text)),
        "fetch call": bool(FETCH_CALL.search(text)),
    }


def insns_from_text(text):
    """shader_ir instructions from listing text (no file, no pc tracking: read_before_write needs neither)."""
    insns = []
    for line in text.splitlines():
        if not line.startswith("\t"):
            continue
        ins = shader_ir.parse_line(line)
        if ins:
            insns.append(ins)
    return insns


def default_disasm(blob):
    """llvm-mc disassembly of one FSSM payload's code range (the same mcpu make_game uses)."""
    start, end, _how = shader_disasm.code_range(blob)
    if end is None:
        return ""
    asm, _bad, _err = shader_disasm.disasm(blob[start:end], "gfx1013")
    return asm


def payloads_of(dxbc_path):
    """(stem, payload index, blob, listed) for every tag-0x0D payload of every container, in index order."""
    with open(dxbc_path, "rb") as f:
        data = f.read()
    try:
        blobs = fssm_unpack.unpack_all(data)
    except (ValueError, OSError):
        return []
    listed = len(blobs) - 1
    return [(os.path.splitext(os.path.basename(dxbc_path))[0], i, blob, i == listed)
            for i, blob in enumerate(blobs)]


def survey(dxbc_dir, disasm_fn=None, limit=None):
    """The census over <dxbc>: counts by stage, hardware id, markers and VS entry VGPRs."""
    index_path = os.path.join(dxbc_dir, "index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, list) or not all(isinstance(e, dict) for e in index):
        raise ValueError("index.json is not a list of shader entries")
    stages = {}
    names = []
    for e in index:
        if isinstance(e.get("file"), str):
            stem = os.path.splitext(e["file"])[0]
            if stem not in stages:
                names.append(stem)
            stages[stem] = e.get("stage")
    if limit is not None:
        names = names[:limit]

    by_stage_hw = Counter()
    markers_by_hw = defaultdict(Counter)
    descriptors_per_blob = Counter()
    api_mismatch = 0
    no_record = 0
    vs_rows = []
    disassembled = 0
    for name in names:
        dxbc_path = os.path.join(dxbc_dir, name + ".dxbc")
        if not os.path.isfile(dxbc_path):
            continue
        rows = payloads_of(dxbc_path)
        for _stem, payload_index, blob, listed in rows:
            records = pal_stages.stage_descriptors(blob)
            if not records:
                no_record += 1
                continue
            first = records[0]
            hw, api = first["hardware_id"], first["api_id"]
            by_stage_hw[(stages.get(name), api, hw)] += 1
            descriptors_per_blob[len(records)] += 1
            if stages.get(name) and api is not None and stages[name] != {0: "vs", 1: "ps", 2: "gs", 3: "cs",
                                                                        6: "ms"}.get(api):
                api_mismatch += 1
            listing = os.path.join(dxbc_dir, "disasm", name + ".s")
            if listed and os.path.isfile(listing):
                with open(listing, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            elif disasm_fn is not None:
                text = disasm_fn(blob)
                disassembled += 1
            else:
                text = None
            if text is not None:
                for key, value in markers(text).items():
                    if value:
                        markers_by_hw[hw][key] += 1
            if api == 0:
                entry = {"hw": hw, "payload": payload_index, "listed": listed}
                if text is not None:
                    _, read_v = read_before_write(insns_from_text(text))
                    low = sorted(r for r in read_v if r < 16)
                    entry.update({
                        "fetch": bool(FETCH_CALL.search(text)),
                        "reads_v0": 0 in read_v,
                        "reads_v3": 3 in read_v,
                        "reads_v1_v2": any(r in (1, 2) for r in read_v),
                        "low_vgprs": low,
                    })
                vs_rows.append(entry)

    return {
        "containers": len(names),
        "payloads": sum(by_stage_hw.values()),
        "payloads_without_record": no_record,
        "descriptors_per_blob": dict(sorted(descriptors_per_blob.items())),
        "api_stage_mismatches": api_mismatch,
        "disassembled_unlisted": disassembled,
        "by_stage_hw": {"|".join(str(x) for x in k): v for k, v in sorted(by_stage_hw.items(),
                                                                         key=lambda kv: (str(kv[0][0]), kv[0][1] or -1, kv[0][2] or -1))},
        "markers_by_hw": {str(hw): dict(sorted(c.items())) for hw, c in sorted(markers_by_hw.items())},
        "vs_payloads": len(vs_rows),
        "vs_by_hw": dict(sorted(Counter(r["hw"] for r in vs_rows).items())),
        "vs_fetch": sum(1 for r in vs_rows if r.get("fetch")),
        "vs_reads_v0": sum(1 for r in vs_rows if r.get("reads_v0")),
        "vs_reads_v3": sum(1 for r in vs_rows if r.get("reads_v3")),
        "vs_reads_v1_v2": sum(1 for r in vs_rows if r.get("reads_v1_v2")),
        "vs_low_vgprs": {str(hw): dict(sorted(Counter(",".join(map(str, r.get("low_vgprs", ())))
                                                                   for r in vs_rows if r["hw"] == hw).items(),
                                              key=lambda kv: -kv[1])[:12])
                         for hw in sorted({r["hw"] for r in vs_rows})},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dxbc", help="<work>/shaders/dxbc from make_game.py")
    ap.add_argument("--json", help="also write the counts as JSON")
    ap.add_argument("--limit", type=int, help="stop after this many containers (index order)")
    a = ap.parse_args(argv)
    index_path = os.path.join(a.dxbc, "index.json")
    if a.json and os.path.exists(a.json) and os.path.exists(index_path) and os.path.samefile(a.json, index_path):
        print(f"vs_layout_census: error: --json would overwrite the input {index_path}", file=sys.stderr)
        return 2
    try:
        r = survey(a.dxbc, disasm_fn=default_disasm, limit=a.limit)
    except (OSError, ValueError) as e:
        print(f"vs_layout_census: error: cannot read {index_path}: {e}", file=sys.stderr)
        return 2
    print(f"{r['containers']} containers, {r['payloads']} payloads "
          f"({r['payloads_without_record']} without a PAL record), descriptors per blob {r['descriptors_per_blob']}")
    print("stage/hw (carve stage|API|PAL hw):", r["by_stage_hw"])
    print("markers by PAL hw:", r["markers_by_hw"])
    print(f"API-vs payloads {r['vs_payloads']} by PAL hw {r['vs_by_hw']}: fetch {r['vs_fetch']}, "
          f"reads v0 {r['vs_reads_v0']}, reads v3 {r['vs_reads_v3']}, reads v1/v2 {r['vs_reads_v1_v2']}")
    print("VS/ES entry VGPRs (low 16, top 12):", r["vs_low_vgprs"])
    print(f"API stage mismatches {r['api_stage_mismatches']}, "
          f"unlisted payloads disassembled {r['disassembled_unlisted']}")
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(r, f, indent=1)
        except OSError as e:
            print(f"vs_layout_census: error: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
