"""Count declared G-buffer signatures and static exports in an indexed shader carve."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_scan import parts, signature_offsets
from fssm_unpack import unpack_all
from shader_ir import ENC, HDR, parse_line


def signature(path):
    return signature_bytes(path.read_bytes())


def signature_bytes(data):
    sig = parts(data)["OSG1"]
    rows = []
    for offset in signature_offsets(sig):
        index, system, ctype, register, mask, _, precision = struct.unpack_from(
            "<4IBB2xI", sig, offset + 8)
        kind = {64: "target", 65: "depth", 67: "depth_ge", 68: "depth_le"}.get(system, f"system{system}")
        rows.append((kind, index, register, mask, ctype, precision))
    shape = Counter(row[0] for row in rows)
    colour = shape["target"]
    depth = sum(shape[key] for key in ("depth", "depth_ge", "depth_le"))
    layout = " | ".join(f"{k}{i} reg={r} mask={m:#x} type={t} precision={p}"
                      for k, i, r, m, t, p in sorted(rows)) or "(empty)"
    return f"{len(rows)} rows, {colour} colour, {depth} depth", layout


def exports(path):
    counts = Counter()
    encoded_bytes = 0
    with path.open(encoding="utf-8") as listing:
        header = listing.readline()
        span = HDR.match(header)
        for line in listing:
            if line.startswith("; ---"):
                break
            encoding = ENC.search(line)
            if encoding and encoding[1].strip():
                encoded_bytes += encoding[1].count(",") + 1
            if line.split(maxsplit=1)[:1] != ["exp"]:
                continue
            ins = parse_line(line)
            target = ins["args"][0].get("t", "")
            if re.fullmatch(r"mrt[0-7z]", target):
                mode = "compr" if "compr" in ins["mods"] else "uncompressed"
                counts[f"{target} {mode}"] += 1
    complete = (span and encoded_bytes > 0 and encoded_bytes == int(span[3], 16) - int(span[2], 16)
                and not re.search(r"\b[1-9]\d* invalid words", header))
    return counts, bool(complete)


def payload_identity(data):
    """Only assign an entry when its container identifies one whole FSSM."""
    try:
        if data is None or not data.startswith(b"DXBC"):
            return "unavailable", None
        identities = {hashlib.sha256(blob).digest() for blob in unpack_all(data)}
    except (ValueError, struct.error, zlib.error):
        return "unavailable", None
    if not identities:
        return "unavailable", None
    if len(identities) > 1:
        return "ambiguous", None
    return "identified", identities.pop()


def census(directory, listings, *, payload_coverage=False):
    predicates = ("all indexed ps", "entry contains deferred (casefold)",
                  "stem contains Deferred (case-sensitive)", "stem contains PS_DeferredTextured (case-sensitive)")
    report = {key: {"coverage": Counter(), "declared_counts": Counter(), "signatures": Counter(), "observed_exports": Counter()}
              for key in predicates}
    identities = {key: Counter() for key in predicates}
    if payload_coverage:
        for group in report.values():
            group["payload_coverage"] = dict(identified=0, ambiguous=0, unavailable=0)
    for row in json.loads((directory / "index.json").read_text(encoding="utf-8")):
        if row["stage"] != "ps":
            continue
        path = directory / row["file"]
        stem = path.stem
        selected = (True, "deferred" in row["name"].casefold(), "Deferred" in stem, "PS_DeferredTextured" in stem)
        coverage = Counter(indexed=1)
        sig = None
        observed = Counter()
        data = None
        try:
            data = path.read_bytes()
            sig = signature_bytes(data)
            coverage["signature_read"] += 1
        except FileNotFoundError:
            coverage["container_missing"] += 1
        except (KeyError, ValueError, struct.error, OSError):
            coverage["signature_unreadable"] += 1
        try:
            observed, complete = exports(listings / (stem + ".s"))
            coverage["listing_complete" if complete else "listing_incomplete"] += 1
        except FileNotFoundError:
            coverage["listing_missing"] += 1
        except (OSError, IndexError, ValueError):
            coverage["listing_unreadable"] += 1
        if payload_coverage:
            status, digest = payload_identity(data)
        for key, include in zip(predicates, selected):
            if include:
                group = report[key]
                if payload_coverage:
                    group["payload_coverage"][status] += 1
                    if digest is not None:
                        identities[key][digest] += 1
                group["coverage"].update(coverage)
                group["observed_exports"].update(observed)
                if sig is not None:
                    shape, layout = sig
                    group["declared_counts"][shape] += 1
                    group["signatures"][layout] += 1
    if payload_coverage:
        for key, counts in identities.items():
            summary = report[key]["payload_coverage"]
            summary["distinct_payloads"] = len(counts)
            summary["payloads_by_indexed_entry_count"] = Counter(counts.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxbc", type=Path, help="shader_carve directory containing index.json")
    parser.add_argument("--disasm", type=Path, help="listing directory (default: DXBC/disasm)")
    parser.add_argument("--payload-coverage", action="store_true",
                        help="count distinct whole FSSM payloads per cohort and histogram their indexed entry counts")
    args = parser.parse_args()
    print(json.dumps(census(args.dxbc, args.disasm or args.dxbc / "disasm",
                            payload_coverage=args.payload_coverage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
