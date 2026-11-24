"""Match `anim.rpf` `SKEL_*` bone lists against decoded skeleton name tables in `packs.rpf` (ASSET-231).

    python -I tools/anim_skeleton_match.py <packs.rpf> <anim.rpf> [--out <report.json>]

Reads both archives in place using `anim_index.scan_rpf`, extracts `SKEL_[A-Za-z0-9_]+` tokens from
inflated streams, and compares the animation bone set against the model skeletons (both exact and
case-insensitive, plus `joaat32` of lowercased ASCII). Writes counts only (no bone names, offsets
or archive bytes) when `--out` is given. Standard library only; no game data.
Usage: `python -I tools/anim_skeleton_match.py --help` lists the options.
"""
import argparse
import json
import mmap
import os
import re
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anim_index  # noqa: E402
from report_paths import same_report_path  # noqa: E402

SKEL_RE = re.compile(rb"SKEL_[A-Za-z0-9_]+")


def joaat32(name: str) -> int:
    """RAGE joaat32 over the lowercased bone name, encoded as hashdict.joaat does (latin-1, '?' for the rest)."""
    h = 0
    for b in name.lower().encode("latin-1", "replace"):
        h = (h + b) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h = (h ^ (h >> 6)) & 0xFFFFFFFF
    h = (h + (h << 3)) & 0xFFFFFFFF
    h = (h ^ (h >> 11)) & 0xFFFFFFFF
    h = (h + (h << 15)) & 0xFFFFFFFF
    return h


def extract_skel_tokens(data: bytes) -> set[str]:
    """Return distinct `SKEL_[A-Za-z0-9_]+` tokens present in `data`."""
    if b"SKEL_" not in data:
        return set()
    return {m.decode("ascii") for m in SKEL_RE.findall(data)}


def scan_archive_skel(path: str) -> dict:
    """Scan an RPF archive's zlib streams and return stream/token counts and the `SKEL_*` token sets."""
    rows, _ = anim_index.scan_rpf(path)
    per_stream = []
    all_tokens = set()
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for row in rows:
            off = row["offset"]
            try:
                data = zlib.decompress(m[off:off + row["stored"] + 16])
            except zlib.error:
                continue
            toks = extract_skel_tokens(data)
            if toks:
                per_stream.append(toks)
                all_tokens |= toks
    return {"streams": len(rows), "skel_streams": len(per_stream), "per_stream": per_stream, "tokens": all_tokens}


def compare_sets(skel_tokens: set[str], anim_per_stream: list[set[str]]) -> dict:
    """Compare animation `SKEL_*` streams/tokens against a model skeleton token set (counts only)."""
    anim_tokens = set().union(*anim_per_stream) if anim_per_stream else set()
    skel_lower = {t.lower() for t in skel_tokens}
    skel_hashes = {joaat32(t) for t in skel_tokens}
    exact = anim_tokens & skel_tokens
    ci = {t for t in anim_tokens if t.lower() in skel_lower}
    hashed = {t for t in anim_tokens if joaat32(t) in skel_hashes}
    all_exact_streams = sum(1 for s in anim_per_stream if s <= skel_tokens)
    all_ci_streams = sum(1 for s in anim_per_stream if all(t.lower() in skel_lower for t in s))
    any_ci_streams = sum(1 for s in anim_per_stream if any(t.lower() in skel_lower for t in s))
    return {
        "distinct_anim_tokens": len(anim_tokens),
        "distinct_skeleton_tokens": len(skel_tokens),
        "matched_exact": len(exact),
        "matched_case_insensitive": len(ci),
        "matched_joaat32": len(hashed),
        "unmatched_case_insensitive": len(anim_tokens) - len(ci),
        "anim_skel_streams": len(anim_per_stream),
        "streams_all_matched_exact": all_exact_streams,
        "streams_all_matched_ci": all_ci_streams,
        "streams_any_matched_ci": any_ci_streams,
    }


def _check_input(ap, label: str, path: str) -> None:
    if not os.path.exists(path):
        ap.error(f"{label} not found: {path}")
    if os.path.isdir(path):
        ap.error(f"{label} is a directory: {path}")
    try:
        if os.path.getsize(path) == 0:
            ap.error(f"{label} is empty: {path}")
    except OSError as exc:
        ap.error(f"cannot read {label} ({path}): {exc}")


def _check_output(ap, out_path: str, inputs: tuple[str, ...]) -> None:
    if any(same_report_path(out_path, inp) for inp in inputs):
        ap.error("--out must not overwrite an input archive")
    if os.path.isdir(out_path):
        ap.error(f"--out is a directory: {out_path}")
    parent = os.path.dirname(os.path.abspath(out_path))
    if not os.path.isdir(parent):
        ap.error(f"--out parent directory does not exist: {parent}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Match anim.rpf SKEL_* bone lists against packs.rpf skeletons (ASSET-231).")
    ap.add_argument("packs", help="Path to packs.rpf")
    ap.add_argument("anim", help="Path to anim.rpf")
    ap.add_argument("--out", help="Optional JSON output path (counts only)")
    a = ap.parse_args(argv)
    _check_input(ap, "packs", a.packs)
    _check_input(ap, "anim", a.anim)
    if a.out:
        _check_output(ap, a.out, (a.packs, a.anim))
    try:
        packs = scan_archive_skel(a.packs)
        anim = scan_archive_skel(a.anim)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    summary = compare_sets(packs["tokens"], anim["per_stream"])
    summary["packs_streams"] = packs["streams"]
    summary["packs_skel_streams"] = packs["skel_streams"]
    summary["anim_streams"] = anim["streams"]
    text = json.dumps(summary, indent=2)
    print(text)
    if a.out:
        try:
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
        except OSError as exc:
            ap.error(f"cannot write --out ({a.out}): {exc}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
