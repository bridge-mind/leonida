"""Measure the two-number pairs in plain timecycle_modifier_data entries.

Usage:
    python -I tools/timecycle_pairs.py <common.rpf | carve_dir | file.xml ...> [--out FILE]

Reads timecycle_modifier_data and timecycle_keyframe_data XML documents from an RPF9
archive (via gamedata_index.ArchiveSource), a carve directory (DirectorySource), or
plain .xml files, and prints an aggregate JSON summary of the (a, b) number pairs in
plain modifiers: counts, ranges, RGB triplet b-equality, out-of-[0,1] variables, and
per-variable b constancy. Modifier patterns compare the parsed values to 0 and 1
without rounding; `distinct_round6` is a separate distribution statistic.

Outputs aggregate counts, ranges, and schema variable names only — never modifier
names, attribute values, or raw XML. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import gamedata_index as gi  # noqa: E402
from report_paths import same_report_path  # noqa: E402

TARGET_ROOTS = ("timecycle_modifier_data", "timecycle_keyframe_data")


def _iter_source_docs(source) -> list[tuple[str, bytes]]:
    rows = source.rows
    n = len(rows)
    i = 0
    docs: list[tuple[str, bytes]] = []
    while i < n:
        data = source.read(rows[i])
        if gi.stream_kind(data, rows[i]) == "decl":
            root = gi.xml_root(data)
            blob = data
            j = i + 1
            while j < n and not gi.xml_complete(root, blob):
                nxt = source.read(rows[j])
                if gi.stream_kind(nxt, rows[j]) in ("decl", "bink2", "rbf0", "psin", "other", "binary"):
                    break
                blob += nxt
                j += 1
            if root in TARGET_ROOTS:
                docs.append((root, blob))
            i = j
        else:
            i += 1
    return docs


def load_documents(paths: list[str]) -> list[tuple[str, bytes]]:
    docs: list[tuple[str, bytes]] = []
    for path in paths:
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, "index.json")):
                docs.extend(_iter_source_docs(gi.DirectorySource(path)))
            else:
                for name in sorted(os.listdir(path)):
                    full = os.path.join(path, name)
                    if os.path.isfile(full) and name.lower().endswith(".xml"):
                        with open(full, "rb") as f:
                            blob = f.read()
                        root = gi.xml_root(blob)
                        if root in TARGET_ROOTS:
                            docs.append((root, blob))
        elif path.lower().endswith(".rpf"):
            src = gi.ArchiveSource(path)
            try:
                docs.extend(_iter_source_docs(src))
            finally:
                src.close()
        else:
            with open(path, "rb") as f:
                blob = f.read()
            root = gi.xml_root(blob)
            if root in TARGET_ROOTS:
                docs.append((root, blob))
    return docs


def summarize_documents(docs: list[tuple[str, bytes]]) -> dict:
    kf_docs = [b for r, b in docs if r == "timecycle_keyframe_data"]
    mod_docs = [b for r, b in docs if r == "timecycle_modifier_data"]

    kf_range: dict[str, tuple[float, float]] = {}
    for blob in kf_docs:
        root = ET.fromstring(blob)
        for region in root.findall(".//region"):
            for child in region:
                text = (child.text or "").strip()
                if not text:
                    continue
                vals = [float(x) for x in text.split()]
                lo, hi = min(vals), max(vals)
                if child.tag in kf_range:
                    p_lo, p_hi = kf_range[child.tag]
                    kf_range[child.tag] = (min(p_lo, lo), max(p_hi, hi))
                else:
                    kf_range[child.tag] = (lo, hi)

    total_mods = 0
    plain_mods = 0
    axis_mods = 0
    token_counts: Counter[int] = Counter()
    mod_b_patterns: Counter[str] = Counter()
    rgb_triplets: Counter[str] = Counter()
    by_var: dict[str, list[tuple[float, float]]] = defaultdict(list)
    var_docs_b0: dict[str, set[int]] = defaultdict(set)
    var_docs_b1: dict[str, set[int]] = defaultdict(set)
    var_same_doc_mixed: set[str] = set()
    as_: list[float] = []
    bs_: list[float] = []

    for doc_idx, blob in enumerate(mod_docs):
        root = ET.fromstring(blob)
        local_var_bs: dict[str, set[float]] = defaultdict(set)
        for mod in root.findall("modifier"):
            total_mods += 1
            if mod.findall("axis") or "numAxes" in mod.attrib:
                axis_mods += 1
                continue
            plain_mods += 1
            mod_vars: dict[str, tuple[float, float]] = {}
            mod_bs: set[float] = set()
            for ch in mod:
                if ch.tag == "submodifiers":
                    continue
                toks = (ch.text or "").strip().split()
                token_counts[len(toks)] += 1
                if len(toks) != 2:
                    continue
                a, b = float(toks[0]), float(toks[1])
                as_.append(a)
                bs_.append(b)
                by_var[ch.tag].append((a, b))
                mod_vars[ch.tag] = (a, b)
                mod_bs.add(b)
                if b == 0.0:
                    var_docs_b0[ch.tag].add(doc_idx)
                    local_var_bs[ch.tag].add(0.0)
                elif b == 1.0:
                    var_docs_b1[ch.tag].add(doc_idx)
                    local_var_bs[ch.tag].add(1.0)
            if not mod_bs:
                mod_b_patterns["no_vars"] += 1
            elif mod_bs <= {0.0}:
                mod_b_patterns["all_b_eq_0"] += 1
            elif mod_bs <= {1.0}:
                mod_b_patterns["all_b_eq_1"] += 1
            elif mod_bs <= {0.0, 1.0}:
                mod_b_patterns["only_0_and_1"] += 1
            elif all(0.0 <= x <= 1.0 for x in mod_bs):
                mod_b_patterns["in_0_1_with_frac"] += 1
            else:
                mod_b_patterns["outside_0_1"] += 1

            for v in mod_vars:
                if v.endswith("_r"):
                    base = v[:-2]
                    vg, vb = base + "_g", base + "_b"
                    if vg in mod_vars and vb in mod_vars:
                        br, bg, bb = mod_vars[v][1], mod_vars[vg][1], mod_vars[vb][1]
                        if br == bg == bb:
                            if br == 0.0:
                                rgb_triplets["eq0_all_equal"] += 1
                            elif br == 1.0:
                                rgb_triplets["eq1_all_equal"] += 1
                            else:
                                rgb_triplets["frac_all_equal"] += 1
                        else:
                            rgb_triplets["unequal_b"] += 1

        for v, bs_set in local_var_bs.items():
            if 0.0 in bs_set and 1.0 in bs_set:
                var_same_doc_mixed.add(v)

    def _dist(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "lt0": sum(1 for x in vals if x < 0.0),
            "eq0": sum(1 for x in vals if x == 0.0),
            "between_0_1": sum(1 for x in vals if 0.0 < x < 1.0),
            "eq1": sum(1 for x in vals if x == 1.0),
            "gt1": sum(1 for x in vals if x > 1.0),
            "min": min(vals),
            "median": round(statistics.median(vals), 6),
            "max": max(vals),
            "distinct_round6": len({round(x, 6) for x in vals}),
        }

    out_of_01_vars = {}
    for v, ps in sorted(by_var.items()):
        if any(b < 0.0 or b > 1.0 for _, b in ps):
            entry = {
                "count": len(ps),
                "a_range": [min(a for a, _ in ps), max(a for a, _ in ps)],
                "b_range": [min(b for _, b in ps), max(b for _, b in ps)],
                "a_eq_b": sum(1 for a, b in ps if a == b),
                "a_lt_b": sum(1 for a, b in ps if a < b),
                "a_gt_b": sum(1 for a, b in ps if a > b),
                "b_lt0": sum(1 for _, b in ps if b < 0.0),
                "b_eq0": sum(1 for _, b in ps if b == 0.0),
                "b_eq1": sum(1 for _, b in ps if b == 1.0),
                "b_gt1": sum(1 for _, b in ps if b > 1.0),
            }
            if v in kf_range:
                entry["keyframe_range"] = [kf_range[v][0], kf_range[v][1]]
            out_of_01_vars[v] = entry

    always_b0 = sum(1 for ps in by_var.values() if all(b == 0.0 for _, b in ps))
    always_b1 = sum(1 for ps in by_var.values() if all(b == 1.0 for _, b in ps))
    both_01 = sorted(set(var_docs_b0) & set(var_docs_b1))
    non01_var_count = sum(1 for ps in by_var.values() if any(b not in (0.0, 1.0) for _, b in ps))

    return {
        "keyframe_docs": len(kf_docs),
        "modifier_docs": len(mod_docs),
        "keyframe_variables": len(kf_range),
        "modifiers": {"total": total_mods, "plain": plain_mods, "axis": axis_mods},
        "plain_token_counts": {str(k): v for k, v in sorted(token_counts.items())},
        "plain_variables_distinct": len(by_var),
        "first_number": _dist(as_),
        "second_number": _dist(bs_),
        "per_modifier_b_pattern": dict(sorted(mod_b_patterns.items())),
        "per_variable_b_constancy": {
            "always_b_eq_0": always_b0,
            "always_b_eq_1": always_b1,
            "both_b0_and_b1": len(both_01),
            "mixed_within_single_doc": len(var_same_doc_mixed),
            "vars_with_b_not_0_or_1": non01_var_count,
        },
        "rgb_triplet_b_equality": dict(sorted(rgb_triplets.items())),
        "vars_with_b_outside_0_1": out_of_01_vars,
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".timecycle_pairs_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze plain timecycle modifier (a, b) number pairs.")
    ap.add_argument("paths", nargs="+", help="RPF9 archive, carve directory, or .xml file(s)")
    ap.add_argument("--out", help="Optional JSON report path")
    args = ap.parse_args(argv)

    for p in args.paths:
        if not os.path.exists(p):
            ap.error(f"no such file or folder: {p}")

    if args.out:
        if os.path.isdir(args.out):
            ap.error(f"--out is a directory: {args.out}")
        for p in args.paths:
            if same_report_path(args.out, p):
                ap.error(f"--out would overwrite input: {args.out}")
            if os.path.isdir(p) and same_report_path(args.out, os.path.join(p, "index.json")):
                ap.error(f"--out would overwrite carve index: {args.out}")

    try:
        if args.out:
            for p in args.paths:
                if not os.path.isdir(p):
                    continue
                # Directory arguments also own the XML files or carved chunks
                # read below. A report must not replace one of those sources.
                carved = os.path.isfile(os.path.join(p, "index.json"))
                extensions = (".bin", ".bk2") if carved else (".xml",)
                with os.scandir(p) as entries:
                    for entry in entries:
                        if (entry.name.lower().endswith(extensions) and entry.is_file()
                                and same_report_path(args.out, entry.path)):
                            ap.error(f"--out would overwrite input: {args.out}")
        docs = load_documents(args.paths)
        summary = summarize_documents(docs)
        if args.out:
            _write_json_atomic(args.out, summary)
        else:
            json.dump(summary, sys.stdout, indent=1, sort_keys=True)
            sys.stdout.write("\n")
    except (OSError, RuntimeError, ET.ParseError, ValueError) as exc:
        print(f"timecycle_pairs: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
