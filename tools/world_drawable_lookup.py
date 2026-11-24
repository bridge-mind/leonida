"""Predict model export offsets from local world_archetypes and rpf9_names reports."""
import argparse
from collections import Counter, defaultdict
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import world_build
import world_names


def report_rows(path, field):
    with open(path, encoding="utf-8-sig") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or not isinstance(report.get(field), list):
        raise ValueError(f"{path}: expected a {field} array")
    if any(not isinstance(row, dict) for row in report[field]):
        raise ValueError(f"{path}: non-object in {field}")
    return report


def lookup(placements, archetypes, names, manifests):
    definitions = defaultdict(set)
    for row in report_rows(archetypes, "archetypes")["archetypes"]:
        definitions[row.get("name")].add((row.get("asset_name"), row.get("asset_type")))

    counts = world_build._placement_counts()
    wanted = Counter()
    outcomes = Counter()
    for row in world_build.iter_placements(placements, counts, world_build.ALL_LOD_LEVELS):
        try:
            key = int(row.get("key"), 16)
        except (TypeError, ValueError):
            outcomes["missing_or_invalid_key"] += 1
            continue
        wanted[row["_arch"], key] += 1

    listed = set()
    for path in manifests:
        entries = report_rows(path, "Items")["Items"]
        index = world_build.model_name_index([("", item) for item in entries], world_names)
        listed.update((h, key) for h, items in index.items() for _, _, key in items)

    offsets = defaultdict(set)
    index = report_rows(names, "files")
    summary = index.get("summary")
    archive = summary.get("archive") if isinstance(summary, dict) else None
    if not isinstance(archive, str) or not archive:
        raise ValueError(f"{names}: missing archive in summary")
    for row in index["files"]:
        offset = row.get("offset")
        if type(offset) is not int or offset < 0 or not isinstance(row.get("names"), list):
            raise ValueError(f"{names}: invalid resource name index row")
        if offset % 4096 != 16:
            continue
        for name in row["names"]:
            if not isinstance(name, str):
                raise ValueError(f"{names}: expected a resource name string")
            stem, _, suffix = name.rpartition(".#")
            asset_type = {"ft": 1, "dr": 2}.get(suffix.lower())
            if not stem or asset_type is None:
                continue
            h = world_names.joaat(stem)
            identity = h, world_names.hash64(stem)
            if identity in wanted and definitions.get(h) == {(h, asset_type)}:
                offsets[identity].add(offset)

    candidates = []
    for identity, n in sorted(wanted.items()):
        h, key = identity
        hits = sorted(offsets.get(identity, ()))
        if identity in listed:
            outcomes["listed_in_manifest"] += n
        elif hits:
            status = "candidate" if len(hits) == 1 else "ambiguous"
            outcomes[status] += n
            candidates.append({"archetype": f"{h:08x}", "key": f"{key:016x}", "offsets": hits,
                               "placements": n, "status": status})
        else:
            outcomes["unresolved"] += n
    return {"archive": archive, "placements": counts, "outcomes": outcomes, "candidates": candidates}


def export_window(report, budget):
    """Maximize single-offset placement predictions within one archive's scan budget."""
    if not 0 < budget <= (1 << 63) - 1:
        raise ValueError("window bytes must be a positive signed 64-bit integer")

    weights = Counter()
    identities = Counter()
    for row in report["candidates"]:
        if row["status"] == "candidate":
            offset, = row["offsets"]
            weights[offset] += row["placements"]
            identities[offset] += 1

    offsets = sorted(weights)
    best = None
    left = total = count = best_total = 0
    for end in offsets:
        total += weights[end]
        count += identities[end]
        while end - offsets[left] >= budget:
            start = offsets[left]
            total -= weights[start]
            count -= identities[start]
            left += 1
        if total > best_total:
            best_total = total
            start = offsets[left]
            best = {"archive": report["archive"], "start": start, "length": end - start + 1,
                    "predicted_placements": total, "predicted_identities": count}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("placements", help="world_index entity JSONL")
    parser.add_argument("archetypes", help="world_archetypes JSON")
    parser.add_argument("names", help="rpf9_names JSON for the source archive")
    parser.add_argument("--manifest", action="append", default=[], help="exclude identities already listed (repeatable)")
    parser.add_argument("--window-bytes", type=int, help="recommend one model scan of at most this many bytes")
    args = parser.parse_args()
    try:
        result = lookup(args.placements, args.archetypes, args.names, args.manifest)
        if args.window_bytes is not None:
            result["export_window"] = export_window(result, args.window_bytes)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
