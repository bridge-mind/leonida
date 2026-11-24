r"""Aggregate census of the `VehSpawnIntensities` table in `CGraphicsDefinitionsData`.

Answers, from the recovered XML document, how the spawn-dirt table is built (ASSET-335, the
`VehSpawnIntensities` half): how many entries it carries, how many intensity bands each entry
holds, how many distinct band signatures there are and how they group, the value ranges of each
band index's `DirtMin`/`DirtMax`/`MudMin`/`MudMax` across those signatures, which band index
carries a mud range (`MudMin`/`MudMax` above zero), and how the rain fields (`DirtRainChance`,
`DirtRainMin`, `DirtRainMax`) group. Numbers and ranges only.

    python -I veh_spawn_intensities.py <CGraphicsDefinitionsData.xml | rpf9_carve dir> [--json FILE]

Accepts one rebuilt XML payload file or a carve directory made by `tools/rpf9_carve.py`.
In both modes the document root must be `CGraphicsDefinitionsData`.
The CLI prints one aggregate JSON object and never prints entry keys, names or per-entry rows:
those are game values. The Python API (`records`) returns entries that contain the table's own
text, so call sites must not write them anywhere public. A bad input gives one ERROR line and
exit 2. Standard library only; no game data is required beyond the local rebuilt file.
Usage: `python -I tools/veh_spawn_intensities.py --help` lists the options.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

DOC_ROOT = "CGraphicsDefinitionsData"
SECTION = "VehSpawnIntensities"
SET_FIELDS = ("DirtMin", "DirtMax", "MudMin", "MudMax")
RAIN_FIELDS = ("DirtRainChance", "DirtRainMin", "DirtRainMax")


class CensusError(ValueError):
    """Invalid input; messages never include input values or paths."""


def _number(item, tag):
    field = item.find(tag)
    if field is None:
        raise CensusError(f"no {tag} element")
    value = field.get("value")
    if value is None:
        raise CensusError(f"{tag} has no value= attribute")
    try:
        number = float(value)
    except ValueError:
        raise CensusError(f"{tag} is not numeric") from None
    if not math.isfinite(number):
        raise CensusError("a value is not finite")
    return number


def records(root):
    """The table's entries as dicts with `key` (game data, local use only), `sets` and `rain`."""
    if root.tag != DOC_ROOT:
        raise CensusError(f"expected {DOC_ROOT} document root")
    section = root.find(SECTION)
    if section is None:
        raise CensusError("no VehSpawnIntensities section")
    out = []
    for item in section.findall("Item"):
        container = item.find("IntensitySets")
        sets = [tuple(_number(sub, f) for f in SET_FIELDS)
                for sub in (container.findall("Item") if container is not None else [])]
        if not sets:
            raise CensusError("entry has no intensity sets")
        out.append({"key": item.get("key") or "", "sets": tuple(sets),
                    "rain": tuple(_number(item, f) for f in RAIN_FIELDS)})
    return out


def aggregate(items):
    """The published shape: counts, group sizes and per-band-index ranges. No keys or names."""
    signatures: dict[tuple, int] = {}
    for item in items:
        signatures[item["sets"]] = signatures.get(item["sets"], 0) + 1
    width = max(len(sig) for sig in signatures)
    per_set = []
    for index in range(width):
        columns = [sig[index] for sig in signatures if len(sig) > index]
        per_set.append({field: [min(col[i] for col in columns), max(col[i] for col in columns)]
                        for i, field in enumerate(SET_FIELDS)})
    rain: dict[tuple, int] = {}
    for item in items:
        rain[item["rain"]] = rain.get(item["rain"], 0) + 1
    return {
        "entries": len(items),
        "signatures": len(signatures),
        "signature_groups": sorted(signatures.values(), reverse=True),
        "set_counts": sorted({len(item["sets"]) for item in items}),
        "per_set_ranges": per_set,
        "bands_with_mud": [sum(1 for sig in signatures if len(sig) > index and sig[index][3] > 0.0)
                           for index in range(width)],
        "rain_groups": len(rain),
        "rain_group_sizes": sorted(rain.values(), reverse=True),
    }


def document_roots(path):
    """Yield ElementTree roots: the one XML file, or the CGraphicsDefinitionsData payloads of a carve."""
    if os.path.isdir(path):
        import gamedata_index as gi

        source = gi.DirectorySource(path)
        positions = {row["off"]: i for i, row in enumerate(source.rows)}
        for row in gi.build_rows(source.rows, source):
            if row.get("type") != "xml" or row.get("root") != DOC_ROOT:
                continue
            first = positions[row["offset"]]
            blob = b"".join(source.read(r) for r in source.rows[first:first + row["chunks"]])
            yield ET.fromstring(blob)
    else:
        yield ET.parse(path).getroot()


def census(path):
    """Read one document, return its aggregate. Errors carry no values or paths."""
    sections = 0
    result = None
    for root in document_roots(path):
        items = records(root)
        sections += 1
        result = aggregate(items)
    if sections == 0:
        raise CensusError("no CGraphicsDefinitionsData document found")
    if sections > 1:
        raise CensusError("more than one CGraphicsDefinitionsData document found")
    return result


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", help="a rebuilt CGraphicsDefinitionsData XML file or a rpf9_carve directory")
    cli.add_argument("--json", help="write the same aggregate census here")
    args = cli.parse_args(argv)
    if args.json and same_report_path(args.input, args.json):
        print("ERROR: --json must not overwrite the input", file=sys.stderr)
        return 2
    try:
        census_out = census(args.input)
    except (OSError, CensusError, ET.ParseError, KeyError, TypeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(census_out, sort_keys=True))
    if args.json:
        try:
            folder = os.path.dirname(os.path.abspath(args.json))
            os.makedirs(folder, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(census_out, f, indent=1, sort_keys=True)
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
    raise SystemExit(main())
