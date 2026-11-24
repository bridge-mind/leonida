"""Audit graphics subgroup-width fidelity from a translator index and Vulkan device properties.

Reports only counts, by default for every graphics shader in the index. With --manifest, use the
same pipeline manifest as shader_link_check.py and also count each graphics pipeline's worst result.
No SPIR-V binary is opened. A report is a subgroup-width assessment, not a Vulkan compile or draw.
Usage: `python -I tools/wave64_pipeline_audit.py --help` lists the options.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from math import prod
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shader_link_check import (ensure_report_is_separate, is_shader_stem, load_pipeline_manifest,  # noqa: E402
                               write_report)
from vk_subgroup_info import query  # noqa: E402

# VkShaderStageFlagBits. A mesh-compute module uses the compute bit when assessed outside a graphics chain.
STAGE_BITS = {"vs": 0x01, "gs": 0x08, "ps": 0x10, "cs": 0x20, "ms": 0x80}
GRAPHICS = ("vs", "gs", "ms", "ps")
OUTCOMES = ("exact", "partition-invariant", "default-width", "approximate", "unknown", "unsupported")
SEVERITY = {key: i for i, key in enumerate(OUTCOMES)}


@dataclass(frozen=True)
class Device:
    default: int
    minimum: int
    maximum: int
    supported: int
    required: int
    control: bool
    max_workgroup_subgroups: int

    @classmethod
    def read(cls, data):
        names = ("subgroup_size", "min_subgroup_size", "max_subgroup_size", "supported_subgroup_stages",
                 "required_subgroup_size_stages", "subgroup_size_control", "max_compute_workgroup_subgroups")
        if not isinstance(data, dict) or any(key not in data for key in names):
            raise ValueError("device info must come from tools/vk_subgroup_info.py --stage-caps")
        values = [data[key] for key in names]
        if type(values[5]) is not bool or any(type(v) is not int or not 0 <= v <= 0xffffffff
                                              for i, v in enumerate(values) if i != 5):
            raise ValueError("invalid subgroup capability values")
        info = cls(*values)
        if info.minimum > info.maximum or (info.minimum and info.default and
                                           not info.minimum <= info.default <= info.maximum):
            raise ValueError("inconsistent device subgroup size range")
        if any(v and v & (v - 1) for v in (info.default, info.minimum, info.maximum)):
            raise ValueError("device subgroup sizes must be powers of two")
        return info


@dataclass(frozen=True)
class Assessment:
    outcome: str
    reason: str


def assess(entry, caps, *, in_chain=False):
    stage = entry["stage"]
    if stage == "ms" and entry.get("mesh_compute"):
        if in_chain:
            return Assessment("unsupported", "mesh-compute-is-not-a-graphics-stage")
        stage = "cs"
    if entry.get("wave64_exchange") == "workgroup":
        if stage in ("ms", "cs"):
            return Assessment("exact", "workgroup-exchange")
        return Assessment("unknown", "unexpected-workgroup-exchange")
    wanted = entry.get("subgroup_size")
    if wanted is None:
        wave64 = entry.get("wave64")
        if type(wave64) is not bool:
            return Assessment("unknown", "missing-wave-width")
        wanted = 64 if wave64 else 32
    if type(wanted) is not int or wanted not in (32, 64):
        return Assessment("unknown", "invalid-translated-wave-width")
    if "wave64" in entry and (type(entry["wave64"]) is not bool or entry["wave64"] != (wanted == 64)):
        return Assessment("unknown", "conflicting-wave-width")
    if not caps.supported & STAGE_BITS[stage]:
        return Assessment("unsupported", "no-subgroup-operations-in-stage")
    if caps.minimum == caps.maximum == wanted:
        return Assessment("exact", "fixed-width")
    if caps.control and caps.required & STAGE_BITS[stage] and caps.minimum <= wanted <= caps.maximum:
        if stage in ("ms", "cs"):
            size = entry.get("local_size")
            if not isinstance(size, list) or len(size) != 3 or any(type(v) is not int or v <= 0 for v in size):
                return Assessment("unknown", "missing-workgroup-size")
            can_require = prod(size) <= wanted * caps.max_workgroup_subgroups
        else:
            can_require = True
        if can_require:
            return Assessment("exact", "required-width")
    if not caps.default:
        return Assessment("unknown", "missing-device-default-width")
    if caps.default == wanted:
        return Assessment("default-width", "matches-device-default")
    if wanted == 64 and caps.default == 32 and stage in ("vs", "gs", "ps"):
        tier = entry.get("wave64_graphics")
        if tier == "A":
            return Assessment("partition-invariant", "tier-A")
        if tier == "B":
            return Assessment("partition-invariant", "tier-B-slot-order-can-change")
        if tier == "C":
            return Assessment("approximate", "tier-C-cross-half-dependence")
        return Assessment("unknown", "missing-partition-analysis")
    return Assessment("unsupported", "translated-width-unavailable")


def empty_counts():
    return {name: 0 for name in OUTCOMES}


def build_report(index, caps, chains=None):
    if chains is None:
        selection = {name for name, row in index.items() if row.get("stage") in GRAPHICS}
    else:
        selection = set()
        for chain in chains:
            for module in chain.modules:
                name = module.path.stem
                row = index.get(name)
                if row is None or row.get("stage") != module.stage:
                    # Reports are shareable aggregate data. Avoid echoing translated shader names.
                    raise ValueError("a manifest entry is missing from the index or has a stage mismatch")
                selection.add(name)

    results = {name: assess(index[name], caps, in_chain=chains is not None) for name in selection}
    by_stage = defaultdict(empty_counts)
    reasons = Counter()
    total = empty_counts()
    for name, result in results.items():
        by_stage[index[name]["stage"]][result.outcome] += 1
        reasons[result.reason] += 1
        total[result.outcome] += 1

    chain_counts = None
    if chains is not None:
        chain_counts = empty_counts()
        for chain in chains:
            worst = max((results[module.path.stem].outcome for module in chain.modules), key=SEVERITY.__getitem__)
            chain_counts[worst] += 1
    return {"version": 1, "device": asdict(caps), "shaders": total, "stages": dict(sorted(by_stage.items())),
            "reasons": dict(sorted(reasons.items())), "pipelines": chain_counts}


def load_json(path):
    try:
        with path.open(encoding="utf-8-sig") as source:
            return json.load(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index", type=Path, help="the translator's spirv/index.json")
    parser.add_argument("--manifest", type=Path, help="audit only the chains from this shader-link manifest")
    parser.add_argument("--device-info", type=Path, help="use saved vk_subgroup_info.py --stage-caps JSON")
    parser.add_argument("--report", type=Path, help="save a JSON report with aggregate counts")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 for a result other than exact or partition-invariant")
    args = parser.parse_args(argv)
    try:
        index = load_json(args.index)
        if not isinstance(index, dict) or any(not isinstance(row, dict) for row in index.values()):
            raise ValueError("expected a translator index object whose entries are objects")
        chains = None
        if args.manifest:
            chains = load_pipeline_manifest(args.manifest, args.index.parent)
        caps = Device.read(load_json(args.device_info) if args.device_info else query(stage_caps=True))
        report = build_report(index, caps, chains)
        if not sum(report["shaders"].values()):
            raise ValueError("the selection contains no graphics shaders")
        if args.report:
            inputs = [path for path in (args.index, args.manifest, args.device_info) if path]
            inputs.extend(args.index.parent / (name + ".spv") for name in index
                          if is_shader_stem(name))
            ensure_report_is_separate(args.report, inputs)
            write_report(args.report, report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"wave64_pipeline_audit: {exc}", file=sys.stderr)
        return 2

    for label, counts in (*report["stages"].items(), ("shaders", report["shaders"]),
                          ("pipelines", report["pipelines"])):
        if counts is not None:
            print(f"{label} ({sum(counts.values())}): " + ", ".join(f"{key}={n}" for key, n in counts.items() if n))
    print(f"Device reports subgroup widths {caps.minimum}..{caps.maximum}, default {caps.default}.")
    if args.strict and any(report["shaders"][status] for status in OUTCOMES[2:]):
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
