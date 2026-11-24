"""Check user-defined varying interfaces between translated graphics stages.

Run `python -I tools/shader_link_check.py --help` for standalone pairs, batch checks,
and reflection of a single SPIR-V module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402
from spirv_interface import (Address, Declaration, InterfaceError, NumericType,  # noqa: E402
                             StageInterface, reflect_module)


STAGES = ("vs", "gs", "ms", "ps")
CHAINS = {("vs", "ps"), ("vs", "gs", "ps"), ("ms", "ps"), ("vs", "gs")}
# A manifest names a shader by its file stem in --shader-dir. The stem must stay one path component inside that
# folder: no separators or drive colon, not "." or "..", and no character a Windows file name can't hold. Anything
# else is allowed: make_game's corpus names use "~", "!", "&", "(" and a leading ".".
BAD_STEM = re.compile(r'[\\/:<>"|?*\x00-\x1f]')


def is_shader_stem(reference):
    return isinstance(reference, str) and reference not in ("", ".", "..") \
        and not BAD_STEM.search(reference) and reference == reference.strip()


@dataclass(frozen=True)
class ModulePath:
    path: Path
    stage: str
    entry: str = "main"


@dataclass(frozen=True)
class Pipeline:
    name: str
    modules: tuple[ModulePath, ...]


@dataclass(frozen=True)
class Problem:
    kind: str
    consumer: str
    message: str
    producer: str | None = None
    location: Address | None = None

    def display(self):
        stages = f"{self.producer} -> {self.consumer}" if self.producer else self.consumer
        where = f" at {self.location.describe()}" if self.location is not None else ""
        return f"{stages}{where} [{self.kind}] {self.message}"


@dataclass
class PipelineResult:
    name: str
    status: str
    links_checked: int
    problems: list[Problem]


def wider_vector_compatible(producer, consumer, allow):
    if not allow or producer.structure or consumer.structure:
        return False
    if producer.indexed_by_vertex_or_primitive or consumer.indexed_by_vertex_or_primitive:
        return False
    p, c = producer.varyings[0].type, consumer.varyings[0].type
    # maintenance4 relaxes the length of vector variables. Scalar vs vector, nested types,
    # and arrays which were unwrapped for a stage's invocation index still need exact types.
    return (isinstance(p, NumericType) and isinstance(c, NumericType)
            and p.kind == c.kind and p.width == c.width and p.components > c.components >= 2)


def compare_stages(producer: StageInterface, consumer: StageInterface, maintenance4=False):
    """Check consumer declarations against the preceding stage's output declarations."""
    output_slots = {slot: declaration
                    for declaration in producer.outputs
                    for varying in declaration.varyings
                    for slot in varying.occupied()}
    problems = []
    for expected in consumer.inputs:
        anchor = expected.varyings[0].address
        actual = output_slots.get(anchor)

        def problem(kind, message):
            return Problem(kind, consumer.stage, message, producer.stage, anchor)

        if actual is None:
            problems.append(problem("missing-output", f"consumer expects {expected.describe()}"))
            continue
        if actual.structure != expected.structure or actual.block != expected.block:
            problems.append(problem("structure-mismatch", f"{actual.describe()} cannot match {expected.describe()}"))
            continue
        if tuple(v.address for v in actual.varyings) != tuple(v.address for v in expected.varyings):
            problems.append(problem("location-mismatch",
                                    f"{actual.describe()} starts or places its members differently from "
                                    f"{expected.describe()}"))
            continue
        if tuple(v.per_primitive for v in actual.varyings) != tuple(v.per_primitive for v in expected.varyings):
            problems.append(problem("per-primitive-mismatch",
                                    f"producer declares {actual.describe()}, consumer expects {expected.describe()}"))
            continue
        if tuple(v.type for v in actual.varyings) != tuple(v.type for v in expected.varyings):
            if wider_vector_compatible(actual, expected, maintenance4):
                continue
            hint = (" (a wider output vector requires maintenance4)"
                    if wider_vector_compatible(actual, expected, True) else "")
            problems.append(problem("type-mismatch",
                                    f"producer declares {actual.describe()}, consumer expects {expected.describe()}"
                                    f"{hint}"))
    return problems


def stage_chain(refs):
    stages = tuple(stage for stage in STAGES if stage in refs)
    if stages not in CHAINS:
        raise ValueError("expected vs+ps, vs+gs, vs+gs+ps, or ms+ps")
    return stages


def reject_duplicate_keys(pairs):
    doc = {}
    for key, value in pairs:
        if key in doc:
            raise ValueError(f"duplicate key {key!r}")
        doc[key] = value
    return doc


def load_pipeline_manifest(path: Path, shader_dir: Path):
    try:
        # utf-8-sig: a hand-written manifest from Notepad or PowerShell 5 Out-File starts with a BOM.
        with path.open(encoding="utf-8-sig") as source:
            doc = json.load(source, object_pairs_hook=reject_duplicate_keys)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if (not isinstance(doc, dict) or set(doc) - {"pipelines", "version"}
            or type(doc.get("version", 1)) is not int or doc.get("version", 1) != 1):
        raise ValueError("expected a version 1 object with a pipelines list")
    if not isinstance(doc.get("pipelines"), list) or not doc["pipelines"]:
        raise ValueError("pipelines must be a nonempty list")
    names = set()
    pipelines = []
    for number, row in enumerate(doc["pipelines"], 1):
        try:
            if not isinstance(row, dict) or set(row) - {"name", *STAGES}:
                raise ValueError("unexpected pipeline field")
            name = row.get("name")
            if not isinstance(name, str) or not name.strip() or any(ord(ch) < 32 for ch in name):
                raise ValueError("expected one printable, nonempty pipeline name")
            if name in names:
                raise ValueError(f"duplicate pipeline name {name!r}")
            stages = stage_chain(row)
            modules = tuple(manifest_module(row[stage], shader_dir, stage) for stage in stages)
            names.add(name)
            pipelines.append(Pipeline(name, modules))
        except ValueError as exc:
            raise ValueError(f"pipelines[{number - 1}]: {exc}") from exc
    return pipelines


def manifest_module(reference, shader_dir, stage):
    entry = "main"
    if isinstance(reference, dict):
        if set(reference) - {"shader", "entry"}:
            raise ValueError(f"{stage}: unexpected shader reference field")
        entry, reference = reference.get("entry", entry), reference.get("shader")
    if not is_shader_stem(reference):
        raise ValueError(f"{stage}: expected a shader name without directories")
    if not isinstance(entry, str) or not entry or "\x00" in entry:
        raise ValueError(f"{stage}: expected a nonempty entry-point name")
    return ModulePath(shader_dir / (reference + ".spv"), stage, entry)


class InterfaceCache:
    def __init__(self):
        self.modules = {}

    def load(self, module):
        if module not in self.modules:
            try:
                self.modules[module] = reflect_module(module.path.read_bytes(), module.stage, module.entry)
            except (OSError, InterfaceError) as exc:
                self.modules[module] = InterfaceError(f"{module.path}: {exc}")
        result = self.modules[module]
        if isinstance(result, InterfaceError):
            raise result
        return result


def check_pipeline(pipeline, cache, maintenance4=False):
    interfaces = {}
    problems = []
    for module in pipeline.modules:
        try:
            interfaces[module.stage] = cache.load(module)
        except (OSError, InterfaceError) as exc:
            problems.append(Problem("module-error", module.stage, str(exc)))
    links = 0
    invalid = bool(problems)
    for producer, consumer in zip(pipeline.modules, pipeline.modules[1:]):
        if producer.stage in interfaces and consumer.stage in interfaces:
            links += 1
            problems.extend(compare_stages(interfaces[producer.stage], interfaces[consumer.stage], maintenance4))
    status = "invalid" if invalid else ("incompatible" if problems else "compatible")
    return PipelineResult(pipeline.name, status, links, problems)


def write_report(path: Path, doc):
    path = path.absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                         dir=path.parent, prefix=".shader-link-", suffix=".tmp") as target:
            temporary = Path(target.name)
            json.dump(doc, target, indent=2, ensure_ascii=False)
            target.write("\n")
        os.chmod(temporary, published_mode(path))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_report_is_separate(output, inputs):
    if output is None:
        return
    out = output.resolve()
    try:
        out_stat = out.stat()
    except (FileNotFoundError, NotADirectoryError):
        out_stat = None
    for source in dict.fromkeys(inputs):
        if out_stat is None:
            aliases = out == source.resolve()
        else:
            try:
                # Existing targets can be compared by file identity, including hard links.
                aliases = os.path.samestat(out_stat, source.stat())
            except (FileNotFoundError, NotADirectoryError):
                aliases = out == source.resolve()
        if aliases:
            raise ValueError(f"report would overwrite an input file: {source}")


def inspect(args):
    ensure_report_is_separate(args.report, [args.module])
    interface = reflect_module(args.module.read_bytes(), args.stage, args.entry)
    doc = {
        "stage": interface.stage,
        "entry": interface.entry,
        "inputs": [declaration_row(item) for item in interface.inputs],
        "outputs": [declaration_row(item) for item in interface.outputs],
    }
    if args.report:
        write_report(args.report, doc)
    else:
        print(json.dumps(doc, indent=2))
    return 0


def declaration_row(declaration):
    return {
        "id": declaration.symbol,
        "structure": declaration.structure,
        "block": declaration.block,
        "indexed_by_vertex_or_primitive": declaration.indexed_by_vertex_or_primitive,
        "varyings": [{**asdict(varying.address), "type": varying.type.describe(),
                      "per_primitive": varying.per_primitive} for varying in declaration.varyings],
    }


def run_checks(args):
    extra_inputs = []
    if args.command == "batch":
        pipelines = load_pipeline_manifest(args.manifest, args.shader_dir)
        extra_inputs = [args.manifest]
    else:
        refs = {}
        for stage in STAGES:
            path = getattr(args, stage)
            entry = getattr(args, f"{stage}_entry")
            if path is not None:
                refs[stage] = ModulePath(path, stage, args.entry if entry is None else entry)
            elif entry is not None:
                raise ValueError(f"--{stage}-entry requires --{stage}")
        pipelines = [Pipeline("pipeline", tuple(refs[stage] for stage in stage_chain(refs)))]
    ensure_report_is_separate(args.report, [*extra_inputs,
                                            *(module.path for pipeline in pipelines for module in pipeline.modules)])
    cache = InterfaceCache()
    results = [check_pipeline(pipeline, cache, args.maintenance4) for pipeline in pipelines]
    summary = {state: sum(result.status == state for result in results)
               for state in ("compatible", "incompatible", "invalid")}
    if args.report:
        write_report(args.report, {"version": 1, "maintenance4": args.maintenance4, "summary": summary,
                                   "pipelines": [asdict(result) for result in results]})
    for result in results:
        if args.quiet and not result.problems:
            continue
        print(f"{result.name}: {result.status} ({result.links_checked} links checked)")
        for problem in result.problems:
            print(f"  {problem.display()}")
    print(", ".join(f"{count} {state}" for state, count in summary.items()))
    if summary["invalid"]:
        return 2
    return int(bool(summary["incompatible"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    direct = commands.add_parser("check", help="check one graphics pipeline from SPIR-V file paths")
    for stage in STAGES:
        direct.add_argument("--" + stage, type=Path, metavar="SPV")
        direct.add_argument(f"--{stage}-entry", metavar="NAME", help=f"override --entry for --{stage}")
    direct.add_argument("--entry", default="main", help="default entry-point name for direct paths (default: main)")
    batch = commands.add_parser("batch", help="check a JSON list of pipelines sharing a shader directory")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--shader-dir", type=Path, required=True, help="directory holding the named .spv files")
    for command in (direct, batch):
        command.add_argument("--maintenance4", action="store_true",
                             help="allow wider output vectors when the pipeline enables maintenance4")
        command.add_argument("--quiet", action="store_true", help="display diagnostics and the summary")
        command.add_argument("--report", type=Path, help="write structured results to JSON")
    single = commands.add_parser("inspect", help="show the reflected user inputs and outputs of one entry point")
    single.add_argument("module", type=Path)
    single.add_argument("--stage", choices=STAGES, help="select a stage when multiple entry points share a name")
    single.add_argument("--entry", default="main")
    single.add_argument("--report", type=Path, help="write the reflection to JSON")
    args = parser.parse_args(argv)
    try:
        return inspect(args) if args.command == "inspect" else run_checks(args)
    except (OSError, ValueError) as exc:
        print(f"shader_link_check: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
