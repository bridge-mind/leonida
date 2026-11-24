#!/usr/bin/env python3
"""RDNA2 ISA coverage of the translator (tools/rdna2spirv.py), measured against AMD's machine-readable ISA XML.

    python -I tools/isa_coverage.py --isa amdgpu_isa_rdna2.xml [--json] [--list] [--llvm-mc llvm-mc]

Needs no game data and no GPU, only the standard library. The XML is AMD's "machine-readable GPU ISA
specification" (MIT licence, see docs/isa_coverage.md for the download line); it is not vendored.

What it reports
  * every RDNA2 instruction in the XML, grouped by encoding (SOP1 ... EXP, plus the DPP/SDWA modifier forms);
  * for each one, what the translator does with it:
      handled-exact   its name is a string the translator compares the opcode with (a == / in test, or a key of
                      a mnemonic table such as NOOPS or F2);
      handled-family  no exact name, but it starts with a prefix the translator tests with op.startswith(...)
                      (v_cmp_, ds_, buffer_atomic_ ...), or ends with a suffix it tests with op.endswith(...) where
                      the test has no startswith (s_*_saveexec_b64 ...; a `<stem> in self.TABLE` beside it limits
                      the family to the stems that table holds);
      refused         the name sits in a test whose whole body is `raise Unsupported(...)`;
      unknown         none of the above: the translator reaches its final `raise Unsupported(op)` for it, or
                      handles it by a rule this static scan cannot see;
  * the names the translator matches that are not RDNA2 instructions (typos, GFX9 or GFX11 spellings, names
    that base_op() can never produce). This list is the most useful output.

Method and limits (static, by ast): the translator dispatches with a long if/elif chain on the opcode string, so
the scan collects the string constants that are compared with `op` (== / != / in / not in, also base_op(...)),
the keys and elements of every dict / set / tuple that holds mnemonics, the prefixes given to op.startswith and the
suffixes given to op.endswith.
It cannot see: a branch that accepts an instruction and then rejects some operand forms (reported separately as
`partial`, when the branch body contains a raise Unsupported), names built by string formatting, and whether the
emitted code is right. "handled" means "has a rule", not "is correct". A broad prefix such as ds_ marks every
ds_* instruction handled-family even where a sub-dispatch then refuses it; the exact names found inside the
sub-dispatch still count as handled-exact, so read handled-family as an upper bound.
Usage: `python -I tools/isa_coverage.py --help` lists the options.
"""
import argparse
import ast
import collections
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSLATOR = os.path.join(HERE, "rdna2spirv.py")
ENV_XML = "LEONIDA_RDNA2_ISA_XML"

# What an opcode string looks like: a lower-case mnemonic starting with a known family word.
MNEMONIC = re.compile(r"^((s|v|ds|buffer|tbuffer|global|flat|scratch|image)(_[a-z0-9]+)+|exp)$")
# base_op() in the translator strips these, so a name that ends in one can never match.
STRIPPED_SUFFIXES = ("_e32", "_e64", "_sdwa", "_dpp")
# Names the translator compares the opcode with: `op`, or a call of base_op().
OP_NAMES = {"op"}

GROUP_ORDER = ["SOP1", "SOP2", "SOPC", "SOPK", "SOPP", "SMEM", "VOP1", "VOP2", "VOP3", "VOP3P", "VOPC", "DS",
               "MUBUF", "MTBUF", "MIMG", "FLAT", "GLOBAL", "SCRATCH", "EXP", "VINTRP", "DPP/SDWA"]
CLASSES = ("handled-exact", "handled-family", "refused", "unknown")


def encoding_group(name):
    """XML encoding name -> the group of the report, or None for an encoding we do not know."""
    n = name.upper()
    if n.startswith("ENC_"):
        n = n[4:]
    if "_VOP_SDWA" in n or "_VOP_DPP" in n:
        return "DPP/SDWA"
    if n.endswith("_INST_LITERAL"):
        n = n[: -len("_INST_LITERAL")]
    if n.startswith("VOP3_SDST_ENC"):
        return "VOP3"
    if n.startswith("MIMG"):
        return "MIMG"
    if n == "FLAT_GLBL":
        return "GLOBAL"
    if n == "FLAT_SCRATCH":
        return "SCRATCH"
    return n if n in GROUP_ORDER else None


def read_isa(path):
    """The XML -> (architecture name, schema version, licence, {mnemonic: [encoding groups]})."""
    root = ET.parse(path).getroot()
    doc = root.find("Document")
    arch = root.findtext("ISA/Architecture/ArchitectureName", "")
    version = doc.findtext("SchemaVersion", "") if doc is not None else ""
    licence = doc.findtext("License", "") if doc is not None else ""
    out = {}
    for ins in root.iterfind("ISA/Instructions/Instruction"):
        name = (ins.findtext("InstructionName") or "").strip().lower()
        if not name:
            continue
        groups = out.setdefault(name, [])
        for enc in ins.iterfind("InstructionEncodings/InstructionEncoding"):
            g = encoding_group(enc.findtext("EncodingName", ""))
            g = g or "OTHER:" + enc.findtext("EncodingName", "?")
            if g not in groups:
                groups.append(g)
    return {"architecture": arch, "schema": version, "license": licence, "instructions": out}


def _strings(node):
    return [x.value for x in ast.walk(node) if isinstance(x, ast.Constant) and isinstance(x.value, str)]


def _mnemonics(node):
    return [s for s in _strings(node) if MNEMONIC.match(s)]


def _is_op(node):
    if isinstance(node, ast.Name):
        return node.id in OP_NAMES
    if isinstance(node, ast.Subscript):          # ins["op"]
        return isinstance(node.slice, ast.Constant) and node.slice.value == "op"
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "base_op"


def _only_raises_unsupported(body):
    return (len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].exc is not None and
            "Unsupported" in ast.unparse(body[0].exc))


def _raises_unsupported(nodes):
    return any(isinstance(x, ast.Raise) and x.exc is not None and "Unsupported" in ast.unparse(x.exc)
               for n in nodes for x in ast.walk(n))


def _op_compare_names(test, positive_only=False):
    """Mnemonics a test compares `op` with (== / in, and unless positive_only != / not in)."""
    names = []
    allowed = (ast.Eq, ast.In) if positive_only else (ast.Eq, ast.NotEq, ast.In, ast.NotIn)
    work = [test]
    while work:
        n = work.pop()
        if positive_only and isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            continue
        if isinstance(n, ast.Compare) and _is_op(n.left) and all(isinstance(o, allowed) for o in n.ops):
            for c in n.comparators:
                names.extend(_mnemonics(c))
        work.extend(reversed(list(ast.iter_child_nodes(n))))
    return names


def _unconditional_op_names(test):
    """Mnemonics `test` matches unconditionally (a positive op == / op in compare, or an `or` of them)."""
    if isinstance(test, ast.Compare) and _is_op(test.left) and all(isinstance(o, (ast.Eq, ast.In)) for o in test.ops):
        return [m for c in test.comparators for m in _mnemonics(c)]
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        parts = [_unconditional_op_names(v) for v in test.values]
        return [m for p in parts for m in p] if parts and all(parts) else []
    return []


def _mnemonic_collection(node):
    """The mnemonics of a dict (its keys) or of a set / tuple / list that holds at least two of them."""
    if isinstance(node, ast.Dict):
        elts = [k for k in node.keys if k is not None]
    elif isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        elts = node.elts
    else:
        return []
    hits = [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str) and
            MNEMONIC.match(e.value)]
    return hits if isinstance(node, ast.Dict) or len(hits) >= 2 else []


def scan_translator(path):
    """Static scan of the translator source. Returns the sets described in the module docstring."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    exact = collections.defaultdict(set)      # name -> {source kind}
    prefixes = collections.defaultdict(set)   # prefix -> {line}
    suffixes = collections.defaultdict(set)
    tables = {}
    refused, partial = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and _is_op(node.left):
            for c in node.comparators:
                for m in _mnemonics(c):
                    exact[m].add("compare")
        elif isinstance(node, ast.Assign):          # NOOPS = {...}, F2 = {"v_add_f32": ...}, BRANCHES = {...}
            keys = (node.value.keys if isinstance(node.value, ast.Dict) else
                    node.value.elts if isinstance(node.value, (ast.Set, ast.Tuple, ast.List)) else [])
            strs = [k.value for k in keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if strs:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        tables[t.id] = tuple(sorted(strs))
            for m in _mnemonic_collection(node.value):
                exact[m].add("table")
        elif isinstance(node, ast.Dict):            # a table written inline
            for m in _mnemonic_collection(node):
                exact[m].add("table")
        if isinstance(node, ast.If):
            uncond = _unconditional_op_names(node.test)
            names = _op_compare_names(node.test, positive_only=True)
            if uncond and _only_raises_unsupported(node.body):
                refused.update(uncond)
            elif names and _raises_unsupported(node.body):
                partial.update(names)
    # Prefix families count only where the instruction dispatch is: Gen._insn and the methods it hands `op` to
    # (self.image_op(op, ...), self.lds_op(op, ...)). Other op.startswith calls in the file (read/write analysis,
    # memory ordering) say nothing about which instructions are translated.
    methods = {fn.name: fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)}
    scope = {"_insn"}
    if "_insn" in methods:
        for c in ast.walk(methods["_insn"]):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and isinstance(c.func.value, ast.Name) and c.func.value.id == "self"
                    and any(isinstance(x, ast.Name) and x.id == "op" for x in c.args[:1])):
                scope.add(c.func.attr)
    for name in scope & set(methods):
        for node in ast.walk(methods[name]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "startswith" and _is_op(node.func.value)):
                for p in _strings(node):
                    if re.match(r"^[a-z0-9_]+$", p):
                        prefixes[p].add(node.lineno)
        for node in _chain_ifs(methods[name].body):
            if _only_raises_unsupported(node.body):
                continue
            for suffix, stems in _suffix_tests(node.test, tables):
                suffixes[(suffix, stems)].add(node.lineno)
    return {"exact": {k: sorted(v) for k, v in exact.items()}, "prefixes": {k: sorted(v) for k, v in prefixes.items()},
            "suffixes": {k: sorted(v) for k, v in suffixes.items()},
            "refused": sorted(refused), "partial": sorted(partial)}


def _op_call(node, attr):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr
            and _is_op(node.func.value))


def _chain_ifs(body):
    out = []
    for st in body:
        while isinstance(st, ast.If):
            out.append(st)
            st = st.orelse[0] if len(st.orelse) == 1 and isinstance(st.orelse[0], ast.If) else None
    return out


def _suffix_tests(test, tables):
    parts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    stems = None
    for x in parts:
        if _op_call(x, "endswith"):
            continue
        t = x.comparators[0] if isinstance(x, ast.Compare) and len(x.ops) == 1 and isinstance(x.ops[0], ast.In) else None
        key = t.attr if isinstance(t, ast.Attribute) else t.id if isinstance(t, ast.Name) else None
        if key not in tables or stems is not None:
            return []
        stems = tables[key]
    return [(s, stems) for x in parts if _op_call(x, "endswith") for s in _strings(x)
            if re.match(r"^_[a-z0-9_]+$", s)]


def in_suffix_family(name, suffixes):
    for suffix, stems in suffixes:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if stems is None or any(stem.endswith("_" + k) for k in stems):
                return True
    return False


def classify(name, scan, handled_elsewhere):
    """One of CLASSES. A name refused by a test and also handled by another test is handled."""
    if name in scan["exact"] and (name not in scan["refused"] or name in handled_elsewhere):
        return "handled-exact"
    if name in scan["refused"]:
        return "refused"
    if any(name.startswith(p) for p in scan["prefixes"]) or in_suffix_family(name, scan.get("suffixes", ())):
        return "handled-family"
    return "unknown"


def accepted_names(path):
    """Names compared or tabled anywhere except inside a test whose body only raises Unsupported."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _unconditional_op_names(node.test) and _only_raises_unsupported(node.body):
            skip.update(id(x) for x in ast.walk(node.test) if isinstance(x, ast.expr))
    names = set()
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Compare) and _is_op(node.left):
            for c in node.comparators:
                names.update(m for m in _mnemonics(c) if not _in_skipped(c, skip))
        elif isinstance(node, (ast.Assign, ast.Dict)):
            names.update(_mnemonic_collection(node.value if isinstance(node, ast.Assign) else node))
    return names


def _in_skipped(node, skip):
    return any(id(x) in skip for x in ast.walk(node) if isinstance(x, ast.expr))


PROBE_TIMEOUT = 30    # seconds llvm-mc gets for one mnemonic


def probe_llvm_mc(llvm_mc, names, mcpu="gfx1013"):
    """Ask llvm-mc about each mnemonic without operands and distinguish tool failures from opcode diagnostics."""
    out = {}
    for n in names:
        try:
            r = subprocess.run([llvm_mc, "-arch=amdgcn", f"-mcpu={mcpu}"], input=n + "\n", capture_output=True,
                               text=True, timeout=PROBE_TIMEOUT)
        except OSError as e:
            return {"error": str(e)}
        except subprocess.TimeoutExpired:
            # A stalled llvm-mc would stall every remaining name for as long again: give up with one message.
            return {"error": f"llvm-mc did not answer within {PROBE_TIMEOUT} s for {n!r}"}
        text = r.stderr + "\n" + r.stdout
        if "error: invalid instruction" in text:
            out[n] = "unknown to LLVM"
        elif "error: instruction not supported on this GPU" in text:
            out[n] = f"LLVM: other GPU only (not {mcpu})"
        elif r.returncode == 0 or "error: too few operands for instruction" in text:
            out[n] = f"LLVM: valid on {mcpu}"
        else:
            diagnostic = next((line.strip() for line in text.splitlines() if line.strip()), "no diagnostic")
            out[n] = f"LLVM probe failed (exit {r.returncode}): {diagnostic}"
    return out


def build_report(isa, scan, accepted):
    instrs = isa["instructions"]
    scan = dict(scan)
    scan["refused"] = sorted(set(scan["refused"]) - accepted)       # refused and nowhere accepted
    scan["exact"] = {k: v for k, v in scan["exact"].items() if k not in scan["refused"] or k in accepted}
    cls = {n: classify(n, scan, accepted) for n in instrs}
    groups = {g: collections.Counter() for g in GROUP_ORDER}
    unknown_by_group = collections.defaultdict(list)
    for n, encs in instrs.items():
        for g in encs:
            groups.setdefault(g, collections.Counter())[cls[n]] += 1
            if cls[n] == "unknown":
                unknown_by_group[g].append(n)
    total = collections.Counter(cls.values())
    names_in_xml = set(instrs)
    # Names the translator matches that are not RDNA2 instructions.
    extra = sorted(n for n in set(scan["exact"]) | set(scan["refused"]) if n not in names_in_xml)
    dead = [n for n in extra if n.endswith(STRIPPED_SUFFIXES)]
    dead_stem = {n: next(n[: -len(s)] for s in STRIPPED_SUFFIXES if n.endswith(s)) for n in dead}
    empty_prefixes = sorted(p for p in scan["prefixes"] if not any(n.startswith(p) for n in names_in_xml))
    partial = sorted(n for n in scan["partial"] if n in names_in_xml)
    return {
        "architecture": isa["architecture"], "schema": isa["schema"], "license": isa["license"],
        "instructions": len(instrs), "totals": {c: total.get(c, 0) for c in CLASSES},
        "groups": {g: {c: groups[g].get(c, 0) for c in CLASSES} | {"total": sum(groups[g].values())}
                   for g in groups if sum(groups[g].values())},
        "class": cls, "unknown_by_group": {g: sorted(v) for g, v in unknown_by_group.items()},
        "refused": sorted(n for n in instrs if cls[n] == "refused"),
        "partial": partial,
        "not_in_rdna2": [{"name": n, "kind": sorted(scan["exact"].get(n, ["refused"])),
                          "never_matches_after_base_op": n in dead_stem,
                          **({"stem": dead_stem[n], "stem_in_rdna2": dead_stem[n] in names_in_xml}
                             if n in dead_stem else {})} for n in extra],
        "prefixes": {p: sum(1 for n in names_in_xml if n.startswith(p)) for p in sorted(scan["prefixes"])},
        "prefixes_without_rdna2_match": empty_prefixes,
        "suffixes": {s: sum(1 for n in names_in_xml if in_suffix_family(n, [(s, st)]))
                     for s, st in sorted(scan.get("suffixes", ()), key=lambda k: (k[0], k[1] or ()))},
    }


def markdown(rep, list_unknown=False):
    t = rep["totals"]
    n = rep["instructions"]
    lines = [f"# RDNA2 ISA coverage of tools/rdna2spirv.py", "",
             f"Spec: {rep['architecture']}, schema {rep['schema']}, licence {rep['license']}. {n} instructions.", "",
             "| class | instructions | share |", "|---|---:|---:|"]
    for c in CLASSES:
        lines.append(f"| {c} | {t[c]} | {100 * t[c] / n:.1f}% |")
    lines += ["", "An instruction with several encodings is counted once in the table above and once per group "
              "below, so the group rows add up to more than the total.", "",
              "| encoding | total | handled-exact | handled-family | refused | unknown |",
              "|---|---:|---:|---:|---:|---:|"]
    for g in sorted(rep["groups"], key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99):
        r = rep["groups"][g]
        lines.append(f"| {g} | {r['total']} | {r['handled-exact']} | {r['handled-family']} | {r['refused']} | "
                     f"{r['unknown']} |")
    lines += ["", f"## Matched by the translator but not an RDNA2 instruction ({len(rep['not_in_rdna2'])})", ""]
    if rep["not_in_rdna2"]:
        for e in rep["not_in_rdna2"]:
            note = ""
            if e["never_matches_after_base_op"]:
                note = f" (ends in a suffix base_op() strips, so it never matches; `{e['stem']}` " + \
                       ("is RDNA2)" if e["stem_in_rdna2"] else "is not RDNA2 either)")
            if e.get("llvm"):
                note += f" [{e['llvm']}]"
            lines.append(f"- `{e['name']}` ({'/'.join(e['kind'])}){note}")
    else:
        lines.append("None.")
    lines += ["", "Prefix families the dispatch tests with op.startswith (RDNA2 instructions each one matches): " +
              ", ".join(f"`{p}` {c}" for p, c in rep["prefixes"].items())]
    if rep["suffixes"]:
        lines += ["", "Suffix families the dispatch tests with op.endswith (RDNA2 instructions each one matches): " +
                  ", ".join(f"`{s}` {c}" for s, c in rep["suffixes"].items())]
    if rep["prefixes_without_rdna2_match"]:
        lines += ["", "Prefixes (op.startswith) that match no RDNA2 instruction: " +
                  ", ".join(f"`{p}`" for p in rep["prefixes_without_rdna2_match"])]
    lines += ["", f"## Refused ({len(rep['refused'])})", "",
              ", ".join(f"`{x}`" for x in rep["refused"]) or "None.",
              "", f"## Handled but with operand forms that raise Unsupported ({len(rep['partial'])})", "",
              ", ".join(f"`{x}`" for x in rep["partial"]) or "None."]
    if list_unknown:
        lines += ["", "## Unknown, by encoding", ""]
        for g in sorted(rep["unknown_by_group"], key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99):
            lines += [f"### {g} ({len(rep['unknown_by_group'][g])})", "",
                      ", ".join(f"`{x}`" for x in rep["unknown_by_group"][g]), ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--isa", default=os.environ.get(ENV_XML), help=f"amdgpu_isa_rdna2.xml (default: ${ENV_XML})")
    ap.add_argument("--translator", default=DEFAULT_TRANSLATOR)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--list", action="store_true", help="also list the unknown instructions per encoding")
    ap.add_argument("--llvm-mc", metavar="EXE", help="classify the non-RDNA2 names with llvm-mc (-mcpu=gfx1013)")
    a = ap.parse_args(argv)
    if not a.isa:
        ap.error(f"--isa is required (or set {ENV_XML})")
    try:
        isa = read_isa(a.isa)
    except OSError as e:
        ap.error(f"--isa {a.isa}: {e.strerror or e}")
    except ET.ParseError as e:
        ap.error(f"--isa {a.isa}: invalid XML ({e})")
    if not isa["instructions"]:
        ap.error(f"--isa {a.isa}: no instructions (not the AMD ISA XML spec?)")
    if "RDNA 2" not in isa["architecture"]:
        print(f"warning: {a.isa} describes {isa['architecture']!r}, not AMD RDNA 2", file=sys.stderr)
    try:
        scan, accepted = scan_translator(a.translator), accepted_names(a.translator)
    except OSError as e:
        ap.error(f"--translator {a.translator}: {e.strerror or e}")
    except UnicodeDecodeError as e:
        ap.error(f"--translator {a.translator}: not valid UTF-8 ({e.reason} at byte {e.start})")
    rep = build_report(isa, scan, accepted)
    if a.llvm_mc:
        probe = probe_llvm_mc(a.llvm_mc, [e["name"] for e in rep["not_in_rdna2"]])
        for e in rep["not_in_rdna2"]:
            e["llvm"] = probe.get(e["name"]) if "error" not in probe else probe["error"]
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        sys.stdout.write(markdown(rep, a.list))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
