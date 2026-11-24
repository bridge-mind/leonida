"""Op-scoped mutation testing for a tests/semantic file: which one-line translator bugs would it miss?

    python -I tools/semantic_mutants.py tests/semantic/test_valu_sad_u8_hw.py --op v_sad_u8
    python -I tools/semantic_mutants.py tests/semantic/test_x.py --op ds_swizzle_b32 --function swizzle
    python -I tools/semantic_mutants.py tests/semantic/test_x.py --lines 8375-8414

The tool finds the code in tools/rdna2spirv.py that translates the op: the `if op == "..."` / `elif op in (...)`
branch naming it (--op), a whole method (--function), or an explicit line range (--lines). It then makes one-line
mutants of that code with standard operators: swapped opcode pairs (OpIAdd/OpISub, OpBitwiseAnd/Or, shifts), swapped
GLSL.std.450 pairs (UMin/UMax, SMin/UMin, NMin/FMin, Floor/Ceil, ...), comparisons made strict or non-strict or of the
other signedness, small `self.cu(N)` constants moved by one, the two arms of an OpSelect swapped, and the last operand
of an emitted instruction replaced by the one before it. Each mutant is written into a private copy of tools/ (the
repository's own tools/ is never modified) and the test file is run against it through LEONIDA_TOOLS_DIR, as
tests/semantic/test_mutations.py does. A mutant the test file still passes is reported as SURVIVED.

A surviving mutant is a lead, not a verdict: some mutants are equivalent (they can't change any result, e.g. a
constant that only feeds a branch the op never takes), so read each one before asking for a new case. Exit status:
0 when every mutant is caught, 1 when any survives, 2 on a usage or baseline error (the unmutated copy must pass
first). Synthetic: no game files, no GPU.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSLATOR = "rdna2spirv.py"

# Whole-token swaps: each pair is tried in both directions where it occurs.
PAIRS = [
    ("OpIAdd", "OpISub"), ("OpBitwiseAnd", "OpBitwiseOr"), ("OpBitwiseOr", "OpBitwiseXor"),
    ("OpShiftLeftLogical", "OpShiftRightLogical"), ("OpShiftRightLogical", "OpShiftRightArithmetic"),
    ("OpFAdd", "OpFSub"), ("OpFMul", "OpFAdd"), ("OpConvertFToS", "OpConvertFToU"),
    ("OpConvertSToF", "OpConvertUToF"), ("OpBitFieldSExtract", "OpBitFieldUExtract"),
    ("OpIEqual", "OpINotEqual"), ("OpULessThan", "OpULessThanEqual"), ("OpUGreaterThan", "OpUGreaterThanEqual"),
    ("OpSLessThan", "OpSLessThanEqual"), ("OpSGreaterThan", "OpSGreaterThanEqual"), ("OpULessThan", "OpSLessThan"),
    ("OpUGreaterThan", "OpSGreaterThan"), ("OpLogicalAnd", "OpLogicalOr"),
    ("UMin", "UMax"), ("SMin", "SMax"), ("FMin", "FMax"), ("NMin", "NMax"), ("SMin", "UMin"), ("SMax", "UMax"),
    ("NMin", "FMin"), ("NMax", "FMax"), ("SClamp", "UClamp"), ("Floor", "Ceil"), ("Trunc", "Floor"),
    ("RoundEven", "Round"),
    ("OpAtomicIAdd", "OpAtomicISub"), ("OpAtomicSMin", "OpAtomicUMin"), ("OpAtomicSMax", "OpAtomicUMax"),
    ("OpAtomicUMin", "OpAtomicUMax"), ("OpAtomicAnd", "OpAtomicOr"),
]
CU = re.compile(r"self\.cu\((0x[0-9A-Fa-f]+|\d+)\)")
SELECT = re.compile(r"(OpSelect %\w+ \{[^}]+\} )(\{[^}]+\}|%\w+)( )(\{[^}]+\}|%\w+)")
LAST_OPERAND = re.compile(r"(\{[A-Za-z_][\w\[\]]*\}) (\{[A-Za-z_][\w\[\]]*\})(\"\))")


@dataclass
class Mutant:
    line: int      # 1-based line in rdna2spirv.py
    label: str
    old: str       # the whole original line
    new: str       # the whole mutated line


def find_branch(lines: list[str], op: str) -> tuple[int, int]:
    """(first, last) 0-based line indices of the if/elif branch whose condition names `op` as a string literal."""
    quoted = f'"{op}"'
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not re.match(r"(el)?if\b", stripped) or "op" not in stripped:
            continue
        cond, j = line, i
        while not cond.rstrip().endswith(":") and j + 1 < len(lines):     # a condition spread over several lines
            j += 1
            cond += lines[j]
        if quoted not in cond:
            continue
        indent = len(line) - len(stripped)
        end = j
        for k in range(j + 1, len(lines)):
            text = lines[k]
            if not text.strip():
                continue
            if len(text) - len(text.lstrip()) <= indent:
                break
            end = k
        return i, end
    raise LookupError(f"no `if op ...` branch names {quoted}; try --function or --lines")


def table_mutants(lines: list[str], op: str, limit: int = 6) -> list[Mutant]:
    """Mutants for an op translated through a lookup table: `"op": "Value"` inside a dict literal (F1, LDS_ATOMICS,
    ...) gets each other value of the same table in turn (at most `limit`), e.g. "InverseSqrt" -> "Sqrt"."""
    out: list[Mutant] = []
    entry = re.compile(rf'"{re.escape(op)}"\s*:\s*"([^"]+)"')
    for i, line in enumerate(lines):
        m = entry.search(line.split("#", 1)[0])
        if not m:
            continue
        start = i                                     # the dict literal holding the entry: from its `{` ...
        while start > 0 and "{" not in lines[start]:
            start -= 1
        depth, end = 0, start                         # ... to the line that closes it
        for k in range(start, len(lines)):
            depth += lines[k].count("{") - lines[k].count("}")
            end = k
            if depth <= 0 and k >= i:
                break
        values: list[str] = []
        for k in range(start, end + 1):
            for v in re.findall(r'"[^"]+"\s*:\s*"([^"]+)"', lines[k].split("#", 1)[0]):
                if v != m.group(1) and v not in values:
                    values.append(v)
        for v in values[:limit]:
            new = line[:m.start(1)] + v + line[m.end(1):]
            out.append(Mutant(i + 1, f'table entry "{op}": "{m.group(1)}" -> "{v}"', line, new))
    return out


def branch_ops(lines: list[str], first: int) -> list[str]:
    """The op names a branch's condition lists (its string literals), in order."""
    cond, j = lines[first], first
    while not cond.rstrip().endswith(":") and j + 1 < len(lines):
        j += 1
        cond += lines[j]
    return re.findall(r'"([a-z][a-z0-9_]*)"', cond)


def find_function(lines: list[str], name: str) -> tuple[int, int]:
    for i, line in enumerate(lines):
        if re.match(rf"\s*def {re.escape(name)}\(", line):
            indent = len(line) - len(line.lstrip())
            end = i
            for k in range(i + 1, len(lines)):
                text = lines[k]
                if text.strip() and len(text) - len(text.lstrip()) <= indent:
                    break
                if text.strip():
                    end = k
            return i, end
    raise LookupError(f"no method named {name}")


def mutants_for(lines: list[str], first: int, last: int) -> list[Mutant]:
    out: list[Mutant] = []
    seen: set[tuple[int, str]] = set()

    def add(i: int, label: str, new: str) -> None:
        if new != lines[i] and (i, new) not in seen:
            seen.add((i, new))
            out.append(Mutant(i + 1, label, lines[i], new))

    for i in range(first, last + 1):
        line = lines[i]
        code = line.split("#", 1)[0] if not line.lstrip().startswith(("f\"", "\"", "'")) else line
        if not code.strip() or code.lstrip().startswith("#"):
            continue
        for a, b in PAIRS:
            for x, y in ((a, b), (b, a)):
                for m in re.finditer(rf"(?<![A-Za-z]){x}(?![A-Za-z])", code):
                    add(i, f"{x} -> {y}", line[:m.start()] + y + line[m.end():])
        if "emit(" in code or "OpSelect" in code:
            for m in CU.finditer(code):
                text = m.group(1)
                value = int(text, 0)
                if value >= 0xFFFFFF00:
                    continue
                for delta in (1, -1):
                    v = value + delta
                    if v < 0:
                        continue
                    shown = hex(v) if text.lower().startswith("0x") else str(v)
                    add(i, f"self.cu({text}) -> self.cu({shown})", line[:m.start(1)] + shown + line[m.end(1):])
            m = SELECT.search(code)
            if m:
                swapped = m.group(1) + m.group(4) + m.group(3) + m.group(2)
                add(i, "OpSelect arms swapped", line[:m.start()] + swapped + line[m.end():])
            m = LAST_OPERAND.search(code)
            if m:
                add(i, f"last operand {m.group(2)} -> {m.group(1)}",
                    line[:m.start()] + f"{m.group(1)} {m.group(1)}{m.group(3)}" + line[m.end():])
    return out


def run_test(test_file: str, tools_dir: str, timeout: int) -> tuple[bool, str]:
    """True when the test file passes against tools_dir."""
    here = os.path.dirname(os.path.abspath(test_file))
    path = os.pathsep.join(filter(None, [here, os.path.join(ROOT, "tests", "semantic"), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, LEONIDA_TOOLS_DIR=tools_dir, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=path)
    try:
        p = subprocess.run([sys.executable, os.path.abspath(test_file)], env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s"
    tail = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, tail[-1] if tail else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("test_file")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--op", help="mutate the if/elif branch that names this op")
    where.add_argument("--function", help="mutate this method of the translator")
    where.add_argument("--lines", help="mutate this 1-based line range, e.g. 8375-8414")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per test run (default 300)")
    ap.add_argument("--list", action="store_true", help="only list the mutants")
    args = ap.parse_args(argv)

    src_path = os.path.join(ROOT, "tools", TRANSLATOR)
    with open(src_path, encoding="utf-8", newline="") as f:
        raw = f.read()
    sep = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.replace("\r\n", "\n").split("\n")
    table: list[Mutant] = table_mutants(lines, args.op) if args.op else []
    try:
        if args.op:
            try:
                first, last = find_branch(lines, args.op)
            except LookupError:
                if not table:
                    raise LookupError(f'no `if op ...` branch or lookup-table entry names "{args.op}"; '
                                      "try --function or --lines") from None
                first = last = -1                     # table-driven only
        elif args.function:
            first, last = find_function(lines, args.function)
        else:
            a, b = (int(x) for x in args.lines.split("-"))
            if not 1 <= a <= b <= len(lines):
                raise LookupError(f"line range {args.lines} is outside 1..{len(lines)}")
            first, last = a - 1, b - 1
    except (LookupError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mutants = (mutants_for(lines, first, last) if first >= 0 else []) + table
    where = f"lines {first + 1}-{last + 1}" if first >= 0 else "no branch"
    print(f"{TRANSLATOR} {where}" + (f" + {len(table)} table-entry mutants" if table else "") +
          f": {len(mutants)} mutants")
    if args.op and first >= 0:
        named = branch_ops(lines, first)
        if len(named) > 1:
            # A shared branch also holds the other ops' code; a test of one op can't be expected to catch those.
            print(f"the branch also handles: {', '.join(o for o in named if o != args.op)}")
    if args.list:
        for m in mutants:
            print(f"  line {m.line}: {m.label}")
        return 0
    if not mutants:
        return 0

    with tempfile.TemporaryDirectory(prefix="semantic-mutants-") as tmp:
        tools = os.path.join(tmp, "tools")
        shutil.copytree(os.path.join(ROOT, "tools"), tools, ignore=shutil.ignore_patterns("__pycache__", "rpf9dump"))
        target = os.path.join(tools, TRANSLATOR)
        ok, last_line = run_test(args.test_file, tools, args.timeout)
        if not ok:
            print(f"error: the test file fails on the unmutated translator ({last_line})", file=sys.stderr)
            return 2
        survived = 0
        t0 = time.time()
        for m in mutants:
            mutated = list(lines)
            mutated[m.line - 1] = m.new
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(sep.join(mutated))
            ok, last_line = run_test(args.test_file, tools, args.timeout)
            if ok:
                survived += 1
            print(f"  {'SURVIVED' if ok else 'caught  '}  line {m.line}: {m.label}"
                  + ("" if ok else f"  ({last_line})"))
        print(f"{len(mutants) - survived} of {len(mutants)} mutants caught, {survived} survived "
              f"({time.time() - t0:.0f} s)")
        return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
