#!/usr/bin/env python3
"""Give set 0's descriptor arrays fixed, non-overlapping Metal argument-buffer IDs (SHADER-161).

SPIRV-Cross (1.4.363.0) puts every runtime descriptor array of an argument buffer at one ID: with Leonida's
set 0 (sampled images at binding 0, samplers at 1, more image arrays at 3+) it comments all but one out as
"Overlapping binding", so the shader can't reach them. This rewrites a translated module for a native Metal
path only (the Vulkan module and layout stay as they are):

- each set-0 runtime descriptor array becomes a sized array with that binding's capacity from
  runtime/src/vulkan_bindless.h (`bindings`), and
- each set-0 Binding becomes the first ID of its range: the capacities of all lower bindings added up.

Convert the result with `spirv-cross --msl --msl-argument-buffers --msl-argument-buffer-tier 1
--msl-decoration-binding`, so the IDs are the same in every shader, whichever bindings it uses. The argument
buffer then holds binding b's element i at ID `metal_id(b) + i` (`--print-layout` prints the table).

Usage: spirv_metal_set0.py in.spv out.spv     |     spirv_metal_set0.py --print-layout
Needs spirv-dis and spirv-as (SPIRV-Tools) on PATH, or SPIRV_DIS / SPIRV_AS.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDLESS_H = os.path.join(ROOT, "runtime", "src", "vulkan_bindless.h")


def capacities(path=BINDLESS_H):
    """Descriptor counts by binding number, from the `bindings` table of vulkan_bindless.h."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"std::array<Binding,\s*(\d+)>\s*bindings\s*=\s*\{\{(.*?)\}\};", src, re.S)
    if not m:
        raise SystemExit(f"{path}: no `bindings` table")
    counts = [int(c) for c in re.findall(r"\{VK_DESCRIPTOR_TYPE_\w+,\s*(\d+)\}", m.group(2))]
    if len(counts) != int(m.group(1)):
        raise SystemExit(f"{path}: `bindings` declares {m.group(1)} entries, found {len(counts)}")
    return counts


def metal_ids(counts):
    """First argument-buffer ID of each binding: the capacities of the lower bindings added up."""
    ids, total = [], 0
    for c in counts:
        ids.append(total)
        total += c
    return ids


def _fresh_id(base, used):
    """Return `base` if unused, otherwise `base_1`, `base_2`, ... and record it in `used`."""
    cand = base
    i = 1
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def rewrite(text, counts):
    """Rewrite the disassembly (spirv-dis --raw-id is not needed; names are kept) of one module."""
    ids = metal_ids(counts)
    lines = text.splitlines()
    sets, binds = {}, {}
    for ln in lines:
        m = re.match(r"\s*OpDecorate (%\S+) DescriptorSet (\d+)$", ln)
        if m:
            sets[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*OpDecorate (%\S+) Binding (\d+)$", ln)
        if m:
            binds[m.group(1)] = int(m.group(2))
    set0 = {v: b for v, b in binds.items() if sets.get(v) == 0}
    for v, b in set0.items():
        if b >= len(counts):
            raise SystemExit(f"{v}: binding {b} isn't in vulkan_bindless.h")

    defs = {}       # result id -> instruction text
    definition_positions: dict[str, int] = {}
    for idx, ln in enumerate(lines):
        m = re.match(r"\s*(%\S+) = (.*)$", ln)
        if m:
            defs[m.group(1)] = m.group(2)
            definition_positions[m.group(1)] = idx
    uint = next((r for r, d in defs.items() if d == "OpTypeInt 32 0"), None)

    candidates = []
    for order, (v, b) in enumerate(set0.items()):
        var = re.match(r"OpVariable (%\S+) UniformConstant$", defs.get(v, ""))
        if not var:
            continue    # the hash table (a storage buffer) and other non-array resources keep their type
        old_ptr = var.group(1)
        ptr = re.match(r"OpTypePointer UniformConstant (%\S+)$", defs.get(old_ptr, ""))
        rt = re.match(r"OpTypeRuntimeArray (%\S+)$", defs.get(ptr.group(1), "")) if ptr else None
        if not rt:
            continue
        candidates.append((b, definition_positions[old_ptr], order, v, old_ptr, rt.group(1)))

    if candidates:
        if uint is None:
            raise SystemExit("module has no 32-bit uint type for the array lengths")
        first_pointer = min(candidate[1] for candidate in candidates)
        uint_position = definition_positions[uint]
        if uint_position > first_pointer:
            # The generated length may precede an original declaration of its integer type.
            lines.insert(first_pointer, lines.pop(uint_position))

    used_ids = set(defs) | set(sets) | set(binds)
    cap_by_bind = {}    # binding -> OpConstant id (emitted once per binding)
    ptr_by_key = {}     # (binding, elem_type) -> new OpTypePointer id
    new_types = []      # inserted right after the variable's pointer type is declared
    retype = {}         # resized variable or copied pointer ID -> new pointer type
    for b, _, _, v, old_ptr, elem in sorted(candidates):
        if (b, elem) in ptr_by_key:
            retype[v] = ptr_by_key[(b, elem)]
            continue
        decl = []
        cap = cap_by_bind.get(b)
        if cap is None:
            cap = _fresh_id(f"%msl_cap_b{b}", used_ids)
            cap_by_bind[b] = cap
            decl.append(f"{cap} = OpConstant {uint} {counts[b]}")
        arr = _fresh_id(f"%msl_arr_b{b}", used_ids)
        nptr = _fresh_id(f"%msl_ptr_b{b}", used_ids)
        decl.extend([
            f"{arr} = OpTypeArray {elem} {cap}",
            f"{nptr} = OpTypePointer UniformConstant {arr}",
        ])
        new_types.append((old_ptr, decl))
        ptr_by_key[(b, elem)] = nptr
        retype[v] = nptr

    out = []
    for ln in lines:
        m = re.match(r"(\s*OpDecorate (%\S+) Binding )(\d+)$", ln)
        if m and m.group(2) in set0:
            ln = f"{m.group(1)}{ids[int(m.group(3))]}"
        m = re.match(r"(\s*(%\S+) = OpVariable )(%\S+)( UniformConstant)$", ln)
        if m and m.group(2) in retype:
            ln = f"{m.group(1)}{retype[m.group(2)]}{m.group(4)}"
        # Retype copies by source ID so arrays sharing a type in other sets are unaffected.
        m = re.match(r"(\s*(%\S+) = OpCopyObject )%\S+ (%\S+)$", ln)
        if m and m.group(3) in retype:
            ptr = retype[m.group(3)]
            retype[m.group(2)] = ptr
            ln = f"{m.group(1)}{ptr} {m.group(3)}"
        out.append(ln)
        m = re.match(r"\s*(%\S+) = OpTypePointer", ln)
        if m:
            for after, decl in new_types:
                if after == m.group(1):
                    out.extend(decl)
    return "\n".join(out) + "\n"


def main(argv):
    rest = argv[1:]
    if rest and rest[0] in ("-h", "--help"):
        for line in __doc__.strip().splitlines()[-2:]:
            print(line)
        return 0
    counts = capacities()
    if argv[1:] == ["--print-layout"]:
        for b, (c, i) in enumerate(zip(counts, metal_ids(counts))):
            print(f"binding {b:2}: {c:5} descriptors at IDs {i}..{i + c - 1}")
        return 0
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    dis, asm = _tools()
    # errors="replace": on Windows text=True decodes with the ANSI code page, and a program that answers in UTF-8
    # (a path with non-ASCII letters in its error message) would otherwise kill the reader thread and lose stderr.
    text = _run([dis, argv[1]], capture_output=True, text=True, errors="replace").stdout
    env = re.search(r"; Version: (\d+)\.(\d+)", text)
    target = f"spv{env.group(1)}.{env.group(2)}" if env else "vulkan1.2"
    _run([asm, "--target-env", target, "-", "-o", argv[2]], input=rewrite(text, counts), text=True)
    return 0


def _tools():
    return os.environ.get("SPIRV_DIS", "spirv-dis"), os.environ.get("SPIRV_AS", "spirv-as")


def _run(cmd, **kwargs):
    """subprocess.run(check=True). Windows does not name the program it could not start, so name it here."""
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except OSError as e:
        if e.filename is None:
            e.filename = cmd[0]
        raise


def cli(argv):
    """main() for the command line: a SPIRV-Tools program that cannot be started, or that rejects its input, is
    one ERROR line and exit 1 instead of a traceback. main() itself still raises, for callers that catch it."""
    try:
        return main(argv)
    except subprocess.CalledProcessError as e:
        program = e.cmd[0] if isinstance(e.cmd, (list, tuple)) else e.cmd
        said = next((line.strip() for line in (e.stderr or "").splitlines() if line.strip()), "")
        print(f"ERROR: {program} failed with exit status {e.returncode}" + (f": {said}" if said else ""),
              file=sys.stderr)
    except OSError as e:
        hint = " (SPIRV-Tools must be on PATH, or set SPIRV_DIS / SPIRV_AS)" if e.filename in _tools() else ""
        print(f"ERROR: {e.filename or 'spirv_metal_set0.py'}: {e.strerror or e}{hint}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(cli(sys.argv))
