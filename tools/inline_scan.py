"""Find the inline command-list writes in an x64 image and say whether each one checks the cursor limit first.

Usage:
    python -I tools/inline_scan.py <image.exe|image.dll|dumpbin-listing.txt> [--base 0x140000000] [--sites]

Code compiled against the console headers writes some commands into the list's buffer itself
(docs/d3d12_command_objects.md, "The list object is wider than its vtable"): it compares the write cursor
`[list+18h]` with a limit (`[list+28h]` for a dispatch, `[list+30h]` for root arguments in the package's DLLs),
writes the packet through the cursor, advances it and stores it back; when the buffer is full it calls the vtable
method instead. A layer that keeps the cursor at the limit sees every command only if every such write checks
first. This tool finds each cursor store-back `mov [list+18h],reg` that follows stores through the cursor, looks
back to the previous store-back of the same list, and sorts the site into:

    checked    a compare of the cursor against `[list+X]` (X != 18h), or against a register, followed by an
               unsigned branch; the branch target's vtable call is reported as the fallback slot
    UNCHECKED  no compare, but the cursor was loaded from the list and the list's own fields are touched
               (+140h..+143h, +344h, +544h): an inline write that a cursor-at-limit layer never sees
    unrelated  no compare and no list field: another structure with a pointer at +18h

A PE is disassembled with MSVC `dumpbin /disasm:nobytes` (it comes with the build tools the runtime needs) and
streamed, so an image of any size works in one pass; a saved dumpbin listing (with or without the byte column)
works too, on any OS. Nothing is executed. The output is RVAs, counts and constants, no code bytes.
An input with no instructions in it (an empty or other text file, a PE dumpbin prints no code for) exits 2
instead of reporting zero sites, so "UNCHECKED 0" always means something was scanned.
The matcher knows the instruction shapes MSVC emits for these writes (see tests/test_inline_scan.py); a shape it
does not know shows up as `unrelated`, so read a few of those by hand on a new image.
"""
import argparse
import collections
import re
import shutil
import subprocess
import sys
from pathlib import Path

LOOKBACK = 64         # instructions searched backwards from a store-back
FALLBACK_WINDOW = 16  # instructions after the branch target to look for the vtable call
PENDING_WINDOW = 400  # instructions to wait for a forward branch target
TAIL_WINDOW = 6       # instructions after a store-back that may still show a list field
REG64 = r"r(?:[abcd]x|[sd]i|bp|sp|8|9|1[0-5])"
DUMPBIN_GLOB = "Program Files*/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"

LINE = re.compile(r"^\s+([0-9A-F]{8,16}): (?:(?:[0-9A-F]{2} )+\s+)?([a-z][a-z0-9]*)\s*(.*)$")
LOAD = re.compile(rf"^({REG64}),qword ptr \[({REG64})\+18h\]$")
STORE_BACK = re.compile(rf"^qword ptr \[({REG64})\+18h\],({REG64})$")
CMP = re.compile(rf"^(?:({REG64}),qword ptr \[({REG64})\+([0-9A-F]+)h?\]|qword ptr \[({REG64})\+([0-9A-F]+)h?\],({REG64}))$")
LEA = re.compile(rf"^({REG64}),\[({REG64})\+([0-9A-F]+)h?\]$")
ADD = re.compile(rf"^({REG64}),([0-9A-F]+)h?$")
STORE = re.compile(r"^(?:byte|word|dword|qword|xmmword|ymmword) ptr \[(\w+)(?:\+([0-9A-F]+)h?)?\],(.+)$")
IMM = re.compile(r"^([0-9A-F]+)h?$")
COPY = re.compile(rf"^({REG64}),({REG64})$")
LIMIT_LOAD = re.compile(rf"^({REG64}),qword ptr \[({REG64})\+([0-9A-F]+)h?\]$")
LIST_FIELDS = re.compile(r"\[(\w+)(?:\+\w+\*\d)?\+0*(?:140|141|143|344|544)h\]")  # fields only a D3D12.X list has
VCALL = re.compile(rf"^qword ptr \[({REG64})\+([0-9A-F]+)h?\]$")
UNSIGNED_JCC = {"jae", "jb", "ja", "jbe"}
REG_LIMIT = -1  # limit compared through a register loaded outside the scanned span
JCC_TARGET = re.compile(r"^([0-9A-F]{8,16})$")


def read_lines(source: Path):
    if source.suffix.lower() in (".exe", ".dll"):
        dumpbin = shutil.which("dumpbin") or next(iter(sorted(Path("C:/").glob(DUMPBIN_GLOB))), None)
        if not dumpbin:
            sys.exit("dumpbin.exe not found: run from a VS developer shell or install the MSVC build tools")
        proc = subprocess.Popen([str(dumpbin), "/nologo", "/disasm:nobytes", str(source)],
                                stdout=subprocess.PIPE, text=True, errors="replace")
        yield from proc.stdout
        if proc.wait():
            sys.exit(f"dumpbin failed with exit code {proc.returncode}")
        return
    with open(source, encoding="utf-8", errors="replace") as f:
        yield from f


def instructions(lines):
    for line in lines:
        m = LINE.match(line)
        if m:
            yield int(m.group(1), 16), m.group(2), m.group(3).strip()


def classify(window, base, stored):
    """Look back from a store-back of `stored` into [base+18h]. Return a site dict or None."""
    chain, size, load, bases, broken = {stored}, 0, None, {base}, False
    touched = {stored}  # every register the cursor passed through, for counting stores
    span = []
    for addr, mnem, ops in reversed(window):
        if mnem == "ret" or (mnem == "mov" and (sb := STORE_BACK.match(ops)) and sb.group(1) in bases):
            break  # the previous site of this list, or the previous function
        span.append((addr, mnem, ops))
        if mnem == "mov" and (cp := COPY.match(ops)) and cp.group(1) in bases:
            bases.add(cp.group(2))  # `mov rbx,rcx`: the list is in both
        if load is None and not broken:
            if mnem == "mov" and (lm := LOAD.match(ops)) and lm.group(1) in chain and lm.group(2) in bases:
                load = addr
            elif mnem == "lea" and (le := LEA.match(ops)) and le.group(1) in chain:
                chain.discard(le.group(1))  # defined here; older values of it are not the cursor
                chain.add(le.group(2))
                touched.add(le.group(2))
                size += int(le.group(3), 16)
            elif mnem == "add" and (am := ADD.match(ops)) and am.group(1) in chain:
                size += int(am.group(2), 16)
            elif ops.split(",", 1)[0] in chain and mnem not in ("cmp", "test") and not mnem.startswith("j"):
                broken = True  # the stored value was computed from something other than the cursor
    span.reverse()
    stores, header, limit, target, evidence = 0, None, None, None, False
    limit_regs = {}  # register -> limit offset, for `mov rdx,[list+30h]` ... `cmp rcx,rdx`
    for i, (addr, mnem, ops) in enumerate(span):
        if mnem == "mov" and (lr := LIMIT_LOAD.match(ops)) and lr.group(2) in bases and int(lr.group(3), 16) != 0x18:
            limit_regs[lr.group(1)] = int(lr.group(3), 16)
        if mnem == "cmp" and (rr := COPY.match(ops)) and (rr.group(1) in touched or rr.group(1) in limit_regs):
            nxt = span[i + 1] if i + 1 < len(span) else None
            if nxt and nxt[1] in UNSIGNED_JCC:
                # the limit came from [list+X] here, or in a block that jumps to this one (REG_LIMIT)
                limit = limit_regs.get(rr.group(2), limit_regs.get(rr.group(1), REG_LIMIT))
                if jt := JCC_TARGET.match(nxt[2]):
                    target = int(jt.group(1), 16)
        if mnem.startswith(("mov", "vmov")) and (st := STORE.match(ops)) and st.group(1) in touched:
            stores += 1
            imm = IMM.match(st.group(3))
            if header is None and not st.group(2) and imm and mnem == "mov":
                header = int(imm.group(1), 16)
        if (lf := LIST_FIELDS.search(ops)) and lf.group(1) in bases:
            evidence = True
        if mnem == "cmp" and (cm := CMP.match(ops)):
            reg_base, disp = (cm.group(2), cm.group(3)) if cm.group(1) else (cm.group(4), cm.group(5))
            if reg_base in bases and int(disp, 16) != 0x18:
                limit = int(disp, 16)
                nxt = span[i + 1] if i + 1 < len(span) else None
                if nxt and nxt[1].startswith("j") and nxt[1] != "jmp" and (jt := JCC_TARGET.match(nxt[2])):
                    target = int(jt.group(1), 16)
    if not stores or base == "rsp":
        return None
    checked = limit is not None
    # A guard-free list write must load the cursor from the list nearby; without that it is some other structure.
    is_list = checked or (evidence and load is not None and not broken)
    return {"load": load, "base": base, "checked": checked, "list": is_list, "limit": limit, "size": size,
            "stores": stores, "header": header, "target": target, "fallback_slot": None,
            "bases": bases, "load_traced": load is not None and not broken}


def scan(lines, stats=None):
    """Yield one dict per cursor store-back site; fallback slots are filled in once the branch target is seen.
    `stats["instructions"]` (if a dict is passed) counts the instructions read."""
    stats = {} if stats is None else stats
    stats["instructions"] = 0
    window = collections.deque(maxlen=LOOKBACK)
    pending, watching, tail = [], [], []
    for idx, (addr, mnem, ops) in enumerate(instructions(lines)):
        stats["instructions"] = idx + 1
        for item in tail[:]:  # list fields written right after the store-back (e.g. clearing +140h) count too
            site, left = item
            tail.remove(item)
            if (lf := LIST_FIELDS.search(ops)) and lf.group(1) in site["bases"] and site["load_traced"]:
                site["list"] = True
            elif left > 1:
                tail.append((site, left - 1))
        for item in watching[:]:
            site, left = item
            watching.remove(item)
            if mnem == "call" and (vm := VCALL.match(ops)):
                site["fallback_slot"] = int(vm.group(2), 16) // 8
            elif left > 1:
                watching.append((site, left - 1))
        for item in pending[:]:
            site, opened = item
            if addr == site["target"]:
                watching.append((site, FALLBACK_WINDOW))
                pending.remove(item)
            elif idx - opened > PENDING_WINDOW:
                pending.remove(item)
        if mnem == "mov" and (sb := STORE_BACK.match(ops)):
            site = classify(window, sb.group(1), sb.group(2))
            if site:
                site["store_back"] = addr
                if site["target"] is not None:
                    pending.append((site, idx))
                if not site["list"]:
                    tail.append((site, TAIL_WINDOW))
                yield site
        window.append((addr, mnem, ops))


def limit_text(limit):
    return "none" if limit is None else "reg" if limit == REG_LIMIT else f"+0x{limit:x}"


def summarize(sites, instructions=None):
    checked = [s for s in sites if s["checked"]]
    unchecked = [s for s in sites if not s["checked"] and s["list"]]
    unrelated = [s for s in sites if not s["list"]]
    plain = lambda v: "none" if v is None else str(v)
    out = [] if instructions is None else [f"instructions scanned: {instructions}"]
    out += [f"list sites {len(checked) + len(unchecked)}: checked {len(checked)}, UNCHECKED {len(unchecked)} "
            f"(plus {len(unrelated)} cursor-like stores with no list field nearby, likely other structures)"]
    for title, key, group, fmt in (("limit offsets (checked)", "limit", checked, limit_text),
                                   ("fallback vtable slots (checked)", "fallback_slot", checked, plain),
                                   ("packet sizes, checked", "size", checked, plain),
                                   ("packet sizes, unchecked", "size", unchecked, plain)):
        hist = collections.Counter(s[key] for s in group)
        out.append(f"{title}: " + ", ".join(f"{fmt(k)} x{n}" for k, n in hist.most_common(12)))
    headers = collections.Counter(s["header"] for s in sites if s["header"] is not None)
    out.append("header dwords: " + ", ".join(f"0x{h:08x} x{n}" for h, n in headers.most_common(12)))
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--base", type=lambda s: int(s, 16), help="image base (default: inferred from the first site)")
    p.add_argument("--sites", action="store_true", help="print every site")
    args = p.parse_args()
    if not args.source.is_file():
        print(f"{args.source} not found", file=sys.stderr)
        return 2
    stats = {}
    sites = list(scan(read_lines(args.source), stats))
    if not stats["instructions"]:
        print(f"{args.source}: no instructions found (not a dumpbin listing, or an image without code)",
              file=sys.stderr)
        return 2
    base = args.base if args.base is not None else min((s["store_back"] for s in sites), default=0) & ~0xFFFFFFF
    print(summarize(sites, stats["instructions"]))
    if args.sites:
        rva = lambda a: "?" if a is None else f"+0x{a - base:x}"
        for s in sites:
            state = f"checked {limit_text(s['limit'])}" if s["checked"] else "UNCHECKED" if s["list"] else "unrelated"
            header = "-" if s["header"] is None else f"0x{s['header']:08x}"
            print(f"  load {rva(s['load'])} store {rva(s['store_back'])} [{s['base']}] {state} size {s['size']} "
                  f"stores {s['stores']} header {header} fallback {s['fallback_slot']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
