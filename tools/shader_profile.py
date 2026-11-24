"""Profile the disassembled shader corpus: which RDNA instructions the translator must support.

Usage:
    python -I shader_profile.py <disasm dir> [--out profile.json] [--per-shader]

Counts each mnemonic (number of shaders using it and total occurrences), groups them by encoding family,
classifies instructions (ALU vs memory vs branch), computes operation ratios, and summarises register pressure
per stage (VGPR/SGPR counts from `profile_shader`; `--per-shader` also lists every shader).
In the JSON report, ratios.alu_to_memory is null when the corpus has ALU instructions and no memory instructions.
`categories_by_stage` uses the carve index first, then a recognized stage prefix in the filename, otherwise `?`.

`undecoded` counts the llvm-mc 'invalid instruction encoding' warnings shader_disasm.py copies into each .s file.
by_word lists the four bytes at each warning position, their total occurrences and how many shaders contain them.
unresolved counts warnings whose position or echoed input gives no complete word. Results depend on LLVM's
version, target and input range, which can include embedded data.

The terminal prints warning totals. The JSON report includes the bytes, so keep reports from real shaders local
(see CONTRIBUTING.md).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

FAMILIES = [("s_buffer_", "smem"), ("s_load", "smem"), ("s_", "scalar"), ("v_", "vector"), ("ds_", "lds"),
            ("buffer_", "buffer"), ("tbuffer_", "buffer"), ("image_", "image"), ("global_", "global"),
            ("flat_", "flat"), ("scratch_", "scratch"), ("exp", "export")]

STAGE_PREFIX = re.compile(r"(ps|vs|gs|hs|ds|cs|ms|as|lib)_", re.IGNORECASE | re.ASCII)
VGPR_RANGE = re.compile(r"\bv\[(\d+):(\d+)\]")
VGPR_SINGLE = re.compile(r"\bv(\d+)\b")
SGPR_RANGE = re.compile(r"\bs\[(\d+):(\d+)\]")
SGPR_SINGLE = re.compile(r"\bs(\d+)\b")
SPECIAL_REG = re.compile(r"\b(vcc|vcc_lo|vcc_hi|exec|exec_lo|exec_hi|m0|scc|flat_scratch|flat_scratch_lo|flat_scratch_hi)\b")


def family(m):
    for p, f in FAMILIES:
        if m.startswith(p):
            return f
    return "other"


def classify_instruction(m: str) -> str:
    """Classify an RDNA mnemonic into 'alu', 'memory', 'branch', 'export', or 'other'."""
    if not m:
        return "other"
    m = m.lower()
    if m.startswith("exp"):
        return "export"
    if m == "s_branch" or m.startswith("s_cbranch_") or m in ("s_setpc_b64", "s_swappc_b64", "s_rfe_b64", "s_call_b64"):
        return "branch"
    if m.startswith(("s_buffer_", "s_load", "s_store_", "s_dcache_", "buffer_", "tbuffer_", "image_", "global_", "flat_", "scratch_", "ds_")):
        return "memory"
    if m.startswith("v_"):
        return "alu"
    if m.startswith("s_"):
        if m in ("s_barrier", "s_nop", "s_endpgm", "s_trap", "s_sleep", "s_icache_inv") or \
           m.startswith(("s_waitcnt", "s_sendmsg", "s_setreg_", "s_getreg_", "s_setprio")):
            return "other"
        return "alu"
    return "other"


def is_alu(m: str) -> bool:
    return classify_instruction(m) == "alu"


def is_memory(m: str) -> bool:
    return classify_instruction(m) == "memory"


def is_branch(m: str) -> bool:
    return classify_instruction(m) == "branch"


def count_categories(mnemonics) -> dict:
    """Count instruction occurrences by high-level category (alu, memory, branch, export, other)."""
    counts = Counter()
    if isinstance(mnemonics, dict):
        for m, count in mnemonics.items():
            counts[classify_instruction(m)] += count
    else:
        for m in mnemonics:
            counts[classify_instruction(m)] += 1
    total = sum(counts.values())
    return {
        "alu": counts["alu"],
        "memory": counts["memory"],
        "branch": counts["branch"],
        "export": counts["export"],
        "other": counts["other"],
        "total": total,
    }


def compute_ratios(counts: dict) -> dict:
    """Calculate ALU vs Memory vs Branch ratios relative to total instructions,
    and relative ratios like alu_to_memory."""
    total = counts.get("total", 0)
    if total <= 0:
        total = sum(counts.get(k, 0) for k in ("alu", "memory", "branch", "export", "other"))
    alu = counts.get("alu", 0)
    mem = counts.get("memory", 0)
    branch = counts.get("branch", 0)
    return {
        "alu_ratio": round(alu / total, 6) if total > 0 else 0.0,
        "memory_ratio": round(mem / total, 6) if total > 0 else 0.0,
        "branch_ratio": round(branch / total, 6) if total > 0 else 0.0,
        "alu_to_memory": round(alu / mem, 6) if mem > 0 else (float("inf") if alu > 0 else 0.0),
    }


def parse_registers(line: str) -> dict:
    """Extract all VGPRs, SGPRs, and special registers referenced in a single disassembly line."""
    code, _, _ = line.partition(";")
    vgprs = set()
    for lo_s, hi_s in VGPR_RANGE.findall(code):
        lo, hi = int(lo_s), int(hi_s)
        vgprs.update(range(lo, hi + 1))
    code_no_vr = VGPR_RANGE.sub("", code)
    for v_s in VGPR_SINGLE.findall(code_no_vr):
        vgprs.add(int(v_s))

    sgprs = set()
    for lo_s, hi_s in SGPR_RANGE.findall(code):
        lo, hi = int(lo_s), int(hi_s)
        sgprs.update(range(lo, hi + 1))
    code_no_sr = SGPR_RANGE.sub("", code)
    for s_s in SGPR_SINGLE.findall(code_no_sr):
        sgprs.add(int(s_s))

    specials = set(SPECIAL_REG.findall(code))
    return {"vgprs": vgprs, "sgprs": sgprs, "special": specials}


def instruction_lines(lines):
    for line in lines:
        if line.startswith("; ---"):
            break
        if line.startswith("\t"):
            code = line.partition(";")[0].strip()
            if code and not code.startswith("."):
                yield code


def register_footprint(lines_or_text) -> dict:
    """Calculate the register footprint (max index, allocation count, and unique sets)
    across disassembly lines or text."""
    if isinstance(lines_or_text, str):
        lines = lines_or_text.splitlines()
    else:
        lines = lines_or_text

    all_vgprs = set()
    all_sgprs = set()
    all_specials = set()

    for line in instruction_lines(lines):
        regs = parse_registers(line)
        all_vgprs.update(regs["vgprs"])
        all_sgprs.update(regs["sgprs"])
        all_specials.update(regs["special"])

    max_vgpr = max(all_vgprs) if all_vgprs else -1
    max_sgpr = max(all_sgprs) if all_sgprs else -1

    return {
        "max_vgpr": max_vgpr,
        "max_sgpr": max_sgpr,
        "vgpr_count": max_vgpr + 1 if max_vgpr >= 0 else 0,
        "sgpr_count": max_sgpr + 1 if max_sgpr >= 0 else 0,
        "unique_vgprs": len(all_vgprs),
        "unique_sgprs": len(all_sgprs),
        "vgprs": sorted(all_vgprs),
        "sgprs": sorted(all_sgprs),
        "special": sorted(all_specials),
    }


def profile_shader(text: str) -> dict:
    """Profile a single shader's disassembly: mnemonics, categories, ratios, and register footprint."""
    lines = text.splitlines()
    mnemonics = Counter(line.split(maxsplit=1)[0] for line in instruction_lines(lines))

    cats = count_categories(mnemonics)
    ratios = compute_ratios(cats)
    reg_fp = register_footprint(lines)

    return {
        "total_instructions": cats["total"],
        "mnemonics": dict(mnemonics),
        "categories": cats,
        "ratios": ratios,
        "registers": reg_fp,
    }


LLVM_MESSAGES = "; --- llvm-mc messages ---"    # the header shader_disasm.py writes above llvm-mc's stderr
WARNING_LOCATION = re.compile(r"; <stdin>:[1-9][0-9]*:([1-9][0-9]*)")
BYTE_ECHO = re.compile(r"; [ \t]*0x[0-9a-fA-F]{2}(?=[ \t]|$)")
WORD = re.compile(r"[ \t]+".join([r"0x([0-9a-fA-F]{2})"] * 4) + r"(?!\S)")    # four complete 0xNN tokens


def warning_words(text: str):
    """Words LLVM warned about in one .s file's llvm-mc messages section, as (words, unresolved).

    Each 'invalid instruction encoding' warning gives one word: the four complete 0xNN tokens starting exactly
    at its one-based column of the source line LLVM echoes right after it, as e.g. 'ff ff ff d7'. A warning whose
    position, echo or tokens don't give such a word counts as unresolved instead."""
    if LLVM_MESSAGES not in text:    # clean shaders have no messages section
        return [], 0
    lines = text.splitlines()
    try:
        start = lines.index(LLVM_MESSAGES) + 1
    except ValueError:
        return [], 0
    words, unresolved = [], 0
    for index in range(start, len(lines)):
        head = lines[index]
        if not head.startswith("; <stdin>:"):
            continue
        location, _, message = head.partition(": ")
        if message != "warning: invalid instruction encoding":
            continue
        echo = lines[index + 1] if index + 1 < len(lines) else ""
        position = WARNING_LOCATION.fullmatch(location)
        word = None
        # A column longer than the echo length in decimal cannot be in range. Check before int() to
        # keep even arbitrarily long malformed locations from aborting the scan.
        if position and len(position[1]) <= len(str(len(echo))) and BYTE_ECHO.match(echo):
            col = int(position[1])
            # The producer added "; ". At a token start, echo[col] is the preceding whitespace and
            # echo[col + 1] is the first byte's '0' (including the column-one case).
            if col < len(echo) - 1 and echo[col] in " \t":
                word = WORD.match(echo, col + 1)
        if word:
            words.append(" ".join(word.groups()).lower())
        else:
            unresolved += 1
    return words, unresolved


def register_summary(reg_stage) -> dict:
    """Per-stage VGPR/SGPR pressure from (vgpr_count, sgpr_count) pairs: max, mean and a VGPR histogram.

    The histogram buckets are upper bounds on the VGPR count (wave32 RDNA2 allocates in blocks of 8 and caps a
    wave at 256), so the counts show how many shaders fit each occupancy step."""
    out = {}
    for stage in sorted(reg_stage):
        pairs = reg_stage[stage]
        if not pairs:
            continue
        v = [p[0] for p in pairs]
        s = [p[1] for p in pairs]
        hist = Counter()
        for c in v:
            hist[next(b for b in (32, 64, 128, 256) if c <= b) if c <= 256 else ">256"] += 1
        out[stage] = {"shaders": len(pairs),
                      "vgpr_max": max(v), "vgpr_mean": round(sum(v) / len(v), 2),
                      "sgpr_max": max(s), "sgpr_mean": round(sum(s) / len(s), 2),
                      "vgpr_hist": {str(k): hist[k] for k in (32, 64, 128, 256, ">256") if hist[k]}}
    return out


def carve_index_path(disasm_dir):
    return os.path.join(os.path.dirname(os.path.realpath(disasm_dir)), "index.json")


def stage_map(disasm_dir) -> dict:
    """{shader file stem: stage} from shader_carve.py's index.json next to the disasm folder (the stage comes from
    each container's program header), as rdna2spirv.stage_of reads it. Most corpus names carry no stage prefix,
    so the name is only a fallback for shaders the index doesn't list. Empty when there is no readable index."""
    path = carve_index_path(disasm_dir)
    try:
        with open(path, encoding="utf-8-sig") as f:
            entries = json.load(f)
        return {os.path.splitext(e["file"])[0]: e["stage"] for e in entries
                if isinstance(e, dict) and isinstance(e.get("file"), str) and isinstance(e.get("stage"), str)}
    except (OSError, ValueError, TypeError):
        return {}


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--per-shader", action="store_true",
                    help="also list each shader's instruction count and VGPR/SGPR footprint")
    a = ap.parse_args()
    if not os.path.isdir(a.src):
        what = "is not a folder" if os.path.exists(a.src) else "does not exist"
        print(f"ERROR: {a.src} {what}. Give the folder of .s files shader_disasm.py writes.", file=sys.stderr)
        return 2
    path = a.out or os.path.join(a.src, "..", "profile.json")
    index = carve_index_path(a.src)
    if same_output_path(index, path):
        print(f"ERROR: output would overwrite the shader carve index {index}", file=sys.stderr)
        return 2
    if os.path.isdir(path):
        print(f"ERROR: cannot write {path}: destination is a directory, not a file", file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(out_dir):        # checked before the scan, which takes minutes on the whole corpus
        print(f"ERROR: the folder for {path} does not exist", file=sys.stderr)
        return 2
    try:
        with os.scandir(a.src) as entries:
            shader_files = [entry for entry in entries if entry.name.endswith(".s") and entry.is_file()]
    except OSError as e:
        print(f"ERROR: cannot list {a.src}: {e.strerror or e}", file=sys.stderr)
        return 2
    for entry in shader_files:
        if same_output_path(entry.path, path):
            print(f"ERROR: output would overwrite input shader {entry.path}", file=sys.stderr)
            return 2
    occ, users = Counter(), Counter()
    word_occ, word_users = Counter(), Counter()    # warned-at word -> warnings, shaders
    unresolved = 0
    categories_by_stage = defaultdict(Counter)
    reg_stage = defaultdict(list)    # stage -> [(vgpr_count, sgpr_count)]
    shaders = []
    stages = stage_map(a.src)
    n = 0
    for entry in shader_files:
        name = entry.name
        n += 1
        stage = stages.get(name[:-2])
        if not stage:
            prefix = STAGE_PREFIX.match(name)
            stage = prefix[1].lower() if prefix else "?"
        try:
            with open(entry.path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            print(f"ERROR: cannot read {entry.path}: {e.strerror or e}", file=sys.stderr)
            return 2
        prof = profile_shader(text)
        regs = prof["registers"]
        reg_stage[stage].append((regs["vgpr_count"], regs["sgpr_count"]))
        if a.per_shader:
            shaders.append({"name": name[:-2], "stage": stage, "instructions": prof["total_instructions"],
                            "vgpr_count": regs["vgpr_count"], "sgpr_count": regs["sgpr_count"]})
        occ.update(prof["mnemonics"])
        users.update(prof["mnemonics"].keys())
        categories_by_stage[stage].update(prof["categories"])
        words, bad = warning_words(text)
        word_occ.update(words)
        word_users.update(set(words))
        unresolved += bad
    fam = defaultdict(lambda: [0, 0])
    for m, c in occ.items():
        fam[family(m)][0] += 1
        fam[family(m)][1] += c
    cats = count_categories(occ)
    ratios = compute_ratios(cats)
    if ratios["alu_to_memory"] == float("inf"):
        ratios["alu_to_memory"] = None
    ranked_users = sorted(users.items(), key=lambda item: (-item[1], item[0]))
    out = {"shaders": n, "mnemonics": len(occ),
           "families": {k: {"mnemonics": v[0], "occurrences": v[1]} for k, v in sorted(fam.items())},
           "categories": cats,
           "categories_by_stage": dict(sorted(categories_by_stage.items())),
           "ratios": ratios,
           "by_mnemonic": [{"m": m, "shaders": shader_count, "count": occ[m]}
                           for m, shader_count in ranked_users],
           "registers": register_summary(reg_stage),
           "undecoded": {"warnings": sum(word_occ.values()) + unresolved, "unresolved": unresolved,
                         "by_word": [{"word": w, "shaders": word_users[w], "count": word_occ[w]}
                                     for w in sorted(word_occ, key=lambda w: (-word_occ[w], -word_users[w], w))]}}
    if a.per_shader:
        out["per_shader"] = sorted(shaders, key=lambda e: e["name"])
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, allow_nan=False)
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e.strerror or e}", file=sys.stderr)
        return 2
    print(f"{n} shaders, {len(occ)} distinct mnemonics -> {os.path.abspath(path)}")
    for k, v in sorted(fam.items(), key=lambda item: (-item[1][1], item[0])):
        print(f"  {k:8} {v[0]:4} mnemonics {v[1]:>12,} uses")
    print(f"  ALU: {cats['alu']} ({ratios['alu_ratio'] * 100:.1f}%)  Memory: {cats['memory']} ({ratios['memory_ratio'] * 100:.1f}%)  Branch: {cats['branch']} ({ratios['branch_ratio'] * 100:.1f}%)")
    cum, total = 0, sum(occ.values())
    for i, (m, c) in enumerate(occ.most_common(), 1):
        cum += c
        if total > 0 and cum / total >= 0.99:
            print(f"  {i} mnemonics cover 99% of all instructions")
            break
    for st, r in out["registers"].items():
        print(f"  {st:2} registers: VGPR max {r['vgpr_max']} mean {r['vgpr_mean']}, "
              f"SGPR max {r['sgpr_max']} mean {r['sgpr_mean']} ({r['shaders']} shaders)")
    print("  most widespread:", ", ".join(f"{m} ({count})" for m, count in ranked_users[:12]))
    print(f"  undecoded: {out['undecoded']['warnings']} LLVM warnings, {len(word_occ)} distinct words, "
          f"{unresolved} unresolved (words only in the JSON report)")


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
