"""Count candidate cross-half wave ops in wave64 graphics shaders over the whole corpus.

    python -I tools/wave64_graphics_scan.py <work>/shaders/dxbc/disasm [--out wave64_graphics_scan.json]

design.md's Wave64 emulation row defers graphics (VS/PS/GS) wave64: emulation covers only compute and mesh
(Vulkan graphics stages have no workgroup memory), so on a 32-wide-subgroup GPU every wave op whose result
depends on lanes outside the executing 32-lane half sees only its own half. This tool sizes that gap before any
design work: it parses the disassembly the shaders stage leaves in <work>/shaders/dxbc/disasm and reports, per
stage, how many wave64 shaders contain each selected opcode. A hit is a *candidate*, not a verdict — whether a
given occurrence actually crosses the 32/32 half depends on operand provenance and run-time values, which a
static scan does not decide:

- whole-wave mask words: `v_mbcnt_lo/hi_u32_b32` (the hi word folds in the other half's exec bits) and
  `s_bcnt0/1_i32_b64`, `s_ff0/1_i32_b64`. The mask-word ops are reported split by source: an exec/vcc source
  takes the translator's `wave_mask` path outright; a register source does only if that register holds a lane
  mask (`holds_lane_mask` tracks register provenance, which a static scan cannot reproduce) — so "other source"
  occurrences stay candidates.
- lane-addressed exchanges: `v_readlane_b32`, `v_writelane_b32`, `v_permlane16_b32`, `v_permlanex16_b32`,
  `v_readfirstlane_b32` (ballot over the whole wave). Upper bound: a lane select below 32 stays on one half,
  and selects are usually dynamic SGPRs, unresolved here.
- whole-wave branches on a 64-bit mask: `s_cbranch_execz/_execnz/_vccz/_vccnz` — the `scc_from_mask` /
  `branch_condition` paths. Mask *arithmetic* (`s_and_saveexec_b64` and friends) is per-lane and exact; only
  the branch taken from it splits.

Wave64 classification reads the adjacent FSSM wave-size flag, as the translator does. Without a usable
header it falls back to mesh shader defaults and instruction markers. Stages come from the translator's
stage lookup. Opcodes are counted from parsed instructions only, never from comments or operand text.

Quad ops and `ds_swizzle`/`ds_bpermute` are left out on purpose: quads never cross the 32/32 half and the
translator models the permute ops per 32-lane half, matching hardware.

Standard library only. Nothing from the game is printed or written: counts only — names, code and data stay in
the work folder.
Usage: `python -I tools/wave64_graphics_scan.py --help` lists the options.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fssm_header  # noqa: E402
import rdna2spirv  # noqa: E402  (same directory; WAVE64_RE, stage_of, base_op)
from report_paths import same_report_path  # noqa: E402
from shader_ir import parse_file  # noqa: E402

# why each op can depend on lanes outside the executing half is in the docstring; this only gates counting
MASK_WORD_OPS = {"s_bcnt0_i32_b64", "s_bcnt1_i32_b64", "s_ff0_i32_b64", "s_ff1_i32_b64"}
SELECTED_OPS = MASK_WORD_OPS | {
    "v_mbcnt_lo_u32_b32",
    "v_mbcnt_hi_u32_b32",
    "v_readlane_b32",
    "v_writelane_b32",
    "v_permlane16_b32",
    "v_permlanex16_b32",
    "v_readfirstlane_b32",
    "s_cbranch_execz",
    "s_cbranch_execnz",
    "s_cbranch_vccz",
    "s_cbranch_vccnz",
}
GRAPHICS = ("vs", "ps", "gs")


def wave64_rule(stage, insns):
    """The translator's fallback when a shader has no FSSM wave-size flag."""
    if stage == "ms":
        return True
    return any(rdna2spirv.WAVE64_RE.search(" ".join([i["op"]] + [x.get("t", "") for x in i["args"]])) or
               (rdna2spirv.base_op(i["op"]).endswith("_b64") and
                any(x.get("t") in ("exec", "vcc") for x in i["args"]))
               for i in insns)


def scan_shader(stage, insns):
    """Selected-opcode occurrence counts for one parsed shader, or None when nothing matches."""
    occurrences, exec_vcc, other = Counter(), Counter(), Counter()
    for i in insns:
        op = rdna2spirv.base_op(i["op"])
        if op not in SELECTED_OPS:
            continue
        occurrences[op] += 1
        if op in MASK_WORD_OPS:
            sources = i["args"][1:]
            (exec_vcc if any(x.get("t") in ("exec", "vcc") for x in sources) else other)[op] += 1
    if not occurrences:
        return None
    return {"occurrences": occurrences,
            "mask_word_source": {"exec_vcc": exec_vcc, "other": other}}


def scanned_paths(disasm):
    """The one enumeration the scanner and the overwrite guard share: regular files named *.s (symlinks followed)."""
    with os.scandir(disasm) as entries:
        return sorted(entry.path for entry in entries
                      if entry.name.endswith(".s") and entry.is_file())


def metadata_paths(listings):
    """The FSSM headers and carve indexes the scan reads for its listings."""
    seen = set()
    for path in listings:
        name = os.path.splitext(os.path.basename(path))[0]
        yield rdna2spirv.native_paths(path, name)[0]
        meta = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "index.json")
        if meta not in seen:
            seen.add(meta)
            yield meta


def scan_directory(disasm):
    """Aggregate selected-opcode counts over one carve disasm directory. Returns per-stage dicts; no names."""
    per_stage = {stage: {"shaders": 0, "wave64": 0, "with_selected_op": 0, "occurrences": Counter(),
                         "mask_word_source": {"exec_vcc": Counter(), "other": Counter()}}
                 for stage in GRAPHICS}
    other = {"shaders": 0, "wave64": 0}
    paths = scanned_paths(disasm)
    for path in paths:
        info = parse_file(path)
        stage = rdna2spirv.stage_of(info["name"], path)
        fssm_path = rdna2spirv.native_paths(path, info["name"])[0]
        try:
            with open(fssm_path, "rb") as f:
                wave_size = fssm_header.wave_size(f.read(fssm_header.WAVE_FLAGS_OFFSET + 4))
        except OSError:
            wave_size = None
        wave64 = wave_size == 64 if wave_size is not None else wave64_rule(stage, info["insns"])
        if stage in per_stage:
            bucket = per_stage[stage]
            bucket["shaders"] += 1
            if not wave64:
                continue
            bucket["wave64"] += 1
            found = scan_shader(stage, info["insns"])
            if found:
                bucket["with_selected_op"] += 1
                bucket["occurrences"].update(found["occurrences"])
                for part in ("exec_vcc", "other"):
                    bucket["mask_word_source"][part].update(found["mask_word_source"][part])
        else:
            other["shaders"] += 1
            other["wave64"] += int(wave64)
    for stage in per_stage:
        bucket = per_stage[stage]
        bucket["occurrences"] = dict(sorted(bucket["occurrences"].items(), key=lambda kv: -kv[1]))
        bucket["mask_word_source"] = {part: dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
                                      for part, cnt in bucket["mask_word_source"].items()}
    return {"per_stage": per_stage, "other_stages": other}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Count candidate cross-half wave ops in wave64 graphics shaders (counts only, no names)")
    ap.add_argument("src", help="<work>/shaders/dxbc/disasm from the shaders stage")
    ap.add_argument("--out", help="write the same aggregates to a JSON file")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.src):
        ap.error(f"not a directory: {args.src}")
    if args.out:
        if os.path.isdir(args.out):
            ap.error(f"{args.out} is a directory, not a JSON file")
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if not os.path.isdir(out_dir):
            ap.error(f"output directory does not exist: {out_dir}")
        # --out must not replace the source directory or any input the scan reads (each scanned listing,
        # its FSSM header, and the carve index), even through a hardlink or symlink alias the
        # name check alone would miss. samefile also catches a case-only spelling on Windows and macOS's
        # default volume, so only when the report already exists does it compare --out against each input
        # rather than statting N listings for nothing.
        if same_report_path(args.src, args.out):
            ap.error("output must not overwrite the source directory")
        listings = scanned_paths(args.src)
        for path in [*listings, *metadata_paths(listings)]:
            if same_report_path(path, args.out):
                ap.error(f"output must not overwrite an input it scans: {path}")
    report = scan_directory(args.src)
    for stage, b in report["per_stage"].items():
        occ = ", ".join("%s x%d" % (op, n) for op, n in b["occurrences"].items()) or "none"
        print("%s: %d shaders, %d wave64, %d with a selected opcode (candidates: %s)"
              % (stage, b["shaders"], b["wave64"], b["with_selected_op"], occ))
        src = b["mask_word_source"]
        parts = ["%s: %s" % (part, ", ".join("%s x%d" % kv for kv in cnt.items()))
                 for part, cnt in src.items() if cnt]
        if parts:
            print("    mask-word sources (%s)" % "; ".join(parts))
    o = report["other_stages"]
    print("other stages: %d shaders, %d wave64 (not counted)" % (o["shaders"], o["wave64"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
