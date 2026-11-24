"""Count the wave32 pixel shaders whose results can depend on the subgroup being 32 lanes wide.

    python -I tools/wave32_pixel_scan.py <work>/shaders/dxbc/disasm [--out wave32_pixel_scan.json]

The counterpart of wave64_graphics_scan.py for the other direction. A wave32 pixel shader is translated for a 32-wide
subgroup (spirv/index.json `subgroup_size` 32). Where the device cannot require 32 for the fragment stage and runs it
64 wide (AMD's Windows driver on RDNA2 reports `requiredSubgroupSizeStages` = COMPUTE only, default size 64:
docs/d3d12_vulkan.md, "Subgroup size"), two hardware waves share one subgroup. Lane-independent code is exact; an
instruction that reads other lanes, the lane number or the wave's mask as a whole sees 64 lanes. This tool reports,
over the whole corpus, how many wave32 pixel shaders contain such an instruction, per family. A hit is a candidate,
not a verdict: whether a given occurrence changes a result depends on run-time values (a v_readfirstlane of a value
that is uniform anyway, a branch that skips a block whose lanes are all off) that a static scan does not decide.

Families (an instruction counts once, in the first that matches):

- lane_id: `v_mbcnt_lo/hi_u32_b32`, whatever the mask operand (with -1 it is the lane number).
- mask_word: the instructions the translator answers from the whole wave's mask (rdna2spirv.Gen.wave_mask) when a
  source holds a lane mask: `s_bcnt0/1_i32_*`, `s_ff0/1_i32_*`, `s_bitcmp0/1_*`, `s_cmp_*` with a lane-mask operand.
  A lane mask is EXEC, or VCC / an SGPR that tools/lane_masks.py finds written as a mask on every path (the
  translator's holds_lane_mask); the same instructions on data are not counted.
- lane_select: `v_readlane_b32`, `v_writelane_b32`, `v_readfirstlane_b32`, except the two the translator lowers
  without a subgroup op: a `v_readlane` that fetches a parked lane mask back, and a `v_readfirstlane` of a mask held
  in a VGPR (lane_masks.py's state).
- permute: `v_permlane16_b32`, `v_permlanex16_b32`, `ds_permute_b32`, `ds_bpermute_b32`, and `ds_swizzle_b32`
  in its bitmask modes (offset bit 15 clear: BITMASK_PERM, SWAP, REVERSE, BROADCAST).
- dpp_row: a DPP instruction with a row control (`row_shl/shr/ror/mirror/half_mirror/share/xmask`) or `dpp8`.
- vote: a whole-wave branch on EXEC or VCC (`s_cbranch_execz/execnz/vccz/vccnz`, which rdna2spirv.branch_condition
  always answers with a subgroup vote), or an SCC reader (`s_cbranch_scc0/1`,
  `s_cselect`, `s_cmov`, `s_addc`, `s_subb`) whose SCC, in the same basic block, last came from a lane-mask SOP:
  a bitwise, saveexec or wrexec op that writes EXEC or reads a lane mask (the translator's scc_from_mask vote).
  An SCC carried across a block boundary is not followed, so this row is a lower bound.
- quad (reported apart, not in `crossing`): `s_wqm_*`, DPP `quad_perm`, and `ds_swizzle_b32` in QUAD_PERM mode
  (offset bit 15 set). A quad is four consecutive lanes, so these give the same result at either width.

`crossing` is the number of wave32 pixel shaders with at least one hit outside `quad`; `quad_only` those whose only
hits are quad ops. `permute` and `dpp_row` cross lanes but, as rdna2spirv.py lowers them, stay inside the source
lane's 16-lane row or 32-lane half (v_permlane*, ds_swizzle, DPP keep the upper lane bits), so 64 wide gives the
same result; lane_id, mask_word, lane_select and vote do not (they take the lane select modulo 32, the mask of
lanes 0-31, or a vote of the whole subgroup), which is what differs (code reading; ps_bitmask_perm's v_mbcnt reads
32 in lanes 32-63 of an RX 6800's 64-wide fragment subgroup). Mask arithmetic that only writes a mask (s_and_saveexec_b32, v_cmp, v_cndmask) is per lane and is
not counted unless its SCC is read.

Wave size and stage are found as in wave64_graphics_scan.py: the FSSM header's wave-size flag next to the listing
(fssm_header.wave_size), else the code-marker rule; the stage from the carve index (rdna2spirv.stage_of). Basic
blocks are split as rdna2spirv.translate does (at branch targets and after branches) for the lane-mask analysis.

Standard library only. Nothing from the game is printed or written: counts only.
Usage: `python -I tools/wave32_pixel_scan.py --help` lists the options.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fssm_header  # noqa: E402
import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402  (stage_of, native_paths, base_op, BRANCHES)
from shader_ir import parse_file  # noqa: E402
from wave64_graphics_scan import scanned_paths, wave64_rule  # noqa: E402

FAMILIES = ("lane_id", "mask_word", "lane_select", "permute", "dpp_row", "vote", "quad")
CROSSING = FAMILIES[:-1]

LANE_ID = {"v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"}
MASK_WORD_PREFIX = ("s_bcnt0_i32_", "s_bcnt1_i32_", "s_ff0_i32_", "s_ff1_i32_", "s_bitcmp0_", "s_bitcmp1_", "s_cmp_")
LANE_SELECT = {"v_readlane_b32", "v_writelane_b32", "v_readfirstlane_b32"}
PERMUTE = {"v_permlane16_b32", "v_permlanex16_b32", "ds_permute_b32", "ds_bpermute_b32"}
DPP_ROW = re.compile(r"^(row_(shl|shr|ror|share|xmask):|row_mirror$|row_half_mirror$|dpp8:)")
MASK_BRANCHES = {"s_cbranch_execz", "s_cbranch_execnz", "s_cbranch_vccz", "s_cbranch_vccnz"}
SCC_READERS = ("s_cbranch_scc0", "s_cbranch_scc1", "s_cselect_", "s_cmov_", "s_cmovk_", "s_addc_", "s_subb_")
SCC_WRITERS = ("s_and_", "s_or_", "s_xor_", "s_andn1_", "s_andn2_", "s_orn1_", "s_orn2_", "s_nand_", "s_nor_",
               "s_xnor_", "s_not_", "s_wqm_", "s_quadmask_", "s_bcnt", "s_cmp_", "s_cmpk_", "s_bitcmp", "s_add_",
               "s_addk_", "s_sub_", "s_addc_", "s_subb_", "s_lshl", "s_lshr", "s_ashr", "s_bfe_", "s_min_", "s_max_",
               "s_abs_", "s_absdiff_", "s_ff0_", "s_ff1_", "s_flbit_", "s_bitset", "s_brev_", "s_bfm_")
MASK_SOP = ("s_and_", "s_or_", "s_xor_", "s_andn1_", "s_andn2_", "s_orn1_", "s_orn2_", "s_nand_", "s_nor_",
            "s_xnor_", "s_not_")


def blocks_of(insns):
    """Basic blocks and successors as rdna2spirv.translate splits them (no irreducible-node splitting), or one block
    when a branch has no encoding to resolve its target with."""
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    branches = [i for i in insns if rdna2spirv.base_op(i["op"]) in rdna2spirv.BRANCHES]
    if not insns or not branches or any(not i.get("enc") or not any(x["k"] == "imm" for x in i["args"])
                                        for i in branches):
        return [insns], [[]]
    leaders = {insns[0]["pc"]}
    for ins in branches:
        leaders.update((target(ins), next_pc(ins)))
    pcs = {i["pc"] for i in insns}
    starts = sorted(p for p in leaders if p in pcs)
    index = {p: n for n, p in enumerate(starts)}
    blocks, cur = [[] for _ in starts], 0
    for ins in insns:
        cur = index.get(ins["pc"], cur)
        blocks[cur].append(ins)
    succ = []
    for n, blk in enumerate(blocks):
        last = next((i for i in blk if rdna2spirv.base_op(i["op"]) == "s_endpgm" or
                     rdna2spirv.base_op(i["op"]) in rdna2spirv.BRANCHES), None)
        fall = [n + 1] if n + 1 < len(blocks) else []
        if last is None:
            succ.append(fall)
        elif rdna2spirv.base_op(last["op"]) == "s_endpgm":
            succ.append([])
        else:
            t = index.get(target(last))
            to = [t] if t is not None else []
            succ.append(list(dict.fromkeys(to if rdna2spirv.base_op(last["op"]) == "s_branch" else to + fall)))
    return blocks, succ


def holds_lane_mask(masks, either, a):
    """rdna2spirv.Gen.holds_lane_mask for a wave32 program, from lane_masks.analyse's state at the instruction."""
    if a["k"] == "special":
        t = a.get("t", "")
        return t.startswith("exec") or (t in ("vcc", "vcc_lo") and "vcc" in masks)
    if a["k"] == "sgpr":
        n = a.get("n", 1)
        if n == 2:
            return any(a["r"] + i in masks for i in range(2)) and \
                all(a["r"] + i in masks or a["r"] + i in either for i in range(2))
        return all(a["r"] + i in masks for i in range(n))
    return False


def swizzle_quad(ins):
    """True / False for a ds_swizzle_b32 in QUAD_PERM / a bitmask mode, from its printed offset."""
    off = next((m[len("offset:"):] for m in ins["mods"] if m.startswith("offset:")), "0")
    if off.startswith("swizzle("):
        return off.startswith("swizzle(QUAD_PERM")
    try:
        return bool(int(off, 0) & 0x8000)
    except ValueError:
        return False


def family_of(ins, masks, either):
    """The family of one instruction (see the module docstring) other than `vote`, or None."""
    op = rdna2spirv.base_op(ins["op"])
    if op in LANE_ID:
        return "lane_id"
    if op.startswith(MASK_WORD_PREFIX):
        # s_cmp_* s0, s1 and s_bitcmp* s0, bit read their first operands; s_bcnt / s_ff write sD from sS.
        a = ins["args"]
        srcs = a[:2] if op.startswith("s_cmp_") else a[:1] if op.startswith("s_bitcmp") else a[1:2]
        return "mask_word" if any(holds_lane_mask(masks, either, x) for x in srcs) else None
    if op in LANE_SELECT:
        a = ins["args"]
        # Not a subgroup op as translated: a lane mask fetched back from the VGPR lane it was parked in, and a
        # v_readfirstlane of a mask held in a VGPR (each lane reads its own copy).
        if op == "v_readlane_b32" and len(a) == 3 and lane_masks.lane_slot(a[1], a[2]) in masks:
            return None
        if op == "v_readfirstlane_b32" and len(a) == 2 and a[1]["k"] == "vgpr" and ("v", a[1]["r"]) in masks:
            return None
        return "lane_select"
    if op in PERMUTE:
        return "permute"
    if op == "ds_swizzle_b32":
        return "quad" if swizzle_quad(ins) else "permute"
    if op.startswith("s_wqm_"):
        return "quad"
    if ins["op"].endswith("_dpp") or any(m.startswith(("quad_perm:", "dpp8:")) for m in ins["mods"]):
        if any(DPP_ROW.match(m) for m in ins["mods"]):
            return "dpp_row"
        if any(m.startswith("quad_perm:") for m in ins["mods"]):
            return "quad"
    return None


def scan_shader(insns):
    """Per-family occurrence counts for one wave32 program (a Counter, empty when nothing matches)."""
    blocks, succ = blocks_of(insns)
    masks_at, _, either_at = lane_masks.analyse(blocks, succ, include_either=True)
    hits = Counter()
    for blk in blocks:
        scc_from_mask = False                       # the last SCC writer in this block was a lane-mask SOP
        for ins in blk:
            op = rdna2spirv.base_op(ins["op"])
            masks, either = masks_at.get(id(ins), {}), either_at.get(id(ins), set())
            fam = family_of(ins, masks, either)
            if fam:
                hits[fam] += 1
            elif op in MASK_BRANCHES or (op.startswith(SCC_READERS) and scc_from_mask):
                hits["vote"] += 1
            if op.startswith(SCC_WRITERS):
                a = ins["args"]
                writes_exec = bool(a) and a[0]["k"] == "special" and a[0].get("t", "").startswith("exec")
                scc_from_mask = (op.startswith(MASK_SOP) or "_saveexec_" in op or "_wrexec_" in op) and \
                    (writes_exec or any(holds_lane_mask(masks, either, x) for x in a[1:3]))
    return hits


def scan_directory(disasm):
    """Aggregate over one carve disasm directory: pixel-shader totals and per-family counts. No names."""
    report = {"ps": 0, "ps_wave32": 0, "crossing": 0, "quad_only": 0,
              "shaders": {f: 0 for f in FAMILIES}, "occurrences": {f: 0 for f in FAMILIES},
              "only_family": {f: 0 for f in CROSSING}}
    for path in scanned_paths(disasm):
        info = parse_file(path)
        if rdna2spirv.stage_of(info["name"], path) != "ps":
            continue
        report["ps"] += 1
        fssm_path = rdna2spirv.native_paths(path, info["name"])[0]
        try:
            with open(fssm_path, "rb") as f:
                wave_size = fssm_header.wave_size(f.read(fssm_header.WAVE_FLAGS_OFFSET + 4))
        except OSError:
            wave_size = None
        wave64 = wave_size == 64 if wave_size is not None else wave64_rule("ps", info["insns"])
        if wave64:
            continue
        report["ps_wave32"] += 1
        hits = scan_shader(info["insns"])
        for fam, n in hits.items():
            report["shaders"][fam] += 1
            report["occurrences"][fam] += n
        crossing = [f for f in CROSSING if hits[f]]
        if crossing:
            report["crossing"] += 1
            if len(crossing) == 1:
                report["only_family"][crossing[0]] += 1
        elif hits["quad"]:
            report["quad_only"] += 1
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count wave32 pixel shaders with lane-crossing or width-dependent "
                                             "instructions (counts only, no names)")
    ap.add_argument("src", help="<work>/shaders/dxbc/disasm from the shaders stage")
    ap.add_argument("--out", help="write the same counts to a JSON file")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.src):
        ap.error(f"not a directory: {args.src}")
    if args.out:
        if os.path.isdir(args.out):
            ap.error(f"{args.out} is a directory, not a JSON file")
        if not os.path.isdir(os.path.dirname(os.path.abspath(args.out))):
            ap.error(f"output directory does not exist: {os.path.dirname(os.path.abspath(args.out))}")
        if os.path.exists(args.out) and any(os.path.samefile(args.out, p) for p in scanned_paths(args.src)):
            ap.error("output must not overwrite a listing it scans")
    r = scan_directory(args.src)
    print("ps: %d shaders, %d wave32; %d with a lane-crossing or width-dependent instruction, %d with quad ops only"
          % (r["ps"], r["ps_wave32"], r["crossing"], r["quad_only"]))
    for fam in FAMILIES:
        only = " (%d with no other family)" % r["only_family"][fam] if fam in r["only_family"] else ""
        print("  %-12s %5d shaders, %6d instructions%s" % (fam, r["shaders"][fam], r["occurrences"][fam], only))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
