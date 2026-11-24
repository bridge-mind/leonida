"""Audit initial SGPR/VGPR read-before-write registers and RSRC2/PSV0 headers across CS, MS, and GS shaders.

Standard library only (no GPU, LLVM, or SPIRV-Tools required). Intended to run locally on a disassembled corpus:

    python3 -I scripts/audit_shader4_regs.py <disasm_dir> [--fssm DIR] [--dxbc DIR] [--json OUT] [--limit N]

Directory layout defaults match make_game.py / tools/rdna2spirv.py:
  <disasm_dir>/<name>.s
  <disasm_dir>/../fssm/<name>.fssm
  <disasm_dir>/../<name>.dxbc
  <disasm_dir>/../index.json   (optional stage map; otherwise stage is inferred from filename prefix)
"""
import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from report_paths import same_report_path  # noqa: E402
import shader_ir  # noqa: E402
from shader_meta import SGPR_RMW  # noqa: E402

NO_DEST_OPS = ("s_cmp", "s_bitcmp", "v_cmpx", "s_set", "s_rfe", "s_branch", "s_cbranch",
               "s_waitcnt", "s_barrier", "s_nop", "s_endpgm", "s_sendmsg", "s_sleep",
               "exp", "buffer_store", "tbuffer_store", "global_store", "flat_store", "scratch_store",
               "s_store", "s_buffer_store", "image_store", "ds_write")
# DS ops that write args[0]; every other DS op without "_rtn" (atomics, ds_write*, src2 and GWS forms) only reads its
# operands. Same rule as tools/lane_masks.py.
DS_HAS_DST = ("ds_read", "ds_swizzle", "ds_permute", "ds_bpermute", "ds_append", "ds_consume", "ds_ordered_count",
              "ds_nop")
WAVE64_RE = re.compile(r"exec_hi|v_mbcnt_hi_u32_b32|_saveexec_b64|_wrexec_b64")
# VOP2/VOP3 "c" dot forms accumulate into D (D = D + dot(S0, S1)); the same list the translator's read-before-write
# tracking uses in tools/rdna2spirv.py.
VOP_DOT_ACCUM = ("v_dot2c_f32_f16", "v_dot2c_i32_i16", "v_dot2c_u32_u16",
                 "v_dot4c_i32_i8", "v_dot4c_u32_u8", "v_dot8c_i32_i4", "v_dot8c_u32_u4")


def base_op(op):
    for sfx in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(sfx):
            return op[:-len(sfx)]
    return op


class ReportInputCollision(Exception):
    pass


def protect_input(report_path, input_path, kind):
    if report_path and os.path.isfile(input_path) and same_report_path(report_path, input_path):
        raise ReportInputCollision(f"--json {report_path!r} must not overwrite an input {kind} {input_path!r}")


def load_stage_map(disasm_dir, *, report_path=None):
    parent = os.path.dirname(os.path.realpath(disasm_dir))
    candidates = (os.path.join(parent, "index.json"), os.path.join(disasm_dir, "index.json"))
    # Protect the fallback index even when the first one supplies the stage map.
    for cand in candidates:
        protect_input(report_path, cand, "index")
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, encoding="utf-8") as _f:
                    data = json.load(_f)
                if isinstance(data, list):
                    return {os.path.splitext(e["file"])[0]: e.get("stage", "?") for e in data if "file" in e}
                if isinstance(data, dict):
                    return {k: v.get("stage", "?") for k, v in data.items() if isinstance(v, dict)}
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return {}


def stage_of(name, stage_map):
    if name in stage_map:
        return stage_map[name]
    if len(name) >= 3 and name[2] == "_":
        return name[:2].lower()
    return "?"


def read_psv0(dxbc_path):
    """Parse DXBC PSV0 chunk for MS (stage 13) or GS metadata; returns dict or None."""
    if not os.path.isfile(dxbc_path):
        return None
    with open(dxbc_path, "rb") as _f:
        d = _f.read()
    if len(d) < 32 or d[:4] != b"DXBC":
        return None
    nchunks = struct.unpack_from("<I", d, 28)[0]
    for k in range(nchunks):
        if 32 + 4 * k + 4 > len(d):
            break
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if off + 8 <= len(d) and d[off:off + 4] == b"PSV0":
            plen = struct.unpack_from("<I", d, off + 4)[0]
            p = d[off + 8:off + 8 + plen]
            if len(p) < 4:
                return None
            rsize = struct.unpack_from("<I", p, 0)[0]
            r = p[4:4 + rsize]
            if len(r) < 48:
                return {"rsize": rsize}
            shared, _, payload = struct.unpack_from("<III", r, 0)
            maxv, maxp = struct.unpack_from("<HH", r, 12)
            tx, ty, tz = struct.unpack_from("<III", r, 36)
            return {
                "psv_stage": r[24],
                "topology": r[27],
                "shared_bytes": shared,
                "payload_bytes": payload,
                "maxv": maxv,
                "maxp": maxp,
                "threads": [tx, ty, tz],
            }
    return None


def decode_cs_rsrc2(fssm_path):
    """Read COMPUTE_PGM_RSRC2 (0x138) and thread-group size (0x13C..0x144) from FSSM."""
    if not os.path.isfile(fssm_path):
        return None
    with open(fssm_path, "rb") as _f:
        hdr = _f.read(0x148)
    if len(hdr) < 0x148:
        return None
    rsrc2, lx, ly, lz = struct.unpack_from("<4I", hdr, 0x138)
    return {
        "rsrc2_hex": f"0x{rsrc2:08x}",
        "scratch_en": rsrc2 & 1,
        "user_sgpr": (rsrc2 >> 1) & 31,
        "trap_present": (rsrc2 >> 6) & 1,
        "tgid_x": (rsrc2 >> 7) & 1,
        "tgid_y": (rsrc2 >> 8) & 1,
        "tgid_z": (rsrc2 >> 9) & 1,
        "tg_size_en": (rsrc2 >> 10) & 1,
        "tidig_comp_cnt": (rsrc2 >> 11) & 3,
        "lds_blocks_512b": (rsrc2 >> 15) & 0x1FF,
        "group_size": [lx, ly, lz],
    }


def read_fssm_words(fssm_path):
    """Read raw u32 words at 0x138..0x144 for non-CS FSSM inspection."""
    if not os.path.isfile(fssm_path):
        return None
    with open(fssm_path, "rb") as _f:
        hdr = _f.read(0x148)
    if len(hdr) < 0x148:
        return None
    w = struct.unpack_from("<4I", hdr, 0x138)
    return [f"0x{x:08x}" for x in w]


def _fmt_s3_arg(a, is_dst=False):
    if is_dst:
        return "D"
    k = a["k"]
    if k == "sgpr":
        if a["r"] == 3 and a.get("n", 1) == 1:
            return "s3"
        return "sN"
    if k == "slist":
        return "s3" if 3 in a["r"] else "sN"
    if k in ("vgpr", "vlist"):
        return "vN"
    if k == "imm":
        v = a["v"]
        return hex(v) if v >= 16 or v < 0 else str(v)
    return a.get("t", k)


def rbw_registers(insns):
    """Straight-line read-before-write SGPR, VGPR sets and s3-read instruction patterns."""
    written_s, written_v = set(), set()
    rbw_s, rbw_v = set(), set()
    s3_patterns = []

    def expand_reg(a):
        if a["k"] in ("vlist", "slist"):
            return [r for r in a["r"] if isinstance(r, int)]
        return range(a["r"], a["r"] + a.get("n", 1))

    for ins in insns:
        op = base_op(ins["op"])
        args = ins["args"]
        if not args:
            continue
        mods = ins.get("mods", ())
        is_gf_atomic = op.startswith(("global_atomic_", "flat_atomic_"))
        is_bi_atomic = op.startswith(("buffer_atomic_", "image_atomic_"))
        is_no_dest = (
            op.startswith(NO_DEST_OPS)
            or (is_gf_atomic and "glc" not in mods)
            or (is_bi_atomic and "glc" not in mods)
            or (op.startswith("ds_") and "_rtn" not in op and not op.startswith(DS_HAS_DST))
            # A buffer load with lds transfers straight to LDS: args[0] is vaddr, not a written VGPR.
            or (op.startswith(("buffer_load", "tbuffer_load")) and "lds" in mods)
        )
        reads_imm_dst = op in SGPR_RMW
        # Instructions that read their destination too (buffer/image atomics with glc read and write vdata at args[0];
        # global/flat atomics with glc write vdst at args[0] and read vaddr/vdata/saddr from args[1:]).
        dst_also_read = (
            reads_imm_dst or "fmac" in op or "mac_" in op or "dot2c_" in op or "dot4c_" in op or "dot8c_" in op
            or op == "v_writelane_b32"
            or op == "v_swap_b32" or (not is_gf_atomic and "_atomic_" in op)
            or op in VOP_DOT_ACCUM
        )
        # A direct-to-LDS MUBUF/MTBUF load writes LDS, not a VGPR: args[0] holds the 4-dword LDS address and is read.
        is_lds_load = op.startswith(("buffer_load", "tbuffer_load")) and "lds" in mods
        is_dual_dst = op.startswith("v_") and (
            "_co_" in op or op.startswith(("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64"))
        )
        # Carry-out / scale-out forms write the VGPR and scalar mask first, even when they also take a carry-in source.
        if is_lds_load:
            dsts, srcs = [], args
        elif not is_no_dest and is_dual_dst and len(args) >= 2:
            dsts = args[:2]
            srcs = args[2:]
        elif is_no_dest:
            dsts = []
            srcs = args
        else:
            dsts = args[:1]
            srcs = (args[:1] if dst_also_read else []) + args[1:]

        reads_s3_rbw = False
        for a in srcs:
            if a["k"] in ("sgpr", "slist"):
                for r in expand_reg(a):
                    if r not in written_s:
                        rbw_s.add(r)
                        if r == 3:
                            reads_s3_rbw = True
            elif a["k"] in ("vgpr", "vlist"):
                for r in expand_reg(a):
                    if r not in written_v:
                        rbw_v.add(r)
        if reads_s3_rbw:
            dst_ids = set() if reads_imm_dst else {id(x) for x in dsts}
            fmt_args = ", ".join(_fmt_s3_arg(x, is_dst=(id(x) in dst_ids)) for x in args)
            s3_patterns.append(f"{op} {fmt_args}".strip())

        for a in dsts:
            if a["k"] in ("sgpr", "slist"):
                written_s.update(expand_reg(a))
            elif a["k"] in ("vgpr", "vlist"):
                written_v.update(expand_reg(a))

    return sorted(rbw_s), sorted(rbw_v), s3_patterns


def detect_wave64(insns):
    return any(
        WAVE64_RE.search(" ".join([i["op"]] + [x.get("t", "") for x in i["args"]]))
        or (base_op(i["op"]).endswith("_b64") and any(x.get("t") in ("exec", "vcc") for x in i["args"]))
        for i in insns
    )


def audit(disasm_dir, fssm_dir=None, dxbc_dir=None, stages=("cs", "ms", "gs"), limit=None, *, report_path=None):
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    parent = os.path.dirname(os.path.realpath(disasm_dir))
    fssm_dir = fssm_dir or os.path.join(parent, "fssm")
    dxbc_dir = dxbc_dir or parent

    with os.scandir(disasm_dir) as it:
        files = sorted(e.name for e in it if e.name.endswith(".s") and e.is_file())
    if report_path:
        for fname in files:
            protect_input(report_path, os.path.join(disasm_dir, fname), "listing")
    stage_map = load_stage_map(disasm_dir, report_path=report_path)
    if limit is not None:
        files = files[:limit]

    report = {
        st: {
            "count": 0,
            "wave64_heuristic": 0,
            "wave32_heuristic": 0,
            "rsrc2_combos": Counter(),
            "psv0_combos": Counter(),
            "fssm_0x138_words": Counter(),
            "rbw_sgpr_counts": Counter(),
            "rbw_vgpr_counts": Counter(),
            "rbw_sgpr_sets": Counter(),
            "rbw_vgpr_sets": Counter(),
            "s3_rbw_op_patterns": Counter(),
        }
        for st in stages
    }

    for fname in files:
        stem = os.path.splitext(fname)[0]
        st = stage_of(stem, stage_map)
        if st not in report:
            continue
        info = shader_ir.parse_file(os.path.join(disasm_dir, fname))
        insns = info["insns"]
        rbw_s, rbw_v, s3_pats = rbw_registers(insns)
        rec = report[st]
        rec["count"] += 1
        if detect_wave64(insns):
            rec["wave64_heuristic"] += 1
        else:
            rec["wave32_heuristic"] += 1
        for r in rbw_s:
            rec["rbw_sgpr_counts"][f"s{r}"] += 1
        for r in rbw_v:
            rec["rbw_vgpr_counts"][f"v{r}"] += 1
        rec["rbw_sgpr_sets"][",".join(f"s{r}" for r in rbw_s) or "(none)"] += 1
        rec["rbw_vgpr_sets"][",".join(f"v{r}" for r in rbw_v) or "(none)"] += 1
        for pat in s3_pats:
            rec["s3_rbw_op_patterns"][pat] += 1

        fssm_path = os.path.join(fssm_dir, stem + ".fssm")
        protect_input(report_path, fssm_path, "FSSM file")
        if st == "cs":
            dec = decode_cs_rsrc2(fssm_path)
            if dec:
                key = (
                    f"user={dec['user_sgpr']} tgid={dec['tgid_x']}{dec['tgid_y']}{dec['tgid_z']} "
                    f"tg_size={dec['tg_size_en']} tidig={dec['tidig_comp_cnt']} "
                    f"scratch={dec['scratch_en']} trap={dec['trap_present']} "
                    f"lds={dec['lds_blocks_512b'] * 512}B"
                )
                rec["rsrc2_combos"][key] += 1
            else:
                rec["rsrc2_combos"]["(missing fssm)"] += 1
        else:
            words = read_fssm_words(fssm_path)
            if words:
                rec["fssm_0x138_words"][" ".join(words)] += 1
            dxbc_path = os.path.join(dxbc_dir, stem + ".dxbc")
            protect_input(report_path, dxbc_path, "DXBC file")
            psv = read_psv0(dxbc_path)
            if psv:
                if "threads" in psv:
                    key = (
                        f"stage={psv['psv_stage']} topo={psv['topology']} "
                        f"threads={'x'.join(map(str, psv['threads']))} "
                        f"maxv={psv['maxv']} maxp={psv['maxp']} "
                        f"payload={psv['payload_bytes']}B shared={psv['shared_bytes']}B"
                    )
                else:
                    key = f"rsize={psv['rsize']}"
                rec["psv0_combos"][key] += 1

    # Convert Counters to plain sorted dicts for JSON/text output.
    out = {}
    for st, rec in report.items():
        out[st] = {
            "count": rec["count"],
            "wave64_heuristic": rec["wave64_heuristic"],
            "wave32_heuristic": rec["wave32_heuristic"],
            "rsrc2_combos": dict(rec["rsrc2_combos"].most_common()),
            "psv0_combos": dict(rec["psv0_combos"].most_common()),
            "fssm_0x138_words": dict(rec["fssm_0x138_words"].most_common(20)),
            "rbw_sgpr_counts": dict(sorted(rec["rbw_sgpr_counts"].items(), key=lambda kv: int(kv[0][1:]))),
            "rbw_vgpr_counts": dict(sorted(rec["rbw_vgpr_counts"].items(), key=lambda kv: int(kv[0][1:]))),
            "top_rbw_sgpr_sets": dict(rec["rbw_sgpr_sets"].most_common(15)),
            "top_rbw_vgpr_sets": dict(rec["rbw_vgpr_sets"].most_common(15)),
            "s3_rbw_op_patterns": dict(rec["s3_rbw_op_patterns"].most_common(20)),
        }
    return out


def print_report(rep):
    for st, rec in rep.items():
        print(f"=== {st.upper()} ({rec['count']} shaders; wave64={rec['wave64_heuristic']}, wave32={rec['wave32_heuristic']}) ===")
        if rec["rsrc2_combos"]:
            print("  COMPUTE_PGM_RSRC2 combinations:")
            for k, v in list(rec["rsrc2_combos"].items())[:20]:
                print(f"    {v:6d}  {k}")
        if rec["psv0_combos"]:
            print("  PSV0 combinations:")
            for k, v in list(rec["psv0_combos"].items())[:20]:
                print(f"    {v:6d}  {k}")
        if rec["fssm_0x138_words"]:
            print("  FSSM 0x138..0x144 raw words (top 10):")
            for k, v in list(rec["fssm_0x138_words"].items())[:10]:
                print(f"    {v:6d}  {k}")
        if rec["rbw_sgpr_counts"]:
            print("  RBW SGPRs:", " ".join(f"{k}:{v}" for k, v in rec["rbw_sgpr_counts"].items()))
        if rec["rbw_vgpr_counts"]:
            print("  RBW VGPRs:", " ".join(f"{k}:{v}" for k, v in rec["rbw_vgpr_counts"].items()))
        if rec["s3_rbw_op_patterns"]:
            print("  s3 RBW instruction patterns:")
            for k, v in list(rec["s3_rbw_op_patterns"].items())[:10]:
                print(f"    {v:6d}  {k}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit CS/MS/GS RSRC2/PSV0 headers and read-before-write registers.")
    ap.add_argument("disasm_dir", help="directory of disassembled .s files")
    ap.add_argument("--fssm", default=None, help="directory of .fssm files (default: <disasm_dir>/../fssm)")
    ap.add_argument("--dxbc", default=None, help="directory of .dxbc files (default: <disasm_dir>/..)")
    ap.add_argument("--json", dest="json_out", default=None, help="optional path to write full JSON report")
    ap.add_argument("--limit", type=int, default=None, help="stop after N .s files")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 0:
        ap.exit(2, f"{ap.prog}: error: --limit must be non-negative\n")
    # A bad path is one ERROR line, not a traceback out of os.listdir: the audit of a whole corpus can
    # run for minutes, and an explicit --fssm/--dxbc that does not exist would otherwise be counted as
    # "(missing fssm)" and print a report that looks right.
    for label, path in (("", a.disasm_dir), ("--fssm: ", a.fssm), ("--dxbc: ", a.dxbc)):
        if path is None:
            continue
        if not os.path.exists(path):
            print(f"ERROR: {label}{path}: no such folder", file=sys.stderr)
            return 2
        if not os.path.isdir(path):
            print(f"ERROR: {label}{path}: not a folder", file=sys.stderr)
            return 2

    report_path = a.json_out if a.json_out and os.path.isfile(a.json_out) else None
    try:
        rep = audit(a.disasm_dir, fssm_dir=a.fssm, dxbc_dir=a.dxbc, limit=a.limit, report_path=report_path)
    except ReportInputCollision as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print_report(rep)
    if a.json_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
            with open(a.json_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rep, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: {a.json_out}: cannot write: {e.strerror or e}", file=sys.stderr)
            return 1
        print(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
