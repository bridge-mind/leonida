"""SHADER-33: which hardware vertex-shader layout the corpus uses (legacy VS or NGG), and which input VGPRs it reads.

    python -I scripts/vs_input_survey.py <work>/shaders/dxbc [--json OUT]

Reads <dxbc>/index.json (stage per shader) and <dxbc>/disasm/<name>.s (shader_disasm.py) for every vertex shader and
prints counts only:

- NGG markers: `exp prim` (primitive export), `s_sendmsg` (NGG needs GS_ALLOC_REQ) and `v_mbcnt` (NGG thread ids).
  A legacy (non-NGG) hardware VS has none of them and exports `pos0` / `param` only.
- Registers read before written, straight-line, from the start of the program: the same rule as #240's
  scripts/audit_shader4_regs.py (by @eIBart0) uses for CS/MS/GS. In the legacy GFX10 VS layout (Mesa
  radv_shader_args.c, declare_vs_input_vgprs, not is_ngg) v0 = vertex id, v1 unused, v2 = primitive id,
  v3 = instance id; in an NGG VS merged with GS the VS inputs start at v5 (vertex id) / v8 (instance id).
- The fetch call: `s_swappc_b64 s[4:5], s[4:5]` at the start of a VS calls the driver's fetch program, which reads
  v0 itself and leaves the attributes in v4 and up.

Standard library only. Nothing from the game is printed or written beyond counts.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import shader_ir  # noqa: E402
from shader_meta import SGPR_RMW  # noqa: E402

NO_DEST_OPS = ("s_cmp", "s_bitcmp", "v_cmpx", "s_set", "s_rfe", "s_branch", "s_cbranch", "s_waitcnt", "s_barrier",
               "s_nop", "s_endpgm", "s_sendmsg", "s_sleep", "exp", "buffer_store", "tbuffer_store", "global_store",
               "flat_store", "scratch_store", "s_store", "s_buffer_store", "image_store", "ds_write")
DUAL_DST_VALU = ("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64")
DS_RETURNING = ("ds_read", "ds_bpermute", "ds_permute", "ds_swizzle", "ds_consume", "ds_append", "ds_ordered_count")
PRIM_EXPORT = re.compile(r"^\s*exp\s+prim\b", re.M)
POS0_EXPORT = re.compile(r"^\s*exp\s+pos0\b", re.M)
PARAM_EXPORT = re.compile(r"^\s*exp\s+param\d+\b", re.M)
SENDMSG = re.compile(r"^\s*s_sendmsg\b", re.M)
MBCNT = re.compile(r"\bv_mbcnt_(lo|hi)_u32_b32\b")
FETCH_CALL = re.compile(r"^\s*s_swappc_b64\s+s\[4:5\],\s*s\[4:5\]", re.M)


def base_op(op):
    for sfx in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(sfx):
            return op[:-len(sfx)]
    return op


def _expand_regs(a):
    """Return ('s' | 'v', iterable of register indices) for an operand, or (None, ())."""
    k = a["k"]
    if k == "sgpr":
        return "s", range(a["r"], a["r"] + a.get("n", 1))
    if k == "vgpr":
        return "v", range(a["r"], a["r"] + a.get("n", 1))
    if k == "slist":
        return "s", (r for r in a["r"] if isinstance(r, int))
    if k == "vlist":
        return "v", (r for r in a["r"] if isinstance(r, int))
    return None, ()


def read_before_write(insns):
    """(SGPRs, VGPRs) read before any write, straight-line from the first instruction."""
    written_s, written_v, read_s, read_v = set(), set(), set(), set()
    for ins in insns:
        op, args = base_op(ins["op"]), ins["args"]
        if not args:
            continue
        mods = ins.get("mods", ())
        is_gf_atomic = op.startswith(("global_atomic_", "flat_atomic_"))
        is_bi_atomic = op.startswith(("buffer_atomic_", "image_atomic_"))
        is_ds_no_dest = op.startswith("ds_") and not op.startswith(DS_RETURNING) and "_rtn" not in op
        no_dest = (
            op.startswith(NO_DEST_OPS)
            or ((is_gf_atomic or is_bi_atomic) and "glc" not in mods)
            or is_ds_no_dest
            or (op.startswith("v_cmp_") and len(args) <= 2)
        )
        if no_dest:
            dsts, srcs = [], args
        elif op == "v_swap_b32" and len(args) >= 2:
            dsts, srcs = args[:2], args[:2]
        elif (op.startswith("v_") and ("_co_" in op or op in DUAL_DST_VALU)
              and len(args) >= 2 and args[1]["k"] in ("sgpr", "special", "slist")):
            dsts, srcs = args[:2], args[2:]          # vdst + scalar carry/scale-out dst
        else:
            dst_read = (
                op in SGPR_RMW or "fmac" in op or "mac_" in op
                or op.startswith(("v_dot2c_", "v_dot4c_", "v_dot8c_"))
                or op == "v_writelane_b32" or is_bi_atomic
            )
            dsts, srcs = args[:1], (args[:1] if dst_read else []) + args[1:]
        for a in srcs:
            kind, regs = _expand_regs(a)
            for r in regs:
                if kind == "s" and r not in written_s:
                    read_s.add(r)
                elif kind == "v" and r not in written_v:
                    read_v.add(r)
        for a in dsts:
            kind, regs = _expand_regs(a)
            if kind == "s":
                written_s.update(regs)
            elif kind == "v":
                written_v.update(regs)
    return read_s, read_v


def survey(dxbc_dir):
    with open(os.path.join(dxbc_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, list) or not all(isinstance(e, dict) for e in index):
        raise ValueError("index.json is not a list of shader entries")
    if not all(isinstance(e.get("file"), str) for e in index if e.get("stage") == "vs"):
        raise ValueError("index.json has a vertex shader entry without a string file")
    names = [os.path.splitext(e["file"])[0] for e in index if e.get("stage") == "vs"]
    markers, vgprs, families = Counter(), Counter(), Counter()
    missing = 0
    for n in names:
        path = os.path.join(dxbc_dir, "disasm", n + ".s")
        if not os.path.isfile(path):
            missing += 1
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        _, read_v = read_before_write(shader_ir.parse_file(path)["insns"])
        for r in read_v:
            if r < 16:
                vgprs[f"v{r}"] += 1
        fetch = bool(FETCH_CALL.search(text))
        markers["exp prim"] += bool(PRIM_EXPORT.search(text))
        markers["s_sendmsg"] += bool(SENDMSG.search(text))
        markers["v_mbcnt"] += bool(MBCNT.search(text))
        markers["exp pos0"] += bool(POS0_EXPORT.search(text))
        markers["exp param"] += bool(PARAM_EXPORT.search(text))
        markers["fetch call"] += fetch
        if fetch:
            families["fetch call, attributes from v4 up"] += 1
        elif 0 in read_v:
            families["reads v0 (vertex id)" + (" and v3 (instance id)" if 3 in read_v else "")] += 1
        else:
            families["reads neither v0 nor a fetch call"] += 1
    return {"vertex_shaders": len(names), "missing_listings": missing, "markers": dict(markers),
            "input_vgprs_read": dict(sorted(vgprs.items(), key=lambda kv: int(kv[0][1:]))),
            "families": dict(families)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dxbc", help="<work>/shaders/dxbc from make_game.py")
    ap.add_argument("--json", help="also write the counts as JSON")
    a = ap.parse_args()
    index_path = os.path.join(a.dxbc, "index.json")
    if a.json and os.path.exists(a.json) and os.path.exists(index_path) and os.path.samefile(a.json, index_path):
        print(f"vs_input_survey: error: --json would overwrite the input {index_path}", file=sys.stderr)
        return 2
    try:
        r = survey(a.dxbc)
    except (OSError, ValueError) as e:  # missing or non-folder <dxbc>, unreadable, non-UTF-8 or malformed index.json
        print(f"vs_input_survey: error: cannot read {index_path}: {e}", file=sys.stderr)
        return 2
    m = r["markers"]
    print(f"{r['vertex_shaders']} vertex shaders ({r['missing_listings']} without a listing)")
    print(f"NGG markers: exp prim {m.get('exp prim', 0)}, s_sendmsg {m.get('s_sendmsg', 0)}, v_mbcnt {m.get('v_mbcnt', 0)}")
    print(f"exports: pos0 {m.get('exp pos0', 0)}, param {m.get('exp param', 0)}; fetch call {m.get('fetch call', 0)}")
    print("input VGPRs read before written (v0..v15):", r["input_vgprs_read"])
    print("families:", r["families"])
    legacy = not (m.get("exp prim") or m.get("s_sendmsg") or m.get("v_mbcnt"))
    print("layout:", "legacy (non-NGG) VS" if legacy else "NGG markers present: check the VS input layout")
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(r, f, indent=1)
        except OSError as e:
            print(f"vs_input_survey: error: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
