"""Parse LLVM AMDGPU disassembly (.s from shader_disasm.py) into a structured instruction list.

Each instruction: {"op": mnemonic, "args": [operand...], "mods": [modifier...], "enc": bytes, "pc": byte offset}
Operand kinds: sgpr s5 / s[4:7], vgpr v3 / v[0:3], special (vcc, exec, m0, scc...), inline constant, literal,
label/offset (for branches), hw register, and raw text for anything else.

As a module: parse_file(path) -> {"name", "code_start", "insns": [...], "invalid": [pc...]}
As a tool:  python -I shader_ir.py <disasm dir> [--limit N]   (parses everything and reports coverage)
Usage: `python -I tools/shader_ir.py --help` lists the options.
"""
import argparse
import os
import re
import sys
from collections import Counter

ENC = re.compile(r";\s*encoding:\s*\[([^\]]*)\]")
HDR = re.compile(r"^;\s*(\S+)\s+code 0x([0-9a-f]+)\.\.0x([0-9a-f]+)")
SPECIAL = {"vcc", "vcc_lo", "vcc_hi", "exec", "exec_lo", "exec_hi", "m0", "scc", "null", "off", "src_scc",
           "src_execz", "src_vccz", "src_lds_direct", "src_shared_base", "src_shared_limit",
           "src_private_base", "src_private_limit", "src_pops_exiting_wave_id", "lds", "flat_scratch",
           "flat_scratch_lo", "flat_scratch_hi", "ttmp", "tba", "tma"}

# operand() runs once per operand of every instruction of every shader, so its patterns are compiled once here
# rather than looked up in re's cache on each call.
_NEG = re.compile(r"neg\((.+)\)")
_ABS = re.compile(r"abs\((.+)\)")
_SEXT = re.compile(r"sext\((.+)\)")
_REG = re.compile(r"([sv])(\d+)")
_REG_RANGE = re.compile(r"([sv])\[(\d+):(\d+)\]")
_TTMP = re.compile(r"ttmp(\d+)|ttmp\[(\d+):(\d+)\]")
_VLIST = re.compile(r"\[(v\d+(?:,\s*v\d+)+)\]")
_SLIST = re.compile(r"\[((?:s\d+|vcc_lo|vcc_hi)(?:\s*,\s*(?:s\d+|vcc_lo|vcc_hi))+)\]")
_ATTR = re.compile(r"attr(\d+)\.([xyzw])")
_TARGET = re.compile(r"(pos|param|mrt|mrtz|null|prim)(\d*)")
_INT = re.compile(r"-?(0x[0-9a-fA-F]+|\d+)")
_FLOAT = re.compile(r"-?\d+\.\d+(e[+-]?\d+)?")
# LLVM 23 prints a literal it could also show as an inline constant as lit(V), e.g. a v_fma_mix* f32 source
# `lit(0x42200000)`, where LLVM 21 printed the inline constant of the low half (0) and dropped the literal.
# Only a numeric constant is unwrapped: lit(v2) is no literal and stays raw (Unsupported), as before.
_LIT = re.compile(r"lit\((-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d+(?:e[+-]?\d+)?)?))\)")


def operand(tok):
    t = tok.strip()
    if "lit(" in t:
        # lit(V) is the constant V. A leading '-' on it is the VOP3 neg modifier, as neg(V) (not integer negation).
        t = _LIT.sub(r"\1", t)
        if t.startswith("-"):
            t = "neg(" + t[1:] + ")"
    neg = t.startswith("-")
    core = t[1:] if neg else t
    absm = len(core) > 2 and core.startswith("|") and core.endswith("|")
    if absm:
        core = core[1:-1]
    # Fast path for the commonest operands: single register (v3, s12), special register, or unsigned decimal int.
    if len(core) > 1 and core[0] in "sv" and core[1:].isdigit():
        return {"k": core[0] + "gpr", "r": int(core[1:]), "n": 1, "neg": neg, "abs": absm}
    if core in SPECIAL:
        return {"k": "special", "t": core, "neg": neg, "abs": absm}
    if core.isascii() and core.isdigit():
        v = int(core, 10)
        if absm:
            out = {"k": "imm", "v": v, "abs": True}
            if neg:
                out["neg"] = True
            return out
        return {"k": "imm", "v": -v if neg else v}
    if "(" in core:
        m = _NEG.fullmatch(core)
        if m:  # VOP3 FP negation: neg(0), neg(1.0), neg(0x3f800000), neg(|1.0|), neg(v1)
            inner = operand(m.group(1))
            if absm:
                # Outer |...| absorbs the inner neg; outer '-' (if any) negates the result.
                inner["abs"] = True
                if neg:
                    inner["neg"] = True
                elif inner.get("k") in ("imm", "fimm"):
                    inner.pop("neg", None)
                else:
                    inner["neg"] = False
            elif not neg:
                # Keep neg as a VOP3 sign-bit modifier on constants instead of integer negation:
                # in LLVM AMDGPU, neg(1) flips bit 31 (0x80000001) while -1 is 0xffffffff.
                new_neg = not inner.get("neg", False)
                if not new_neg and inner.get("k") in ("imm", "fimm"):
                    inner.pop("neg", None)
                else:
                    inner["neg"] = new_neg
            return inner
        m = _ABS.fullmatch(core)
        if m:  # VOP3 FP absolute value: abs(v0), abs(0xbf800000), neg(abs(1.0))
            inner = operand(m.group(1))
            inner["abs"] = True
            if neg:
                inner["neg"] = True
            elif inner.get("k") in ("imm", "fimm"):
                inner.pop("neg", None)
            else:
                inner["neg"] = False
            return inner
        m = _SEXT.fullmatch(core)
        if m:  # SDWA integer sign-extension of the selected source bits
            inner = operand(m.group(1))
            inner["sext"] = True
            if absm:
                inner["abs"] = True
            inner["neg"] = inner.get("neg", False) or neg
            return inner
    if "[" in core:
        m = _REG_RANGE.fullmatch(core)
        if m:
            lo, hi = int(m.group(2)), int(m.group(3))
            return {"k": m.group(1) + "gpr", "r": lo, "n": hi - lo + 1, "neg": neg, "abs": absm}
        m = _VLIST.fullmatch(core)
        if m:  # NSA image address: non-contiguous VGPR list
            return {"k": "vlist", "r": [int(x.strip()[1:]) for x in m.group(1).split(",")]}
        m = _SLIST.fullmatch(core)
        if m:  # LLVM VCCTuple scalar register list, e.g. [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]
            items = [int(x[1:]) if x.startswith("s") else x for x in (t.strip() for t in m.group(1).split(","))]
            return {"k": "slist", "r": items, "n": len(items)}
    if core.startswith("ttmp"):
        m = _TTMP.fullmatch(core)
        if m:
            return {"k": "ttmp", "t": core}
    if core.startswith("attr"):
        m = _ATTR.fullmatch(core)
        if m:
            return {"k": "attr", "a": int(m.group(1)), "c": "xyzw".index(m.group(2))}
    if core in ("p0", "p10", "p20"):
        return {"k": "interp", "t": core}
    m = _TARGET.fullmatch(core)
    if m and core not in ("null",):
        return {"k": "target", "t": core}
    if _INT.fullmatch(core):
        # int(core, 0) rejects the leading-zero decimals this regex accepts ('010', '08', '0001'),
        # which aborts a whole parse_file() run on one disassembly line.
        neg_core = core.startswith("-")
        digits = core[1:] if neg_core else core
        v = int(digits, 16) if digits.lower().startswith("0x") else int(digits, 10)
        if neg_core:
            v = -v
        if absm:
            out = {"k": "imm", "v": v, "abs": True}
            if neg:
                out["neg"] = True
            return out
        return {"k": "imm", "v": -v if neg else v}
    if _FLOAT.fullmatch(core):
        if absm:
            out = {"k": "fimm", "v": float(core), "abs": True}
            if neg:
                out["neg"] = True
            return out
        return {"k": "fimm", "v": float(core) * (-1 if neg else 1), "abs": False}
    if core.startswith("hwreg(") or core.startswith("sendmsg(") or core.startswith("vmcnt(") or \
            core.startswith("expcnt(") or core.startswith("lgkmcnt(") or core.startswith("vscnt(") or \
            core.startswith("depctr") or core.startswith("instid") or core.startswith("quad_perm") or \
            core.startswith("row_") or core.startswith("dpp") or core.startswith("sel:") or "(" in core:
        return {"k": "special", "t": core}
    return {"k": "raw", "t": t}


def split_args(s):
    if "(" not in s and "[" not in s:
        parts = s.split(",")
        if parts and not parts[-1].strip():
            parts.pop()
        return parts
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


SENDMSG = re.compile(r"sendmsg\([^)]*\)")

MODWORDS = re.compile(r"\b(offset:(?:swizzle\([^)]*\)|\S+)|offen|idxen|glc|slc|dlc|lds|tfe|unorm|da|d16|a16|nv|r128|lwe|clamp|"
                      r"mul:\d+|div:\d+|op_sel(?:_hi)?:\[[^\]]*\]|neg_lo:\[[^\]]*\]|neg_hi:\[[^\]]*\]|"
                      r"dmask:\S+|dim:\S+|format:\S+|done|vm|compr|row_mask:\S+|bank_mask:\S+|"
                      r"bound_ctrl:\S+|fi:\S+|offset0:\S+|offset1:\S+|gds|inst_offset:\S+|"
                      r"dst_sel:\S+|dst_unused:\S+|src0_sel:\S+|src1_sel:\S+|high|dpp8:\[[^\]]*\]|"
                      r"row_mirror|row_half_mirror|row_shl:\d+|row_shr:\d+|row_ror:\d+|row_share:\d+|"
                      r"row_xmask:\d+|quad_perm:\[[^\]]*\])(?=\s|,|$)")


def parse_line(line):
    code, _, comment = line.partition(";")
    enc = None
    if comment and "encoding" in comment:
        m = ENC.search(line)
        if m:
            enc = bytes(int(x, 16) for x in m.group(1).split(",") if x.strip())
    code = code.strip()
    if not code:
        return None
    fields = code.split(maxsplit=1)
    op = fields[0]
    rest = fields[1] if len(fields) > 1 else ""
    # sendmsg(MSG_GS, GS_OP_NOP) contains a comma: keep it as one operand.
    if "sendmsg(" in rest:
        rest = SENDMSG.sub(lambda m: m.group(0).replace(", ", "|").replace(",", "|"), rest)
    mods = []
    if rest:
        def _take_mod(match):
            mods.append(match.group(1))
            return ""
        rest = MODWORDS.sub(_take_mod, rest)
    parts = [a for a in split_args(rest) if a.strip()]
    if op in ("s_waitcnt", "s_waitcnt_depctr"):
        # LLVM separates named wait counters with spaces.
        parts = [token for part in parts for token in part.split()]
    elif parts and len(parts[0].split()) == 2:
        # 'exp pos0 v3, ...' / 'exp null off, ...': the target is space-separated from the first operand.
        parts = parts[0].split() + parts[1:]
    args = [operand(a) for a in parts]
    if op.startswith(("v_add_f64", "v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64",
                      "v_frexp_mant_f64", "v_frexp_exp_i32_f64", "v_trunc_f64", "v_ceil_f64",
                      "v_rndne_f64", "v_floor_f64")):
        for src, spelling in zip(args[1:], parts[1:]):
            # LLVM prints inline integers in decimal and the separate literal dword in hex,
            # even for two sources with the same numeric value (e.g. 0x1 and 1).
            if src["k"] == "imm" and ("0x" in spelling or not -16 <= src["v"] <= 64):
                src["f64_literal"] = True
    return {"op": op, "args": args, "mods": mods, "enc": enc.hex() if enc else None}


def parse_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    info = {"name": os.path.splitext(os.path.basename(path))[0], "insns": [], "invalid": 0}
    m = HDR.match(lines[0]) if lines else None
    pc = int(m.group(2), 16) if m else 0
    info["code_start"] = pc
    # the end of the code range the header declares (exclusive), None without a header: undecodable words can sit
    # between the last printed instruction and it
    info["code_end"] = int(m.group(3), 16) if m else None
    start = 1 if (lines and not lines[0].startswith("\t")) else 0
    for ln in lines[start:]:
        if ln.startswith("; ---"):
            break
        if not ln.startswith("\t") or ln.lstrip().startswith("."):
            continue
        ins = parse_line(ln)
        if not ins:
            continue
        ins["pc"] = pc
        if ins["enc"]:
            pc += len(ins["enc"]) // 2
        info["insns"].append(ins)
    tail = "\n".join(lines)
    info["invalid"] = tail.count("invalid instruction encoding")
    return info


def positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Parse a folder of .s disassembly and report operand coverage.")
    ap.add_argument("src", help="folder of .s files written by shader_disasm.py")
    ap.add_argument("--limit", type=positive_int, help="parse only the first N files, sorted by name")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.src):
        ap.error(f"{a.src} is not a folder")
    try:
        with os.scandir(a.src) as entries:
            files = sorted(entry.path for entry in entries
                           if entry.name.endswith(".s") and entry.is_file())[:a.limit]
    except OSError as e:
        print(f"ERROR: cannot list {a.src}: {e.strerror or e}", file=sys.stderr)
        return 2
    kinds, raw, n_ins, no_enc = Counter(), Counter(), 0, 0
    for path in files:
        try:
            info = parse_file(path)
        except OSError as e:
            print(f"ERROR: cannot read {e.filename or path}: {e.strerror or e}", file=sys.stderr)
            return 2
        for ins in info["insns"]:
            n_ins += 1
            no_enc += ins["enc"] is None
            for a in ins["args"]:
                kinds[a["k"]] += 1
                if a["k"] == "raw":
                    raw[a["t"]] += 1
    print(f"{len(files)} shaders, {n_ins:,} instructions, {no_enc} without encoding bytes")
    print("operand kinds:", dict(kinds.most_common()))
    print("unparsed (raw) operands:", raw.most_common(15))


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
