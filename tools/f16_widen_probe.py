r"""Measure how RDNA2 (GFX10) widens a half-precision NaN in a typed buffer load (FUZZ-5 follow-up).

`tools/mem_diff.py` reported one class of mismatch: a typed buffer load of a half NaN. Its Python reference decodes
16_FLOAT through `struct.unpack('<e')`, which on Python 3.14 keeps the NaN payload (CPython gh-130317), while the
translated SPIR-V decodes it through `UnpackHalf2x16` (`Gen.decode_format`, `tools/rdna2spirv.py`) and the mini
interpreter gives the canonical quiet NaN (0x7FC00000 / 0xFFC00000). Which one is right had never been
measured. The same halves also go through `v_cvt_f32_f16`, the instruction a driver lowers `UnpackHalf2x16` to,
so the probe reports that separately.

Each kernel here stores one chosen half word into the output buffer with an untyped `buffer_store_dword`, waits,
loads it back with `buffer_load_format_*` (MUBUF) or `tbuffer_load_format_*` (MTBUF), waits, and stores the
resulting f32 lanes over the first words of the output buffer. A wave32 and wave64 run per case.

Usage:
    python -I tools/f16_widen_probe.py [--waves 32,64] [--json out.json]
    python -I tools/f16_widen_probe.py --self-test        # pure-Python model checks, no GPU

Exit 0 when the sweep ran (every case produced the expected lane count), 1 on a report-write error, 2 on a CLI
argument error, 77 when tools/hw_oracle is unavailable or the GPU is not GFX10. Standard library only; nothing
here reads game files.
"""

import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# (label, half word) -- the fuzz finds, both specials, two controls, and an sNaN/qNaN whose payload is a single bit.
HALVES = (
    ("snan_7ddb", 0x7DDB),
    ("qnan_7fe6", 0x7FE6),
    ("payload_7fbb", 0x7FBB),
    ("neg_payload_ffb2", 0xFFB2),
    ("payload_7fcb", 0x7FCB),
    ("neg_payload_ff96", 0xFF96),
    ("snan_min_7c01", 0x7C01),
    ("neg_snan_min_fc01", 0xFC01),
    ("one_3c00", 0x3C00),
    ("neg_two_c000", 0xC000),
)

# (GFX10 format code, name, widths) for the three 16-bit float buffer formats.
FLOAT_FORMATS = (
    (13, "16_FLOAT", (16,)),
    (29, "16_16_FLOAT", (16, 16)),
    (71, "16_16_16_16_FLOAT", (16, 16, 16, 16)),
)
INSTRUCTIONS = ("x", "xy", "xyzw")
FORMS = ("buffer", "tbuffer")

SEL_XYZW = (4, 5, 6, 7)  # DST_SEL X<-X, Y<-Y, Z<-Z, W<-W


def vsharp_d3(code, sel=SEL_XYZW):
    sx, sy, sz, sw = sel
    return sx | (sy << 3) | (sz << 6) | (sw << 9) | (code << 12) | (1 << 24) | (3 << 28)


def h2f_payload(h):
    """f16 -> f32 bits keeping sign and NaN payload (the 'widen by shifting' rule, and Python 3.14's '<e')."""
    h &= 0xFFFF
    sign = (h >> 15) << 31
    exp = (h >> 10) & 0x1F
    frac = h & 0x03FF
    if exp == 0x1F:
        return sign | 0x7F800000 | (frac << 13)
    if exp == 0 and frac:
        b = frac.bit_length()
        return sign | ((b + 102) << 23) | ((frac << (24 - b)) & 0x7FFFFF)
    if exp == 0:
        return sign
    return sign | ((exp + 112) << 23) | (frac << 13)


def h2f_canonical(h):
    """f16 -> f32 bits with every NaN collapsed to the canonical quiet NaN, sign kept (mini interpreter rule)."""
    h &= 0xFFFF
    if (h >> 10) & 0x1F == 0x1F and h & 0x03FF:
        return ((h >> 15) << 31) | 0x7FC00000
    return h2f_payload(h)


def h2f_quiet(h):
    """f16 -> f32 bits keeping sign and payload with the quiet bit forced (a sNaN quieted, payload kept)."""
    bits = h2f_payload(h)
    if (h >> 10) & 0x1F == 0x1F and h & 0x03FF:
        bits |= 1 << 22
    return bits


def classify(bits, half):
    """How an observed f32 word relates to the half it came from."""
    payload = h2f_payload(half)
    canonical = h2f_canonical(half)
    quiet = h2f_quiet(half)
    if bits == payload:
        return "payload"
    if bits == quiet and quiet != payload:
        return "payload_quiet"
    if bits == canonical and canonical != payload:
        return "canonical"
    if bits == 0:
        return "zero"
    return "other"


def _kernel(asm_mod, inst, code, half, wave):
    ncomp = len(inst.rsplit("_", 1)[1])
    vreg = "v4" if ncomp == 1 else f"v[4:{3 + ncomp}]"
    load = (
        f"{inst} {vreg}, off, s[8:11], 0 offset:32"
        if inst.startswith("buffer_")
        else f"{inst} {vreg}, off, s[8:11], 0 format:{code} offset:32"
    )
    lines = [
        "s_mov_b32 s8, s2",
        "s_and_b32 s9, s3, 0xffff",
        "s_mov_b32 s10, 0x100",
        f"s_mov_b32 s11, 0x{vsharp_d3(code):x}",
        "v_mov_b32 v4, 0",
        "buffer_store_dword v4, off, s[8:11], 0 offset:36",
        f"v_mov_b32 v4, 0x{half:x}",
        "buffer_store_dword v4, off, s[8:11], 0 offset:32 glc",
        "s_waitcnt_vscnt null, 0x0",
        load,
        "s_waitcnt vmcnt(0)",
    ]
    for i in range(ncomp):
        lines.append(f"buffer_store_dword v{4 + i}, off, s[8:11], 0 offset:{4 * i}")
    lines += ["s_waitcnt_vscnt null, 0x0", "s_endpgm"]
    return asm_mod.Kernel(lines, wave, 1, 16, raw=True, vgprs=12, sgprs=16)


def _alu_kernel(asm_mod, half, wave):
    """The half word through v_cvt_f32_f16: what a driver lowers UnpackHalf2x16 to."""
    lines = [
        "s_mov_b32 s8, s2",
        "s_and_b32 s9, s3, 0xffff",
        "s_mov_b32 s10, 0x100",
        f"s_mov_b32 s11, 0x{vsharp_d3(13):x}",
        f"v_mov_b32 v4, 0x{half:x}",
        "buffer_store_dword v4, off, s[8:11], 0 offset:32 glc",
        "s_waitcnt_vscnt null, 0x0",
        "buffer_load_dword v4, off, s[8:11], 0 offset:32",
        "s_waitcnt vmcnt(0)",
        "v_cvt_f32_f16 v5, v4",
        "buffer_store_dword v5, off, s[8:11], 0 offset:0",
        "s_waitcnt_vscnt null, 0x0",
        "s_endpgm",
    ]
    return asm_mod.Kernel(lines, wave, 1, 16, raw=True, vgprs=12, sgprs=16)


def cases():
    """(case name, format code, form, instruction, lane count) for every sweep entry."""
    for code, name, widths in FLOAT_FORMATS:
        for form in FORMS:
            for suffix in INSTRUCTIONS:
                ncomp = len(suffix)
                yield f"fmt{code}/{name}/{form}_{suffix}", code, form, suffix, ncomp


def sweep(oracle, asm_mod, wave):
    out = {}
    order, kernels = [], []
    for case, code, form, suffix, n in cases():
        for label, half in HALVES:
            order.append((case, code, form, suffix, n, label, half))
            kernels.append(
                _kernel(asm_mod, f"{form}_load_format_{suffix}", code, half, wave)
            )
    got = oracle.run(kernels, words=16)
    for (case, code, form, suffix, n, label, half), words in zip(order, got):
        if isinstance(words, str):
            raise RuntimeError(f"{case} {label} wave{wave}: {words}")
        lanes = words[:n]
        case_out = out.setdefault(case, {})
        case_out[label] = {
            "half": f"0x{half:04x}",
            "words": [f"0x{w:08x}" for w in lanes],
            "classes": [classify(w, half) for w in lanes],
        }
    order, kernels = [], []
    for label, half in HALVES:
        order.append(("alu/v_cvt_f32_f16", label, half))
        kernels.append(_alu_kernel(asm_mod, half, wave))
    got = oracle.run(kernels, words=16)
    for (case, label, half), words in zip(order, got):
        if isinstance(words, str):
            raise RuntimeError(f"{case} {label} wave{wave}: {words}")
        out.setdefault(case, {})[label] = {
            "half": f"0x{half:04x}",
            "words": [f"0x{words[0]:08x}"],
            "classes": [classify(words[0], half)],
        }
    return out


def self_test():
    # Payload rule, bit for bit.
    assert h2f_payload(0x3C00) == 0x3F800000
    assert h2f_payload(0xC000) == 0xC0000000
    assert h2f_payload(0x7DDB) == 0x7FBB6000  # sNaN, payload kept, quiet bit not forced
    assert h2f_payload(0x7FE6) == 0x7FFCC000
    assert h2f_payload(0xFFB2) == 0xFFF64000
    assert h2f_payload(0x7C01) == 0x7F802000
    assert h2f_payload(0x0001) == 0x33800000  # smallest subnormal half
    # Canonical rule, and the quiet-bit variant.
    assert h2f_canonical(0x7DDB) == 0x7FC00000
    assert h2f_canonical(0xFFB2) == 0xFFC00000
    assert h2f_canonical(0x7FE6) == 0x7FC00000
    assert h2f_canonical(0x3C00) == 0x3F800000
    assert h2f_quiet(0x7DDB) == 0x7FFB6000
    assert h2f_quiet(0x7FE6) == 0x7FFCC000
    assert h2f_quiet(0x3C00) == 0x3F800000
    # classify() names the three outcomes apart.
    assert classify(0x7FBB6000, 0x7DDB) == "payload"
    assert classify(0x7FC00000, 0x7DDB) == "canonical"
    assert classify(0x7FFB6000, 0x7DDB) == "payload_quiet"
    assert classify(0x7FBB6001, 0x7DDB) == "other"
    assert classify(0x00000000, 0x7DDB) == "zero"
    # Every control widens exactly in all three models.
    for h in (0x3C00, 0xC000):
        assert h2f_payload(h) == h2f_canonical(h) == h2f_quiet(h)
        assert struct.unpack(
            "<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", h))[0])
        )[0] == h2f_payload(h)
    print("self-test OK")


def parse_waves(raw):
    tokens = [w.strip() for w in raw.split(",") if w.strip()]
    if not tokens:
        return None, "--waves must not be empty"
    waves = []
    for tok in tokens:
        try:
            w = int(tok)
        except ValueError:
            return None, f"invalid wave size {tok!r} (expected 32 or 64)"
        if w not in (32, 64):
            return None, f"unsupported wave size {w} (expected 32 or 64)"
        waves.append(w)
    return waves, None


def validate_json_path(path):
    if not path or not path.strip():
        return "--json path must not be empty"
    if "\x00" in path:
        return "--json path contains a NUL byte"
    if (
        path.endswith(os.sep)
        or (os.altsep and path.endswith(os.altsep))
        or os.path.isdir(path)
    ):
        return f"--json path is a directory: {path}"
    parent = os.path.dirname(os.path.abspath(path))
    while parent:
        if os.path.isdir(parent):
            break
        if os.path.lexists(parent):
            return f"--json parent is not a directory: {parent}"
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    return None


def write_json_report(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"cannot create directory for --json {path}: {e.strerror or e}"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        return f"cannot write --json {path}: {e.strerror or e}"
    return None


def print_table(out, wave):
    print(f"wave{wave}:")
    for case, info in out.items():
        print(f"  {case}")
        for label, _half in HALVES:
            e = info[label]
            print(
                f"    {e['half']} -> {' '.join(e['words'])}  [{' '.join(e['classes'])}]"
            )


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--waves", default="32,64", help="wave sizes to run (default: 32,64)"
    )
    ap.add_argument(
        "--json", dest="json_out", help="write the full JSON report to this path"
    )
    ap.add_argument(
        "--self-test", action="store_true", help="check the Python model without a GPU"
    )
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    waves, err = parse_waves(args.waves)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        err = validate_json_path(args.json_out)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return 77
    if asm.family(oracle.arch) != "gfx10":
        print(f"SKIP: {oracle.arch} is not GFX10 (RDNA1/RDNA2)", file=sys.stderr)
        return 77
    out = {"arch": oracle.arch, "waves": {}}
    for wave in waves:
        report = sweep(oracle, asm, wave)
        out["waves"][str(wave)] = report
        print_table(report, wave)
    if args.json_out is not None:
        err = write_json_report(args.json_out, out)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
