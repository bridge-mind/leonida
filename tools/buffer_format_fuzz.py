r"""Hardware oracle for the GFX10 buffer data formats a V# or MTBUF can name (standard library only).

For every format code 1..77 (GFX10_FORMAT_*, tools/gfx10_formats.json) that the GPU also has, one raw kernel per batch
of 16-byte elements runs `buffer_load_format_xyzw` (the format comes from the V#) or `tbuffer_load_format_xyzw` (the
format comes from the instruction) over many bit patterns, or `buffer_store_format_xyzw` of many register values into
a buffer pre-filled with 0xDEAD0000, and every result word is compared with MODEL, the CPU decode / encode this file defines. The
translator's emission (Gen.decode_format / Gen.encode_format in tools/rdna2spirv.py) is held to the same MODEL by
tests/semantic/test_buffer_formats_all.py, so the chain is hardware -> MODEL -> translated SPIR-V.

The Radeon 880M (gfx1150) is RDNA3.5, not RDNA2, and numbers its formats differently (GFX11 has 63; its 10_11_11 / 11_11_10
families keep only FLOAT and it has no SCALED 10_10_10_2), so a format is mapped by NAME and the 14 GFX10-only codes
(30..35, 37..42, 46, 47) cannot be run there. On a GFX10 device (the Linux DRM backend, tools/hw_oracle/drm.py, native
GFX10 numbers) the default run covers all 77. The model was validated on the 880M and on a real RDNA2 (gfx1036,
@AwkGrep256, #2022: 1,330,469 elements on the 63 codes and 199,144 on the 14 others, 0 mismatches), with the same
two-component buffer_store_format_* routing (x, w) on both (docs/buffer_formats.md).

Usage:
    python -I tools/buffer_format_fuzz.py --self-test
    python -I tools/buffer_format_fuzz.py --oracle hw [--mode load|store|all] [--path mubuf|mtbuf|both]
                                          [--formats 56,36,...] [--seed S] [--negative-control]
Exit code: 0 all match, 1 a mismatch, 2 a usage or harness error, 77 no hardware oracle. Nothing here reads game files.
"""
import argparse
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PREFILL = 0xDEAD0000        # what the launcher writes into the output before a kernel runs (hw_oracle.c)

# GFX11's unified format numbers by name, from the order the AMD assembler (LLVM 23.1.5, -mcpu=gfx1150) gives
# `format:N`; GFX11 has no code for the other GFX10 buffer formats (its names below are all it knows).
GFX11_CODES = {
    "8_UNORM": 1, "8_SNORM": 2, "8_USCALED": 3, "8_SSCALED": 4, "8_UINT": 5, "8_SINT": 6, "16_UNORM": 7,
    "16_SNORM": 8, "16_USCALED": 9, "16_SSCALED": 10, "16_UINT": 11, "16_SINT": 12, "16_FLOAT": 13,
    "8_8_UNORM": 14, "8_8_SNORM": 15, "8_8_USCALED": 16, "8_8_SSCALED": 17, "8_8_UINT": 18, "8_8_SINT": 19,
    "32_UINT": 20, "32_SINT": 21, "32_FLOAT": 22, "16_16_UNORM": 23, "16_16_SNORM": 24, "16_16_USCALED": 25,
    "16_16_SSCALED": 26, "16_16_UINT": 27, "16_16_SINT": 28, "16_16_FLOAT": 29, "10_11_11_FLOAT": 30,
    "11_11_10_FLOAT": 31, "10_10_10_2_UNORM": 32, "10_10_10_2_SNORM": 33, "10_10_10_2_UINT": 34,
    "10_10_10_2_SINT": 35, "2_10_10_10_UNORM": 36, "2_10_10_10_SNORM": 37, "2_10_10_10_USCALED": 38,
    "2_10_10_10_SSCALED": 39, "2_10_10_10_UINT": 40, "2_10_10_10_SINT": 41, "8_8_8_8_UNORM": 42,
    "8_8_8_8_SNORM": 43, "8_8_8_8_USCALED": 44, "8_8_8_8_SSCALED": 45, "8_8_8_8_UINT": 46, "8_8_8_8_SINT": 47,
    "32_32_UINT": 48, "32_32_SINT": 49, "32_32_FLOAT": 50, "16_16_16_16_UNORM": 51, "16_16_16_16_SNORM": 52,
    "16_16_16_16_USCALED": 53, "16_16_16_16_SSCALED": 54, "16_16_16_16_UINT": 55, "16_16_16_16_SINT": 56,
    "16_16_16_16_FLOAT": 57, "32_32_32_UINT": 58, "32_32_32_SINT": 59, "32_32_32_FLOAT": 60,
    "32_32_32_32_UINT": 61, "32_32_32_32_SINT": 62, "32_32_32_32_FLOAT": 63,
}


def load_gfx10_names():
    with open(os.path.join(HERE, "gfx10_formats.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v[len("GFX10_FORMAT_"):] for k, v in raw.items() if 1 <= int(k) <= 77}


GFX10_NAMES = load_gfx10_names()           # code -> "8_8_8_8_UNORM"


def parse_name(name):
    """(component widths from bit 0, numeric format). Names list the widths from the top bit down, so the last
    number of 2_10_10_10 (50) is the x field in bits 9:0 and 10_10_10_2 (44) has a 2-bit x in bits 1:0."""
    *w, nfmt = name.split("_")
    return tuple(int(x) for x in reversed(w)), nfmt


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def frac_to_f32(q: Fraction) -> int:
    """f32 bits of a non-negative fraction, correctly rounded (ties to even)."""
    if q == 0:
        return 0
    num, den = q.numerator, q.denominator
    e = num.bit_length() - den.bit_length()
    if Fraction(num, den) < Fraction(2) ** e:
        e -= 1                                  # 2^e <= q < 2^(e+1)
    e = max(e, -126)
    scaled = q / (Fraction(2) ** (e - 23))     # mantissa units of 2^(e-23)
    m = scaled.numerator // scaled.denominator
    rem = scaled - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m & 1):
        m += 1
    if e == -126 and m < (1 << 23):
        return m                                # denormal range is never reached by these formats
    if m >= 1 << 24:
        m >>= 1
        e += 1
    return ((e + 127) << 23) | (m - (1 << 23))


def sign_extend(v, w):
    return v - (1 << w) if v >> (w - 1) & 1 else v


# ---- the model: what a load returns, what a store writes ----

# Small unsigned floats: 5 exponent bits, bias 15, no sign. 11 bits carry 6 mantissa bits, 10 bits carry 5.
def ufloat_decode(raw, w):
    mbits = w - 5
    e, m = raw >> mbits, raw & ((1 << mbits) - 1)
    if e == 31:
        return 0x7F800000 | (m << (23 - mbits))      # infinity, or a NaN whose payload moves up unquieted (measured)
    if e == 0:
        return frac_to_f32(Fraction(m, 1 << mbits) * Fraction(2) ** -14)
    return frac_to_f32((1 + Fraction(m, 1 << mbits)) * Fraction(2) ** (e - 15))


def half_decode(raw):
    h = struct.unpack("<e", struct.pack("<H", raw))[0]
    if math.isnan(h):       # sign and payload move up unquieted (measured); Python would quiet a signalling NaN
        return (raw & 0x8000) << 16 | 0x7F800000 | ((raw & 0x3FF) << 13)
    return f32_bits(h)


def decode_field(nfmt, w, raw):
    """One component's result word for the field value `raw` (w bits)."""
    if nfmt == "UNORM":
        return frac_to_f32(Fraction(raw, (1 << w) - 1))
    if nfmt == "SNORM":
        s = sign_extend(raw, w)
        if s == -(1 << (w - 1)):
            return 0xBF800000
        q = frac_to_f32(Fraction(abs(s), (1 << (w - 1)) - 1))
        return q | (0x80000000 if s < 0 else 0)
    if nfmt == "USCALED":
        return frac_to_f32(Fraction(raw))
    if nfmt == "SSCALED":
        s = sign_extend(raw, w)
        return frac_to_f32(Fraction(abs(s))) | (0x80000000 if s < 0 else 0)
    if nfmt == "UINT":
        return raw
    if nfmt == "SINT":
        return sign_extend(raw, w) & 0xFFFFFFFF
    if nfmt == "FLOAT":
        if w == 32:
            return raw
        if w == 16:
            return half_decode(raw)
        return ufloat_decode(raw, w)
    raise ValueError(nfmt)


def decode_element(name, data, path="mubuf"):
    """The four result words of an xyzw load of the element at data[0:]. Through a V# (DST_SEL = XYZW) a channel the
    format lacks repeats component (c mod n), which is what the translator does and the hardware returned. tbuffer
    takes its selectors from the instruction (X000, XY00, XYZ0, XYZW by the format's component count): the rest is 0."""
    widths, nfmt = parse_name(name)
    out, bit = [], 0
    big = int.from_bytes(data[:16], "little")
    for w in widths:
        out.append(decode_field(nfmt, w, (big >> bit) & ((1 << w) - 1)))
        bit += w
    n = len(out)
    if path == "mtbuf":
        return out + [0] * (4 - n)
    return [out[c % n] for c in range(4)]


def element_size(name):
    return sum(parse_name(name)[0]) // 8


# ---- the data: patterns per format ----

def edge_values(w, rng):
    vals = {0, 1, 2, (1 << w) - 1, (1 << w) - 2, 1 << (w - 1), (1 << (w - 1)) - 1, (1 << (w - 1)) + 1}
    vals |= {rng.getrandbits(w) for _ in range(4)}
    return sorted(v for v in vals if 0 <= v < 1 << w)


def pack(widths, comps, rng, size):
    big, bit = 0, 0
    for w, v in zip(widths, comps):
        big |= (v & ((1 << w) - 1)) << bit
        bit += w
    tail = rng.getrandbits(128 - bit) << bit            # bytes past the element are random: they must not leak in
    return (big | tail).to_bytes(16, "little")


def load_patterns(name, rng, budget):
    """16-byte elements for one format: every value of each field (all of it up to 12 bits, a 16-bit float's 65536,
    edges and random above that) with random other fields, then edge-value combinations and plain random bytes."""
    widths, nfmt = parse_name(name)
    size = element_size(name)
    elems = [bytes(16), b"\xff" * 16]
    for c, w in enumerate(widths):
        if w == 32:
            values = edge_values(32, rng) + [0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80000000, 0x3F800000,
                                             0xBF800000] + [rng.getrandbits(32) for _ in range(256)]
        elif w <= 12 or (w == 16 and nfmt == "FLOAT"):
            values = range(1 << w)
        else:
            values = edge_values(w, rng) + [rng.getrandbits(w) for _ in range(512)]
        for v in values:
            comps = [rng.getrandbits(x) for x in widths]
            comps[c] = v
            elems.append(pack(widths, comps, rng, size))
    for _ in range(budget):
        comps = [rng.choice(edge_values(w, rng)) for w in widths]
        elems.append(pack(widths, comps, rng, size))
        elems.append(rng.getrandbits(128).to_bytes(16, "little"))
    return elems


def f32_edges(rng):
    base = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 255.0, 255.5, 256.0, 1e-8, -1e-8, 1e30, -1e30, 65504.0, 65505.0,
            65520.0, 65519.0, 131008.0, 1e-5, 6.1e-5, 5.9e-8, 3.0e-8, 1.5, 2.5, 0.49999997, 0.50000006, 100.5, 101.5,
            1023.0, 1023.5, 1024.0, -1023.0, -1024.0, -511.5, 511.5, 512.0, 511.0, 3.0, -3.0, 0.3, -0.3]
    bits = [f32_bits(x) for x in base] + [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x00000001,
                                          0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF]
    return bits + [rng.getrandbits(32) for _ in range(40)] + [f32_bits(rng.uniform(-3, 3)) for _ in range(40)] + \
        [f32_bits(rng.uniform(-300, 300)) for _ in range(40)] + [f32_bits(math.ldexp(rng.uniform(1, 2), rng.randrange(-20, 18)))
                                                                  for _ in range(60)]


def int_edges(rng):
    base = [0, 1, 2, 3, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 32767, 32768, 65535, 65536, 0x7FFFFFFF,
            0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFF80, 0xFFFFFF7F, 0xFFFFFE00, 0xFFFFFC00, 0xFFFF8000, 0xFFFF7FFF,
            0xFFFFFFFF - 1023, 0xFFFFFFFF - 511, 0xFFFFFFFF - 4095]
    return base + [rng.getrandbits(32) for _ in range(60)] + [rng.getrandbits(rng.randrange(1, 17)) for _ in range(60)] + \
        [(-rng.getrandbits(rng.randrange(1, 17))) & 0xFFFFFFFF for _ in range(60)]


def store_inputs(name, rng):
    """Register values (4 dwords per element) for a store: every channel takes values from the right domain, so a
    float format sees f32 edges, an integer format integer edges. Channel 0 sweeps while the others hold other values."""
    widths, nfmt = parse_name(name)
    vals = int_edges(rng) if nfmt in ("UINT", "SINT") else f32_edges(rng)
    if nfmt in ("UNORM", "SNORM"):
        vals += [f32_bits(rng.uniform(-1.1, 1.1)) for _ in range(120)]
        vals += [f32_bits(k / ((1 << w) - 1) + d) for w in set(widths) for k in (0, 1, 2, (1 << w) - 2, (1 << w) - 1)
                 for d in (0.0, 1e-9, -1e-9, 0.5 / ((1 << w) - 1), -0.5 / ((1 << w) - 1))]
    elems = []
    for i, v in enumerate(vals):
        row = [v] + [vals[(i * 7 + 13 * k + 5) % len(vals)] for k in (1, 2, 3)]
        elems.append(row)
        elems.append([row[1], v, row[3], row[2]])
        elems.append([row[2], row[3], v, row[1]])
        elems.append([row[3], row[2], row[1], v])
    return elems


# ---- the kernels ----

SEL_CODES = {"0": 0, "1": 1, "X": 4, "Y": 5, "Z": 6, "W": 7}


def v_sharp_word3(arch_family, code, sel="XYZW"):
    w3 = sum(SEL_CODES[ch] << (3 * i) for i, ch in enumerate(sel)) | code << 12 | 3 << 28    # DST_SEL, FORMAT, OOB_SELECT = raw
    if arch_family == "gfx10":
        w3 |= 1 << 23                                                  # RESOURCE_LEVEL
    return w3


def _longs(words):
    return [f"  .long {', '.join(f'0x{w:08x}' for w in words[i:i + 8])}" for i in range(0, len(words), 8)]


def vgpr_clear():
    return [f"v_mov_b32 v{8 + i}, 0x{PREFILL + 0x100 + i:08x}" for i in range(4)]


def load_lines(uid, family, ref, path, elems):
    """Lane i loads element i of the table in the kernel's own code with `buffer_load_format_xyzw`/tbuffer and stores
    the four result registers at i * 16. ref: the format code for the V# (mubuf) or the BUF_FMT_ name (mtbuf)."""
    code = ref if path == "mubuf" else 22
    w3 = v_sharp_word3(family, code)
    lines = ["s_getpc_b64 s[4:5]", f".Lpc{uid}:", f"s_add_u32 s4, s4, .Ldat{uid}-.Lpc{uid}", "s_addc_u32 s5, s5, 0",
             "s_mov_b32 s8, s4", "s_and_b32 s9, s5, 0xffff", "s_mov_b32 s10, 0x100000", f"s_mov_b32 s11, 0x{w3:08x}",
             "v_lshlrev_b32 v1, 4, v0"] + vgpr_clear()
    if path == "mubuf":
        lines.append("buffer_load_format_xyzw v[8:11], v1, s[8:11], 0 offen")
    else:
        lines.append(f"tbuffer_load_format_xyzw v[8:11], v1, s[8:11], 0 format:[BUF_FMT_{ref}] offen")
    lines += ["s_waitcnt vmcnt(0)", "global_store_b128 v1, v[8:11], s[2:3]", "s_endpgm", ".p2align 4", f".Ldat{uid}:"]
    words = []
    for e in elems:
        words += list(struct.unpack("<4I", e))
    return lines + _longs(words)


def store_lines(uid, family, ref, path, rows, instr="xyzw", sel="XYZW"):
    """Lane i reads its four register values from the table, stores them with `buffer_store_format_<instr>` / tbuffer
    (instr xyzw or xy: the first two registers) into the output buffer at i * 16 (a V# whose base is the output pointer)."""
    regs = "v[8:11]" if instr == "xyzw" else "v[8:9]"
    code = ref if path == "mubuf" else 22
    w3 = v_sharp_word3(family, code, sel)
    lines = ["s_getpc_b64 s[4:5]", f".Lpc{uid}:", f"s_add_u32 s4, s4, .Ldat{uid}-.Lpc{uid}", "s_addc_u32 s5, s5, 0",
             "s_mov_b32 s12, s4", "s_and_b32 s13, s5, 0xffff", "s_mov_b32 s14, 0x100000", f"s_mov_b32 s15, 0x{v_sharp_word3(family, 22):08x}",
             "s_mov_b32 s8, s2", "s_and_b32 s9, s3, 0xffff", "s_mov_b32 s10, 0x100000", f"s_mov_b32 s11, 0x{w3:08x}",
             "v_lshlrev_b32 v1, 4, v0", "buffer_load_dwordx4 v[8:11], v1, s[12:15], 0 offen", "s_waitcnt vmcnt(0)"]
    if path == "mubuf":
        lines.append(f"buffer_store_format_{instr} {regs}, v1, s[8:11], 0 offen")
    else:
        lines.append(f"tbuffer_store_format_{instr} {regs}, v1, s[8:11], 0 format:[BUF_FMT_{ref}] offen")
    lines += ["s_endpgm", ".p2align 4", f".Ldat{uid}:"]
    words = []
    for r in rows:
        words += r
    return lines + _longs(words)


# ---- running ----

CHUNK = 1024        # elements (lanes) per kernel: one workgroup of 1024 threads, 4096 output words


def hw_code(family, code, name):
    """The format number this GPU family uses for GFX10 format `name` (None: the family has no such format)."""
    if family == "gfx10":
        return code
    return GFX11_CODES.get(name)


def run_loads(oracle, family, fmts, path, rng, budget, negative=None):
    """-> {code: (cases, mismatches [(element bytes, want words, got words)])}"""
    from hw_oracle import asm
    jobs, kernels = [], []
    for code, name in fmts:
        hc = hw_code(family, code, name)
        elems = load_patterns(name, rng, budget)
        ref = hc if path == "mubuf" else name
        if negative is not None:
            ref = negative(code, name, ref)
        for lo in range(0, len(elems), CHUNK):
            part = elems[lo:lo + CHUNK]
            uid = len(kernels)
            kernels.append(asm.Kernel(load_lines(uid, family, ref, path, part), 32, len(part), len(part) * 4, raw=True))
            jobs.append((code, name, part))
    results = oracle.run(kernels, words=CHUNK * 4)
    out = {}
    for (code, name, part), res in zip(jobs, results):
        cases, bad = out.setdefault(code, [0, []])
        if isinstance(res, str):
            bad.append((b"", [], [res]))
            continue
        for i, e in enumerate(part):
            got = res[4 * i:4 * i + 4]
            want = decode_element(name, e, path)
            cases += 1
            if not words_equal(got, want, name):
                bad.append((e, want, got))
        out[code][0] = cases
    return out


def words_equal(got, want, name):
    """Bit equality, NaN payloads included (the hardware keeps them)."""
    return list(got) == list(want)


# ---- stores ----

def to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rne(x):
    """Round half to even of a Fraction or float to an int."""
    q = Fraction(x)
    f = q.numerator // q.denominator
    r = q - f
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and f & 1):
        f += 1
    return f


def rne_away(x):
    """Round half away from zero of a non-negative Fraction."""
    return (x + Fraction(1, 2)).__floor__()


def small_float_encode(bits, mbits, signed):
    """f32 bits -> a 5-exponent-bit float with `mbits` mantissa bits (and a sign bit when `signed`), truncating the
    mantissa (round toward zero, measured). Overflow, NaN and the sign of an unsigned format: see the measurements."""
    x = to_f32(bits)
    sign = (0x8000 if signed else 0) if bits >> 31 else 0
    if math.isnan(x):
        return sign | (31 << mbits) | ((bits >> (23 - mbits) & ((1 << mbits) - 1)) or 1)   # a payload that shifts out stays a NaN
    if not signed and bits >> 31:
        return 0
    if math.isinf(x):
        return sign | (31 << mbits)
    q = abs(Fraction(x))
    if q == 0:
        return sign
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2) ** e:
        e -= 1
    if e < -14:             # denormal: multiples of 2^(-14 - mbits)
        return sign | (q / Fraction(2) ** (-14 - mbits)).__floor__()
    m = ((q / Fraction(2) ** e - 1) * (1 << mbits)).__floor__()
    if e > 15:
        return sign | (30 << mbits) | ((1 << mbits) - 1)      # too large for the format: the largest finite value
    return sign | ((e + 15) << mbits) | m


def ufloat_encode(bits, w):
    return small_float_encode(bits, w - 5, False)


def half_encode(bits):
    return small_float_encode(bits, 10, True)


def encode_field(nfmt, w, reg):
    if nfmt in ("UINT", "SINT"):
        if nfmt == "UINT":
            return min(reg, (1 << w) - 1)
        s = reg - (1 << 32) if reg >> 31 else reg
        return max(-(1 << (w - 1)), min((1 << (w - 1)) - 1, s)) & ((1 << w) - 1)
    if nfmt == "FLOAT":
        if w == 32:
            return reg
        if w == 16:
            return half_encode(reg)
        return ufloat_encode(reg, w)
    if nfmt == "USCALED":       # measured on gfx1150: a store converts a SCALED format as its NORM one
        nfmt = "UNORM"
    elif nfmt == "SSCALED":
        nfmt = "SNORM"
    x = to_f32(reg)
    if math.isnan(x):
        return 0
    fx = Fraction(x) if not math.isinf(x) else Fraction(1 << 40) * (1 if x > 0 else -1)
    if nfmt == "UNORM":
        return rne(max(Fraction(0), min(Fraction(1), fx)) * ((1 << w) - 1))
    if nfmt == "SNORM":
        v = max(Fraction(-1), min(Fraction(1), fx)) * ((1 << (w - 1)) - 1)
        return (-rne_away(-v) if v < 0 else rne_away(v)) & ((1 << w) - 1)
    if nfmt == "USCALED":
        return max(0, min((1 << w) - 1, rne(fx)))
    if nfmt == "SSCALED":
        return max(-(1 << (w - 1)), min((1 << (w - 1)) - 1, rne(fx))) & ((1 << w) - 1)
    raise ValueError(nfmt)


def encode_element(name, regs, instr="xyzw", route="mubuf", sel="XYZW"):
    """The bytes a store of registers x, y, z, w writes (the element's size; nothing else is touched). instr "xy" gives
    only x and y (z and w are 0). route "mubuf" (`buffer_store_format_*`, the format from the V#): component c takes
    channel c, except a two-component format, whose second component is channel y when the V#'s DST_SEL is Y in y and a
    constant (0 or 1) in z (the runtime's XY01), channel w otherwise (identity XYZW: `_xy` then writes (x, 0)). Measured
    on the Radeon 880M (gfx1150) for sixteen selector sets and on a real RDNA2, gfx1036 (@AwkGrep256, #2022: XYZW; the
    narrow-integer fixtures: XY01). route "mtbuf" (`tbuffer_store_format_*`): component c takes channel c."""
    widths, nfmt = parse_name(name)
    chan = list(regs[:4]) if instr == "xyzw" else list(regs[:2]) + [0, 0]
    if route == "mubuf" and len(widths) == 2:
        chan = [chan[0], chan[1] if sel[1] == "Y" and sel[2] in "01" else chan[3]]
    big, bit = 0, 0
    for c, w in enumerate(widths):
        big |= encode_field(nfmt, w, chan[c]) << bit
        bit += w
    return big.to_bytes(bit // 8, "little")


def run_stores(oracle, family, fmts, path, rng, instr="xyzw", sel="XYZW"):
    """-> {code: [elements, mismatches, 0]}. A mismatch is a store that differs from the model of its route."""
    from hw_oracle import asm
    jobs, kernels = [], []
    for code, name in fmts:
        hc = hw_code(family, code, name)
        rows = store_inputs(name, rng)
        ref = hc if path == "mubuf" else name
        for lo in range(0, len(rows), CHUNK):
            part = rows[lo:lo + CHUNK]
            uid = len(kernels)
            kernels.append(asm.Kernel(store_lines(uid, family, ref, path, part, instr, sel), 32, len(part), len(part) * 4, raw=True))
            jobs.append((code, name, part))
    results = oracle.run(kernels, words=CHUNK * 4)
    out = {}
    for (code, name, part), res in zip(jobs, results):
        rec = out.setdefault(code, [0, [], 0])
        if isinstance(res, str):
            rec[1].append((b"", b"", res.encode()))
            continue
        size = element_size(name)
        for i, row in enumerate(part):
            got = struct.pack("<4I", *res[4 * i:4 * i + 4])
            want = bytearray(struct.pack("<4I", *([PREFILL] * 4)))
            want[:size] = encode_element(name, row, instr, path, sel)
            rec[0] += 1
            if bytes(want) != got:
                rec[1].append((row, bytes(want), got))
    return out


# ---- the command line ----

GFX10_ONLY = tuple(c for c, n in GFX10_NAMES.items() if n not in GFX11_CODES)    # 30..35, 37..42: no GFX11 spelling


def select_formats(spec, family="gfx11"):
    """The formats to run: the ones in `spec`, else every code the GPU family has (GFX10: all 77, GFX11: the 63 that have a
    GFX11 spelling)."""
    codes = {n: c for c, n in GFX10_NAMES.items()}
    if not spec:
        return [(c, n) for c, n in GFX10_NAMES.items() if family == "gfx10" or n in GFX11_CODES]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        code = int(tok) if tok.isdigit() else codes.get(tok)
        if code not in GFX10_NAMES:
            raise ValueError(f"unknown format {tok!r}; give a GFX10 code (1..77) or name such as 8_8_8_8_UNORM")
        out.append((code, GFX10_NAMES[code]))
    return out


def self_test():
    """The MODEL on values worked out by hand, and round trips; no GPU."""
    problems = []

    def check(cond, what):
        if not cond:
            problems.append(what)

    one11 = 15 << 6
    check(decode_element("8_8_8_8_UNORM", bytes([0, 255, 128, 51]))[:4] == [0, 0x3F800000, frac_to_f32(Fraction(128, 255)),
                                                                          frac_to_f32(Fraction(51, 255))], "unorm8")
    check(decode_element("8_SINT", bytes([0xFF]))[0] == 0xFFFFFFFF, "8_SINT -1")
    check(decode_element("8_SSCALED", bytes([0x80]))[0] == 0xC3000000, "8_SSCALED -128.0")
    check(decode_element("10_11_11_FLOAT", one11.to_bytes(4, "little"))[0] == 0x3F800000, "11 bit float 1.0")
    check(decode_element("10_11_11_FLOAT", (0x7C1).to_bytes(4, "little"))[0] == 0x7F820000, "11 bit float NaN payload")
    check(decode_element("10_11_11_FLOAT", b"\1\0\0\0")[0] == 0x35800000, "11 bit float denormal 2^-20")
    check(decode_element("8_UNORM", b"\x33")[:4] == [frac_to_f32(Fraction(51, 255))] * 4, "missing channels repeat")
    check(decode_element("8_UNORM", b"\x33", "mtbuf")[1:] == [0, 0, 0], "tbuffer: the rest is 0")
    check(encode_element("10_11_11_FLOAT", [0x3F800000, 0, 0, 0])[:2] == one11.to_bytes(2, "little"), "encode 1.0")
    check(encode_field("FLOAT", 11, 0xBF800000) == 0, "negative to an unsigned float is 0")
    check(encode_field("FLOAT", 11, 0x47800000) == 0x7BF, "too large is the largest finite value")
    check(encode_field("SNORM", 2, 0x3F000000) == 1 and encode_field("SNORM", 2, 0xBF000000) == 3, "2 bit SNORM ties away")
    check(encode_field("USCALED", 8, 0x3F800000) == 255, "a SCALED store converts as NORM")
    check(encode_field("SINT", 8, 0x80000000) == 0x80 and encode_field("UINT", 8, 0x1FF) == 0xFF, "integer stores saturate")
    rng = random.Random(1)
    for code, name in GFX10_NAMES.items():             # decode -> encode -> decode is stable for every format
        widths, nfmt = parse_name(name)
        if nfmt in ("USCALED", "SSCALED"):
            continue
        for e in load_patterns(name, rng, 2)[:200]:
            d = decode_element(name, e)
            if nfmt == "FLOAT" and any((x & 0x7F800000) == 0x7F800000 for x in d[:len(widths)]):
                continue                               # inf and NaN do not round trip through RTZ and saturation
            re_ = encode_element(name, d)
            d2 = decode_element(name, re_ + bytes(16 - len(re_)))
            if nfmt == "SNORM":
                continue                               # -1.0 decodes from two codes; stored once as the symmetric one
            check(d2 == d, f"round trip {name} {e[:len(re_)].hex()} -> {re_.hex()}")
            if problems:
                break
    for p in problems:
        print("SELF-TEST FAIL:", p)
    print("self-test:", "FAILED" if problems else "ok")
    return 1 if problems else 0


def negative_control(oracle, family, rng):
    """The hardware must disagree with a kernel whose V# names another format (a model that was never compared
    with real words would pass this too)."""
    fmts = select_formats("8_8_8_8_UNORM,16_16_SNORM,10_11_11_FLOAT,2_10_10_10_UINT,32_32_FLOAT")
    swapped = run_loads(oracle, family, fmts, "mubuf", rng, 8, negative=lambda c, n, ref: 22 if ref != 22 else 21)
    ok = True
    for code, name in fmts:
        cases, bad = swapped[code]
        ok &= len(bad) > 0
        print(f"  wrong V# format for {name}: {len(bad)} of {cases} elements differ ({'caught' if bad else 'NOT CAUGHT'})")
    return 0 if ok else 1


def report(title, results, fmts):
    """Prints a line per format; returns (validated elements, mismatches, known-difference elements)."""
    bad_total = cases_total = known_total = 0
    for code, name in fmts:
        cases, bad, *rest = results[code]
        known = rest[0] if rest else 0
        cases_total += cases - known
        known_total += known
        bad_total += len(bad)
        note = f"  (+{known} known difference)" if known else ""
        print(f"  {title:<14} {code:>3} {name:<22} {cases - known:>7} validated  {'ok' if not bad else f'MISMATCH {len(bad)}'}{note}")
        for first, want, got in bad[:3]:
            shown = first.hex() if isinstance(first, bytes) else [f"{w:08x}" for w in first]
            print("      ", shown, want if isinstance(want, bytes) else [f"{w:08x}" for w in want],
                  got if isinstance(got, bytes) else [f"{w:08x}" if isinstance(w, int) else w for w in got])
    return cases_total, bad_total, known_total


def main():
    ap = argparse.ArgumentParser(description="Buffer data formats on the GPU against the CPU model.")
    ap.add_argument("--oracle", default="", help="hw: run the kernels on the AMD GPU through tools/hw_oracle/")
    ap.add_argument("--mode", choices=("load", "store", "all"), default="all")
    ap.add_argument("--path", choices=("mubuf", "mtbuf", "both"), default="both")
    ap.add_argument("--formats", default="", help="GFX10 codes or names, comma separated (default: every GFX10 code "
                                                   "the GPU family has)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=40, help="random elements per format on top of the sweeps")
    ap.add_argument("--hw-arch", default=None)
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.oracle != "hw":
        ap.error("--oracle hw or --self-test")
    # An explicit --formats needs no GPU to be parsed: a typo is a usage error here, before the oracle is opened
    # (it used to be reported only after it, or masked by "hardware oracle unavailable" without one).
    explicit = None
    if a.formats:
        try:
            explicit = select_formats(a.formats)
        except ValueError as e:
            ap.error(str(e))
    from hw_oracle import runner, asm
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    family = asm.family(oracle.arch)
    fmts = explicit if explicit is not None else select_formats("", family)
    if family == "gfx12":
        print("GFX12 numbers its formats again; not built here")
        oracle.close()
        return 77
    print(f"hardware oracle: {oracle.device} ({oracle.arch}, {family} format numbers)")
    if family == "gfx11":
        print("note: GFX11 is RDNA3 or newer, not RDNA2: a match supports the model and does not prove RDNA2; "
              f"GFX10-only codes {', '.join(map(str, GFX10_ONLY))} cannot be run here")
        fmts = [(c, n) for c, n in fmts if n in GFX11_CODES]
    rng = random.Random(a.seed)
    try:
        if a.negative_control:
            return negative_control(oracle, family, rng)
        paths = ("mubuf", "mtbuf") if a.path == "both" else (a.path,)
        total = bad = known = 0
        for path in paths:
            if a.mode in ("load", "all"):
                c, b_, k = report(f"load {path}", run_loads(oracle, family, fmts, path, rng, a.budget), fmts)
                total, bad, known = total + c, bad + b_, known + k
            if a.mode in ("store", "all"):
                for instr in ("xyzw", "xy"):
                    c, b_, k = report(f"store {instr} {path}", run_stores(oracle, family, fmts, path, rng, instr), fmts)
                    total, bad, known = total + c, bad + b_, known + k
                if path == "mubuf":     # the runtime's selectors for a two-component format: X, Y, 0, 1
                    two = [(c_, n_) for c_, n_ in fmts if len(parse_name(n_)[0]) == 2]
                    for instr in ("xyzw", "xy"):
                        c, b_, k = report(f"store {instr} XY01", run_stores(oracle, family, two, path, rng, instr, "XY01"), two)
                        total, bad, known = total + c, bad + b_, known + k
        print(f"{total} validated elements, {bad} mismatches" + (f"; plus {known} known-difference elements" if known else ""))
        return 1 if bad else 0
    finally:
        oracle.close()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
