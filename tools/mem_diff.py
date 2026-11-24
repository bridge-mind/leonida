r"""Differential fuzzer for the translator's memory instructions (FUZZ-5; standard library only).

Seeded programs made of scalar and buffer memory instructions are run two ways and compared bit-exact:
  * a small reference model in this file, written from the RDNA2 buffer and scalar-memory addressing rules
    (below), which shares no code with the translator;
  * tools/rdna2spirv.py, executed by tests/semantic/spirv_mini_interp.py.

Both the exported words (mrt0) and the final memory contents are compared, so stores are checked too.

Addressing modelled (AMD RDNA2 ISA, buffer resource descriptor and MUBUF/SMEM addressing):
  * V# base   = word0 | (word1 & 0xFFFF) << 32;  stride = word1[29:16].
  * MUBUF     = base + soffset + inst_offset + (offen ? voffset : 0) + (idxen ? vindex * stride : 0),
                with the offsets summed in 32 bits. Swizzled buffers are not generated.
  * s_load    = 64-bit pointer s[p:p+1] + soffset (u32) + imm.
  * s_buffer_load = V# base + soffset + imm.

Every generated access lies inside its buffer (index < num_records for structured buffers, the whole access
below num_records for raw ones) and is dword aligned. Untyped buffer accesses use a word3 with only OOB_SELECT
(STRUCTURED with a stride, RAW without). Out-of-bounds behaviour, the
s_buffer_load alignment truncation and swizzling depend on descriptor fields docs/design.md lists as not modelled,
so they are left to a GPU-confirmed follow-up instead of being guessed here.

Typed buffer loads (buffer_load_format_*, tbuffer_load_format_*) are decoded from the GFX10 format names in
tools/gfx10_formats.json, not from the translator's table. MUBUF applies DST_SEL and repeats missing channels
(*c* mod *n*). MTBUF uses instruction-format channels in order, zero-fills absent channels and returns zero
for an INVALID V# format. UNORM/SNORM use correctly rounded division, and MUBUF DST_SEL 1 is integer 1
for UINT/SINT (SHADER-86, RX 6700).

Usage:
    python -I tools/mem_diff.py [--programs N] [--seed S] [--max-ops N]
    python -I tools/mem_diff.py --self-test

Exit code 0 when every program matches, 1 on a mismatch (the first one is printed with its listing), 2 on a
harness error or when every program was skipped (nothing was compared). Skips are counted with their reasons, never silent.
"""
import argparse
import gc
import os
import random
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import rdna2spirv  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

VSHARP = 0          # s[0:3]: the buffer descriptor
POINTER = 4         # s[4:5]: a 64-bit pointer for s_load
SOFF = 6            # s6: a scalar offset register
SDST = 8            # s[8:15]: scalar load results
VADDR = 2           # v[2:3]: index / offset (v2 alone when only one is used)
VDATA = 4           # v[4:7]: buffer load / store data
EXPORT = (8, 9, 10, 11)
REGION = 0x1000     # bytes of seeded memory per buffer
WORDS = REGION // 4


class Program:
    """One generated case: its assembly lines, the seeded memory and the buffers it may touch."""

    def __init__(self):
        self.lines = []
        self.memory = {}


# ---------------------------------------------------------------- typed-buffer formats (reference decode)

def _load_format_names():
    path = os.path.join(HERE, "gfx10_formats.json")
    import json
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


FORMAT_NAMES = _load_format_names()     # GFX10 FORMAT code -> name (Mesa gfx10-rsrc.json), data only
ONE_F32 = 0x3F800000


def parse_format(name):
    """'GFX10_FORMAT_16_16_SNORM' -> ([16, 16] for X, Y from bit 0, 'SNORM'). The name lists the channel widths
    from the top bit down (2_10_10_10 is X10 Y10 Z10 W2 from bit 0), so the list is reversed. Returns None for
    names this model does not decode (packed floats, scaled, sRGB, block formats)."""
    parts = name.replace("GFX10_FORMAT_", "").split("_")
    kind = parts[-1]
    try:
        widths = [int(p) for p in parts[:-1]]
    except ValueError:
        return None
    if kind not in ("UNORM", "SNORM", "UINT", "SINT", "FLOAT") or not widths:
        return None
    if kind == "FLOAT" and any(w not in (16, 32) for w in widths):
        return None
    if sum(widths) % 8:
        return None
    return list(reversed(widths)), kind


def f32_round(q):
    """Nearest-even binary32 of an exact rational (Fraction), as bits."""
    from fractions import Fraction
    if q == 0:
        return 0
    d = float(q)
    cands = set()
    for scale in (1.0, 1 + 2 ** -26, 1 - 2 ** -26):
        try:
            cands.add(struct.unpack("<I", struct.pack("<f", d * scale))[0])
        except OverflowError:
            pass
    def err(b):
        return abs(Fraction(struct.unpack("<f", struct.pack("<I", b))[0]) - q)
    return min(cands, key=lambda b: (err(b), b & 1))


def f16_to_f32(h):
    """16_FLOAT -> f32 bits. Exact for finite and infinite halves; a NaN shifts its 10-bit payload into the f32
    mantissa and keeps the sign and the quiet bit exactly as the half has them (measured on gfx1036 with
    tools/f16_widen_probe.py: 0x7ddb -> 0x7fbb6000, 0x7fe6 -> 0x7ffcc000). struct's '<e' is not used: before
    Python 3.14 it drops the payload and returns a canonical NaN, from 3.14 it keeps it, and the hardware keeps it."""
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


def decode_element(raw, widths, kind):
    """Decode one element (an int of sum(widths) bits) -> 4 result words; missing channels repeat (*c* mod *n*)."""
    from fractions import Fraction
    out, bit = [], 0
    for w in widths:
        v = (raw >> bit) & ((1 << w) - 1)
        bit += w
        if kind == "UNORM":
            out.append(f32_round(Fraction(v, (1 << w) - 1)))
        elif kind == "SNORM":
            s = v - (1 << w) if v >> (w - 1) else v
            q = max(Fraction(s, (1 << (w - 1)) - 1), Fraction(-1))
            bits_ = f32_round(abs(q))
            out.append(bits_ | (0x80000000 if q < 0 else 0))
        elif kind == "UINT":
            out.append(v)
        elif kind == "SINT":
            out.append((v - (1 << w) if v >> (w - 1) else v) & MASK32)
        elif w == 32:
            out.append(v)
        else:                                           # FLOAT 16 -> f32 bits (exact, NaN payload kept)
            out.append(f16_to_f32(v))
    n = len(out)
    while len(out) < 4:
        out.append(out[len(out) % n])
    return out


# ---------------------------------------------------------------- reference model

class RefMachine:
    """Executes the subset of RDNA2 this fuzzer emits, for one lane."""

    def __init__(self, memory):
        self.mem = memory
        self.s = {}
        self.v = {}

    def word(self, addr):
        if addr & 3:
            raise AssertionError(f"reference model: unaligned dword at {addr:#x}")
        return self.mem.get(addr & MASK64, 0)

    def put(self, addr, value):
        if addr & 3:
            raise AssertionError(f"reference model: unaligned dword at {addr:#x}")
        self.mem[addr & MASK64] = value & MASK32

    def scalar(self, text):
        text = text.strip()
        if text.startswith("s") and text[1:].isdigit():
            return self.s.get(int(text[1:]), 0)
        if text.startswith("v") and text[1:].isdigit():
            return self.v.get(int(text[1:]), 0)
        return int(text, 0) & MASK32

    def run(self, lines):
        for line in lines:
            self.step(line)

    def step(self, line):
        op, _, rest = line.partition(" ")
        args = [a.strip() for a in rest.split(",")] if rest else []
        mods = []
        if args:
            tail = args[-1].split()
            args[-1], mods = tail[0], tail[1:]
        if op == "s_mov_b32":
            self.s[int(args[0][1:])] = self.scalar(args[1])
        elif op == "v_mov_b32":
            src = args[1]
            self.v[int(args[0][1:])] = self.scalar(src)
        elif op.startswith("s_load_dword") or op.startswith("s_buffer_load_dword"):
            n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
            dst = reg_first(args[0])
            base_reg = reg_first(args[1])
            lo, hi = self.s.get(base_reg, 0), self.s.get(base_reg + 1, 0)
            if op.startswith("s_buffer"):
                hi &= 0xFFFF
            addr = (hi << 32 | lo) + self.scalar(args[2])
            for m in mods:
                if m.startswith("offset:"):
                    addr += int(m.split(":", 1)[1], 0)
            for i in range(n):
                self.s[dst + i] = self.word((addr + 4 * i) & MASK64)
        elif op.startswith("buffer_load_dword") or op.startswith("buffer_store_dword"):
            n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
            vdata = reg_first(args[0])
            addr = self._mubuf_addr(args[1], args[2], args[3], mods)
            for i in range(n):
                if op.startswith("buffer_load"):
                    self.v[vdata + i] = self.word(addr + 4 * i)
                else:
                    self.put(addr + 4 * i, self.v.get(vdata + i, 0))
        elif op in BUFFER_SUB:
            addr = self._mubuf_addr(args[1], args[2], args[3], mods)
            self.v[reg_first(args[0])] = self._sub_dword(addr, *BUFFER_SUB[op])
        elif op.startswith(("buffer_load_format_", "tbuffer_load_format_")):
            ncomp = len(op.rsplit("_", 1)[1])
            vdata, rsrc = reg_first(args[0]), reg_first(args[2])
            vaddr = None if args[1] == "off" else reg_first(args[1])
            d0, d1, d3 = self.s.get(rsrc, 0), self.s.get(rsrc + 1, 0), self.s.get(rsrc + 3, 0)
            base, stride = (d1 & 0xFFFF) << 32 | d0, (d1 >> 16) & 0x3FFF
            offset = self.scalar(args[3]) + sum(int(m.split(":", 1)[1], 0) for m in mods if m.startswith("offset:"))
            regs = [] if vaddr is None else [vaddr, vaddr + 1]
            if "idxen" in mods:
                offset += self.v.get(regs.pop(0), 0) * stride
            if "offen" in mods:
                offset += self.v.get(regs.pop(0), 0)
            addr = base + (offset & MASK32)
            fm = next((m for m in mods if m.startswith("format:")), None)
            tbuffer = op.startswith("tbuffer")
            descriptor_format = (d3 >> 12) & 0x7F
            if tbuffer:
                if descriptor_format == 0:
                    for c in range(ncomp):
                        self.v[vdata + c] = 0
                    return
                # LLVM omits the default MTBUF instruction format from its listing.
                fmt = DEFAULT_TBUFFER_FORMAT if fm is None else fm.split("BUF_FMT_")[-1].rstrip("]")
                name = "GFX10_FORMAT_" + fmt
            else:
                name = FORMAT_NAMES[descriptor_format]
            widths, kind = parse_format(name)
            nbytes = sum(widths) // 8
            raw = 0
            for i in range(nbytes):                     # little-endian bytes from dword memory
                a = addr + i
                raw |= ((self.word(a & ~3) >> (8 * (a & 3))) & 0xFF) << (8 * i)
            comps = decode_element(raw, widths, kind)
            if tbuffer:
                for c in range(ncomp):
                    self.v[vdata + c] = comps[c] if c < len(widths) else 0
                return
            one = 1 if kind in ("UINT", "SINT") else ONE_F32
            for c in range(ncomp):
                sel = (d3 >> (3 * c)) & 7
                self.v[vdata + c] = {0: 0, 1: one, 4: comps[0], 5: comps[1], 6: comps[2], 7: comps[3]}.get(sel, 0)
        elif op.startswith("buffer_atomic_"):
            kind = op[len("buffer_atomic_"):]
            vdata = reg_first(args[0])
            addr = self._mubuf_addr(args[1], args[2], args[3], mods)
            old = self._apply_atomic(kind, addr, self.v.get(vdata, 0), self.v.get(vdata + 1, 0))
            if "glc" in mods:
                self.v[vdata] = old
        elif op.startswith("global_atomic_"):
            kind = op[len("global_atomic_"):]
            glc = "glc" in mods
            vaddr, vdata_arg, saddr = (args[1], args[2], args[3]) if glc else (args[0], args[1], args[2])
            addr = self._global_addr(vaddr, saddr, mods)
            vdata = reg_first(vdata_arg)
            old = self._apply_atomic(kind, addr, self.v.get(vdata, 0), self.v.get(vdata + 1, 0))
            if glc:
                self.v[reg_first(args[0])] = old
        elif op.startswith("global_load") or op.startswith("global_store"):
            load = op.startswith("global_load")
            data, vaddr = (reg_first(args[0]), args[1]) if load else (reg_first(args[1]), args[0])
            addr = self._global_addr(vaddr, args[2], mods)
            sub = GLOBAL_SUB_SPEC.get(op)
            if sub:
                self.v[data] = self._sub_dword(addr, *sub)
            else:
                n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
                for i in range(n):
                    if load:
                        self.v[data + i] = self.word(addr + 4 * i)
                    else:
                        self.put(addr + 4 * i, self.v.get(data + i, 0))
        elif op.startswith("s_waitcnt") or op == "exp" or op == "s_endpgm":
            pass
        else:
            raise AssertionError(f"reference model: unexpected op {op}")

    def _mubuf_addr(self, vaddr_arg, rsrc_arg, soff_arg, mods):
        vaddr = None if vaddr_arg == "off" else reg_first(vaddr_arg)
        rsrc = reg_first(rsrc_arg)
        d0, d1 = self.s.get(rsrc, 0), self.s.get(rsrc + 1, 0)
        base, stride = (d1 & 0xFFFF) << 32 | d0, (d1 >> 16) & 0x3FFF
        offset = self.scalar(soff_arg) + sum(int(m.split(":", 1)[1], 0) for m in mods if m.startswith("offset:"))
        regs = [] if vaddr is None else [vaddr, vaddr + 1]
        if "idxen" in mods:
            offset += self.v.get(regs.pop(0), 0) * stride
        if "offen" in mods:
            offset += self.v.get(regs.pop(0), 0)
        return base + (offset & MASK32)

    def _global_addr(self, vaddr, saddr, mods):
        inst = sum(int(m.split(":", 1)[1], 0) for m in mods if m.startswith("offset:"))
        if saddr == "off":
            v0 = reg_first(vaddr)
            addr = self.v.get(v0 + 1, 0) << 32 | self.v.get(v0, 0)
        else:
            s0 = reg_first(saddr)
            addr = (self.s.get(s0 + 1, 0) << 32 | self.s.get(s0, 0)) + self.v.get(reg_first(vaddr), 0)
        return (addr + inst) & MASK64            # the printed offset is already signed (llvm-mc)

    def _sub_dword(self, addr, bits, sign):
        shift = (addr & 3) * 8
        value = (self.word(addr & ~3) >> shift) & ((1 << bits) - 1)
        if sign and value >> (bits - 1):
            value -= 1 << bits
        return value & MASK32

    def _apply_atomic(self, kind, addr, src, cmp_val=0):
        old = self.word(addr)
        s32 = lambda x: x - (1 << 32) if x >> 31 else x  # noqa: E731
        new = {"add": old + src, "sub": old - src, "umin": min(old, src), "umax": max(old, src),
               "smin": old if s32(old) <= s32(src) else src, "smax": old if s32(old) >= s32(src) else src,
               "and": old & src, "or": old | src, "xor": old ^ src, "swap": src,
               "cmpswap": src if old == cmp_val else old}[kind]
        self.put(addr, new)
        return old


def reg_first(text):
    """'v4' / 's[8:11]' / 'v[2:3]' -> 4 / 8 / 2."""
    text = text.strip()
    if "[" in text:
        return int(text[2:].split(":")[0])
    return int(text[1:])


def reg_range(kind, first, n):
    return f"{kind}{first}" if n == 1 else f"{kind}[{first}:{first + n - 1}]"


# ---------------------------------------------------------------- program generator

def _seed_region(rng, memory, base):
    for i in range(WORDS):
        memory[base + 4 * i] = rng.getrandbits(32)


def _buffer_base(rng):
    """A 48-bit, region-aligned base address. The high 16 bits exercise V# word1[15:0]."""
    hi = rng.choice((0, 0, 1, 0x7F, 0xFFFF))
    lo = rng.randrange(1, 0x10000) * REGION
    return (hi << 32 | lo) & ((1 << 48) - 1)


def _soffset(rng, prog, budget):
    """A scalar offset operand: an inline constant or s6 seeded with a dword-aligned value below budget."""
    value = rng.randrange(0, max(1, budget // 4)) * 4
    if value <= 64 and rng.random() < 0.5:
        return str(value), value
    prog.lines.append(f"s_mov_b32 s{SOFF}, {value:#x}")
    return f"s{SOFF}", value


def _emit_vsharp(rng, prog, base, stride, num_records):
    word1 = (base >> 32) & 0xFFFF | (stride & 0x3FFF) << 16
    # word3: only OOB_SELECT (bits 29:28), the bounds check the accesses stay inside of: STRUCTURED for a stride,
    # RAW without (the two RADV uses); every generated access is in range under either
    for i, w in enumerate((base & MASK32, word1, num_records, OOB_STRUCTURED if stride else OOB_RAW)):
        prog.lines.append(f"s_mov_b32 s{VSHARP + i}, {w:#x}")


def gen_s_load(rng, prog):
    base = _buffer_base(rng)
    _seed_region(rng, prog.memory, base)
    n = rng.choice((1, 2, 4, 8))
    prog.lines.append(f"s_mov_b32 s{POINTER}, {base & MASK32:#x}")
    prog.lines.append(f"s_mov_b32 s{POINTER + 1}, {base >> 32:#x}")
    room = REGION - 4 * n
    form = rng.choice(("imm", "sgpr", "both"))
    if form == "imm":
        off = rng.randrange(0, room // 4 + 1) * 4
        operand = f"{off:#x}"
    elif form == "sgpr":
        off = rng.randrange(0, room // 4 + 1) * 4
        prog.lines.append(f"s_mov_b32 s{SOFF}, {off:#x}")
        operand = f"s{SOFF}"
    else:
        a = rng.randrange(0, room // 8 + 1) * 4
        b = rng.randrange(0, (room - a) // 4 + 1) * 4
        prog.lines.append(f"s_mov_b32 s{SOFF}, {a:#x}")
        operand = f"s{SOFF} offset:{b:#x}"
    prog.lines.append(f"s_load_dword{'' if n == 1 else 'x' + str(n)} {reg_range('s', SDST, n)}, "
                      f"s[{POINTER}:{POINTER + 1}], {operand}")
    prog.lines.append("s_waitcnt lgkmcnt(0)")
    return [("s", SDST + i) for i in range(n)]


def gen_s_buffer_load(rng, prog):
    base = _buffer_base(rng)
    _seed_region(rng, prog.memory, base)
    n = rng.choice((1, 2, 4, 8))
    _emit_vsharp(rng, prog, base, 0, REGION)
    room = REGION - 4 * n
    if rng.random() < 0.5:
        off = rng.randrange(0, room // 4 + 1) * 4
        operand = f"{off:#x}"
    else:
        off = rng.randrange(0, room // 4 + 1) * 4
        prog.lines.append(f"s_mov_b32 s{SOFF}, {off:#x}")
        operand = f"s{SOFF}"
    prog.lines.append(f"s_buffer_load_dword{'' if n == 1 else 'x' + str(n)} {reg_range('s', SDST, n)}, "
                      f"s[{VSHARP}:{VSHARP + 3}], {operand}")
    prog.lines.append("s_waitcnt lgkmcnt(0)")
    return [("s", SDST + i) for i in range(n)]


def _mubuf_operands(rng, prog, n):
    """Pick a raw or structured layout and an in-bounds address. Returns the operand text after vdata."""
    base = _buffer_base(rng)
    _seed_region(rng, prog.memory, base)
    size = 4 * n
    structured = rng.random() < 0.5
    inst = rng.randrange(0, 16) * 4
    if structured:
        stride = rng.choice((4, 8, 12, 16, 32, 48, 64, 256))
        stride = max(stride, size)            # the element holds the whole access
        records = REGION // stride
        index = rng.randrange(0, records)
        inst = min(inst, stride - size)
        inst -= inst % 4
        soff_text, soff = _soffset(rng, prog, stride - size - inst + 4)
        within = soff + inst
        voff = None
        if within + size <= stride and rng.random() < 0.5:
            voff = rng.randrange(0, (stride - size - within) // 4 + 1) * 4
        _emit_vsharp(rng, prog, base, stride, records)
        prog.lines.append(f"v_mov_b32 v{VADDR}, {index:#x}")
        if voff is not None:
            prog.lines.append(f"v_mov_b32 v{VADDR + 1}, {voff:#x}")
            return f"v[{VADDR}:{VADDR + 1}], s[{VSHARP}:{VSHARP + 3}], {soff_text} idxen offen offset:{inst}"
        return f"v{VADDR}, s[{VSHARP}:{VSHARP + 3}], {soff_text} idxen offset:{inst}"
    soff_text, soff = _soffset(rng, prog, REGION // 4)
    room = REGION - size - soff - inst
    voff = rng.randrange(0, room // 4 + 1) * 4
    _emit_vsharp(rng, prog, base, 0, REGION)
    if rng.random() < 0.8:
        prog.lines.append(f"v_mov_b32 v{VADDR}, {voff:#x}")
        return f"v{VADDR}, s[{VSHARP}:{VSHARP + 3}], {soff_text} offen offset:{inst}"
    return f"off, s[{VSHARP}:{VSHARP + 3}], {soff_text} offset:{inst}"


def gen_buffer_load(rng, prog):
    n = rng.choice((1, 2, 3, 4))
    operands = _mubuf_operands(rng, prog, n)
    prog.lines.append(f"buffer_load_dword{'' if n == 1 else 'x' + str(n)} {reg_range('v', VDATA, n)}, {operands}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA + i) for i in range(n)]


def gen_buffer_store(rng, prog):
    n = rng.choice((1, 2, 3, 4))
    for i in range(n):
        prog.lines.append(f"v_mov_b32 v{VDATA + i}, {rng.getrandbits(32):#x}")
    operands = _mubuf_operands(rng, prog, n)
    prog.lines.append(f"buffer_store_dword{'' if n == 1 else 'x' + str(n)} {reg_range('v', VDATA, n)}, {operands}")
    # Read the same place back so the export sees the stored words too; memory is compared as a whole anyway.
    prog.lines.append(f"buffer_load_dword{'' if n == 1 else 'x' + str(n)} {reg_range('v', VDATA, n)}, {operands}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA + i) for i in range(n)]


BUFFER_SUB = {"buffer_load_ubyte": (8, False), "buffer_load_sbyte": (8, True),
              "buffer_load_ushort": (16, False), "buffer_load_sshort": (16, True)}


def gen_buffer_sub(rng, prog):
    name = rng.choice(tuple(BUFFER_SUB))
    size = 1 if "byte" in name else 2
    operands = _mubuf_operands(rng, prog, 1)
    head, inst_str = operands.rsplit("offset:", 1)
    inst = int(inst_str, 0) + rng.randrange(0, 4 // size) * size
    prog.lines.append(f"{name} v{VDATA}, {head}offset:{inst}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA)]


ATOMICS = ("add", "sub", "umin", "umax", "smin", "smax", "and", "or", "xor", "swap", "cmpswap")


def gen_buffer_atomic(rng, prog):
    kind = rng.choice(ATOMICS)
    glc = rng.random() < 0.7
    prog.lines.append(f"v_mov_b32 v{VDATA}, {rng.getrandbits(32):#x}")
    operands = _mubuf_operands(rng, prog, 1)
    if kind == "cmpswap":
        # Half the time compare against the word that is really there, so the swap happens.
        prog.lines.append(f"buffer_load_dword v{VDATA + 1}, {operands}" if rng.random() < 0.5
                          else f"v_mov_b32 v{VDATA + 1}, {rng.getrandbits(32):#x}")
        prog.lines.append("s_waitcnt vmcnt(0)")
        vdata = f"v[{VDATA}:{VDATA + 1}]"
    else:
        vdata = f"v{VDATA}"
    prog.lines.append(f"buffer_atomic_{kind} {vdata}, {operands}{' glc' if glc else ''}")
    prog.lines.append(f"buffer_load_dword v{VDATA + 2}, {operands}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA), ("v", VDATA + 2)]


GLOBAL_SUB_SPEC = {"global_load_ubyte": (8, False), "global_load_sbyte": (8, True),
                   "global_load_ushort": (16, False), "global_load_sshort": (16, True)}
GLOBAL_SUB = tuple(GLOBAL_SUB_SPEC)


def _global_addr_operands(rng, prog, width, align, sub_size=0):
    """Emit a 64-bit VGPR address or SGPR base + 32-bit VGPR offset reaching an in-bounds address."""
    base = _buffer_base(rng) + REGION            # room below the base for negative offsets
    _seed_region(rng, prog.memory, base - REGION)
    _seed_region(rng, prog.memory, base)
    inst = rng.randrange(-2048 // 4, 2048 // 4) * 4
    if sub_size:
        inst += rng.randrange(0, 4 // sub_size) * sub_size
    target = rng.randrange(-REGION // align, (REGION - width) // align + 1) * align
    voff = target - inst
    if voff >= 0 and rng.random() < 0.5:
        prog.lines += [f"s_mov_b32 s{POINTER}, {base & MASK32:#x}", f"s_mov_b32 s{POINTER + 1}, {base >> 32:#x}",
                       f"v_mov_b32 v{VADDR}, {voff:#x}"]
        vaddr, saddr = f"v{VADDR}", f"s[{POINTER}:{POINTER + 1}]"
    else:
        a = base + voff                              # 64-bit VGPR address
        prog.lines += [f"v_mov_b32 v{VADDR}, {a & MASK32:#x}", f"v_mov_b32 v{VADDR + 1}, {a >> 32:#x}"]
        vaddr, saddr = f"v[{VADDR}:{VADDR + 1}]", "off"
    off = f" offset:{inst}" if inst else ""
    return vaddr, saddr, off


def gen_global(rng, prog):
    """global_load/store with a 64-bit VGPR address or an SGPR base + 32-bit VGPR offset, and a signed 12-bit
    immediate printed the way llvm-mc prints it for gfx10 (decimal, negative values with a minus sign)."""
    op = rng.choice(("load", "load", "store", "sub"))
    n = rng.choice((1, 2, 3, 4)) if op != "sub" else 1
    width = 4 * n
    sub_size = 0
    if op == "sub":
        name = rng.choice(GLOBAL_SUB)
        sub_size = 1 if "byte" in name else 2
        width = sub_size
    vaddr, saddr, off = _global_addr_operands(rng, prog, width, 4 if op != "sub" else width, sub_size)
    if op == "store":
        for i in range(n):
            prog.lines.append(f"v_mov_b32 v{VDATA + i}, {rng.getrandbits(32):#x}")
        suffix = "" if n == 1 else f"x{n}"
        prog.lines.append(f"global_store_dword{suffix} {vaddr}, {reg_range('v', VDATA, n)}, {saddr}{off}")
        prog.lines.append(f"global_load_dword{suffix} {reg_range('v', VDATA, n)}, {vaddr}, {saddr}{off}")
    elif op == "sub":
        prog.lines.append(f"{name} v{VDATA}, {vaddr}, {saddr}{off}")
    else:
        suffix = "" if n == 1 else f"x{n}"
        prog.lines.append(f"global_load_dword{suffix} {reg_range('v', VDATA, n)}, {vaddr}, {saddr}{off}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA + i) for i in range(n)]


def gen_global_atomic(rng, prog):
    kind = rng.choice(ATOMICS)
    glc = rng.random() < 0.7
    prog.lines.append(f"v_mov_b32 v{VDATA}, {rng.getrandbits(32):#x}")
    vaddr, saddr, off = _global_addr_operands(rng, prog, 4, 4)
    if kind == "cmpswap":
        prog.lines.append(f"global_load_dword v{VDATA + 1}, {vaddr}, {saddr}{off}" if rng.random() < 0.5
                          else f"v_mov_b32 v{VDATA + 1}, {rng.getrandbits(32):#x}")
        prog.lines.append("s_waitcnt vmcnt(0)")
        vdata = f"v[{VDATA}:{VDATA + 1}]"
    else:
        vdata = f"v{VDATA}"
    vdst = f"v{VDATA + 2}, " if glc else ""
    prog.lines.append(f"global_atomic_{kind} {vdst}{vaddr}, {vdata}, {saddr}{off}{' glc' if glc else ''}")
    prog.lines.append(f"global_load_dword v{VDATA + 3}, {vaddr}, {saddr}{off}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA + 2 if glc else VDATA), ("v", VDATA + 3)]


DEFAULT_TBUFFER_FORMAT = "8_UNORM"
# Exercise the implicit LLVM spelling alongside the five MTBUF formats found in the 1.0.2.1 corpus.
TBUFFER_FORMATS = ("16_16_SNORM", "8_8_8_8_UNORM", "8_8_8_8_UINT", "2_10_10_10_UNORM", "16_16_16_16_UINT",
                   DEFAULT_TBUFFER_FORMAT)
XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9
OOB_STRUCTURED, OOB_RAW = 1 << 28, 3 << 28       # V# word3 OOB_SELECT


def _typed_codes():
    """Formats buffer_load_format is generated with: the ones the translator claims to decode (its scope), each
    checked here against the reference decode of its GFX10 name."""
    return sorted(c for c in rdna2spirv.Gen.FORMATS if parse_format(FORMAT_NAMES[c]))


def gen_buffer_format(rng, prog):
    ncomp = rng.choice((1, 2, 3, 4))
    suffix = "xyzw"[:ncomp]
    tbuffer = rng.random() < 0.5
    if tbuffer:
        fname = rng.choice(TBUFFER_FORMATS)
        widths, kind = parse_format("GFX10_FORMAT_" + fname)
        word3 = (20 << 12) | XYZW                      # Bound 32_UINT V#; the instruction selects the format.
    else:
        code = rng.choice(_typed_codes())
        widths, kind = parse_format(FORMAT_NAMES[code])
        sels = [rng.choice((0, 1, 4, 5, 6, 7)) for _ in range(4)]
        word3 = code << 12 | sels[0] | sels[1] << 3 | sels[2] << 6 | sels[3] << 9
    size = sum(widths) // 8
    base = _buffer_base(rng)
    _seed_region(rng, prog.memory, base)
    structured = rng.random() < 0.5
    if structured:
        stride = size * rng.choice((1, 2, 3, 4))
        stride += -stride % 4
        records = REGION // stride
        index = rng.randrange(0, records)
        _emit_vsharp(rng, prog, base, stride, records)
        prog.lines[-1] = f"s_mov_b32 s{VSHARP + 3}, {word3 | OOB_STRUCTURED:#x}"
        prog.lines.append(f"v_mov_b32 v{VADDR}, {index:#x}")
        operands = f"v{VADDR}, s[{VSHARP}:{VSHARP + 3}], 0"
        addr_mods = "idxen"
    else:
        voff = rng.randrange(0, (REGION - 16) // size) * size
        _emit_vsharp(rng, prog, base, 0, REGION)
        prog.lines[-1] = f"s_mov_b32 s{VSHARP + 3}, {word3 | OOB_RAW:#x}"
        prog.lines.append(f"v_mov_b32 v{VADDR}, {voff:#x}")
        operands = f"v{VADDR}, s[{VSHARP}:{VSHARP + 3}], 0"
        addr_mods = "offen"
    vdata = reg_range("v", VDATA, ncomp)
    if tbuffer:
        format_mod = "" if fname == DEFAULT_TBUFFER_FORMAT else f" format:[BUF_FMT_{fname}]"
        prog.lines.append(f"tbuffer_load_format_{suffix} {vdata}, {operands}{format_mod} {addr_mods}")
    else:
        prog.lines.append(f"buffer_load_format_{suffix} {vdata}, {operands} {addr_mods}")
    prog.lines.append("s_waitcnt vmcnt(0)")
    return [("v", VDATA + c) for c in range(ncomp)]


GENERATORS = {
    "s_load": gen_s_load,
    "s_buffer_load": gen_s_buffer_load,
    "buffer_load": gen_buffer_load,
    "buffer_sub": gen_buffer_sub,
    "buffer_store": gen_buffer_store,
    "buffer_atomic": gen_buffer_atomic,
    "global": gen_global,
    "global_atomic": gen_global_atomic,
    "buffer_format": gen_buffer_format,
}


def generate_program(rng, n_ops, families=None):
    """A program of n_ops memory operations; its last four result registers are exported through v8..v11."""
    prog = Program()
    names = sorted(families or GENERATORS)
    results = []
    for _ in range(n_ops):
        family = rng.choice(names)
        prog.lines.append(f"; {family}")
        results = GENERATORS[family](rng, prog) or results
    picks = [results[min(i, len(results) - 1)] for i in range(4)] if results else []
    for dst, (kind, reg) in zip(EXPORT, picks):
        if not (kind == "v" and reg == dst):
            prog.lines.append(f"v_mov_b32 v{dst}, {kind}{reg}")
    return prog


# ---------------------------------------------------------------- running both sides

def _listing(lines):
    body = [line for line in lines if not line.startswith(";")]
    return ("; mem_diff case  code 0x0..0x1000\n" + "".join("\t" + line + "\n" for line in body)
            + f"\texp mrt0 {', '.join('v%d' % r for r in EXPORT)} done vm\n\ts_endpgm\n")


def run_one(name, prog):
    """Returns (status, detail): status is match, mismatch or skip."""
    ref = RefMachine(dict(prog.memory))
    ref.run([line for line in prog.lines if not line.startswith(";")])
    want = [ref.v.get(r, 0) for r in EXPORT]
    with tempfile.TemporaryDirectory(prefix="mem-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_listing(prog.lines))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spvasm = rdna2spirv.translate(path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return "skip", f"translator unsupported: {e}"
    memory = dict(prog.memory)
    try:
        out = run_fragment(spvasm, memory=memory)
    except Exception as e:  # noqa: BLE001 - the interpreter is half of the comparison
        return "skip", f"mini_interp unsupported: {type(e).__name__}: {e}"
    got = list(out.get("%out_mrt0", ()))
    if got != want:
        bad = [k for k in range(4) if k >= len(got) or got[k] != want[k]]
        i = bad[0]
        g = f"{got[i]:#010x}" if i < len(got) else "missing"
        detail = f"export word {i}: expected {want[i]:#010x}, translated {g}"
        return "mismatch", detail
    if memory != ref.mem:
        bad = sorted(a for a in set(memory) | set(ref.mem) if memory.get(a, 0) != ref.mem.get(a, 0))
        a = bad[0]
        return "mismatch", (f"memory at {a:#x} ({len(bad)} word(s) differ): expected {ref.mem.get(a, 0):#010x}, "
                            f"translated {memory.get(a, 0):#010x}")
    return "match", ""


def run_suite(programs, seed, max_ops, families=None):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "harness-error": 0}
    first = None
    rng = random.Random(seed)
    for i in range(programs):
        prog = generate_program(rng, rng.randrange(1, max_ops + 1), families)
        name = f"ps_mem_{i:05d}"
        try:
            status, detail = run_one(name, prog)
        except Exception as e:  # noqa: BLE001 - harness bugs must be loud
            counts["harness-error"] += 1
            first = first or f"{name}: harness error {type(e).__name__}: {e}"
            continue
        if status == "match":
            counts["pass"] += 1
        elif status == "mismatch":
            counts["mismatch"] += 1
            if first is None:
                first = f"{name} (seed {seed} program {i}): {detail}\n" + "\n".join("\t" + x for x in prog.lines)
        else:
            counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
    return counts, first


def self_test():
    """Canaries: a hand-written case matches, and a corrupted reference is caught."""
    prog = Program()
    prog.memory = {0x2000: 0x11111111, 0x2004: 0x22222222, 0x2008: 0x33333333, 0x200C: 0x44444444}
    prog.lines = ["s_mov_b32 s0, 0x2000", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x10", "s_mov_b32 s3, 0x30000000",
                  "v_mov_b32 v2, 4", "buffer_load_dwordx3 v[8:10], v2, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
                  "v_mov_b32 v11, 7"]
    status, detail = run_one("ps_mem_canary", prog)
    if status != "match":
        print(f"canary FAILED: hand-written case gave {status}: {detail}")
        return False
    ref = RefMachine(dict(prog.memory))
    ref.run(prog.lines)
    if [ref.v.get(r) for r in EXPORT] != [0x22222222, 0x33333333, 0x44444444, 7]:
        print("canary FAILED: the reference model read the wrong words")
        return False
    real_word = RefMachine.word
    RefMachine.word = lambda self, addr: real_word(self, addr) ^ 1      # a reference that is off by one bit
    try:
        status, _ = run_one("ps_mem_canary_bad", prog)
    finally:
        RefMachine.word = real_word
    if status != "mismatch":
        print("canary FAILED: a corrupted reference was not caught")
        return False
    # The f16 reference keeps a NaN payload and its quiet bit, as gfx1036 does (tools/f16_widen_probe.py).
    if decode_element(0x7DDB, [16], "FLOAT")[0] != 0x7FBB6000:
        print("canary FAILED: the f16 NaN payload was not kept")
        return False
    print("canaries: match ok, mismatch detected")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Differential fuzzer for memory instructions (FUZZ-5).")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=3)
    ap.add_argument("--family", action="append", choices=sorted(GENERATORS), help="restrict to one family (repeatable)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    if a.self_test:
        return 0 if self_test() else 1
    counts, first = run_suite(a.programs, a.seed, a.max_ops, a.family)
    skips = sum(counts["skip"].values())
    print(f"{a.programs} programs (seed {a.seed}): {counts['pass']} pass, {counts['mismatch']} mismatch, "
          f"{skips} skipped, {counts['harness-error']} harness errors")
    for reason, n in sorted(counts["skip"].items()):
        print(f"  skip x{n}: {reason}")
    if first:
        print("FIRST FAILURE:\n" + first)
    if counts["harness-error"]:
        return 2
    if counts["mismatch"]:
        return 1
    if not counts["pass"]:
        print("no program ran: every one was skipped")
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
