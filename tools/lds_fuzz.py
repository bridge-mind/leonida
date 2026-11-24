r"""Differential fuzzer for the translator's LDS and ds_* instructions (FUZZ-6), standard library only.

Generates seeded random compute programs (one 64-thread group, wave64 or wave32, 4 KiB LDS) that run a sequence
of ds_* instructions and compares every thread's result registers and the final LDS contents, bit for bit, with an
independent model of the RDNA2 instructions written here (it does not import the translator). Two oracles:

    translator   (default) translate with rdna2spirv.py and run on the GPU with vk_compute_test
    hw           run the program natively on an AMD GPU (tools/hw_oracle/); tests the MODEL, wave32 and wave64

    python -I tools/lds_fuzz.py --programs 200 --seed 1 [--wave 64|32|both] [--emulate] [--build runtime/build]
    python -I tools/lds_fuzz.py --oracle hw --programs 2000 --seed 1       # the model against the AMD GPU; 77 if
                                                                           # the hardware oracle's tools are missing
    python -I tools/lds_fuzz.py --oracle translator,hw ...                 # both, and what the translator did
    python -I tools/lds_fuzz.py --show 1:7        # print program 7 of seed 1 and the model's result
    python -I tools/lds_fuzz.py --model-only ...  # generate and run the model only (no GPU)

Programs are deterministic: every op either touches addresses no other lane touches in the same op, or is an
order-independent atomic without return (add/sub/min/max/and/or/xor). ds_swizzle runs with all
lanes active (the ISA behaviour of a source lane that is off in EXEC isn't covered here, see --swizzle-exec);
ds_bpermute also runs under random EXEC masks (an inactive source lane reads 0). Swizzle offsets are
limited to the two patterns the LLVM swizzle() macros encode: bit 15 set = quad permute (4 x 2-bit selectors in
bits 7:0), clear = bit mask (and = bits 4:0, or = bits 9:5, xor = bits 14:10, within each 32-lane group).

Options that widen the op mix (each changes the programs a seed generates, and their names; without them a seed
gives the same programs as before): --permute adds ds_permute_b32, --swizzle-exec runs ds_swizzle under EXEC masks,
--threads N or mix sets the group size (32, 64, 128, 256; a partial wave or several waves, joined by s_barrier),
--extended is all three. ds_permute_b32 and ds_swizzle under EXEC carry a hypothesis (the model's reading of the ISA
text, which nothing measured before the hardware oracle): a mismatch there is a falsified claim on the hardware and
a "hypothesis" line apart from the others on the translator.

Each mismatch is printed with the words that differ and the op that last wrote them, and a --show line that
prints the program. Exit code: 0 when every program that ran matches, 1 on a mismatch, 2 on a harness error
(assembler, translator or vk_compute_test failure) or when no program could run (every one skipped), 77 when
--oracle hw was asked for and the hardware oracle's tools or GPU are missing.

A wave32 program is skipped where vk_compute_test can't pin 32-lane subgroups ("compute subgroup size: 32 not
available", e.g. Mesa llvmpipe, whose subgroups are 8 wide): its words would come from another wave size. Wave64
programs translated with --emulate use no subgroup operations, so they are compared at any subgroup size.
Usage: `python -I tools/lds_fuzz.py --help` lists the options.
"""
import argparse
import copy
import json
import os
import random
import re
import struct
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402
from compute_fixture import compute_header, find_test_program, unpinned_subgroup  # noqa: E402

THREADS = 64                          # the default group size
GROUP_SIZES = (32, 64, 128, 256)      # --threads mix
LDS_BYTES = 4096
LDS_DWORDS = LDS_BYTES // 4
POOL = list(range(20, 36))            # result / data registers stored at the end
ADDR = 6                              # per-op address register
TMP = 7
RSRC2 = 0x8088 & ~(0x1FF << 15) | ((LDS_BYTES // 512) << 15)
OUT_WORDS = THREADS * len(POOL) + LDS_DWORDS
M32 = 0xFFFFFFFF
HW_VGPRS = 40                         # the programs use v0..v35
EXEC_SAVE = 10                        # s10 (s[10:11] in wave64) holds the entry EXEC in the native kernels

READS = {   # name: (bytes, kind)
    "ds_read_b32": (4, "b32"), "ds_read_b64": (8, "wide"), "ds_read_b96": (12, "wide"),
    "ds_read_b128": (16, "wide"),
    "ds_read_u8": (1, "u"), "ds_read_i8": (1, "i"), "ds_read_u16": (2, "u"), "ds_read_i16": (2, "i"),
    "ds_read_u16_d16": (2, "u_lo"), "ds_read_u16_d16_hi": (2, "u_hi"),
    "ds_read_u8_d16": (1, "u_lo"), "ds_read_u8_d16_hi": (1, "u_hi"),
    "ds_read_i8_d16": (1, "i_lo"), "ds_read_i8_d16_hi": (1, "i_hi"),
}
WRITES = {  # name: (bytes, data bit offset)
    "ds_write_b32": (4, 0), "ds_write_b64": (8, 0), "ds_write_b96": (12, 0), "ds_write_b128": (16, 0),
    "ds_write_b8": (1, 0), "ds_write_b16": (2, 0), "ds_write_b8_d16_hi": (1, 16), "ds_write_b16_d16_hi": (2, 16),
}
SHARED_ATOMICS = ("add", "sub", "min_i32", "max_i32", "min_u32", "max_u32", "and", "or", "xor")
RTN_ATOMICS = SHARED_ATOMICS
EXEC_KINDS = ("atomic", "atomic_rtn", "bpermute", "cmpst", "permute", "read", "read2", "swizzle", "write",
              "write2", "wrxchg", "wrxchg2")
ATOMIC_NAME = {"add": "ds_add%s_u32", "sub": "ds_sub%s_u32", "min_i32": "ds_min%s_i32", "max_i32": "ds_max%s_i32",
               "min_u32": "ds_min%s_u32", "max_u32": "ds_max%s_u32", "and": "ds_and%s_b32", "or": "ds_or%s_b32",
               "xor": "ds_xor%s_b32"}

# What the model gets wrong on purpose (--perturb, --negative-control): each name switches one deliberate error
# into model_op / atomic / swizzle_src, to prove the comparison can fail. name: (op family it changes, description)
PERTURBATIONS = {
    "max_i32_unsigned": ("max_i32", "ds_max_i32 compares as unsigned"),
    "min_u32_signed": ("min_u32", "ds_min_u32 compares as signed"),
    "cmpst_swap": ("cmpst", "ds_cmpst stores data0 and compares with data1 (operands swapped)"),
    "i8_zero_extend": ("read_i8", "ds_read_i8 zero-extends"),
    "i16_zero_extend": ("read_i16", "ds_read_i16 zero-extends"),
    "d16_hi_zeroes_low": ("read_d16_hi", "the d16_hi reads clear the low half instead of keeping it"),
    "u8_d16_clobbers_hi": ("read_d16_lo", "the d16 (low half) reads clear the high half instead of keeping it"),
    "b16_hi_shift_0": ("write_d16_hi", "ds_write_b16_d16_hi stores the low half"),
    "write2_swapped": ("write2", "ds_write2 stores its data registers swapped"),
    "st64_unit_4": ("read2", "the st64 forms scale their offsets by 4, not 256"),
    "wrxchg2_swapped": ("wrxchg2", "ds_wrxchg2 stores data1 in its first element"),
    "bpermute_inactive_reads": ("bpermute", "ds_bpermute reads an inactive source lane's register, not 0"),
    "bpermute_wave64_cross": ("bpermute", "a wave64 ds_bpermute reaches all 64 lanes"),
    "swizzle_quad_reversed": ("swizzle", "the quad-permute selectors are taken in reverse lane order"),
    "swizzle_or_xor_swapped": ("swizzle", "the bit-mask or and xor fields are swapped"),
    "permute_no_zero": ("permute", "ds_permute_b32 leaves an unwritten lane's register alone instead of writing 0"),
    "permute_lowest_wins": ("permute", "ds_permute_b32 lets the lowest active source lane win a collision"),
}
PERTURB = set()


class perturbed:
    """Context manager: the model with the named deliberate errors switched on."""

    def __init__(self, *names):
        self.names = set(names)
        unknown = self.names - set(PERTURBATIONS)
        if unknown:
            raise ValueError(f"unknown perturbation {sorted(unknown)}")

    def __enter__(self):
        self.old = set(PERTURB)
        PERTURB.update(self.names)
        return self

    def __exit__(self, *exc):
        PERTURB.clear()
        PERTURB.update(self.old)


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def atomic(op, old, data):
    if op == "add":
        return (old + data) & M32
    if op == "sub":
        return (old - data) & M32
    if op == "min_i32":
        return old if s32(old) <= s32(data) else data
    if op == "max_i32":
        if "max_i32_unsigned" in PERTURB:
            return max(old, data)
        return old if s32(old) >= s32(data) else data
    if op == "min_u32":
        if "min_u32_signed" in PERTURB:
            return old if s32(old) <= s32(data) else data
        return min(old, data)
    if op == "max_u32":
        return max(old, data)
    if op == "and":
        return old & data
    if op == "or":
        return old | data
    if op == "xor":
        return old ^ data
    raise ValueError(op)


def swizzle_src(lane, offset):
    """Source lane for ds_swizzle_b32 (lane = index in the wave)."""
    if offset & 0x8000:
        sh = 2 * (lane & 3)
        if "swizzle_quad_reversed" in PERTURB:
            sh = 6 - sh
        sel = (offset >> sh) & 3
        return (lane & ~3) | sel
    and_m, or_m, xor_m = offset & 31, (offset >> 5) & 31, (offset >> 10) & 31
    if "swizzle_or_xor_swapped" in PERTURB:
        or_m, xor_m = xor_m, or_m
    return (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m)


def op_family(op):
    """The name of the op family a generated op belongs to (the per-family tables and the perturbations use it)."""
    k = op["kind"]
    if k == "read":
        name = op["name"]
        if "d16" in name:
            return "read_d16_hi" if name.endswith("_hi") else "read_d16_lo"
        if name in ("ds_read_u8", "ds_read_i8", "ds_read_u16", "ds_read_i16"):
            return "read_" + name.split("_")[-1]
        return "read_b32" if name == "ds_read_b32" else "read_wide"
    if k == "write":
        name = op["name"]
        if name.endswith("d16_hi"):
            return "write_d16_hi"
        return "write_small" if name in ("ds_write_b8", "ds_write_b16") else "write_b32" if name == "ds_write_b32" \
            else "write_wide"
    if k == "atomic":
        return "atomic_" + op["op"]
    if k == "atomic_rtn":
        return "atomic_rtn_" + op["op"]
    return k


class Program:
    """A generated program: a list of ops, the wave size and the initial values."""

    def __init__(self, seed, index, wave, no_exec=(), threads=THREADS, permute=False, swizzle_exec=False):
        """threads: the group size (a multiple of 32 that divides 1024), or "mix" to draw one from GROUP_SIZES. permute
        adds ds_permute_b32 and swizzle_exec runs ds_swizzle under EXEC masks. With the defaults the program is the
        one earlier versions generated for the seed; each option changes the draw sequence, so it changes the name."""
        extra = (f"t{threads}" if threads != THREADS else "") + ("p" if permute else "") + \
                ("x" if swizzle_exec else "")
        self.rng = random.Random(f"{seed}:{index}:{wave}" + (f":{extra}" if extra else ""))
        self.name = f"cs_ldsfuzz_s{seed}_{index}_w{wave}" + (f"_{extra}" if extra else "")
        self.wave = wave
        rng = self.rng
        if threads == "mix":
            threads = rng.choice(GROUP_SIZES)
        if not isinstance(threads, int) or threads <= 0 or threads % 32 or LDS_DWORDS % threads:
            raise ValueError(f"group size {threads!r}: a multiple of 32 that divides {LDS_DWORDS}")
        self.threads = threads
        self.out_words = threads * len(POOL) + LDS_DWORDS
        self.permute, self.swizzle_exec = permute, swizzle_exec
        self.lds_xor = rng.getrandbits(32)
        self.lds_mul = rng.getrandbits(32) | 1
        self.pool_init = {r: (rng.randrange(1, 1 << 24), rng.getrandbits(32)) for r in POOL}
        self.ops = [self.random_op() for _ in range(rng.randrange(4, 13))]
        for op in self.ops:      # the mask is still drawn, so the rest of the program is the same
            if op["kind"] in no_exec:
                op["exec"] = None
                op.pop("hyp", None)

    def with_ops(self, ops):
        """A copy of the program running only these ops."""
        c = copy.copy(self)
        c.ops = list(ops)
        return c

    @property
    def hyp_ops(self):
        return {i for i, op in enumerate(self.ops) if op.get("hyp")}

    @property
    def partial_wave(self):
        return self.threads % self.wave != 0

    def exec_mask(self):
        rng = self.rng
        if rng.random() < 0.6:
            return None
        choice = rng.choice(("rand", "low", "high", "one", "none_hi"))
        if choice == "rand":
            return rng.getrandbits(64)
        if choice == "low":
            return 0x00000000FFFFFFFF
        if choice == "high":
            return 0xFFFFFFFF00000000
        if choice == "one":
            return 1 << rng.randrange(64)
        return rng.getrandbits(32)

    def lane_addresses(self, stride, base):
        return [(t * stride + base) for t in range(self.threads)]

    def disjoint(self, spans):
        """spans: per thread, list of (byte address, bytes). True when no byte is touched by two threads."""
        owner = {}
        for t, ss in enumerate(spans):
            for a, n in ss:
                if a < 0 or a + n > LDS_BYTES:
                    return False
                for b in range(a, a + n):
                    if owner.setdefault(b, t) != t:
                        return False
        return True

    def pick_addr(self, size, align, distinct, extra=()):
        """(stride, base, offset) so that every thread's access fits and, if distinct, threads don't overlap.
        extra: further byte offsets each thread also touches (read2/write2 second slot)."""
        rng = self.rng
        for _ in range(200):
            if distinct:
                stride = rng.choice([s for s in (align, 2 * align, 4, 8, 16, 32, 48, 64) if s % align == 0 and s >= 1])
            else:
                stride = rng.choice([0, align, 4, 8, 16, 32]) // align * align
            top = LDS_BYTES - stride * (self.threads - 1) - size - max(extra, default=0)
            if top < 0:
                continue
            off = rng.choice([0, 0, rng.randrange(0, min(top, 0xFFFF) + 1)]) // align * align
            base_top = top - off
            if base_top < 0:
                continue
            base = rng.randrange(0, base_top + 1) // align * align
            spans = [[(a + off, size)] + [(a + off + e, size) for e in extra] for a in self.lane_addresses(stride, base)]
            if not distinct or self.disjoint(spans):
                return stride, base, off
        return None

    def random_op(self):
        rng = self.rng
        kinds = ("read", "read", "write", "write", "read2", "write2", "atomic", "atomic_rtn",
                 "wrxchg", "wrxchg2", "cmpst", "swizzle", "bpermute") + (("permute",) if self.permute else ())
        while True:
            op = self.make(rng.choice(kinds))
            if op:
                return op

    def make(self, kind):
        rng = self.rng
        if kind == "read":
            name = rng.choice(sorted(READS))
            size = READS[name][0]
            align = min(size, 4) if size < 4 else (16 if size > 8 else size)
            a = self.pick_addr(size, align, distinct=False)
            if not a:
                return None
            n = (size + 3) // 4 if size >= 4 else 1
            dst = rng.randrange(POOL[0], POOL[-1] - n + 2)
            return {"kind": kind, "name": name, "addr": a, "dst": dst, "exec": self.exec_mask()}
        if kind == "write":
            name = rng.choice(sorted(WRITES))
            size = WRITES[name][0]
            align = min(size, 4) if size < 4 else (16 if size > 8 else size)
            a = self.pick_addr(size, align, distinct=True)
            if not a:
                return None
            n = (size + 3) // 4 if size >= 4 else 1
            src = rng.randrange(POOL[0], POOL[-1] - n + 2)
            return {"kind": kind, "name": name, "addr": a, "src": src, "exec": self.exec_mask()}
        if kind in ("read2", "write2"):
            st64 = rng.random() < 0.3
            unit = 256 if st64 else 4
            o0, o1 = rng.randrange(0, 4 if st64 else 64), rng.randrange(0, 4 if st64 else 64)
            if kind == "write2" and o0 == o1:
                return None
            lo = min(o0, o1) * unit
            extra = (max(o0, o1) * unit - lo,)
            a = self.pick_addr(4, 4, distinct=(kind == "write2"), extra=extra)
            if not a:
                return None
            stride, base, _ = a
            # the instruction's offsets are offset0/offset1; place the lower one at the picked spot
            base = base + a[2]
            if base - lo < 0:
                return None
            base -= lo
            op = {"kind": kind, "st64": st64, "o0": o0, "o1": o1, "addr": (stride, base, 0), "exec": self.exec_mask()}
            if kind == "read2":
                op["dst"] = rng.randrange(POOL[0], POOL[-1])
            else:
                op["src0"], op["src1"] = rng.choice(POOL), rng.choice(POOL)
            return op
        if kind == "wrxchg2":
            st64 = rng.random() < 0.3
            unit = 256 if st64 else 4
            o0, o1 = rng.randrange(0, 4 if st64 else 64), rng.randrange(0, 4 if st64 else 64)
            lo = min(o0, o1) * unit
            extra = (max(o0, o1) * unit - lo,)
            a = self.pick_addr(4, 4, distinct=True, extra=extra)
            if not a:
                return None
            stride, base, _ = a
            base = base + a[2]
            if base - lo < 0:
                return None
            base -= lo
            # the data pair is read before `vdst` is written, so `vdst` may alias it (measured on gfx1100)
            srcs = rng.sample(POOL, 2)
            dst = rng.randrange(POOL[0], POOL[-1])          # dst + 1 stays inside POOL
            return {"kind": kind, "st64": st64, "o0": o0, "o1": o1, "addr": (stride, base, 0), "dst": dst,
                    "src0": srcs[0], "src1": srcs[1], "exec": self.exec_mask()}
        if kind == "atomic":
            a = self.pick_addr(4, 4, distinct=False)
            if not a:
                return None
            return {"kind": kind, "op": rng.choice(SHARED_ATOMICS), "addr": a, "src": rng.choice(POOL),
                    "exec": self.exec_mask()}
        if kind in ("atomic_rtn", "wrxchg", "cmpst"):
            a = self.pick_addr(4, 4, distinct=True)
            if not a:
                return None
            op = {"kind": kind, "addr": a, "src": rng.choice(POOL), "dst": rng.choice(POOL), "exec": self.exec_mask()}
            if kind == "atomic_rtn":
                op["op"] = rng.choice(RTN_ATOMICS)
            if kind == "cmpst":
                op["src2"] = rng.choice(POOL)
                op["cmp_hit"] = rng.random() < 0.6   # make the compare value equal the memory word for some lanes
            return op
        if kind == "swizzle":
            if rng.random() < 0.5:
                offset = 0x8000 | rng.getrandbits(8)
            else:
                offset = rng.getrandbits(15)
            op = {"kind": kind, "offset": offset, "src": rng.choice(POOL), "dst": rng.choice(POOL), "exec": None}
            if self.swizzle_exec:
                op["exec"] = self.exec_mask()
                if op["exec"] is not None:
                    op["hyp"] = ("an active lane whose source lane is off in EXEC reads 0, like ds_bpermute; measured on gfx1150, "
                                 "RDNA2 unmeasured")
            return op
        if kind == "bpermute":
            return {"kind": kind, "mul": rng.randrange(0, 64), "add": rng.randrange(0, 64),
                    "offset": rng.choice([0, 0, 4 * rng.randrange(0, 64)]), "src": rng.choice(POOL),
                    "dst": rng.choice(POOL), "exec": self.exec_mask()}
        if kind == "permute":
            # allow both odd (bijective) and even (colliding) multipliers; RDNA2 hw (gfx1030, SHADER-133) and gfx1150
            # confirm the highest active source lane wins a collision, unaddressed active lanes get 0, halves separate
            return {"kind": kind, "mul": rng.randrange(0, 64), "add": rng.randrange(0, 64),
                    "offset": rng.choice([0, 0, 4 * rng.randrange(0, 64)]), "src": rng.choice(POOL),
                    "dst": rng.choice(POOL), "exec": self.exec_mask(),
                    "hyp": "only active lanes send, highest active sender wins collisions, a lane nobody sends to "
                           "gets 0, each 32-lane half is separate (measured on gfx1030 SHADER-133 and gfx1150)"}
        raise ValueError(kind)

    # ---- the program text ----

    def asm(self):
        """The program for the translator (llvm-mc gfx1013 text, buffer stores through the resource in s[0:3])."""
        head = [f"; {self.name}: tools/lds_fuzz.py (FUZZ-6)", f"; group: {self.threads},1,1", f"; rsrc2: {RSRC2:#x}"]
        if self.wave == 64:
            head.append("; mattr: +wavefrontsize64,-wavefrontsize32")
        return "\n".join(head + self.body(self.ops)) + "\n"

    def hw_lines(self, arch, ops=None):
        """The program as a native kernel body for tools/hw_oracle: the entry EXEC is saved and restored (a partial
        wave has fewer lanes than -1), the stores go through the output pointer in s[2:3], and the ds_* mnemonics are
        the ones the architecture names them, and so are the barrier and the LDS waits (GFX12, asm.sync_lines)."""
        from hw_oracle import asm
        ops = self.ops if ops is None else ops
        lines = self.body(ops, hw=True, arch=arch)
        return [out for ln in lines for out in asm.sync_lines(asm.ds_line(ln, arch), arch)]

    def body(self, ops, hw=False, arch="gfx1150"):
        w64 = self.wave == 64
        T = self.threads
        out = []
        if hw:
            from hw_oracle import asm
            gfx10 = asm.family(arch) == "gfx10"
            e = EXEC_SAVE
            full = f"s_mov_b64 exec, s[{e}:{e + 1}]" if w64 else f"s_mov_b32 exec_lo, s{e}"
            out.append(f"s_mov_b64 s[{e}:{e + 1}], exec" if w64 else f"s_mov_b32 s{e}, exec_lo")

            def store(data, addr, off=None):
                return (f"{'global_store_dword' if gfx10 else 'global_store_b32'} v{addr}, v{data}, s[2:3]"
                        + (f" offset:{off}" if off is not None else ""))
        else:
            full = "s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1"
            out.append(full)

            def store(data, addr, off=None):
                return f"buffer_store_dword v{data}, v{addr}, s[0:3], 0 offen" + (f" offset:{off}" if off is not None else "")
        for r, (k, c) in self.pool_init.items():
            out += [f"v_mul_u32_u24_e32 v{r}, {k:#x}, v0", f"v_xor_b32_e32 v{r}, {c:#x}, v{r}"]
        for j in range(LDS_DWORDS // T):
            out += [f"v_add_nc_u32_e32 v{TMP}, {T * j}, v0",
                    f"v_lshlrev_b32_e32 v{ADDR}, 2, v{TMP}",
                    f"v_mul_lo_u32 v{TMP}, v{TMP}, {self.lds_mul:#x}",
                    f"v_xor_b32_e32 v{TMP}, {self.lds_xor:#x}, v{TMP}",
                    f"ds_write_b32 v{ADDR}, v{TMP}"]
        out += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
        for op in ops:
            out += self.asm_op(op, w64, full)
            out += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
        for i, r in enumerate(POOL):
            out += [f"v_mul_u32_u24_e32 v{ADDR}, {4 * len(POOL)}, v0", store(r, ADDR, 4 * i)]
        for j in range(LDS_DWORDS // T):
            out += [f"v_add_nc_u32_e32 v{TMP}, {T * j}, v0",
                    f"v_lshlrev_b32_e32 v{TMP}, 2, v{TMP}",
                    f"ds_read_b32 v{ADDR}, v{TMP}",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_add_nc_u32_e32 v{TMP}, {4 * T * len(POOL)}, v{TMP}",
                    store(ADDR, TMP)]
        out.append("s_endpgm")
        return out

    def asm_addr(self, stride, base):
        return [f"v_mul_u32_u24_e32 v{ADDR}, {stride}, v0", f"v_add_nc_u32_e32 v{ADDR}, {base:#x}, v{ADDR}"]

    def asm_exec(self, mask, w64):
        if mask is None:
            return []
        mask &= (1 << min(self.threads, self.wave)) - 1     # lanes a partial wave doesn't have stay off
        if w64:
            return [f"s_mov_b32 exec_lo, {mask & M32:#x}", f"s_mov_b32 exec_hi, {mask >> 32:#x}"]
        return [f"s_mov_b32 exec_lo, {mask & M32:#x}"]

    def asm_op(self, op, w64, full):
        k = op["kind"]
        lines = []
        if k in ("read", "write", "atomic", "atomic_rtn", "wrxchg", "wrxchg2", "cmpst", "read2", "write2"):
            stride, base, off = op["addr"]
            lines += self.asm_addr(stride, base)
        if k == "cmpst" and op["cmp_hit"]:
            # compare value = the word currently in memory for every lane (so the store happens): read it first
            lines += [f"ds_read_b32 v{op['src']}, v{ADDR} offset:{op['addr'][2]}", "s_waitcnt lgkmcnt(0)"]
        if k in ("bpermute", "permute"):
            lines += [f"v_mul_u32_u24_e32 v{ADDR}, {op['mul']}, v0", f"v_add_nc_u32_e32 v{ADDR}, {op['add']}, v{ADDR}",
                      f"v_lshlrev_b32_e32 v{ADDR}, 2, v{ADDR}"]
        lines += self.asm_exec(op["exec"], w64)
        if k == "read":
            size = READS[op["name"]][0]
            n = (size + 3) // 4 if size >= 4 else 1
            dst = f"v{op['dst']}" if n == 1 else f"v[{op['dst']}:{op['dst'] + n - 1}]"
            lines.append(f"{op['name']} {dst}, v{ADDR} offset:{op['addr'][2]}")
        elif k == "write":
            size = WRITES[op["name"]][0]
            n = (size + 3) // 4 if size >= 4 else 1
            src = f"v{op['src']}" if n == 1 else f"v[{op['src']}:{op['src'] + n - 1}]"
            lines.append(f"{op['name']} v{ADDR}, {src} offset:{op['addr'][2]}")
        elif k == "read2":
            nm = "ds_read2st64_b32" if op["st64"] else "ds_read2_b32"
            lines.append(f"{nm} v[{op['dst']}:{op['dst'] + 1}], v{ADDR} offset0:{op['o0']} offset1:{op['o1']}")
        elif k == "write2":
            nm = "ds_write2st64_b32" if op["st64"] else "ds_write2_b32"
            lines.append(f"{nm} v{ADDR}, v{op['src0']}, v{op['src1']} offset0:{op['o0']} offset1:{op['o1']}")
        elif k == "atomic":
            lines.append(f"{ATOMIC_NAME[op['op']] % ''} v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "atomic_rtn":
            lines.append(f"{ATOMIC_NAME[op['op']] % '_rtn'} v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "wrxchg":
            lines.append(f"ds_wrxchg_rtn_b32 v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "wrxchg2":
            nm = "ds_wrxchg2st64_rtn_b32" if op["st64"] else "ds_wrxchg2_rtn_b32"
            lines.append(f"{nm} v[{op['dst']}:{op['dst'] + 1}], v{ADDR}, v{op['src0']}, v{op['src1']} "
                         f"offset0:{op['o0']} offset1:{op['o1']}")
        elif k == "cmpst":
            lines.append(f"ds_cmpst_rtn_b32 v{op['dst']}, v{ADDR}, v{op['src']}, v{op['src2']} offset:{op['addr'][2]}")
        elif k == "swizzle":
            lines.append(f"ds_swizzle_b32 v{op['dst']}, v{op['src']} offset:{op['offset']:#x}")
        elif k == "bpermute":
            lines.append(f"ds_bpermute_b32 v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['offset']}")
        elif k == "permute":
            lines.append(f"ds_permute_b32 v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['offset']}")
        if op["exec"] is not None:
            lines.append(full)
        return lines

    # ---- the model ----

    def model(self):
        """Run the program on the reference model. Returns (regs[t][r], lds bytearray)."""
        T = self.threads
        regs = [{r: ((t * k) & M32) ^ c for r, (k, c) in self.pool_init.items()} for t in range(T)]
        lds = bytearray(LDS_BYTES)
        for i in range(LDS_DWORDS):
            struct.pack_into("<I", lds, 4 * i, ((i * self.lds_mul) & M32) ^ self.lds_xor)
        self.writer = [{r: None for r in POOL} for _ in range(T)]
        for i, op in enumerate(self.ops):
            if op["kind"] == "cmpst" and op["cmp_hit"]:
                for t in range(T):
                    self.writer[t][op["src"]] = i
            self.model_op(op, regs, lds)
            for t in range(T):
                if self.active(t, op["exec"]):
                    for r in self.dst_regs(op):
                        self.writer[t][r] = i
        return regs, lds

    @staticmethod
    def dst_regs(op):
        k = op["kind"]
        if k == "read":
            size = READS[op["name"]][0]
            return range(op["dst"], op["dst"] + (size // 4 if size >= 4 else 1))
        if k in ("read2", "wrxchg2"):
            return range(op["dst"], op["dst"] + 2)
        if k in ("atomic_rtn", "wrxchg", "cmpst", "swizzle", "bpermute", "permute"):
            return (op["dst"],)
        return ()

    def describe(self, i):
        if i is None:
            return "initial value"
        op = self.ops[i]
        what = op.get("name") or op.get("op") or ("st64" if op.get("st64") else "")
        mask = "" if op["exec"] is None else f", EXEC {op['exec']:#x}"
        return " ".join(x for x in (f"op {i}", op["kind"], what) if x) + mask

    def active(self, t, mask):
        if mask is None:
            return True
        return bool((mask >> (t & (self.wave - 1))) & 1)

    def wave_lane(self, t):
        return t & (self.wave - 1)

    def model_op(self, op, regs, lds):
        k = op["kind"]
        T = self.threads
        rd = lambda a: struct.unpack_from("<I", lds, a)[0]               # noqa: E731
        act = [self.active(t, op["exec"]) for t in range(T)]
        if "addr" in op:
            stride, base, off = op["addr"]
            addr = [t * stride + base + off for t in range(T)]
        if k == "read":
            size, how = READS[op["name"]]
            new = []
            for t in range(T):
                if not act[t]:
                    new.append(None)
                    continue
                a = addr[t]
                if size >= 4:
                    new.append([rd(a + 4 * i) for i in range(size // 4)])
                    continue
                raw = int.from_bytes(lds[a:a + size], "little")
                bits = 8 * size
                sext = raw - (1 << bits) if raw >> (bits - 1) else raw
                if (op["name"] == "ds_read_i8" and "i8_zero_extend" in PERTURB) or \
                        (op["name"] == "ds_read_i16" and "i16_zero_extend" in PERTURB):
                    sext = raw
                old = regs[t][op["dst"]]
                if how == "u":
                    v = raw
                elif how == "i":
                    v = sext & M32
                elif how in ("u_lo", "i_lo"):
                    keep = 0 if "u8_d16_clobbers_hi" in PERTURB else old & 0xFFFF0000
                    v = keep | ((raw if how == "u_lo" else sext) & 0xFFFF)
                else:
                    keep = 0 if "d16_hi_zeroes_low" in PERTURB else old & 0xFFFF
                    v = keep | (((raw if how == "u_hi" else sext) & 0xFFFF) << 16)
                new.append([v])
            for t in range(T):
                if new[t] is not None:
                    for i, v in enumerate(new[t]):
                        regs[t][op["dst"] + i] = v
        elif k == "write":
            size, shift = WRITES[op["name"]]
            if op["name"] == "ds_write_b16_d16_hi" and "b16_hi_shift_0" in PERTURB:
                shift = 0
            for t in range(T):
                if not act[t]:
                    continue
                if size >= 4:
                    for i in range(size // 4):
                        struct.pack_into("<I", lds, addr[t] + 4 * i, regs[t][op["src"] + i])
                else:
                    v = (regs[t][op["src"]] >> shift) & ((1 << (8 * size)) - 1)
                    lds[addr[t]:addr[t] + size] = v.to_bytes(size, "little")
        elif k in ("read2", "write2"):
            unit = 256 if op["st64"] else 4
            if op["st64"] and "st64_unit_4" in PERTURB:
                unit = 4
            stride, base, _ = op["addr"]
            for t in range(T):
                if not act[t]:
                    continue
                a = t * stride + base
                a0, a1 = a + op["o0"] * unit, a + op["o1"] * unit
                if k == "read2":
                    regs[t][op["dst"]], regs[t][op["dst"] + 1] = rd(a0), rd(a1)
                else:
                    d0, d1 = regs[t][op["src0"]], regs[t][op["src1"]]
                    if "write2_swapped" in PERTURB:
                        d0, d1 = d1, d0
                    struct.pack_into("<I", lds, a0, d0)
                    struct.pack_into("<I", lds, a1, d1)
        elif k == "atomic":
            for t in range(T):
                if act[t]:
                    struct.pack_into("<I", lds, addr[t], atomic(op["op"], rd(addr[t]), regs[t][op["src"]]))
        elif k in ("atomic_rtn", "wrxchg", "cmpst"):
            if k == "cmpst" and op["cmp_hit"]:
                for t in range(T):          # the ds_read_b32 before the EXEC change runs in every lane
                    regs[t][op["src"]] = rd(addr[t])
            ret = {}
            for t in range(T):
                if not act[t]:
                    continue
                old = rd(addr[t])
                data = regs[t][op["src"]]
                if k == "atomic_rtn":
                    new = atomic(op["op"], old, data)
                elif k == "wrxchg":
                    new = data
                elif "cmpst_swap" in PERTURB:
                    new = data if old == regs[t][op["src2"]] else old
                else:
                    new = regs[t][op["src2"]] if old == data else old
                struct.pack_into("<I", lds, addr[t], new)
                ret[t] = old
            for t, v in ret.items():
                regs[t][op["dst"]] = v
        elif k == "wrxchg2":
            unit = 256 if op["st64"] else 4
            if op["st64"] and "st64_unit_4" in PERTURB:
                unit = 4
            stride, base, _ = op["addr"]
            for t in range(T):
                if not act[t]:
                    continue
                a = t * stride + base
                a0, a1 = a + op["o0"] * unit, a + op["o1"] * unit
                d0, d1 = regs[t][op["src0"]], regs[t][op["src1"]]
                if "wrxchg2_swapped" in PERTURB:
                    d0, d1 = d1, d0
                old0 = rd(a0)
                struct.pack_into("<I", lds, a0, d0)
                old1 = rd(a1)               # after the first store, so a repeated address returns data0
                struct.pack_into("<I", lds, a1, d1)
                regs[t][op["dst"]], regs[t][op["dst"] + 1] = old0, old1
        elif k == "swizzle":
            src = [regs[t][op["src"]] for t in range(T)]
            for t in range(T):
                if not act[t]:
                    continue
                wl = self.wave_lane(t)
                s = t - wl + swizzle_src(wl, op["offset"])
                regs[t][op["dst"]] = src[s] if act[s] else 0        # a source lane that is off reads 0 (see "hyp")
        elif k == "bpermute":
            # each 32-lane group (a wave32, or a half of a wave64) reads its own lanes; an inactive source gives 0
            src = [regs[t][op["src"]] for t in range(T)]
            new = {}
            for t in range(T):
                if act[t]:
                    word = ((t * op["mul"] + op["add"]) * 4 + op["offset"]) >> 2
                    if self.wave == 64 and "bpermute_wave64_cross" in PERTURB:
                        s = (t & ~63) | (word & 63)
                    else:
                        s = (t & ~31) | (word & 31)
                    new[t] = src[s] if s < T and (act[s] or "bpermute_inactive_reads" in PERTURB) else 0
            for t, v in new.items():
                regs[t][op["dst"]] = v
        elif k == "permute":
            # forward scatter within each 32-lane half: active lanes send to lane (addr + offset) / 4 of their 32-lane
            # group; highest-numbered active source wins collisions; active destination lanes nobody targets get 0
            src = [regs[t][op["src"]] for t in range(T)]
            sent = {}
            lanes = range(T - 1, -1, -1) if "permute_lowest_wins" in PERTURB else range(T)
            for t in lanes:
                if act[t]:
                    word = ((t * op["mul"] + op["add"]) * 4 + op["offset"]) >> 2
                    sent[(t & ~31) | (word & 31)] = src[t]
            for t in range(T):
                if act[t] and not ("permute_no_zero" in PERTURB and t not in sent):
                    regs[t][op["dst"]] = sent.get(t, 0)
        else:
            raise ValueError(k)


def programs(seed, count, waves, no_exec=(), **opts):
    return [Program(seed, i, w, no_exec, **opts) for i in range(count) for w in waves]


def parse_dump(data, threads=THREADS):
    words = struct.unpack(f"<{len(data) // 4}I", data)
    return parse_words(words, threads)


def parse_words(words, threads=THREADS):
    regs = [{r: words[t * len(POOL) + i] for i, r in enumerate(POOL)} for t in range(threads)]
    lds = words[threads * len(POOL):threads * len(POOL) + LDS_DWORDS]
    return regs, lds


def compare(p, got_regs, got_lds):
    want_regs, want_lds = p.model()
    diffs = []
    for t in range(p.threads):
        for r in POOL:
            if got_regs[t][r] != want_regs[t][r]:
                diffs.append(f"thread {t} v{r}: got {got_regs[t][r]:#010x}, want {want_regs[t][r]:#010x}"
                             f" (last written by {p.describe(p.writer[t][r])})")
    for i in range(LDS_DWORDS):
        w = struct.unpack_from("<I", want_lds, 4 * i)[0]
        if got_lds[i] != w:
            diffs.append(f"LDS dword {i} (byte {4 * i:#x}): got {got_lds[i]:#010x}, want {w:#010x}")
    return diffs


# ---- the translator oracle ----

def pc_tail_env():
    """True when LEONIDA_PC_TAIL_UBO asks for rdna2spirv.py --pc-tail-ubo (128-byte push constants)."""
    return (os.environ.get("LEONIDA_PC_TAIL_UBO") or "").strip().lower() in ("1", "true", "yes", "on")


def translate(progs, work, emulate, pc_tail=False):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    for d in (disasm, fssm, spirv):   # a reused --work must not leave an earlier run's listings or SPIR-V behind
        if d.exists():
            shutil.rmtree(d)
    for d in (disasm, fssm):
        d.mkdir(parents=True)
    mc = mc_command.mc_command()
    for p in progs:
        text = p.asm()
        asm_path = work / (p.name + ".asm")
        asm_path.write_text(text, encoding="utf-8", newline="\n")
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = mc_command.run(cmd, input=text, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            raise RuntimeError(f"{p.name} did not assemble:\n{r.stderr}")
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (p.name + ".s")).write_text(f"; {p.name}\n{code}", encoding="utf-8", newline="\n")
        (fssm / (p.name + ".fssm")).write_bytes(compute_header(asm_path))
    cmd = [sys.executable, "-I", os.path.join(HERE, "rdna2spirv.py"), str(disasm), "--out", str(spirv)]
    if emulate:
        cmd.append("--wave64-emulate")
    if pc_tail:
        cmd.append("--pc-tail-ubo")
    r = subprocess.run(cmd, capture_output=True, text=True)
    index = spirv / "index.json"
    if not index.exists():
        raise RuntimeError(f"rdna2spirv.py exited {r.returncode} without an index.json:\n{(r.stdout + r.stderr)[-2000:]}")
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}


def run_gpu(p, spv, build, index_entry, emulate, dump, pc_tail=False):
    exe = os.path.join(build, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(exe):
        sys.exit(f"vk_compute_test not found in {build} or its RelWithDebInfo/Release folder; "
                 "build it first (cmake --build <build dir> --target vk_compute_test)")
    env = dict(os.environ, VKC_OUT_WORDS=str(p.out_words), VKC_DUMP=str(dump))
    if pc_tail:
        env["LEONIDA_PC_TAIL_UBO"] = "1"
    args = [exe, str(spv), "1", "dump"]
    if p.wave == 64 and not emulate:
        args.append("--require-wave64")
    elif p.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    if index_entry.get("subgroup_size") and not emulate:
        env["VKC_SUBGROUP_SIZE"] = str(index_entry["subgroup_size"])
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr).strip()


def show_cmd(p, no_exec, opts):
    flags = "".join(f" --no-exec {k}" for k in no_exec)
    if opts.get("permute"):
        flags += " --permute"
    if opts.get("swizzle_exec"):
        flags += " --swizzle-exec"
    if opts.get("threads", THREADS) != THREADS:
        flags += f" --threads {opts['threads']}"
    return flags


def run_translator(progs, a, work, tally):
    """The translator oracle: translate every program with a whole number of waves and run it on the GPU. Returns the
    exit code of this oracle (0, 1, 2) and fills tally["translator"][name] with the verdict."""
    pc_tail = bool(getattr(a, "pc_tail_ubo", False))
    try:
        todo = [p for p in progs if not p.partial_wave]
        for p in progs:
            if p.partial_wave:
                print(f"SKIP {p.name}: translator: a wave64 group of {p.threads} threads is a partial wave "
                      "(the translator cannot launch one)")
        if todo:
            index = translate(todo, work, a.emulate, pc_tail=True) if pc_tail else translate(todo, work, a.emulate)
        else:
            index = {}
    except RuntimeError as e:
        print(e)
        return 2
    failed = hyp_only = untranslated = errors = 0
    skipped = len(progs) - len(todo)
    for p in todo:
        spv = work / "spirv" / (p.name + ".spv")
        if p.name not in index or not spv.is_file():
            untranslated += 1
            print(f"NOT TRANSLATED {p.name}")
            tally.setdefault("translator", {})[p.name] = "not translated"
            continue
        dump = work / (p.name + ".bin")
        dump.unlink(missing_ok=True)
        rc, msg = (run_gpu(p, spv, a.build, index[p.name], a.emulate, dump, pc_tail=True)
                   if pc_tail else run_gpu(p, spv, a.build, index[p.name], a.emulate, dump))
        if rc == 77:
            skipped += 1
            print(f"SKIP {p.name}: {msg.splitlines()[-1] if msg else 'unsupported'}")
            tally.setdefault("translator", {})[p.name] = "skipped"
            continue
        # A wave32 program needs 32-lane subgroups. Where vk_compute_test couldn't pin them, its words come from
        # another wave size and are no evidence either way. --wave64-emulate output uses no subgroup operations.
        lanes = unpinned_subgroup(msg) if rc == 0 and p.wave == 32 else None
        if lanes:
            skipped += 1
            print(f"SKIP {p.name}: the device can't run it at {lanes} lanes "
                  f"(vk_compute_test: compute subgroup size: {lanes} not available)")
            tally.setdefault("translator", {})[p.name] = "skipped"
            continue
        data = dump.read_bytes() if dump.is_file() else b""
        if rc != 0 or len(data) != 4 * p.out_words:   # dump mode itself never fails a run
            errors += 1
            print(f"ERROR {p.name}: exit {rc}, {len(data)} of {4 * p.out_words} dump bytes: {msg[-400:]}")
            tally.setdefault("translator", {})[p.name] = "error"
            continue
        regs, lds = parse_dump(data, p.threads)
        diffs = compare(p, regs, lds)
        if diffs and p.hyp_ops:
            hyp_only += 1
            tally.setdefault("translator", {})[p.name] = "hypothesis"
            print(f"HYPOTHESIS MISMATCH {p.name} ({len(diffs)} words; program has a hypothesis op); print it with "
                  f"--show {a.seed}:{p.name.split('_')[3]}:{p.wave}{show_cmd(p, a.no_exec, a.opts)}")
            for d in diffs[:a.max_report]:
                print("   ", d)
        elif diffs:
            failed += 1
            tally.setdefault("translator", {})[p.name] = "mismatch"
            print(f"MISMATCH {p.name} ({len(diffs)} words); print it with --show {a.seed}:"
                  f"{p.name.split('_')[3]}:{p.wave}{show_cmd(p, a.no_exec, a.opts)}")
            for d in diffs[:a.max_report]:
                print("   ", d)
        else:
            tally.setdefault("translator", {})[p.name] = "match"
    mode = "emulated" if a.emulate else "native"
    ran = len(progs) - untranslated - skipped - errors
    extra = f", {hyp_only} differ in a program with a hypothesis op" if hyp_only else ""
    print(f"{ran - failed - hyp_only}/{ran} programs that ran match ({mode}); {failed} mismatched{extra}, "
          f"{skipped} skipped, {untranslated} not translated, {errors} harness errors; work dir {work}")
    if errors or ran == 0:
        return 2
    return 1 if failed or untranslated else 0


# ---- the hardware oracle ----

def hw_skip_reason(p, arch):
    """Why a program cannot run natively on this GPU, or None. GFX11 renamed the ds_* loads and stores, which
    asm.ds_line respells, and swaps ds_cmpstore's data operands (measured on gfx1150 and on gfx1201); GFX12 also
    spells the barrier and the LDS wait differently, which asm.sync_lines respells. Everything else the generator
    emits is spelled the same on GFX10, GFX11 and GFX12 (assembled for gfx1030, gfx1100, gfx1150 and gfx1201), so
    the other skips are the assembler's own rejections, which the runner reports.

    One behaviour differs on GFX12: a wave64 ds_bpermute_b32 reaches all 64 lanes there, while RDNA2 (the console's
    generation, and what this model describes) keeps each 32-lane half separate. Measured with every lane asking for
    lane (l + 32) mod 64: an RX 9070 XT (gfx1201) returns lane l ^ 32, a gfx1036 iGPU returns lane l, and on gfx1201
    300 of 300 wave64 programs match --perturb bpermute_wave64_cross. So those programs can't check the RDNA2 model on
    GFX12 and are skipped there; wave32 and every other wave64 program still run."""
    from hw_oracle import asm
    if asm.family(arch) == "gfx12" and p.wave == 64 and "bpermute_wave64_cross" not in PERTURB \
            and any(op["kind"] == "bpermute" for op in p.ops):
        return "a wave64 ds_bpermute_b32 reaches all 64 lanes on GFX12 (measured on gfx1201), not each 32-lane half"
    return None


def hw_kernel(p, arch, ops=None):
    from hw_oracle import asm
    return asm.Kernel(p.hw_lines(arch, ops), p.wave, p.threads, p.out_words, raw=True, lds=LDS_BYTES, vgprs=HW_VGPRS)


def run_hw(oracle, progs, a, tally, results):
    """The hardware oracle: run every program natively (wave32 or wave64, its real group size, its EXEC masks and
    barriers) and compare every stored word with the model. results gets (program, regs, lds) of each run."""
    stats = {}
    kernels, which = [], []
    verdicts = tally.setdefault("hw", {})
    for p in progs:
        st = stats.setdefault(f"wave{p.wave} hw", {"programs": 0, "match": 0, "mismatch": 0, "hyp_only": 0,
                                                    "hyp_confirmed": 0, "skipped": 0, "rejected": 0})
        st["programs"] += 1
        why = hw_skip_reason(p, oracle.arch)
        if why:
            st["skipped"] += 1
            verdicts[p.name] = "skipped"
            print(f"SKIP {p.name} [hw]: {why}")
            continue
        kernels.append(hw_kernel(p, oracle.arch))
        which.append(p)
    harness = 0
    try:
        ran = oracle.run(kernels)
    except RuntimeError as e:
        if re.search(r"LDS|group segment|words", str(e)):
            print("hardware oracle unavailable: this backend cannot run these programs (no LDS support, or fewer "
                  f"output words than they store): {str(e)[:200]}")
            return 77
        raise
    for p, got in zip(which, ran):
        st = stats[f"wave{p.wave} hw"]
        if isinstance(got, str):
            st["rejected"] += 1
            verdicts[p.name] = "rejected"
            print(f"SKIP {p.name} [hw]: {got}")
            continue
        if len(got) != p.out_words:
            harness += 1
            verdicts[p.name] = "error"
            print(f"ERROR {p.name} [hw]: {len(got)} of {p.out_words} words")
            continue
        regs, lds = parse_words(got, p.threads)
        results.append((p, regs, lds))
        diffs = compare(p, regs, lds)
        hyp = bool(p.hyp_ops)
        if not diffs:
            st["match"] += 1
            st["hyp_confirmed"] += hyp
            verdicts[p.name] = "match"
            continue
        # On the hardware a failing hypothesis is a falsified claim, and it could hide a model bug elsewhere in the
        # program, so it is a mismatch like any other; hyp_only counts the ones that had a hypothesis op.
        st["mismatch"] += 1
        st["hyp_only"] += hyp
        verdicts[p.name] = "mismatch"
        print(f"MISMATCH {p.name} [hw] ({len(diffs)} words){' (program has a hypothesis op)' if hyp else ''}; "
              f"print it with --show {a.seed}:{p.name.split('_')[3]}:{p.wave}{show_cmd(p, a.no_exec, a.opts)}")
        for d in diffs[:a.max_report]:
            print("   ", d)
        if a.minimize:
            ops = minimize_hw(p, oracle)
            print("  minimized (ops that still differ on the hardware):")
            for i in range(len(ops)):
                print(f"    {ops[i]}" + ("   ; hypothesis op" if ops[i].get("hyp") else ""))
        if a.keep:
            os.makedirs(a.keep, exist_ok=True)
            Path(a.keep, f"{p.name}_hw.s").write_text("\n".join(p.hw_lines(oracle.arch)) + "\n", encoding="utf-8", newline="\n")
    print("hardware summary:")
    for key, st in sorted(stats.items()):
        print(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    bad = sum(st["mismatch"] for st in stats.values())
    ran = sum(st["match"] + st["mismatch"] for st in stats.values())
    if harness or ran == 0:
        return 2
    return 1 if bad else 0


def minimize_hw(p, oracle):
    """Drop ops while the program still differs on the hardware; returns the remaining ops."""
    def fails_with(ops):
        q = p.with_ops(ops)
        got = oracle.run([hw_kernel(q, oracle.arch)])[0]
        if isinstance(got, str) or len(got) != p.out_words:
            return False
        regs, lds = parse_words(got, p.threads)
        return bool(compare(q, regs, lds))
    ops = list(p.ops)
    for i in range(len(ops) - 1, -1, -1):
        trial = ops[:i] + ops[i + 1:]
        if fails_with(trial):
            ops = trial
    return ops


def family_table(results, tally_hw):
    """Per op family: ops executed in programs that matched on the hardware, in programs that did not."""
    fam = {}
    for p, regs, lds in results:
        ok = tally_hw.get(p.name) == "match"
        for op in p.ops:
            row = fam.setdefault(op_family(op), [0, 0, 0, 0])
            row[0 if ok else 1] += 1
            if op["exec"] is not None:
                row[2 if ok else 3] += 1
    return fam


def attribute(results):
    """Per op family: words that differ and were last written by an op of that family (registers only; LDS words
    are counted under 'LDS')."""
    out = {}
    for p, regs, lds in results:
        want_regs, want_lds = p.model()
        for t in range(p.threads):
            for r in POOL:
                if regs[t][r] != want_regs[t][r]:
                    w = p.writer[t][r]
                    key = "initial value" if w is None else op_family(p.ops[w])
                    out[key] = out.get(key, 0) + 1
        for i in range(LDS_DWORDS):
            if lds[i] != struct.unpack_from("<I", want_lds, 4 * i)[0]:
                out["LDS"] = out.get("LDS", 0) + 1
    return out


def negative_control(results, only=None):
    """Re-judge the hardware words already collected with each deliberately wrong model. Returns {name: (programs
    that contain the op family, programs the wrong model mismatched on)}."""
    out = {}
    for name, (fam, _) in PERTURBATIONS.items():
        if only and name != only:
            continue
        cases = bad = 0
        with perturbed(name):
            for p, regs, lds in results:
                if not any(op_family(op) == fam or op_family(op).startswith(fam + "_") or
                           op_family(op).endswith("_" + fam) for op in p.ops):
                    continue
                cases += 1
                bad += bool(compare(p, regs, lds))
        out[name] = (cases, bad)
    return out


def show(seed, index, wave, no_exec=(), arch=None, **opts):
    p = Program(seed, index, wave, no_exec, **opts)
    print("\n".join(p.hw_lines(arch)) + "\n" if arch else p.asm())
    regs, _ = p.model()
    print("; model, thread 0:", " ".join(f"v{r}={regs[0][r]:#x}" for r in POOL))
    for i in sorted(p.hyp_ops):
        print(f"; hypothesis, op {i}: {p.ops[i]['hyp']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--programs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wave", choices=("64", "32", "both"), default="both")
    ap.add_argument("--emulate", action="store_true", help="translate with --wave64-emulate, run at subgroup 32")
    ap.add_argument("--pc-tail-ubo", action="store_true",
                    help="translate with --pc-tail-ubo and run vk_compute_test with LEONIDA_PC_TAIL_UBO=1 "
                         "(128-byte push constants; also set by LEONIDA_PC_TAIL_UBO=1)")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"))
    ap.add_argument("--work", default=None, help="keep each run's generated .asm and SPIR-V in a new subfolder here")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--show", default=None, metavar="SEED:INDEX[:WAVE]")
    ap.add_argument("--max-report", type=int, default=12)
    ap.add_argument("--only", default=None, metavar="I,J,...",
                    help="run only these program indices of the seed (e.g. the ones a report names)")
    ap.add_argument("--no-exec", action="append", default=[], choices=EXEC_KINDS,
                    help="run this op kind (e.g. bpermute) with all lanes active, to isolate a known finding")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated: translator (default; needs llvm-mc, the Vulkan SDK and vk_compute_test), hw "
                         "(natively on the AMD GPU, wave32 and wave64; tools/hw_oracle/), or translator,hw")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected); with "
                    "--show, prints the native kernel body for it")
    ap.add_argument("--threads", default=str(THREADS), metavar="N|mix",
                    help="group size: 32, 64 (default), 128, 256, or mix (one of them per program)")
    ap.add_argument("--permute", action="store_true", help="also generate ds_permute_b32 (carries a hypothesis)")
    ap.add_argument("--swizzle-exec", action="store_true",
                    help="also run ds_swizzle_b32 under EXEC masks (carries a hypothesis)")
    ap.add_argument("--extended", action="store_true", help="--permute --swizzle-exec --threads mix")
    ap.add_argument("--minimize", action="store_true", help="reduce each program that differs on the hardware")
    ap.add_argument("--keep", help="write the native kernel body of each hardware mismatch here")
    ap.add_argument("--perturb", choices=sorted(PERTURBATIONS),
                    help="judge the hardware against a deliberately wrong model: mismatches must appear")
    ap.add_argument("--negative-control", action="store_true",
                    help="after the hardware run, re-judge its words with every perturbed model; exit 1 if one is "
                         "not caught")
    a = ap.parse_args()
    a.pc_tail_ubo = a.pc_tail_ubo or pc_tail_env()
    if a.programs < 1 or a.max_report < 1:
        ap.error("--programs and --max-report must be at least 1")
    if a.extended:
        a.permute = a.swizzle_exec = True
        if a.threads == str(THREADS):
            a.threads = "mix"
    try:
        threads = a.threads if a.threads == "mix" else int(a.threads)
        if threads != "mix":
            Program(0, 0, 64, threads=threads)
    except ValueError:
        ap.error("--threads takes 32, 64, 128, 256 or mix (a multiple of 32 that divides 1024)")
    a.opts = {}
    if threads != THREADS:
        a.opts["threads"] = threads
    if a.permute:
        a.opts["permute"] = True
    if a.swizzle_exec:
        a.opts["swizzle_exec"] = True
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    if (a.perturb or a.negative_control or a.minimize or a.keep) and "hw" not in backends and not a.model_only:
        ap.error("--perturb, --negative-control, --minimize and --keep need --oracle hw")
    if a.show is not None:
        parts = a.show.split(":")
        try:
            if len(parts) not in (2, 3):
                raise ValueError
            seed, index = int(parts[0]), int(parts[1])
            wave = int(parts[2]) if len(parts) == 3 else 64
        except ValueError:
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        if index < 0 or wave not in (32, 64):
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        if a.hw_arch:
            show(seed, index, wave, tuple(a.no_exec), arch=a.hw_arch, **a.opts)
        else:
            show(seed, index, wave, tuple(a.no_exec), **a.opts)
        return 0
    waves = (64, 32) if a.wave == "both" else (int(a.wave),)
    progs = programs(a.seed, a.programs, waves, tuple(a.no_exec), **a.opts)
    if a.only is not None:
        try:
            keep = {int(x) for x in a.only.split(",")}
        except ValueError:
            ap.error("--only takes a comma-separated list of program indices")
        progs = [p for p in progs if int(p.name.split("_")[3]) in keep]
        if not progs:
            ap.error("--only matches no program (the indices must be below --programs)")
    if a.model_only:
        for p in progs:
            p.model()
        print(f"{len(progs)} programs generated and modelled")
        return 0
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        from hw_oracle import asm
        why = asm.rdna23_skip_reason(oracle.arch)
        if why:
            oracle.close()
            print(f"hardware oracle unavailable: {why}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    tally, rcs, results = {}, [], []
    try:
        if "translator" in backends:
            a.build = find_test_program(a.build, "vk_compute_test") or a.build   # a Visual Studio config subfolder
            try:
                parent = Path(a.work).resolve() if a.work else None
                if parent is not None:
                    parent.mkdir(parents=True, exist_ok=True)
                work = Path(tempfile.mkdtemp(prefix="lds_fuzz_", dir=parent))
            except OSError as e:
                print(f"ERROR: cannot create work directory: {e}", file=sys.stderr)
                return 2
            print(f"work dir {work}")
            rcs.append(run_translator(progs, a, work, tally))
        if oracle:
            if a.perturb:
                PERTURB.add(a.perturb)      # the hardware side only: the translator is judged by the right model
            hw_rc = run_hw(oracle, progs, a, tally, results)
            PERTURB.discard(a.perturb)
            if hw_rc == 77:
                return 77
            rcs.append(hw_rc)
            if "translator" in backends:
                for p in progs:
                    if tally["hw"].get(p.name) == "mismatch":
                        print(f"{p.name}: hardware and model differ; translator: "
                              f"{tally['translator'].get(p.name, 'not run')}")
            if results:
                print("op families (ops run in programs that matched on the hardware / that differed; "
                      "of those, under a partial EXEC mask):")
                for fam, (m, x, me, xe) in sorted(family_table(results, tally["hw"]).items()):
                    print(f"  {fam:<22} {m:>6} / {x:<5}  masked {me:>5} / {xe}")
                bad = attribute(results)
                if bad:
                    print("differing words by the family of the op that last wrote them: " +
                          ", ".join(f"{k} {v}" for k, v in sorted(bad.items())))
            if a.negative_control or a.perturb:
                if a.perturb:
                    n_bad = sum(v == "mismatch" for v in tally["hw"].values())
                    print(f"perturbed model '{a.perturb}' ({PERTURBATIONS[a.perturb][1]}): {n_bad} of "
                          f"{len(results)} programs mismatch" + ("" if n_bad else " -- NOT DETECTED"))
                    if hw_rc != 2:          # a harness error stays one; otherwise a mismatch is the pass
                        rcs[-1] = 0 if n_bad else 1
                if a.negative_control:
                    print("negative control (wrong models judged against the same hardware words):")
                    silent = []
                    for name, (cases, bad) in negative_control(results).items():
                        note = "" if not cases else "" if bad else "  -- NOT DETECTED"
                        if cases and not bad:
                            silent.append(name)
                        print(f"  {name:<26} {cases:>5} programs with the op, {bad:>5} mismatch{note}"
                              + ("  (no program has this op: not exercised)" if not cases else ""))
                    if silent:
                        print(f"canary FAILED: perturbations not detected: {', '.join(silent)}")
                        rcs.append(1)
    finally:
        PERTURB.clear()
        if oracle:
            oracle.close()
    return 2 if 2 in rcs else 1 if 1 in rcs else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
