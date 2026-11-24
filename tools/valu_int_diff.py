r"""Differential fuzzer for the translator's 32-bit and 16-bit integer VALU instructions (standard library only).

Seeded programs made of 32-bit integer, bitfield, SAD/lerp, 16-bit op_sel, and packed 16-bit VALU instructions are
run two ways and compared bit-exact on all four exported VGPRs:
  * an independent Python reference model written from the AMD RDNA2 ISA pseudocode;
  * tools/rdna2spirv.py, executed by tests/semantic/spirv_mini_interp.py (plus optional spirv-as / spirv-val).

Usage:
    python -I tools/valu_int_diff.py [--programs N] [--seed S] [--max-ops N] [--spirv-val]
    python -I tools/valu_int_diff.py --self-test

Exit code 0 when every program matches, 1 on a mismatch (the first one is printed with its listing), 2 on a
harness error.
"""
import argparse
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

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
MASK16 = 0xFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
EDGE32 = (
    0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
    0x7FFFFF, 0x800000, 0xFFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF,
    0x01020304, 0x80FF017F, 0x12345678, 0xFEDCBA98,
)
INLINE_INTS = (0, 1, 2, 3, 4, 8, 15, 16, 31, 64, -1, -2, -4, -16)


def _s32(x):
    x &= MASK32
    return x - 0x100000000 if (x & 0x80000000) else x


def _s16(x):
    x &= MASK16
    return x - 0x10000 if (x & 0x8000) else x


def _s24(x):
    x &= 0xFFFFFF
    return x - 0x1000000 if (x & 0x800000) else x


def _s8(x):
    x &= 0xFF
    return x - 0x100 if (x & 0x80) else x


def _clamp_u32(v):
    return 0 if v < 0 else (MASK32 if v > MASK32 else v)


def _clamp_u16(v):
    return 0 if v < 0 else (MASK16 if v > MASK16 else v)


def _clamp_i16(v):
    return -0x8000 if v < -0x8000 else (0x7FFF if v > 0x7FFF else v & MASK16)


def _half(word, sel):
    return (word >> (16 if sel else 0)) & MASK16


def _write_half(old, val, sel):
    val &= MASK16
    if sel:
        return (old & MASK16) | (val << 16)
    return (old & 0xFFFF0000) | val


def eval_ref_op(op, srcs, old_dst, clamp=False, op_sel=None, op_sel_hi=None):
    """Evaluate one RDNA2 integer VALU instruction against the ISA pseudocode."""
    op_sel = op_sel or [0, 0, 0, 0]
    op_sel_hi = op_sel_hi or [1, 1, 1]
    a = [s & MASK32 for s in srcs]

    if op == "v_add_nc_u32":
        s = a[0] + a[1]
        return _clamp_u32(s) if clamp else (s & MASK32)
    if op == "v_sub_nc_u32":
        d = a[0] - a[1]
        return _clamp_u32(d) if clamp else (d & MASK32)
    if op == "v_subrev_nc_u32":
        d = a[1] - a[0]
        return _clamp_u32(d) if clamp else (d & MASK32)
    if op == "v_add3_u32":
        return (a[0] + a[1] + a[2]) & MASK32
    if op == "v_mul_lo_u32":
        return (a[0] * a[1]) & MASK32
    if op == "v_mul_i32_i24":
        return (_s24(a[0]) * _s24(a[1])) & MASK32
    if op == "v_mul_hi_i32_i24":
        return ((_s24(a[0]) * _s24(a[1])) >> 32) & MASK32
    if op == "v_mul_u32_u24":
        return ((a[0] & 0xFFFFFF) * (a[1] & 0xFFFFFF)) & MASK32
    if op == "v_mul_hi_u32_u24":
        return (((a[0] & 0xFFFFFF) * (a[1] & 0xFFFFFF)) >> 32) & MASK32
    if op == "v_and_b32":
        return a[0] & a[1]
    if op == "v_or_b32":
        return a[0] | a[1]
    if op == "v_xor_b32":
        return a[0] ^ a[1]
    if op == "v_not_b32":
        return (~a[0]) & MASK32
    if op == "v_lshlrev_b32":
        return (a[1] << (a[0] & 31)) & MASK32
    if op == "v_lshrrev_b32":
        return (a[1] >> (a[0] & 31)) & MASK32
    if op == "v_ashrrev_i32":
        return (_s32(a[1]) >> (a[0] & 31)) & MASK32
    if op == "v_bfe_u32":
        off = a[1] & 31
        w = a[2] & 31
        return 0 if w == 0 else ((a[0] >> off) & ((1 << w) - 1)) & MASK32
    if op == "v_bfe_i32":
        off = a[1] & 31
        w = min(a[2] & 31, 32 - off)
        if w == 0:
            return 0
        val = (a[0] >> off) & ((1 << w) - 1)
        if val & (1 << (w - 1)):
            val -= 1 << w
        return val & MASK32
    if op == "v_bfi_b32":
        return ((a[0] & a[1]) | ((~a[0] & MASK32) & a[2])) & MASK32
    if op == "v_bfm_b32":
        w = a[0] & 31
        off = a[1] & 31
        return (((1 << w) - 1) << off) & MASK32
    if op == "v_alignbit_b32":
        sh = a[2] & 31
        return (((a[0] << 32) | a[1]) >> sh) & MASK32
    if op == "v_alignbyte_b32":
        sh = (a[2] & 3) * 8
        return (((a[0] << 32) | a[1]) >> sh) & MASK32
    if op == "v_min_u32":
        return min(a[0], a[1])
    if op == "v_max_u32":
        return max(a[0], a[1])
    if op == "v_min_i32":
        return a[0] if _s32(a[0]) < _s32(a[1]) else a[1]
    if op == "v_max_i32":
        return a[0] if _s32(a[0]) > _s32(a[1]) else a[1]
    if op == "v_min3_u32":
        return min(a[0], a[1], a[2])
    if op == "v_max3_u32":
        return max(a[0], a[1], a[2])
    if op == "v_min3_i32":
        return min(a, key=_s32)
    if op == "v_max3_i32":
        return max(a, key=_s32)
    if op == "v_med3_u32":
        return sorted(a)[1]
    if op == "v_med3_i32":
        return sorted(a, key=_s32)[1]
    if op == "v_ffbh_u32":
        return MASK32 if a[0] == 0 else (31 - (a[0].bit_length() - 1))
    if op == "v_ffbh_i32":
        s = _s32(a[0])
        if s in (0, -1):
            return MASK32
        x = (~a[0] & MASK32) if s < 0 else a[0]
        return 31 - (x.bit_length() - 1)
    if op == "v_ffbl_b32":
        return MASK32 if a[0] == 0 else ((a[0] & -a[0]).bit_length() - 1)
    if op == "v_bcnt_u32_b32":
        return (a[0].bit_count() + a[1]) & MASK32
    if op == "v_bfrev_b32":
        return int(f"{a[0]:032b}"[::-1], 2)
    if op == "v_lerp_u8":
        out = 0
        for i in range(4):
            b0 = (a[0] >> (8 * i)) & 0xFF
            b1 = (a[1] >> (8 * i)) & 0xFF
            b2 = (a[2] >> (8 * i)) & 1
            out |= ((b0 + b1 + b2) >> 1) << (8 * i)
        return out
    if op in ("v_sad_u8", "v_sad_hi_u8"):
        s = sum(abs(((a[0] >> (8 * i)) & 0xFF) - ((a[1] >> (8 * i)) & 0xFF)) for i in range(4))
        return ((s << 16) + a[2] if op == "v_sad_hi_u8" else s + a[2]) & MASK32
    if op == "v_sad_u16":
        s = sum(abs(((a[0] >> (16 * i)) & MASK16) - ((a[1] >> (16 * i)) & MASK16)) for i in range(2))
        return (s + a[2]) & MASK32
    if op == "v_sad_u32":
        return (abs(a[0] - a[1]) + a[2]) & MASK32
    if op == "v_msad_u8":
        s = 0
        for i in range(4):
            b1 = (a[1] >> (8 * i)) & 0xFF
            if b1 != 0:
                b0 = (a[0] >> (8 * i)) & 0xFF
                s += abs(b0 - b1)
        return (s + a[2]) & MASK32

    # 16-bit VOP3 ops with op_sel:[s0, s1, (s2,) dst]
    if op in ("v_add_nc_u16", "v_sub_nc_u16", "v_add_nc_i16", "v_sub_nc_i16",
              "v_mul_lo_u16", "v_min_u16", "v_max_u16", "v_min_i16", "v_max_i16"):
        h0 = _half(a[0], op_sel[0])
        h1 = _half(a[1], op_sel[1])
        dst_sel = op_sel[2]
        if op == "v_add_nc_u16":
            r = _clamp_u16(h0 + h1) if clamp else ((h0 + h1) & MASK16)
        elif op == "v_sub_nc_u16":
            r = _clamp_u16(h0 - h1) if clamp else ((h0 - h1) & MASK16)
        elif op == "v_add_nc_i16":
            v = _s16(h0) + _s16(h1)
            r = _clamp_i16(v) if clamp else (v & MASK16)
        elif op == "v_sub_nc_i16":
            v = _s16(h0) - _s16(h1)
            r = _clamp_i16(v) if clamp else (v & MASK16)
        elif op == "v_mul_lo_u16":
            r = (h0 * h1) & MASK16
        elif op == "v_min_u16":
            r = min(h0, h1)
        elif op == "v_max_u16":
            r = max(h0, h1)
        elif op == "v_min_i16":
            r = h0 if _s16(h0) < _s16(h1) else h1
        else:
            r = h0 if _s16(h0) > _s16(h1) else h1
        return _write_half(old_dst, r, dst_sel)

    if op in ("v_min3_u16", "v_max3_u16", "v_min3_i16", "v_max3_i16", "v_med3_u16", "v_med3_i16"):
        hs = [_half(a[i], op_sel[i]) for i in range(3)]
        dst_sel = op_sel[3]
        if op == "v_min3_u16":
            r = min(hs)
        elif op == "v_max3_u16":
            r = max(hs)
        elif op == "v_min3_i16":
            r = min(hs, key=_s16)
        elif op == "v_max3_i16":
            r = max(hs, key=_s16)
        elif op == "v_med3_u16":
            r = sorted(hs)[1]
        else:
            r = sorted(hs, key=_s16)[1]
        return _write_half(old_dst, r, dst_sel)

    # Packed 16-bit VOP3P ops
    if op in ("v_pk_add_u16", "v_pk_add_i16", "v_pk_sub_u16", "v_pk_sub_i16", "v_pk_mul_lo_u16",
              "v_pk_min_u16", "v_pk_max_u16", "v_pk_min_i16", "v_pk_max_i16",
              "v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16"):
        out = 0
        for lane in range(2):
            sel0 = op_sel_hi[0] if lane else op_sel[0]
            sel1 = op_sel_hi[1] if lane else op_sel[1]
            h0 = _half(a[0], sel0)
            h1 = _half(a[1], sel1)
            if op == "v_pk_add_u16":
                r = _clamp_u16(h0 + h1) if clamp else ((h0 + h1) & MASK16)
            elif op == "v_pk_add_i16":
                r = _clamp_i16(_s16(h0) + _s16(h1)) if clamp else ((_s16(h0) + _s16(h1)) & MASK16)
            elif op == "v_pk_sub_u16":
                r = _clamp_u16(h0 - h1) if clamp else ((h0 - h1) & MASK16)
            elif op == "v_pk_sub_i16":
                r = _clamp_i16(_s16(h0) - _s16(h1)) if clamp else ((_s16(h0) - _s16(h1)) & MASK16)
            elif op == "v_pk_mul_lo_u16":
                r = (h0 * h1) & MASK16
            elif op == "v_pk_min_u16":
                r = min(h0, h1)
            elif op == "v_pk_max_u16":
                r = max(h0, h1)
            elif op == "v_pk_min_i16":
                r = h0 if _s16(h0) < _s16(h1) else h1
            elif op == "v_pk_max_i16":
                r = h0 if _s16(h0) > _s16(h1) else h1
            elif op == "v_pk_lshlrev_b16":
                r = (h1 << (h0 & 15)) & MASK16
            elif op == "v_pk_lshrrev_b16":
                r = (h1 >> (h0 & 15)) & MASK16
            else:
                r = (_s16(h1) >> (h0 & 15)) & MASK16
            out |= (r & MASK16) << (16 * lane)
        return out

    raise ValueError(f"unknown op {op}")


# (op, n_srcs, supports_clamp, mode: 'plain' | 'op16_2' | 'op16_3' | 'pk16')
OPS = [
    ("v_add_nc_u32", 2, True, "plain"),
    ("v_sub_nc_u32", 2, True, "plain"),
    ("v_subrev_nc_u32", 2, True, "plain"),
    ("v_add3_u32", 3, False, "plain"),
    ("v_mul_lo_u32", 2, False, "plain"),
    ("v_mul_i32_i24", 2, False, "plain"),
    ("v_mul_hi_i32_i24", 2, False, "plain"),
    ("v_mul_u32_u24", 2, False, "plain"),
    ("v_mul_hi_u32_u24", 2, False, "plain"),
    ("v_and_b32", 2, False, "plain"),
    ("v_or_b32", 2, False, "plain"),
    ("v_xor_b32", 2, False, "plain"),
    ("v_not_b32", 1, False, "plain"),
    ("v_lshlrev_b32", 2, False, "plain"),
    ("v_lshrrev_b32", 2, False, "plain"),
    ("v_ashrrev_i32", 2, False, "plain"),
    ("v_bfe_u32", 3, False, "plain"),
    ("v_bfe_i32", 3, False, "plain"),
    ("v_bfi_b32", 3, False, "plain"),
    ("v_bfm_b32", 2, False, "plain"),
    ("v_alignbit_b32", 3, False, "plain"),
    ("v_alignbyte_b32", 3, False, "plain"),
    ("v_min_u32", 2, False, "plain"),
    ("v_max_u32", 2, False, "plain"),
    ("v_min_i32", 2, False, "plain"),
    ("v_max_i32", 2, False, "plain"),
    ("v_min3_u32", 3, False, "plain"),
    ("v_max3_u32", 3, False, "plain"),
    ("v_min3_i32", 3, False, "plain"),
    ("v_max3_i32", 3, False, "plain"),
    ("v_med3_u32", 3, False, "plain"),
    ("v_med3_i32", 3, False, "plain"),
    ("v_ffbh_u32", 1, False, "plain"),
    ("v_ffbh_i32", 1, False, "plain"),
    ("v_ffbl_b32", 1, False, "plain"),
    ("v_bcnt_u32_b32", 2, False, "plain"),
    ("v_bfrev_b32", 1, False, "plain"),
    ("v_lerp_u8", 3, False, "plain"),
    ("v_sad_u8", 3, False, "plain"),
    ("v_sad_hi_u8", 3, False, "plain"),
    ("v_sad_u16", 3, False, "plain"),
    ("v_sad_u32", 3, False, "plain"),
    ("v_msad_u8", 3, False, "plain"),
    ("v_add_nc_u16", 2, True, "op16_2"),
    ("v_sub_nc_u16", 2, True, "op16_2"),
    ("v_add_nc_i16", 2, True, "op16_2"),
    ("v_sub_nc_i16", 2, True, "op16_2"),
    ("v_mul_lo_u16", 2, False, "op16_2"),
    ("v_min_u16", 2, False, "op16_2"),
    ("v_max_u16", 2, False, "op16_2"),
    ("v_min_i16", 2, False, "op16_2"),
    ("v_max_i16", 2, False, "op16_2"),
    ("v_min3_u16", 3, False, "op16_3"),
    ("v_max3_u16", 3, False, "op16_3"),
    ("v_min3_i16", 3, False, "op16_3"),
    ("v_max3_i16", 3, False, "op16_3"),
    ("v_med3_u16", 3, False, "op16_3"),
    ("v_med3_i16", 3, False, "op16_3"),
    ("v_pk_add_u16", 2, True, "pk16"),
    ("v_pk_add_i16", 2, True, "pk16"),
    ("v_pk_sub_u16", 2, True, "pk16"),
    ("v_pk_sub_i16", 2, True, "pk16"),
    ("v_pk_mul_lo_u16", 2, False, "pk16"),
    ("v_pk_min_u16", 2, False, "pk16"),
    ("v_pk_max_u16", 2, False, "pk16"),
    ("v_pk_min_i16", 2, False, "pk16"),
    ("v_pk_max_i16", 2, False, "pk16"),
    ("v_pk_lshlrev_b16", 2, False, "pk16"),
    ("v_pk_lshrrev_b16", 2, False, "pk16"),
    ("v_pk_ashrrev_i16", 2, False, "pk16"),
]


def _rand_val(rng):
    r = rng.random()
    if r < 0.45:
        return rng.choice(EDGE32)
    if r < 0.65:
        return rng.choice(INLINE_INTS) & MASK32
    return rng.getrandbits(32)


def build_program(rng, max_ops=16, ops=OPS):
    """Generate a random integer VALU program and its expected mrt0 (v0..v3) words."""
    vgprs = [_rand_val(rng) for _ in range(4)]
    sgprs = [_rand_val(rng) for _ in range(4)]
    lines = []
    for i, val in enumerate(vgprs):
        lines.append(f"v_mov_b32_e32 v{i}, 0x{val:08x}")
    for i, val in enumerate(sgprs):
        lines.append(f"s_mov_b32 s{i}, 0x{val:08x}")

    n_ops = rng.randint(1, max_ops)
    for _ in range(n_ops):
        op, n_srcs, can_clamp, mode = rng.choice(ops)
        dst_idx = rng.randrange(4)
        clamp = can_clamp and (rng.random() < 0.35)

        src_tokens = []
        src_vals = []
        sgpr_used = False
        lit_used = False
        for _ in range(n_srcs):
            pick = rng.random()
            if pick < 0.62 or mode == "pk16":
                vi = rng.randrange(4)
                src_tokens.append(f"v{vi}")
                src_vals.append(vgprs[vi])
            elif pick < 0.80 and not sgpr_used and mode != "pk16":
                si = rng.randrange(4)
                src_tokens.append(f"s{si}")
                src_vals.append(sgprs[si])
                sgpr_used = True
            elif pick < 0.92:
                k = rng.choice(INLINE_INTS)
                src_tokens.append(str(k))
                src_vals.append(k & MASK32)
            elif not lit_used and mode != "pk16" and not clamp:
                k = _rand_val(rng)
                src_tokens.append(f"0x{k:08x}")
                src_vals.append(k)
                lit_used = True
            else:
                vi = rng.randrange(4)
                src_tokens.append(f"v{vi}")
                src_vals.append(vgprs[vi])

        op_sel = [0, 0, 0, 0]
        op_sel_hi = [1, 1, 1]
        mods = []
        if mode == "op16_2":
            op_sel = [rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 1), 0]
            if any(op_sel[:3]):
                mods.append(f"op_sel:[{op_sel[0]},{op_sel[1]},{op_sel[2]}]")
        elif mode == "op16_3":
            op_sel = [rng.randint(0, 1) for _ in range(4)]
            if any(op_sel):
                mods.append(f"op_sel:[{op_sel[0]},{op_sel[1]},{op_sel[2]},{op_sel[3]}]")
        elif mode == "pk16":
            op_sel = [rng.randint(0, 1), rng.randint(0, 1), 0, 0]
            op_sel_hi = [rng.randint(0, 1), rng.randint(0, 1), 1]
            if op_sel[0] or op_sel[1]:
                mods.append(f"op_sel:[{op_sel[0]},{op_sel[1]}]")
            if op_sel_hi != [1, 1, 1]:
                mods.append(f"op_sel_hi:[{op_sel_hi[0]},{op_sel_hi[1]}]")
        if clamp:
            mods.append("clamp")

        if op in ("v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32") and clamp:
            mnem = op + "_e64"
        elif mode == "op16_2" and not op.startswith(("v_add_nc_", "v_sub_nc_")):
            mnem = op + "_e64"
        else:
            mnem = op
        line = f"{mnem} v{dst_idx}, " + ", ".join(src_tokens)
        if mods:
            line += " " + " ".join(mods)
        lines.append(line)
        vgprs[dst_idx] = eval_ref_op(op, src_vals, vgprs[dst_idx], clamp=clamp, op_sel=op_sel, op_sel_hi=op_sel_hi)

    lines.append("exp mrt0 v0, v1, v2, v3 done vm")
    lines.append("s_endpgm")
    return lines, vgprs


def translate_and_run(lines, name="ps_fuzz_valu_int", spirv_tools=None):
    with tempfile.TemporaryDirectory() as tmp:
        dis_dir = os.path.join(tmp, "disasm")
        os.mkdir(dis_dir)
        dis = os.path.join(dis_dir, f"{name}.s")
        dxbc = os.path.join(tmp, f"{name}.dxbc")
        with open(dis, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t" + "\n\t".join(lines) + "\n")
        with open(dxbc, "wb") as fh:
            fh.write(STUB_DXBC)
        spv_asm = os.path.join(tmp, f"{name}.spvasm")
        spv_bin = os.path.join(tmp, f"{name}.spv")
        spv_text = rdna2spirv.translate(dis)
        if spirv_tools:
            with open(spv_asm, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(spv_text)
            spirv_as, spirv_val = spirv_tools
            r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", "-o", spv_bin, spv_asm],
                                capture_output=True, text=True)
            if r1.returncode != 0:
                raise RuntimeError(f"spirv-as failed:\n{r1.stderr}")
            r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_bin],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                raise RuntimeError(f"spirv-val failed:\n{r2.stderr}")
        got = run_fragment(spv_text)
        return [w & MASK32 for w in got["%out_mrt0"]]


def run_fuzz(programs=200, seed=1, max_ops=16, spirv_val=False):
    spirv_tools = None
    if spirv_val:
        sas = shutil.which("spirv-as")
        sval = shutil.which("spirv-val")
        if not sas or not sval:
            raise RuntimeError("--spirv-val requested but spirv-as/spirv-val not found in PATH")
        spirv_tools = (sas, sval)
    rng = random.Random(seed)
    for i in range(programs):
        lines, want = build_program(rng, max_ops=max_ops)
        got = translate_and_run(lines, name=f"ps_fuzz_{seed}_{i}", spirv_tools=spirv_tools)
        if got != want:
            return i, lines, want, got
    return None


def self_test():
    # 1. Negative controls: verify reference detects real semantic bugs if perturbed.
    assert eval_ref_op("v_add_nc_u32", [0xFFFFFFFF, 5], 0, clamp=True) == 0xFFFFFFFF
    assert eval_ref_op("v_add_nc_u32", [0xFFFFFFFF, 5], 0, clamp=False) == 4
    assert eval_ref_op("v_sub_nc_i16", [0x00008000, 0x00000001], 0xABCD0000, clamp=True,
                       op_sel=[0, 0, 0, 0]) == 0xABCD8000
    assert eval_ref_op("v_pk_add_u16", [0xFFFF0001, 0x00020003], 0, clamp=True) == 0xFFFF0004
    assert eval_ref_op("v_alignbit_b32", [0x12345678, 0x9ABCDEF0, 8], 0) == 0x789ABCDE
    assert eval_ref_op("v_lerp_u8", [0x01020304, 0x03040506, 0x01000100], 0) == 0x02030405
    # 2. Run a deterministic 50-program batch against rdna2spirv + interpreter.
    fail = run_fuzz(programs=50, seed=7, max_ops=12, spirv_val=False)
    if fail is not None:
        idx, lines, want, got = fail
        raise AssertionError(f"self-test mismatch at program {idx}: want={want} got={got}\n" + "\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Differential fuzzer for 32-bit and 16-bit integer VALU instructions.")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=16)
    ap.add_argument("--spirv-val", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test OK")
        return 0
    try:
        fail = run_fuzz(programs=args.programs, seed=args.seed, max_ops=args.max_ops, spirv_val=args.spirv_val)
    except Exception as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 2
    if fail is not None:
        idx, lines, want, got = fail
        print(f"MISMATCH at program {idx} (seed {args.seed}):")
        print("  want:", [f"0x{w:08x}" for w in want])
        print("  got: ", [f"0x{w:08x}" for w in got])
        print("\n".join(f"\t{ln}" for ln in lines))
        return 1
    print(f"{args.programs} programs (seed {args.seed}): {args.programs} pass, 0 mismatch")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
