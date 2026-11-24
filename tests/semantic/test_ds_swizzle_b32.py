"""ds_swizzle_b32 vD, vS offset:swizzle(...): every swizzle mode run across the lanes of the translated SPIR-V.

Standard library only; no GPU, no LLVM, no game files.

ds_swizzle_b32 is a fixed cross-lane gather inside each group of 32 lanes ([0..31] in wave32; [0..31] and [32..63]
independently in wave64). An active destination lane reads the selected source lane's vS; a source lane that is off
in EXEC (or absent from a partial wave) reads 0; a destination lane that is off keeps its vD (docs/design.md,
ds_swizzle_b32 row). The 16-bit offset selects the source lane (RDNA2 ISA, DS_SWIZZLE_B32):
  * bit 15 set (QUAD_PERM): src = (lane & ~3) | offset[2 * (lane & 3) + 1 : 2 * (lane & 3)];
  * bit 15 clear (BITMASK_PERM, and the SWAP / REVERSE / BROADCAST shorthands of LLVM's disassembler):
    src = ((lane & and_mask) | or_mask) ^ xor_mask, with and/or/xor in offset bits [4:0] / [9:5] / [14:10].

Already covered on main: test_ds_permute.py (QUAD_PERM,1,0,3,2 and SWAP,1 with disabled neighbours, the
fragment_swizzle_exec fixture) and test_wave64_halves.py (QUAD_PERM,3,2,1,0 and SWAP,16 in a wave64 vertex shader).
This file adds the other QUAD_PERM patterns, BITMASK_PERM, REVERSE and BROADCAST, compute shaders in wave32, wave64
on one 64-wide and on two 32-wide subgroups, partial waves (20 and 40 lanes), EXEC masks, aliasing and raw bit
patterns, and a pixel shader on tests/semantic/spirv_wave_interp.py. The expected words come from hand-worked
constants and a plain-Python model of the 16-bit offset encoding, not from the translator.

The --wave64-emulate lowering goes through Workgroup memory and barriers, which spirv_mini_interp.py rejects, so it
is checked structurally only. The translator parses the swizzle(...) text LLVM's disassembler prints; a raw numeric
offset is refused rather than guessed.

    python -I tests/semantic/test_ds_swizzle_b32.py -v
"""
import gc
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

MASK32 = 0xFFFFFFFF
ALL32 = 0xFFFFFFFF
ALL64 = 0xFFFFFFFFFFFFFFFF
OLD_SENTINEL = 0xBEEF0000
HEAP_FILL = 0xEEEEEEEE
# V# in s[0:3]: base 0, stride 0, num_records 0x10000 bytes (OOB_SELECT = 3).
VSHARP = [0, 0, 0x10000, 0x31016FAC]
INPUT_BASE = 0x8000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


# ---- Independent RDNA2 ISA reference model (16-bit offset encoding & macro encoders) ----

def encode_quad_perm(a: int, b: int, c: int, d: int) -> int:
    """RDNA2 16-bit ds_swizzle offset for QUAD_PERM(a, b, c, d): bit 15 = 1, 2-bit lanes in [7:0]."""
    for v in (a, b, c, d):
        assert 0 <= v <= 3
    return 0x8000 | a | (b << 2) | (c << 4) | (d << 6)


def encode_bitmask_perm(pattern: str) -> int:
    """RDNA2 16-bit ds_swizzle offset for BITMASK_PERM("c4c3c2c1c0"):
    bit 15 = 0, and_mask = [4:0], or_mask = [9:5], xor_mask = [14:10]."""
    assert len(pattern) == 5 and all(ch in "01pi" for ch in pattern)
    and_m = or_m = xor_m = 0
    for k, ch in enumerate(pattern):
        bit = 1 << (4 - k)
        if ch in "pi":
            and_m |= bit
        if ch == "1":
            or_m |= bit
        if ch == "i":
            xor_m |= bit
    return and_m | (or_m << 5) | (xor_m << 10)


def encode_swap(stride: int) -> int:
    """RDNA2 16-bit offset for SWAP(stride): lane ^ stride (bitmask with and=31, or=0, xor=stride)."""
    assert stride in (1, 2, 4, 8, 16)
    return 0x1F | (stride << 10)


def encode_reverse(width: int) -> int:
    """RDNA2 16-bit offset for REVERSE(width): lane ^ (width - 1)."""
    assert width in (2, 4, 8, 16, 32)
    return 0x1F | ((width - 1) << 10)


def encode_broadcast(group_size: int, lane_idx: int) -> int:
    """RDNA2 16-bit offset for BROADCAST(group_size, lane_idx): (lane & ~(group_size - 1)) | lane_idx."""
    assert group_size in (2, 4, 8, 16, 32) and 0 <= lane_idx < group_size
    and_m = (~(group_size - 1)) & 0x1F
    return and_m | (lane_idx << 5)


def isa_swizzle_src_lane(lane: int, offset16: int) -> int:
    """RDNA2 ISA source lane within a 32-lane group from the 16-bit offset field."""
    if offset16 & 0x8000:
        sel = (offset16 >> (2 * (lane & 3))) & 3
        return (lane & ~3) | sel
    and_m = offset16 & 0x1F
    or_m = (offset16 >> 5) & 0x1F
    xor_m = (offset16 >> 10) & 0x1F
    return (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m)


def ref_swizzle(src_vals: list[int], dst_old: list[int], offset16: int, exec_mask: int,
                zero_inactive_src: bool = True) -> list[int]:
    """Independent RDNA2 ISA reference model for `ds_swizzle_b32`."""
    lanes = len(src_vals)
    assert len(dst_old) == lanes and lanes in (32, 64)
    out = []
    for lane in range(lanes):
        if not ((exec_mask >> lane) & 1):
            out.append(dst_old[lane] & MASK32)
            continue
        src = isa_swizzle_src_lane(lane, offset16)
        if zero_inactive_src and not ((exec_mask >> src) & 1):
            out.append(0)
        else:
            out.append(src_vals[src] & MASK32)
    return out


# ---- Harness helpers ----

class _Heap(dict):
    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


_TMP_DIR = None
_DISASM_DIR = None
_FSSM_DIR = None


def _make_fssm(threads: int, rsrc2: int = 0x8088) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, threads, 1, 1)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _DISASM_DIR, _FSSM_DIR
    _TMP_DIR = tempfile.TemporaryDirectory()
    _DISASM_DIR = os.path.join(_TMP_DIR.name, "disasm")
    _FSSM_DIR = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(_DISASM_DIR)
    os.mkdir(_FSSM_DIR)


def tearDownModule():
    global _TMP_DIR, _DISASM_DIR, _FSSM_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _DISASM_DIR = None
        _FSSM_DIR = None


def translate_cs(name: str, lines: list[str], threads: int, emulate: bool = False) -> tuple[str, object]:
    assert name.startswith("cs_"), f"compute shader name must start with cs_: {name!r}"
    if _TMP_DIR is None:
        setUpModule()
    with open(os.path.join(_FSSM_DIR, name + ".fssm"), "wb") as f:
        f.write(_make_fssm(threads))
    src_path = os.path.join(_DISASM_DIR, name + ".s")
    asm = f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(src_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    old_emu = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src_path)
            gen = rdna2spirv.translate.last_gen
            gc.collect()
    finally:
        rdna2spirv.WAVE64_EMULATE = old_emu
    return spv, gen


def translate_ps(name: str, lines: list[str]) -> tuple[str, object]:
    assert name.startswith("ps_"), f"pixel shader name must start with ps_: {name!r}"
    if _TMP_DIR is None:
        setUpModule()
    with open(os.path.join(_TMP_DIR.name, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    src_path = os.path.join(_DISASM_DIR, name + ".s")
    asm = f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(src_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
        gen = rdna2spirv.translate.last_gen
        gc.collect()
    return spv, gen


def run_cs(name: str, lines: list[str], threads: int, subgroup_size: int,
           init_inputs: list[int] | None = None, out_slots: int = 1) -> list[int]:
    """Translate and run a compute shader; return `threads * out_slots` 32-bit output words."""
    spv, _ = translate_cs(name, lines, threads, emulate=False)
    mem = _Heap()
    if init_inputs is not None:
        for i, val in enumerate(init_inputs):
            mem[INPUT_BASE + 4 * i] = val & MASK32
    si.run_workgroup(
        spv, threads, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    out_bytes = 4 * threads * out_slots
    stray = sorted(a for a in mem if a < INPUT_BASE and (a % 4 != 0 or a >= out_bytes))
    if stray:
        raise AssertionError(f"{name}: unexpected stores outside out[0..{threads * out_slots - 1}]: {stray}")
    return [mem.get(4 * k) for k in range(threads * out_slots)]


def exec_mask_lines(mask: int, wave: int) -> list[str]:
    lines = [f"s_mov_b32 exec_lo, {mask & MASK32:#x}"]
    if wave == 64:
        lines.append(f"s_mov_b32 exec_hi, {(mask >> 32) & MASK32:#x}")
    return lines


def restore_exec_lines(wave: int) -> list[str]:
    return ["s_mov_b64 exec, -1"] if wave == 64 else ["s_mov_b32 exec_lo, -1"]


def store_slots_lines(regs: list[str], threads: int) -> list[str]:
    lines = ["v_lshlrev_b32_e32 v21, 2, v0"]
    for slot, reg in enumerate(regs):
        if slot == 0:
            lines.append(f"buffer_store_dword {reg}, v21, s[0:3], 0 offen")
        else:
            lines.append(f"buffer_store_dword {reg}, v21, s[0:3], 0 offen offset:{4 * slot * threads}")
    return lines


def _run_swizzle_batch(name: str, swz_specs: list[tuple[str, int]], init_vals: list[int],
                       wave: int, subgroup_size: int, exec_mask: int | None = None) -> list[list[int]]:
    """Batch up to 16 `ds_swizzle_b32` operations into a single compute shader and return per-slot lane vectors."""
    assert 1 <= len(swz_specs) <= 16
    lines = []
    if wave == 64:
        lines.append("s_mov_b64 exec, -1")
    lines += [
        "v_lshlrev_b32_e32 v20, 2, v0",
        f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
        "buffer_load_dword v1, v20, s[0:3], 0 offen",
    ]
    regs = []
    for idx in range(len(swz_specs)):
        dst = f"v{2 + idx}"
        regs.append(dst)
        lines.append(f"v_or_b32_e32 {dst}, {OLD_SENTINEL:#x}, v0")
    if exec_mask is not None:
        lines.extend(exec_mask_lines(exec_mask, wave))
    for idx, (mod_str, _) in enumerate(swz_specs):
        lines.append(f"ds_swizzle_b32 v{2 + idx}, v1 offset:{mod_str}")
    lines.append("s_waitcnt lgkmcnt(0)")
    if exec_mask is not None:
        lines.extend(restore_exec_lines(wave))
    lines.extend(store_slots_lines(regs, wave))
    flat = run_cs(name, lines, wave, subgroup_size, init_inputs=init_vals, out_slots=len(swz_specs))
    return [flat[slot * wave:(slot + 1) * wave] for slot in range(len(swz_specs))]


# ---- Test suites ----

class OracleTests(unittest.TestCase):
    """Cross-check hand-worked constants and macro encoders against the 16-bit RDNA2 ISA model."""

    def test_quad_perm_hand_worked_constants(self):
        vals = [100 + l for l in range(32)]
        old = [OLD_SENTINEL | l for l in range(32)]

        # Broadcast lane 0/1/2/3 inside every quad.
        for k in range(4):
            got = ref_swizzle(vals, old, encode_quad_perm(k, k, k, k), ALL32)
            self.assertEqual(got, [100 + (l & ~3) + k for l in range(32)])

        # Reverse quad [3,2,1,0]: quad 0 -> [103, 102, 101, 100], quad 7 -> [131, 130, 129, 128].
        rev = ref_swizzle(vals, old, encode_quad_perm(3, 2, 1, 0), ALL32)
        self.assertEqual(rev[:4], [103, 102, 101, 100])
        self.assertEqual(rev[28:32], [131, 130, 129, 128])

        # Swap pairs [1,0,3,2] and rotate [1,2,3,0].
        swap = ref_swizzle(vals, old, encode_quad_perm(1, 0, 3, 2), ALL32)
        self.assertEqual(swap[:8], [101, 100, 103, 102, 105, 104, 107, 106])
        rot = ref_swizzle(vals, old, encode_quad_perm(1, 2, 3, 0), ALL32)
        self.assertEqual(rot[:8], [101, 102, 103, 100, 105, 106, 107, 104])

    def test_bitmask_shorthand_encoding_equivalences(self):
        # SWAP, REVERSE, BROADCAST must produce the exact same 16-bit offsets as BITMASK_PERM.
        self.assertEqual(encode_swap(1), encode_bitmask_perm("ppppi"))
        self.assertEqual(encode_swap(2), encode_bitmask_perm("pppip"))
        self.assertEqual(encode_swap(4), encode_bitmask_perm("ppipp"))
        self.assertEqual(encode_swap(8), encode_bitmask_perm("pippp"))
        self.assertEqual(encode_swap(16), encode_bitmask_perm("ipppp"))

        self.assertEqual(encode_reverse(2), encode_bitmask_perm("ppppi"))
        self.assertEqual(encode_reverse(4), encode_bitmask_perm("pppii"))
        self.assertEqual(encode_reverse(8), encode_bitmask_perm("ppiii"))
        self.assertEqual(encode_reverse(16), encode_bitmask_perm("piiii"))
        self.assertEqual(encode_reverse(32), encode_bitmask_perm("iiiii"))

        self.assertEqual(encode_broadcast(8, 0), encode_bitmask_perm("pp000"))
        self.assertEqual(encode_broadcast(8, 5), encode_bitmask_perm("pp101"))
        self.assertEqual(encode_broadcast(16, 0), encode_bitmask_perm("p0000"))
        self.assertEqual(encode_broadcast(32, 0), encode_bitmask_perm("00000"))
        self.assertEqual(encode_broadcast(32, 31), encode_bitmask_perm("11111"))

    def test_inactive_source_zero_fill_in_isa_model(self):
        # ISA model: an active lane reading an inactive source lane gets 0; an inactive destination keeps its vD.
        vals = [100 + l for l in range(32)]
        old = [OLD_SENTINEL | l for l in range(32)]
        mask_no_0 = ALL32 ^ 1
        got = ref_swizzle(vals, old, encode_broadcast(32, 0), mask_no_0, zero_inactive_src=True)
        self.assertEqual(got[0], old[0])
        self.assertEqual(got[1:], [0] * 31)


class Wave32QuadPermTests(unittest.TestCase):
    """End-to-end 32-lane execution for QUAD_PERM mode."""

    def test_quad_perm_broadcasts_reverse_swap_rotate_and_asymmetric(self):
        init_vals = [(0x10000000 + l * 0x010101) & MASK32 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        specs = [
            # Identity
            ("swizzle(QUAD_PERM,0,1,2,3)", encode_quad_perm(0, 1, 2, 3)),
            # Broadcast lane 0, 1, 2, 3 of each quad
            ("swizzle(QUAD_PERM,0,0,0,0)", encode_quad_perm(0, 0, 0, 0)),
            ("swizzle(QUAD_PERM,1,1,1,1)", encode_quad_perm(1, 1, 1, 1)),
            ("swizzle(QUAD_PERM,2,2,2,2)", encode_quad_perm(2, 2, 2, 2)),
            ("swizzle(QUAD_PERM,3,3,3,3)", encode_quad_perm(3, 3, 3, 3)),
            # Reverse quad [3,2,1,0]
            ("swizzle(QUAD_PERM,3,2,1,0)", encode_quad_perm(3, 2, 1, 0)),
            # Swap pairs [1,0,3,2] and swap half-quads [2,3,0,1]
            ("swizzle(QUAD_PERM,1,0,3,2)", encode_quad_perm(1, 0, 3, 2)),
            ("swizzle(QUAD_PERM,2,3,0,1)", encode_quad_perm(2, 3, 0, 1)),
            # Rotate left [1,2,3,0] and rotate right [3,0,1,2]
            ("swizzle(QUAD_PERM,1,2,3,0)", encode_quad_perm(1, 2, 3, 0)),
            ("swizzle(QUAD_PERM,3,0,1,2)", encode_quad_perm(3, 0, 1, 2)),
            # Non-bijective / arbitrary quad patterns
            ("swizzle(QUAD_PERM,2,0,3,2)", encode_quad_perm(2, 0, 3, 2)),
            ("swizzle(QUAD_PERM,3,3,0,1)", encode_quad_perm(3, 3, 0, 1)),
        ]
        results = _run_swizzle_batch("cs_w32_swz_quad", specs, init_vals, 32, 32)
        for idx, (mod_str, off16) in enumerate(specs):
            want = ref_swizzle(init_vals, old_vals, off16, ALL32)
            with self.subTest(pattern=mod_str):
                self.assertEqual(results[idx], want)

        # Explicit hand-checked spot checks (independent of ref_swizzle).
        self.assertEqual(results[0], init_vals)                                # identity
        self.assertEqual(results[1][:8], [init_vals[0]] * 4 + [init_vals[4]] * 4)  # bcast 0
        self.assertEqual(results[5][:4], [init_vals[3], init_vals[2], init_vals[1], init_vals[0]])
        self.assertEqual(results[6][:4], [init_vals[1], init_vals[0], init_vals[3], init_vals[2]])
        self.assertEqual(results[8][:4], [init_vals[1], init_vals[2], init_vals[3], init_vals[0]])

    def test_quad_sum_reduction(self):
        # 2-step quad reduction: swap pairs [1,0,3,2], then swap half-quads [2,3,0,1].
        init_vals = [11 * l + 3 for l in range(32)]
        got = run_cs("cs_w32_swz_quad_reduce", [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,2,3,0,1)",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v1, v21, s[0:3], 0 offen",
        ], 32, 32, init_inputs=init_vals)
        want = [sum(init_vals[(l & ~3):(l & ~3) + 4]) for l in range(32)]
        self.assertEqual(got, want)


class Wave32BitmaskAndShorthandTests(unittest.TestCase):
    """End-to-end 32-lane execution for BITMASK_PERM, SWAP, REVERSE, and BROADCAST modes."""

    def test_bitmask_perm_xor_butterflies_broadcasts_and_mixed(self):
        init_vals = [0x5000 + l * 19 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        patterns = [
            # Identity
            "ppppp",
            # XOR butterfly strides 1, 2, 4, 8, 16 (swap halves)
            "ppppi",
            "pppip",
            "ppipp",
            "pippp",
            "ipppp",
            # Combined XOR masks ("pippi" = lane^9 as in ps_bitmask_perm.asm, full reverse, checkerboard)
            "pippi",
            "iiiii",
            "ipipi",
            # AND/OR broadcasts across 4, 8, 16, 32 lanes
            "ppp00",
            "pp000",
            "p0000",
            "00000",
            "11111",
            # Mixed AND/OR/XOR ("10p0i" = ((lane & 5) | 16) ^ 1 as in ps_bitmask_perm.asm)
            "10p0i",
            "01i1p",
        ]
        specs = [(f'swizzle(BITMASK_PERM,"{p}")', encode_bitmask_perm(p)) for p in patterns]
        results = _run_swizzle_batch("cs_w32_swz_bitmask", specs, init_vals, 32, 32)
        for idx, (mod_str, off16) in enumerate(specs):
            want = ref_swizzle(init_vals, old_vals, off16, ALL32)
            with self.subTest(pattern=mod_str):
                self.assertEqual(results[idx], want)

        # Spot-check key patterns directly (independent of ref_swizzle):
        # "ipppp" swaps [0..15] <-> [16..31]
        self.assertEqual(results[5], init_vals[16:32] + init_vals[0:16])
        # "pippi" = lane ^ 9
        self.assertEqual(results[6], [init_vals[l ^ 9] for l in range(32)])
        # "00000" broadcasts lane 0 across all 32 lanes; "11111" broadcasts lane 31
        self.assertEqual(results[12], [init_vals[0]] * 32)
        self.assertEqual(results[13], [init_vals[31]] * 32)
        # "10p0i" = ((lane & 5) | 16) ^ 1
        self.assertEqual(results[14], [init_vals[((l & 5) | 16) ^ 1] for l in range(32)])

    def test_swap_reverse_and_broadcast_shorthands(self):
        init_vals = [0x9000 + l * 23 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        specs = [
            # SWAP strides 1, 2, 4, 8, 16
            ("swizzle(SWAP,1)", encode_swap(1)),
            ("swizzle(SWAP,2)", encode_swap(2)),
            ("swizzle(SWAP,4)", encode_swap(4)),
            ("swizzle(SWAP,8)", encode_swap(8)),
            ("swizzle(SWAP,16)", encode_swap(16)),
            # REVERSE widths 2, 4, 8, 16, 32
            ("swizzle(REVERSE,2)", encode_reverse(2)),
            ("swizzle(REVERSE,4)", encode_reverse(4)),
            ("swizzle(REVERSE,8)", encode_reverse(8)),
            ("swizzle(REVERSE,16)", encode_reverse(16)),
            ("swizzle(REVERSE,32)", encode_reverse(32)),
            # BROADCAST group sizes 2, 4, 8, 16, 32 with zero and non-zero lane indices
            ("swizzle(BROADCAST,2,1)", encode_broadcast(2, 1)),
            ("swizzle(BROADCAST,4,2)", encode_broadcast(4, 2)),
            ("swizzle(BROADCAST,8,0)", encode_broadcast(8, 0)),
            ("swizzle(BROADCAST,16,0)", encode_broadcast(16, 0)),
            ("swizzle(BROADCAST,32,0)", encode_broadcast(32, 0)),
            ("swizzle(BROADCAST,32,19)", encode_broadcast(32, 19)),
        ]
        results = _run_swizzle_batch("cs_w32_swz_shorthands", specs, init_vals, 32, 32)
        for idx, (mod_str, off16) in enumerate(specs):
            want = ref_swizzle(init_vals, old_vals, off16, ALL32)
            with self.subTest(pattern=mod_str):
                self.assertEqual(results[idx], want)

        # Spot checks: SWAP,16 swaps halves; REVERSE,32 reverses all 32; BROADCAST,8,0 broadcasts 0/8/16/24.
        self.assertEqual(results[4], init_vals[16:32] + init_vals[0:16])
        self.assertEqual(results[9], list(reversed(init_vals)))
        self.assertEqual(results[12], [init_vals[l & ~7] for l in range(32)])
        self.assertEqual(results[13], [init_vals[l & ~15] for l in range(32)])
        self.assertEqual(results[14], [init_vals[0]] * 32)
        self.assertEqual(results[15], [init_vals[19]] * 32)

    def test_butterfly_32lane_sum_reduction(self):
        # Full 32-lane XOR butterfly reduction mixing SWAP and BITMASK_PERM.
        init_vals = [7 * l + 5 for l in range(32)]
        expected_sum = sum(init_vals)
        got = run_cs("cs_w32_swz_butterfly", [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppppi")',
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ppipp")',
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"ipppp")',
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, v1, v2",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v1, v21, s[0:3], 0 offen",
        ], 32, 32, init_inputs=init_vals)
        self.assertEqual(got, [expected_sum] * 32)


class Wave32EdgeCaseTests(unittest.TestCase):
    """Aliasing, raw 32-bit pattern preservation, EXEC destination masking, and partial waves."""

    def test_destination_aliasing_and_raw_bit_patterns(self):
        # Include NaN payloads, -0.0, +inf, 0xFFFFFFFF, 0x00000000 to verify bit-exact raw shuffle.
        special = [0x7FC01234, 0xFFC0DEAD, 0x80000000, 0x7F800000, 0xFFFFFFFF, 0x00000000, 0x00000001, 0xDEADBEEF]
        init_vals = [special[l % len(special)] ^ (l << 8) for l in range(32)]
        got = run_cs("cs_w32_swz_alias", [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            "v_lshlrev_b32_e32 v21, 2, v0",
            # Alias v1, v1 in-place
            "ds_swizzle_b32 v1, v1 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "s_waitcnt lgkmcnt(0)",
            # Alias v0, v0 in-place
            'ds_swizzle_b32 v0, v0 offset:swizzle(BITMASK_PERM,"ipppp")',
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v1, v21, s[0:3], 0 offen",
            f"buffer_store_dword v0, v21, s[0:3], 0 offen offset:{4 * 32}",
        ], 32, 32, init_inputs=init_vals, out_slots=2)
        self.assertEqual(got[:32], ref_swizzle(init_vals, init_vals, encode_quad_perm(3, 2, 1, 0), ALL32))
        self.assertEqual(got[32:], [l ^ 16 for l in range(32)])

    def test_exec_destination_masking_with_active_sources(self):
        init_vals = [0x2000 + l * 13 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]

        # 1. Whole active quads (quads 0, 2, 5, 7 active -> mask 0xF0F00F0F): QUAD_PERM, SWAP,1/2, REVERSE,4
        #    only access lanes within the active quads, while inactive quads preserve OLD_SENTINEL | lane.
        mask_quads = 0xF0F00F0F
        specs_q = [
            ("swizzle(QUAD_PERM,3,2,1,0)", encode_quad_perm(3, 2, 1, 0)),
            ("swizzle(QUAD_PERM,1,2,3,0)", encode_quad_perm(1, 2, 3, 0)),
            ("swizzle(SWAP,1)", encode_swap(1)),
            ("swizzle(REVERSE,4)", encode_reverse(4)),
            ('swizzle(BITMASK_PERM,"ppp00")', encode_bitmask_perm("ppp00")),
        ]
        res_q = _run_swizzle_batch("cs_w32_swz_exec_quads", specs_q, init_vals, 32, 32, exec_mask=mask_quads)
        for idx, (mod_str, off16) in enumerate(specs_q):
            with self.subTest(mask=hex(mask_quads), pattern=mod_str):
                self.assertEqual(res_q[idx], ref_swizzle(init_vals, old_vals, off16, mask_quads))

        # 2. Active even lanes (0x55555555) broadcasting lane 0 across 8, 16, 32 lanes (all source lanes
        #    0, 8, 16, 24 are even, hence active); odd lanes keep OLD_SENTINEL | lane.
        mask_even = 0x55555555
        specs_e = [
            ("swizzle(BROADCAST,8,0)", encode_broadcast(8, 0)),
            ("swizzle(BROADCAST,16,0)", encode_broadcast(16, 0)),
            ("swizzle(BROADCAST,32,0)", encode_broadcast(32, 0)),
            ('swizzle(BITMASK_PERM,"ipppp")', encode_bitmask_perm("ipppp")),  # even^16 is even (active)
        ]
        res_e = _run_swizzle_batch("cs_w32_swz_exec_even", specs_e, init_vals, 32, 32, exec_mask=mask_even)
        for idx, (mod_str, off16) in enumerate(specs_e):
            with self.subTest(mask=hex(mask_even), pattern=mod_str):
                self.assertEqual(res_e[idx], ref_swizzle(init_vals, old_vals, off16, mask_even))

    def test_inactive_source_lane_reads_zero(self):
        # EXEC without lane 0: BROADCAST,32,0 and QUAD_PERM,0,0,0,0 read lane 0 (or quad 0's lane 0) from an
        # inactive source and get 0; lane 0 keeps its vD; REVERSE,32 lane 31 reads inactive lane 0.
        init_vals = [0x3000 + l * 7 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        mask = ALL32 ^ 1
        specs = [
            ("swizzle(BROADCAST,32,0)", encode_broadcast(32, 0)),
            ("swizzle(QUAD_PERM,0,0,0,0)", encode_quad_perm(0, 0, 0, 0)),
            ("swizzle(REVERSE,32)", encode_reverse(32)),
        ]
        res = _run_swizzle_batch("cs_w32_swz_inact_src", specs, init_vals, 32, 32, exec_mask=mask)
        for idx, (mod_str, off16) in enumerate(specs):
            with self.subTest(pattern=mod_str):
                self.assertEqual(res[idx], ref_swizzle(init_vals, old_vals, off16, mask))
        self.assertEqual(res[0], [OLD_SENTINEL] + [0] * 31)
        self.assertEqual(res[1][:5], [OLD_SENTINEL, 0, 0, 0, 0x3000 + 4 * 7])
        self.assertEqual(res[2][31], 0)

    def test_partial_wave_20_lanes(self):
        # 20 threads on a 32-wide subgroup: lanes 20..31 do not exist.
        # Quads [0..3]..[16..19] are complete, so QUAD_PERM / SWAP,1 / SWAP,2 / REVERSE,4 stay in-bounds.
        # Under EXEC = 0xFFFF (lanes 16..19 off), SWAP,8 on lanes 0..15 reads lanes 0..15; lanes 16..19 keep vD.
        got = run_cs("cs_w32_swz_partial20", [
            "v_add_nc_u32_e32 v1, 700, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v4, {OLD_SENTINEL:#x}, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,2,3,0)",
            "ds_swizzle_b32 v3, v1 offset:swizzle(BROADCAST,4,3)",
            "s_mov_b32 exec_lo, 0xffff",
            "ds_swizzle_b32 v4, v1 offset:swizzle(SWAP,8)",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            *store_slots_lines(["v2", "v3", "v4"], 20),
        ], 20, 32, out_slots=3)
        vals = [700 + l for l in range(20)] + [0] * 12
        old = [OLD_SENTINEL | l for l in range(20)] + [0] * 12
        self.assertEqual(got[0:20], ref_swizzle(vals, old, encode_quad_perm(1, 2, 3, 0), 0xFFFFF)[:20])
        self.assertEqual(got[20:40], ref_swizzle(vals, old, encode_broadcast(4, 3), 0xFFFFF)[:20])
        self.assertEqual(got[40:60], ref_swizzle(vals, old, encode_swap(8), 0xFFFF)[:20])

        # SWAP,16: lanes 4..15 read absent lanes 20..31 and get 0 (the translator's model for an absent source,
        # as in test_ds_permute.py; unverified on hardware); lanes 0..3 and 16..19 swap with each other.
        got = run_cs("cs_w32_swz_partial20_absent", [
            "v_add_nc_u32_e32 v1, 700, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(0)",
            *store_slots_lines(["v2"], 20),
        ], 20, 32)
        self.assertEqual(got, [716, 717, 718, 719] + [0] * 12 + [700, 701, 702, 703])


class Wave64ExecutionTests(unittest.TestCase):
    """End-to-end 64-lane execution on BOTH one 64-wide subgroup AND two 32-wide subgroups."""

    def test_all_five_modes_stay_isolated_per_32lane_half_on_sg64_and_sg32(self):
        init_vals = [0x4000 + l * 31 for l in range(64)]
        old_vals = [OLD_SENTINEL | l for l in range(64)]
        specs = [
            # QUAD_PERM
            ("swizzle(QUAD_PERM,0,0,0,0)", encode_quad_perm(0, 0, 0, 0)),
            ("swizzle(QUAD_PERM,3,2,1,0)", encode_quad_perm(3, 2, 1, 0)),
            ("swizzle(QUAD_PERM,1,0,3,2)", encode_quad_perm(1, 0, 3, 2)),
            ("swizzle(QUAD_PERM,1,2,3,0)", encode_quad_perm(1, 2, 3, 0)),
            # BITMASK_PERM (XOR butterflies, AND/OR broadcasts, mixed)
            ('swizzle(BITMASK_PERM,"ppppi")', encode_bitmask_perm("ppppi")),
            ('swizzle(BITMASK_PERM,"ipppp")', encode_bitmask_perm("ipppp")),
            ('swizzle(BITMASK_PERM,"00000")', encode_bitmask_perm("00000")),
            ('swizzle(BITMASK_PERM,"11111")', encode_bitmask_perm("11111")),
            ('swizzle(BITMASK_PERM,"pippi")', encode_bitmask_perm("pippi")),
            ('swizzle(BITMASK_PERM,"10p0i")', encode_bitmask_perm("10p0i")),
            # SWAP
            ("swizzle(SWAP,4)", encode_swap(4)),
            ("swizzle(SWAP,16)", encode_swap(16)),
            # REVERSE
            ("swizzle(REVERSE,8)", encode_reverse(8)),
            ("swizzle(REVERSE,32)", encode_reverse(32)),
            # BROADCAST
            ("swizzle(BROADCAST,16,5)", encode_broadcast(16, 5)),
            ("swizzle(BROADCAST,32,0)", encode_broadcast(32, 0)),
        ]
        for subgroup_size in (64, 32):
            results = _run_swizzle_batch(
                f"cs_w64_swz_all_sg{subgroup_size}", specs, init_vals, 64, subgroup_size,
            )
            for idx, (mod_str, off16) in enumerate(specs):
                want = ref_swizzle(init_vals, old_vals, off16, ALL64)
                with self.subTest(subgroup_size=subgroup_size, pattern=mod_str):
                    self.assertEqual(results[idx], want)

            # Explicitly confirm upper-half [32..63] isolation from lower-half [0..31]:
            # BITMASK_PERM "00000" and BROADCAST,32,0 -> lanes 0..31 get init[0], lanes 32..63 get init[32].
            self.assertEqual(results[6], [init_vals[0]] * 32 + [init_vals[32]] * 32)
            self.assertEqual(results[15], [init_vals[0]] * 32 + [init_vals[32]] * 32)
            # REVERSE,32 -> low half reversed within 0..31, high half reversed within 32..63.
            self.assertEqual(results[13], list(reversed(init_vals[:32])) + list(reversed(init_vals[32:])))

    def test_wave64_asymmetric_exec_mask_and_butterfly_halves(self):
        init_vals = [0x3000 + l * 7 for l in range(64)]
        old_vals = [OLD_SENTINEL | l for l in range(64)]
        # Low half: quads 0, 3, 5 active; high half: quads 1, 2, 6, 7 active.
        exec_asym = (0xFF000FF0 << 32) | 0x00F0F00F
        specs = [
            ("swizzle(QUAD_PERM,3,2,1,0)", encode_quad_perm(3, 2, 1, 0)),
            ("swizzle(SWAP,2)", encode_swap(2)),
            ('swizzle(BITMASK_PERM,"ppp11")', encode_bitmask_perm("ppp11")),
        ]
        for subgroup_size in (64, 32):
            res = _run_swizzle_batch(
                f"cs_w64_swz_exec_sg{subgroup_size}", specs, init_vals, 64, subgroup_size, exec_mask=exec_asym,
            )
            for idx, (mod_str, off16) in enumerate(specs):
                with self.subTest(subgroup_size=subgroup_size, pattern=mod_str):
                    self.assertEqual(res[idx], ref_swizzle(init_vals, old_vals, off16, exec_asym))

        # 5-stage butterfly reduction in wave64 produces independent sums for [0..31] and [32..63].
        sum_lo = sum(init_vals[:32])
        sum_hi = sum(init_vals[32:])
        for subgroup_size in (64, 32):
            with self.subTest(butterfly_subgroup=subgroup_size):
                got = run_cs(f"cs_w64_swz_butterfly_sg{subgroup_size}", [
                    "s_mov_b64 exec, -1",
                    "v_lshlrev_b32_e32 v20, 2, v0",
                    f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
                    "buffer_load_dword v1, v20, s[0:3], 0 offen",
                    "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v1, v2",
                    "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v1, v2",
                    "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,4)",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v1, v2",
                    "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v1, v2",
                    "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,16)",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v1, v2",
                    "v_lshlrev_b32_e32 v21, 2, v0",
                    "buffer_store_dword v1, v21, s[0:3], 0 offen",
                ], 64, subgroup_size, init_inputs=init_vals)
                self.assertEqual(got, [sum_lo] * 32 + [sum_hi] * 32)

    def test_wave64_partial_wave_40_lanes(self):
        # 40 threads on a 64-wide subgroup: [0..31] complete, [32..39] complete 8-lane block, [40..63] missing.
        exec_low32 = 0x00000000FFFFFFFF
        got = run_cs("cs_w64_swz_partial40", [
            "s_mov_b64 exec, -1",
            "v_add_nc_u32_e32 v1, 200, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            "ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,8)",
            *exec_mask_lines(exec_low32, 64),
            "ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,16)",
            "s_waitcnt lgkmcnt(0)",
            *restore_exec_lines(64),
            *store_slots_lines(["v2", "v3"], 40),
        ], 40, 64, out_slots=2)
        vals64 = [200 + l for l in range(40)] + [0] * 24
        old64 = [OLD_SENTINEL | l for l in range(40)] + [0] * 24
        self.assertEqual(got[:40], ref_swizzle(vals64, old64, encode_reverse(8), ALL64)[:40])
        self.assertEqual(got[40:], ref_swizzle(vals64, old64, encode_swap(16), exec_low32)[:40])


class PixelShaderWaveInterpTests(unittest.TestCase):
    """End-to-end 32-lane pixel shader execution via `spirv_wave_interp.run_wave`."""

    def test_ps_bitmask_quad_and_broadcast_via_wave_interp(self):
        # Hand-written PS mirroring runtime/test/shaders/ps_bitmask_perm.asm plus QUAD_PERM & BROADCAST.
        spv, _ = translate_ps("ps_swz_wave32", [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            'ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,"pippi")',
            'ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM,"10p0i")',
            "ds_swizzle_b32 v3, v0 offset:swizzle(QUAD_PERM,1,2,3,0)",
            "s_waitcnt lgkmcnt(0)",
            "exp mrt0 v0, v1, v2, v3 done vm",
        ])
        res = wi.run_wave(spv, wave=32)
        for lane in range(32):
            got = res["outputs"][lane]["%out_mrt0"]
            want = (
                lane,
                lane ^ 9,
                ((lane & 5) | 16) ^ 1,
                (lane & ~3) | ((lane + 1) & 3),
            )
            self.assertEqual(got, want, f"lane {lane}: got {got}, want {want}")


class EmulationAndParsingTests(unittest.TestCase):
    """Structural SPIR-V checks for `--wave64-emulate`; unparsed offsets raise Unsupported."""

    def test_wave64_emulate_lowers_all_five_modes_via_x_buf_without_subgroup_ops(self):
        cases = [
            ("quad", "swizzle(QUAD_PERM,3,2,1,0)", r"%cfffffffc\b"),
            ("bitmask", 'swizzle(BITMASK_PERM,"10p0i")', r"%cffffffe5\b"),
            ("swap", "swizzle(SWAP,16)", r"OpBitwiseXor %uint %\w+ %c10\b"),
            ("reverse", "swizzle(REVERSE,32)", r"OpBitwiseXor %uint %\w+ %c1f\b"),
            ("broadcast", "swizzle(BROADCAST,16,5)", r"%cfffffff0\b"),
        ]
        for label, mod_str, expected_pat in cases:
            with self.subTest(mode=label):
                lines = [
                    "s_mov_b64 exec, -1",
                    "v_add_nc_u32_e32 v1, 100, v0",
                    f"ds_swizzle_b32 v2, v1 offset:{mod_str}",
                    "s_waitcnt lgkmcnt(0)",
                    "v_lshlrev_b32_e32 v21, 2, v0",
                    "buffer_store_dword v2, v21, s[0:3], 0 offen",
                ]
                spv, gen = translate_cs(f"cs_w64_emu_swz_{label}", lines, 64, emulate=True)
                self.assertTrue(gen.wave64 and gen.emulate)
                self.assertNotIn("OpGroupNonUniform", spv)
                self.assertNotIn("%sg_lane", spv)
                self.assertIn("%x_buf = OpVariable", spv)
                self.assertGreaterEqual(spv.count("OpControlBarrier"), 2)
                self.assertRegex(spv, expected_pat)
                # spirv_mini_interp.py deliberately rejects the Workgroup-memory round loop.
                with self.assertRaises(si.UnsupportedConstruct):
                    si.run_workgroup(
                        spv, 64, 32,
                        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32},
                        push_constants=VSHARP + [0] * 28, memory={},
                    )

        # Wave32 shaders are unaffected by WAVE64_EMULATE = True.
        w32_lines = [
            "ds_swizzle_b32 v2, v0 offset:swizzle(SWAP,8)",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        spv_plain, _ = translate_cs("cs_w32_emu_swz_nop", w32_lines, 32, emulate=False)
        spv_emu, gen_emu = translate_cs("cs_w32_emu_swz_nop", w32_lines, 32, emulate=True)
        self.assertFalse(gen_emu.wave64 or gen_emu.emulate)
        self.assertEqual(spv_plain, spv_emu)

    def test_raw_numeric_offsets_and_unknown_swizzle_kind_raise_unsupported(self):
        # A numeric offset in the bitmask or QUAD_PERM range is decoded since #4177; 0xC000..0xFFFF is a separate mode
        # on RDNA2 (measured on gfx1036 for all 16,384 offsets, docs/design.md "ds_swizzle_b32"), so it and an unknown
        # kind are refused, not guessed.
        for bad_mod in ("offset:0xc000", "offset:0xffff", "offset:swizzle(UNKNOWN,1)"):
            with self.subTest(mod=bad_mod):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_cs("cs_w32_swz_bad", [
                        f"ds_swizzle_b32 v2, v0 {bad_mod}",
                        "s_waitcnt lgkmcnt(0)",
                        "v_lshlrev_b32_e32 v21, 2, v0",
                        "buffer_store_dword v2, v21, s[0:3], 0 offen",
                    ], 32)

class NegativeControls(unittest.TestCase):
    """Sabotage translated SPIR-V like a translator regression; multi-lane execution must catch it."""

    def _run(self, spv: str, threads: int, subgroup_size: int) -> list[int]:
        mem = _Heap()
        si.run_workgroup(
            spv, threads, subgroup_size,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        return [mem.get(4 * k) for k in range(threads)]

    def test_sabotage_bitmask_upper_half_mask_dropped_in_wave64(self):
        # Dropping 0xFFFFFFE0 -> 0 causes upper half [32..63] to read from lower half [0..31].
        spv, _ = translate_cs("cs_w64_neg_bm", [
            "s_mov_b64 exec, -1",
            "v_add_nc_u32_e32 v1, 100, v0",
            'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"00000")',
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ], 64)
        want = [100] * 32 + [132] * 32
        self.assertEqual(self._run(spv, 64, 64), want)

        bad, n = re.subn(r"OpBitwiseAnd %uint (%\w+) %cffffffe0\b", r"OpBitwiseAnd %uint \1 %c0", spv)
        self.assertEqual(n, 1)
        got_bad = self._run(bad, 64, 64)
        self.assertNotEqual(got_bad, want)
        self.assertEqual(got_bad, [100] * 64)

    def test_sabotage_quad_base_mask_shuffle_and_exec_guard(self):
        spv, _ = translate_cs("cs_w32_neg_quad", [
            "v_add_nc_u32_e32 v1, 300, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            "s_mov_b32 exec_lo, 0x0f0f0f0f",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ], 32)
        want = ref_swizzle([300 + l for l in range(32)], [OLD_SENTINEL | l for l in range(32)],
                           encode_quad_perm(3, 2, 1, 0), 0x0F0F0F0F)
        self.assertEqual(self._run(spv, 32, 32), want)

        # 1. Dropping quad base mask 0xFFFFFFFC sends every quad to quad 0.
        lane = re.search(r"OpBitwiseAnd %uint (%\w+) %c3\b", spv).group(1)   # sel = lane & 3; base = lane & ~3
        bad_base, n1 = re.subn(rf"OpBitwiseAnd %uint {lane} %cfffffffc\b", rf"OpBitwiseAnd %uint {lane} %c0", spv)
        self.assertEqual(n1, 1)
        self.assertNotEqual(self._run(bad_base, 32, 32), want)

        # 2. Replacing OpGroupNonUniformShuffle with OpCopyObject turns swizzle into own-lane copy.
        bad_shuf, n2 = re.subn(
            r"(%\w+) = OpGroupNonUniformShuffle %uint %c3 (%\w+) %\w+",
            r"\1 = OpCopyObject %uint \2",
            spv,
        )
        self.assertEqual(n2, 1)
        self.assertNotEqual(self._run(bad_shuf, 32, 32), want)

        # 3. Making the destination EXEC mask all ones (the write after the shuffle) overwrites inactive quads.
        at = spv.index("OpGroupNonUniformShuffle %uint %c3")
        tail, n3 = re.subn(
            r"(%\w+) = OpSelect %uint %\w+ %cffffffff %c0\n(\s*)(%\w+) = OpNot %uint \1\b",
            r"\1 = OpCopyObject %uint %cffffffff\n\2\3 = OpNot %uint \1",
            spv[at:],
            count=1,
        )
        bad_exec = spv[:at] + tail
        self.assertEqual(n3, 1)
        got_exec = self._run(bad_exec, 32, 32)
        self.assertNotEqual(got_exec, want)
        self.assertNotEqual(got_exec[4], OLD_SENTINEL | 4)


if __name__ == "__main__":
    unittest.main()
