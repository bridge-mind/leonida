"""Semantic value and translation tests for RDNA2 cross-lane permutes:
  v_permlane16_b32   D = vS[(l & ~15) | sel[l & 15]]
  v_permlanex16_b32  D = vS[((l & ~15) ^ 16) | sel[l & 15]]

where the 4-bit lane selector nibbles `sel[0..7]` come from `src1` (bits `[4*k+3 : 4*k]`) and
`sel[8..15]` come from `src2` (bits `[4*(k-8)+3 : 4*(k-8)]`), applied independently inside each
32-lane row (`[0..15]` and `[16..31]`, and in wave64 `[32..47]` and `[48..63]`). Only lanes active
in `EXEC` write `vD`; inactive destination lanes preserve their prior `vD`.

Executed end-to-end through `tools/rdna2spirv.py` and `tests/semantic/spirv_mini_interp.py`
(`run_workgroup`) across 32 lanes (wave32) and 64 lanes (wave64 on both one 64-wide subgroup and
two 32-wide subgroups), as well as partial waves (20 and 40 lanes), and checked against
hand-computed constants and an independent plain-Python RDNA2 ISA reference model. Standard library
only; zero game data; hand-written RDNA2 disassembly listings only.

Documented findings (observed in `tools/rdna2spirv.py`; no shared files modified):
  1. `op_sel:[fi, bound_ctrl]` and inactive source lanes (`tools/rdna2spirv.py` lines 4608-4632):
     In RDNA2 ISA (measured on RDNA2 gfx1036 and RDNA3.5 gfx1150; see `docs/shader_validation.md`
     and `tools/crosslane_fuzz.py`), when a source lane is inactive in `EXEC` (`EXEC[src] == 0`):
       - `fi=1` (`op_sel:[1,0]`, `op_sel:[1,1]`): reads the inactive source lane's pre-existing `vS`.
       - `fi=0, bound_ctrl=0` (`op_sel:[0,0]` or omitted): keeps the destination lane's prior `vD`.
       - `fi=0, bound_ctrl=1` (`op_sel:[0,1]`): writes `0` to the destination lane.
     `tools/rdna2spirv.py` lines 4608-4632 do not inspect `mods` (`op_sel:[fi, bound_ctrl]`) and emit
     `self.g_shuffle(r, v, src)` unconditionally without checking the source lane's `%active` bit.
     Therefore:
       - When all accessed source lanes are active in `EXEC`, all four `op_sel` modes (`[0,0]`,
         `[1,0]`, `[0,1]`, `[1,1]`) and omitted `op_sel` match the RDNA2 ISA.
       - When a source lane is inactive in `EXEC` and `fi=1` (`op_sel:[1,0]`, `op_sel:[1,1]`), the
         translation matches the RDNA2 ISA (reading the inactive source lane's pre-existing `vS`).
       - When a source lane is inactive in `EXEC` and `fi=0` (`op_sel:[0,0]`/omitted,
         `op_sel:[0,1]`), the translator currently reads `vS[src]` instead of keeping `vD` or
         writing `0`. Those inactive-source `fi=0` cases are tested in the plain-Python ISA
         reference tests (`OracleTests`) and excluded from passing end-to-end assertions rather
         than relaxing expected values; structural SPIR-V translation is verified in
         `EmulationAndFindingTests`.
  2. `--wave64-emulate` (`tools/rdna2spirv.py` lines 2671-2810, 2890-2895, 4608-4632):
     With `WAVE64_EMULATE = True`, wave64 compute/mesh shaders lower `g_shuffle` and `sub_lane`
     through `%x_buf` Workgroup-memory exchanges, `OpControlBarrier`, and a round loop (`OpLoopMerge`
     with a backward branch). `tests/semantic/spirv_mini_interp.py` rejects `Workgroup` barriers and
     non-forward branches (`UnsupportedConstruct`), so `--wave64-emulate` is covered by translation
     and structural SPIR-V invariants here (and rejected cleanly by `run_workgroup`).
  3. Wave64 native on two 32-wide subgroups (`subgroup_size = 32`):
     Because `v_permlane16_b32` and `v_permlanex16_b32` never cross the 32-lane row boundary
     (`0..31` vs `32..63`), both instructions stay exact across all 64 lanes even on two 32-wide
     subgroups without `--wave64-emulate`.
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

MASK32 = 0xFFFFFFFF
ALL32 = 0xFFFFFFFF
ALL64 = 0xFFFFFFFFFFFFFFFF
OLD_SENTINEL = 0xCAFE0000
HEAP_FILL = 0xEEEEEEEE
# V# in s[0:3]: base 0, stride 0, num_records 0x10000 bytes (OOB_SELECT = 3).
VSHARP = [0, 0, 0x10000, 0x31016FAC]
INPUT_BASE = 0x8000

ID_S1 = 0x76543210
ID_S2 = 0xFEDCBA98
REV8_S1 = 0x01234567
REV8_S2 = 0x89ABCDEF
REV16_S1 = 0x89ABCDEF
REV16_S2 = 0x01234567
PAIR_SWAP_S1 = 0x10325476
PAIR_SWAP_S2 = 0x98BADCFE
ASYM_S1 = 0x3A0F5C91
ASYM_S2 = 0xE27B48D6


def ref_permlane(x16: bool, src_vals: list[int], dst_old: list[int], s1: int, s2: int,
                 exec_mask: int, fi: int = 0, bound_ctrl: int = 0) -> list[int]:
    """Independent RDNA2 ISA reference model for v_permlane16_b32 / v_permlanex16_b32."""
    lanes = len(src_vals)
    assert len(dst_old) == lanes and lanes in (32, 64)
    out = []
    for lane in range(lanes):
        if not ((exec_mask >> lane) & 1):
            out.append(dst_old[lane] & MASK32)
            continue
        within = lane & 15
        sel_word = s2 if within >= 8 else s1
        nib = (sel_word >> (4 * (lane & 7))) & 0xF
        half = lane & ~15
        row = (half ^ 16) if x16 else half
        src_lane = row | nib
        if ((exec_mask >> src_lane) & 1) or fi:
            out.append(src_vals[src_lane] & MASK32)
        elif bound_ctrl:
            out.append(0)
        else:
            out.append(dst_old[lane] & MASK32)
    return out


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
    with open(src_path, "w", encoding="utf-8", newline="") as f:
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
    """Emit buffer_store_dword for each VGPR in `regs` at `4 * (slot * threads + v0)`."""
    lines = ["v_lshlrev_b32_e32 v21, 2, v0"]
    for slot, reg in enumerate(regs):
        if slot == 0:
            lines.append(f"buffer_store_dword {reg}, v21, s[0:3], 0 offen")
        else:
            lines.append(f"buffer_store_dword {reg}, v21, s[0:3], 0 offen offset:{4 * slot * threads}")
    return lines


class OracleTests(unittest.TestCase):
    """Cross-check hand-worked constants and ISA semantics against ref_permlane."""

    def test_hand_worked_identity_and_half_swap(self):
        vals32 = [100 + l for l in range(32)]
        old32 = [OLD_SENTINEL | l for l in range(32)]
        # v_permlane16_b32 identity leaves all 32 lanes unchanged.
        self.assertEqual(ref_permlane(False, vals32, old32, ID_S1, ID_S2, ALL32), vals32)
        # v_permlanex16_b32 swaps [0..15] and [16..31].
        self.assertEqual(
            ref_permlane(True, vals32, old32, ID_S1, ID_S2, ALL32),
            [100 + (l ^ 16) for l in range(32)],
        )

        # Wave64 swaps within each 32-lane row: [0..15]<->[16..31] and [32..47]<->[48..63].
        vals64 = [500 + l for l in range(64)]
        old64 = [OLD_SENTINEL | l for l in range(64)]
        got64 = ref_permlane(True, vals64, old64, ID_S1, ID_S2, ALL64)
        self.assertEqual((got64[0], got64[15], got64[16], got64[31]), (516, 531, 500, 515))
        self.assertEqual((got64[32], got64[47], got64[48], got64[63]), (548, 563, 532, 547))

    def test_hand_worked_nibble_patterns_and_exec_opsel_rules(self):
        vals32 = [1000 + l for l in range(32)]
        old32 = [OLD_SENTINEL | l for l in range(32)]

        # Broadcast 0 from s1 (lanes 0..7) and 15 from s2 (lanes 8..15).
        got_bcast = ref_permlane(False, vals32, old32, 0x00000000, 0xFFFFFFFF, ALL32)
        self.assertEqual(got_bcast[0:8], [1000] * 8)
        self.assertEqual(got_bcast[8:16], [1015] * 8)
        self.assertEqual(got_bcast[16:24], [1016] * 8)
        self.assertEqual(got_bcast[24:32], [1031] * 8)

        # Full 16-lane reversal: s1=0x89abcdef (lanes 0..7 -> 15..8), s2=0x01234567 (lanes 8..15 -> 7..0).
        got_rev16 = ref_permlane(False, vals32, old32, REV16_S1, REV16_S2, ALL32)
        self.assertEqual(got_rev16[0:16], [1000 + (15 - l) for l in range(16)])
        self.assertEqual(got_rev16[16:32], [1016 + (15 - l) for l in range(16)])

        # Sub-half 8-lane reversal: s1=0x01234567 (0..7 -> 7..0), s2=0x89abcdef (8..15 -> 15..8).
        got_rev8 = ref_permlane(True, vals32, old32, REV8_S1, REV8_S2, ALL32)
        self.assertEqual(got_rev8[0:8], [1016 + (7 - l) for l in range(8)])
        self.assertEqual(got_rev8[8:16], [1016 + (15 - l) for l in range(8)])

        # op_sel:[fi, bound_ctrl] when lane 0 is INACTIVE in EXEC (0xFFFFFFFE):
        # lanes 1..15 broadcast local lane 0 (inactive in half 0!), lanes 16..31 broadcast local lane 16 (active!).
        exec_no_0 = ALL32 ^ 1
        # fi=0, bound_ctrl=0 -> lanes 1..15 keep old32, lane 0 keeps old32[0], lanes 16..31 read 1016.
        r00 = ref_permlane(False, vals32, old32, 0, 0, exec_no_0, fi=0, bound_ctrl=0)
        self.assertEqual(r00[0], old32[0])
        self.assertEqual(r00[1:16], old32[1:16])
        self.assertEqual(r00[16:32], [1016] * 16)

        # fi=0, bound_ctrl=1 -> lanes 1..15 write 0, lane 0 keeps old32[0], lanes 16..31 read 1016.
        r01 = ref_permlane(False, vals32, old32, 0, 0, exec_no_0, fi=0, bound_ctrl=1)
        self.assertEqual(r01[0], old32[0])
        self.assertEqual(r01[1:16], [0] * 15)
        self.assertEqual(r01[16:32], [1016] * 16)

        # fi=1, bound_ctrl=0 or 1 -> lanes 1..15 read inactive lane 0's vals32[0], lane 0 keeps old32[0].
        for bc in (0, 1):
            r1x = ref_permlane(False, vals32, old32, 0, 0, exec_no_0, fi=1, bound_ctrl=bc)
            self.assertEqual(r1x[0], old32[0])
            self.assertEqual(r1x[1:16], [1000] * 15)
            self.assertEqual(r1x[16:32], [1016] * 16)


class Wave32ExecutionTests(unittest.TestCase):
    """End-to-end 32-lane execution for v_permlane16_b32 and v_permlanex16_b32."""

    def _run_batch(self, specs: list[tuple[bool, int, int, str, str]], init_vals: list[int],
                   old_vals: list[int], exec_mask: int = ALL32) -> list[list[int]]:
        """Run a list of (x16, s1, s2, src_form, extra_mods) batched into one shader."""
        lines = [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
        ]
        regs = []
        for idx, _ in enumerate(specs):
            dst = f"v{2 + idx}"
            regs.append(dst)
            lines.append(f"v_or_b32_e32 {dst}, {OLD_SENTINEL:#x}, v0")
        lines.extend(exec_mask_lines(exec_mask, 32))
        for idx, (x16, s1, s2, src_expr, extra) in enumerate(specs):
            dst = f"v{2 + idx}"
            op = "v_permlanex16_b32" if x16 else "v_permlane16_b32"
            if src_expr == "sgpr":
                lines += [f"s_mov_b32 s4, {s1:#x}", f"s_mov_b32 s5, {s2:#x}",
                          f"{op} {dst}, v1, s4, s5{extra}"]
            elif src_expr == "lit_sgpr":
                lines += [f"s_mov_b32 s5, {s2:#x}", f"{op} {dst}, v1, {s1:#x}, s5{extra}"]
            elif src_expr == "sgpr_lit":
                lines += [f"s_mov_b32 s4, {s1:#x}", f"{op} {dst}, v1, s4, {s2:#x}{extra}"]
            elif src_expr == "inline":
                lines.append(f"{op} {dst}, v1, {s1}, {s2}{extra}")
            elif src_expr == "same_sgpr":
                assert s1 == s2
                lines += [f"s_mov_b32 s4, {s1:#x}", f"{op} {dst}, v1, s4, s4{extra}"]
            elif src_expr == "e64_sgpr":
                lines += [f"s_mov_b32 s4, {s1:#x}", f"s_mov_b32 s5, {s2:#x}",
                          f"{op}_e64 {dst}, v1, s4, s5{extra}"]
            else:
                raise ValueError(src_expr)
        lines.extend(restore_exec_lines(32))
        lines.extend(store_slots_lines(regs, 32))
        flat = run_cs("cs_w32_permlane_batch", lines, 32, 32, init_inputs=init_vals, out_slots=len(specs))
        return [flat[slot * 32:(slot + 1) * 32] for slot in range(len(specs))]

    def test_half_isolation_vs_cross_half_swap_and_nibble_selectors(self):
        init_vals = [(0x10000000 + l * 0x010101) & MASK32 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        specs = [
            # Identity (SGPR+SGPR, literal+SGPR, SGPR+literal, _e64)
            (False, ID_S1, ID_S2, "sgpr", ""),
            (True, ID_S1, ID_S2, "sgpr", ""),
            (False, ID_S1, ID_S2, "lit_sgpr", ""),
            (True, ID_S1, ID_S2, "sgpr_lit", ""),
            (True, ID_S1, ID_S2, "e64_sgpr", ""),
            # Broadcasts with inline constants (0, -1, 15, -16) and same-SGPR
            (False, 0, 0, "inline", ""),
            (True, 0, 0, "inline", ""),
            (False, -1, -1, "inline", ""),
            (True, 0, -1, "inline", ""),
            (False, -16, 15, "inline", ""),          # -16 = 0xfffffff0, 15 = 0x0000000f
            (True, 0x55555555, 0x55555555, "same_sgpr", ""),
            # 8-lane reversal vs full 16-lane reversal
            (False, REV8_S1, REV8_S2, "sgpr", ""),
            (True, REV8_S1, REV8_S2, "sgpr", ""),
            (False, REV16_S1, REV16_S2, "sgpr", ""),
            (True, REV16_S1, REV16_S2, "sgpr", ""),
            # Pairwise swap & asymmetric non-bijective permutation
            (False, PAIR_SWAP_S1, PAIR_SWAP_S2, "sgpr", ""),
            (True, ASYM_S1, ASYM_S2, "sgpr", ""),
        ]
        results = self._run_batch(specs, init_vals, old_vals)
        for idx, (x16, s1, s2, form, _) in enumerate(specs):
            s1_u = s1 & MASK32
            s2_u = s2 & MASK32
            want = ref_permlane(x16, init_vals, old_vals, s1_u, s2_u, ALL32)
            with self.subTest(idx=idx, x16=x16, form=form, s1=hex(s1_u), s2=hex(s2_u)):
                self.assertEqual(results[idx], want)

        # Spot-check half-staying vs half-swapping explicitly:
        self.assertEqual(results[0], init_vals)
        self.assertEqual(results[1], init_vals[16:32] + init_vals[0:16])

    def test_all_16_nibble_broadcasts_for_both_ops(self):
        init_vals = [0x9000 + l * 17 for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        specs = []
        for nib in range(16):
            sel = 0x11111111 * nib
            specs.append((False, sel, sel, "same_sgpr", ""))
            specs.append((True, sel, sel, "same_sgpr", ""))
        # Batch 16 specs at a time to stay well within VGPR file (v2..v17).
        for base in (0, 16):
            chunk = specs[base:base + 16]
            results = self._run_batch(chunk, init_vals, old_vals)
            for idx, (x16, s1, s2, _, _) in enumerate(chunk):
                want = ref_permlane(x16, init_vals, old_vals, s1, s2, ALL32)
                with self.subTest(x16=x16, nib=s1 & 0xF):
                    self.assertEqual(results[idx], want)

    def test_butterfly_32lane_sum_reduction(self):
        # Real compiler idiom: xor 1, 2, 4, 8 via v_permlane16_b32 then xor 16 via v_permlanex16_b32.
        init_vals = [3 * l + 5 for l in range(32)]
        expected_sum = sum(init_vals)
        got = run_cs("cs_w32_permlane_butterfly", [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            # xor 1: s1=0x67452301, s2=0xefcdab89
            "s_mov_b32 s5, 0xefcdab89",
            "v_permlane16_b32 v2, v1, 0x67452301, s5",
            "v_add_nc_u32_e32 v1, v1, v2",
            # xor 2: s1=0x54761032, s2=0xdcfe98ba
            "s_mov_b32 s5, 0xdcfe98ba",
            "v_permlane16_b32 v2, v1, 0x54761032, s5",
            "v_add_nc_u32_e32 v1, v1, v2",
            # xor 4: s1=0x32107654, s2=0xba98fedc
            "s_mov_b32 s5, 0xba98fedc",
            "v_permlane16_b32 v2, v1, 0x32107654, s5",
            "v_add_nc_u32_e32 v1, v1, v2",
            # xor 8: s1=0xfedcba98, s2=0x76543210
            "s_mov_b32 s5, 0x76543210",
            "v_permlane16_b32 v2, v1, 0xfedcba98, s5",
            "v_add_nc_u32_e32 v1, v1, v2",
            # xor 16: identity nibbles via v_permlanex16_b32
            "s_mov_b32 s4, 0x76543210",
            "s_mov_b32 s5, 0xfedcba98",
            "v_permlanex16_b32 v2, v1, s4, s5",
            "v_add_nc_u32_e32 v1, v1, v2",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v1, v21, s[0:3], 0 offen",
        ], 32, 32, init_inputs=init_vals)
        self.assertEqual(got, [expected_sum] * 32)

    def test_exec_destination_masking_and_all_op_sel_with_active_sources(self):
        # Active lanes in each 16-lane half are {0, 2, 4, 6, 8, 9, 10, 11};
        # selectors ONLY read from active lanes {0, 2, 4, 6, 8, 9, 10, 11}, so all 4 op_sel modes match ISA.
        exec_mask = 0x0F550F55
        s1_sel = 0x64206420  # nibbles in {0, 2, 4, 6} (all active)
        s2_sel = 0xBA98BA98  # nibbles in {8, 9, 10, 11} (all active)
        init_vals = [0x100 + l for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        specs = []
        for x16 in (False, True):
            for mods in ("", " op_sel:[0,0]", " op_sel:[1,0]", " op_sel:[0,1]", " op_sel:[1,1]"):
                specs.append((x16, s1_sel, s2_sel, "sgpr", mods))
        results = self._run_batch(specs, init_vals, old_vals, exec_mask=exec_mask)
        for idx, (x16, s1, s2, _, mods) in enumerate(specs):
            want = ref_permlane(x16, init_vals, old_vals, s1, s2, exec_mask)
            with self.subTest(x16=x16, mods=mods or "default"):
                self.assertEqual(results[idx], want)

    def test_op_sel_fi1_reads_inactive_source_lanes(self):
        # Lanes 0, 16 are OFF in EXEC; active lanes modify v1 to 9999 AFTER pre-filling v1,
        # then v_permlane16_b32 / v_permlanex16_b32 with op_sel:[1,0] and op_sel:[1,1] broadcast
        # local lane 0 (inactive lane 0 / inactive lane 16!). Active lanes must see the inactive
        # lanes' pre-existing values (2000 and 2016), while inactive lanes 0 and 16 keep OLD.
        exec_mask = ALL32 ^ ((1 << 0) | (1 << 16))
        init_vals = [2000 + l for l in range(32)]
        old_vals = [OLD_SENTINEL | l for l in range(32)]
        updated_vals = [init_vals[l] if not ((exec_mask >> l) & 1) else 9999 for l in range(32)]

        lines = [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v4, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v5, {OLD_SENTINEL:#x}, v0",
            *exec_mask_lines(exec_mask, 32),
            "v_mov_b32_e32 v1, 9999",  # inactive lanes 0 & 16 keep 2000 & 2016
            "v_permlane16_b32 v2, v1, 0, 0 op_sel:[1,0]",
            "v_permlane16_b32 v3, v1, 0, 0 op_sel:[1,1]",
            "v_permlanex16_b32 v4, v1, 0, 0 op_sel:[1,0]",
            "v_permlanex16_b32 v5, v1, 0, 0 op_sel:[1,1]",
            *restore_exec_lines(32),
            *store_slots_lines(["v2", "v3", "v4", "v5"], 32),
        ]
        flat = run_cs("cs_w32_permlane_fi1", lines, 32, 32, init_inputs=init_vals, out_slots=4)
        for slot, x16 in enumerate((False, False, True, True)):
            got = flat[slot * 32:(slot + 1) * 32]
            want = ref_permlane(x16, updated_vals, old_vals, 0, 0, exec_mask, fi=1, bound_ctrl=slot & 1)
            self.assertEqual(got, want)
        self.assertEqual(flat[0], OLD_SENTINEL | 0)
        self.assertEqual(flat[1], 2000)
        self.assertEqual(flat[16], OLD_SENTINEL | 16)
        self.assertEqual(flat[17], 2016)
        # Slot 2 (permlanex16 op_sel:[1,0]): active lanes 1..15 read inactive lane 16 (2016).
        self.assertEqual(flat[2 * 32 + 1], 2016)
        self.assertEqual(flat[2 * 32 + 17], 2000)

    def test_destination_aliasing_v1_and_v0(self):
        init_vals = [300 + l for l in range(32)]
        # 1. Alias v1, v1 (with EXEC mask & fi=1 reading inactive lane 0 before overwrite)
        exec_mask = ALL32 ^ (1 << 0)
        got_v1 = run_cs("cs_w32_permlane_alias_v1", [
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            "v_mov_b32_e32 v2, v1",
            *exec_mask_lines(exec_mask, 32),
            "s_mov_b32 s4, 0x00000000",
            "s_mov_b32 s5, 0x00000000",
            "v_permlane16_b32 v1, v1, s4, s5 op_sel:[1,0]",
            "v_permlanex16_b32 v2, v2, 0x76543210, 0xfedcba98 op_sel:[1,0]",
            *restore_exec_lines(32),
            *store_slots_lines(["v1", "v2"], 32),
        ], 32, 32, init_inputs=init_vals, out_slots=2)
        self.assertEqual(got_v1[:32], ref_permlane(False, init_vals, init_vals, 0, 0, exec_mask, fi=1))
        self.assertEqual(got_v1[32:], ref_permlane(True, init_vals, init_vals, ID_S1, ID_S2, exec_mask, fi=1))

        # 2. Alias v0, v0 directly (save output byte offset 4 * v0 in v21 first)
        got_v0_16 = run_cs("cs_w32_permlane16_alias_v0", [
            "v_lshlrev_b32_e32 v21, 2, v0",
            f"s_mov_b32 s4, {REV16_S1:#x}",
            f"s_mov_b32 s5, {REV16_S2:#x}",
            "v_permlane16_b32 v0, v0, s4, s5",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ], 32, 32)
        self.assertEqual(got_v0_16, ref_permlane(False, list(range(32)), list(range(32)), REV16_S1, REV16_S2, ALL32))

        got_v0_x16 = run_cs("cs_w32_permlanex16_alias_v0", [
            "v_lshlrev_b32_e32 v21, 2, v0",
            f"s_mov_b32 s4, {REV16_S1:#x}",
            f"s_mov_b32 s5, {REV16_S2:#x}",
            "v_permlanex16_b32 v0, v0, s4, s5",
            "buffer_store_dword v0, v21, s[0:3], 0 offen",
        ], 32, 32)
        self.assertEqual(got_v0_x16, ref_permlane(True, list(range(32)), list(range(32)), REV16_S1, REV16_S2, ALL32))

    def test_partial_wave_20_lanes_avoids_missing_lanes(self):
        # 20 threads on a 32-wide subgroup: lanes 20..31 do not exist.
        # 1. All 20 existing lanes active (0xFFFFF): selectors in {0..3} read existing active lanes 0..3 / 16..19.
        exec_20 = 0xFFFFF
        got_all20 = run_cs("cs_w32_partial20_all", [
            "v_add_nc_u32_e32 v1, 700, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            *exec_mask_lines(exec_20, 32),
            "v_permlane16_b32 v2, v1, 0x32103210, 0x01230123",
            "v_permlanex16_b32 v3, v1, 0x32103210, 0x01230123",
            *restore_exec_lines(32),
            *store_slots_lines(["v2", "v3"], 20),
        ], 20, 32, out_slots=2)
        vals_padded = [700 + l for l in range(20)] + [0] * 12
        old_padded = [OLD_SENTINEL | l for l in range(20)] + [0] * 12
        self.assertEqual(got_all20[:20], ref_permlane(False, vals_padded, old_padded, 0x32103210, 0x01230123, exec_20)[:20])
        self.assertEqual(got_all20[20:40], ref_permlane(True, vals_padded, old_padded, 0x32103210, 0x01230123, exec_20)[:20])

        # 2. EXEC = 0xFFFF (lanes 16..19 inactive): permlane16 reads [0..15] and preserves OLD on 16..19;
        # permlanex16 op_sel:[1,0] (fi=1) reads inactive lanes 16..19 while inactive lanes 16..19 discard UNDEF
        # shuffles from missing lanes 20..31 and keep OLD.
        exec_16 = 0xFFFF
        got_masked = run_cs("cs_w32_partial20_masked", [
            "v_add_nc_u32_e32 v1, 700, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            *exec_mask_lines(exec_16, 32),
            "v_permlane16_b32 v2, v1, 0x01234567, 0x89abcdef",
            "v_permlanex16_b32 v3, v1, 0x32103210, 0x01230123 op_sel:[1,0]",
            *restore_exec_lines(32),
            *store_slots_lines(["v2", "v3"], 20),
        ], 20, 32, out_slots=2)
        self.assertEqual(got_masked[:20], ref_permlane(False, vals_padded, old_padded, REV8_S1, REV8_S2, exec_16)[:20])
        self.assertEqual(got_masked[20:40], ref_permlane(True, vals_padded, old_padded, 0x32103210, 0x01230123, exec_16, fi=1)[:20])


class Wave64ExecutionTests(unittest.TestCase):
    """End-to-end 64-lane execution across one 64-wide subgroup AND two 32-wide subgroups."""

    def test_wave64_halves_selectors_exec_and_fi1_on_subgroup_64_and_32(self):
        init_vals = [0x4000 + l * 31 for l in range(64)]
        old_vals = [OLD_SENTINEL | l for l in range(64)]
        # Asymmetric 64-bit EXEC: active lanes in each 16-lane quarter are {0, 2, 4, 6, 8, 9, 10, 11}.
        exec_active_only = 0x0F550F550F550F55
        # Inactive source lane 0 and lane 32 (one in each 32-lane row) for fi=1 test.
        exec_no_0_32 = ALL64 ^ ((1 << 0) | (1 << 32))
        updated_fi1 = [init_vals[l] if not ((exec_no_0_32 >> l) & 1) else (8800 + l) for l in range(64)]

        lines = [
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {INPUT_BASE:#x}, v20",
            "buffer_load_dword v1, v20, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, v1",
            # Slot 0: v_permlane16_b32 identity (stays in [0..15], [16..31], [32..47], [48..63])
            f"s_mov_b32 s4, {ID_S1:#x}",
            f"s_mov_b32 s5, {ID_S2:#x}",
            "v_permlane16_b32 v2, v1, s4, s5",
            # Slot 1: v_permlanex16_b32 identity (swaps [0..15]<->[16..31] and [32..47]<->[48..63])
            "v_permlanex16_b32 v3, v1, s4, s5",
            # Slot 2: v_permlane16_b32 asymmetric non-bijective selectors
            f"s_mov_b32 s6, {ASYM_S1:#x}",
            f"s_mov_b32 s7, {ASYM_S2:#x}",
            "v_permlane16_b32 v4, v1, s6, s7",
            # Slot 3: v_permlanex16_b32_e64 full 16-lane reversal
            f"s_mov_b32 s8, {REV16_S2:#x}",
            f"v_permlanex16_b32_e64 v5, v1, {REV16_S1:#x}, s8",
            # Slot 4: 64-bit EXEC destination masking with active sources (v6 pre-loaded with OLD)
            f"v_or_b32_e32 v6, {OLD_SENTINEL:#x}, v0",
            "s_mov_b32 s9, 0xba98ba98",
            *exec_mask_lines(exec_active_only, 64),
            "v_permlanex16_b32 v6, v1, 0x64206420, s9",
            *restore_exec_lines(64),
            # Slot 5: fi=1 reading inactive source lane 0 (in low 32) and lane 32 (in high 32)
            f"v_or_b32_e32 v7, {OLD_SENTINEL:#x}, v0",
            *exec_mask_lines(exec_no_0_32, 64),
            "v_add_nc_u32_e32 v10, 8800, v0",
            "v_permlane16_b32 v7, v10, 0, 0 op_sel:[1,0]",
            # Slot 6: fi=1 permlanex16 reading inactive lane 32 from lanes 48..63 & lane 0 from 16..31
            *restore_exec_lines(64),
            f"v_or_b32_e32 v8, {OLD_SENTINEL:#x}, v0",
            *exec_mask_lines(exec_no_0_32, 64),
            "v_permlanex16_b32 v8, v10, 0, 0 op_sel:[1,1]",
            *restore_exec_lines(64),
            *store_slots_lines(["v2", "v3", "v4", "v5", "v6", "v7", "v8"], 64),
        ]

        expected_slots = [
            ref_permlane(False, init_vals, old_vals, ID_S1, ID_S2, ALL64),
            ref_permlane(True, init_vals, old_vals, ID_S1, ID_S2, ALL64),
            ref_permlane(False, init_vals, old_vals, ASYM_S1, ASYM_S2, ALL64),
            ref_permlane(True, init_vals, old_vals, REV16_S1, REV16_S2, ALL64),
            ref_permlane(True, init_vals, old_vals, 0x64206420, 0xBA98BA98, exec_active_only),
            ref_permlane(False, updated_fi1, old_vals, 0, 0, exec_no_0_32, fi=1, bound_ctrl=0),
            ref_permlane(True, updated_fi1, old_vals, 0, 0, exec_no_0_32, fi=1, bound_ctrl=1),
        ]

        for subgroup_size in (64, 32):
            with self.subTest(subgroup_size=subgroup_size):
                flat = run_cs(f"cs_w64_permlane_sg{subgroup_size}", lines, 64, subgroup_size,
                              init_inputs=init_vals, out_slots=7)
                for slot, want in enumerate(expected_slots):
                    got = flat[slot * 64:(slot + 1) * 64]
                    self.assertEqual(got, want, f"subgroup={subgroup_size}, slot={slot}")

    def test_wave64_destination_aliasing_v0_and_v1(self):
        for subgroup_size in (64, 32):
            with self.subTest(subgroup_size=subgroup_size):
                got = run_cs(f"cs_w64_permlane_alias_sg{subgroup_size}", [
                    "s_mov_b64 exec, -1",
                    "v_lshlrev_b32_e32 v21, 2, v0",
                    "v_add_nc_u32_e32 v1, 500, v0",
                    f"s_mov_b32 s4, {PAIR_SWAP_S1:#x}",
                    f"s_mov_b32 s5, {PAIR_SWAP_S2:#x}",
                    "v_permlane16_b32 v1, v1, s4, s5",
                    "v_permlanex16_b32 v0, v0, s4, s5",
                    "buffer_store_dword v1, v21, s[0:3], 0 offen",
                    f"buffer_store_dword v0, v21, s[0:3], 0 offen offset:{4 * 64}",
                ], 64, subgroup_size, out_slots=2)
                self.assertEqual(
                    got[:64],
                    ref_permlane(False, [500 + l for l in range(64)], [0] * 64, PAIR_SWAP_S1, PAIR_SWAP_S2, ALL64),
                )
                self.assertEqual(
                    got[64:],
                    ref_permlane(True, list(range(64)), list(range(64)), PAIR_SWAP_S1, PAIR_SWAP_S2, ALL64),
                )

    def test_wave64_partial_wave_40_lanes(self):
        # 40 threads in a 64-wide subgroup: lanes 32..39 exist in quarter [32..47]; lanes 40..63 do not.
        # Active lanes 0..31 run v_permlanex16_b32 across [0..15] <-> [16..31], and lanes 32..39 run
        # v_permlane16_b32 within [32..39]; inactive lanes 32..39 on permlanex16 discard UNDEF from [48..63].
        exec_low32 = 0x00000000FFFFFFFF
        got = run_cs("cs_w64_partial40_permlane", [
            "s_mov_b64 exec, -1",
            "v_add_nc_u32_e32 v1, 200, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"v_or_b32_e32 v3, {OLD_SENTINEL:#x}, v0",
            "v_permlane16_b32 v2, v1, 0x32107654, 0x32107654",
            *exec_mask_lines(exec_low32, 64),
            "v_permlanex16_b32 v3, v1, 0x01234567, 0x89abcdef",
            *restore_exec_lines(64),
            *store_slots_lines(["v2", "v3"], 40),
        ], 40, 64, out_slots=2)
        vals64 = [200 + l for l in range(40)] + [0] * 24
        old64 = [OLD_SENTINEL | l for l in range(40)] + [0] * 24
        self.assertEqual(got[:40], ref_permlane(False, vals64, old64, 0x32107654, 0x32107654, ALL64)[:40])
        self.assertEqual(got[40:], ref_permlane(True, vals64, old64, REV8_S1, REV8_S2, exec_low32)[:40])


class EmulationAndFindingTests(unittest.TestCase):
    """Translation / structural SPIR-V checks for --wave64-emulate and documented op_sel finding."""

    def test_wave64_emulate_lowers_permlane_through_x_buf_without_subgroup_ops(self):
        base = [
            "s_mov_b64 exec, -1",
            "v_add_nc_u32_e32 v1, 100, v0",
            f"s_mov_b32 s4, {ID_S1:#x}",
            f"s_mov_b32 s5, {ID_S2:#x}",
        ]
        tail = [
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ]
        spv_16, gen_16 = translate_cs("cs_w64_emu_perm16", base + ["v_permlane16_b32 v2, v1, s4, s5"] + tail,
                                     64, emulate=True)
        spv_x16, gen_x16 = translate_cs("cs_w64_emu_permx16", base + ["v_permlanex16_b32 v2, v1, s4, s5"] + tail,
                                       64, emulate=True)
        for spv, gen in ((spv_16, gen_16), (spv_x16, gen_x16)):
            self.assertTrue(gen.wave64 and gen.emulate)
            self.assertNotIn("OpGroupNonUniform", spv)
            self.assertNotIn("%sg_lane", spv)
            self.assertIn("%x_buf = OpVariable", spv)
            self.assertGreaterEqual(spv.count("OpControlBarrier"), 2)
            # Nibble extraction (width = 4) and 16-lane half mask (0xFFFFFFF0)
            self.assertRegex(spv, r"OpBitFieldUExtract %uint %\w+ %\w+ %c4\b")
            self.assertIn("OpBitwiseAnd %uint %", spv)
            self.assertIn("%cfffffff0", spv)
            # spirv_mini_interp.py deliberately rejects the emulated Workgroup/loop scheduler.
            with self.assertRaises(si.UnsupportedConstruct):
                si.run_workgroup(
                    spv, 64, 32,
                    inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32},
                    push_constants=VSHARP + [0] * 28, memory={},
                )

        # v_permlanex16_b32 XORs the masked half base with 16 (%c10); v_permlane16_b32 does not.
        xor16_pat = r"OpBitwiseAnd %uint %\w+ %cfffffff0\n\s*%\w+ = OpBitwiseXor %uint %\w+ %c10\b"
        self.assertIsNone(re.search(xor16_pat, spv_16))
        self.assertIsNotNone(re.search(xor16_pat, spv_x16))

        # Wave32 shaders are unaffected by WAVE64_EMULATE=True.
        w32_lines = ["v_permlanex16_b32 v2, v0, 0x76543210, 0xfedcba98",
                     "v_lshlrev_b32_e32 v21, 2, v0", "buffer_store_dword v2, v21, s[0:3], 0 offen"]
        spv_w32_plain, _ = translate_cs("cs_w32_emu_nop", w32_lines, 32, emulate=False)
        spv_w32_emu, gen_w32_emu = translate_cs("cs_w32_emu_nop", w32_lines, 32, emulate=True)
        self.assertFalse(gen_w32_emu.wave64 or gen_w32_emu.emulate)
        self.assertEqual(spv_w32_plain, spv_w32_emu)

    def test_op_sel_fi0_translates_cleanly_while_inactive_src_handling_is_excluded(self):
        # Documented finding (tools/rdna2spirv.py lines 4608-4632): op_sel:[0,0] and op_sel:[0,1]
        # are accepted by shader_ir/rdna2spirv without raising Unsupported, and lower to
        # OpGroupNonUniformShuffle without inspecting mods or source %active; inactive-source fi=0
        # value assertions stay excluded from Wave32ExecutionTests / Wave64ExecutionTests.
        for op in ("v_permlane16_b32", "v_permlanex16_b32"):
            for opsel in ("op_sel:[0,0]", "op_sel:[0,1]"):
                with self.subTest(op=op, opsel=opsel):
                    spv, gen = translate_cs("cs_w32_opsel_fi0", [
                        "v_mov_b32_e32 v1, v0",
                        "s_mov_b32 exec_lo, 0xfffffffe",
                        f"{op} v2, v1, 0, 0 {opsel}",
                        "s_mov_b32 exec_lo, -1",
                        "v_lshlrev_b32_e32 v21, 2, v0",
                        "buffer_store_dword v2, v21, s[0:3], 0 offen",
                    ], 32)
                    self.assertFalse(gen.wave64)
                    self.assertIn("OpGroupNonUniformShuffle %uint %c3", spv)
                    self.assertRegex(spv, r"OpBitFieldUExtract %uint %\w+ %\w+ %c4\b")


class NegativeControls(unittest.TestCase):
    """Sabotage the translated SPIR-V like a translator regression; multi-lane execution must detect it."""

    def _run(self, spv: str, threads: int, subgroup_size: int) -> list[int]:
        mem = _Heap()
        si.run_workgroup(
            spv, threads, subgroup_size,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        return [mem.get(4 * k) for k in range(threads)]

    def test_sabotage_permlanex16_xor16_dropped(self):
        spv, _ = translate_cs("cs_w64_neg_x16", [
            "s_mov_b64 exec, -1",
            "v_add_nc_u32_e32 v1, 100, v0",
            f"v_permlanex16_b32 v2, v1, {ID_S1:#x}, {ID_S2:#x}",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ], 64)
        want = ref_permlane(True, [100 + l for l in range(64)], [0] * 64, ID_S1, ID_S2, ALL64)
        self.assertEqual(self._run(spv, 64, 64), want)

        bad, n = re.subn(r"(%\w+) = OpBitwiseXor %uint (%\w+) %c10\b", r"\1 = OpCopyObject %uint \2", spv)
        self.assertEqual(n, 1)
        got_bad = self._run(bad, 64, 64)
        self.assertNotEqual(got_bad, want)
        self.assertEqual(got_bad, [100 + l for l in range(64)])  # collapsed to permlane16

    def test_sabotage_half_base_dropped_or_shuffle_removed(self):
        spv, _ = translate_cs("cs_w32_neg_perm16", [
            "v_add_nc_u32_e32 v1, 200, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"s_mov_b32 exec_lo, {0x0F550F55:#x}",
            "s_mov_b32 s4, 0x64206420",
            "s_mov_b32 s5, 0xba98ba98",
            "v_permlane16_b32 v2, v1, s4, s5",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ], 32)
        want = ref_permlane(False, [200 + l for l in range(32)], [OLD_SENTINEL | l for l in range(32)],
                            0x64206420, 0xBA98BA98, 0x0F550F55)
        self.assertEqual(self._run(spv, 32, 32), want)

        # 1. Dropping the 0xFFFFFFF0 half-base mask sends lanes 16..31 into [0..15].
        bad_half, n1 = re.subn(r"OpBitwiseAnd %uint (%\w+) %cfffffff0\b", r"OpBitwiseAnd %uint \1 %c0", spv)
        self.assertEqual(n1, 1)
        self.assertNotEqual(self._run(bad_half, 32, 32), want)

        # 2. Swapping the hi8 OpSelect arms swaps s1 (lanes 0..7) and s2 (lanes 8..15).
        bad_sel, n2 = re.subn(
            r"(%\w+ = OpUGreaterThanEqual %bool %\w+ %c8\n(?:.*\n){0,5}?\s*%\w+ = OpSelect %uint %\w+ )(%\w+) (%\w+)",
            r"\1\3 \2",
            spv,
        )
        self.assertEqual(n2, 1)
        self.assertNotEqual(self._run(bad_sel, 32, 32), want)

        # 3. Replacing OpGroupNonUniformShuffle with OpCopyObject turns permute into own-lane copy.
        bad_shuf, n3 = re.subn(
            r"(%\w+) = OpGroupNonUniformShuffle %uint %c3 (%\w+) %\w+",
            r"\1 = OpCopyObject %uint \2",
            spv,
        )
        self.assertEqual(n3, 2)   # the data shuffle and, with FI=0 (no op_sel), the source lane's activity shuffle (SHADER-74)
        self.assertNotEqual(self._run(bad_shuf, 32, 32), want)

        # 4. The FI/BOUND_CTRL result is a bitwise merge, not a select (#5895): an active lane whose source is off in
        #    EXEC keeps the old value. Every even lane selects an odd (off) source, so the unsabotaged run keeps OLD
        #    everywhere; forcing the ok-mask on makes the active lanes take the off source's value instead — the
        #    MoltenVK symptom the masks exist to prevent.
        spv_fi, _ = translate_cs("cs_w32_neg_perm_fi", [
            "v_add_nc_u32_e32 v1, 200, v0",
            f"v_or_b32_e32 v2, {OLD_SENTINEL:#x}, v0",
            f"s_mov_b32 exec_lo, {0x55555555:#x}",
            "s_mov_b32 s4, 0x11111111",
            "s_mov_b32 s5, 0x11111111",
            "v_permlane16_b32 v2, v1, s4, s5",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v21, 2, v0",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
        ], 32)
        want_fi = [OLD_SENTINEL | l for l in range(32)]   # FI=0 with an off source keeps OLD; odd lanes are off in EXEC
        self.assertEqual(self._run(spv_fi, 32, 32), want_fi)
        ok_mask = re.search(r"(%\w+) = OpSelect %uint %\w+ (%cffffffff) %c0(\n(?:.*\n){0,2}?\s*%\w+ = OpAccessChain %_ptr_Function_uint %vgpr)", spv_fi)
        self.assertIsNotNone(ok_mask)
        bad_ok = spv_fi[:ok_mask.start()] + ok_mask.group(1) + " = OpCopyObject %uint " + ok_mask.group(2) + ok_mask.group(3) + spv_fi[ok_mask.end():]
        got_bad_ok = self._run(bad_ok, 32, 32)
        self.assertNotEqual(got_bad_ok, want_fi)
        self.assertNotEqual(got_bad_ok[0], OLD_SENTINEL)   # the active lane now takes the off source's value

        # 5. Forcing the destination EXEC mask on overwrites inactive destination lanes: dst() combines with the
        #    same masks, and its exec mask is the select that its own OpNot consumes on the next line.
        exec_mask = re.search(r"(%\w+) = OpSelect %uint %\w+ (%cffffffff) %c0(?=\n\s*%\w+ = OpNot %uint \1\n)", spv)
        self.assertIsNotNone(exec_mask)
        bad_exec = spv[:exec_mask.start()] + exec_mask.group(1) + " = OpCopyObject %uint " + exec_mask.group(2) + spv[exec_mask.end():]
        got_bad_exec = self._run(bad_exec, 32, 32)
        self.assertNotEqual(got_bad_exec, want)
        self.assertNotEqual(got_bad_exec[1], OLD_SENTINEL | 1)


if __name__ == "__main__":
    unittest.main()
