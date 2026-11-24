"""Semantic interpreter tests for RDNA2 vector bitfield operations (v_bfe_u32, v_bfe_i32, v_bfi_b32).

Covers:
  - RDNA2 Vector ALU Bitfield Instructions:
      * v_bfe_u32 D, S0, S1, S2:
          Unsigned bitfield extract. Extracts S2 (width) bits from S0 (base) starting at
          bit S1 (offset).
          D = (S0 >> (S1 & 31)) & ((1 << eff_width) - 1)
          where eff_width = min(S2 & 31, 32 - (S1 & 31)).
          If width == 0, D = 0.
      * v_bfe_i32 D, S0, S1, S2:
          Signed bitfield extract. Extracts S2 (width) bits from S0 (base) starting at
          bit S1 (offset) and sign-extends the result from bit (eff_width - 1) to 32 bits.
          If width == 0, D = 0.
      * v_bfi_b32 D, S0, S1, S2:
          Bitfield insert with bitmask selection.
          D = (S0 & S1) | (~S0 & S2)
          where S0 is bitmask, S1 is replacement field, S2 is base dword.
  - Verification layers:
      1. Pure Python reference mathematical oracles.
      2. Simulated VALU register file interpreter (AluBitfieldInterpreter).
      3. SPIR-V lowering and execution via rdna2spirv and spirv_mini_interp.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned dword."""
    return val & MASK32


def i32(val: int) -> int:
    """Convert 32-bit unsigned dword to signed two's complement integer."""
    val = val & MASK32
    return val - (1 << 32) if val >> 31 else val


def make_mask(offset: int, width: int) -> int:
    """Generate a contiguous 32-bit mask of given width at given offset."""
    off = offset & 31
    wid = width & 31
    if wid == 0:
        return 0
    eff_wid = min(wid, 32 - off)
    if eff_wid >= 32:
        return MASK32
    return (((1 << eff_wid) - 1) << off) & MASK32


def pack_field(base: int, val: int, offset: int, width: int) -> int:
    """Pack value into base at [offset +: width] using v_bfi_b32 semantics."""
    mask = make_mask(offset, width)
    field = (val << (offset & 31)) & mask
    return ref_v_bfi_b32(mask, field, base)


def ref_v_bfe_u32(base: int, offset: int, width: int) -> int:
    """Reference implementation of RDNA2 v_bfe_u32 ALU instruction.

    D = (base >> (offset & 31)) & ((1 << eff_width) - 1)
    Clamps effective width to (32 - (offset & 31)) per RDNA2 ISA / SPIR-V lower bounds.
    """
    b = base & MASK32
    off = offset & 31
    wid = width & 31
    eff_wid = min(wid, 32 - off)
    if eff_wid == 0:
        return 0
    return ((b >> off) & ((1 << eff_wid) - 1)) & MASK32


def ref_v_bfe_i32(base: int, offset: int, width: int) -> int:
    """Reference implementation of RDNA2 v_bfe_i32 ALU instruction.

    Extracts eff_wid bits and sign-extends from bit (eff_wid - 1).
    If eff_wid == 0, returns 0.
    """
    b = base & MASK32
    off = offset & 31
    wid = width & 31
    eff_wid = min(wid, 32 - off)
    if eff_wid == 0:
        return 0
    field = (b >> off) & ((1 << eff_wid) - 1)
    if field & (1 << (eff_wid - 1)):
        field -= 1 << eff_wid
    return field & MASK32


def ref_v_bfi_b32(mask: int, field: int, base: int) -> int:
    """Reference implementation of RDNA2 v_bfi_b32 ALU instruction.

    D = (mask & field) | (~mask & base)
    """
    m = mask & MASK32
    f = field & MASK32
    b = base & MASK32
    return ((m & f) | ((~m & MASK32) & b)) & MASK32


class AluBitfieldInterpreter:
    """Simulated RDNA2 VALU register machine with VGPR state."""

    def __init__(self, num_vgprs: int = 64) -> None:
        self.vgprs = [0] * num_vgprs

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg] & MASK32

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def _resolve_src(self, operand: int | str) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        op_str = str(operand).strip().lower()
        if op_str.startswith("v") and op_str[1:].isdigit():
            idx = int(op_str[1:])
            return self.get_vgpr(idx)
        if op_str.startswith("0x") or op_str.startswith("-0x"):
            return int(op_str, 16) & MASK32
        return int(op_str) & MASK32

    def _resolve_dst_reg(self, operand: int | str) -> int:
        if isinstance(operand, int):
            return operand
        op_str = str(operand).strip().lower()
        if op_str.startswith("v") and op_str[1:].isdigit():
            return int(op_str[1:])
        raise ValueError(f"Invalid VGPR destination: {operand}")

    def step(self, mnemonic: str, dst: int | str, *srcs: int | str) -> int:
        d_idx = self._resolve_dst_reg(dst)
        m = mnemonic.strip().lower()
        if m in ("v_mov_b32", "v_mov_b32_e32", "v_mov_b32_e64"):
            res = self._resolve_src(srcs[0])
        elif m in ("v_bfe_u32", "v_bfe_u32_e64"):
            base = self._resolve_src(srcs[0])
            off = self._resolve_src(srcs[1])
            wid = self._resolve_src(srcs[2])
            res = ref_v_bfe_u32(base, off, wid)
        elif m in ("v_bfe_i32", "v_bfe_i32_e64"):
            base = self._resolve_src(srcs[0])
            off = self._resolve_src(srcs[1])
            wid = self._resolve_src(srcs[2])
            res = ref_v_bfe_i32(base, off, wid)
        elif m in ("v_bfi_b32", "v_bfi_b32_e64"):
            mask = self._resolve_src(srcs[0])
            field = self._resolve_src(srcs[1])
            base = self._resolve_src(srcs[2])
            res = ref_v_bfi_b32(mask, field, base)
        elif m in ("v_and_b32", "v_and_b32_e32", "v_and_b32_e64"):
            res = (self._resolve_src(srcs[0]) & self._resolve_src(srcs[1])) & MASK32
        elif m in ("v_or_b32", "v_or_b32_e32", "v_or_b32_e64"):
            res = (self._resolve_src(srcs[0]) | self._resolve_src(srcs[1])) & MASK32
        elif m in ("v_xor_b32", "v_xor_b32_e32", "v_xor_b32_e64"):
            res = (self._resolve_src(srcs[0]) ^ self._resolve_src(srcs[1])) & MASK32
        elif m in ("v_not_b32", "v_not_b32_e32", "v_not_b32_e64"):
            res = (~self._resolve_src(srcs[0])) & MASK32
        elif m in ("v_lshlrev_b32", "v_lshlrev_b32_e32", "v_lshlrev_b32_e64"):
            shift = self._resolve_src(srcs[0]) & 31
            val = self._resolve_src(srcs[1])
            res = (val << shift) & MASK32
        elif m in ("v_lshrrev_b32", "v_lshrrev_b32_e32", "v_lshrrev_b32_e64"):
            shift = self._resolve_src(srcs[0]) & 31
            val = self._resolve_src(srcs[1])
            res = (val >> shift) & MASK32
        elif m in ("v_ashrrev_i32", "v_ashrrev_i32_e32", "v_ashrrev_i32_e64"):
            shift = self._resolve_src(srcs[0]) & 31
            val = i32(self._resolve_src(srcs[1]))
            res = (val >> shift) & MASK32
        else:
            raise NotImplementedError(f"Instruction {mnemonic} not modeled in AluBitfieldInterpreter")

        self.set_vgpr(d_idx, res)
        return res

    def run_program(self, instructions: list[tuple]) -> None:
        for instr in instructions:
            mnemonic, dst, *srcs = instr
            self.step(mnemonic, dst, *srcs)


def run_rdna_ps(lines: list[str], name: str = "ps_alu_bitfield") -> list[int]:
    """Assemble and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, all_lines)
    return mrt0_words(spv)


def run_rdna_cs_wave(
    lines: list[str],
    num_lanes: int = 16,
    initial_mem: dict[int, int] | None = None,
    name: str = "cs_simd_wave",
) -> dict[int, int]:
    """Assemble and run a compute shader workgroup with linear local IDs."""
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x8088, num_lanes, 1, 1)
    all_lines = list(lines) + ["s_endpgm"]
    spv = translate_lines(name, all_lines, fssm=bytes(fssm))
    mem = dict(initial_mem or {})
    return si.run_workgroup(
        spv,
        num_lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
        memory=mem,
    )


class SimdWaveBitfieldModel:
    """Deterministic SIMD Wave model executing vector bitfield operations with EXEC masks."""

    def __init__(self, wave_size: int = 32, num_vgprs: int = 64) -> None:
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.vgprs = [[0] * num_vgprs for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg]

    def v_bfe_u32(self, dst: int, src_base: int, src_off: int, src_wid: int) -> None:
        for lane in range(self.wave_size):
            if self.exec_mask & (1 << lane):
                base = self.vgprs[lane][src_base]
                off = self.vgprs[lane][src_off]
                wid = self.vgprs[lane][src_wid]
                self.vgprs[lane][dst] = ref_v_bfe_u32(base, off, wid)

    def v_bfe_i32(self, dst: int, src_base: int, src_off: int, src_wid: int) -> None:
        for lane in range(self.wave_size):
            if self.exec_mask & (1 << lane):
                base = self.vgprs[lane][src_base]
                off = self.vgprs[lane][src_off]
                wid = self.vgprs[lane][src_wid]
                self.vgprs[lane][dst] = ref_v_bfe_i32(base, off, wid)

    def v_bfi_b32(self, dst: int, src_mask: int, src_fld: int, src_base: int) -> None:
        for lane in range(self.wave_size):
            if self.exec_mask & (1 << lane):
                mask = self.vgprs[lane][src_mask]
                fld = self.vgprs[lane][src_fld]
                base = self.vgprs[lane][src_base]
                self.vgprs[lane][dst] = ref_v_bfi_b32(mask, fld, base)


class TestBitfieldReferenceOracle(unittest.TestCase):
    """Unit tests for pure Python reference oracles."""

    def test_ref_bfe_u32_basic_fields(self):
        cases = [
            (0x12345678, 4, 8, 0x67),
            (0xDEADBEEF, 8, 12, 0xDBE),
            (0xABCD1234, 16, 16, 0xABCD),
            (0xFEDCBA98, 0, 4, 0x8),
            (0xFEDCBA98, 0, 8, 0x98),
            (0xFEDCBA98, 0, 16, 0xBA98),
            (0x87654321, 28, 4, 0x8),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_v_bfe_u32(base, off, wid), expected)

    def test_ref_bfe_u32_zero_width_is_zero(self):
        bases = [0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678, 0x80000000]
        offsets = [0, 1, 7, 15, 16, 30, 31]
        for b in bases:
            for off in offsets:
                with self.subTest(base=hex(b), off=off):
                    self.assertEqual(ref_v_bfe_u32(b, off, 0), 0)

    def test_ref_bfe_u32_single_bit_extractions(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                self.assertEqual(ref_v_bfe_u32(val, bit, 1), 1)
                self.assertEqual(ref_v_bfe_u32(~val & MASK32, bit, 1), 0)

    def test_ref_bfe_u32_offset_modulo_32(self):
        base = 0x12345678
        self.assertEqual(ref_v_bfe_u32(base, 36, 8), ref_v_bfe_u32(base, 4, 8))
        self.assertEqual(ref_v_bfe_u32(base, 68, 8), ref_v_bfe_u32(base, 4, 8))

    def test_ref_bfe_i32_positive_values_zero_extended(self):
        cases = [
            (0x76543210, 28, 4, 0x00000007),
            (0x00007FFF, 0, 16, 0x00007FFF),
            (0x00003FFF, 0, 15, 0x00003FFF),
            (0x1234007F, 0, 8, 0x0000007F),
            (0x55555555, 0, 2, 0x00000001),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_v_bfe_i32(base, off, wid), expected)

    def test_ref_bfe_i32_negative_values_sign_extended(self):
        cases = [
            (0xFFFFFFF0, 2, 4, 0xFFFFFFFC),
            (0x80000000, 31, 1, 0xFFFFFFFF),
            (0x00000080, 0, 8, 0xFFFFFF80),
            (0x00008000, 0, 16, 0xFFFF8000),
            (0xAAAAAAAA, 0, 2, 0xFFFFFFFE),
            (0xFFFFFFFF, 0, 4, 0xFFFFFFFF),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_v_bfe_i32(base, off, wid), expected)

    def test_ref_bfe_i32_zero_width_is_zero(self):
        bases = [0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x80000000, 0x7FFFFFFF]
        for b in bases:
            for off in (0, 1, 15, 31):
                with self.subTest(base=hex(b), off=off):
                    self.assertEqual(ref_v_bfe_i32(b, off, 0), 0)

    def test_ref_bfe_i32_single_bit_signs(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                # When bit is 1 and width is 1, bit (width - 1) is 1 -> sign extends to -1 (0xFFFFFFFF)
                self.assertEqual(ref_v_bfe_i32(val, bit, 1), MASK32)
                # When bit is 0 and width is 1, bit (width - 1) is 0 -> zero extends to 0
                self.assertEqual(ref_v_bfe_i32(~val & MASK32, bit, 1), 0)

    def test_ref_bfi_b32_basic_cases(self):
        cases = [
            (0x00FF0000, 0x12345678, 0xA0B0C0D0, 0xA034C0D0),
            (0x000000FF, 0xAABBCCDD, 0x11223344, 0x112233DD),
            (0xFF000000, 0xAABBCCDD, 0x11223344, 0xAA223344),
            (0x0000FF00, 0xAABBCCDD, 0x11223344, 0x1122CC44),
            (0xF0F0F0F0, 0xAAAAAAAA, 0x55555555, 0xA5A5A5A5),
            (0x0F0F0F0F, 0xAAAAAAAA, 0x55555555, 0x5A5A5A5A),
        ]
        for mask, fld, base, expected in cases:
            with self.subTest(mask=hex(mask), fld=hex(fld), base=hex(base)):
                self.assertEqual(ref_v_bfi_b32(mask, fld, base), expected)

    def test_ref_bfi_b32_mask_zero_preserves_base(self):
        for base in [0x0, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF, 0x80000001]:
            with self.subTest(base=hex(base)):
                self.assertEqual(ref_v_bfi_b32(0, 0xCAFEBABE, base), base)

    def test_ref_bfi_b32_mask_all_ones_replaces_base(self):
        for fld in [0x0, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF, 0x80000001]:
            with self.subTest(fld=hex(fld)):
                self.assertEqual(ref_v_bfi_b32(MASK32, fld, 0xCAFEBABE), fld)

    def test_ref_bfi_b32_identity_field_equals_base(self):
        for val in [0x0, 0xFFFFFFFF, 0x12345678, 0x87654321, 0x55555555]:
            for mask in [0x0, 0xFFFFFFFF, 0xF0F0F0F0, 0x00FF00FF]:
                self.assertEqual(ref_v_bfi_b32(mask, val, val), val)

    def test_ref_bfi_b32_duality_inverted_mask(self):
        mask = 0x3CA55AC3
        s1 = 0xDEADBEEF
        s2 = 0x01234567
        inv_mask = (~mask) & MASK32
        self.assertEqual(ref_v_bfi_b32(mask, s1, s2), ref_v_bfi_b32(inv_mask, s2, s1))


class TestAluBitfieldInterpreter(unittest.TestCase):
    """Tests for simulated register interpreter executing bitfield operations."""

    def test_interpreter_mov_and_bfe_u32(self):
        interp = AluBitfieldInterpreter()
        interp.step("v_mov_b32", "v1", 0x12345678)
        interp.step("v_mov_b32", "v2", 4)
        interp.step("v_mov_b32", "v3", 8)
        interp.step("v_bfe_u32", "v0", "v1", "v2", "v3")
        self.assertEqual(interp.get_vgpr(0), 0x67)

    def test_interpreter_bfe_i32_sign_extension(self):
        interp = AluBitfieldInterpreter()
        interp.step("v_mov_b32", "v1", 0xFFFFFFF0)
        interp.step("v_mov_b32", "v2", 2)
        interp.step("v_mov_b32", "v3", 4)
        interp.step("v_bfe_i32", "v0", "v1", "v2", "v3")
        self.assertEqual(interp.get_vgpr(0), 0xFFFFFFFC)

    def test_interpreter_bfi_b32(self):
        interp = AluBitfieldInterpreter()
        interp.step("v_mov_b32", "v1", 0x00FF0000)
        interp.step("v_mov_b32", "v2", 0x12345678)
        interp.step("v_mov_b32", "v3", 0xA0B0C0D0)
        interp.step("v_bfi_b32", "v0", "v1", "v2", "v3")
        self.assertEqual(interp.get_vgpr(0), 0xA034C0D0)

    def test_interpreter_in_place_overwrite(self):
        interp = AluBitfieldInterpreter()
        interp.set_vgpr(0, 0x12345678)
        interp.step("v_bfe_u32", "v0", "v0", 8, 8)
        self.assertEqual(interp.get_vgpr(0), 0x56)

    def test_interpreter_operand_aliasing(self):
        interp = AluBitfieldInterpreter()
        interp.set_vgpr(1, 0x0000FFFF)
        interp.set_vgpr(0, 0x12345678)
        interp.step("v_bfi_b32", "v0", "v1", "v0", "v0")
        self.assertEqual(interp.get_vgpr(0), 0x12345678)


class TestRdna2SpirvTranslationBfeBfi(unittest.TestCase):
    """Verifies that rdna2spirv lowers v_bfe and v_bfi instructions matching oracle."""

    def test_rdna2_bfe_u32_translation_matches_oracle(self):
        cases = [
            (0x12345678, 4, 8),
            (0xDEADBEEF, 8, 12),
            (0xABCD1234, 16, 16),
            (0xFFFFFFFF, 0, 8),
            (0x80000000, 31, 1),
        ]
        for base, off, wid in cases:
            expected = ref_v_bfe_u32(base, off, wid)
            words = run_rdna_ps([
                f"v_mov_b32 v1, {base:#x}",
                f"v_mov_b32 v2, {off}",
                f"v_mov_b32 v3, {wid}",
                "v_bfe_u32 v0, v1, v2, v3",
                "v_mov_b32 v1, v0",
                "v_mov_b32 v2, v0",
                "v_mov_b32 v3, v0",
            ], name=f"ps_bfe_u_{off}_{wid}")
            self.assertEqual(words[0], expected)

    def test_rdna2_bfe_i32_translation_matches_oracle(self):
        cases = [
            (0xFFFFFFF0, 2, 4),
            (0x80000000, 31, 1),
            (0x00007FFF, 0, 16),
            (0x00008000, 0, 16),
        ]
        for base, off, wid in cases:
            expected = ref_v_bfe_i32(base, off, wid)
            words = run_rdna_ps([
                f"v_mov_b32 v1, {base:#x}",
                f"v_mov_b32 v2, {off}",
                f"v_mov_b32 v3, {wid}",
                "v_bfe_i32 v0, v1, v2, v3",
                "v_mov_b32 v1, v0",
                "v_mov_b32 v2, v0",
                "v_mov_b32 v3, v0",
            ], name=f"ps_bfe_i_{off}_{wid}")
            self.assertEqual(words[0], expected)

    def test_rdna2_bfi_b32_translation_matches_oracle(self):
        cases = [
            (0x00FF0000, 0x12345678, 0xA0B0C0D0),
            (0x000000FF, 0xAABBCCDD, 0x11223344),
            (0xF0F0F0F0, 0xAAAAAAAA, 0x55555555),
        ]
        for mask, fld, base in cases:
            expected = ref_v_bfi_b32(mask, fld, base)
            words = run_rdna_ps([
                f"v_mov_b32 v1, {mask:#x}",
                f"v_mov_b32 v2, {fld:#x}",
                f"v_mov_b32 v3, {base:#x}",
                "v_bfi_b32 v0, v1, v2, v3",
                "v_mov_b32 v1, v0",
                "v_mov_b32 v2, v0",
                "v_mov_b32 v3, v0",
            ], name=f"ps_bfi_{mask & 0xFFFF:#x}")
            self.assertEqual(words[0], expected)


class TestBitfieldRoundtripAndComposition(unittest.TestCase):
    """Verifies roundtrip invariants, untouched bits preservation, and mask composition."""

    def test_extract_insert_roundtrip_exhaustive_offsets(self):
        """For any word, extracting [off : off + wid] and re-inserting restores original word."""
        test_patterns = [
            0x12345678,
            0xDEADBEEF,
            0xA5A5A5A5,
            0x00000001,
            0x80000000,
            0xFFFFFFFF,
            0x00000000,
            0xCAFEBABE,
        ]
        for base in test_patterns:
            for off in (0, 1, 4, 7, 8, 12, 16, 20, 24, 28, 31):
                for wid in (1, 2, 4, 8, 12, 16, 24, 32 - off):
                    if off + wid > 32 or wid <= 0:
                        continue
                    with self.subTest(base=hex(base), off=off, wid=wid):
                        extracted = ref_v_bfe_u32(base, off, wid)
                        restored = pack_field(base, extracted, off, wid)
                        self.assertEqual(restored, base)
                        isolated = pack_field(0, extracted, off, wid)
                        mask = make_mask(off, wid)
                        self.assertEqual(isolated, base & mask)

    def test_untouched_bits_preservation(self):
        """Inserting a new field value modifies only the masked bits, leaving untouched bits intact."""
        base = 0xDEADBEEF
        for off in (0, 4, 8, 16, 24):
            for wid in (1, 4, 8, 12):
                if off + wid > 32:
                    continue
                mask = make_mask(off, wid)
                new_val = 0x5A
                inserted = pack_field(base, new_val, off, wid)
                self.assertEqual(inserted & (~mask & MASK32), base & (~mask & MASK32))
                extracted_new = ref_v_bfe_u32(inserted, off, wid)
                self.assertEqual(extracted_new, new_val & ((1 << wid) - 1))

    def test_rgba8_pack_unpack_roundtrip(self):
        """Simulate packing and unpacking RGBA8 color components."""
        r, g, b, a = 0x12, 0x34, 0x56, 0x78
        packed = 0
        packed = pack_field(packed, r, 0, 8)
        packed = pack_field(packed, g, 8, 8)
        packed = pack_field(packed, b, 16, 8)
        packed = pack_field(packed, a, 24, 8)
        self.assertEqual(packed, 0x78563412)

        out_r = ref_v_bfe_u32(packed, 0, 8)
        out_g = ref_v_bfe_u32(packed, 8, 8)
        out_b = ref_v_bfe_u32(packed, 16, 8)
        out_a = ref_v_bfe_u32(packed, 24, 8)
        self.assertEqual((out_r, out_g, out_b, out_a), (r, g, b, a))

    def test_hdr_r10g10b10a2_pack_unpack(self):
        """Simulate packing and unpacking HDR10 R10G10B10A2 format."""
        r10 = 0x3FF
        g10 = 0x200
        b10 = 0x155
        a2 = 0x3

        packed = 0
        packed = pack_field(packed, r10, 0, 10)
        packed = pack_field(packed, g10, 10, 10)
        packed = pack_field(packed, b10, 20, 10)
        packed = pack_field(packed, a2, 30, 2)

        self.assertEqual(ref_v_bfe_u32(packed, 0, 10), r10)
        self.assertEqual(ref_v_bfe_u32(packed, 10, 10), g10)
        self.assertEqual(ref_v_bfe_u32(packed, 20, 10), b10)
        self.assertEqual(ref_v_bfe_u32(packed, 30, 2), a2)

    def test_rgb565_pack_unpack(self):
        """Simulate packing and unpacking 16-bit RGB565 in low word."""
        r5 = 0x1F
        g6 = 0x2A
        b5 = 0x0E

        packed = 0
        packed = pack_field(packed, r5, 0, 5)
        packed = pack_field(packed, g6, 5, 6)
        packed = pack_field(packed, b5, 11, 5)

        self.assertEqual(ref_v_bfe_u32(packed, 0, 5), r5)
        self.assertEqual(ref_v_bfe_u32(packed, 5, 6), g6)
        self.assertEqual(ref_v_bfe_u32(packed, 11, 5), b5)
        self.assertEqual(packed >> 16, 0)

    def test_bfi_boolean_xor_equivalence(self):
        """Verify algebraic equivalence: BFI(m, f, b) == ((m & (f ^ b)) ^ b)."""
        test_tuples = [
            (0x00FF0000, 0x12345678, 0x87654321),
            (0xF0F0F0F0, 0xAAAAAAAA, 0x55555555),
            (0x12345678, 0xDEADBEEF, 0xCAFEBABE),
            (0xFFFFFFFF, 0x11111111, 0x22222222),
            (0x00000000, 0x11111111, 0x22222222),
        ]
        for m, f, b in test_tuples:
            bfi_std = ref_v_bfi_b32(m, f, b)
            bfi_xor = ((m & (f ^ b)) ^ b) & MASK32
            self.assertEqual(bfi_std, bfi_xor)

    def test_shader_extract_insert_roundtrip(self):
        """Run translated shader performing roundtrip extraction and re-insertion."""
        base = 0x12345678
        lines = [
            f"v_mov_b32 v1, {base:#x}",
            "v_mov_b32 v2, 8",
            "v_mov_b32 v3, 8",
            "v_bfe_u32 v4, v1, v2, v3",
            "v_add_nc_u32 v4, 1, v4",
            "v_lshlrev_b32 v4, 8, v4",
            "v_mov_b32 v5, 0x0000FF00",
            "v_bfi_b32 v0, v5, v4, v1",
            "v_mov_b32 v1, v0",
            "v_mov_b32 v2, v0",
            "v_mov_b32 v3, v0",
        ]
        words = run_rdna_ps(lines, name="ps_bfe_bfi_roundtrip")
        self.assertEqual(words[0], 0x12345778)


class TestBitfieldClampBoundaries(unittest.TestCase):
    """Verifies edge cases for width/offset clamp boundaries, zero width, and modulo 32."""

    def test_offset_plus_width_exceeds_32_clamped(self):
        """When offset + width > 32, width clamps to min(width, 32 - offset)."""
        cases = [
            (0xFFFFFFFF, 24, 16, 8, 0x000000FF),
            (0xDEADBEEF, 28, 8, 4, 0x0000000D),
            (0x80000000, 31, 31, 1, 0x00000001),
            (0x12345678, 16, 20, 16, 0x00001234),
        ]
        for base, off, wid, eff_wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_v_bfe_u32(base, off, wid), expected)
                words = run_rdna_ps([
                    f"v_mov_b32 v1, {base:#x}",
                    f"v_mov_b32 v2, {off}",
                    f"v_mov_b32 v3, {wid}",
                    "v_bfe_u32 v0, v1, v2, v3",
                    "v_mov_b32 v1, v0",
                    "v_mov_b32 v2, v0",
                    "v_mov_b32 v3, v0",
                ], name=f"ps_clamp_u_{off}_{wid}")
                self.assertEqual(words[0], expected)

    def test_signed_clamp_boundary_sign_extension(self):
        """When clamped, sign extension for v_bfe_i32 extends from (eff_width - 1)."""
        cases = [
            (0xC0000000, 30, 8, 0xFFFFFFFF),
            (0x80000000, 30, 8, 0xFFFFFFFE),
            (0x40000000, 30, 8, 0x00000001),
            (0x80000000, 31, 16, 0xFFFFFFFF),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_v_bfe_i32(base, off, wid), expected)
                words = run_rdna_ps([
                    f"v_mov_b32 v1, {base:#x}",
                    f"v_mov_b32 v2, {off}",
                    f"v_mov_b32 v3, {wid}",
                    "v_bfe_i32 v0, v1, v2, v3",
                    "v_mov_b32 v1, v0",
                    "v_mov_b32 v2, v0",
                    "v_mov_b32 v3, v0",
                ], name=f"ps_clamp_i_{off}_{wid}")
                self.assertEqual(words[0], expected)

    def test_offset_modulo_32_wrapping(self):
        """Offset operand is taken modulo 32."""
        base = 0x12345678
        for extra in (32, 64, 96):
            self.assertEqual(ref_v_bfe_u32(base, 4 + extra, 8), 0x67)
            self.assertEqual(ref_v_bfe_u32(base, 0 + extra, 4), 0x8)

    def test_width_modulo_32_wrapping_and_zero(self):
        """Width is taken modulo 32; width=0 or width=32 (32 & 31 = 0) returns 0."""
        for base in (0xFFFFFFFF, 0x12345678, 0xDEADBEEF):
            for off in (0, 7, 16, 31):
                self.assertEqual(ref_v_bfe_u32(base, off, 0), 0)
                self.assertEqual(ref_v_bfe_u32(base, off, 32), 0)
                self.assertEqual(ref_v_bfe_i32(base, off, 0), 0)
                self.assertEqual(ref_v_bfe_i32(base, off, 32), 0)


class TestSimdDivergence(unittest.TestCase):
    """Verifies SIMD lane divergence and EXEC mask write inhibition."""

    def test_simd_divergent_per_lane_extraction(self):
        """16 SIMD compute lanes evaluate lane-dependent bitfield offsets."""
        lanes = 16
        base = 0x76543210
        lines = [
            f"v_mov_b32_e32 v1, {base:#x}",
            "v_bfe_u32 v2, v1, v0, 4",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_rdna_cs_wave(lines, num_lanes=lanes, name="cs_simd_divergent")
        for i in range(lanes):
            expected = ref_v_bfe_u32(base, i, 4)
            self.assertEqual(mem[4 * i], expected, f"Mismatch at lane {i}")

    def test_simd_exec_mask_write_inhibition(self):
        """Lanes deactivated by v_cmpx do not write to destination memory."""
        lanes = 16
        initial_mem = {4 * i: 0xDEADBEEF for i in range(lanes)}
        lines = [
            "v_mov_b32_e32 v1, 0x12345678",
            "v_cmpx_lt_u32_e32 v0, 8",
            "v_bfe_u32 v2, v1, 4, 8",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_rdna_cs_wave(lines, num_lanes=lanes, initial_mem=initial_mem, name="cs_simd_exec_mask")
        for i in range(lanes):
            if i < 8:
                self.assertEqual(mem[4 * i], 0x67, f"Active lane {i} should be updated")
            else:
                self.assertEqual(mem[4 * i], 0xDEADBEEF, f"Inactive lane {i} should remain unchanged")

    def test_simd_wave_model_wave32_and_wave64(self):
        """Test pure-Python SimdWaveBitfieldModel across wave32 and wave64 configurations."""
        for wave_size in (32, 64):
            model = SimdWaveBitfieldModel(wave_size=wave_size)
            for lane in range(wave_size):
                model.set_lane_vgpr(lane, 1, 0x12345678)
                model.set_lane_vgpr(lane, 2, lane & 15)
                model.set_lane_vgpr(lane, 3, 4)
                model.set_lane_vgpr(lane, 0, 0xAAAAAAAA)

            even_mask = sum(1 << l for l in range(wave_size) if l % 2 == 0)
            model.set_exec(even_mask)
            model.v_bfe_u32(0, 1, 2, 3)

            for lane in range(wave_size):
                if lane % 2 == 0:
                    expected = ref_v_bfe_u32(0x12345678, lane & 15, 4)
                    self.assertEqual(model.get_lane_vgpr(lane, 0), expected)
                else:
                    self.assertEqual(model.get_lane_vgpr(lane, 0), 0xAAAAAAAA)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring sensitivity to subtle bitfield errors."""

    def test_negative_unsigned_bfe_sign_extend_detected(self):
        """Mutated v_bfe_u32 that incorrectly sign-extends must fail assertion."""
        def buggy_bfe_u32(base: int, offset: int, width: int) -> int:
            return ref_v_bfe_i32(base, offset, width)

        correct = ref_v_bfe_u32(0x80000000, 31, 1)
        buggy = buggy_bfe_u32(0x80000000, 31, 1)
        self.assertEqual(correct, 1)
        self.assertNotEqual(buggy, correct)

    def test_negative_signed_bfe_zero_extend_detected(self):
        """Mutated v_bfe_i32 that incorrectly zero-extends must fail assertion."""
        def buggy_bfe_i32(base: int, offset: int, width: int) -> int:
            return ref_v_bfe_u32(base, offset, width)

        correct = ref_v_bfe_i32(0x80000000, 31, 1)
        buggy = buggy_bfe_i32(0x80000000, 31, 1)
        self.assertEqual(correct, MASK32)
        self.assertNotEqual(buggy, correct)

    def test_negative_bfi_inverted_operands_detected(self):
        """Mutated v_bfi_b32 that inverts wrong operand must fail assertion."""
        def buggy_bfi_b32(mask: int, field: int, base: int) -> int:
            return ((mask & base) | ((~mask & MASK32) & field)) & MASK32

        mask, field, base = 0x00FF0000, 0x12345678, 0xA0B0C0D0
        correct = ref_v_bfi_b32(mask, field, base)
        buggy = buggy_bfi_b32(mask, field, base)
        self.assertEqual(correct, 0xA034C0D0)
        self.assertNotEqual(buggy, correct)

    def test_negative_width_zero_nonzero_result_detected(self):
        """Mutated zero-width returning base must fail assertion."""
        def buggy_bfe_zero_width(base: int, offset: int, width: int) -> int:
            if width == 0:
                return base & MASK32
            return ref_v_bfe_u32(base, offset, width)

        correct = ref_v_bfe_u32(0xDEADBEEF, 4, 0)
        buggy = buggy_bfe_zero_width(0xDEADBEEF, 4, 0)
        self.assertEqual(correct, 0)
        self.assertNotEqual(buggy, correct)

    def test_negative_clamp_out_of_bounds_detected(self):
        """Mutated BFE that shifts without clamping width beyond 32 must diverge."""
        def unclamped_bfe_i32(base: int, offset: int, width: int) -> int:
            val = ((base & MASK32) >> offset) & ((1 << width) - 1)
            if val & (1 << (width - 1)):
                val -= 1 << width
            return val & MASK32

        # At off=30, wid=8: clamped effective width is 2 (bits 30..31 of 0xFFFFFFFF are 11 -> sign-extends to 0xFFFFFFFF).
        # Unclamped width=8 checks bit 7 (which is 0!), incorrectly treating it as positive 3 (0x00000003).
        clamped = ref_v_bfe_i32(0xFFFFFFFF, 30, 8)
        unclamped = unclamped_bfe_i32(0xFFFFFFFF, 30, 8)
        self.assertEqual(clamped, MASK32)
        self.assertEqual(unclamped, 0x3)
        self.assertNotEqual(clamped, unclamped)

    def test_negative_simd_inactive_lane_leak_detected(self):
        """Mutated SIMD execution that updates inactive lanes must fail assertion."""
        model = SimdWaveBitfieldModel(wave_size=32)
        for l in range(32):
            model.set_lane_vgpr(l, 0, 0x11111111)
            model.set_lane_vgpr(l, 1, 0x22222222)
            model.set_lane_vgpr(l, 2, 0)
            model.set_lane_vgpr(l, 3, 4)

        model.set_exec(0x1)
        model.v_bfe_u32(0, 1, 2, 3)

        leaked_model = SimdWaveBitfieldModel(wave_size=32)
        for l in range(32):
            leaked_model.set_lane_vgpr(l, 0, 0x2)

        self.assertEqual(model.get_lane_vgpr(5, 0), 0x11111111)
        self.assertNotEqual(leaked_model.get_lane_vgpr(5, 0), model.get_lane_vgpr(5, 0))


if __name__ == "__main__":
    unittest.main()
