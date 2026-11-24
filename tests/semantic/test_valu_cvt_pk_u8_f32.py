"""Semantic value tests for RDNA2 v_cvt_pk_u8_f32 (VOP3).

AMD RDNA2 ISA semantics for `v_cvt_pk_u8_f32 vD, src0, src1, src2`:
  - Converts binary32 `src0` to an 8-bit unsigned integer using IEEE-754 round-to-nearest-even (`RoundEven`),
    clamped to [0, 255] (negative values and -inf -> 0; values >= 255.0 and +inf -> 255; NaN -> 0).
  - Inserts that byte into byte lane `src1[1:0]` (`src1 & 3`, i.e. bits [7:0], [15:8], [23:16], or [31:24])
    of `src2`, leaving the other 3 bytes of `src2` untouched in `vD`.

Executed end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py (single-lane pixel
shaders via run_fragment and multi-lane compute shaders via run_workgroup), and checked against hand-worked
constants and an independent pure-Python binary32 Fraction/RTE reference oracle. Standard library only;
hand-written RDNA2 disassembly listings only.

NaN inputs (docs/design.md, "Float-to-byte pack" row): `FClamp` with a NaN operand is undefined in
  GLSL.std.450, so tools/rdna2spirv.py finds a NaN from its bits and selects +0 before `FClamp`, then
  `RoundEven` + `OpConvertFToU`. `OracleTests` verifies the RDNA2 ISA `NaN -> 0` rule (also measured on
  hardware, #358), and `NaNHandlingTests` runs qNaN, -qNaN, sNaN and NaN under `abs`/`neg` through
  `spirv_mini_interp.py` and checks byte 0.
"""
import gc
import os
import random
import re
import struct
import sys
import tempfile
import unittest
import warnings
from fractions import Fraction

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
OLD = 0xDEADBEEF
CANARY = 0xCAFEBABE
HEAP_FILL = 0xEEEEEEEE
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# Representative binary32 bit encodings.
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_SUB_MIN = 0x00000001
F32_POS_SUB_MID = 0x00400000
F32_POS_SUB_MAX = 0x007FFFFF
F32_NEG_SUB_MIN = 0x80000001
F32_NEG_SUB_MAX = 0x807FFFFF
F32_POS_NORM_MIN = 0x00800000
F32_NEG_NORM_MIN = 0x80800000
F32_FLT_MAX = 0x7F7FFFFF
F32_NEG_FLT_MAX = 0xFF7FFFFF
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_QNAN = 0x7FC00000
F32_NEG_QNAN = 0xFFC00000
F32_QNAN_PAYLOAD = 0x7FC12345
F32_SNAN = 0x7F800001
F32_NEG_SNAN = 0xFF800001


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return F32_POS_INF if x > 0 else F32_NEG_INF


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


# -----------------------------------------------------------------------------
# Independent IEEE-754 Binary32 Fraction/RTE Reference Oracle
# -----------------------------------------------------------------------------

def f32_bits_to_fraction(bits: int) -> Fraction:
    """Exact rational value of a finite IEEE-754 binary32 bit encoding."""
    bits &= MASK32
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError(f"non-finite binary32 {bits:#010x}")
    if exp == 0:
        return sign * Fraction(man, 1 << 23) * (Fraction(2) ** -126)
    return sign * Fraction((1 << 23) | man, 1 << 23) * (Fraction(2) ** (exp - 127))


def rne_nonneg_fraction(q: Fraction) -> int:
    """Round a non-negative rational to the nearest integer, ties to even."""
    n = q.numerator // q.denominator
    rem = q - n
    half = Fraction(1, 2)
    if rem > half or (rem == half and (n & 1)):
        n += 1
    return n


def ref_cvt_pk_u8_byte(x_bits: int) -> int:
    """RDNA2 flt32_to_uint8: binary32 bits -> IEEE-754 RTE clamped [0, 255] (NaN -> 0)."""
    bits = x_bits & MASK32
    if is_f32_nan_bits(bits):
        return 0
    if (bits >> 31) or (bits & 0x7FFFFFFF) == 0:
        return 0
    if bits == F32_POS_INF:
        return 255
    q = f32_bits_to_fraction(bits)
    if q >= 255:
        return 255
    return min(255, rne_nonneg_fraction(q))


def ref_cvt_pk_u8_f32(x_bits: int, slot: int, base: int) -> int:
    """RDNA2 V_CVT_PK_U8_F32: insert ref_cvt_pk_u8_byte(x_bits) at byte (slot & 3) of base."""
    u = ref_cvt_pk_u8_byte(x_bits)
    sh = 8 * (slot & 3)
    return ((base & MASK32) & ~(0xFF << sh)) | (u << sh)


# Hand-worked reference expectations: (label, x_bits, slot, base, expected_dword).
HAND_WORKED = (
    ("slot0_old", f32_bits(18.0), 0, OLD, 0xDEADBE12),
    ("slot1_old", f32_bits(52.0), 1, OLD, 0xDEAD34EF),
    ("slot2_old", f32_bits(86.0), 2, OLD, 0xDE56BEEF),
    ("slot3_old", f32_bits(120.0), 3, OLD, 0x78ADBEEF),
    ("slot4_wraps_0", f32_bits(0xAB), 4, OLD, 0xDEADBEAB),
    ("slot5_wraps_1", f32_bits(0xCD), 5, OLD, 0xDEADCDEF),
    ("slot7_wraps_3", f32_bits(0x42), 7, OLD, 0x42ADBEEF),
    ("slot_neg1_wraps_3", f32_bits(0x99), -1, OLD, 0x99ADBEEF),
    ("tie_0_5_even", f32_bits(0.5), 0, OLD, 0xDEADBE00),
    ("tie_1_5_even", f32_bits(1.5), 1, OLD, 0xDEAD02EF),
    ("tie_2_5_even", f32_bits(2.5), 2, OLD, 0xDE02BEEF),
    ("tie_253_5_even", f32_bits(253.5), 0, OLD, 0xDEADBEFE),
    ("tie_254_5_even", f32_bits(254.5), 1, OLD, 0xDEADFEEF),
    ("clamp_255_49", f32_bits(255.49), 2, OLD, 0xDEFFBEEF),
    ("clamp_255_5", f32_bits(255.5), 3, OLD, 0xFFADBEEF),
    ("neg_0_001", f32_bits(-0.001), 0, OLD, 0xDEADBE00),
    ("neg_1000", f32_bits(-1000.0), 1, OLD, 0xDEAD00EF),
    ("neg_inf", F32_NEG_INF, 2, OLD, 0xDE00BEEF),
    ("pos_inf", F32_POS_INF, 3, OLD, 0xFFADBEEF),
    ("subnormal_pos", F32_POS_SUB_MIN, 0, OLD, 0xDEADBE00),
    ("subnormal_neg", F32_NEG_SUB_MAX, 1, OLD, 0xDEAD00EF),
    ("qnan", F32_QNAN, 0, OLD, 0xDEADBE00),
    ("neg_qnan", F32_NEG_QNAN, 2, OLD, 0xDE00BEEF),
    ("snan", F32_SNAN, 3, OLD, 0x00ADBEEF),
)


# -----------------------------------------------------------------------------
# Translation & Execution Helpers
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_valu_cvt_pk_u8_f32.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_valu_cvt_pk_u8_f32.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_valu_cvt_pk_u8_f32.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_valu_cvt_pk_u8_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_valu_cvt_pk_u8_f32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_mrt0(lines: list[str], regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0")) -> tuple[int, ...]:
    spv = translate_ps(list(lines) + [f"exp mrt0 {', '.join(regs)} done vm"])
    out = si.run_fragment(spv)
    if list(out) != ["%out_mrt0"]:
        raise AssertionError(f"expected %out_mrt0 only; got {sorted(out)}")
    return tuple(out["%out_mrt0"])


def run_cs(lines: list[str], group_size: int, subgroup_size: int, memory: dict[int, int]) -> dict[int, int]:
    if _TMP_DIR is None:
        setUpModule()
    # rsrc2 = 0x0088: USER_SGPR = 4 (s[0:3] V#), TGID_X_EN = 1 (s4), TIDIG_COMP_CNT = 0 (v0 = local_id.x).
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_valu_cvt_pk_u8_f32\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    return si.run_workgroup(
        spv, group_size, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )


class _Heap(dict):
    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


# -----------------------------------------------------------------------------
# 1. Reference Oracle Self-Tests
# -----------------------------------------------------------------------------

class OracleTests(unittest.TestCase):
    def test_hand_worked_constants_match_oracle(self):
        for label, x_bits, slot, base, want in HAND_WORKED:
            with self.subTest(case=label):
                self.assertEqual(ref_cvt_pk_u8_f32(x_bits, slot, base), want)

    def test_oracle_agrees_with_python_float_round_for_finite_values(self):
        for k in range(256):
            for delta in (-0.25, 0.0, 0.25, 0.5):
                x = k + delta
                xb = f32_bits(x)
                expected_u = min(255, max(0, round(bits_f32(xb))))
                self.assertEqual(ref_cvt_pk_u8_byte(xb), expected_u, f"x={x}")
        # Verify 1-ULP neighbors around .5 ties.
        for tie in (0.5, 1.5, 2.5, 127.5, 128.5, 253.5, 254.5):
            tb = f32_bits(tie)
            for b in (tb - 1, tb, tb + 1):
                expected_u = min(255, max(0, round(bits_f32(b))))
                self.assertEqual(ref_cvt_pk_u8_byte(b), expected_u, f"b={b:#010x}")


# -----------------------------------------------------------------------------
# 2. Byte Slots (0..3), Slot Masking (slot & 3), and Untouched Bytes
# -----------------------------------------------------------------------------

class ByteSlotAndMaskingTests(unittest.TestCase):
    def test_all_four_slots_preserve_other_three_bytes_of_base(self):
        # Distinct byte values including 0x00 and 0xFF (slot 3 gives 0xFFADBEEF / 0x00ADBEEF).
        byte_vals = (0.0, 1.0, 18.0, 127.0, 128.0, 254.0, 255.0)
        for suffix in ("", "_e64"):
            for slot in range(4):
                for start in range(0, len(byte_vals), 4):
                    chunk = byte_vals[start:start + 4]
                    lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
                    for lane, val in enumerate(chunk):
                        xb = f32_bits(val)
                        dst = f"v{10 + lane}"
                        lines += [
                            f"v_mov_b32 {dst}, {CANARY:#x}",
                            f"v_mov_b32 v1, {xb:#x}",
                            f"v_mov_b32 v2, {slot}",
                            f"v_cvt_pk_u8_f32{suffix} {dst}, v1, v2, v3",
                        ]
                        expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
                    while len(expected) < 4:
                        lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                        expected.append(0)
                    with self.subTest(suffix=suffix or "bare", slot=slot, batch=start // 4):
                        got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                        self.assertEqual(got, tuple(expected))

    def test_slot_and_3_masking_across_vgpr_sgpr_and_inline_operands(self):
        slots = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 0xFFFFFFFF, 0xFFFFFFFC, 0x80000001, 0x1234567A]
        payloads = [17.0, 65.0, 145.0, 243.0]

        for op_kind in ("vgpr", "sgpr", "inline"):
            for start in range(0, len(slots), 4):
                chunk = slots[start:start + 4]
                lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
                for idx, slot in enumerate(chunk):
                    x = payloads[(start + idx) % len(payloads)]
                    xb = f32_bits(x)
                    dst = f"v{10 + idx}"
                    lines.append(f"v_mov_b32 {dst}, {CANARY:#x}")
                    lines.append(f"v_mov_b32 v1, {xb:#x}")
                    if op_kind == "vgpr":
                        lines.append(f"v_mov_b32 v2, {slot & MASK32:#x}")
                        slot_arg = "v2"
                    elif op_kind == "sgpr":
                        lines.append(f"s_mov_b32 s4, {slot & MASK32:#x}")
                        slot_arg = "s4"
                    else:
                        slot_arg = str(slot) if -16 <= slot <= 64 else f"{slot & MASK32:#x}"
                    lines.append(f"v_cvt_pk_u8_f32_e64 {dst}, v1, {slot_arg}, v3")
                    expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
                while len(expected) < 4:
                    lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                    expected.append(0)
                with self.subTest(op_kind=op_kind, chunk=chunk):
                    got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                    self.assertEqual(got, tuple(expected))

    def test_base_from_sgpr_inline_and_sequential_four_byte_pack(self):
        # src2 from SGPR, inline 0, and inline -1.
        got_bases = run_ps_mrt0([
            f"v_mov_b32 v1, {f32_bits(0x4A):#x}",
            f"s_mov_b32 s5, {OLD:#x}",
            "v_cvt_pk_u8_f32 v10, v1, 2, s5",     # 0xDE4ABEEF
            "v_cvt_pk_u8_f32 v11, v1, 1, 0",      # 0x00004A00
            "v_cvt_pk_u8_f32 v12, v1, 3, -1",     # 0x4AFFFFFF
            "v_cvt_pk_u8_f32 v13, v1, 0, -1",     # 0xFFFFFF4A
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got_bases, (
            ref_cvt_pk_u8_f32(f32_bits(0x4A), 2, OLD),
            ref_cvt_pk_u8_f32(f32_bits(0x4A), 1, 0),
            ref_cvt_pk_u8_f32(f32_bits(0x4A), 3, MASK32),
            ref_cvt_pk_u8_f32(f32_bits(0x4A), 0, MASK32),
        ))

        # Pack 4 floats sequentially into bytes 0, 1, 2, 3 of v0 starting from OLD.
        got_packed = run_ps_mrt0([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {f32_bits(0x11):#x}",
            f"v_mov_b32 v2, {f32_bits(0x22):#x}",
            f"v_mov_b32 v3, {f32_bits(0x33):#x}",
            f"v_mov_b32 v4, {f32_bits(0x44):#x}",
            "v_cvt_pk_u8_f32 v0, v1, 0, v0",
            "v_cvt_pk_u8_f32 v0, v2, 1, v0",
            "v_cvt_pk_u8_f32 v0, v3, 2, v0",
            "v_cvt_pk_u8_f32 v0, v4, 3, v0",
        ])[0]
        self.assertEqual(got_packed, 0x44332211)


# -----------------------------------------------------------------------------
# 3. Clamping, Special Float Inputs, and .5 Round-to-Nearest-Even Ties
# -----------------------------------------------------------------------------

class ClampingAndSpecialInputsTests(unittest.TestCase):
    def test_negative_overflow_and_subnormal_clamping(self):
        cases = [
            # Negatives -> 0
            ("pos_zero", F32_POS_ZERO),
            ("neg_zero", F32_NEG_ZERO),
            ("neg_tiny", f32_bits(-1e-30)),
            ("neg_0_001", f32_bits(-0.001)),
            ("neg_half", f32_bits(-0.5)),
            ("neg_one", f32_bits(-1.0)),
            ("neg_127_5", f32_bits(-127.5)),
            ("neg_1000", f32_bits(-1000.0)),
            ("neg_flt_max", F32_NEG_FLT_MAX),
            ("neg_inf", F32_NEG_INF),
            # Overflow -> 255
            ("255_0", f32_bits(255.0)),
            ("255_001", f32_bits(255.001)),
            ("255_49", f32_bits(255.49)),
            ("255_5", f32_bits(255.5)),
            ("256_0", f32_bits(256.0)),
            ("512_0", f32_bits(512.0)),
            ("1e10", f32_bits(1e10)),
            ("flt_max", F32_FLT_MAX),
            ("pos_inf", F32_POS_INF),
            # Tiny subnormals & near-zero normals -> 0
            ("pos_sub_min", F32_POS_SUB_MIN),
            ("pos_sub_mid", F32_POS_SUB_MID),
            ("pos_sub_max", F32_POS_SUB_MAX),
            ("neg_sub_min", F32_NEG_SUB_MIN),
            ("neg_sub_max", F32_NEG_SUB_MAX),
            ("pos_norm_min", F32_POS_NORM_MIN),
            ("neg_norm_min", F32_NEG_NORM_MIN),
            # ULP boundaries around 0.5, 1.5, 254.5, 255.0
            ("0_5_minus_ulp", f32_bits(0.5) - 1),
            ("0_5_plus_ulp", f32_bits(0.5) + 1),
            ("1_5_minus_ulp", f32_bits(1.5) - 1),
            ("1_5_plus_ulp", f32_bits(1.5) + 1),
            ("254_5_minus_ulp", f32_bits(254.5) - 1),
            ("254_5_plus_ulp", f32_bits(254.5) + 1),
            ("255_0_minus_ulp", f32_bits(255.0) - 1),
            ("255_0_plus_ulp", f32_bits(255.0) + 1),
        ]
        for start in range(0, len(cases), 4):
            chunk = cases[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, (_, xb) in enumerate(chunk):
                slot = (start + idx) & 3
                dst = f"v{10 + idx}"
                lines += [
                    f"v_mov_b32 v1, {xb:#x}",
                    f"v_cvt_pk_u8_f32 {dst}, v1, {slot}, v3",
                ]
                expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
            while len(expected) < 4:
                lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                expected.append(0)
            with self.subTest(cases=[lbl for lbl, _ in chunk]):
                got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                self.assertEqual(got, tuple(expected))

    def test_seeded_finite_sweep(self):
        rng = random.Random(0xC08E)
        samples = [(f32_bits(rng.uniform(-20.0, 280.0)), rng.randrange(0, 8)) for _ in range(64)]
        for start in range(0, len(samples), 4):
            chunk = samples[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, (xb, slot) in enumerate(chunk):
                dst = f"v{10 + idx}"
                lines += [
                    f"v_mov_b32 v1, {xb:#x}",
                    f"v_cvt_pk_u8_f32 {dst}, v1, {slot}, v3",
                ]
                expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
            with self.subTest(batch=start // 4):
                self.assertEqual(run_ps_mrt0(lines, ("v10", "v11", "v12", "v13")), tuple(expected))


class RoundToNearestEvenTieTests(unittest.TestCase):
    def test_all_half_ties_from_0_5_to_255_5_and_255_49(self):
        ties = [k + 0.5 for k in range(256)] + [255.49]
        for start in range(0, len(ties), 4):
            chunk = ties[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, x in enumerate(chunk):
                xb = f32_bits(x)
                slot = (start + idx) & 3
                dst = f"v{10 + idx}"
                lines += [
                    f"v_mov_b32 v1, {xb:#x}",
                    f"v_cvt_pk_u8_f32 {dst}, v1, {slot}, v3",
                ]
                expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
            while len(expected) < 4:
                lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                expected.append(0)
            with self.subTest(ties=chunk):
                got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                self.assertEqual(got, tuple(expected))

    def test_all_exact_integers_0_to_255(self):
        for start in range(0, 256, 4):
            chunk = range(start, start + 4)
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, k in enumerate(chunk):
                xb = f32_bits(float(k))
                dst = f"v{10 + idx}"
                lines += [
                    f"v_mov_b32 v1, {xb:#x}",
                    f"v_cvt_pk_u8_f32 {dst}, v1, {idx}, v3",
                ]
                expected.append(ref_cvt_pk_u8_f32(xb, idx, OLD))
            with self.subTest(start=start):
                self.assertEqual(run_ps_mrt0(lines, ("v10", "v11", "v12", "v13")), tuple(expected))


# -----------------------------------------------------------------------------
# 4. VOP3 Modifiers, SGPR/Inline Float Sources, Destination Aliasing, EXEC
# -----------------------------------------------------------------------------

class SourceModifiersAndOperandsTests(unittest.TestCase):
    def test_vop3_source_modifiers(self):
        cases = [
            (f32_bits(-42.5), "-v1", 0, 42),       # -(-42.5) = 42.5 -> 42 (even)
            (f32_bits(-43.5), "-v1", 1, 44),       # -(-43.5) = 43.5 -> 44 (even)
            (f32_bits(42.5), "-v1", 2, 0),         # -(42.5) = -42.5 -> 0
            (F32_NEG_INF, "-v1", 3, 255),          # -(-inf) = +inf -> 255
            (F32_POS_INF, "-v1", 0, 0),            # -(+inf) = -inf -> 0
            (f32_bits(-13.5), "|v1|", 1, 14),      # |-13.5| = 13.5 -> 14 (even)
            (f32_bits(-14.5), "|v1|", 2, 14),      # |-14.5| = 14.5 -> 14 (even)
            (f32_bits(-300.0), "|v1|", 3, 255),    # |-300.0| = 300.0 -> 255
            (F32_NEG_INF, "|v1|", 0, 255),         # |-inf| = +inf -> 255
            (f32_bits(10.0), "-|v1|", 1, 0),       # -|10.0| = -10.0 -> 0
            (f32_bits(-10.0), "-|v1|", 2, 0),      # -|-10.0| = -10.0 -> 0
            (F32_NEG_INF, "-|v1|", 3, 0),          # -|-inf| = -inf -> 0
            (f32_bits(-1.5), "neg(v1)", 0, 2),     # neg(-1.5) = 1.5 -> 2
            (f32_bits(-2.5), "abs(v1)", 1, 2),     # abs(-2.5) = 2.5 -> 2
            (f32_bits(-7.5), "-|s4|", 2, 0),
            (f32_bits(-7.5), "|s4|", 3, 8),
        ]
        for start in range(0, len(cases), 4):
            chunk = cases[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, (xb, expr, slot, want_byte) in enumerate(chunk):
                dst = f"v{10 + idx}"
                reg = "s4" if "s4" in expr else "v1"
                lines += [
                    f"{'s_mov_b32' if reg == 's4' else 'v_mov_b32'} {reg}, {xb:#x}",
                    f"v_cvt_pk_u8_f32_e64 {dst}, {expr}, {slot}, v3",
                ]
                sh = 8 * (slot & 3)
                expected.append((OLD & ~(0xFF << sh)) | (want_byte << sh))
            while len(expected) < 4:
                lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                expected.append(0)
            with self.subTest(cases=[(e, s) for _, e, s, _ in chunk]):
                got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                self.assertEqual(got, tuple(expected))

    def test_inline_constant_and_sgpr_float_sources(self):
        inline_cases = [
            ("0.5", 0, 0),
            ("1.0", 1, 1),
            ("2.0", 2, 2),
            ("4.0", 3, 4),
            ("-1.0", 0, 0),
            ("-0.5", 1, 0),
            ("-2.0", 2, 0),
            ("-4.0", 3, 0),
            ("0.0", 0, 0),
            ("0", 1, 0),
        ]
        for start in range(0, len(inline_cases), 4):
            chunk = inline_cases[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, (lit, slot, want_byte) in enumerate(chunk):
                dst = f"v{10 + idx}"
                lines.append(f"v_cvt_pk_u8_f32 {dst}, {lit}, {slot}, v3")
                sh = 8 * (slot & 3)
                expected.append((OLD & ~(0xFF << sh)) | (want_byte << sh))
            while len(expected) < 4:
                lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                expected.append(0)
            with self.subTest(inlines=[c[0] for c in chunk]):
                got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                self.assertEqual(got, tuple(expected))

        # SGPR float sources across edge values.
        sgpr_cases = [
            (f32_bits(0.5), 0),
            (f32_bits(1.0), 1),
            (f32_bits(2.0), 2),
            (f32_bits(4.0), 3),
            (f32_bits(-1.0), 0),
            (f32_bits(127.5), 1),
            (f32_bits(254.5), 2),
            (F32_POS_INF, 3),
        ]
        for start in range(0, len(sgpr_cases), 4):
            chunk = sgpr_cases[start:start + 4]
            lines, expected = [f"v_mov_b32 v3, {OLD:#x}"], []
            for idx, (xb, slot) in enumerate(chunk):
                dst = f"v{10 + idx}"
                lines += [
                    f"s_mov_b32 s4, {xb:#x}",
                    f"v_cvt_pk_u8_f32_e64 {dst}, s4, {slot}, v3",
                ]
                expected.append(ref_cvt_pk_u8_f32(xb, slot, OLD))
            with self.subTest(batch=start // 4):
                got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                self.assertEqual(got, tuple(expected))

    def test_destination_aliasing(self):
        # 1. dst == src2: v_cvt_pk_u8_f32 v0, v1, 2, v0
        # 2. dst == src0: v_cvt_pk_u8_f32 v0, v0, 1, v3
        # 3. dst == src1: v_cvt_pk_u8_f32 v0, v1, v0, v3
        # 4. dst == src0 == src2: v_cvt_pk_u8_f32 v0, v0, 1, v0 (v0 holds 12.5 = 0x41480000 -> byte 12 at slot 1)
        got = run_ps_mrt0([
            f"v_mov_b32 v3, {OLD:#x}",
            f"v_mov_b32 v1, {f32_bits(86.0):#x}",
            f"v_mov_b32 v10, {OLD:#x}",
            "v_cvt_pk_u8_f32 v10, v1, 2, v10",
            f"v_mov_b32 v11, {f32_bits(35.5):#x}",
            "v_cvt_pk_u8_f32 v11, v11, 1, v3",
            "v_mov_b32 v12, 7",
            "v_cvt_pk_u8_f32 v12, v1, v12, v3",
            f"v_mov_b32 v13, {f32_bits(12.5):#x}",
            "v_cvt_pk_u8_f32 v13, v13, 1, v13",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (
            ref_cvt_pk_u8_f32(f32_bits(86.0), 2, OLD),
            ref_cvt_pk_u8_f32(f32_bits(35.5), 1, OLD),
            ref_cvt_pk_u8_f32(f32_bits(86.0), 7, OLD),
            ref_cvt_pk_u8_f32(f32_bits(12.5), 1, f32_bits(12.5)),
        ))

        # 5. All four operands alias v0: 0x41480003 is 12.500003 -> byte 13 (0x0D) at slot (0x41480003 & 3 == 3).
        all_alias_in = 0x41480003
        got_all = run_ps_mrt0([
            f"v_mov_b32 v0, {all_alias_in:#x}",
            "v_cvt_pk_u8_f32 v0, v0, v0, v0",
        ])[0]
        self.assertEqual(got_all, ref_cvt_pk_u8_f32(all_alias_in, all_alias_in, all_alias_in))
        self.assertEqual(got_all, 0x0D480003)


class ExecMaskingTests(unittest.TestCase):
    def test_single_lane_exec_mask_preserves_destination(self):
        # Inactive lane leaves v0 untouched; disabled instruction before re-enabling EXEC is skipped.
        got = run_ps_mrt0([
            f"v_mov_b32 v1, {f32_bits(77.0):#x}",
            f"v_mov_b32 v3, {CANARY:#x}",
            f"v_mov_b32 v10, {OLD:#x}",
            f"v_mov_b32 v11, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_cvt_pk_u8_f32 v10, v1, 1, v3",
            "v_cvt_pk_u8_f32 v11, v1, 0, v3",
            "s_mov_b32 exec_lo, -1",
            "v_cvt_pk_u8_f32 v11, v1, 2, v3",
        ], ("v10", "v11", "v10", "v11"))
        self.assertEqual(got[:2], (OLD, ref_cvt_pk_u8_f32(f32_bits(77.0), 2, CANARY)))

    def test_multilane_compute_exec_masking(self):
        # 16 lanes in lockstep: odd lanes are masked off via v_cmpx_eq_u32 before v_cvt_pk_u8_f32,
        # so active even lanes write the converted byte into OLD while inactive odd lanes keep OLD.
        lane_floats = [f32_bits(float(i * 15) + 0.5) for i in range(16)]
        mem = _Heap({0x100 + 4 * i: w for i, w in enumerate(lane_floats)})
        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            f"v_mov_b32_e32 v2, {OLD:#x}",
            f"v_mov_b32_e32 v3, {OLD:#x}",
            "v_and_b32_e32 v4, 1, v0",
            "v_cmpx_eq_u32_e32 0, v4",            # keep even lanes active
            "v_cvt_pk_u8_f32_e64 v2, v1, v0, v3",
            "s_mov_b32 exec_lo, -1",
            "buffer_store_dword v2, v5, s[0:3], 0 offen",
        ], group_size=16, subgroup_size=32, memory=mem)

        got = [mem.get(4 * i) for i in range(16)]
        want = [
            ref_cvt_pk_u8_f32(lane_floats[i], i, OLD) if (i & 1) == 0 else OLD
            for i in range(16)
        ]
        self.assertEqual(got, want)


# -----------------------------------------------------------------------------
# 5. NaN Inputs
# -----------------------------------------------------------------------------

class NaNHandlingTests(unittest.TestCase):
    """A NaN source gives byte 0 (measured on RDNA2, #358): the translator selects +0 before FClamp,
    which GLSL.std.450 leaves undefined for a NaN operand, so the interpreter now runs these cases."""

    def test_nan_gives_zero_byte(self):
        nan_cases = [
            ("qnan", F32_QNAN, "v1"),
            ("neg_qnan", F32_NEG_QNAN, "v1"),
            ("qnan_payload", F32_QNAN_PAYLOAD, "v1"),
            ("snan", F32_SNAN, "v1"),
            ("neg_snan", F32_NEG_SNAN, "v1"),
            ("qnan_abs", F32_NEG_QNAN, "|v1|"),
            ("qnan_neg", F32_QNAN, "-v1"),
        ]
        for label, xb, expr in nan_cases:
            with self.subTest(case=label):
                spv = translate_ps([
                    f"v_mov_b32 v1, {xb:#x}",
                    f"v_mov_b32 v3, {OLD:#x}",
                    f"v_cvt_pk_u8_f32_e64 v0, {expr}, 0, v3",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                ])
                self.assertIn("glsl FClamp", spv)
                self.assertIn("glsl RoundEven", spv)
                self.assertIn("OpConvertFToU", spv)
                self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], ref_cvt_pk_u8_f32(xb, 0, OLD))


# -----------------------------------------------------------------------------
# 6. Negative Controls (SPIR-V Sabotage Detection)
# -----------------------------------------------------------------------------

class NegativeControlsTests(unittest.TestCase):
    def test_sabotage_round_even_to_trunc(self):
        # 1.5 -> RoundEven gives 2, whereas Trunc gives 1.
        good = translate_ps([
            f"v_mov_b32 v1, {f32_bits(1.5):#x}",
            f"v_mov_b32 v3, {OLD:#x}",
            "v_cvt_pk_u8_f32 v0, v1, 1, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        want = ref_cvt_pk_u8_f32(f32_bits(1.5), 1, OLD)
        self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], want)

        bad, n = re.subn(r"\bRoundEven\b", "Trunc", good)
        self.assertEqual(n, 1)
        got_bad = si.run_fragment(bad)["%out_mrt0"][0]
        self.assertNotEqual(got_bad, want)
        self.assertEqual(got_bad, 0xDEAD01EF)

    def test_sabotage_missing_slot_shift(self):
        # Removing the * 8 shift (OpIMul %uint %slot3 %c8 -> %c0) writes slot 0 instead of slot 2.
        good = translate_ps([
            f"v_mov_b32 v1, {f32_bits(0x42):#x}",
            f"v_mov_b32 v3, {OLD:#x}",
            "v_cvt_pk_u8_f32 v0, v1, 2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        want = ref_cvt_pk_u8_f32(f32_bits(0x42), 2, OLD)
        self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], want)

        bad, n = re.subn(r"OpIMul %uint (%\w+) %c8\b", r"OpIMul %uint \1 %c0", good)
        self.assertEqual(n, 1)
        got_bad = si.run_fragment(bad)["%out_mrt0"][0]
        self.assertNotEqual(got_bad, want)
        self.assertEqual(got_bad, ref_cvt_pk_u8_f32(f32_bits(0x42), 0, OLD))

    def test_sabotage_missing_slot_mask(self):
        # Widening slot & 3 to slot & 7 breaks slot 5 (shift = 40 >= 32 -> undefined in SPIR-V).
        good = translate_ps([
            f"v_mov_b32 v1, {f32_bits(0x42):#x}",
            f"v_mov_b32 v2, 5",
            f"v_mov_b32 v3, {OLD:#x}",
            "v_cvt_pk_u8_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        want = ref_cvt_pk_u8_f32(f32_bits(0x42), 5, OLD)
        self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], want)

        bad, n = re.subn(r"OpBitwiseAnd %uint (%\w+) %c3\b", r"OpBitwiseAnd %uint \1 %c7", good)
        self.assertEqual(n, 1)
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(bad)


if __name__ == "__main__":
    unittest.main()
