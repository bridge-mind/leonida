"""Semantic value tests for f32 class, round-to-even, and rcp_iflag VALU ops:
  v_cmp_class_f32   D.u[lane] = ((class_bit(S0.f32) & S1.u[9:0]) != 0) if EXEC[lane] else 0
  v_cmpx_class_f32  EXEC[lane] &= ((class_bit(S0.f32) & S1.u[9:0]) != 0)  (preserves VCC)
  v_rndne_f32       D.f32 = roundToIntegralTiesToEven(S0.f32)
  v_rcp_iflag_f32   D.f32 = 1.0 / S0.f32  (RDNA2 1-ULP approximate reciprocal, no FP flags)

IEEE-754 binary32 class mask bits for v_cmp_class_f32 / v_cmpx_class_f32 (AMD RDNA2 ISA, S1[9:0]):
  bit 0 (0x001): sNaN         bit 5 (0x020): -0.0
  bit 1 (0x002): qNaN         bit 6 (0x040): +0.0
  bit 2 (0x004): -inf         bit 7 (0x080): +subnormal
  bit 3 (0x008): -normal      bit 8 (0x100): +normal
  bit 4 (0x010): -subnormal   bit 9 (0x200): +inf
Bits [31:10] of the mask operand are ignored.

Executed end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py (both single-lane
run_fragment and multi-lane compute run_workgroup), and checked against hand-computed constants and independent
plain-Python RDNA2 ISA reference functions. Standard library only; hand-written RDNA2 listings only.

Findings & exclusions (no shared files modified):
  1. v_rcp_iflag_f32 lowers to `OpFDiv %float %f1 {x}` followed by the VOP3 output modifiers (`clamp`, `mul:2`,
     `mul:4`, `div:2`) through `clamp_mods`, as `v_rcp_f32` does. When this file was written the translator dropped
     those modifiers and spirv_mini_interp.py had no `OpFDiv`; both have changed since (FUZZ-3, ed6da6b6), so
     RcpIflagF32OracleAndTranslationTests now executes it on exact inputs (powers of two, so the 1-ULP hardware
     approximation and the interpreter's exact division agree) with each output modifier. A zero operand stays
     out: `OpFDiv` by zero is undefined in SPIR-V and the interpreter refuses it (docs/design.md, "Denormal flush
     and special results", Not modelled yet). gfx1036 hardware agrees with the translator on 2,570 of 3,000 fuzzed
     programs with clamp/omod and differs on none (`tools/f32_fuzz.py --oracle translator,hw --ops
     v_rcp_iflag_f32`, seeds 1 to 3; the rest are division by zero or cases the model does not judge).
  2. NaN caveats in SPIR-V / spirv_mini_interp.py:
     - For `v_rndne_f32` on NaN (`qNaN` / `sNaN`), SPIR-V / GLSL.std.450 `RoundEven` leaves NaN payload/quieting
       unspecified (spirv_mini_interp.py returns the input NaN unchanged; hardware quiets `sNaN`). Tests assert
       that the result is a binary32 NaN rather than pinning an exact NaN payload.
     - For `v_cmp_class_f32` / `v_cmpx_class_f32` VOP3 source modifiers (`-`, `|..|`), tools/rdna2spirv.py emits
       float `OpFNegate`/`FAbs`, which in spirv_mini_interp.py goes through Python float packing and quiets `sNaN`
       (see tests/semantic/test_interp.py). Source modifiers are therefore tested on `+-inf`, `+-normal`,
       `+-subnormal`, `+-0.0`, and `qNaN`; raw `sNaN` (`+sNaN` / `-sNaN`) is tested without float modifiers.
"""
import gc
import math
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
HEAP_FILL = 0xEEEEEEEE

# Class mask bits (AMD RDNA2 ISA V_CMP_CLASS_F32).
CLASS_SNAN = 1 << 0
CLASS_QNAN = 1 << 1
CLASS_NINF = 1 << 2
CLASS_NNORM = 1 << 3
CLASS_NSUBNORM = 1 << 4
CLASS_NZERO = 1 << 5
CLASS_PZERO = 1 << 6
CLASS_PSUBNORM = 1 << 7
CLASS_PNORM = 1 << 8
CLASS_PINF = 1 << 9
ALL_CLASS_MASK = 0x3FF


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(float("inf"), x)


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


# -----------------------------------------------------------------------------
# Independent RDNA2 ISA Reference Functions & Hand-Worked Constants
# -----------------------------------------------------------------------------

def f32_class_bit(bits: int) -> int:
    """Return the single RDNA2 class bit (1 << 0..9) for IEEE-754 binary32 word `bits`."""
    bits &= MASK32
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    if exp == 0xFF:
        if man == 0:
            return CLASS_NINF if sign else CLASS_PINF
        return CLASS_QNAN if (man & 0x400000) else CLASS_SNAN
    if exp == 0:
        if man == 0:
            return CLASS_NZERO if sign else CLASS_PZERO
        return CLASS_NSUBNORM if sign else CLASS_PSUBNORM
    return CLASS_NNORM if sign else CLASS_PNORM


def ref_cmp_class_f32(x_bits: int, mask: int) -> bool:
    """RDNA2 ISA V_CMP_CLASS_F32: True iff x's IEEE-754 class bit is selected in mask[9:0]."""
    return bool(f32_class_bit(x_bits) & (mask & ALL_CLASS_MASK))


def ref_rndne_f32(x_bits: int) -> int:
    """RDNA2 ISA V_RNDNE_F32: IEEE-754 roundToIntegralTiesToEven preserving sign of zero."""
    bits = x_bits & MASK32
    if is_f32_nan_bits(bits):
        return bits | 0x00400000
    x = bits_f32(bits)
    if not math.isfinite(x):
        return bits
    return f32_bits(math.copysign(float(round(x)), x))


def ref_rcp_iflag_f32(x_bits: int) -> int:
    """RDNA2 ISA V_RCP_IFLAG_F32 (idealized exact 1/x for powers of two, specials, and normals)."""
    bits = x_bits & MASK32
    if is_f32_nan_bits(bits):
        return 0x7FC00000
    sign = bits & 0x80000000
    absu = bits & 0x7FFFFFFF
    if absu == 0:
        return sign | 0x7F800000   # +-0.0 -> +-inf
    if absu == 0x7F800000:
        return sign                # +-inf -> +-0.0
    return f32_bits(1.0 / bits_f32(bits))


# Hand-worked classification samples covering all 10 IEEE-754 classes (including boundaries and signs).
CLASS_SAMPLES = (
    ("pos_snan_min", 0x7F800001, CLASS_SNAN),
    ("pos_snan_max", 0x7FBFFFFF, CLASS_SNAN),
    ("neg_snan_min", 0xFF800001, CLASS_SNAN),
    ("neg_snan_max", 0xFFBFFFFF, CLASS_SNAN),
    ("pos_qnan", 0x7FC00000, CLASS_QNAN),
    ("pos_qnan_payload", 0x7FFFFFFF, CLASS_QNAN),
    ("neg_qnan", 0xFFC00000, CLASS_QNAN),
    ("neg_qnan_payload", 0xFFFFFFFF, CLASS_QNAN),
    ("neg_inf", 0xFF800000, CLASS_NINF),
    ("neg_one", 0xBF800000, CLASS_NNORM),
    ("neg_norm_min", 0x80800000, CLASS_NNORM),
    ("neg_norm_max", 0xFF7FFFFF, CLASS_NNORM),
    ("neg_subnorm_min", 0x80000001, CLASS_NSUBNORM),
    ("neg_subnorm_max", 0x807FFFFF, CLASS_NSUBNORM),
    ("neg_zero", 0x80000000, CLASS_NZERO),
    ("pos_zero", 0x00000000, CLASS_PZERO),
    ("pos_subnorm_min", 0x00000001, CLASS_PSUBNORM),
    ("pos_subnorm_max", 0x007FFFFF, CLASS_PSUBNORM),
    ("pos_one", 0x3F800000, CLASS_PNORM),
    ("pos_norm_min", 0x00800000, CLASS_PNORM),
    ("pos_norm_max", 0x7F7FFFFF, CLASS_PNORM),
    ("pos_inf", 0x7F800000, CLASS_PINF),
)

# Hand-worked V_RNDNE_F32 expectations (binary32 bits in -> binary32 bits out).
RNDNE_HAND_WORKED = (
    ("pos_zero", 0x00000000, 0x00000000),
    ("neg_zero", 0x80000000, 0x80000000),
    ("pos_half", 0x3F000000, 0x00000000),          # +0.5 -> +0.0 (even)
    ("neg_half", 0xBF000000, 0x80000000),          # -0.5 -> -0.0 (even, sign preserved)
    ("pos_1_5", 0x3FC00000, 0x40000000),           # +1.5 -> +2.0 (even)
    ("neg_1_5", 0xBFC00000, 0xC0000000),           # -1.5 -> -2.0 (even)
    ("pos_2_5", 0x40200000, 0x40000000),           # +2.5 -> +2.0 (even)
    ("neg_2_5", 0xC0200000, 0xC0000000),           # -2.5 -> -2.0 (even)
    ("pos_3_5", 0x40600000, 0x40800000),           # +3.5 -> +4.0 (even)
    ("neg_3_5", 0xC0600000, 0xC0800000),           # -3.5 -> -4.0 (even)
    ("pos_4_5", 0x40900000, 0x40800000),           # +4.5 -> +4.0 (even)
    ("neg_4_5", 0xC0900000, 0xC0800000),           # -4.5 -> -4.0 (even)
    ("pos_half_minus_ulp", 0x3EFFFFFF, 0x00000000),
    ("pos_half_plus_ulp", 0x3F000001, 0x3F800000),
    ("neg_half_plus_ulp", 0xBEFFFFFF, 0x80000000),
    ("neg_half_minus_ulp", 0xBF000001, 0xBF800000),
    ("pos_1_5_minus_ulp", 0x3FBFFFFF, 0x3F800000),
    ("pos_1_5_plus_ulp", 0x3FC00001, 0x40000000),
    ("pos_2_5_minus_ulp", 0x401FFFFF, 0x40000000),
    ("pos_2_5_plus_ulp", 0x40200001, 0x40400000),
    ("pos_quarter", 0x3E800000, 0x00000000),
    ("neg_quarter", 0xBE800000, 0x80000000),
    ("pos_three_quarter", 0x3F400000, 0x3F800000),
    ("neg_three_quarter", 0xBF400000, 0xBF800000),
    ("pos_subnorm_min", 0x00000001, 0x00000000),
    ("pos_subnorm_max", 0x007FFFFF, 0x00000000),
    ("neg_subnorm_min", 0x80000001, 0x80000000),
    ("neg_subnorm_max", 0x807FFFFF, 0x80000000),
    ("pos_norm_min", 0x00800000, 0x00000000),
    ("neg_norm_min", 0x80800000, 0x80000000),
    ("pos_2p22_plus_half", 0x4A800001, 0x4A800000),   # 4194304.5 -> 4194304.0 (even)
    ("pos_2p22_plus_1_5", 0x4A800003, 0x4A800004),    # 4194305.5 -> 4194306.0 (even)
    ("neg_2p22_minus_half", 0xCA800001, 0xCA800000),  # -4194304.5 -> -4194304.0 (even)
    ("neg_2p22_minus_1_5", 0xCA800003, 0xCA800004),   # -4194305.5 -> -4194306.0 (even)
    ("pos_2p23_minus_half", 0x4AFFFFFF, 0x4B000000),  # 8388607.5 -> 8388608.0 (even)
    ("pos_2p23_plus_1", 0x4B000001, 0x4B000001),      # 8388609.0 -> exact
    ("neg_2p23_minus_1", 0xCB000001, 0xCB000001),     # -8388609.0 -> exact
    ("flt_max", 0x7F7FFFFF, 0x7F7FFFFF),
    ("neg_flt_max", 0xFF7FFFFF, 0xFF7FFFFF),
    ("pos_inf", 0x7F800000, 0x7F800000),
    ("neg_inf", 0xFF800000, 0xFF800000),
)

# Hand-worked V_RCP_IFLAG_F32 expectations (reference oracle only; excluded from spirv_mini_interp execution).
RCP_IFLAG_HAND_WORKED = (
    ("pos_one", 0x3F800000, 0x3F800000),         # 1.0 -> 1.0
    ("neg_one", 0xBF800000, 0xBF800000),         # -1.0 -> -1.0
    ("pos_two", 0x40000000, 0x3F000000),         # 2.0 -> 0.5
    ("neg_two", 0xC0000000, 0xBF000000),         # -2.0 -> -0.5
    ("pos_half", 0x3F000000, 0x40000000),        # 0.5 -> 2.0
    ("neg_half", 0xBF000000, 0xC0000000),        # -0.5 -> -2.0
    ("pos_four", 0x40800000, 0x3E800000),        # 4.0 -> 0.25
    ("neg_quarter", 0xBE800000, 0xC0800000),     # -0.25 -> -4.0
    ("pos_2p20", 0x49800000, 0x35800000),        # 2^20 -> 2^-20
    ("neg_2pm20", 0xB5800000, 0xC9800000),       # -2^-20 -> -2^20
    ("pos_1_5", 0x3FC00000, 0x3F2AAAAB),         # 1.5 -> ~0.6666667
    ("neg_3_0", 0xC0400000, 0xBEAAAAAB),         # -3.0 -> ~-0.33333334
    ("pos_zero", 0x00000000, 0x7F800000),        # +0.0 -> +inf
    ("neg_zero", 0x80000000, 0xFF800000),        # -0.0 -> -inf
    ("pos_inf", 0x7F800000, 0x00000000),         # +inf -> +0.0
    ("neg_inf", 0xFF800000, 0x80000000),         # -inf -> -0.0
    ("pos_qnan", 0x7FC00000, 0x7FC00000),        # qNaN -> qNaN
    ("pos_snan", 0x7F800001, 0x7FC00000),        # sNaN -> qNaN
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
    _PS_PATH = os.path.join(disasm_dir, "ps_valu_f32_class_rndne.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_valu_f32_class_rndne.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_valu_f32_class_rndne.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_valu_f32_class_rndne.dxbc"), "wb") as f:
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
    asm = "; test_valu_f32_class_rndne\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
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


def run_ps_raw(lines: list[str]) -> dict:
    spv = translate_ps(lines)
    return si.run_fragment(spv)


def run_cs(lines: list[str], group_size: int, subgroup_size: int, memory: dict[int, int]) -> dict[int, int]:
    if _TMP_DIR is None:
        setUpModule()
    # rsrc2 = 0x0088: USER_SGPR = 4 (s[0:3] V#), TGID_X_EN = 1 (s4), TIDIG_COMP_CNT = 0 (v0 = local_id.x).
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_valu_f32_class_rndne\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
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
    """Flat memory initialized to HEAP_FILL for unwritten addresses."""

    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


# -----------------------------------------------------------------------------
# Oracle Self-Tests
# -----------------------------------------------------------------------------

class OracleTests(unittest.TestCase):
    """Independent hand-worked constants agree with the plain-Python ISA reference functions."""

    def test_class_samples_each_match_exactly_one_bit(self):
        self.assertEqual(len({bit for _, _, bit in CLASS_SAMPLES}), 10)
        for name, x_bits, expected_bit in CLASS_SAMPLES:
            with self.subTest(sample=name, bits=f"{x_bits:#010x}"):
                self.assertEqual(f32_class_bit(x_bits), expected_bit)
                for b in range(10):
                    self.assertEqual(ref_cmp_class_f32(x_bits, 1 << b), (1 << b) == expected_bit)
                self.assertFalse(ref_cmp_class_f32(x_bits, 0))
                self.assertTrue(ref_cmp_class_f32(x_bits, ALL_CLASS_MASK))

    def test_rndne_hand_worked_constants_match_reference(self):
        for name, x_bits, want in RNDNE_HAND_WORKED:
            with self.subTest(case=name, x=f"{x_bits:#010x}"):
                self.assertEqual(ref_rndne_f32(x_bits), want)

    def test_rcp_iflag_hand_worked_constants_match_reference(self):
        for name, x_bits, want in RCP_IFLAG_HAND_WORKED:
            with self.subTest(case=name, x=f"{x_bits:#010x}"):
                self.assertEqual(ref_rcp_iflag_f32(x_bits), want)


# -----------------------------------------------------------------------------
# 1. v_cmp_class_f32
# -----------------------------------------------------------------------------

class CmpClassF32Tests(unittest.TestCase):
    def test_all_10_class_bits_exhaustive_matrix_e32_and_e64(self):
        # Every sample x every single-bit mask (0..9) + mask 0 + 0x3FF, batched 4 per shader.
        masks = [1 << b for b in range(10)] + [0, ALL_CLASS_MASK]
        cases = [(name, x_bits, mask) for name, x_bits, _ in CLASS_SAMPLES for mask in masks]
        for form in ("e32", "e64"):
            for start in range(0, len(cases), 4):
                chunk = cases[start:start + 4]
                lines = []
                expected = []
                for slot, (_, x_bits, mask) in enumerate(chunk):
                    dst = f"v{10 + slot}"
                    lines += [f"v_mov_b32 v1, {x_bits:#x}", f"v_mov_b32 v2, {mask:#x}"]
                    if form == "e32":
                        lines += ["v_cmp_class_f32_e32 vcc_lo, v1, v2", f"v_cndmask_b32_e32 {dst}, 0, 1, vcc_lo"]
                    else:
                        lines += ["v_cmp_class_f32_e64 s4, v1, v2", f"v_cndmask_b32_e64 {dst}, 0, 1, s4"]
                    expected.append(int(ref_cmp_class_f32(x_bits, mask)))
                while len(expected) < 4:
                    lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                    expected.append(0)
                with self.subTest(form=form, batch=start // 4):
                    self.assertEqual(run_ps_mrt0(lines, ("v10", "v11", "v12", "v13")), tuple(expected))

    def test_multi_bit_masks_and_ignored_upper_mask_bits(self):
        multi_masks = [
            0x000,                   # none
            0x3FF,                   # all 10 classes
            0x003,                   # any NaN (sNaN | qNaN)
            0x204,                   # any Inf (-inf | +inf)
            0x060,                   # any zero (-0.0 | +0.0)
            0x090,                   # any subnormal (-subnorm | +subnorm)
            0x108,                   # any normal (-norm | +norm)
            0x1E0,                   # non-negative finite (+0 | +subnorm | +norm)
            0x03C,                   # negative non-NaN (-inf | -norm | -subnorm | -0)
            0x207,                   # non-finite (NaNs | Infs)
            0xFFFFFC00,              # upper bits [31:10] only -> always 0
            0x80000100,              # upper bit + +normal -> matches +normal only
        ]
        # Representative value from each of the 10 classes.
        reps = [
            0x7F800001, 0x7FC00000, 0xFF800000, 0xBF800000, 0x80000001,
            0x80000000, 0x00000000, 0x00000001, 0x3F800000, 0x7F800000,
        ]
        cases = [(x_bits, m) for x_bits in reps for m in multi_masks]
        for start in range(0, len(cases), 4):
            chunk = cases[start:start + 4]
            lines, expected = [], []
            for slot, (x_bits, mask) in enumerate(chunk):
                dst = f"v{10 + slot}"
                lines += [
                    f"v_mov_b32 v1, {x_bits:#x}",
                    f"v_mov_b32 v2, {mask:#x}",
                    "v_cmp_class_f32_e64 s4, v1, v2",
                    f"v_cndmask_b32_e64 {dst}, 0, 1, s4",
                ]
                expected.append(int(ref_cmp_class_f32(x_bits, mask)))
            while len(expected) < 4:
                lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                expected.append(0)
            with self.subTest(batch=start // 4):
                self.assertEqual(run_ps_mrt0(lines, ("v10", "v11", "v12", "v13")), tuple(expected))

    def test_encodings_inline_constants_sgprs_and_modifiers(self):
        # 2-arg _e32 (implicit vcc), bare mnemonic, inline float/int constants, SGPR src0/src1, and wave64 s[4:5].
        got = run_ps_mrt0([
            "v_mov_b32 v1, 0x3f800000",          # +1.0 (+normal)
            "v_mov_b32 v2, 0x100",
            "v_cmp_class_f32_e32 v1, v2",        # 2-operand form writes %vcc
            "v_cndmask_b32_e32 v10, 0, 1, vcc_lo",
            "v_cmp_class_f32 vcc_lo, -0.0, 32",  # bare mnemonic, inline -0.0 and mask 32 (-0.0)
            "v_cndmask_b32_e32 v11, 0, 1, vcc_lo",
            "s_mov_b32 s5, 0x80000001",          # -subnormal in SGPR
            "s_mov_b32 s6, 0x010",
            "v_cmp_class_f32_e64 s4, s5, s6",
            "v_cndmask_b32_e64 v12, 0, 1, s4",
            "v_cmp_class_f32_e64 vcc_lo, 0.5, -1",  # inline 0.5 & -1 (all 10 class bits)
            "v_cndmask_b32_e32 v13, 0, 1, vcc_lo",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (1, 1, 1, 1))

        # Source modifiers (-, |..|, -|..|) apply before classification on non-sNaN values.
        mod_cases = [
            (0x80000000, "|v1|", CLASS_PZERO, 1),       # |-0.0| -> +0.0
            (0x80000000, "|v1|", CLASS_NZERO, 0),
            (0x00000000, "-v1", CLASS_NZERO, 1),        # -(+0.0) -> -0.0
            (0xFF800000, "-v1", CLASS_PINF, 1),         # -(-inf) -> +inf
            (0x7F800000, "-|v1|", CLASS_NINF, 1),       # -|+inf| -> -inf
            (0x80000001, "|v1|", CLASS_PSUBNORM, 1),    # |-subnorm| -> +subnorm
            (0x00000001, "-v1", CLASS_NSUBNORM, 1),     # -(+subnorm) -> -subnorm
            (0x3F800000, "-v1", CLASS_NNORM, 1),        # -(+1.0) -> -1.0
            (0xFFC00000, "|v1|", CLASS_QNAN, 1),        # |-qNaN| -> qNaN
        ]
        for x_bits, src_expr, mask, want in mod_cases:
            with self.subTest(x=f"{x_bits:#010x}", src=src_expr, mask=hex(mask)):
                res = run_ps_mrt0([
                    f"v_mov_b32 v1, {x_bits:#x}",
                    f"v_mov_b32 v2, {mask:#x}",
                    f"v_cmp_class_f32_e64 s4, {src_expr}, v2",
                    "v_cndmask_b32_e64 v0, 0, 1, s4",
                ])[0]
                self.assertEqual(res, want)

        # Wave64 s[4:5] and vcc destinations.
        w64 = run_ps_mrt0([
            "s_mov_b64 exec, -1",
            "v_mov_b32 v1, 0x7f800000",          # +inf
            "v_mov_b32 v2, 0x200",
            "v_cmp_class_f32_e64 s[4:5], v1, v2",
            "v_cndmask_b32_e64 v10, 0, 1, s[4:5]",
            "v_cmp_class_f32_e32 vcc, 0x7f800000, v2",
            "v_cndmask_b32_e32 v11, 0, 1, vcc",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(w64, (1, 1, 0, 0))

    def test_inactive_exec_lane_writes_zero_to_mask_dst(self):
        # Hardware VOPC rule: inactive lanes write 0 into the destination mask even when operands match.
        got = run_ps_mrt0([
            "v_mov_b32 v1, 0x3f800000",          # +1.0 (+normal)
            "v_mov_b32 v2, 0x100",               # matches +normal when active
            "s_mov_b32 vcc_lo, -1",              # pre-fill with all-1s; must be cleared to 0
            "s_mov_b32 s4, -1",
            "s_mov_b32 exec_lo, 0",              # disable lane BEFORE cmp_class
            "v_cmp_class_f32_e32 vcc_lo, v1, v2",
            "v_cmp_class_f32_e64 s4, v1, v2",
            "s_mov_b32 exec_lo, -1",
            "v_cndmask_b32_e32 v10, 0, 1, vcc_lo",
            "v_cndmask_b32_e64 v11, 0, 1, s4",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (0, 0, 0, 0))


# -----------------------------------------------------------------------------
# 2. v_cmpx_class_f32
# -----------------------------------------------------------------------------

class CmpxClassF32Tests(unittest.TestCase):
    def test_single_lane_exec_update_vcc_preservation_and_kill(self):
        # Matching lane keeps EXEC active and preserves VCC; non-matching lane disables EXEC and preserves v0.
        for mnemonic in ("v_cmpx_class_f32_e32", "v_cmpx_class_f32_e64", "v_cmpx_class_f32"):
            with self.subTest(mnemonic=mnemonic, match=True):
                got = run_ps_mrt0([
                    "s_mov_b32 vcc_lo, -1",
                    "v_mov_b32 v0, 0x11111111",
                    "v_mov_b32 v1, 0x7fc00000",      # qNaN
                    "v_mov_b32 v2, 0x002",
                    f"{mnemonic} v1, v2",            # matches -> EXEC stays 1, VCC stays -1
                    "v_cndmask_b32_e32 v0, 0x22222222, 0x33333333, vcc_lo",
                    "s_mov_b32 exec_lo, -1",
                ])[0]
                self.assertEqual(got, 0x33333333)

            with self.subTest(mnemonic=mnemonic, match=False):
                got = run_ps_mrt0([
                    "s_mov_b32 vcc_lo, -1",
                    "v_mov_b32 v0, 0x11111111",
                    "v_mov_b32 v1, 0x80000000",      # -0.0
                    "v_mov_b32 v2, 0x040",           # +0.0 only -> mismatch
                    f"{mnemonic} v1, v2",            # clears EXEC, must NOT alter VCC
                    "v_mov_b32 v0, 0x99999999",      # masked out while EXEC == 0
                    "s_mov_b32 exec_lo, -1",
                    "v_cndmask_b32_e32 v10, 0x44, 0x55, vcc_lo",
                ], ("v0", "v10", "v0", "v10"))
                self.assertEqual(got, (0x11111111, 0x55, 0x11111111, 0x55))

        # Without restoring EXEC before exp/s_endpgm, a non-matching PS lane executes OpKill -> {}.
        self.assertEqual(run_ps_raw([
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x040",
            "v_cmpx_class_f32_e32 v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]), {})
        self.assertEqual(run_ps_raw([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x020",
            "v_cmpx_class_f32_e32 v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]), {"%out_mrt0": (0x12345678,) * 4})

    def test_single_lane_intersects_exec_and_supports_source_modifiers(self):
        # A pre-disabled lane MUST stay disabled even when v_cmpx_class_f32 matches.
        got = run_ps_mrt0([
            "v_mov_b32 v0, 0xAAAAAAAA",
            "v_mov_b32 v1, 0x7f800000",          # +inf
            "v_mov_b32 v2, 0x200",               # +inf mask
            "s_mov_b32 exec_lo, 0",
            "v_cmpx_class_f32_e64 v1, v2",
            "v_mov_b32 v0, 0xBBBBBBBB",          # must NOT write
            "s_mov_b32 exec_lo, -1",
        ])[0]
        self.assertEqual(got, 0xAAAAAAAA)

        # VOP3 source modifiers: |-0.0| matches +0.0 (0x040).
        got_mod = run_ps_mrt0([
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x040",
            "v_cmpx_class_f32_e64 |v1|, v2",
            "v_mov_b32 v0, 0x22222222",
            "s_mov_b32 exec_lo, -1",
        ])[0]
        self.assertEqual(got_mod, 0x22222222)

    def test_multilane_exec_masking_intersection_and_vcc_preservation(self):
        # 16 lanes in lockstep: lanes 0..11 hold distinct floats covering all 10 IEEE-754 classes;
        # lanes 12..15 hold matching values too, but lane 0 and lane 15 are disabled BEFORE v_cmpx_class_f32.
        # Stage 1 (e32): keep {sNaN, qNaN, -inf, -subnorm, -0.0, +0.0, +inf} = 0x277.
        # Stage 2 (e64): narrow further to {qNaN, -0.0, +0.0, +inf} = 0x262.
        # VCC is set beforehand for odd lanes (`(lane & 1) != 0`): active surviving lanes write 0x22 (VCC=1)
        # or 0x11 (VCC=0), while dropped or pre-disabled lanes leave memory at HEAP_FILL (0xEEEEEEEE).
        lane_inputs = [
            0x7FC00000,  # lane 0: +qNaN (matches both, BUT pre-disabled -> must NOT write)
            0x7F800001,  # lane 1: +sNaN (matches stage 1, dropped by stage 2)
            0x7FC00000,  # lane 2: +qNaN (survives both, even -> 0x11)
            0xFF800000,  # lane 3: -inf  (matches stage 1, dropped by stage 2)
            0xBF800000,  # lane 4: -1.0  (-normal, dropped by stage 1)
            0x80000001,  # lane 5: -subnorm (matches stage 1, dropped by stage 2)
            0x80000000,  # lane 6: -0.0  (survives both, even -> 0x11)
            0x00000000,  # lane 7: +0.0  (survives both, odd -> 0x22)
            0x00000001,  # lane 8: +subnorm (dropped by stage 1)
            0x3F800000,  # lane 9: +1.0 (+normal, dropped by stage 1)
            0x7F800000,  # lane 10: +inf (survives both, even -> 0x11)
            0xFFC00000,  # lane 11: -qNaN (survives both, odd -> 0x22)
            0xFF800001,  # lane 12: -sNaN (dropped by stage 2)
            0x7F800000,  # lane 13: +inf (survives both, odd -> 0x22)
            0x80000000,  # lane 14: -0.0 (survives both, even -> 0x11)
            0x7FC00000,  # lane 15: +qNaN (matches both, BUT pre-disabled -> must NOT write)
        ]
        mask1 = CLASS_SNAN | CLASS_QNAN | CLASS_NINF | CLASS_NSUBNORM | CLASS_NZERO | CLASS_PZERO | CLASS_PINF  # 0x277
        mask2 = CLASS_QNAN | CLASS_NZERO | CLASS_PZERO | CLASS_PINF                                             # 0x262

        mem = _Heap({0x100 + 4 * i: w for i, w in enumerate(lane_inputs)})
        run_cs([
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x100, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            f"v_mov_b32_e32 v2, {mask1:#x}",
            f"v_mov_b32_e32 v7, {mask2:#x}",
            "v_and_b32_e32 v8, 1, v0",
            "v_cmp_ne_u32_e32 vcc_lo, 0, v8",    # VCC = 1 on odd lanes; must survive both cmpx
            "v_cmpx_ne_u32_e32 0, v0",           # disable lane 0
            "v_cmpx_ne_u32_e32 15, v0",          # disable lane 15
            "v_cmpx_class_f32_e32 v1, v2",
            "v_cmpx_class_f32_e64 v1, v7",
            "v_mov_b32_e32 v3, 0x11",
            "v_mov_b32_e32 v4, 0x22",
            "v_cndmask_b32_e32 v9, v3, v4, vcc_lo",
            "buffer_store_dword v9, v5, s[0:3], 0 offen",
        ], group_size=16, subgroup_size=32, memory=mem)

        got = [mem.get(4 * i) for i in range(16)]
        want = []
        for lane, x_bits in enumerate(lane_inputs):
            active = (lane not in (0, 15)) and ref_cmp_class_f32(x_bits, mask1) and ref_cmp_class_f32(x_bits, mask2)
            want.append((0x22 if (lane & 1) else 0x11) if active else HEAP_FILL)
        self.assertEqual(got, want)
        self.assertEqual(got, [
            HEAP_FILL, HEAP_FILL, 0x11, HEAP_FILL,
            HEAP_FILL, HEAP_FILL, 0x11, 0x22,
            HEAP_FILL, HEAP_FILL, 0x11, 0x22,
            HEAP_FILL, 0x22, 0x11, HEAP_FILL,
        ])

    def test_wave64_multilane_cmpx_class_updates_all_64_lanes(self):
        # 64-lane Wave64 compute run verifying both low half (lanes 0..31) and high half (lanes 32..63).
        lane_inputs = [CLASS_SAMPLES[i % len(CLASS_SAMPLES)][1] for i in range(64)]
        mask = CLASS_SNAN | CLASS_NSUBNORM | CLASS_PINF  # 0x211
        mem = _Heap({0x200 + 4 * i: w for i, w in enumerate(lane_inputs)})
        run_cs([
            "s_mov_b64 exec, -1",                # WAVE64_RE marker -> wave64 mode
            "v_lshlrev_b32_e32 v5, 2, v0",
            "v_add_nc_u32_e32 v6, 0x200, v5",
            "buffer_load_dword v1, v6, s[0:3], 0 offen",
            f"v_mov_b32_e32 v2, {mask:#x}",
            "v_cmpx_class_f32_e32 v1, v2",
            "v_add_nc_u32_e32 v3, 0x100, v0",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
        ], group_size=64, subgroup_size=64, memory=mem)

        got = [mem.get(4 * i) for i in range(64)]
        want = [(0x100 + i) if ref_cmp_class_f32(x_bits, mask) else HEAP_FILL for i, x_bits in enumerate(lane_inputs)]
        self.assertEqual(got, want)


# -----------------------------------------------------------------------------
# 3. v_rndne_f32
# -----------------------------------------------------------------------------

class RndneF32Tests(unittest.TestCase):
    def test_edge_values_e32_e64_and_bare(self):
        for suffix in ("_e32", "_e64", ""):
            for start in range(0, len(RNDNE_HAND_WORKED), 4):
                chunk = RNDNE_HAND_WORKED[start:start + 4]
                lines, expected = [], []
                for slot, (_, x_bits, want) in enumerate(chunk):
                    dst = f"v{10 + slot}"
                    lines += [f"v_mov_b32 v1, {x_bits:#x}", f"v_rndne_f32{suffix} {dst}, v1"]
                    expected.append(want)
                while len(expected) < 4:
                    lines.append(f"v_mov_b32 v{10 + len(expected)}, 0")
                    expected.append(0)
                with self.subTest(suffix=suffix or "bare", batch=start // 4):
                    got = run_ps_mrt0(lines, ("v10", "v11", "v12", "v13"))
                    self.assertEqual(got, tuple(expected),
                                     f"got {[hex(w) for w in got]}, want {[hex(w) for w in expected]}")

    def test_nan_inputs_preserve_nan(self):
        nan_words = (0x7FC00000, 0xFFC00000, 0x7FC12345, 0x7F800001, 0xFF800001)
        for suffix in ("_e32", "_e64"):
            for x_bits in nan_words:
                with self.subTest(suffix=suffix, x=f"{x_bits:#010x}"):
                    got = run_ps_mrt0([f"v_mov_b32 v1, {x_bits:#x}", f"v_rndne_f32{suffix} v0, v1"])[0]
                    self.assertTrue(is_f32_nan_bits(got), f"expected NaN, got {got:#010x}")

    def test_modifiers_inline_constants_sgpr_aliasing_and_exec(self):
        # Source modifiers (-v1, |v1|, -|v1|) and output modifiers (clamp, mul:2, mul:4, div:2).
        got_mods = run_ps_mrt0([
            "v_mov_b32 v1, 0x40200000",          # +2.5
            "v_mov_b32 v2, 0xc0200000",          # -2.5
            "v_mov_b32 v3, 0x3fc00000",          # +1.5 -> rndne = 2.0
            "v_rndne_f32_e64 v10, -v1",          # rndne(-2.5) = -2.0
            "v_rndne_f32_e64 v11, |v2|",         # rndne(2.5) = 2.0
            "v_rndne_f32_e64 v12, -|v1| clamp",  # max(min(-2.0, 1.0), 0.0) = 0.0
            "v_rndne_f32_e64 v13, v3 mul:2",     # 2.0 * 2 = 4.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got_mods, (f32_bits(-2.0), f32_bits(2.0), f32_bits(0.0), f32_bits(4.0)))

        got_mods2 = run_ps_mrt0([
            "v_mov_b32 v1, 0x3fc00000",          # +1.5 -> rndne = 2.0
            "v_mov_b32 v2, 0x3ecccccd",          # ~0.4 -> rndne = +0.0
            "v_rndne_f32_e64 v10, v1 mul:4",     # 2.0 * 4 = 8.0
            "v_rndne_f32_e64 v11, v1 div:2",     # 2.0 * 0.5 = 1.0
            "v_rndne_f32_e64 v12, v2 mul:2 clamp",  # 0.0
            "v_rndne_f32_e64 v13, v1 clamp",     # clamp(2.0, 0, 1) = 1.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got_mods2, (f32_bits(8.0), f32_bits(1.0), f32_bits(0.0), f32_bits(1.0)))

        # Inline float constants, SGPR source, dst aliasing (v0, v0), and inactive EXEC lane preservation.
        got_misc = run_ps_mrt0([
            "v_rndne_f32_e32 v10, -0.5",         # inline -0.5 -> -0.0
            "s_mov_b32 s4, 0xbf000000",          # -0.5 in SGPR
            "v_rndne_f32_e32 v11, s4",
            "v_mov_b32 v12, 0xc0600000",         # -3.5
            "v_rndne_f32_e64 v12, v12",          # aliases source -> -4.0
            "v_mov_b32 v13, 0xdeadbeef",
            "s_mov_b32 exec_lo, 0",
            "v_rndne_f32_e32 v13, 1.0",          # inactive lane: must keep 0xdeadbeef
            "s_mov_b32 exec_lo, -1",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got_misc, (0x80000000, 0x80000000, f32_bits(-4.0), 0xDEADBEEF))


# -----------------------------------------------------------------------------
# 4. v_rcp_iflag_f32
# -----------------------------------------------------------------------------

class RcpIflagF32OracleAndTranslationTests(unittest.TestCase):
    """Checks the translation of v_rcp_iflag_f32 and executes it with source and output modifiers.

    See module docstring, finding 1: exact (power-of-two) inputs only, and no zero operand.
    """

    def test_executes_with_output_modifiers(self):
        got = run_ps_mrt0([
            "v_mov_b32 v1, 0x40000000",                # +2.0
            "v_rcp_iflag_f32_e64 v10, v1 mul:2",       # 0.5 * 2 = 1.0
            "v_rcp_iflag_f32_e64 v11, v1 mul:4",       # 0.5 * 4 = 2.0
            "v_rcp_iflag_f32_e64 v12, v1 div:2",       # 0.5 / 2 = 0.25
            "v_rcp_iflag_f32_e64 v13, -v1 clamp",      # clamp(-0.5, 0, 1) = +0.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (f32_bits(1.0), f32_bits(2.0), f32_bits(0.25), f32_bits(0.0)))

        got2 = run_ps_mrt0([
            "v_mov_b32 v1, 0x3e800000",                # +0.25
            "v_rcp_iflag_f32_e64 v10, v1 clamp",       # clamp(4.0, 0, 1) = 1.0
            "v_rcp_iflag_f32_e64 v11, -|v1|",          # -4.0
            "v_rcp_iflag_f32_e64 v12, v1 div:2 clamp", # clamp(2.0, 0, 1) = 1.0
            "v_rcp_iflag_f32_e32 v13, v1",             # 4.0, no modifiers
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got2, (f32_bits(1.0), f32_bits(-4.0), f32_bits(1.0), f32_bits(4.0)))
        for x_bits, want in ((0x40000000, 0x3F000000), (0x3E800000, 0x40800000), (0xC0000000, 0xBF000000)):
            with self.subTest(x=f"{x_bits:#010x}"):
                self.assertEqual(ref_rcp_iflag_f32(x_bits), want)

    def test_dropping_output_modifiers_is_detected(self):
        # Negative control: if the translator stopped applying the modifier, mul:2 would read back 0.5, not 1.0.
        good = translate_ps(["v_mov_b32 v1, 0x40000000", "v_rcp_iflag_f32_e64 v0, v1 mul:2",
                             "exp mrt0 v0, v0, v0, v0 done vm"])
        self.assertEqual(si.run_fragment(good)["%out_mrt0"][0], f32_bits(1.0))
        bad, n = re.subn(r"^(\s*%\w+) = OpFMul %float (%\w+) %\w+$", r"\1 = OpFMul %float \2 %f1", good,
                         count=1, flags=re.M)
        self.assertEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], f32_bits(1.0))

    def test_translates_to_op_fdiv_with_no_contraction(self):
        for line in (
            "v_rcp_iflag_f32_e32 v0, v1",
            "v_rcp_iflag_f32_e64 v0, v1",
            "v_rcp_iflag_f32 v0, v1",
            "v_rcp_iflag_f32_e64 v0, -|s4|",
            "v_rcp_iflag_f32_e32 v0, 2.0",
        ):
            with self.subTest(line=line):
                spv = translate_ps([
                    "v_mov_b32 v1, 0x40000000",
                    "s_mov_b32 s4, 0xbf800000",
                    line,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                ])
                m = re.search(r"^\s*(%\w+) = OpFDiv %float %f1 %\w+$", spv, re.M)
                self.assertIsNotNone(m, f"expected OpFDiv %float %f1 in translated SPIR-V for {line!r}")
                self.assertIn(f"OpDecorate {m.group(1)} NoContraction", spv)


# -----------------------------------------------------------------------------
# 5. Negative Controls (SPIR-V sabotage detection)
# -----------------------------------------------------------------------------

class NegativeControlsTests(unittest.TestCase):
    def test_sabotaged_spirv_is_detected(self):
        # 1. v_cmp_class_f32: inverting quiet NaN bit check (UGreaterThanEqual -> ULessThan) swaps sNaN and qNaN.
        good_cls = translate_ps([
            "v_mov_b32 v1, 0x7f800001",          # +sNaN
            "v_mov_b32 v2, 0x001",               # CLASS_SNAN
            "v_cmp_class_f32_e32 vcc_lo, v1, v2",
            "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        self.assertEqual(si.run_fragment(good_cls)["%out_mrt0"][0], 1)
        bad_cls, n = re.subn(r"OpUGreaterThanEqual %bool (%\w+) %c400000",
                             r"OpULessThan %bool \1 %c400000", good_cls)
        self.assertEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad_cls)["%out_mrt0"][0], 1)

        # 2. v_cmpx_class_f32: replacing active intersection AND with OR reactivates a pre-disabled lane.
        good_cmpx = translate_ps([
            "v_mov_b32 v0, 0xAAAAAAAA",
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x100",
            "s_mov_b32 exec_lo, 0",
            "v_cmpx_class_f32_e32 v1, v2",
            "v_mov_b32 v0, 0xBBBBBBBB",
            "s_mov_b32 exec_lo, -1",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        self.assertEqual(si.run_fragment(good_cmpx)["%out_mrt0"][0], 0xAAAAAAAA)
        # Sabotage the final `acc = acc & active` right before `OpStore %active`.
        bad_cmpx, n = re.subn(
            r"(%\w+) = OpLogicalAnd %bool (%\w+) (%\w+)\s*\n\s*OpStore %active \1",
            r"\1 = OpLogicalOr %bool \2 \3\n         OpStore %active \1",
            good_cmpx,
        )
        self.assertEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad_cmpx)["%out_mrt0"][0], 0xAAAAAAAA)

        # 3. v_rndne_f32: replacing RoundEven with Trunc breaks tie-to-even (2.5 -> 2.0) and -0.5 (-0.0).
        good_rnd = translate_ps([
            "v_mov_b32 v1, 0x3fc00000",          # +1.5 -> 2.0 (Trunc gives 1.0)
            "v_rndne_f32_e32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ])
        self.assertEqual(si.run_fragment(good_rnd)["%out_mrt0"][0], f32_bits(2.0))
        bad_rnd, n = re.subn(r"OpExtInst %float %glsl RoundEven", "OpExtInst %float %glsl Trunc", good_rnd)
        self.assertEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad_rnd)["%out_mrt0"][0], f32_bits(2.0))


if __name__ == "__main__":
    unittest.main()
