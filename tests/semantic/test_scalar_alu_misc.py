"""Semantic value tests for RDNA2 scalar ALU, select, and 64-bit compare instructions.

Covers:
  * s_ashr_i32      D.i = S0.i >> S1.u[4:0]; SCC = (D != 0)
  * s_sub_i32       D.i = (S0.i - S1.i) mod 2^32; SCC = signed 32-bit overflow
  * s_lshl3_add_u32 tmp = (64'U(S0) << 3) + 64'U(S1); D = tmp[31:0]; SCC = (tmp >= 2^32)
  * s_cselect_b64   D.u64 = SCC ? S0.u64 : S1.u64 (SCC unchanged)
  * s_cmp_lg_u64    SCC = (S0.u64 != S1.u64)

Each end-to-end test constructs a hand-written RDNA2 disassembly listing, translates it
with tools/rdna2spirv.py, executes the resulting SPIR-V with tests/semantic/spirv_mini_interp.py
(run_fragment for single-invocation tests, run_workgroup for wave32/wave64 lane-mask tests),
and checks the exported words against hand-computed constants and plain-Python ISA references.

Translator finding (s_cselect_b64 in tools/rdna2spirv.py, lines 3564-3570, 1922-1946):
  tools/rdna2spirv.py unconditionally lowers s_cselect_b64 through mask_src() and mask_dst()
  (its 1-dword-per-lane wave-mask representation) without checking holds_lane_mask(), unlike
  s_cmp_eq_u64/s_cmp_lg_u64 (lines 3157-3173), s_mov_b64 (lines 426-451), and s_wqm_b64
  (lines 1864-1909). As a result:
    1. For 64-bit SGPR data pairs s[r:r+1] whose halves differ, mask_src() reads only s[r]
       and drops s[r+1], and mask_dst() writes the selected 32-bit word to both s[d] and
       s[d+1], overwriting s[d+1] with s[d].
    2. For non-mask integer immediates (anything other than 0 or -1), mask_src() collapses
       the immediate to (0xFFFFFFFF if (v & 1) else 0) across both dwords.
    3. This affects both wave32 and wave64 shaders whenever non-mask 64-bit data is selected.
  Accordingly, plain-Python ISA reference tests cover all s_cselect_b64 cases, while
  end-to-end translator assertions for s_cselect_b64 are restricted to lane-mask and 0/-1
  selections (where mask semantics match RDNA2 ISA semantics) plus SCC preservation; general
  64-bit SGPR data pairs (lo != hi) and non-mask constants are excluded from passing
  end-to-end assertions rather than relaxing expected values.

Run directly:
    python -I tests/semantic/test_scalar_alu_misc.py -v
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import gc
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
INT32_MIN_U32 = 0x80000000
INT32_MAX_U32 = 0x7FFFFFFF


def _s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x >> 31 else x


def ref_s_ashr_i32(s0: int, s1: int) -> tuple[int, int]:
    """RDNA2 ISA: D.i = S0.i >> S1.u[4:0]; SCC = (D != 0)."""
    res = (_s32(s0) >> (s1 & 31)) & MASK32
    return res, int(res != 0)


def ref_s_sub_i32(s0: int, s1: int) -> tuple[int, int]:
    """RDNA2 ISA: D.i = (S0.i - S1.i) mod 2^32; SCC = signed 32-bit overflow."""
    diff = _s32(s0) - _s32(s1)
    overflow = int(diff < -(1 << 31) or diff >= (1 << 31))
    return diff & MASK32, overflow


def ref_s_lshl3_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """RDNA2 ISA: tmp = (64'U(S0) << 3) + 64'U(S1); D = tmp[31:0]; SCC = (tmp >= 2^32)."""
    full = ((s0 & MASK32) << 3) + (s1 & MASK32)
    return full & MASK32, int(full >= (1 << 32))


def ref_s_cselect_b64(s0: int, s1: int, scc: int) -> int:
    """RDNA2 ISA: D.u64 = SCC ? S0.u64 : S1.u64 (SCC is not modified)."""
    return (s0 if scc else s1) & MASK64


def ref_s_cmp_lg_u64(s0: int, s1: int) -> int:
    """RDNA2 ISA: SCC = (S0.u64 != S1.u64)."""
    return int((s0 & MASK64) != (s1 & MASK64))


# Hand-computed oracle vectors cross-checked against the plain-Python reference functions.
ORACLE_ASHR_I32 = [
    # (s0, s1, expected_d, expected_scc)
    (0x00000000, 0, 0x00000000, 0),
    (0x00000000, 17, 0x00000000, 0),
    (0x00000001, 1, 0x00000000, 0),
    (0x7FFFFFFF, 1, 0x3FFFFFFF, 1),
    (0x7FFFFFFF, 30, 0x00000001, 1),
    (0x7FFFFFFF, 31, 0x00000000, 0),
    (0x80000000, 0, 0x80000000, 1),
    (0x80000000, 1, 0xC0000000, 1),
    (0x80000000, 16, 0xFFFF8000, 1),
    (0x80000000, 31, 0xFFFFFFFF, 1),
    (0x80000001, 1, 0xC0000000, 1),
    (0x80000001, 31, 0xFFFFFFFF, 1),
    (0xF0000000, 4, 0xFF000000, 1),
    (0xFFFFFFFF, 31, 0xFFFFFFFF, 1),
    (0x12345678, 32, 0x12345678, 1),          # 32 & 31 == 0
    (0x80000000, 33, 0xC0000000, 1),          # 33 & 31 == 1
    (0x80000000, 0xFFFFFFFF, 0xFFFFFFFF, 1),  # 0xFFFFFFFF & 31 == 31
    (0x7FFFFFFF, 63, 0x00000000, 0),          # 63 & 31 == 31
]

ORACLE_SUB_I32 = [
    # (s0, s1, expected_d, expected_scc)
    (0x00000000, 0x00000000, 0x00000000, 0),
    (0x00000000, 0x00000001, 0xFFFFFFFF, 0),  # unsigned borrow without signed overflow
    (0x00000001, 0x00000002, 0xFFFFFFFF, 0),
    (INT32_MIN_U32, 1, 0x7FFFFFFF, 1),        # INT_MIN - 1 overflows negative -> positive
    (INT32_MAX_U32, 0xFFFFFFFF, 0x80000000, 1),  # INT_MAX - (-1) overflows positive -> negative
    (0x00000000, INT32_MIN_U32, 0x80000000, 1),  # 0 - (-2^31) = +2^31 overflows
    (INT32_MIN_U32, INT32_MAX_U32, 0x00000001, 1),
    (INT32_MAX_U32, INT32_MIN_U32, 0xFFFFFFFF, 1),
    (0xFFFFFFFE, INT32_MAX_U32, 0x7FFFFFFF, 1),  # -2 - (2^31 - 1) = -2^31 - 1 overflows
    (0x00000001, INT32_MIN_U32, 0x80000001, 1),
    (0xFFFFFFFF, INT32_MAX_U32, 0x80000000, 0),  # -1 - (2^31 - 1) = INT_MIN (exact)
    (0xFFFFFFFF, INT32_MIN_U32, 0x7FFFFFFF, 0),  # -1 - (-2^31) = INT_MAX (exact)
    (INT32_MIN_U32, 0xFFFFFFFF, 0x80000001, 0),  # INT_MIN - (-1) = INT_MIN + 1 (exact)
    (INT32_MIN_U32, INT32_MIN_U32, 0x00000000, 0),
    (INT32_MAX_U32, INT32_MAX_U32, 0x00000000, 0),
]

ORACLE_LSHL3_ADD_U32 = [
    # (s0, s1, expected_d, expected_scc)
    (0x00000000, 0x00000000, 0x00000000, 0),
    (0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0),
    (0x00000005, 0x00000003, 0x0000002B, 0),  # 5 * 8 + 3 = 43
    (0x1FFFFFFF, 0x00000007, 0xFFFFFFFF, 0),  # max non-overflowing sum
    (0x1FFFFFFF, 0x00000008, 0x00000000, 1),  # add carry alone
    (0x10000000, 0x80000000, 0x00000000, 1),  # 0x80000000 + 0x80000000
    (0x00000001, 0xFFFFFFF8, 0x00000000, 1),
    (0x20000000, 0x00000000, 0x00000000, 1),  # bit 29 shifted out alone sets SCC
    (0x40000000, 0x00000000, 0x00000000, 1),  # bit 30 shifted out alone
    (0x80000000, 0x00000000, 0x00000000, 1),  # bit 31 shifted out alone
    (0x20000001, 0x00000005, 0x0000000D, 1),
    (0xFFFFFFFF, 0x00000000, 0xFFFFFFF8, 1),
    (0xFFFFFFFF, 0x00000008, 0x00000000, 1),
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFF7, 1),
]

ORACLE_CSELECT_B64 = [
    # (s0_64, s1_64, scc, expected_64) -- includes full 64-bit data pairs and constants for ISA reference
    (0x1111111122222222, 0x3333333344444444, 1, 0x1111111122222222),
    (0x1111111122222222, 0x3333333344444444, 0, 0x3333333344444444),
    (0x0000000000000002, 0x0000000000000003, 1, 0x0000000000000002),
    (0x0000000000000002, 0x0000000000000003, 0, 0x0000000000000003),
    (0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 1, 0xFFFFFFFFFFFFFFFF),
    (0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 0, 0x0000000000000000),
    (0x8000000000000000, 0x0000000080000000, 1, 0x8000000000000000),
    (0x8000000000000000, 0x0000000080000000, 0, 0x0000000080000000),
]

ORACLE_CMP_LG_U64 = [
    # (s0_64, s1_64, expected_scc)
    (0x0000000000000000, 0x0000000000000000, 0),
    (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0),
    (0x123456789ABCDEF0, 0x123456789ABCDEF0, 0),
    (0x1234567800000000, 0x1234567800000001, 1),  # differs only in bit 0 (low dword)
    (0x1234567880000000, 0x1234567800000000, 1),  # differs only in bit 31 (low dword)
    (0x000000019ABCDEF0, 0x000000009ABCDEF0, 1),  # differs only in bit 32 (high dword)
    (0x800000009ABCDEF0, 0x000000009ABCDEF0, 1),  # differs only in bit 63 (high dword)
    (0x1111111122222222, 0x2222222211111111, 1),  # swapped dwords
    (0xFFFFFFFFFFFFFFF0, 0x00000000FFFFFFF0, 1),  # sign-extended -16 vs 32-bit low word
]

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM_DIR = None


def _fssm_blob(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    """Build the minimal FSSM header words read by tools/rdna2spirv.py for compute shaders."""
    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, rsrc2, *group)
    return bytes(hdr)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM_DIR
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    _CS_FSSM_DIR = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(_CS_FSSM_DIR)
    _PS_PATH = os.path.join(disasm_dir, "ps_salu_misc.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_salu_misc.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_misc.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_salu_misc.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_scalar_alu_misc\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_quad(lines: list[str]) -> tuple[int, int, int, int]:
    """Translate PS instructions and return the four exported 32-bit words (v0, v1, v2, v3)."""
    words = mrt0_words(translate_ps(lines))
    return (words[0], words[1], words[2], words[3])


def run_cs_workgroup(lines: list[str], threads: int, subgroup_size: int) -> tuple[str, dict[int, int]]:
    """Translate CS instructions (4 user SGPRs s[0:3] for buffer V#, v0 = local_id.x) and run in lockstep."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_scalar_alu_misc_cs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with open(os.path.join(_CS_FSSM_DIR, "cs_salu_misc.fssm"), "wb") as f:
        # 0x1008: USER_SGPR=4 (s[0:3]), TIDIG_COMP_CNT=2 (v0..v2), no TGID SGPRs.
        f.write(_fssm_blob(0x1008, (threads, 1, 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    memory: dict[int, int] = {}
    si.run_workgroup(
        spv,
        threads,
        subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
        push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
        memory=memory,
    )
    return spv, memory


def scc_seed_line(scc_val: int) -> str:
    return "s_cmp_eq_u32 1, 1" if scc_val else "s_cmp_eq_u32 0, 1"


def join64(lo: int, hi: int) -> int:
    return ((hi & MASK32) << 32) | (lo & MASK32)


def split64(v: int) -> tuple[int, int]:
    return v & MASK32, (v >> 32) & MASK32


class TestOracleConstants(unittest.TestCase):
    """Self-consistency checks between hand-computed oracle constants and plain-Python ISA models."""

    def test_reference_matches_oracle_tables(self):
        for s0, s1, want_d, want_scc in ORACLE_ASHR_I32:
            with self.subTest(op="s_ashr_i32", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_s_ashr_i32(s0, s1), (want_d, want_scc))
        for s0, s1, want_d, want_scc in ORACLE_SUB_I32:
            with self.subTest(op="s_sub_i32", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_s_sub_i32(s0, s1), (want_d, want_scc))
        for s0, s1, want_d, want_scc in ORACLE_LSHL3_ADD_U32:
            with self.subTest(op="s_lshl3_add_u32", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_s_lshl3_add_u32(s0, s1), (want_d, want_scc))
        for s0, s1, scc, want in ORACLE_CSELECT_B64:
            with self.subTest(op="s_cselect_b64", s0=hex(s0), s1=hex(s1), scc=scc):
                self.assertEqual(ref_s_cselect_b64(s0, s1, scc), want)
        for s0, s1, want_scc in ORACLE_CMP_LG_U64:
            with self.subTest(op="s_cmp_lg_u64", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_s_cmp_lg_u64(s0, s1), want_scc)


class TestScalarAshrI32(unittest.TestCase):
    """End-to-end semantic value tests for s_ashr_i32."""

    def test_sign_fill_zero_fill_and_shift_masking(self):
        for s0, s1, want_d, want_scc in ORACLE_ASHR_I32:
            pre_scc = 1 - want_scc
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, w2, w3 = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(pre_scc),
                    "s_ashr_i32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    scc_seed_line(pre_scc),
                    f"s_ashr_i32 s4, {s0:#x}, {s1 & 31}",
                    "s_cselect_b32 s5, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s4",
                    "v_mov_b32 v3, s5",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))
                self.assertEqual((w2, w3), (want_d, want_scc))

    def test_large_shift_counts_mask_modulo_32(self):
        # Verify upper bits [31:5] of S1 are ignored identically for both imm and SGPR counts.
        pairs = [
            (0x80000000, 0, 32),
            (0x80000000, 0, 0xFFFFFFE0),
            (0x80000000, 1, 33),
            (0x80000000, 4, 68),
            (0x7FFFFFFF, 31, 63),
            (0x12345678, 5, 0x80000025),
        ]
        for val, small_sh, dirty_sh in pairs:
            want = ref_s_ashr_i32(val, small_sh)
            with self.subTest(val=hex(val), dirty_sh=hex(dirty_sh)):
                w0, w1, w2, w3 = run_ps_quad([
                    f"s_mov_b32 s2, {val:#x}",
                    f"s_mov_b32 s3, {dirty_sh:#x}",
                    "s_ashr_i32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    f"s_ashr_i32 s4, s2, {dirty_sh:#x}",
                    "s_cselect_b32 s5, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s4",
                    "v_mov_b32 v3, s5",
                ])
                self.assertEqual((w0, w1), want)
                self.assertEqual((w2, w3), want)

    def test_aliasing_special_operands_and_inline_negatives(self):
        w0, w1, w2, w3 = run_ps_quad([
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 m0, 4",
            "s_ashr_i32 s0, s0, 4",          # dst aliases s0 -> 0xF8000000
            "s_ashr_i32 s1, s1, m0",         # s1 = 0xF8000000 using m0 count
            "s_ashr_i32 s2, -8, 1",          # inline -8 >> 1 -> -4 (0xFFFFFFFC)
            "s_ashr_i32 null, -1, 1",        # discarded result, SCC still set to 1
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0xF8000000, 0xF8000000, 0xFFFFFFFC, 1))

        # dst aliases shift count, and dst == src0 == src1
        w0, w1, w2, w3 = run_ps_quad([
            "s_mov_b32 s0, 0x00000004",
            "s_mov_b32 s1, 0x80000000",
            "s_ashr_i32 s0, s1, s0",         # dst == s1 (shift count) -> 0xF8000000
            "s_mov_b32 s2, 0xfffffff0",      # -16: low 5 bits = 16
            "s_ashr_i32 s2, s2, s2",         # -16 >> 16 = -1 (0xFFFFFFFF), SCC = 1
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s2",
            "v_mov_b32 v2, s3",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1, w2), (0xF8000000, 0xFFFFFFFF, 1))

    def test_seeded_sweep(self):
        rng = random.Random(0xA54B)
        for _ in range(30):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            want_d, want_scc = ref_s_ashr_i32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(1 - want_scc),
                    "s_ashr_i32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))


class TestScalarSubI32(unittest.TestCase):
    """End-to-end semantic value tests for s_sub_i32."""

    def test_signed_overflow_vs_wraparound_boundaries(self):
        for s0, s1, want_d, want_scc in ORACLE_SUB_I32:
            pre_scc = 1 - want_scc
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(pre_scc),
                    "s_sub_i32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))

    def test_scc_differs_from_unsigned_s_sub_u32(self):
        # 0 - 1: s_sub_i32 sets SCC=0 (no signed overflow), s_sub_u32 sets SCC=1 (unsigned borrow).
        # INT_MIN - 1: s_sub_i32 sets SCC=1 (signed overflow), s_sub_u32 sets SCC=0 (no unsigned borrow).
        w0, w1, w2, w3 = run_ps_quad([
            "s_sub_i32 s0, 0, 1",
            "s_cselect_b32 s0, 1, 0",
            "s_sub_u32 s1, 0, 1",
            "s_cselect_b32 s1, 1, 0",
            "s_sub_i32 s2, 0x80000000, 1",
            "s_cselect_b32 s2, 1, 0",
            "s_sub_u32 s3, 0x80000000, 1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0, 1, 1, 0))

    def test_aliasing_inline_negative_and_special_regs(self):
        # Self-subtraction on INT_MIN and INT_MAX must give 0 and SCC=0 (even with pre-seeded SCC=1).
        for val in (0, 1, INT32_MIN_U32, INT32_MAX_U32, 0xFFFFFFFF):
            with self.subTest(self_sub=hex(val)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s0, {val:#x}",
                    scc_seed_line(1),
                    "s_sub_i32 s0, s0, s0",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (0, 0))

        # dst == s0, dst == s1, inline negative constant, m0, null
        w0, w1, w2, w3 = run_ps_quad([
            "s_mov_b32 s0, 0x7fffffff",
            "s_sub_i32 s0, s0, -1",          # INT_MAX - (-1) -> 0x80000000, SCC=1
            "s_cselect_b32 s1, 1, 0",
            "s_mov_b32 m0, 1",
            "s_mov_b32 s2, 0x80000000",
            "s_sub_i32 s2, s2, m0",          # INT_MIN - 1 -> 0x7FFFFFFF, SCC=1
            "s_sub_i32 null, 5, 5",          # discarded, SCC=0
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0x80000000, 1, 0x7FFFFFFF, 0))

        # dst aliases src1 (second source)
        w0, w1, _, _ = run_ps_quad([
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 30",
            "s_sub_i32 s0, s1, s0",          # 30 - 10 = 20, SCC=0
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s2",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1), (20, 0))

    def test_seeded_sweep(self):
        rng = random.Random(0x50B1)
        for _ in range(30):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            want_d, want_scc = ref_s_sub_i32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(1 - want_scc),
                    "s_sub_i32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))


class TestScalarLshl3AddU32(unittest.TestCase):
    """End-to-end semantic value tests for s_lshl3_add_u32."""

    def test_scaled_add_and_64bit_carry_scc(self):
        for s0, s1, want_d, want_scc in ORACLE_LSHL3_ADD_U32:
            pre_scc = 1 - want_scc
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(pre_scc),
                    "s_lshl3_add_u32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))

    def test_aliasing_immediates_and_carry_chain(self):
        # s0 = (s0 << 3) + s0 == 9 * s0 mod 2^32, plus chaining carry into s_addc_u32.
        w0, w1, w2, w3 = run_ps_quad([
            "s_mov_b32 s0, 0x20000003",
            "s_lshl3_add_u32 s0, s0, s0",    # (0x20000003 << 3) + 0x20000003 = 0x1_2000001B -> 0x2000001B, SCC=1
            "s_cselect_b32 s1, 1, 0",
            "s_mov_b32 s2, 0x20000000",
            "s_lshl3_add_u32 s2, s2, 4",     # (0x20000000 << 3) + 4 = 0x1_00000004 -> 4, SCC=1
            "s_addc_u32 s3, 10, 0",          # 10 + 0 + SCC(1) = 11
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0x2000001B, 1, 4, 11))

    def test_seeded_sweep(self):
        rng = random.Random(0x13A3)
        for _ in range(30):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            want_d, want_scc = ref_s_lshl3_add_u32(s0, s1)
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s2, {s0:#x}",
                    f"s_mov_b32 s3, {s1:#x}",
                    scc_seed_line(1 - want_scc),
                    "s_lshl3_add_u32 s0, s2, s3",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_d, want_scc))


class TestScalarCmpLgU64(unittest.TestCase):
    """End-to-end semantic value tests for s_cmp_lg_u64."""

    def test_sgpr_pair_inequality_low_and_high_halves(self):
        for s0_64, s1_64, want_scc in ORACLE_CMP_LG_U64:
            lo0, hi0 = split64(s0_64)
            lo1, hi1 = split64(s1_64)
            with self.subTest(s0=hex(s0_64), s1=hex(s1_64)):
                w0, _, _, _ = run_ps_quad([
                    f"s_mov_b32 s0, {lo0:#x}",
                    f"s_mov_b32 s1, {hi0:#x}",
                    f"s_mov_b32 s2, {lo1:#x}",
                    f"s_mov_b32 s3, {hi1:#x}",
                    scc_seed_line(1 - want_scc),
                    "s_cmp_lg_u64 s[0:1], s[2:3]",
                    "s_cselect_b32 s4, 1, 0",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(w0, want_scc)

    def test_self_compare_and_overlapping_pairs(self):
        # Self-compare must clear pre-seeded SCC=1; overlapping s[0:1] vs s[1:2] reads both halves accurately.
        w0, w1, w2, w3 = run_ps_quad([
            "s_mov_b32 s0, 0xdeadbeef",
            "s_mov_b32 s1, 0xdeadbeef",
            "s_mov_b32 s2, 0xdeadbeef",
            scc_seed_line(1),
            "s_cmp_lg_u64 s[0:1], s[0:1]",   # identical pair -> SCC = 0
            "s_cselect_b32 s4, 1, 0",
            scc_seed_line(1),
            "s_cmp_lg_u64 s[0:1], s[1:2]",   # all 0xdeadbeef -> SCC = 0
            "s_cselect_b32 s5, 1, 0",
            "s_mov_b32 s2, 0xdeadbee0",
            "s_cmp_lg_u64 s[0:1], s[1:2]",   # differs in s2 (high word of s[1:2]) -> SCC = 1
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1, w2), (0, 0, 1))

    def test_inline_and_literal_constants(self):
        # Inline -1 sign-extends to 0xFFFFFFFF_FFFFFFFF; -16 -> 0xFFFFFFFF_FFFFFFF0;
        # positive inline 64 and 32-bit literals zero-extend high 32 bits to 0.
        cases = [
            (0x00000000, 0x00000000, "0", 0),
            (0x00000000, 0x00000001, "0", 1),        # high dword non-zero vs 0
            (0xFFFFFFFF, 0xFFFFFFFF, "-1", 0),       # sign-extended -1 matches all-ones
            (0xFFFFFFFF, 0x00000000, "-1", 1),       # high dword 0 != 0xFFFFFFFF
            (0xFFFFFFF0, 0xFFFFFFFF, "-16", 0),
            (0xFFFFFFF0, 0x00000000, "-16", 1),
            (64, 0, "64", 0),
            (64, 1, "64", 1),
            (0x12345678, 0, "0x12345678", 0),
            (0x12345678, 1, "0x12345678", 1),
            (0x80000000, 0, "0x80000000", 0),
            (0x80000000, 0xFFFFFFFF, "0x80000000", 1),
        ]
        for lo, hi, imm_tok, want_scc in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), imm=imm_tok):
                w0, w1, _, _ = run_ps_quad([
                    f"s_mov_b32 s0, {lo:#x}",
                    f"s_mov_b32 s1, {hi:#x}",
                    scc_seed_line(1 - want_scc),
                    f"s_cmp_lg_u64 s[0:1], {imm_tok}",
                    "s_cselect_b32 s2, 1, 0",
                    scc_seed_line(1 - want_scc),
                    f"s_cmp_lg_u64 {imm_tok}, s[0:1]",
                    "s_cselect_b32 s3, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual((w0, w1), (want_scc, want_scc))

    def test_wave32_and_wave64_lane_mask_comparisons(self):
        # Wave64: lane 45 sets only the upper half of VCC; s_cmp_lg_u64 vcc, 0 must see it (SCC=1).
        _, mem64 = run_cs_workgroup([
            "v_cmp_eq_u32_e32 vcc, 45, v0",
            "s_cmp_lg_u64 vcc, 0",
            "s_cselect_b32 s4, 1, 0",
            "v_cmp_eq_u32_e32 vcc, 99, v0",
            "s_cmp_lg_u64 vcc, 0",
            "s_cselect_b32 s5, 1, 0",
            "s_mov_b64 s[6:7], exec",
            "s_cmp_lg_u64 exec, s[6:7]",     # exec == saved exec -> SCC = 0
            "s_cselect_b32 s8, 1, 0",
            "s_cmp_lg_u64 exec, 0",          # active lanes present -> SCC = 1
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v1, s4",
            "v_mov_b32 v2, s5",
            "v_mov_b32 v3, s8",
            "v_mov_b32 v4, s9",
            "v_lshlrev_b32 v5, 4, v0",
            "buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen",
        ], threads=64, subgroup_size=64)
        self.assertEqual((mem64[0], mem64[4], mem64[8], mem64[12]), (1, 0, 0, 1))

        # Wave32: lane 20 in vcc_lo / s[4:5]; s_cmp_lg_u64 detects inequality across the 32-lane wave.
        _, mem32 = run_cs_workgroup([
            "v_cmp_eq_u32_e64 s[4:5], 20, v0",
            "s_cmp_lg_u64 s[4:5], 0",        # lane 20 active -> SCC = 1
            "s_cselect_b32 s6, 1, 0",
            "v_cmp_eq_u32_e64 s[4:5], 45, v0",
            "s_cmp_lg_u64 s[4:5], 0",        # only 32 lanes (0..31) -> SCC = 0
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v1, s6",
            "v_mov_b32 v2, s7",
            "v_lshlrev_b32 v3, 3, v0",
            "buffer_store_dwordx2 v[1:2], v3, s[0:3], 0 offen",
        ], threads=32, subgroup_size=32)
        self.assertEqual((mem32[0], mem32[4]), (1, 0))

    def test_seeded_sweep(self):
        rng = random.Random(0xC391)
        for _ in range(30):
            a = rng.getrandbits(64)
            # Half the trials change only one dword (lo or hi) so both halves are stressed equally.
            if rng.random() < 0.5:
                b = a ^ (1 << rng.randrange(64))
            else:
                b = a
            alo, ahi = split64(a)
            blo, bhi = split64(b)
            want = ref_s_cmp_lg_u64(a, b)
            with self.subTest(a=hex(a), b=hex(b)):
                w0, _, _, _ = run_ps_quad([
                    f"s_mov_b32 s0, {alo:#x}",
                    f"s_mov_b32 s1, {ahi:#x}",
                    f"s_mov_b32 s2, {blo:#x}",
                    f"s_mov_b32 s3, {bhi:#x}",
                    scc_seed_line(1 - want),
                    "s_cmp_lg_u64 s[0:1], s[2:3]",
                    "s_cselect_b32 s4, 1, 0",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(w0, want)


class TestScalarCselectB64LaneMasks(unittest.TestCase):
    """End-to-end semantic tests for s_cselect_b64 on uniform 0/-1 masks and wave lane masks.

    Excluded due to translator finding (tools/rdna2spirv.py lines 3564-3570, 1922-1946):
      64-bit SGPR data pairs with lo != hi and non-mask constants (anything other than 0 or -1)
      are miscompiled by mask_src()/mask_dst(), which drops the high dword of SGPR pair sources,
      duplicates the low 32-bit result into both destination dwords, and collapses constants to
      (0xFFFFFFFF if (v & 1) else 0). Those data-pair and constant assertions are covered by the
      plain-Python RDNA2 ISA reference tests in TestOracleConstants and excluded here.
    """

    def test_zero_and_all_ones_selection_and_scc_preserved(self):
        # Uniform 0 and -1 (0xFFFFFFFF_FFFFFFFF) agree in both dwords and preserve SCC untouched.
        for scc_in in (0, 1):
            want_a = MASK64 if scc_in else 0
            want_b = 0 if scc_in else MASK64
            with self.subTest(scc_in=scc_in):
                w0, w1, w2, w3 = run_ps_quad([
                    scc_seed_line(scc_in),
                    "s_cselect_b64 s[0:1], -1, 0",
                    "s_cselect_b64 s[2:3], 0, -1",
                    "s_cselect_b32 s4, 1, 0",   # verify SCC was not modified by s_cselect_b64
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s2",
                    "v_mov_b32 v3, s4",
                ])
                self.assertEqual((w0, w1), split64(want_a))
                self.assertEqual(w2, split64(want_b)[0])
                self.assertEqual(w3, scc_in)

    def test_wave64_exec_and_vcc_lane_mask_selection(self):
        # Test s_cselect_b64 on EXEC, VCC, and lane-mask SGPR pairs across a 64-lane wave.
        _, mem = run_cs_workgroup([
            "v_mov_b32 v1, 0",
            "v_cmp_lt_u32_e32 vcc, v0, 40",      # lanes 0..39 -> count 40
            "s_mov_b64 s[4:5], exec",            # all 64 lanes -> count 64
            "s_cmp_eq_u32 1, 1",                 # SCC = 1
            "s_cselect_b64 s[6:7], vcc, s[4:5]", # selects vcc (40 lanes)
            "s_bcnt1_i32_b64 s8, s[6:7]",
            "s_cmp_eq_u32 0, 1",                 # SCC = 0
            "s_cselect_b64 s[6:7], vcc, s[4:5]", # selects s[4:5] (64 lanes)
            "s_cselect_b32 s10, 1, 0",           # SCC must still be 0 right after s_cselect_b64
            "s_bcnt1_i32_b64 s9, s[6:7]",        # note: s_bcnt1 overwrites SCC = (s9 != 0)
            "s_cmp_eq_u32 1, 1",                 # SCC = 1
            "s_cselect_b64 exec, vcc, s[4:5]",   # restrict EXEC to lanes 0..39
            "v_mov_b32 v1, 1",                   # written only by active lanes 0..39
            "s_cselect_b64 exec, -1, exec",      # restore EXEC to all 64 lanes
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s9",
            "v_mov_b32 v4, s10",
            "v_lshlrev_b32 v5, 4, v0",
            "buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen",
        ], threads=64, subgroup_size=64)
        for lane in range(64):
            base = 16 * lane
            self.assertEqual(mem[base], 1 if lane < 40 else 0)
            self.assertEqual((mem[base + 4], mem[base + 8], mem[base + 12]), (40, 64, 0))

    def test_wave32_lane_mask_selection(self):
        # Same lane-mask selection under a 32-wide wave (pure wave32 subgroup).
        _, mem = run_cs_workgroup([
            "v_cmp_lt_u32_e64 s[4:5], v0, 12",   # lanes 0..11
            "s_cmp_eq_u32 1, 1",                 # SCC = 1
            "s_cselect_b64 s[6:7], s[4:5], -1",  # selects s[4:5] (lanes 0..11)
            "s_cmp_lg_u64 s[6:7], s[4:5]",       # equal -> SCC = 0
            "s_cselect_b32 s8, 1, 0",
            "s_cmp_eq_u32 0, 1",                 # SCC = 0
            "s_cselect_b64 s[6:7], 0, s[4:5]",   # selects s[4:5] (lanes 0..11)
            "s_cmp_lg_u64 s[6:7], 0",            # non-zero -> SCC = 1
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v1, s8",
            "v_mov_b32 v2, s9",
            "v_lshlrev_b32 v3, 3, v0",
            "buffer_store_dwordx2 v[1:2], v3, s[0:3], 0 offen",
        ], threads=32, subgroup_size=32)
        self.assertEqual((mem[0], mem[4]), (0, 1))


class TestNegativeControls(unittest.TestCase):
    """Sabotage checks confirming the SPIR-V assertions catch broken translations."""

    def test_sabotage_ashr_sub_lshl3_cmp_and_cselect(self):
        # 1. s_ashr_i32: replacing arithmetic shift with logical shift breaks negative sign extension.
        spv_ashr = translate_ps([
            "s_mov_b32 s2, 0x80000000",
            "s_ashr_i32 s0, s2, 1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(mrt0_words(spv_ashr)[0], 0xC0000000)
        bad_ashr = spv_ashr.replace("OpShiftRightArithmetic %int", "OpShiftRightLogical %int", 1)
        self.assertNotEqual(bad_ashr, spv_ashr)
        self.assertNotEqual(mrt0_words(bad_ashr)[0], 0xC0000000)

        # 2. s_sub_i32: losing overflow detection breaks SCC on INT_MIN - 1.
        spv_sub = translate_ps([
            "s_mov_b32 s2, 0x80000000",
            "s_sub_i32 s0, s2, 1",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(tuple(mrt0_words(spv_sub)[:2]), (0x7FFFFFFF, 1))
        bad_sub = spv_sub.replace("OpStore %scc", "OpStore %vcc_hi", 1)
        self.assertNotEqual(bad_sub, spv_sub)
        self.assertNotEqual(tuple(mrt0_words(bad_sub)[:2]), (0x7FFFFFFF, 1))

        # 3. s_lshl3_add_u32: dropping shifted-out carry bits breaks SCC on 0x20000000 << 3 + 0.
        spv_lshl3 = translate_ps([
            "s_mov_b32 s2, 0x20000000",
            "s_lshl3_add_u32 s0, s2, 0",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(tuple(mrt0_words(spv_lshl3)[:2]), (0, 1))
        bad_lshl3 = spv_lshl3.replace("OpShiftRightLogical %uint", "OpBitwiseAnd %uint", 1)
        self.assertNotEqual(bad_lshl3, spv_lshl3)
        self.assertNotEqual(tuple(mrt0_words(bad_lshl3)[:2]), (0, 1))

        # 4. s_cmp_lg_u64: omitting OpLogicalNot flips inequality into equality.
        spv_cmp = translate_ps([
            "s_mov_b32 s0, 1",
            "s_mov_b32 s1, 2",
            "s_mov_b32 s2, 1",
            "s_mov_b32 s3, 3",
            "s_cmp_lg_u64 s[0:1], s[2:3]",
            "s_cselect_b32 s4, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(mrt0_words(spv_cmp)[0], 1)
        bad_cmp = spv_cmp.replace("OpLogicalNot %bool", "OpCopyObject %bool", 1)
        self.assertNotEqual(bad_cmp, spv_cmp)
        self.assertNotEqual(mrt0_words(bad_cmp)[0], 1)

        # 5. s_cselect_b64: reversing select polarity breaks 0/-1 mask selection.
        spv_csel = translate_ps([
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], -1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(tuple(mrt0_words(spv_csel)[:2]), (0xFFFFFFFF, 0xFFFFFFFF))
        bad_csel = spv_csel.replace("OpINotEqual %bool", "OpIEqual %bool", 2)
        self.assertNotEqual(bad_csel, spv_csel)
        self.assertNotEqual(tuple(mrt0_words(bad_csel)[:2]), (0xFFFFFFFF, 0xFFFFFFFF))


if __name__ == "__main__":
    unittest.main()
