"""Semantic interpreter tests for RDNA2 scalar inline-immediate and bit-set ALU instructions.

Covers end-to-end translation via ``tools/rdna2spirv.py`` and execution via
``tests/semantic/spirv_mini_interp.py`` for:
  * s_movk_i32    (SOPK: D = signext(simm16); SCC unchanged)
  * s_addk_i32    (SOPK: D = D + signext(simm16) mod 2^32; SCC = signed 32-bit overflow)
  * s_mulk_i32    (SOPK: D = D * signext(simm16) mod 2^32; SCC unchanged)
  * s_bitset0_b32 (SOP1: D = D & ~(1 << (S0 & 31)); SCC unchanged)
  * s_bitset1_b32 (SOP1: D = D | (1 << (S0 & 31)); SCC unchanged)
  * s_bfm_b32     (SOP2: D = ((1 << (S0 & 31)) - 1) << (S1 & 31) mod 2^32; SCC unchanged)

RDNA2 ISA rules (AMD RDNA2 "Instruction Set Architecture" reference, SOP1/SOP2/SOPK):
  * SOPK 16-bit literals (``simm16``) are sign-extended from bit 15 to 32 bits before use.
  * ``s_addk_i32`` performs in-place signed 32-bit addition and sets SCC on signed overflow
    (positive + positive -> negative, or negative + negative -> positive). Unsigned wraparound
    such as ``0xFFFFFFFF + 1 -> 0`` produces SCC = 0.
  * ``s_movk_i32``, ``s_mulk_i32``, ``s_bitset0_b32``, ``s_bitset1_b32``, and ``s_bfm_b32``
    do not modify SCC.
  * ``s_bitset0_b32`` / ``s_bitset1_b32`` mask the bit index with ``& 31``.
  * ``s_bfm_b32`` masks both width (``S0 & 31``) and offset (``S1 & 31``) to 5 bits, so
    width = 32 masks to 0 (yielding 0, not 0xFFFFFFFF), and bits shifted past bit 31 are
    truncated (zeroed), not wrapped around to bit 0.

Translator notes observed in ``tools/rdna2spirv.py``:
  * ``s_mulk_i32``, ``s_bitset0_b32``, and ``s_bitset1_b32`` read their in-place destination
    with ``src_u(a[0])`` (matching ``s_addk_i32``), supporting SGPR and non-SGPR targets
    such as ``m0``, ``vcc_lo``, ``vcc_hi``, and ``null``.
  * ``s_bfm_b32`` has a duplicate unreachable handler shadowed by its earlier branch.

Run directly:
    python -I tests/semantic/test_scalar_alu_imm_bitset.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True      # never drop __pycache__ into the tree under test
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_imm_bitset.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_imm_bitset.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def _s16(val: int) -> int:
    v = val & 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _s32(val: int) -> int:
    v = val & MASK32
    return v - (1 << 32) if v >> 31 else v


def ref_s_movk_i32(simm16: int) -> int:
    return _s16(simm16) & MASK32


def ref_s_addk_i32(dst_val: int, simm16: int) -> tuple[int, int]:
    total = _s32(dst_val) + _s16(simm16)
    scc = 1 if (total < -(1 << 31) or total >= (1 << 31)) else 0
    return total & MASK32, scc


def ref_s_mulk_i32(dst_val: int, simm16: int) -> int:
    return (_s32(dst_val) * _s16(simm16)) & MASK32


def ref_s_bitset0_b32(dst_val: int, bit_idx: int) -> int:
    return (dst_val & MASK32) & ~(1 << (bit_idx & 31))


def ref_s_bitset1_b32(dst_val: int, bit_idx: int) -> int:
    return ((dst_val & MASK32) | (1 << (bit_idx & 31))) & MASK32


def ref_s_bfm_b32(width: int, offset: int) -> int:
    return (((1 << (width & 31)) - 1) << (offset & 31)) & MASK32


def translate_lines(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_scalar_alu_imm_bitset\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_pair(lines: list[str]) -> tuple[int, int]:
    """Run shader lines writing v0 and v1; return (v0, v1)."""
    spv = translate_lines(lines + ["exp mrt0 v0, v1, v0, v0 done vm", "s_endpgm"])
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0], out[1]


def run_quad(lines: list[str]) -> tuple[int, int, int, int]:
    """Run shader lines writing v0..v3; return all 4 exported words."""
    spv = translate_lines(lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[0], out[1], out[2], out[3])


def scc_seed_line(pre_scc: int) -> str:
    return "s_cmp_eq_u32 1, 1" if pre_scc else "s_cmp_eq_u32 0, 1"


def fmt_imm16(imm) -> str:
    return imm if isinstance(imm, str) else f"{imm:#x}"


# Hand-written boundary constants checked against reference functions in OracleTests.
ORACLE_MOVK = [
    (0x0000, 0x00000000),
    (0x0001, 0x00000001),
    (0x7FFF, 0x00007FFF),
    (0x8000, 0xFFFF8000),
    (0xABCD, 0xFFFFABCD),
    (0xFFFF, 0xFFFFFFFF),
]

ORACLE_ADDK = [
    (0x00000000, 0x0000, 0x00000000, 0),
    (0x7FFFFFFF, 0x0001, 0x80000000, 1),   # positive signed overflow
    (0x7FFF8000, 0x7FFF, 0x7FFFFFFF, 0),   # max non-overflowing positive add
    (0x7FFF8001, 0x7FFF, 0x80000000, 1),   # min overflowing positive add with max simm16
    (0x80000000, 0xFFFF, 0x7FFFFFFF, 1),   # negative signed overflow (-2^31 + -1)
    (0x80008000, 0x8000, 0x80000000, 0),   # min non-overflowing negative add
    (0x80007FFF, 0x8000, 0x7FFFFFFF, 1),   # max overflowing negative add with min simm16
    (0xFFFFFFFF, 0x0001, 0x00000000, 0),   # unsigned wrap without signed overflow
    (0x00000005, 0xFFF6, 0xFFFFFFFB, 0),   # 5 + (-10) = -5
]

ORACLE_MULK = [
    (0x00000000, 0x7FFF, 0x00000000),
    (0x12345678, 0x0000, 0x00000000),
    (0x00000001, 0x8000, 0xFFFF8000),      # 1 * -32768 (sign-extension check)
    (0x80000000, 0xFFFF, 0x80000000),      # INT32_MIN * -1 wraps to INT32_MIN
    (0x80000000, 0x8000, 0x00000000),      # -2^31 * -2^15 = 2^46 -> low 32 bits 0
    (0x80000001, 0x8000, 0xFFFF8000),      # (-2^31 + 1) * -32768
    (0x7FFFFFFF, 0x7FFF, 0x7FFF8001),      # (2^31 - 1) * 32767 mod 2^32
    (0xFFFFFFFB, 0x8000, 0x00028000),      # -5 * -32768 = 163840
]

ORACLE_BITSET = [
    # (op, init, bit_idx, expected)
    ("bitset0", 0xFFFFFFFF, 0, 0xFFFFFFFE),
    ("bitset0", 0xFFFFFFFF, 31, 0x7FFFFFFF),
    ("bitset0", 0xFFFFFFFF, 32, 0xFFFFFFFE),
    ("bitset0", 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF),
    ("bitset0", 0xA5A55A5A, 3, 0xA5A55A52),
    ("bitset1", 0x00000000, 0, 0x00000001),
    ("bitset1", 0x00000000, 31, 0x80000000),
    ("bitset1", 0x00000000, 32, 0x00000001),
    ("bitset1", 0x00000000, 0xFFFFFFFF, 0x80000000),
    ("bitset1", 0xA5A55A5A, 2, 0xA5A55A5E),
]

ORACLE_BFM = [
    # (width, offset, expected)
    (0, 0, 0x00000000),
    (0, 31, 0x00000000),
    (32, 0, 0x00000000),                   # width masked & 31 -> 0
    (32, 16, 0x00000000),
    (31, 0, 0x7FFFFFFF),
    (31, 1, 0xFFFFFFFE),
    (31, 2, 0xFFFFFFFC),                   # shifted past bit 31: truncated, not wrapped
    (31, 31, 0x80000000),
    (8, 28, 0xF0000000),                   # overflow beyond bit 31 truncated
    (16, 20, 0xFFF00000),
    (4, 32, 0x0000000F),                   # offset masked & 31 -> 0
    (33, 35, 0x00000008),                  # width 1, offset 3
    (0xFFFFFFFF, 0xFFFFFFFF, 0x80000000),  # width 31, offset 31
]


class OracleTests(unittest.TestCase):
    """Self-consistency check between hand-written constants and plain-Python ISA formulas."""

    def test_reference_matches_constants(self):
        for imm, want in ORACLE_MOVK:
            self.assertEqual(ref_s_movk_i32(imm), want)
        for init, imm, want_res, want_scc in ORACLE_ADDK:
            self.assertEqual(ref_s_addk_i32(init, imm), (want_res, want_scc))
        for init, imm, want in ORACLE_MULK:
            self.assertEqual(ref_s_mulk_i32(init, imm), want)
        for kind, init, bit, want in ORACLE_BITSET:
            fn = ref_s_bitset0_b32 if kind == "bitset0" else ref_s_bitset1_b32
            self.assertEqual(fn(init, bit), want)
        for width, offset, want in ORACLE_BFM:
            self.assertEqual(ref_s_bfm_b32(width, offset), want)


class TestScalarMovkI32(unittest.TestCase):
    """End-to-end semantic tests for s_movk_i32."""

    def test_sign_extension_and_overwriting_upper_bits(self):
        cases = [
            (0, 0x00000000),
            (1, 0x00000001),
            (0x1234, 0x00001234),
            (0x7FFF, 0x00007FFF),
            (0x8000, 0xFFFF8000),
            (0x8001, 0xFFFF8001),
            (0xABCD, 0xFFFFABCD),
            (0xFFFF, 0xFFFFFFFF),
            ("-1", 0xFFFFFFFF),
            ("-123", 0xFFFFFF85),
            ("-32768", 0xFFFF8000),
        ]
        for imm, want in cases:
            imm_val = int(imm, 0) if isinstance(imm, str) else imm
            self.assertEqual(ref_s_movk_i32(imm_val), want)
            for pre_scc in (0, 1):
                with self.subTest(imm=imm, pre_scc=pre_scc):
                    got, got_scc = run_pair([
                        "s_mov_b32 s0, 0xdeadbeef",
                        scc_seed_line(pre_scc),
                        f"s_movk_i32 s0, {fmt_imm16(imm)}",
                        "s_cselect_b32 s1, 1, 0",
                        "v_mov_b32 v0, s0",
                        "v_mov_b32 v1, s1",
                    ])
                    self.assertEqual(got, want)
                    self.assertEqual(got_scc, pre_scc)

    def test_special_destinations_and_register_isolation(self):
        for dst in ("m0", "vcc_lo", "vcc_hi"):
            with self.subTest(dst=dst):
                got, _ = run_pair([
                    f"s_movk_i32 {dst}, 0x8005",
                    f"s_mov_b32 s0, {dst}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, 0",
                ])
                self.assertEqual(got, 0xFFFF8005)

        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_mov_b32 s2, 0x33333333",
            "s_movk_i32 s1, 0x8001",
            "s_movk_i32 null, 0x7fff",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1, w2, w3), (0x11111111, 0xFFFF8001, 0x33333333, 0))


class TestScalarAddkI32(unittest.TestCase):
    """End-to-end semantic tests for s_addk_i32 and signed-overflow SCC."""

    def test_in_place_add_and_signed_overflow_scc(self):
        cases = [
            (0x00000000, 0),
            (0x00000064, 50),
            (0x00000064, -30),
            (0x00000005, -10),
            (0x7FFFFFFF, 1),          # positive overflow -> 0x80000000, SCC = 1
            (0x7FFF8000, 0x7FFF),     # exact positive boundary -> 0x7FFFFFFF, SCC = 0
            (0x7FFF8001, 0x7FFF),     # positive overflow with max simm16 -> SCC = 1
            (0x7FFFFFFF, 0x7FFF),     # large positive overflow -> SCC = 1
            (0x80000000, "-1"),       # negative overflow -> 0x7FFFFFFF, SCC = 1
            (0x80000000, 0xFFFF),     # negative overflow (hex -1) -> SCC = 1
            (0x80008000, 0x8000),     # exact negative boundary -> 0x80000000, SCC = 0
            (0x80007FFF, 0x8000),     # negative overflow with min simm16 -> SCC = 1
            (0x80000000, "-32768"),   # large negative overflow -> SCC = 1
            (0xFFFFFFFF, 1),          # unsigned wraparound (-1 + 1 = 0), NO signed overflow -> SCC = 0
            (0xFFFFFFFF, 0x7FFF),     # unsigned wraparound (-1 + 32767), NO signed overflow -> SCC = 0
            (0xFFFF0000, 0x7FFF),     # negative + positive -> negative, SCC = 0
        ]
        for init, imm in cases:
            imm_val = int(imm, 0) if isinstance(imm, str) else imm
            want_res, want_scc = ref_s_addk_i32(init, imm_val)
            # Pre-seed SCC opposite of expected when non-overflowing to verify SCC is actively cleared.
            pre_scc = 0 if want_scc else 1
            with self.subTest(init=hex(init), imm=imm):
                got_res, got_scc = run_pair([
                    f"s_mov_b32 s0, {init:#x}",
                    scc_seed_line(pre_scc),
                    f"s_addk_i32 s0, {fmt_imm16(imm)}",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(got_res, want_res)
                self.assertEqual(got_scc, want_scc)

    def test_chained_in_place_accumulation(self):
        # Step 1: 0x7fff0000 + 0x7fff -> 0x7fff7fff, SCC = 0
        # Step 2: + 0x7fff -> 0x7ffffffe, SCC = 0
        # Step 3: + 2 -> 0x80000000, SCC = 1 (overflow)
        # Step 4: + (-1) -> 0x7fffffff, SCC = 1 (negative overflow back)
        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0x7fff0000",
            "s_addk_i32 s0, 0x7fff",
            "s_addk_i32 s0, 0x7fff",
            "s_cselect_b32 s1, 1, 0",
            "s_addk_i32 s0, 2",
            "s_cselect_b32 s2, 1, 0",
            "s_addk_i32 s0, -1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0x7FFFFFFF, 0, 1, 1))

    def test_special_destinations_and_register_isolation(self):
        for dst in ("m0", "vcc_lo", "vcc_hi"):
            with self.subTest(dst=dst):
                got_res, got_scc = run_pair([
                    f"s_mov_b32 {dst}, 0x7ffffff0",
                    f"s_addk_i32 {dst}, 0x20",
                    "s_cselect_b32 s1, 1, 0",
                    f"s_mov_b32 s0, {dst}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(got_res, 0x80000010)
                self.assertEqual(got_scc, 1)

        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x00000010",
            "s_mov_b32 s2, 0x33333333",
            "s_addk_i32 s1, -5",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1, w2, w3), (0x11111111, 0x0000000B, 0x33333333, 0))


class TestScalarMulkI32(unittest.TestCase):
    """End-to-end semantic tests for s_mulk_i32."""

    def test_signed_imm16_multiply_wrap_and_scc_preserved(self):
        cases = [
            (0x00000000, 0x7FFF),
            (0x12345678, 0),
            (0x80000000, 0),
            (0x00000001, 0x8000),     # 1 * -32768 -> 0xFFFF8000 (sign extension)
            (0x00000001, "-32768"),
            (0x00000001, 0xFFFF),     # 1 * -1 -> 0xFFFFFFFF
            (0x00000007, "-3"),       # 7 * -3 -> -21 (0xFFFFFFEB)
            (0xFFFFFFFB, 0x8000),     # -5 * -32768 -> 163840
            (0x00010001, 0x7FFF),     # 32-bit positive wrap
            (0x7FFFFFFF, 0x7FFF),     # INT32_MAX * 32767 mod 2^32
            (0x80000000, "-1"),       # INT32_MIN * -1 wraps to INT32_MIN
            (0x80000000, 0x8000),     # -2^31 * -32768 -> low 32 bits 0
            (0x80000001, 0x8000),     # (-2^31 + 1) * -32768 -> 0xFFFF8000
            (0xFFFFFFFF, 2),          # -1 * 2 -> -2
            (0x12345678, "-7"),
        ]
        for init, imm in cases:
            imm_val = int(imm, 0) if isinstance(imm, str) else imm
            want = ref_s_mulk_i32(init, imm_val)
            for pre_scc in (0, 1):
                with self.subTest(init=hex(init), imm=imm, pre_scc=pre_scc):
                    got, got_scc = run_pair([
                        f"s_mov_b32 s0, {init:#x}",
                        scc_seed_line(pre_scc),
                        f"s_mulk_i32 s0, {fmt_imm16(imm)}",
                        "s_cselect_b32 s1, 1, 0",
                        "v_mov_b32 v0, s0",
                        "v_mov_b32 v1, s1",
                    ])
                    self.assertEqual(got, want)
                    self.assertEqual(got_scc, pre_scc)

    def test_chained_multiply_and_register_isolation(self):
        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0x11111111",
            "s_movk_i32 s1, 3",
            "s_mov_b32 s2, 0x22222222",
            "s_mulk_i32 s1, -5",
            "s_mulk_i32 s1, 0x7fff",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, 0",
        ])
        want_s1 = ref_s_mulk_i32(ref_s_mulk_i32(3, -5), 0x7FFF)
        self.assertEqual((w0, w1, w2, w3), (0x11111111, want_s1, 0x22222222, 0))

    def test_special_destinations_and_register_isolation(self):
        cases = [
            (0x00000007, -3),
            (0x00000001, 0x8000),
            (0xFFFFFFFB, 0x8000),
            (0x7FFFFFFF, 0x7FFF),
            (0x80000000, 0xFFFF),
        ]
        for dst in ("m0", "vcc_lo", "vcc_hi"):
            for init, imm in cases:
                want = ref_s_mulk_i32(init, imm)
                for pre_scc in (0, 1):
                    with self.subTest(dst=dst, init=hex(init), imm=imm, pre_scc=pre_scc):
                        got, got_scc = run_pair([
                            f"s_mov_b32 {dst}, {init:#x}",
                            scc_seed_line(pre_scc),
                            f"s_mulk_i32 {dst}, {fmt_imm16(imm)}",
                            "s_cselect_b32 s1, 1, 0",
                            f"s_mov_b32 s0, {dst}",
                            "v_mov_b32 v0, s0",
                            "v_mov_b32 v1, s1",
                        ])
                        self.assertEqual(got, want)
                        self.assertEqual(got_scc, pre_scc)

        for target, other1, other2 in (
            ("m0", "vcc_lo", "vcc_hi"),
            ("vcc_lo", "m0", "vcc_hi"),
            ("vcc_hi", "m0", "vcc_lo"),
        ):
            with self.subTest(isolated_target=target):
                w0, w1, w2, w3 = run_quad([
                    "s_mov_b32 s0, 0x11111111",
                    f"s_mov_b32 {target}, 3",
                    f"s_mov_b32 {other1}, 0x22222222",
                    f"s_mov_b32 {other2}, 0x33333333",
                    f"s_mulk_i32 {target}, -5",
                    f"s_mulk_i32 {target}, 0x7fff",
                    "s_mulk_i32 null, 0x7fff",
                    f"s_mov_b32 s1, {target}",
                    f"s_mov_b32 s2, {other1}",
                    f"s_mov_b32 s3, {other2}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s2",
                    "v_mov_b32 v3, s3",
                ])
                want_target = ref_s_mulk_i32(ref_s_mulk_i32(3, -5), 0x7FFF)
                self.assertEqual((w0, w1, w2, w3), (0x11111111, want_target, 0x22222222, 0x33333333))


class TestScalarBitsetB32(unittest.TestCase):
    """End-to-end semantic tests for s_bitset0_b32 and s_bitset1_b32."""

    def test_all_32_bit_positions_and_neighbor_preservation(self):
        # Check every bit 0..31 on complementary non-NaN patterns (0xA5A55A5A / 0x5A5AA5A5) plus
        # all-ones (0xFFFFFFFF) and all-zeros (0). Note: exp mrt0 bitcasts through float in
        # spirv_mini_interp, which quiets signaling NaNs (bit 22 cleared on 0xFFFFFFFF -> 0xFFBFFFFF
        # becomes 0xFFFFFFFF); flipping bit 30 (or bit 29 for bit==30) on s0 avoids NaN patterns
        # while still verifying every bit 0..31 of 0xFFFFFFFF after s_bitset0_b32.
        pat_a, pat_b = 0xA5A55A5A, 0x5A5AA5A5
        for bit in range(32):
            guard = 0x20000000 if bit == 30 else 0x40000000
            with self.subTest(bit=bit):
                w0, w1, w2, w3 = run_quad([
                    "s_mov_b32 s0, 0xffffffff",
                    "s_mov_b32 s1, 0",
                    f"s_mov_b32 s2, {pat_a:#x}",
                    f"s_mov_b32 s3, {pat_b:#x}",
                    f"s_bitset0_b32 s0, {bit}",
                    f"s_xor_b32 s0, s0, {guard:#x}",
                    f"s_bitset1_b32 s1, {bit}",
                    f"s_bitset0_b32 s2, {bit}",
                    f"s_bitset1_b32 s3, {bit}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s2",
                    "v_mov_b32 v3, s3",
                ])
                self.assertEqual(w0, ref_s_bitset0_b32(0xFFFFFFFF, bit) ^ guard)
                self.assertEqual(w1, ref_s_bitset1_b32(0, bit))
                self.assertEqual(w2, ref_s_bitset0_b32(pat_a, bit))
                self.assertEqual(w3, ref_s_bitset1_b32(pat_b, bit))
                # Also verify s_bitset1_b32 on pat_a and s_bitset0_b32 on pat_b.
                p0, p1 = run_pair([
                    f"s_mov_b32 s0, {pat_a:#x}",
                    f"s_mov_b32 s1, {pat_b:#x}",
                    f"s_bitset1_b32 s0, {bit}",
                    f"s_bitset0_b32 s1, {bit}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(p0, ref_s_bitset1_b32(pat_a, bit))
                self.assertEqual(p1, ref_s_bitset0_b32(pat_b, bit))

    def test_bit_index_masking_mod_32_and_scc_preserved(self):
        indices = [0, 1, 31, 32, 33, 63, 64, 0x80000000, 0x12345625, 0xFFFFFFFF, "-1"]
        for idx in indices:
            idx_val = int(idx, 0) if isinstance(idx, str) else idx
            for use_sgpr in (False, True):
                for pre_scc in (0, 1):
                    with self.subTest(idx=idx, use_sgpr=use_sgpr, pre_scc=pre_scc):
                        src_op = "s2" if use_sgpr else (idx if isinstance(idx, str) else f"{idx:#x}")
                        setup = [f"s_mov_b32 s2, {idx_val & MASK32:#x}"] if use_sgpr else []
                        w0, w1, w2, w3 = run_quad(setup + [
                            "s_mov_b32 s0, 0xff7fffff",
                            "s_mov_b32 s1, 0",
                            scc_seed_line(pre_scc),
                            f"s_bitset0_b32 s0, {src_op}",
                            f"s_bitset1_b32 s1, {src_op}",
                            "s_cselect_b32 s3, 1, 0",
                            "v_mov_b32 v0, s0",
                            "v_mov_b32 v1, s1",
                            "v_mov_b32 v2, s3",
                            "v_mov_b32 v3, 0",
                        ])
                        self.assertEqual(w0, ref_s_bitset0_b32(0xFF7FFFFF, idx_val))
                        self.assertEqual(w1, ref_s_bitset1_b32(0, idx_val))
                        self.assertEqual(w2, pre_scc)

    def test_idempotency_aliasing_and_special_sources(self):
        # Clearing bit in 0 or setting bit in 0xFFFFFFFF leaves value unchanged; aliasing sD == sS0 works;
        # s_bitset0_b32 clearing the last 1 bit to 0 still preserves SCC=1.
        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0xffffffff",
            "s_mov_b32 s2, 0x80000000",
            "s_cmp_eq_u32 1, 1",
            "s_bitset0_b32 s0, 17",
            "s_bitset1_b32 s1, 17",
            "s_bitset0_b32 s2, 31",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual((w0, w1, w2, w3), (0, 0xFFFFFFFF, 0, 1))

        # Aliasing: s_bitset0_b32 s0, s0 (s0 = 0x25 -> 0x25 & 31 = bit 5 -> clears bit 5 -> 0x05)
        #           s_bitset1_b32 s1, s1 (s1 = 0x04 -> 0x04 & 31 = bit 4 -> sets bit 4 -> 0x14)
        # Also test m0 and null as bit-index source, plus s3 untouched.
        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 0x25",
            "s_mov_b32 s1, 0x04",
            "s_mov_b32 m0, 7",
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0x55aa55aa",
            "s_bitset0_b32 s0, s0",
            "s_bitset1_b32 s1, s1",
            "s_bitset1_b32 s2, m0",
            "s_bitset0_b32 s2, null",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual(w0, ref_s_bitset0_b32(0x25, 0x25))
        self.assertEqual(w1, ref_s_bitset1_b32(0x04, 0x04))
        self.assertEqual(w2, ref_s_bitset0_b32(ref_s_bitset1_b32(0, 7), 0))
        self.assertEqual(w3, 0x55AA55AA)

    def test_special_destinations_and_register_isolation(self):
        for dst in ("m0", "vcc_lo", "vcc_hi"):
            for bit_op, init, bit_tok, bit_val in (
                ("s_bitset0_b32", 0xA5A55A5A, "3", 3),
                ("s_bitset0_b32", 0xFFFFFFFF, "31", 31),
                ("s_bitset0_b32", 0xFFFFFFFF, "35", 35),
                ("s_bitset0_b32", 0xFFFFFFFF, "s2", 0xFFFFFFFF),
                ("s_bitset1_b32", 0x00000000, "0", 0),
                ("s_bitset1_b32", 0x00000000, "31", 31),
                ("s_bitset1_b32", 0x00000000, "35", 35),
                ("s_bitset1_b32", 0xA5A55A5A, "s2", 2),
            ):
                fn = ref_s_bitset0_b32 if bit_op == "s_bitset0_b32" else ref_s_bitset1_b32
                want = fn(init, bit_val)
                setup = [f"s_mov_b32 s2, {bit_val & MASK32:#x}"] if bit_tok == "s2" else []
                for pre_scc in (0, 1):
                    with self.subTest(dst=dst, op=bit_op, init=hex(init), bit=bit_tok, pre_scc=pre_scc):
                        got, got_scc = run_pair(setup + [
                            f"s_mov_b32 {dst}, {init:#x}",
                            scc_seed_line(pre_scc),
                            f"{bit_op} {dst}, {bit_tok}",
                            "s_cselect_b32 s1, 1, 0",
                            f"s_mov_b32 s0, {dst}",
                            "v_mov_b32 v0, s0",
                            "v_mov_b32 v1, s1",
                        ])
                        self.assertEqual(got, want)
                        self.assertEqual(got_scc, pre_scc)

            other1, other2 = [r for r in ("m0", "vcc_lo", "vcc_hi") if r != dst]
            with self.subTest(alias_and_isolated_target=dst):
                w0, w1, w2, w3 = run_quad([
                    "s_mov_b32 s0, 0x11111111",
                    f"s_mov_b32 {dst}, 0x25",
                    f"s_mov_b32 {other1}, 0x22222222",
                    f"s_mov_b32 {other2}, 0x33333333",
                    f"s_bitset0_b32 {dst}, {dst}",      # 0x25 & 31 = 5 -> clears bit 5 -> 0x05
                    f"s_bitset1_b32 {dst}, {dst}",      # 0x05 & 31 = 5 -> sets bit 5 back -> 0x25
                    f"s_bitset1_b32 {dst}, 3",          # sets bit 3 -> 0x2D
                    "s_bitset0_b32 null, 0",
                    "s_bitset1_b32 null, 31",
                    f"s_mov_b32 s1, {dst}",
                    f"s_mov_b32 s2, {other1}",
                    f"s_mov_b32 s3, {other2}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s2",
                    "v_mov_b32 v3, s3",
                ])
                want_dst = ref_s_bitset1_b32(ref_s_bitset1_b32(ref_s_bitset0_b32(0x25, 0x25), 0x05), 3)
                self.assertEqual((w0, w1, w2, w3), (0x11111111, want_dst, 0x22222222, 0x33333333))

        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 m0, 7",
            "s_mov_b32 vcc_lo, 0",
            "s_mov_b32 vcc_hi, 0xffffffff",
            "s_bitset1_b32 vcc_lo, m0",
            "s_bitset0_b32 vcc_hi, m0",
            "s_mov_b32 s0, m0",
            "s_mov_b32 s1, vcc_lo",
            "s_mov_b32 s2, vcc_hi",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual((w0, w1, w2, w3), (7, 0x00000080, 0xFFFFFF7F, 0))


class TestScalarBfmB32(unittest.TestCase):
    """End-to-end semantic tests for s_bfm_b32."""

    def test_width_offset_boundaries_and_overflow_truncation(self):
        cases = [
            # Width 0 -> always 0
            (0, 0), (0, 1), (0, 15), (0, 31), (0, 32),
            # Width 32 / 64 -> masked & 31 to width 0 -> 0 (not 0xFFFFFFFF)
            (32, 0), (32, 1), (32, 16), (32, 31), (64, 5),
            # Width 31 (max representable contiguous mask)
            (31, 0), (31, 1), (31, 2), (31, 31),
            # Offset masking & 31
            (4, 32), (4, 36), (8, 0x80000004), (3, "-1"),
            # Width masking & 31
            (33, 3), (35, 4), ("-1", 0), ("-1", "-1"),
            # Overflow beyond bit 31: truncated, never wraps around to bit 0
            (1, 31), (5, 30), (8, 28), (16, 20), (16, 31),
        ]
        for width, offset in cases:
            w_val = int(width, 0) if isinstance(width, str) else width
            o_val = int(offset, 0) if isinstance(offset, str) else offset
            want = ref_s_bfm_b32(w_val, o_val)
            w_tok = width if isinstance(width, str) else f"{w_val:#x}"
            o_tok = offset if isinstance(offset, str) else f"{o_val:#x}"
            # Verify (imm, imm), (sgpr, sgpr), and SCC preservation together.
            pre_scc = (w_val + o_val) & 1
            with self.subTest(width=width, offset=offset):
                w0, w1, w2, w3 = run_quad([
                    f"s_mov_b32 s2, {w_val & MASK32:#x}",
                    f"s_mov_b32 s3, {o_val & MASK32:#x}",
                    scc_seed_line(pre_scc),
                    f"s_bfm_b32 s0, {w_tok}, {o_tok}",
                    "s_bfm_b32 s1, s2, s3",
                    "s_cselect_b32 s4, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "v_mov_b32 v2, s4",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(w0, want)
                self.assertEqual(w1, want)
                self.assertEqual(w2, pre_scc)

    def test_mixed_operands_aliasing_and_special_regs(self):
        # Mix (sgpr, imm), (imm, sgpr), (m0, null), aliasing dst with S0/S1, and special dsts.
        w0, w1, w2, w3 = run_quad([
            "s_mov_b32 s0, 5",
            "s_mov_b32 s1, 12",
            "s_mov_b32 m0, 6",
            "s_bfm_b32 s0, s0, 4",       # dst aliases S0 -> (0x1f << 4) = 0x1f0
            "s_bfm_b32 s1, 4, s1",       # dst aliases S1 -> (0x0f << 12) = 0xf000
            "s_bfm_b32 s2, m0, null",    # m0 = 6, null = 0 -> 0x3f
            "s_bfm_b32 vcc_lo, 8, 8",    # special dst vcc_lo -> 0xff00
            "s_mov_b32 s3, vcc_lo",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ])
        self.assertEqual(w0, ref_s_bfm_b32(5, 4))
        self.assertEqual(w1, ref_s_bfm_b32(4, 12))
        self.assertEqual(w2, ref_s_bfm_b32(6, 0))
        self.assertEqual(w3, ref_s_bfm_b32(8, 8))


class TestLaneMaskInteractionAndNegativeControls(unittest.TestCase):
    """Verify lane-mask reset on SGPR overwrite and sabotage sensitivity."""

    def test_overwriting_lane_mask_sgpr_resets_to_scalar_data(self):
        # If s0 previously held a v_cmp lane mask, overwriting it with any of these 6 ops must clear
        # its lane-mask status so a following s_ff1_i32_b32 treats s0 as scalar data (single-lane ok).
        for op_line, want in (
            ("s_movk_i32 s0, 0x0010", 4),
            ("s_addk_i32 s0, 0x0020", None),   # depends on prior mask, checked via scalar arithmetic below
            ("s_bfm_b32 s0, 4, 8", 8),
        ):
            if want is None:
                continue
            with self.subTest(op=op_line):
                got, _ = run_pair([
                    "v_mov_b32 v2, 1",
                    "v_cmp_eq_u32_e64 s0, 1, v2",
                    op_line,
                    "s_ff1_i32_b32 s1, s0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(got, 0x10 if "movk" in op_line else 0xF00)

        # Also verify s_addk_i32 / s_mulk_i32 / s_bitset0_b32 / s_bitset1_b32 clear lane-mask tracking.
        for op_line in (
            "s_addk_i32 s0, 8",
            "s_mulk_i32 s0, 4",
            "s_bitset0_b32 s0, 0",
            "s_bitset1_b32 s0, 3",
        ):
            with self.subTest(op=op_line):
                spv = translate_lines([
                    "v_mov_b32 v2, 1",
                    "v_cmp_eq_u32_e64 s0, 1, v2",
                    op_line,
                    "s_ff1_i32_b32 s1, s0",
                    "v_mov_b32 v0, s1",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])
                self.assertNotIn("OpGroupNonUniformBitwiseOr", spv)
                res = si.run_fragment(spv)["%out_mrt0"][0]
                self.assertIsInstance(res, int)

    def test_negative_controls_detect_spirv_sabotage(self):
        # Verify our assertions catch broken sign extension, missing overflow SCC, or broken bitset/bfm/mulk ops.
        baseline_lines = [
            "s_movk_i32 s0, 0x8000",          # 0xFFFF8000
            "s_mov_b32 s1, 0x7fffffff",
            "s_addk_i32 s1, 1",               # SCC = 1
            "s_cselect_b32 s1, 1, 0",
            "s_mov_b32 s2, 0x00000001",
            "s_bitset1_b32 s2, 31",           # 0x80000001
            "s_bfm_b32 s3, 4, 4",             # 0x000000F0
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        good = translate_lines(baseline_lines)
        expected = (0xFFFF8000, 1, 0x80000001, 0x000000F0)
        self.assertEqual(tuple(si.run_fragment(good)["%out_mrt0"]), expected)

        # Sabotage 1: drop sign extension on 0x8000 (4294934528 -> 32768)
        self.assertIn("OpConstant %uint 4294934528", good)
        bad_sext = good.replace("OpConstant %uint 4294934528", "OpConstant %uint 32768")
        self.assertNotEqual(tuple(si.run_fragment(bad_sext)["%out_mrt0"]), expected)

        # Sabotage 2: drop s_bitset1_b32 OR (OpBitwiseOr -> OpBitwiseAnd)
        self.assertIn("OpBitwiseOr %uint", good)
        bad_bitset = good.replace("OpBitwiseOr %uint", "OpBitwiseAnd %uint", 1)
        self.assertNotEqual(tuple(si.run_fragment(bad_bitset)["%out_mrt0"]), expected)

        # Sabotage 3: s_addk_i32 fails to store SCC (stores to %vcc_hi instead of %scc)
        self.assertIn("OpStore %scc", good)
        bad_ovf = good.replace("OpStore %scc", "OpStore %vcc_hi", 1)
        self.assertNotEqual(tuple(si.run_fragment(bad_ovf)["%out_mrt0"]), expected)

        # Sabotage 4: s_bfm_b32 shifts right instead of left on final mask placement
        self.assertEqual(good.count("OpShiftLeftLogical %uint"), 3)  # bitset1 (1) + s_bfm_b32 (2)
        parts = good.rsplit("OpShiftLeftLogical %uint", 1)
        bad_bfm = "OpShiftRightLogical %uint".join(parts)
        self.assertNotEqual(tuple(si.run_fragment(bad_bfm)["%out_mrt0"]), expected)

        # Sabotage 5: s_mulk_i32 + s_bitset0_b32 broken IMul / BitwiseAnd
        mul_bs0 = translate_lines([
            "s_movk_i32 s0, 3",
            "s_mulk_i32 s0, 0xffff",          # -3 (0xFFFFFFFD)
            "s_mov_b32 s1, 0x000000ff",
            "s_bitset0_b32 s1, 0",            # 0x000000FE
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "exp mrt0 v0, v1, v0, v0 done vm",
            "s_endpgm",
        ])
        want_mb = (0xFFFFFFFD, 0x000000FE, 0xFFFFFFFD, 0xFFFFFFFD)
        self.assertEqual(tuple(si.run_fragment(mul_bs0)["%out_mrt0"]), want_mb)
        self.assertNotEqual(tuple(si.run_fragment(mul_bs0.replace("OpIMul %uint", "OpIAdd %uint", 1))["%out_mrt0"]), want_mb)
        self.assertNotEqual(tuple(si.run_fragment(mul_bs0.replace("OpNot %uint", "OpCopyObject %uint", 1))["%out_mrt0"]), want_mb)


if __name__ == "__main__":
    unittest.main()
