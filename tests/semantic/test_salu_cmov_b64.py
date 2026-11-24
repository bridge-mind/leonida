from __future__ import annotations

"""Semantic tests for RDNA2 s_cmov_b64 and s_cselect_b64 on 64-bit scalar *data* pairs.

Rule (s_cmov_b64): D[63:0] = SCC ? S0[63:0] : D[63:0]; SCC read, not written (RDNA2 ISA SOP1).
Rule (s_cselect_b64): D[63:0] = SCC ? S0[63:0] : S1[63:0]; SCC read, not written (RDNA2 ISA SOP2).

Only plain data is supported. A lane-mask register holds one lane's bit in this
translator, so selecting it as 64-bit data would silently read back a per-lane
bool instead of the wave's mask (issue #210). Those operands stay rejected.

Every program here is hand-written; the translator and mini interpreter from
this repository do the work. No LLVM, SPIR-V tools, GPU or game files are used.

Run directly:
    python -I tests/semantic/test_salu_cmov_b64.py -v
"""
import os
import re
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    dis = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(dis)
    _PS_PATH = os.path.join(dis, "ps_cmov64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cmov64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_cmov64\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def run_ps(lines):
    spv = translate_ps(lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return mrt0_words(spv)


def scc_line(scc):
    return "s_cmp_eq_u32 1, 1" if scc else "s_cmp_eq_u32 0, 1"


def load_pair_lines(reg, val64):
    return [f"s_mov_b32 s{reg}, {val64 & MASK32:#x}",
            f"s_mov_b32 s{reg + 1}, {(val64 >> 32) & MASK32:#x}"]


def export_pair_lines(reg):
    return [f"v_mov_b32 v0, s{reg}", f"v_mov_b32 v1, s{reg + 1}"]


def ref_cmov64(old, src, scc):
    return (src if scc else old) & MASK64


def ref_cselect_b64(s0: int, s1: int, scc: int) -> int:
    """Reference oracle for RDNA2 s_cselect_b64: D = SCC ? S0 : S1 (64-bit scalar data)."""
    return (s0 if scc else s1) & MASK64


class DataPairTests(unittest.TestCase):
    def test_plain_data_pair_is_accepted(self):
        # The explicit non-mask case: both dwords come from plain immediate moves,
        # not from EXEC, VCC or a compare. D is an ordinary SGPR data pair.
        old = 0xDEADBEEF_11223344
        src = 0x55667788_90ABCDEF
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cmov64(old, src, scc)
                words = run_ps(
                    load_pair_lines(2, old)
                    + load_pair_lines(4, src)
                    + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]",
                       "s_cselect_b32 s10, 1, 0"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, s10", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words, [want & MASK32, want >> 32, scc, 0])

    def test_edge_values(self):
        pairs = [
            (0, 0xFFFFFFFF_FFFFFFFF),
            (0x00000001_00000000, 0x80000000_7FFFFFFF),
            (0x7FFFFFFF_FFFFFFFF, 0x80000000_00000000),
            (0x00000002_80000001, 0x00000002_80000002),
        ]
        for old, src in pairs:
            for scc in (0, 1):
                with self.subTest(old=hex(old), src=hex(src), scc=scc):
                    want = ref_cmov64(old, src, scc)
                    words = run_ps(
                        load_pair_lines(2, old)
                        + load_pair_lines(4, src)
                        + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_immediate_and_null_sources(self):
        old = 0x55667788_11223344
        # shader_ir leaves a literal wider than 32 bits as a raw operand, which
        # src64_data does not accept (the same holds for s_mov_b64); use the
        # immediate forms the parser actually delivers.
        for src_asm, src in (("0x12345678", 0x12345678), ("42", 42),
                             ("0", 0), ("-1", MASK64), ("-2", MASK64 - 1), ("null", 0)):
            for scc in (0, 1):
                with self.subTest(src=src_asm, scc=scc):
                    want = ref_cmov64(old, src, scc)
                    words = run_ps(
                        load_pair_lines(2, old)
                        + [scc_line(scc), f"s_cmov_b64 s[2:3], {src_asm}"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_destination_aliasing(self):
        a64 = 0x55667788_11223344
        b64 = 0xDDEEFF00_99AABBCC
        # s[3:4] = (s4 << 32) | s3, with s4 = low(a64), s3 = low(b64).
        overlap_val = ((a64 & MASK32) << 32) | (b64 & MASK32)
        for source, src_val in (("s[4:5]", a64), ("s[3:4]", overlap_val)):
            for scc in (0, 1):
                with self.subTest(source=source, scc=scc):
                    want = ref_cmov64(a64, src_val, scc)
                    lines = load_pair_lines(3, b64) + load_pair_lines(4, a64)
                    lines += [scc_line(scc), f"s_cmov_b64 s[4:5], {source}"]
                    lines += export_pair_lines(4) + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    words = run_ps(lines)
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_constant_pairs_are_not_mistaken_for_masks(self):
        # 0 / -1 are EITHER in tools/lane_masks.py (the same bits as data), so
        # they must not be mistaken for lane masks here.
        for src_asm, src in (("-1", MASK64), ("0", 0)):
            for scc in (0, 1):
                with self.subTest(src=src_asm, scc=scc):
                    want = ref_cmov64(0x11223344_55667788, src, scc)
                    words = run_ps(
                        load_pair_lines(2, 0x11223344_55667788)
                        + [f"s_mov_b64 s[4:5], {src_asm}", scc_line(scc),
                           "s_cmov_b64 s[2:3], s[4:5]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_scc_is_not_written(self):
        old, src = 0x11111111_22222222, 0x33333333_44444444
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cmov64(old, src, scc)
                words = run_ps(
                    load_pair_lines(2, old)
                    + load_pair_lines(4, src)
                    + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]",
                       "s_addc_u32 s10, 10, 20"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, s10", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words, [want & MASK32, want >> 32, 30 + scc, 0])


class RejectionTests(unittest.TestCase):
    MASK_SOURCE = "s_cmov_b64 lane-mask source"
    MASK_DEST = "s_cmov_b64 lane-mask destination"
    DEST = "s_cmov_b64 destination"

    def test_lane_mask_source_rejected_with_message(self):
        cmp = "v_cmp_ne_u32_e64 s[20:21], 0, v0"
        cases = [
            ([cmp, "s_mov_b64 s[22:23], s[20:21]"], "s[22:23]"),
            (["s_mov_b64 s[22:23], exec"], "s[22:23]"),
            ([], "exec"),
            # A zero-extended 32-bit mask: one dword is a mask, the other is
            # an EITHER constant. holds_lane_mask() alone misses this pair.
            ([cmp, "s_mov_b32 s2, s20", "s_mov_b32 s3, 0"], "s[2:3]"),
            ([cmp, "s_mov_b32 s2, 0", "s_mov_b32 s3, s20"], "s[2:3]"),
        ]
        for prologue, src in cases:
            with self.subTest(src=src, prologue=prologue[-1] if prologue else "none"):
                lines = prologue + [scc_line(1), f"s_cmov_b64 s[2:3], {src}",
                                    "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.MASK_SOURCE)):
                    translate_ps(lines)

    def test_mask_destination_rejected_with_message(self):
        cmp = "v_cmp_ne_u32_e64 s[20:21], 0, v0"
        cases = [
            [cmp, "s_cmov_b64 s[20:21], s[4:5]"],
            [cmp, "s_mov_b32 s2, s20", "s_mov_b32 s3, 0", "s_cmov_b64 s[2:3], s[4:5]"],
        ]
        for lines in cases:
            with self.subTest(lines=lines[-1]):
                full = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.MASK_DEST)):
                    translate_ps(full)

    def test_ever_mask_pair_rejected_after_overwrite(self):
        # A register that held a mask earlier, then plain data, is refused: at a
        # CFG join lane_masks.py can drop the register (mask on one path, data on
        # another), so "not a mask now" does not prove it is data on every path.
        lines = ["v_cmp_ne_u32_e64 s[20:21], 0, v0",
                 "s_mov_b64 s[20:21], 0", "s_mov_b64 s[2:3], 0",
                 "s_cmov_b64 s[2:3], s[20:21]",
                 "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.MASK_SOURCE)):
            translate_ps(lines)

    def test_invalid_destinations_rejected_with_message(self):
        for line in ("s_cmov_b64 exec, s[4:5]", "s_cmov_b64 vcc, s[4:5]",
                     "s_cmov_b64 m0, s[4:5]", "s_cmov_b64 s4, s[4:5]",
                     "s_cmov_b64 null, s[4:5]"):
            with self.subTest(line=line):
                full = [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.DEST)):
                    translate_ps(full)


class ScalarCselectB64OracleTests(unittest.TestCase):
    def test_distinct_dword_data_pairs(self):
        pairs = [
            (0x55667788_11223344, 0xDDEEFF00_99AABBCC),
            (0x12345678_9ABCDEF0, 0xFEDCBA98_76543210),
            (0x00000002_80000001, 0x80000000_7FFFFFFE),
            (0x01020304_05060708, 0x11223344_55667788),
        ]
        for a64, b64 in pairs:
            for scc in (0, 1):
                with self.subTest(a=hex(a64), b=hex(b64), scc=scc):
                    want = ref_cselect_b64(a64, b64, scc)
                    words = run_ps(
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [scc_line(scc), "s_cselect_b64 s[2:3], s[4:5], s[6:7]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_edge_and_boundary_patterns(self):
        patterns = [
            (0, 0xFFFFFFFF_FFFFFFFF),
            (0xFFFFFFFF_FFFFFFFF, 0),
            (0x80000000_00000000, 0x7FFFFFFF_FFFFFFFF),
            (0x00000000_00000001, 0x80000000_00000000),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
            (0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000),
        ]
        for a64, b64 in patterns:
            for scc in (0, 1):
                with self.subTest(a=hex(a64), b=hex(b64), scc=scc):
                    want = ref_cselect_b64(a64, b64, scc)
                    words = run_ps(
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [scc_line(scc), "s_cselect_b64 s[2:3], s[4:5], s[6:7]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_immediate_operand_forms(self):
        cases = [
            ("42", "-2", 42, 0xFFFFFFFF_FFFFFFFE),
            ("0x10", "0x80000000", 0x10, 0x00000000_80000000),
            ("-16", "64", 0xFFFFFFFF_FFFFFFF0, 64),
            ("0x12345678", "0", 0x12345678, 0),
            ("0", "-1", 0, 0xFFFFFFFF_FFFFFFFF),
            ("-1", "0", 0xFFFFFFFF_FFFFFFFF, 0),
            ("null", "0x55", 0, 0x55),
            ("0xAA", "null", 0xAA, 0),
        ]
        for s0_asm, s1_asm, s0_val, s1_val in cases:
            for scc in (0, 1):
                with self.subTest(s0=s0_asm, s1=s1_asm, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    words = run_ps(
                        [scc_line(scc), f"s_cselect_b64 s[2:3], {s0_asm}, {s1_asm}"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_oracle_dwords_fidelity(self):
        a64 = 0xBBBBBBBB_AAAAAAAA
        b64 = 0xDDDDDDDD_CCCCCCCC
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cselect_b64(a64, b64, scc)
                words = run_ps(
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [scc_line(scc), "s_cselect_b64 s[2:3], s[4:5], s[6:7]"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words[0], want & MASK32)
                self.assertEqual(words[1], (want >> 32) & MASK32)


class ScalarCselectB64RoutingTests(unittest.TestCase):
    def test_scc_generation_and_dependency(self):
        # Verify SCC dependency when SCC is produced by different compare and arithmetic instructions.
        a64 = 0x11111111_22222222
        b64 = 0x33333333_44444444
        cases = [
            ("s_cmp_eq_u32 7, 7", 1),
            ("s_cmp_eq_u32 7, 8", 0),
            ("s_cmp_lg_u32 10, 20", 1),
            ("s_cmp_lg_u32 10, 10", 0),
            ("s_cmp_gt_i32 5, -1", 1),
            ("s_cmp_gt_i32 -5, 1", 0),
            ("s_cmp_lt_u32 1, 100", 1),
            ("s_cmp_lt_u32 100, 1", 0),
            ("s_add_u32 s10, 0xFFFFFFFF, 1", 1),
            ("s_add_u32 s10, 10, 20", 0),
        ]
        for scc_inst, expected_scc in cases:
            with self.subTest(inst=scc_inst, expected_scc=expected_scc):
                want = ref_cselect_b64(a64, b64, expected_scc)
                words = run_ps(
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [scc_inst, "s_cselect_b64 s[2:3], s[4:5], s[6:7]"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_scc_preservation_and_chaining(self):
        # s_cselect_b64 must not modify SCC. Verify by chaining multiple s_cselect_b64 ops
        # followed by s_addc_u32 and s_cselect_b32 which consume SCC.
        p1_a, p1_b = 0x01020304_05060708, 0x11121314_15161718
        p2_a, p2_b = 0x21222324_25262728, 0x31323334_35363738
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want1 = ref_cselect_b64(p1_a, p1_b, scc)
                want2 = ref_cselect_b64(p2_a, p2_b, scc)
                words = run_ps(
                    load_pair_lines(4, p1_a)
                    + load_pair_lines(6, p1_b)
                    + load_pair_lines(8, p2_a)
                    + load_pair_lines(10, p2_b)
                    + [
                        scc_line(scc),
                        "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                        "s_cselect_b64 s[12:13], s[8:9], s[10:11]",
                        "s_cselect_b32 s15, 1, 0",
                        "s_addc_u32 s14, 100, 200",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s12",
                        "v_mov_b32 v2, s14",
                        "v_mov_b32 v3, s15",
                    ]
                )
                self.assertEqual(words[0], want1 & MASK32)
                self.assertEqual(words[1], want2 & MASK32)
                self.assertEqual(words[2], 300 + scc)
                self.assertEqual(words[3], scc)

    def test_sgpr_pair_routing_layouts(self):
        # Test routing between diverse register allocations and non-sequential pair positions.
        a64 = 0xCAFEFEED_12345678
        b64 = 0xDEADBEEF_87654321
        routings = [
            (2, 4, 6),
            (8, 12, 16),
            (20, 24, 28),
            (14, 6, 22),
            (28, 2, 10),
            (4, 18, 8),
        ]
        for dst, s0, s1 in routings:
            for scc in (0, 1):
                with self.subTest(dst=dst, s0=s0, s1=s1, scc=scc):
                    want = ref_cselect_b64(a64, b64, scc)
                    words = run_ps(
                        load_pair_lines(s0, a64)
                        + load_pair_lines(s1, b64)
                        + [scc_line(scc), f"s_cselect_b64 s[{dst}:{dst + 1}], s[{s0}:{s0 + 1}], s[{s1}:{s1 + 1}]"]
                        + export_pair_lines(dst)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_sgpr_pair_and_immediate_mixed_routing(self):
        # Test routing when one operand is an SGPR pair and the other is an immediate constant.
        val64 = 0x01234567_89ABCDEF
        immediates = [
            ("0", 0),
            ("-1", MASK64),
            ("42", 42),
            ("0x12345678", 0x12345678),
            ("-2", MASK64 - 1),
        ]
        for imm_asm, imm_val in immediates:
            for sgpr_first in (True, False):
                for scc in (0, 1):
                    s0_asm, s1_asm = ("s[4:5]", imm_asm) if sgpr_first else (imm_asm, "s[4:5]")
                    s0_val, s1_val = (val64, imm_val) if sgpr_first else (imm_val, val64)
                    with self.subTest(s0=s0_asm, s1=s1_asm, scc=scc):
                        want = ref_cselect_b64(s0_val, s1_val, scc)
                        words = run_ps(
                            load_pair_lines(4, val64)
                            + [scc_line(scc), f"s_cselect_b64 s[2:3], {s0_asm}, {s1_asm}"]
                            + export_pair_lines(2)
                            + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                        )
                        self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_bidirectional_operand_swap_routing(self):
        # Swapping operand positions between two register pairs routes opposing values.
        pair_a = 0xAAAA0000_11112222
        pair_b = 0xBBBB0000_33334444
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want_fwd = ref_cselect_b64(pair_a, pair_b, scc)
                want_rev = ref_cselect_b64(pair_b, pair_a, scc)
                words = run_ps(
                    load_pair_lines(4, pair_a)
                    + load_pair_lines(6, pair_b)
                    + [
                        scc_line(scc),
                        "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                        "s_cselect_b64 s[8:9], s[6:7], s[4:5]",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, s8",
                        "v_mov_b32 v3, s9",
                    ]
                )
                self.assertEqual(words[:2], [want_fwd & MASK32, (want_fwd >> 32) & MASK32])
                self.assertEqual(words[2:], [want_rev & MASK32, (want_rev >> 32) & MASK32])


class ScalarCselectB64AliasingAndControlsTests(unittest.TestCase):
    DEST = "s_cselect_b64 destination"

    def test_destination_exact_aliasing(self):
        # D == S0, D == S1, and D == S0 == S1.
        a64 = 0x55667788_11223344
        b64 = 0xDDEEFF00_99AABBCC
        cases = [
            ("s[4:5]", "s[6:7]", a64, b64),
            ("s[6:7]", "s[4:5]", b64, a64),
            ("s[4:5]", "s[4:5]", a64, a64),
        ]
        for s0_spec, s1_spec, s0_val, s1_val in cases:
            for scc in (0, 1):
                with self.subTest(s0=s0_spec, s1=s1_spec, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [scc_line(scc), f"s_cselect_b64 s[4:5], {s0_spec}, {s1_spec}"]
                        + export_pair_lines(4)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    words = run_ps(lines)
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_partial_register_overlap_aliasing(self):
        # Destination s[4:5] overlaps s[3:4] on s4, and s[5:6] on s5.
        s3, s4, s5, s6 = 0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001
        p34 = (s4 << 32) | s3
        p56 = (s6 << 32) | s5
        for s0_spec, s1_spec, s0_val, s1_val in (
            ("s[3:4]", "s[5:6]", p34, p56),
            ("s[5:6]", "s[3:4]", p56, p34),
        ):
            for scc in (0, 1):
                with self.subTest(s0=s0_spec, s1=s1_spec, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    words = run_ps([
                        f"s_mov_b32 s3, {s3:#x}",
                        f"s_mov_b32 s4, {s4:#x}",
                        f"s_mov_b32 s5, {s5:#x}",
                        f"s_mov_b32 s6, {s6:#x}",
                        scc_line(scc),
                        f"s_cselect_b64 s[4:5], {s0_spec}, {s1_spec}",
                        "v_mov_b32 v0, s4",
                        "v_mov_b32 v1, s5",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                    ])
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_condition_inversion_equivalence(self):
        # cselect(A, B, cond) == cselect(B, A, not cond).
        # We test both external validation and explicit in-shader SCC inversion.
        a64 = 0x13579BDF_02468ACE
        b64 = 0xCAFE1234_BEEF5678
        for scc in (0, 1):
            with self.subTest(scc=scc):
                words = run_ps(
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [
                        scc_line(scc),
                        "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                        "s_cselect_b32 s10, 0, 1",
                        "s_cmp_lg_u32 s10, 0",
                        "s_cselect_b64 s[8:9], s[6:7], s[4:5]",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, s8",
                        "v_mov_b32 v3, s9",
                    ]
                )
                want = ref_cselect_b64(a64, b64, scc)
                self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])
                self.assertEqual(words[2:], [want & MASK32, (want >> 32) & MASK32])

    def test_negative_controls_condition_exclusion(self):
        # Negative control: verify that when SCC=0, S0 is NOT returned; when SCC=1, S1 is NOT returned.
        a64 = 0xAAAAAAAAAAAAAAAA
        b64 = 0x5555555555555555
        for scc, unselected in ((0, a64), (1, b64)):
            with self.subTest(scc=scc, unselected=hex(unselected)):
                words = run_ps(
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [scc_line(scc), "s_cselect_b64 s[2:3], s[4:5], s[6:7]"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                )
                res64 = (words[1] << 32) | words[0]
                self.assertNotEqual(res64, unselected)

    def test_invalid_destinations_rejected_with_message(self):
        # Destination must be a 2-SGPR pair for 64-bit scalar data select.
        for line in (
            "s_cselect_b64 s4, s[6:7], s[8:9]",
            "s_cselect_b64 m0, s[6:7], s[8:9]",
            "s_cselect_b64 null, s[6:7], s[8:9]",
            "s_cselect_b64 s[4:7], s[8:9], s[10:11]",
        ):
            with self.subTest(line=line):
                full = [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.DEST)):
                    translate_ps(full)


if __name__ == "__main__":
    unittest.main()
