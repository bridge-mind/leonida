"""Value tests for integer VALU instructions the translator handles but no other test reaches:

  v_bfm_b32       D = ((1 << (S0 & 31)) - 1) << (S1 & 31)
  v_and_or_b32    D = (S0 & S1) | S2
  v_add_lshl_u32  D = ((S0 + S1) mod 2^32) << (S2 & 31)
  v_lshl_or_b32   D = (S0 << (S1 & 31)) | S2
  v_mad_i32_i24   D = sext(S0[23:0]) * sext(S1[23:0]) + S2   (mod 2^32)
  v_mul_lo_u32    D = (S0 * S1) mod 2^32

Each case is a synthetic hand-written pixel-shader listing (operands loaded into VGPRs with v_mov_b32, one
instruction, exported to mrt0), translated by tools/rdna2spirv.py and executed by spirv_mini_interp.py. The
expected words are hand-computed constants per the RDNA2 ISA; a plain-Python reference (no translator) is checked
against the same constants so a typo in either is caught. Standard library only.
"""
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF


def sext24(v):
    v &= 0xFFFFFF
    return v - (1 << 24) if v & 0x800000 else v


REFERENCE = {
    "v_bfm_b32": lambda s0, s1: (((1 << (s0 & 31)) - 1) << (s1 & 31)) & MASK32,
    "v_and_or_b32": lambda s0, s1, s2: ((s0 & s1) | s2) & MASK32,
    "v_add_lshl_u32": lambda s0, s1, s2: (((s0 + s1) & MASK32) << (s2 & 31)) & MASK32,
    "v_lshl_or_b32": lambda s0, s1, s2: ((s0 << (s1 & 31)) | s2) & MASK32,
    "v_mad_i32_i24": lambda s0, s1, s2: (sext24(s0) * sext24(s1) + s2) & MASK32,
    "v_mul_lo_u32": lambda s0, s1: (s0 * s1) & MASK32,
}

# (sources..., expected) -- expected words computed by hand from the ISA formulas above.
CASES = {
    "v_bfm_b32": [
        (0, 5, 0x00000000),                    # width 0 -> empty mask
        (8, 4, 0x00000FF0),
        (31, 0, 0x7FFFFFFF),                   # width 31
        (31, 1, 0xFFFFFFFE),
        (32, 0, 0x00000000),                   # width 32 & 31 = 0
        (36, 40, 0x00000F00),                  # width 4, offset 8 after masking
        (4, 30, 0xC0000000),                   # 0xF << 30: top two bits wrap out past bit 31
        (0xFFFFFFFF, 0xFFFFFFFF, 0x80000000),  # width 31, offset 31: only bit 31 survives
    ],
    "v_and_or_b32": [
        (0xF0F0F0F0, 0xFF00FF00, 0x0000000F, 0xF000F00F),
        (0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000),
        (0x12345678, 0xFFFFFFFF, 0x80000000, 0x92345678),
        (0xAAAAAAAA, 0x55555555, 0x55555555, 0x55555555),
        (0xDEADBEEF, 0x0000FFFF, 0xFFFF0000, 0xFFFFBEEF),
        (0x0000000C, 0x0000000A, 0x00000003, 0x0000000B),  # (S0|S1)&S2 or S0&(S1|S2) would differ
        (0x000000FF, 0x0000000F, 0x0000000F, 0x0000000F),  # overlapping bits: OR, not ADD (ADD gives 0x1E)
        (0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0xFFFFFFFF),  # ADD would wrap to 0
    ],
    "v_add_lshl_u32": [
        (1, 2, 3, 0x00000018),
        (0xFFFFFFFF, 2, 4, 0x00000010),        # add wraps to 1, then << 4
        (0x80000000, 0x80000000, 0, 0x00000000),
        (0x40000000, 0x40000000, 1, 0x00000000),
        (0x12345678, 0, 36, 0x23456780),       # shift 36 & 31 = 4
        (3, 0, 33, 0x00000006),                # shift 33 & 31 = 1
        (0xFFFFFFF0, 0x11, 31, 0x80000000),    # wrapping add gives 1, then << 31
        (2, 3, 0, 0x00000005),
    ],
    "v_lshl_or_b32": [
        (1, 4, 1, 0x00000011),
        (0x0000FFFF, 16, 0x0000FFFF, 0xFFFFFFFF),
        (0x80000001, 1, 0, 0x00000002),        # bit 31 shifted out
        (5, 33, 0x100, 0x0000010A),            # shift 33 & 31 = 1
        (0xABCD, 48, 0x12, 0xABCD0012),        # shift 48 & 31 = 16
        (1, 31, 1, 0x80000001),
        (0xF, 2, 0x3C, 0x0000003C),            # OR, not ADD (ADD would give 0x78)
    ],
    "v_mad_i32_i24": [
        (2, 3, 4, 0x0000000A),
        (0x800000, 1, 0, 0xFF800000),          # -2^23 * 1
        (0xFFFFFF, 0xFFFFFF, 0, 0x00000001),   # -1 * -1
        (0x800000, 0x7FFFFF, 0, 0x00800000),   # -2^23 * (2^23-1) = -2^46 + 2^23
        (0x7FFFFF, 0x7FFFFF, 0, 0xFF000001),   # (2^23-1)^2 = 2^46 - 2^24 + 1
        (0xAB000005, 0xFF000003, 0, 0x0000000F),  # bits 31:24 ignored: 5 * 3
        (0x12FFFFFF, 0x7F000002, 0, 0xFFFFFFFE),  # bits 31:24 ignored: -1 * 2
        (3, 4, 0xFFFFFFF6, 0x00000002),        # 12 + (-10)
        (0xFFFFFE, 5, 0xFFFFFFFF, 0xFFFFFFF5), # -2 * 5 + (-1) = -11
        (0xFFFFFF, 1, 1, 0x00000000),          # -1 + 1 wraps to 0
    ],
    "v_mul_lo_u32": [
        (6, 7, 0x0000002A),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x00000001),  # (2^32-1)^2 = 2^64 - 2^33 + 1
        (0x00010000, 0x00010000, 0x00000000),  # 2^32
        (0x80000000, 3, 0x80000000),
        (0xFFFFFFFF, 2, 0xFFFFFFFE),
        (0x00010001, 0x0000FFFF, 0xFFFFFFFF),  # (2^16+1)(2^16-1) = 2^32 - 1
        (0x00010001, 0x00010001, 0x00020001),  # 2^32 + 2^17 + 1
        (0x12345678, 0x9ABCDEF0, 0x242D2080),
    ],
}

_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_int_misc.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_op(op, srcs):
    """Translate and run `op v0, v1, v2[, v3]` with the sources loaded by v_mov_b32; return the mrt0 word."""
    regs = [f"v{i + 1}" for i in range(len(srcs))]
    lines = ["; test_int_misc"]
    lines += [f"\tv_mov_b32 {r}, {s & MASK32:#x}" for r, s in zip(regs, srcs)]
    lines.append(f"\t{op} v0, {', '.join(regs)}")
    lines += ["\texp mrt0 v0, v0, v0, v0 done vm", "\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_int_misc.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{op}: the four exported words differ: {[hex(w) for w in words]}")
    return words[0]


class ReferenceMatchesHandConstants(unittest.TestCase):
    def test_reference(self):
        for op, cases in CASES.items():
            for *srcs, expected in cases:
                with self.subTest(op=op, srcs=[hex(s) for s in srcs]):
                    self.assertEqual(REFERENCE[op](*srcs), expected)


class TranslatedValues(unittest.TestCase):
    def check(self, op):
        for *srcs, expected in CASES[op]:
            with self.subTest(srcs=[hex(s) for s in srcs]):
                got = run_op(op, srcs)
                self.assertEqual(got, expected, f"{op} {[hex(s) for s in srcs]}: got {got:#010x}, "
                                                f"expected {expected:#010x}")

    def test_v_bfm_b32(self):
        self.check("v_bfm_b32")

    def test_v_and_or_b32(self):
        self.check("v_and_or_b32")

    def test_v_add_lshl_u32(self):
        self.check("v_add_lshl_u32")

    def test_v_lshl_or_b32(self):
        self.check("v_lshl_or_b32")

    def test_v_mad_i32_i24(self):
        self.check("v_mad_i32_i24")

    def test_v_mul_lo_u32(self):
        self.check("v_mul_lo_u32")


if __name__ == "__main__":
    unittest.main()
