"""Semantic interpreter tests for buffer_load_format_* (the V# FORMAT decode in tools/rdna2spirv.py).

The element is read from PhysicalStorageBuffer memory at V#.base + offset. An element under 4 bytes starts at a byte
shift inside its first dword, and the translation merges dword 1 above it: (w0 >> bsh) | (w1 << (32 - bsh)), picking
w0 when bsh is 0. The interpreter rejects shifts of 32 or more (undefined in SPIR-V, even on the discarded side of an
OpSelect), so every dword-aligned load here also checks that the merge keeps its shift below 32.

Integer and f16 formats are used: their decode is exact, so the expected words need no rounding model.
"""
from __future__ import annotations

import atexit
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
BASE = 0x1000
MEMORY = {BASE: 0x11223344, BASE + 4: 0xDEADBEEF, BASE + 8: 0xCAFEF00D, BASE + 12: 0x0BADC0DE}
# V# FORMAT codes (word3 bits 12-18), from rdna2spirv.FORMATS
FMT_U32, FMT_U32X2, FMT_U32X4, FMT_UINT8X4, FMT_UINT16X2, FMT_UINT16X4, FMT_F16 = 20, 62, 75, 60, 27, 69, 13
FMT_F32, FMT_F32X2, FMT_F32X3, FMT_UNORM8, FMT_UNORM8X4 = 22, 64, 74, 1, 56
FMT_UNORM16, FMT_SNORM16, FMT_UNORM16X2 = 7, 8, 23
DST_SEL_XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9
# DST_SEL (X, 0, 1, 1): code 4, 0, 1, 1 — SHADER-86 measures float 1.0 vs integer 1
DST_SEL_X_0_1_1 = 4 | 0 << 3 | 1 << 6 | 1 << 9

_TMP_DIR = None
_SRC_PATH = None
_EXIT_CLEANUP = False


def setUpModule():
    global _TMP_DIR, _SRC_PATH, _EXIT_CLEANUP
    if _TMP_DIR is not None:
        return  # load_format() may have created it already; replacing it would leak the first one
    _TMP_DIR = tempfile.TemporaryDirectory()
    # test_norm_decode imports load_format() and runs after this module's teardown, which recreates the folder;
    # unittest never tears this module down again. Registered once, after the first TemporaryDirectory, so it runs
    # before tempfile's own exit finalizer (atexit is last in, first out) and that finalizer stays quiet;
    # tearDownModule() cleans whichever folder is live at exit.
    if not _EXIT_CLEANUP:
        atexit.register(tearDownModule)
        _EXIT_CLEANUP = True
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_buf_fmt.s")
    with open(os.path.join(_TMP_DIR.name, "ps_buf_fmt.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def load_format(fmt: int, offset: int, ncomp: int = 4, dst_sel: int = DST_SEL_XYZW,
                memory: dict | None = None, d16: bool = False, *, op: str = "buffer_load_format",
                mods: str = "") -> tuple:
    """Return four MRT words for a formatted load. D16 packs two components per register.
    For a fixed-format load, pass its instruction name in op and its optional format modifier in mods."""
    if _TMP_DIR is None:
        setUpModule()
    suffix = ("d16_" if d16 else "") + "xyzw"[:ncomp]
    nregs = (ncomp + 1) // 2 if d16 else ncomp
    vdata = "v2" if nregs == 1 else f"v[2:{1 + nregs}]"
    regs = [f"v{2 + min(i, nregs - 1)}" for i in range(4)]
    seed = "\tv_mov_b32 v2, 0x12345678\n\tv_mov_b32 v3, 0xa9bcdef0\n" if d16 else ""
    asm = (
        "; test_buffer_load_format\n"
        f"\ts_mov_b32 s0, {BASE:#x}\n"
        "\ts_mov_b32 s1, 0\n"                  # base high bits 0, stride 0 (raw)
        "\ts_mov_b32 s2, 0x100\n"              # num_records (bytes)
        f"\ts_mov_b32 s3, {fmt << 12 | dst_sel | 0x30000000:#x}\n"
        f"\tv_mov_b32 v1, {offset:#x}\n"
        f"{seed}"
        f"\t{op}_{suffix} {vdata}, v1, s[0:3], 0{mods} offen\n"
        "\ts_waitcnt vmcnt(0)\n"
        f"\texp mrt0 {', '.join(regs)} done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv, memory=dict(memory if memory is not None else MEMORY))["%out_mrt0"]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f16_word(h: int) -> int:
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", h))[0]))[0]


class DwordAlignedElements(unittest.TestCase):
    def test_u32(self):
        for off, want in ((0, 0x11223344), (4, 0xDEADBEEF), (12, 0x0BADC0DE)):
            with self.subTest(off=off):
                self.assertEqual(load_format(FMT_U32, off, 1)[0], want)

    def test_u32x2_and_x4(self):
        self.assertEqual(load_format(FMT_U32X2, 4, 2)[:2], (0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_U32X4, 0), (0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x0BADC0DE))

    def test_uint8x4(self):
        self.assertEqual(load_format(FMT_UINT8X4, 4), (0xEF, 0xBE, 0xAD, 0xDE))

    def test_uint16x2_and_x4(self):
        self.assertEqual(load_format(FMT_UINT16X2, 4, 2)[:2], (0xBEEF, 0xDEAD))
        self.assertEqual(load_format(FMT_UINT16X4, 4), (0xBEEF, 0xDEAD, 0xF00D, 0xCAFE))


class SubDwordElements(unittest.TestCase):
    def test_f16_in_each_half(self):
        for off, half in ((4, 0xBEEF), (6, 0xDEAD), (8, 0xF00D), (10, 0xCAFE)):
            with self.subTest(off=off):
                self.assertEqual(load_format(FMT_F16, off, 1)[0], f16_word(half))


class D16DestinationHalves(unittest.TestCase):
    def test_uint32_saturates_before_packing(self):
        words = (0xfffe, 0x10000, 0x10001, 0xffffffff)
        memory = {BASE + 4 * i: word for i, word in enumerate(words)}
        for op, fmt, mods in (("buffer_load_format", FMT_U32X4, ""),
                              ("tbuffer_load_format", FMT_U32, " format:[BUF_FMT_32_32_32_32_UINT]")):
            with self.subTest(op=op):
                self.assertEqual(load_format(fmt, 0, 4, memory=memory, op=op, mods=mods), words)
                self.assertEqual(load_format(fmt, 0, 4, memory=memory, d16=True, op=op, mods=mods)[:2],
                                 (0xfffffffe, 0xffffffff))
                self.assertEqual(load_format(fmt, 0, 3, memory=memory, d16=True, op=op, mods=mods)[:2],
                                 (0xfffffffe, 0xa9bcffff))

    def test_odd_and_even_component_writes(self):
        # Little-endian UINT8 components at byte 0 are 44,33,22,11 (hex).
        for offset, words in (
            (0, ((0x12340044,), (0x00330044,), (0x00330044, 0xA9BC0022), (0x00330044, 0x00110022))),
            (0x100, ((0x12340000,), (0,), (0, 0xA9BC0000), (0, 0))),
        ):
            for ncomp, want in enumerate(words, 1):
                with self.subTest(offset=offset, ncomp=ncomp):
                    got = load_format(FMT_UINT8X4, offset, ncomp, d16=True)
                    self.assertEqual(got[:len(want)], want)


class D16NormalizedRounding(unittest.TestCase):
    """d16 narrows a normalized value to f16 with round to nearest even (#489; measured for typed buffer loads on an
    RX 6700, BUILD-161). These bytes are among the about half of unorm8 values where round toward zero, which
    PackHalf2x16 becomes on AMD drivers, gives 1 ulp less."""

    def test_unorm8_x(self):
        # byte 5 is 0xbe: 190/255 is 0x39f6 to nearest even, 0x39f5 toward zero
        self.assertEqual(load_format(FMT_UNORM8, 5, 1, d16=True)[0], 0x123439F6)

    def test_unorm8x4_xyzw(self):
        # bytes 4..7 are 0xef, 0xbe, 0xad, 0xde: 0x3b7f, 0x39f6, 0x396d, 0x3af7 (0x39f5 and 0x3af6 toward zero)
        self.assertEqual(load_format(FMT_UNORM8X4, 4, 4, d16=True)[:2], (0x39F63B7F, 0x3AF7396D))


class D16NormalizedRoundsOnce(unittest.TestCase):
    """BUILD-161, measured on an RX 6700 (gfx1031) with tools/d16_format_probe.py: d16 rounds the exact UNORM/SNORM
    quotient once, straight to f16. The f32 decode followed by f16_bits rounds twice, which is 1 ulp high where the
    f32 lands exactly on an f16 halfway point: UNORM16 0xffaf and 0xffef, SNORM16 0x7ff7 and 0x8009."""

    def d16_x(self, fmt, half, op="buffer_load_format", mods=""):
        return load_format(fmt, 0, 1, memory={BASE: half}, d16=True, op=op, mods=mods)[0]

    def test_unorm16_and_snorm16_halfway_values(self):
        for fmt, half, want in ((FMT_UNORM16, 0xFFEF, 0x3BFF), (FMT_UNORM16, 0xFFAF, 0x3BFD),
                                (FMT_SNORM16, 0x7FF7, 0x3BFF), (FMT_SNORM16, 0x8009, 0xBBFF)):
            with self.subTest(fmt=fmt, half=hex(half)):
                self.assertEqual(self.d16_x(fmt, half), 0x12340000 | want)

    def test_mtbuf_takes_the_same_rule(self):
        self.assertEqual(self.d16_x(FMT_UNORM16, 0xFFEF, "tbuffer_load_format", " format:[BUF_FMT_16_UNORM]"),
                         0x12343BFF)
        self.assertEqual(self.d16_x(FMT_SNORM16, 0x8009, "tbuffer_load_format", " format:[BUF_FMT_16_SNORM]"),
                         0x1234BBFF)

    def test_neighbours_and_the_f32_result_are_unchanged(self):
        self.assertEqual(self.d16_x(FMT_UNORM16, 0xFFEE), 0x12343BFF)      # 65518/65535: not a halfway f32
        self.assertEqual(self.d16_x(FMT_UNORM16, 0xFFFF), 0x12343C00)
        self.assertEqual(load_format(FMT_UNORM16, 0, 1, memory={BASE: 0xFFEF})[0], 0x3F7FF000)

    def test_a_float_format_rounds_toward_zero(self):
        # A 32_FLOAT element is narrowed toward zero, not to nearest even: gfx1036 gives 0x3bff and 0xbbfd for these
        # two halfway values (SHADER-358; more in test_buffer_f16_narrowing_hw.py).
        self.assertEqual(self.d16_x(FMT_F32, 0x3F7FF000), 0x12343BFF)
        self.assertEqual(self.d16_x(FMT_F32, 0xBF7FB000), 0x1234BBFD)

    def test_both_halves_of_a_pair(self):
        got = load_format(FMT_UNORM16X2, 0, 2, memory={BASE: 0xFFAF << 16 | 0xFFEF}, d16=True)[0]
        self.assertEqual(got, 0x3BFD3BFF)


class CrossingDwordBoundary(unittest.TestCase):
    """An element that starts mid-dword must read the following dwords, not re-read dword 0.

    #234 left this open: `need` used to be ceil(size/4), so a 4-byte format at byte 3
    (or an 8-byte format at byte 6) took its tail from the first dword. MEMORY is
    0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x0BADC0DE at 0x1000..0x100C.
    """

    def test_uint8x4_at_byte_3(self):
        # Bytes at 0x1003..0x1006: 0x11, 0xEF, 0xBE, 0xAD.
        self.assertEqual(load_format(FMT_UINT8X4, 3), (0x11, 0xEF, 0xBE, 0xAD))

    def test_uint8x4_at_byte_1(self):
        # Bytes at 0x1001..0x1004: 0x33, 0x22, 0x11, 0xEF.
        self.assertEqual(load_format(FMT_UINT8X4, 1), (0x33, 0x22, 0x11, 0xEF))

    def test_uint16x2_at_byte_2(self):
        # 4-byte element at 0x1002: 0x1122, then 0xBEEF (low half of the next dword).
        self.assertEqual(load_format(FMT_UINT16X2, 2, 2)[:2], (0x1122, 0xBEEF))

    def test_uint16x4_at_byte_6(self):
        # 8-byte element at 0x1006: 0xDEAD, 0xF00D, 0xCAFE, 0xC0DE.
        self.assertEqual(load_format(FMT_UINT16X4, 6), (0xDEAD, 0xF00D, 0xCAFE, 0xC0DE))


class MissingComponentsRepeat(unittest.TestCase):
    """SHADER-86 / RX 6700: missing channels are c mod n, not 0 or 1."""

    def test_one_component_formats_xxxx(self):
        self.assertEqual(load_format(FMT_U32, 4), (0xDEADBEEF,) * 4)
        self.assertEqual(load_format(FMT_F32, 4), (0xDEADBEEF,) * 4)
        # 8_UNORM at offset 4: low byte 0xEF → 239/255; all four channels match.
        want = f32_bits(239.0 / 255.0)
        self.assertEqual(load_format(FMT_UNORM8, 4), (want,) * 4)

    def test_two_and_three_component_xyxy_and_xyzx(self):
        self.assertEqual(load_format(FMT_U32X2, 4), (0xDEADBEEF, 0xCAFEF00D, 0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_F32X2, 4), (0xDEADBEEF, 0xCAFEF00D, 0xDEADBEEF, 0xCAFEF00D))
        self.assertEqual(load_format(FMT_F32X3, 0), (0x11223344, 0xDEADBEEF, 0xCAFEF00D, 0x11223344))


class UnormCorrectlyRounded(unittest.TestCase):
    """SHADER-86: UNORM is the correctly rounded quotient by (2^n-1), not a reciprocal multiply. Raw 3 → 0x3c40c0c1."""

    def test_unorm8x4_raw_3(self):
        mem = {BASE: 0x03030303}
        want = 0x3C40C0C1
        self.assertEqual(load_format(FMT_UNORM8X4, 0, memory=mem), (want,) * 4)
        # f32 reciprocal multiply disagrees (BetterJdue: 504 mismatches on rgba8).
        recip = f32_bits(struct.unpack("<f", struct.pack("<f", 3.0))[0]
                         * struct.unpack("<f", struct.pack("<f", 1.0 / 255.0))[0])
        self.assertNotEqual(want, recip)


class DstSelOneByFormat(unittest.TestCase):
    """SHADER-86: DST_SEL 1 is 1.0f for float/unorm, integer 1 for UINT."""

    def test_dst_sel_one_float_vs_uint(self):
        # DST_SEL (X, 0, 1, 1): Z and W come from the constant-1 code.
        f32 = load_format(FMT_F32, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(f32, (0xDEADBEEF, 0, 0x3F800000, 0x3F800000))
        u32 = load_format(FMT_U32, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u32, (0xDEADBEEF, 0, 1, 1))
        unorm = load_format(FMT_UNORM8, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(unorm[2:], (0x3F800000, 0x3F800000))
        u8 = load_format(FMT_UINT8X4, 4, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u8[2:], (1, 1))

    def test_dst_sel_one_out_of_bounds(self):
        """SHADER-429: Out-of-bounds loads return 0 for fetched channels, but DST_SEL=1 stays intact."""
        f32 = load_format(FMT_F32, 0x100, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(f32, (0, 0, 0x3F800000, 0x3F800000))
        u32 = load_format(FMT_U32, 0x100, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u32, (0, 0, 1, 1))
        unorm = load_format(FMT_UNORM8, 0x100, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(unorm, (0, 0, 0x3F800000, 0x3F800000))
        u8 = load_format(FMT_UINT8X4, 0x100, dst_sel=DST_SEL_X_0_1_1)
        self.assertEqual(u8, (0, 0, 1, 1))

    def test_dst_sel_one_out_of_bounds_d16(self):
        """SHADER-429: Out-of-bounds loads preserve DST_SEL=1 in packed d16 destinations."""
        f32_d16 = load_format(FMT_F32, 0x100, dst_sel=DST_SEL_X_0_1_1, d16=True)
        self.assertEqual(f32_d16[:2], (0, 0x3C003C00))
        u32_d16 = load_format(FMT_U32, 0x100, dst_sel=DST_SEL_X_0_1_1, d16=True)
        self.assertEqual(u32_d16[:2], (0, 0x00010001))


class TbufferDefaultFormat(unittest.TestCase):
    """GFX10 MTBUF: an omitted format: is BUF_FMT_8_UNORM (code 1), not the V# FORMAT. llvm-mc encodes
    `format:[BUF_FMT_8_UNORM]` and no modifier identically and disassembles both without it.

    The V# says FMT_U32, which would return the whole dword. The bytes are UNORM endpoints, 0xFF -> 1.0 and
    0x00 -> 0.0, so the expected words are exact.
    """

    MEM = {BASE: 0x112200FF}
    EXPLICIT = " format:[BUF_FMT_8_UNORM]"

    def tload(self, offset, ncomp, mods=""):
        return load_format(FMT_U32, offset, ncomp, memory=self.MEM, op="tbuffer_load_format", mods=mods)

    def test_x_unorm_endpoints_ignore_descriptor_format(self):
        for off, want in ((0, 0x3F800000), (1, 0)):
            with self.subTest(off=off):
                self.assertEqual(self.tload(off, 1)[0], want)

    def test_every_suffix_matches_explicit_8_unorm(self):
        for ncomp in (1, 2, 3, 4):
            with self.subTest(suffix="xyzw"[:ncomp]):
                got = self.tload(0, ncomp)[:ncomp]
                self.assertEqual(got[0], 0x3F800000)
                self.assertEqual(got, self.tload(0, ncomp, self.EXPLICIT)[:ncomp])

    def test_packed_load_uses_instruction_format(self):
        # The V# declares one u32. The instruction specifies four u8s at byte 4: EF BE AD DE.
        expected = ((0x123400EF,), (0x00BE00EF,), (0x00BE00EF, 0xA9BC00AD), (0x00BE00EF, 0x00DE00AD))
        for ncomp, want in enumerate(expected, 1):
            with self.subTest(ncomp=ncomp):
                got = load_format(FMT_U32, 4, ncomp, d16=True, op="tbuffer_load_format",
                                  mods=" format:[BUF_FMT_8_8_8_8_UINT]")
                self.assertEqual(got[:len(want)], want)

    def test_fixed_format_identity_ignores_descriptor_selectors(self):
        # Descriptor selectors are 0,0,0,1. MTBUF supplies its own X000, XY00 or XYZW.
        for fmt, source, want in (("32_UINT", 0x44, (0x44, 0, 0, 0)),
                                  ("16_16_UINT", 0x00330044, (0x44, 0x33, 0, 0)),
                                  ("8_8_8_8_UINT", 0x11223344, (0x44, 0x33, 0x22, 0x11))):
            for d16 in (False, True):
                with self.subTest(fmt=fmt, d16=d16):
                    got = load_format(FMT_U32, 0, 4, dst_sel=1 << 9, memory={BASE: source},
                                      d16=d16, op="tbuffer_load_format",
                                      mods=f" format:[BUF_FMT_{fmt}]")
                    if d16:
                        self.assertEqual(got[:2], (want[0] | want[1] << 16, want[2] | want[3] << 16))
                    else:
                        self.assertEqual(got, want)
        self.assertEqual(load_format(FMT_UINT8X4, 0, 4, dst_sel=1 << 9), (0, 0, 0, 1))

    def test_invalid_descriptor_suppresses_fixed_format_load(self):
        # Same address, 256-byte bound and instruction format. Only the V# FORMAT changes.
        cases = ((False, 4, (0x44, 0x33, 0x22, 0x11), (0, 0, 0, 0)),
                 (True, 3, (0x00330044, 0xa9bc0022), (0, 0xa9bc0000)))
        for d16, n, bound, unbound in cases:
            for fmt, want in ((FMT_U32, bound), (0, unbound)):
                with self.subTest(d16=d16, fmt=fmt):
                    got = load_format(fmt, 0, n, memory={BASE: 0x11223344}, d16=d16,
                                      op="tbuffer_load_format", mods=" format:[BUF_FMT_8_8_8_8_UINT]")
                    self.assertEqual(got[:len(want)], want)

    def test_packed_default_8_unorm(self):
        # The first byte is 0xff: 1.0 in f16 is 0x3c00. Preserve the upper destination half.
        got = load_format(FMT_U32, 0, 1, memory=self.MEM, d16=True, op="tbuffer_load_format")
        self.assertEqual(got[0], 0x12343C00)

    def test_explicit_invalid_still_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            self.tload(0, 1, " format:[BUF_FMT_INVALID]")


class InterpreterConversions(unittest.TestCase):
    """The ops the format decode needs from spirv_mini_interp: integer to float and GLSL FMin/FMax."""

    HEADER = (
        "OpCapability Shader\n%glsl = OpExtInstImport \"GLSL.std.450\"\nOpMemoryModel Logical GLSL450\n"
        "OpEntryPoint Fragment %main \"main\" %out\n%void = OpTypeVoid\n%fn = OpTypeFunction %void\n"
        "%uint = OpTypeInt 32 0\n%float = OpTypeFloat 32\n%_ptr_Output_uint = OpTypePointer Output %uint\n"
        "%out = OpVariable %_ptr_Output_uint Output\n"
    )

    def run_expr(self, consts, body):
        text = (self.HEADER + consts + "%main = OpFunction %void None %fn\n%entry = OpLabel\n" + body
                + "%bits = OpBitcast %uint %r\nOpStore %out %bits\nOpReturn\nOpFunctionEnd\n")
        return si.run_fragment(text)["%out"][0]

    def test_convert_u_and_s_to_f(self):
        consts = "%c = OpConstant %uint 4294967295\n"
        self.assertEqual(self.run_expr(consts, "%r = OpConvertUToF %float %c\n"), 0x4F800000)   # 4294967296.0
        self.assertEqual(self.run_expr(consts, "%r = OpConvertSToF %float %c\n"), 0xBF800000)   # -1.0

    def test_fmin_fmax(self):
        consts = "%a = OpConstant %float 1.5\n%b = OpConstant %float -2\n"
        self.assertEqual(self.run_expr(consts, "%r = OpExtInst %float %glsl FMax %a %b\n"), 0x3FC00000)
        self.assertEqual(self.run_expr(consts, "%r = OpExtInst %float %glsl FMin %a %b\n"), 0xC0000000)

    def test_fdiv_correctly_rounded(self):
        consts = "%a = OpConstant %float 3\n%b = OpConstant %float 255\n"
        self.assertEqual(self.run_expr(consts, "%r = OpFDiv %float %a %b\n"), 0x3C40C0C1)


if __name__ == "__main__":
    unittest.main()
