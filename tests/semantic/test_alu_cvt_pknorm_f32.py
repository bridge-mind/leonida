"""Semantic interpreter tests for RDNA2 v_cvt_pknorm_i16_f32 and v_cvt_pknorm_u16_f32.

D[15:0] = norm16(S0), D[31:16] = norm16(S1): the f32 is clamped to [-1, 1] (i16) or [0, 1] (u16), scaled by 32767 or
65535 and rounded to nearest even. Mesa ACO emits these opcodes for NIR pack_snorm_2x16 / pack_unorm_2x16, defined in
nir_opcodes.py as fround_even(fclamp(x) * 32767.0 / 65535.0). NaN gives 0 (as v_cvt_i32_f32), and the exact product
is rounded once, not the f32 product: both measured on an RDNA2 iGPU (gfx1036, SHADER-339; the rounding is
SHADER-363). NEAR_TIES are inputs whose f32 product lands exactly on k + 0.5 while the exact one does not; rounding the
f32 product would be one off there. The whole dword is written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python reference. D leaves through exp as two
16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could
change.
"""
import os
import random
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def norm16(b: int, snorm: bool) -> int:
    x = f32(b)
    if x != x:
        return 0
    lo = -1.0 if snorm else 0.0
    x = min(max(x, lo), 1.0)
    # The exact product is rounded once to nearest even, as RDNA2 does (gfx1036, SHADER-339/363), not the f32
    # product. Fraction keeps it exact; round() on a Fraction is round-half-even.
    return round(Fraction(x) * (32767 if snorm else 65535)) & 0xFFFF


OPS = {"v_cvt_pknorm_i16_f32": True, "v_cvt_pknorm_u16_f32": False}
VALUES = [bits(v) for v in (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-30, 0.25, 0.75, 0.999, -0.999,
                             1 / 32767, 0.5 / 32767, 1.5 / 32767, 2.5 / 32767, 0.5 / 65535, 1.5 / 65535, 2.5 / 65535,
                             float("inf"), float("-inf"))] + [0x7FC00000, 0xFFC00000, 0x00000001, 0x80000001]
# (op, S0, S1, the dword an RDNA2 iGPU (gfx1036) gave, tools/pknorm_probe.py --oracle hw). S0 or S1 is a near tie:
# below k + 0.5 exactly (1.5 - 1.4e-9 gives 1, 3.5 - ... gives 3) or above it (256.5 + 1.5e-5 gives 257).
MEASURED = [
    ("v_cvt_pknorm_i16_f32", 0x38400180, 0x00000000, 0x00000001),
    ("v_cvt_pknorm_i16_f32", 0x38E001C0, 0x80000000, 0x00000003),
    ("v_cvt_pknorm_i16_f32", 0x39300160, 0x3F800000, 0x7FFF0005),
    ("v_cvt_pknorm_i16_f32", 0x3C004101, 0xBF800000, 0x80010101),
    ("v_cvt_pknorm_i16_f32", 0x3F000000, 0x3C014103, 0x01034000),
    ("v_cvt_pknorm_i16_f32", 0xBF000000, 0x3C024105, 0x0105C000),
    ("v_cvt_pknorm_u16_f32", 0x37C000C0, 0x40000000, 0xFFFF0001),
    ("v_cvt_pknorm_u16_f32", 0x386000E0, 0x80000000, 0x00000003),
    ("v_cvt_pknorm_u16_f32", 0x38B000B0, 0x3F800000, 0xFFFF0005),
    ("v_cvt_pknorm_u16_f32", 0x3B008081, 0xBF800000, 0x00000081),
    ("v_cvt_pknorm_u16_f32", 0x3F000000, 0x3B028083, 0x00838000),
    ("v_cvt_pknorm_u16_f32", 0xBF000000, 0x3B048085, 0x00850000),
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_pknorm.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_pknorm.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_cvt_pknorm\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int, mods: str = "") -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", "v_mov_b32 v0, 0xdeadbeef",
                      f"{op} v0, {mods}v1, v2"])


class CvtPknormTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(norm16(bits(1.0), True), 0x7FFF)
        self.assertEqual(norm16(bits(-2.0), True), 0x8001)           # -32767, not -32768
        self.assertEqual(norm16(bits(1.0), False), 0xFFFF)
        self.assertEqual(norm16(bits(-1.0), False), 0)
        self.assertEqual(norm16(bits(0.5), False), 32768)            # 32767.5 rounds to even
        self.assertEqual(norm16(bits(0.5), True), 16384)             # 16383.5 rounds to even
        self.assertEqual(norm16(0x7FC00000, True), 0)

    def test_near_ties_round_once_as_measured_on_rdna2(self):
        for op, s0, s1, want in MEASURED:
            with self.subTest(op=op, s0=hex(s0), s1=hex(s1)):
                self.assertEqual(norm16(s0, OPS[op]) | norm16(s1, OPS[op]) << 16, want)
                self.assertEqual(run_op(op, s0, s1), want)

    def test_values(self):
        for op, snorm in OPS.items():
            for x in VALUES:
                for y in VALUES[::5]:
                    with self.subTest(op=op, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(op, x, y), norm16(x, snorm) | norm16(y, snorm) << 16)

    def test_seeded(self):
        rng = random.Random(0xC0DE)
        for _ in range(40):
            op = rng.choice(list(OPS))
            x, y = bits(rng.uniform(-1.5, 1.5)), bits(rng.uniform(-1.5, 1.5))
            with self.subTest(op=op, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y), norm16(x, OPS[op]) | norm16(y, OPS[op]) << 16)

    def test_source_and_output_modifiers(self):
        half = bits(0.5)
        self.assertEqual(run_op("v_cvt_pknorm_i16_f32", half, half, "-"), (-16384 & 0xFFFF) | 16384 << 16)
        got = run_lines([f"v_mov_b32 v1, {bits(-0.5):#x}", f"v_mov_b32 v2, {half:#x}",
                         "v_cvt_pknorm_u16_f32 v0, |v1|, -v2"])
        self.assertEqual(got, 32768)                                  # |-0.5| -> 32768, -0.5 -> 0
        for op, snorm in OPS.items():
            for x, y in ((bits(-2.0), bits(2.0)), (bits(0.5), bits(-0.5)), (0x7FC00000, bits(1.0))):
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    got_clamp = run_lines([
                        f"v_mov_b32 v1, {x:#x}",
                        f"v_mov_b32 v2, {y:#x}",
                        f"{op} v0, v1, v2 clamp",
                    ])
                    self.assertEqual(got_clamp, norm16(x, snorm) | (norm16(y, snorm) << 16))
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_cvt_pknorm_i16_f32 v0, v1, v2 mul:2"])


if __name__ == "__main__":
    unittest.main()
