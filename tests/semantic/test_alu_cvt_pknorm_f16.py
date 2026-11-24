"""Semantic interpreter + spirv-val tests for RDNA2 v_cvt_pknorm_i16_f16 and v_cvt_pknorm_u16_f16.

D[15:0] = norm16(S0.f16), D[31:16] = norm16(S1.f16): each selected f16 half (op_sel[0..1], with neg/abs source
modifiers) is clamped to [-1, 1] (i16) or [0, 1] (u16), scaled by 32767 or 65535, and rounded once to nearest even
(NaN -> 0). Compared against an exact rational fractions.Fraction reference so near-tie f16 inputs that would
double-round in naive binary32 OpFMul are checked directly.
"""
from fractions import Fraction
import math
import os
import random
import shutil
import struct
import subprocess
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
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def f16_to_float(bits16: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits16 & 0xFFFF))[0]


def float_to_f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def ref_norm16_f16(bits16: int, snorm: bool) -> int:
    x = f16_to_float(bits16)
    if math.isnan(x):
        return 0
    lo = -1.0 if snorm else 0.0
    if x <= lo:
        cl = Fraction(-1 if snorm else 0)
    elif x >= 1.0:
        cl = Fraction(1)
    else:
        cl = Fraction(x)
    scale = 32767 if snorm else 65535
    return round(cl * scale) & 0xFFFF


OPS = {"v_cvt_pknorm_i16_f16": True, "v_cvt_pknorm_u16_f16": False}
EDGE_F16 = [
    0x0000,  # +0.0
    0x8000,  # -0.0
    0x0001,  # smallest positive f16 denormal (2^-24)
    0x8001,  # smallest negative f16 denormal (-2^-24)
    0x03FF,  # largest positive f16 denormal
    0x0400,  # smallest positive f16 normal (2^-14)
    0x3400,  # +0.25
    0xB400,  # -0.25
    0x37FF,  # 0.5 - 2^-12 (near-tie)
    0xB7FF,  # -(0.5 - 2^-12)
    0x3800,  # +0.5 (exact tie -> even)
    0xB800,  # -0.5 (exact tie -> even)
    0x3801,  # 0.5 + 2^-11 (double-rounding trap in naive f32 OpFMul)
    0xB801,  # -(0.5 + 2^-11) (double-rounding trap in naive f32 OpFMul)
    0x3BFF,  # 1.0 - 2^-11
    0xBBFF,  # -(1.0 - 2^-11)
    0x3C00,  # +1.0
    0xBC00,  # -1.0
    0x4000,  # +2.0 (clamped)
    0xC000,  # -2.0 (clamped)
    0x7C00,  # +inf
    0xFC00,  # -inf
    0x7E00,  # +qNaN
    0xFE00,  # -qNaN
    0x7C01,  # +sNaN
    0xFC01,  # -sNaN
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_pknorm_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_pknorm_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    out_lines = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = "; test_cvt_pknorm_f16\n" + "".join(f"\t{ln}\n" for ln in out_lines)
    asm += "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines) -> int:
    spv = translate_lines(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_op(op: str, s0: int, s1: int, mod0: str = "", mod1: str = "", extra: str = "") -> int:
    tail = f" {extra}" if extra else ""
    return run_lines([
        f"v_mov_b32 v1, {s0:#x}",
        f"v_mov_b32 v2, {s1:#x}",
        "v_mov_b32 v0, 0xdeadbeef",
        f"{op} v0, {mod0}v1, {mod1}v2{tail}",
    ])


class CvtPknormF16Tests(unittest.TestCase):
    def test_double_rounding_traps(self):
        # 0x3801 = 0.5 + 2^-11: exact products are 16399.49951171875 (snorm -> 16399) and
        # 32799.49951171875 (unorm -> 32799). Naive f32 OpFMul rounds both to .5 -> 16400 / 32800.
        self.assertEqual(ref_norm16_f16(0x3801, True), 16399)
        self.assertEqual(ref_norm16_f16(0xB801, True), (-16399) & 0xFFFF)
        self.assertEqual(ref_norm16_f16(0x3801, False), 32799)
        self.assertEqual(run_op("v_cvt_pknorm_i16_f16", 0x3801, 0xB801),
                         16399 | (((-16399) & 0xFFFF) << 16))
        self.assertEqual(run_op("v_cvt_pknorm_u16_f16", 0x3801, 0x37FF),
                         32799 | (ref_norm16_f16(0x37FF, False) << 16))

    def test_edge_values(self):
        for op, snorm in OPS.items():
            for idx, x in enumerate(EDGE_F16):
                y = EDGE_F16[(idx * 7 + 3) % len(EDGE_F16)]
                # Put garbage in the unselected high halves to prove low-half extraction ignores them.
                s0 = x | (0x7E00 << 16)
                s1 = y | (0x3C00 << 16)
                want = ref_norm16_f16(x, snorm) | (ref_norm16_f16(y, snorm) << 16)
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    self.assertEqual(run_op(op, s0, s1), want)

    def test_opsel_and_source_modifiers_and_inline_const(self):
        # op_sel picks high/low halves independently
        s0 = 0x3801 | (0xBC00 << 16)  # lo = 0.5+2^-11, hi = -1.0
        s1 = 0x3800 | (0x3C00 << 16)  # lo = 0.5,       hi = +1.0
        self.assertEqual(
            run_op("v_cvt_pknorm_i16_f16", s0, s1, extra="op_sel:[1,0]"),
            ((-32767) & 0xFFFF) | (16384 << 16),
        )
        self.assertEqual(
            run_op("v_cvt_pknorm_u16_f16", s0, s1, extra="op_sel:[0,1]"),
            32799 | (65535 << 16),
        )
        self.assertEqual(
            run_lines([
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_cvt_pknorm_i16_f16 v0, |v1|, -v2 op_sel:[1,1]",
            ]),
            32767 | (((-32767) & 0xFFFF) << 16),
        )
        # Inline f16 constant and SGPR source
        got = run_lines([
            "s_mov_b32 s0, 0x38003801",
            "v_cvt_pknorm_i16_f16 v0, -0.5, s0 op_sel:[0,0]",
        ])
        self.assertEqual(got, ((-16384) & 0xFFFF) | (16399 << 16))

    def test_seeded_pairs(self):
        rng = random.Random(0xF160)
        for _ in range(40):
            op = rng.choice(list(OPS))
            snorm = OPS[op]
            x = rng.randrange(0x10000)
            y = rng.randrange(0x10000)
            sel0 = rng.randrange(2)
            sel1 = rng.randrange(2)
            s0 = (x << 16) | rng.randrange(0x10000) if sel0 else x | (rng.randrange(0x10000) << 16)
            s1 = (y << 16) | rng.randrange(0x10000) if sel1 else y | (rng.randrange(0x10000) << 16)
            want = ref_norm16_f16(x, snorm) | (ref_norm16_f16(y, snorm) << 16)
            with self.subTest(op=op, x=hex(x), y=hex(y), sel=(sel0, sel1)):
                self.assertEqual(run_op(op, s0, s1, extra=f"op_sel:[{sel0},{sel1}]"), want)

    def test_clamp_and_opsel_dst(self):
        s0 = 0x3C003400  # hi = +1.0, lo = +0.25
        s1 = 0xBC003801  # hi = -1.0, lo = 0.5+2^-11
        self.assertEqual(run_op("v_cvt_pknorm_i16_f16", s0, s1, extra="clamp"), 0x400F2000)
        self.assertEqual(run_op("v_cvt_pknorm_u16_f16", s0, s1, extra="clamp"), 0x801F4000)
        want_dhi_i16 = {
            "op_sel:[0,0,1]": 0x20000000,
            "op_sel:[0,1,1]": 0x20000000,
            "op_sel:[1,0,1]": 0x7FFF0000,
            "op_sel:[1,1,1]": 0x7FFF0000,
            "op_sel:[1,0,1] clamp": 0x7FFF0000,
        }
        want_dhi_u16 = {
            "op_sel:[0,0,1]": 0x40000000,
            "op_sel:[0,1,1]": 0x40000000,
            "op_sel:[1,0,1]": 0xFFFF0000,
            "op_sel:[1,1,1]": 0xFFFF0000,
            "op_sel:[1,0,1] clamp": 0xFFFF0000,
        }
        for mod, want in want_dhi_i16.items():
            with self.subTest(op="v_cvt_pknorm_i16_f16", mod=mod):
                self.assertEqual(run_op("v_cvt_pknorm_i16_f16", s0, s1, extra=mod), want)
        for mod, want in want_dhi_u16.items():
            with self.subTest(op="v_cvt_pknorm_u16_f16", mod=mod):
                self.assertEqual(run_op("v_cvt_pknorm_u16_f16", s0, s1, extra=mod), want)

    def test_rejected_modifiers_and_spirv_val(self):
        for bad in (
            "v_cvt_pknorm_u16_f16 v0, v1, v2 mul:2",
            "v_cvt_pknorm_i16_f16 v0, v1, v2 mul:4",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 div:2",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 op_sel_hi:[1,1]",
            "v_cvt_pknorm_i16_f16 v0, v1, v2 neg_lo:[1,0]",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 neg_hi:[0,1]",
            "v_cvt_pknorm_i16_f16 v0, v1, v2 op_sel:[0,0,0,1]",
            "v_cvt_pknorm_u16_f16 v0, v1, v2 op_sel:[2,0,0]",
        ):
            with self.subTest(bad=bad), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines([bad])
        if SPIRV_AS and SPIRV_VAL:
            spv = translate_lines([
                "v_mov_b32 v1, 0xbc003801",
                "v_mov_b32 v2, 0x3c003800",
                "v_cvt_pknorm_i16_f16 v0, |v1|, -v2 op_sel:[1,0]",
                "v_cvt_pknorm_u16_f16 v0, v0, 0.5 op_sel:[0,0]",
            ])
            spv_bin = os.path.join(_TMP_DIR.name, "cvt_pknorm_f16.spv")
            p1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", spv_bin],
                                input=spv.encode("utf-8"), capture_output=True)
            self.assertEqual(p1.returncode, 0, p1.stderr.decode("utf-8", "replace"))
            p2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_bin], capture_output=True)
            self.assertEqual(p2.returncode, 0, p2.stderr.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
