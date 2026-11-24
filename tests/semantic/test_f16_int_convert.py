"""Semantic tests for the translator's integer f32 -> f16 conversion (SHADER-56).

`Gen.f16_bits` replaces GLSL.std.450 PackHalf2x16 at the f16 ALU sites, because drivers fold a PackHalf2x16 ->
UnpackHalf2x16 round trip to a no-op (#513, #512). These tests run `v_cvt_f16_f32` (round to nearest even) and
`v_cvt_pkrtz_f16_f32` (round toward zero) through tools/rdna2spirv.py and the mini interpreter, and compare the
f16 bits with a Python reference over edge values (ties, the subnormal boundary, the overflow boundary, +-0,
+-inf, NaN) and a seeded sweep over every f32 exponent that can reach an f16. Each exported word has a zero high
half, so no word is a NaN bit pattern and the float export can't change it (#509).
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ref_rte(bits):
    """IEEE binary16 round to nearest even of the f32 `bits`; f32 denormals -> +-0; NaN -> None."""
    sign = (bits >> 16) & 0x8000
    a = bits & 0x7FFFFFFF
    if a > 0x7F800000:
        return None
    if a < 0x00800000:
        return sign
    try:
        return struct.unpack("<H", struct.pack("<e", f32(bits)))[0]
    except OverflowError:
        return sign | 0x7C00


def ref_rtz(bits):
    """binary16 round toward zero; finite values above the range -> +-65504; inf -> inf; NaN -> None."""
    sign = (bits >> 16) & 0x8000
    a = bits & 0x7FFFFFFF
    if a > 0x7F800000:
        return None
    if a == 0x7F800000:
        return sign | 0x7C00
    if a < 0x00800000:
        return sign
    x = Fraction(f32(a))
    lo, hi = 0, 0x7BFF                 # largest finite f16 magnitude <= x (binary search over the ordered bits)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if Fraction(struct.unpack("<e", struct.pack("<H", mid))[0]) <= x:
            lo = mid
        else:
            hi = mid - 1
    return sign | lo


_TMP = None


def tmpdir():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_f16cvt.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    return _TMP.name


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def convert(op, values):
    """Four f32 bit patterns -> the four f16 results of `op` (low halves)."""
    lines = []
    for i, v in enumerate(values):
        lines += [f"v_mov_b32 v{i}, 0", f"v_mov_b32 v{4 + i}, {v:#x}"]
        if op == "v_cvt_pkrtz_f16_f32":
            lines += [f"v_cvt_pkrtz_f16_f32 v{i}, v{4 + i}, 0", f"v_and_b32 v{i}, 0xffff, v{i}"]
        else:
            lines.append(f"{op} v{i}, v{4 + i}")
    asm = "; ps_f16cvt\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    path = os.path.join(tmpdir(), "disasm", "ps_f16cvt.s")
    with open(path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    return list(si.run_fragment(spv)["%out_mrt0"])


EDGES = [
    0x00000000, 0x80000000, 0x3F800000, 0xBF800000,      # +-0, +-1
    0x3F801000, 0x3F803000, 0x3F801001, 0x3F802FFF,      # ties to even (down / up), just above / below a tie
    0x477FE000, 0x477FEFFF, 0x477FF000, 0x47800000,      # 65504, just under the RTE overflow tie, the tie, 65536
    0x387FC000, 0x38800000, 0x387FE000, 0x33800000,      # just under 2^-14, 2^-14, subnormal tie, 2^-24
    0x33000000, 0x33000001, 0x32FFFFFF, 0x337FFFFF,      # 2^-25 (tie -> 0), just above, just below, under 2^-24
    0x35000000, 0x34C00000, 0x7F7FFFFF, 0x00000001,      # subnormals, f32 max, an f32 denormal
    0x7F800000, 0xFF800000, 0x7FC00000, 0xFFA3A095,      # +-inf, NaNs
]


class F16IntConvertTests(unittest.TestCase):
    def check(self, op, ref, values):
        for i in range(0, len(values), 4):
            chunk = (values[i:i + 4] + [0, 0, 0, 0])[:4]
            got = convert(op, chunk)
            for v, g in zip(chunk, got):
                want = ref(v)
                with self.subTest(op=op, x=hex(v)):
                    self.assertEqual(g >> 16, 0, "high half must stay 0")
                    if want is None:
                        self.assertEqual(g & 0x7E00, 0x7E00, f"want a quiet f16 NaN, got {g:#06x}")
                        self.assertEqual(g & 0x8000, (v >> 16) & 0x8000)
                    else:
                        self.assertEqual(g, want, f"{op}({v:#010x}) = {g:#06x}, want {want:#06x}")

    def test_rte_edges(self):
        self.check("v_cvt_f16_f32", ref_rte, EDGES)

    def test_rtz_edges(self):
        self.check("v_cvt_pkrtz_f16_f32", ref_rtz, EDGES)

    def sweep(self, seed, n):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            e = rng.randint(100, 145)                  # every exponent from below 2^-25 to above 65536
            out.append(rng.getrandbits(1) << 31 | e << 23 | rng.getrandbits(23))
        return out

    def test_rte_sweep(self):
        self.check("v_cvt_f16_f32", ref_rte, self.sweep(0x5156, 400))

    def test_rtz_sweep(self):
        self.check("v_cvt_pkrtz_f16_f32", ref_rtz, self.sweep(0x5157, 200))

    def test_no_packhalf_at_alu_sites(self):
        path = os.path.join(tmpdir(), "disasm", "ps_f16cvt.s")
        with open(path, "w", encoding="utf-8") as f:
            f.write("; ps_f16cvt\n\tv_mov_b32 v1, 0x3e000000\n\tv_cvt_f16_f32 v0, v1\n\tv_add_f16 v0, v0, v0\n"
                    "\tv_pk_add_f16 v2, v0, v0\n\tv_cvt_pkrtz_f16_f32 v3, v1, v1\n\tv_fma_f16 v0, v0, v0, v0\n"
                    "\texp mrt0 v0, v2, v3, v0 done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(path)
        self.assertNotIn("PackHalf2x16", spv)


if __name__ == "__main__":
    unittest.main()
