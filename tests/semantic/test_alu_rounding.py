"""Semantic interpreter tests for RDNA2 v_floor, v_ceil, v_trunc and v_rndne, in f32 and f16.

Each op rounds to an integral value in the operand's format: floor toward -inf, ceil toward +inf, trunc toward zero,
rndne to the nearest with ties to even. As in IEEE 754 roundToIntegral, a zero result keeps the operand's sign
(ceil(-0.5) = -0.0), and infinities come back unchanged. The f16 forms write the low half of the destination and keep
the high half (docs/design.md, "16-bit results"). Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported word with a
plain-Python reference. NaN inputs are left out (their payload is not modelled), and so are f32 denormals (the
float mode that would flush them is not modelled either).
"""
import math
import os
import random
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
OLD = 0xAAAA5555                     # the f16 destination before the op; its high half must survive
ROUND = {"floor": math.floor, "ceil": math.ceil, "trunc": math.trunc, "rndne": round}

F32_VALUES = [0x00000000, 0x80000000, 0x3F000000, 0xBF000000, 0x3FC00000, 0xBFC00000, 0x40200000, 0xC0200000,
              0x40600000, 0xC0600000, 0x3F7FFFFF, 0xBF7FFFFF, 0x4B000001, 0xCB7FFFFF, 0x7F7FFFFF, 0x7F800000,
              0xFF800000, 0x3EAAAAAB, 0x42C80000]
F16_VALUES = [0x0000, 0x8000, 0x3800, 0xB800, 0x3E00, 0xBE00, 0x4100, 0xC100, 0x4300, 0xC300, 0x3BFF, 0xBBFF,
              0x0001, 0x8001, 0x03FF, 0x6400, 0x63FF, 0xE3FF, 0x7BFF, 0x7C00, 0xFC00, 0x3555]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def ref(kind: str, x: float) -> float:
    if not math.isfinite(x):
        return x
    return math.copysign(float(ROUND[kind](x)), x)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_rounding.s")
    with open(os.path.join(_TMP_DIR.name, "ps_rounding.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, old: int = 0) -> int:
    """Seed v1 and the destination v0, run ``line``, return v0."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v0, {old:#x}", line]
    asm = "; test_rounding\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(f2u(ref("floor", -2.5)), 0xC0400000)
        self.assertEqual(f2u(ref("ceil", -0.5)), 0x80000000)
        self.assertEqual(f2u(ref("trunc", -0.5)), 0x80000000)
        self.assertEqual(f2u(ref("rndne", 2.5)), 0x40000000)
        self.assertEqual(f2u(ref("rndne", -0.5)), 0x80000000)
        self.assertEqual(f2h(ref("floor", h2f(0x8001))), 0xBC00)       # smallest negative denormal -> -1


class RoundingF32Tests(unittest.TestCase):
    def test_each_op_on_edge_values(self):
        for kind in ROUND:
            for x in F32_VALUES:
                with self.subTest(op=f"v_{kind}_f32", x=hex(x)):
                    self.assertEqual(run(f"v_{kind}_f32_e32 v0, v1", x), f2u(ref(kind, u2f(x))))

    def test_seeded_values(self):
        rng = random.Random(0xF1)
        for _ in range(30):
            # Exponents around the integral boundary (2^-2 .. 2^25) are where the rounding matters.
            x = rng.getrandbits(1) << 31 | rng.randrange(125, 152) << 23 | rng.getrandbits(23)
            kind = rng.choice(list(ROUND))
            with self.subTest(op=f"v_{kind}_f32", x=hex(x)):
                self.assertEqual(run(f"v_{kind}_f32_e32 v0, v1", x), f2u(ref(kind, u2f(x))))


class RoundingF16Tests(unittest.TestCase):
    def test_each_op_keeps_the_high_half(self):
        for kind in ROUND:
            for h in F16_VALUES:
                with self.subTest(op=f"v_{kind}_f16", x=hex(h)):
                    got = run(f"v_{kind}_f16_e32 v0, v1", 0x12340000 | h, OLD)
                    self.assertEqual(got, OLD & 0xFFFF0000 | f2h(ref(kind, h2f(h))))

    def test_op_sel_reads_the_high_half(self):
        v1 = 0xBE00 << 16 | 0x4100                                       # hi -1.5, lo 2.5
        self.assertEqual(run("v_floor_f16_e64 v0, v1 op_sel:[1,0]", v1, OLD), OLD & 0xFFFF0000 | f2h(-2.0))
        self.assertEqual(run("v_ceil_f16_e64 v0, v1 op_sel:[0,1]", v1, OLD), f2h(3.0) << 16 | OLD & 0xFFFF)


if __name__ == "__main__":
    unittest.main()
