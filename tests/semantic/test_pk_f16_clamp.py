"""Semantic tests for the clamp bit on packed f16 VOP3P ops (v_pk_add/mul/fma/max/min_f16 ... clamp).

With clamp, each f16 half of the result is clamped to [0.0, 1.0] (RDNA2 ISA, VOP3P CLMP), the same as the scalar
f16 ops. The translator clamps the vec2 with GLSL FClamp; the semantic interpreter used to take FClamp/FMax on
scalars only and failed with a TypeError on any packed clamp. Both halves leave through exp with a zero high half.
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

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f2h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def pack(lo, hi):
    return f2h(lo) | f2h(hi) << 16


_TMP = None


def run(lines):
    """Run the lines, return (low half, high half) of v0 as f16 bits."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_pkclamp.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    path = os.path.join(_TMP.name, "disasm", "ps_pkclamp.s")
    lines = list(lines) + ["v_and_b32 v1, 0xffff, v0", "v_lshrrev_b32 v2, 16, v0"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("; ps_pkclamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v1, v2, v1, v1 done vm\n\ts_endpgm\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def clamp01(x):
    return min(max(x, 0.0), 1.0)


# (op, a, b, c or None, python result per half)
CASES = [
    ("v_pk_add_f16", (0.75, -3.0), (0.5, 1.0), None, lambda a, b, c: a + b),
    ("v_pk_add_f16", (0.25, 0.5), (0.25, 0.25), None, lambda a, b, c: a + b),
    ("v_pk_mul_f16", (4.0, -0.5), (0.5, 0.5), None, lambda a, b, c: a * b),
    ("v_pk_max_f16", (2.0, -1.0), (0.5, -2.0), None, lambda a, b, c: max(a, b)),
    ("v_pk_min_f16", (2.0, 0.125), (8.0, 0.25), None, lambda a, b, c: min(a, b)),
    ("v_pk_fma_f16", (2.0, -1.0), (1.5, 0.5), (0.25, 0.25), lambda a, b, c: a * b + c),
]


class PackedClampTests(unittest.TestCase):
    def test_clamp_each_half(self):
        for op, a, b, c, fn in CASES:
            for clamp in (False, True):
                srcs = "v4, v5" + (", v6" if c else "")
                lines = [f"v_mov_b32 v4, {pack(*a):#x}", f"v_mov_b32 v5, {pack(*b):#x}"]
                if c:
                    lines.append(f"v_mov_b32 v6, {pack(*c):#x}")
                lines.append(f"{op} v0, {srcs}" + (" clamp" if clamp else ""))
                want = tuple(f2h(clamp01(r) if clamp else r)
                             for r in (fn(a[0], b[0], c[0] if c else 0), fn(a[1], b[1], c[1] if c else 0)))
                with self.subTest(op=op, clamp=clamp):
                    self.assertEqual(run(lines), want)


if __name__ == "__main__":
    unittest.main()
