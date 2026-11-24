"""Semantic interpreter tests for RDNA2 s_bitset0_b64 and s_bitset1_b64.

D[S0 & 63] = 0 (bitset0) or 1 (bitset1) in the SGPR pair D, which is both source and destination; the other 63
bits are kept and SCC is not written. The pair is 64-bit data here, as for s_lshl_b64. Each case translates a
synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the pair
with a plain-Python reference. The pair leaves through exp as four 16-bit halves ORed into 0x40000000, so no export
channel is a NaN or denormal pattern that a float round trip could change; SCC comes back through s_cselect_b32.
"""
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

M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = [0, M64, 0x0123456789ABCDEF, 0x8000000000000001]
BITS = [0, 1, 31, 32, 33, 62, 63, 64, 95, 0xFFFFFFFF]


def ref(op: str, old: int, b: int) -> int:
    m = 1 << (b & 63)
    return (old | m) if op == "s_bitset1_b64" else (old & ~m & M64)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_bitset64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_bitset64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, return (s[0:1] as one 64-bit value, SCC)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = ["s_cselect_b32 s20, 0x40000001, 0x40000000"]
    for i, (r, sh) in enumerate(((0, 0), (0, 16), (1, 0), (1, 16))):
        tail += [f"s_lshr_b32 s{10 + i}, s{r}, {sh}", f"s_and_b32 s{10 + i}, s{10 + i}, 0xffff",
                 f"s_or_b32 s{10 + i}, s{10 + i}, 0x40000000", f"v_mov_b32 v{i}, s{10 + i}"]
    tail.append("v_mov_b32 v4, s20")
    asm = ("; test_salu_bitset64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v0, v1, v2, v3 vm\n\texp mrt1 v4, v4, v4, v4 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    w, scc = out["%out_mrt0"], out["%out_mrt1"][0]
    assert all(x >> 16 == 0x4000 for x in w) and scc >> 1 == 0x20000000, ([hex(x) for x in w], hex(scc))
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w)), scc & 1


def run_op(op: str, old: int, b: int, scc_in: int = 0) -> tuple:
    return run_lines([f"s_mov_b32 s0, {old & 0xFFFFFFFF:#x}", f"s_mov_b32 s1, {old >> 32:#x}", f"s_mov_b32 s2, {b:#x}",
                      f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1", f"{op} s[0:1], s2"])


class BitsetB64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("s_bitset1_b64", 0, 33), 1 << 33)
        self.assertEqual(ref("s_bitset0_b64", M64, 63), M64 >> 1)
        self.assertEqual(ref("s_bitset1_b64", 0, 64), 1)                    # index taken mod 64

    def test_values_and_scc_kept(self):
        for op in ("s_bitset0_b64", "s_bitset1_b64"):
            for old in OLD:
                for b in BITS:
                    for scc_in in (0, 1):
                        with self.subTest(op=op, old=hex(old), bit=b, scc=scc_in):
                            self.assertEqual(run_op(op, old, b, scc_in), (ref(op, old, b), scc_in))

    def test_seeded(self):
        rng = random.Random(0xB17)
        for _ in range(20):
            op, old, b = rng.choice(("s_bitset0_b64", "s_bitset1_b64")), rng.getrandbits(64), rng.getrandbits(32)
            with self.subTest(op=op, old=hex(old), bit=hex(b)):
                self.assertEqual(run_op(op, old, b)[0], ref(op, old, b))

    def test_inline_index(self):
        self.assertEqual(run_lines(["s_mov_b32 s0, 0", "s_mov_b32 s1, 0", "s_bitset1_b64 s[0:1], 40"])[0], 1 << 40)

    def test_lane_mask_destination_is_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["s_bitset1_b64 exec, 3"])


if __name__ == "__main__":
    unittest.main()
