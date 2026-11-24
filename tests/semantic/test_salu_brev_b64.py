"""Semantic interpreter tests for RDNA2 s_brev_b64.

D = the 64 bits of S0 in reverse order (bit i goes to bit 63 - i), so D.lo = reverse(S0.hi) and D.hi =
reverse(S0.lo). SCC is not written. The SGPR pair is 64-bit data here, as for s_lshl_b64. Each case translates a
synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the
pair with a plain-Python reference. The pair leaves through exp as four 16-bit halves ORed into 0x40000000, so no
export channel is a NaN or denormal pattern that a float round trip could change; SCC comes back through
s_cselect_b32.
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
VALUES = [0, 1, M64, 1 << 63, 1 << 32, 0xFFFFFFFF, 0x0123456789ABCDEF, 0x8000000000000001, 0xF0F0F0F00F0F0F0F]


def brev64(x: int) -> int:
    return int(f"{x & M64:064b}"[::-1], 2)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_brev64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_brev64.dxbc"), "wb") as f:
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
    asm = ("; test_salu_brev64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
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


def run_op(x: int, scc_in: int) -> tuple:
    return run_lines([f"s_mov_b32 s2, {x & 0xFFFFFFFF:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"s_mov_b32 s4, {scc_in}",
                      "s_cmp_eq_u32 s4, 1", "s_brev_b64 s[0:1], s[2:3]"])


class BrevB64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(brev64(1), 1 << 63)
        self.assertEqual(brev64(0xFFFFFFFF), 0xFFFFFFFF00000000)
        self.assertEqual(brev64(brev64(0x0123456789ABCDEF)), 0x0123456789ABCDEF)

    def test_values_and_scc_kept(self):
        for x in VALUES:
            for scc_in in (0, 1):
                with self.subTest(x=hex(x), scc=scc_in):
                    self.assertEqual(run_op(x, scc_in), (brev64(x), scc_in))

    def test_seeded(self):
        rng = random.Random(0xB64)
        for _ in range(20):
            x = rng.getrandbits(64)
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(x, 0)[0], brev64(x))

    def test_inline_constant_source(self):
        # -1 as a 64-bit inline constant is all ones; 1 reverses to bit 63.
        self.assertEqual(run_lines(["s_brev_b64 s[0:1], -1"])[0], M64)
        self.assertEqual(run_lines(["s_brev_b64 s[0:1], 1"])[0], 1 << 63)

    def test_lane_mask_source_is_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["s_brev_b64 s[0:1], exec"])


if __name__ == "__main__":
    unittest.main()
