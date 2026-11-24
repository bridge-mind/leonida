"""Semantic interpreter tests for RDNA2 s_quadmask_b32, s_quadmask_b64 and s_bitreplicate_b64_b32 on scalar data.

RDNA2 ISA SOP1:
  * s_quadmask_b32: D[i] = OR of S0[4i+3:4i] for i in 0..7, the rest of D is 0; SCC = D != 0.
  * s_quadmask_b64: the same for the 16 quads of the 64-bit S0, D[63:16] = 0; SCC = D != 0.
  * s_bitreplicate_b64_b32: D[2i+1] = D[2i] = S0[i] for i in 0..31; SCC is not written.

Each case translates a synthetic pixel shader with tools/rdna2spirv.py and runs it in
tests/semantic/spirv_mini_interp.py. Every result word leaves through exp as two 16-bit halves ORed into
0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could change. A lane-mask
source or destination stays Unsupported: one invocation holds only its own lane's bit.
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

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
POISON_SCC = {0: "s_cmp_eq_u32 0, 0", 1: "s_cmp_eq_u32 0, 1"}   # SCC = 1 / SCC = 0 before the op


def quadmask(x: int, quads: int) -> int:
    return sum(1 << i for i in range(quads) if (x >> (4 * i)) & 0xF)


def bitreplicate(x: int) -> int:
    return sum(3 << (2 * i) for i in range(32) if (x >> i) & 1)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_quadmask.s")
    with open(os.path.join(_TMP_DIR.name, "ps_quadmask.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_quadmask\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run(lines):
    """Run `lines`, which leave D.lo in s10, D.hi in s11 and SCC in s12; return (D.lo, D.hi, SCC)."""
    tail = ["s_cselect_b32 s12, 1, 0"]
    for i, s in enumerate((10, 11)):
        tail += [f"v_mov_b32 v{10 + i}, s{s}",
                 f"v_and_b32 v{2 * i}, 0xffff, v{10 + i}", f"v_lshrrev_b32 v{2 * i + 1}, 16, v{10 + i}",
                 f"v_or_b32 v{2 * i}, 0x40000000, v{2 * i}", f"v_or_b32 v{2 * i + 1}, 0x40000000, v{2 * i + 1}"]
    tail += ["v_mov_b32 v4, s12", "v_or_b32 v4, 0x40000000, v4"]
    spv = translate(list(lines) + tail + ["exp mrt0 v0, v1, v2, v3 vm", "exp mrt1 v4, v4, v4, v4 done vm",
                                          "s_endpgm"])
    out = si.run_fragment(spv)
    w = list(out["%out_mrt0"]) + [out["%out_mrt1"][0]]
    assert all(v >> 16 == 0x4000 for v in w), [hex(v) for v in w]
    return (w[0] & 0xFFFF) | (w[1] & 0xFFFF) << 16, (w[2] & 0xFFFF) | (w[3] & 0xFFFF) << 16, w[4] & 0xFFFF


def load64(reg: int, v: int):
    return [f"s_mov_b32 s{reg}, {v & M32:#x}", f"s_mov_b32 s{reg + 1}, {v >> 32:#x}"]


VALUES32 = [0, 1, 0xF, 0x10, 0x80000000, 0x10000000, 0x01020408, 0x12345678, 0xF0F0F0F0, 0x0F0F0F0F, 0x88888888,
            0x00010000, 0xFFFF, 0xFFFF0000, M32]
VALUES64 = [0, 1, 1 << 63, 1 << 32, 0x00000001_00000000, 0x80000000_00000001, 0x12345678_9ABCDEF0,
            0xF0F0F0F0_0F0F0F0F, 0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000, M64]


class QuadmaskBitreplicateTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(quadmask(0x01020408, 8), 0x55)
        self.assertEqual(quadmask(0x80000000, 8), 0x80)
        self.assertEqual(quadmask(1 << 63, 16), 0x8000)
        self.assertEqual(bitreplicate(0x80000001), 0xC0000000_00000003)
        self.assertEqual(bitreplicate(0x00010000), 1 << 32 | 1 << 33)

    def test_quadmask_b32(self):
        for x in VALUES32:
            for poison in (0, 1):
                with self.subTest(x=hex(x), poison=poison):
                    want = quadmask(x, 8)
                    got = run([f"s_mov_b32 s2, {x:#x}", "s_mov_b32 s11, 0x5eed", POISON_SCC[poison],
                               "s_quadmask_b32 s10, s2"])
                    self.assertEqual(got, (want, 0x5eed, int(want != 0)))

    def test_quadmask_b64(self):
        for x in VALUES64:
            for poison in (0, 1):
                with self.subTest(x=hex(x), poison=poison):
                    want = quadmask(x, 16)
                    got = run(load64(2, x) + [POISON_SCC[poison], "s_quadmask_b64 s[10:11], s[2:3]"])
                    self.assertEqual(got, (want, 0, int(want != 0)))

    def test_bitreplicate_keeps_scc(self):
        for x in VALUES32:
            for poison in (0, 1):
                with self.subTest(x=hex(x), poison=poison):
                    want = bitreplicate(x)
                    got = run([f"s_mov_b32 s2, {x:#x}", POISON_SCC[poison], "s_bitreplicate_b64_b32 s[10:11], s2"])
                    self.assertEqual(got, (want & M32, want >> 32, 1 - poison))

    def test_immediates_and_aliasing(self):
        # A b64 inline constant is sign-extended: -1 has all 16 quads set.
        self.assertEqual(run(["s_quadmask_b64 s[10:11], -1"]), (0xFFFF, 0, 1))
        self.assertEqual(run(["s_quadmask_b32 s10, 0x80000010", "s_mov_b32 s11, 0"]), (0x82, 0, 1))
        self.assertEqual(run([POISON_SCC[0], "s_bitreplicate_b64_b32 s[10:11], 0x80000001"]), (3, 0xC0000000, 1))
        # Destination overlapping the source: all of S0 is read first.
        self.assertEqual(run(["s_mov_b32 s10, 0x12345678", POISON_SCC[0], "s_bitreplicate_b64_b32 s[10:11], s10"]),
                         (bitreplicate(0x12345678) & M32, bitreplicate(0x12345678) >> 32, 1))
        self.assertEqual(run(load64(10, 0x10000000_00000001) + ["s_quadmask_b64 s[10:11], s[10:11]"]),
                         (0x8001, 0, 1))

    def test_seeded(self):
        rng = random.Random(0x0AD)
        for _ in range(20):
            x = rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64)
            with self.subTest(x=hex(x)):
                self.assertEqual(run(load64(2, x) + ["s_quadmask_b64 s[10:11], s[2:3]"])[:2], (quadmask(x, 16), 0))
                lo = x & M32
                self.assertEqual(run([f"s_mov_b32 s2, {lo:#x}", "s_bitreplicate_b64_b32 s[10:11], s2"])[:2],
                                 (bitreplicate(lo) & M32, bitreplicate(lo) >> 32))

    def test_lane_masks_rejected(self):
        for lines in (["v_cmp_gt_u32_e32 vcc, 5, v0", "s_quadmask_b64 s[10:11], vcc"],
                      ["s_quadmask_b32 s10, exec_lo"],
                      ["s_quadmask_b64 exec, s[2:3]"],
                      ["v_cmp_gt_u32_e64 s[2:3], 5, v0", "s_quadmask_b64 s[10:11], s[2:3]"]):
            with self.subTest(lines=lines):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
