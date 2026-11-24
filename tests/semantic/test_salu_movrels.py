"""Semantic interpreter tests for RDNA2 s_movrels_b32 and s_movrels_b64 (relative SGPR reads through M0).

RDNA2 ISA SOP1: s_movrels_b32 D = SGPR[S0 + M0]; s_movrels_b64 reads the pair SGPR[S0 + M0], SGPR[S0 + M0 + 1].
SCC is not written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py and runs it in
tests/semantic/spirv_mini_interp.py; results leave through exp as 16-bit halves ORed into 0x40000000. s_movreld (the
relative write) stays Unsupported. Standard library only; no GPU, no game files.
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
POISON_SCC = {0: "s_cmp_eq_u32 0, 0", 1: "s_cmp_eq_u32 0, 1"}   # SCC = 1 / SCC = 0 before the op


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_movrels.s")
    with open(os.path.join(_TMP_DIR.name, "ps_movrels.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_movrels\n" + "".join(f"\t{ln}\n" for ln in lines)
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


SEED = ["s_mov_b32 s20, 0x11111111", "s_mov_b32 s21, 0x22222222", "s_mov_b32 s22, 0x33333333",
        "s_mov_b32 s23, 0x44444444", "s_mov_b32 s24, 0x55555555", "s_mov_b32 s25, 0x66666666"]
WORDS = [0x11111111 * k for k in range(1, 7)]


class MovrelsTests(unittest.TestCase):
    def test_b32_reads_s0_plus_m0(self):
        for m0 in range(6):
            with self.subTest(m0=m0):
                got = run(SEED + [f"s_mov_b32 m0, {m0}", "s_movrels_b32 s10, s20", "s_mov_b32 s11, 0x5eed"])
                self.assertEqual(got[:2], (WORDS[m0], 0x5eed))

    def test_b32_base_register_and_m0_add(self):
        # S0 = s22, M0 = 3: SGPR[25].
        self.assertEqual(run(SEED + ["s_mov_b32 m0, 3", "s_movrels_b32 s10, s22", "s_mov_b32 s11, 0"])[0], WORDS[5])

    def test_b64_reads_the_pair(self):
        for m0 in (0, 1, 2, 4):
            with self.subTest(m0=m0):
                got = run(SEED + [f"s_mov_b32 m0, {m0}", "s_movrels_b64 s[10:11], s[20:21]"])
                self.assertEqual(got[:2], (WORDS[m0], WORDS[m0 + 1]))

    def test_m0_from_a_register_and_scc_kept(self):
        for poison in (0, 1):
            with self.subTest(poison=poison):
                got = run(SEED + ["s_mov_b32 s5, 4", "s_mov_b32 m0, s5", POISON_SCC[poison],
                                  "s_movrels_b32 s10, s20", "s_mov_b32 s11, 0"])
                self.assertEqual(got, (WORDS[4], 0, 1 - poison))

    def test_destination_overlapping_the_source(self):
        # D = s[21:22] overlaps the pair read (s[21:22] with M0 = 0 -> itself; M0 = 1 -> s[22:23]).
        got = run(SEED + ["s_mov_b32 m0, 1", "s_movrels_b64 s[21:22], s[21:22]", "s_mov_b32 s10, s21",
                          "s_mov_b32 s11, s22"])
        self.assertEqual(got[:2], (WORDS[2], WORDS[3]))

    def test_rejected_forms(self):
        for lines in (["s_movreld_b32 s10, s20"], ["s_movrels_b32 s10, 5"], ["s_movrels_b64 s10, s[20:21]"]):
            with self.subTest(lines=lines):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
