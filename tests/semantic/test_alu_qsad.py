"""Semantic interpreter tests for RDNA2 v_qsad_pk_u16_u8 and v_mqsad_pk_u16_u8.

Four byte-SADs at once (RDNA ISA pseudocode). S0 is 64 bits, S1 32 bits, S2 and D 64 bits holding four u16 fields:
  D.u16[k] = (S2.u16[k] + sum over the 4 bytes of |W_k.byte - S1.byte|) & 0xFFFF, W_k = S0 bits [8k+31 : 8k]
so the four windows slide over S0 one byte at a time. v_mqsad_pk_u16_u8 is masked: a byte where S1's byte is 0 adds
nothing (as v_msad_u8). Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares D with a plain-Python reference. D leaves through exp as four
16-bit fields ORed into 0x40000000, so no export channel is a NaN or denormal pattern that a float round trip could
change.
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


def ref(op: str, x: int, y: int, acc: int) -> int:
    out = 0
    for k in range(4):
        win = (x >> (8 * k)) & 0xFFFFFFFF
        s = 0
        for i in range(4):
            a, b = (win >> (8 * i)) & 0xFF, (y >> (8 * i)) & 0xFF
            if op == "v_mqsad_pk_u16_u8" and b == 0:
                continue
            s += abs(a - b)
        out |= ((((acc >> (16 * k)) & 0xFFFF) + s) & 0xFFFF) << (16 * k)
    return out


OPS = ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8")
X64 = [0, M64, 0x0807060504030201, 0xFF00FF00FF00FF00, 0x0123456789ABCDEF, 0x8080808080808080]
Y32 = [0, 0xFFFFFFFF, 0x04030201, 0x00FF00FF, 0x80808080, 0x12000034]
ACC = [0, 0xFFFFFFFFFFFFFFFF, 0x0001000200030004, 0xFC00FC00FC00FC00]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_qsad.s")
    with open(os.path.join(_TMP_DIR.name, "ps_qsad.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> int:
    """Run the lines, return v[0:1] (exported as four 16-bit fields, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v20, 0xffff, v0", "v_lshrrev_b32 v21, 16, v0", "v_and_b32 v22, 0xffff, v1",
                           "v_lshrrev_b32 v23, 16, v1"] + [f"v_or_b32 v{r}, 0x40000000, v{r}" for r in range(20, 24)]
    asm = "; test_qsad\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v20, v21, v22, v23 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(v >> 16 == 0x4000 for v in w), [hex(v) for v in w]
    return sum((v & 0xFFFF) << (16 * i) for i, v in enumerate(w))


def run_op(op: str, x: int, y: int, acc: int) -> int:
    return run_lines([f"v_mov_b32 v2, {x & 0xFFFFFFFF:#x}", f"v_mov_b32 v3, {x >> 32:#x}", f"v_mov_b32 v4, {y:#x}",
                      f"v_mov_b32 v6, {acc & 0xFFFFFFFF:#x}", f"v_mov_b32 v7, {acc >> 32:#x}",
                      f"{op} v[0:1], v[2:3], v4, v[6:7]"])


class QsadTests(unittest.TestCase):
    def test_reference(self):
        # Window k of 0x0807060504030201 is bytes k+1..k+4; against S1 = 0x04030201 the SADs are 0, 4, 8, 12.
        self.assertEqual(ref("v_qsad_pk_u16_u8", 0x0807060504030201, 0x04030201, 0), 0x000C000800040000)
        self.assertEqual(ref("v_qsad_pk_u16_u8", 0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF), 0x03FB03FB03FB03FB)   # wraps
        self.assertEqual(ref("v_mqsad_pk_u16_u8", M64, 0, 5), 5)                                          # all masked

    def test_edge_values(self):
        for op in OPS:
            for x in X64:
                for y in Y32:
                    acc = ACC[(x ^ y) % len(ACC)]
                    with self.subTest(op=op, x=hex(x), y=hex(y), acc=hex(acc)):
                        self.assertEqual(run_op(op, x, y, acc), ref(op, x, y, acc))

    def test_seeded(self):
        rng = random.Random(0x95AD)
        for _ in range(30):
            op = rng.choice(OPS)
            x, y, acc = rng.getrandbits(64), rng.getrandbits(32), rng.getrandbits(64)
            if op == "v_mqsad_pk_u16_u8" and rng.random() < 0.5:
                y &= rng.choice([0xFF00FF00, 0x00FFFF00, 0xFFFFFF00])
            with self.subTest(op=op, x=hex(x), y=hex(y), acc=hex(acc)):
                self.assertEqual(run_op(op, x, y, acc), ref(op, x, y, acc))

    def test_null_pixel_source_and_null_accumulator(self) -> None:
        x, y, acc = 0x0807060504030201, 0x04000201, 0x0004000300020001
        setup = [f"v_mov_b32 v2, {x & 0xffffffff:#x}", f"v_mov_b32 v3, {x >> 32:#x}", f"v_mov_b32 v4, {y:#x}",
                 f"v_mov_b32 v6, {acc & 0xffffffff:#x}", f"v_mov_b32 v7, {acc >> 32:#x}"]
        for op in OPS:
            for source, accumulator, expected in (("null", "v[6:7]", ref(op, 0, y, acc)),
                                                   ("v[2:3]", "null", ref(op, x, y, 0))):
                with self.subTest(op=op, source=source, accumulator=accumulator):
                    self.assertEqual(run_lines(setup + [f"{op} v[0:1], {source}, v4, {accumulator}"]), expected)

    def test_sgpr_pair_and_inline_accumulator(self):
        got = run_lines(["s_mov_b32 s2, 0x04030201", "s_mov_b32 s3, 0x08070605", "v_mov_b32 v4, 0x04030201",
                         "v_qsad_pk_u16_u8 v[0:1], s[2:3], v4, 1"])
        self.assertEqual(got, ref("v_qsad_pk_u16_u8", 0x0807060504030201, 0x04030201, 1))

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_qsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7] clamp"])


if __name__ == "__main__":
    unittest.main()
