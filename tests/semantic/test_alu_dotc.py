"""Semantic interpreter and spirv-val tests for RDNA2 v_dot2c_i32_i16 and v_dot8c_i32_i4.

v_dot2c_i32_i16 computes D = S0.i16[0]*S1.i16[0] + S0.i16[1]*S1.i16[1] + D (VOP2/VOP3).
v_dot8c_i32_i4 computes D = sum_{k=0..7}(S0.i4[k]*S1.i4[k]) + D (VOP2/VOP3).
Both wrap at 32 bits by default and saturate to [INT32_MIN, INT32_MAX] with VOP3 clamp.
"""
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
M32 = 0xFFFFFFFF
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def sext(val: int, width: int) -> int:
    mask = (1 << width) - 1
    v = val & mask
    return v - (1 << width) if v & (1 << (width - 1)) else v


def ref_dotc(x: int, y: int, acc: int, width: int, clamp: bool) -> int:
    total = sext(acc, 32)
    for off in range(0, 32, width):
        total += sext(x >> off, width) * sext(y >> off, width)
    if clamp:
        total = max(INT32_MIN, min(INT32_MAX, total))
    return total & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_dotc.s")
    with open(os.path.join(_TMP_DIR.name, "ps_dotc.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_dotc\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_dotc(insn: str, x: int, y: int, acc: int) -> int:
    spv = translate_lines([
        f"v_mov_b32 v1, {x & M32:#x}",
        f"v_mov_b32 v2, {y & M32:#x}",
        f"v_mov_b32 v0, {acc & M32:#x}",
        insn,
    ])
    return si.run_fragment(spv)["%out_mrt0"][0]


class DotcTests(unittest.TestCase):
    def test_dot2c_i32_i16_edges_wrap_and_clamp(self):
        cases = (
            (0x00020003, 0x00040005, 10),              # 2*4 + 3*5 + 10 = 33
            (0xFFFE0003, 0x0004FFFB, 0xFFFFFFF0),      # (-2)*4 + 3*(-5) - 16 = -39
            (0x80008000, 0x80008000, 0),               # (-32768)^2 * 2 = 2^31 (wraps to INT32_MIN, clamps to INT32_MAX)
            (0x80008000, 0x80008000, 0x7FFFFFFF),      # large positive overflow
            (0x80008000, 0x7FFF7FFF, 0x80000000),      # large negative overflow
        )
        for x, y, acc in cases:
            with self.subTest(x=hex(x), y=hex(y), acc=hex(acc)):
                self.assertEqual(run_dotc("v_dot2c_i32_i16_e32 v0, v1, v2", x, y, acc), ref_dotc(x, y, acc, 16, False))
                self.assertEqual(run_dotc("v_dot2c_i32_i16_e64 v0, v1, v2 clamp", x, y, acc), ref_dotc(x, y, acc, 16, True))

    def test_dot8c_i32_i4_edges_wrap_and_clamp(self):
        cases = (
            (0x12345678, 0x87654321, 100),
            (0x88888888, 0x88888888, 0x7FFFFFF0),      # 8 * (-8)*(-8) = +512 -> positive overflow
            (0x88888888, 0x77777777, 0x80000010),      # 8 * (-8)*(7) = -448 -> negative overflow
            (0xFFFFFFFF, 0x11111111, 0),               # 8 * (-1)*(1) = -8
        )
        for x, y, acc in cases:
            with self.subTest(x=hex(x), y=hex(y), acc=hex(acc)):
                self.assertEqual(run_dotc("v_dot8c_i32_i4_e32 v0, v1, v2", x, y, acc), ref_dotc(x, y, acc, 4, False))
                self.assertEqual(run_dotc("v_dot8c_i32_i4_e64 v0, v1, v2 clamp", x, y, acc), ref_dotc(x, y, acc, 4, True))

    def test_seeded_sweep(self):
        rng = random.Random(0xD07C)
        for _ in range(30):
            x, y, acc = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            for op, width in (("v_dot2c_i32_i16", 16), ("v_dot8c_i32_i4", 4)):
                self.assertEqual(run_dotc(f"{op}_e32 v0, v1, v2", x, y, acc), ref_dotc(x, y, acc, width, False))
                self.assertEqual(run_dotc(f"{op}_e64 v0, v1, v2 clamp", x, y, acc), ref_dotc(x, y, acc, width, True))

    def test_aliasing_sgpr_and_spirv_val(self):
        spv = translate_lines([
            "s_mov_b32 s4, 0x00020003",
            "v_mov_b32 v0, 0x00040005",
            "v_dot2c_i32_i16_e32 v0, s4, v0",
            "v_dot8c_i32_i4_e64 v0, 0x11111111, v0 clamp",
        ])
        validate_spirv(spv)
        # First op: s4=(2,3), v0=(4,5), acc=0x00040005 -> 2*4 + 3*5 + 0x00040005
        mid = ref_dotc(0x00020003, 0x00040005, 0x00040005, 16, False)
        want = ref_dotc(0x11111111, mid, mid, 4, True)
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], want)

    def test_rejects_unsupported_modifiers(self):
        for line in (
            "v_dot2c_i32_i16_e64 v0, v1, v2 op_sel:[0,0,0]",
            "v_dot8c_i32_i4_e64 v0, -v1, v2",
            "v_dot2c_i32_i16_e64 v0, |v1|, v2",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines([line])


if __name__ == "__main__":
    unittest.main()
