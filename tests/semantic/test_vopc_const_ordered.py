"""Semantic interpreter tests for the constant and ordered/unordered RDNA2 VOPC compares.

  v_cmp_f_{i32,u32}, v_cmp_f_{f32,f16}:  always false
  v_cmp_t_{i32,u32}, v_cmp_tru_{f32,f16}: always true
  v_cmp_o_{f32,f16}: ordered, true when neither source is NaN
  v_cmp_u_{f32,f16}: unordered, true when either source is NaN
The result goes to VCC (VOPC), to an SGPR mask (VOP3) or, for v_cmpx, to EXEC. Each case translates a synthetic pixel
shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares lane 0's bit with a
plain-Python reference. The bit leaves through exp as 0x40000000 or 0x40000001 (always a normal float).
"""
import math
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
F32 = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
       0x00000001]
F16 = [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x0001]


def is_nan32(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def is_nan16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x3FF) != 0


def ref(cond: str, nan_x: bool, nan_y: bool) -> int:
    return int({"f": False, "t": True, "tru": True, "o": not (nan_x or nan_y), "u": nan_x or nan_y}[cond])


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_vopc_const.s")
    with open(os.path.join(_TMP_DIR.name, "ps_vopc_const.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_bit(x: int, y: int, line: str, mask: str = "vcc_lo", preset: int = 0, *,
            active: bool = True, restore_exec: bool = False) -> int | None:
    """v1 = x, v2 = y, the mask preset to all ones or zero, then ``line``; return lane 0's bit of ``mask``."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", f"s_mov_b32 vcc_lo, {-1 if preset else 0}",
             f"s_mov_b32 s10, {-1 if preset else 0}"]
    if not active:
        lines.append("s_mov_b32 exec_lo, 0")
    lines.append(line)
    if restore_exec:
        lines.append("s_mov_b32 exec_lo, -1")
    lines.append(f"v_cndmask_b32 v0, 0x40000000, 0x40000001, {mask}")
    asm = "; test_vopc_const\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv).get("%out_mrt0")
    if w is None:
        return None                       # the lane was switched off (v_cmpx with a false result)
    assert w[0] >> 1 == 0x20000000, hex(w[0])
    return w[0] & 1


class ConstOrderedCompareTests(unittest.TestCase):
    def test_reference(self):
        self.assertTrue(is_nan32(0x7F800001) and not is_nan32(0x7F800000))
        self.assertTrue(is_nan16(0x7C01) and not is_nan16(0x7C00))
        self.assertEqual(ref("o", False, True), 0)
        self.assertEqual(ref("u", False, True), 1)

    def test_float_compares(self):
        for ty, values, nan in (("f32", F32, is_nan32), ("f16", F16, is_nan16)):
            for cond in ("f", "tru", "o", "u"):
                for x in values:
                    for y in values[::3]:
                        with self.subTest(op=f"v_cmp_{cond}_{ty}", x=hex(x), y=hex(y)):
                            want = ref(cond, nan(x), nan(y))
                            got = run_bit(x, y, f"v_cmp_{cond}_{ty} vcc_lo, v1, v2", preset=1 - want)
                            self.assertEqual(got, want)

    def test_integer_constant_compares(self):
        for ty in ("u32", "i32"):
            for cond in ("f", "t"):
                for x, y in ((0, 0), (1, 2), (0xFFFFFFFF, 0)):
                    with self.subTest(op=f"v_cmp_{cond}_{ty}", x=x, y=y):
                        want = ref(cond, False, False)
                        self.assertEqual(run_bit(x, y, f"v_cmp_{cond}_{ty} vcc_lo, v1, v2", preset=1 - want), want)

    def test_64bit_constant_predicates(self) -> None:
        with open(os.path.join(HERE, "fixtures", "vopc_constant_64.asm"), encoding="ascii") as fixture:
            instructions = [line.strip() for line in fixture if line.startswith("v_cmp")]
        self.assertEqual(len(instructions), 12)
        for line in instructions:
            cmpx = line.startswith("v_cmpx_")
            want = "_f_" not in line
            mask = "s10" if " s10," in line else "vcc_lo"
            for active in (False, True):
                with self.subTest(line=line, active=active):
                    # v1 is the high word of the first source: a NaN for f64 cases.
                    if cmpx:
                        self.assertEqual(run_bit(0x7ff80000, 1, line, preset=1, active=active),
                                         1 if active and want else None)
                    else:
                        self.assertEqual(run_bit(0x7ff80000, 1, line, mask=mask, preset=1 - int(active and want),
                                                 active=active, restore_exec=True), int(active and want))
            if cmpx or mask == "s10":
                with self.subTest(line=line, destination="VCC unchanged"):
                    preset = int(not want)
                    self.assertEqual(run_bit(0x7ff80000, 1, line, preset=preset, restore_exec=True), preset)

    def test_vop3_form_writes_the_named_mask(self):
        self.assertEqual(run_bit(0x7FC00000, 0x3F800000, "v_cmp_u_f32_e64 s10, v1, v2", mask="s10"), 1)
        self.assertEqual(run_bit(0x7FC00000, 0x3F800000, "v_cmp_o_f32_e64 s10, v1, v2", mask="s10", preset=1), 0)
        self.assertEqual(run_bit(5, 6, "v_cmp_t_u32_e64 s10, v1, v2", mask="s10"), 1)
        # VCC is left alone when the VOP3 form names another mask.
        self.assertEqual(run_bit(5, 6, "v_cmp_t_u32_e64 s10, v1, v2", mask="vcc_lo"), 0)

    def test_cmpx_writes_exec(self):
        # v_cmpx: EXEC &= result. A true result keeps the lane (it exports); a false one switches it off.
        self.assertEqual(run_bit(0x7FC00000, 0, "v_cmpx_u_f32 v1, v2", preset=1), 1)
        self.assertIsNone(run_bit(0x3F800000, 0, "v_cmpx_u_f32 v1, v2", preset=1))
        self.assertEqual(run_bit(0, 0, "v_cmpx_t_i32 v1, v2", preset=1), 1)
        self.assertIsNone(run_bit(0, 0, "v_cmpx_f_u32 v1, v2", preset=1))


if __name__ == "__main__":
    unittest.main()
