"""Semantic interpreter tests for RDNA2 32-bit scalar inverted logic ops.

Covers s_not_b32, s_andn2_b32, s_orn2_b32, s_nand_b32, s_nor_b32 and s_xnor_b32,
verifying both the 32-bit result D and the SCC flag (int(D != 0)). Each case
translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported words with a
plain-Python reference.
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x12345678, 0xA5A55A5A]

BINARY_OPS = {
    "s_andn2_b32": lambda a, b: (a & (~b & MASK32)) & MASK32,
    "s_orn2_b32": lambda a, b: (a | (~b & MASK32)) & MASK32,
    "s_nand_b32": lambda a, b: (~(a & b)) & MASK32,
    "s_nor_b32": lambda a, b: (~(a | b)) & MASK32,
    "s_xnor_b32": lambda a, b: (~(a ^ b)) & MASK32,
}

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_scalar_inv_logic.s")
    with open(os.path.join(_TMP_DIR.name, "ps_scalar_inv_logic.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines) -> str:
    """Translate synthetic RDNA2 lines into SPIR-V assembly text."""
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_scalar_alu_inv_logic\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines) -> tuple:
    """Run the lines, export v0 and v1, return those two words."""
    spv = translate_lines(lines)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class ScalarInvLogicTests(unittest.TestCase):
    def test_s_not_b32_and_scc(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                want = (~x) & MASK32
                got = run_lines([
                    f"s_mov_b32 s2, {x:#x}",
                    "s_not_b32 s0, s2",
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(got, (want, int(want != 0)))

    def test_binary_inverted_logic_and_scc(self):
        for op, ref_fn in BINARY_OPS.items():
            for a in VALUES:
                for b in VALUES:
                    with self.subTest(op=op, a=hex(a), b=hex(b)):
                        want = ref_fn(a, b)
                        got = run_lines([
                            f"s_mov_b32 s2, {a:#x}",
                            f"s_mov_b32 s3, {b:#x}",
                            f"{op} s0, s2, s3",
                            "s_cselect_b32 s1, 1, 0",
                            "v_mov_b32 v0, s0",
                            "v_mov_b32 v1, s1",
                        ])
                        self.assertEqual(got, (want, int(want != 0)))

    def test_zero_result_clears_scc(self):
        zero_cases = [
            ("s_not_b32 s0, s2", 0xFFFFFFFF, 0),
            ("s_andn2_b32 s0, s2, s3", 0x12345678, 0x12345678),
            ("s_orn2_b32 s0, s2, s3", 0, 0xFFFFFFFF),
            ("s_nand_b32 s0, s2, s3", 0xFFFFFFFF, 0xFFFFFFFF),
            ("s_nor_b32 s0, s2, s3", 0xFFFFFFFF, 0),
            ("s_xnor_b32 s0, s2, s3", 0xA5A55A5A, 0xA5A55A5A ^ MASK32),
        ]
        for instr, a, b in zero_cases:
            with self.subTest(instr=instr, a=hex(a), b=hex(b)):
                got = run_lines([
                    "s_cmp_eq_u32 1, 1",  # pre-set SCC=1 to ensure op clears it
                    f"s_mov_b32 s2, {a:#x}",
                    f"s_mov_b32 s3, {b:#x}",
                    instr,
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ])
                self.assertEqual(got, (0, 0))

    def test_asymmetry_andn2_orn2(self):
        pairs = [
            (0x12345678, 0xA5A55A5A),
            (0x80000000, 0x7FFFFFFF),
            (0xFFFFFFFF, 0),
        ]
        for op in ("s_andn2_b32", "s_orn2_b32"):
            ref_fn = BINARY_OPS[op]
            for a, b in pairs:
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    res_ab = run_lines([
                        f"s_mov_b32 s2, {a:#x}",
                        f"s_mov_b32 s3, {b:#x}",
                        f"{op} s0, s2, s3",
                        "s_cselect_b32 s1, 1, 0",
                        "v_mov_b32 v0, s0",
                        "v_mov_b32 v1, s1",
                    ])
                    res_ba = run_lines([
                        f"s_mov_b32 s2, {b:#x}",
                        f"s_mov_b32 s3, {a:#x}",
                        f"{op} s0, s2, s3",
                        "s_cselect_b32 s1, 1, 0",
                        "v_mov_b32 v0, s0",
                        "v_mov_b32 v1, s1",
                    ])
                    want_ab = ref_fn(a, b)
                    want_ba = ref_fn(b, a)
                    self.assertNotEqual(want_ab, want_ba)
                    self.assertEqual(res_ab, (want_ab, int(want_ab != 0)))
                    self.assertEqual(res_ba, (want_ba, int(want_ba != 0)))
                    self.assertNotEqual(res_ab[0], res_ba[0])

    def test_negative_controls_catch_spirv_regressions(self):
        # 1) Sabotage logical operator in s_andn2_b32 (OpBitwiseAnd -> OpBitwiseOr)
        spv_logic = translate_lines([
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0xa5a55a5a",
            "s_andn2_b32 s0, s2, s3",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ])
        self.assertIn("OpBitwiseAnd %uint", spv_logic)
        bad_logic = spv_logic.replace("OpBitwiseAnd %uint", "OpBitwiseOr %uint", 1)
        want = BINARY_OPS["s_andn2_b32"](0x12345678, 0xA5A55A5A)
        got_bad_logic = tuple(si.run_fragment(bad_logic)["%out_mrt0"][:2])
        self.assertNotEqual(got_bad_logic, (want, int(want != 0)))

        # 2) Sabotage SCC nonzero check in s_not_b32 (OpINotEqual -> OpIEqual)
        spv_scc = translate_lines([
            "s_mov_b32 s2, 0xffffffff",
            "s_not_b32 s0, s2",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ])
        self.assertIn("OpINotEqual %bool", spv_scc)
        bad_scc = spv_scc.replace("OpINotEqual %bool", "OpIEqual %bool", 1)
        got_bad_scc = tuple(si.run_fragment(bad_scc)["%out_mrt0"][:2])
        self.assertNotEqual(got_bad_scc, (0, 0))


if __name__ == "__main__":
    unittest.main()
