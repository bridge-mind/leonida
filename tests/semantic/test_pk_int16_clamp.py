"""Semantic SPIR-V and interpreter tests for RDNA2 VOP3P clamp on v_pk_mul_lo_u16 and v_pk_{max,min}_{u16,i16}.

Covers every probe case from tools/pk_int16_clamp_probe.py (overflowing products wrapping modulo 2^16, op_sel and
op_sel_hi permutations, inline/literal constants, SGPR operands, destination aliasing), checks SPIR-V emits no
UMin/SClamp saturation on mul_lo/max/min clamp (unlike v_pk_add_u16 clamp), verifies inactive EXEC preservation and
rejected invalid modifiers, and validates SPIR-V with spirv-as/spirv-val when available. Standard library only.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pk_int16_clamp_probe as probe  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
_TMP = None


def _translate(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_pk_int16_clamp.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    src = os.path.join(_TMP.name, "disasm", "ps_pk_int16_clamp.s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("; ps_pk_int16_clamp\n" + "".join(f"\t{ln}\n" for ln in lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def _run_v2(body_lines):
    lines = list(body_lines) + [
        "v_and_b32 v5, 0xffff, v2",
        "v_lshrrev_b32 v6, 16, v2",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
        "exp mrt0 v5, v6, v5, v5 done vm",
        "s_endpgm",
    ]
    spv = _translate(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16), spv


def _ps_case_lines(op, c, clamp):
    return [ln.replace("v_mov_b32_e32 ", "v_mov_b32 ") for ln in probe.case_instruction(op, c, clamp)]


class PkInt16ClampSemanticTests(unittest.TestCase):
    def test_all_probe_cases_plain_and_clamp(self):
        ctrl_diffs = 0
        for op in probe.OPS:
            for c in probe.CASES:
                want_plain = probe.expected_pk_int16(op, c.src0, c.src1, sel=c.sel, selhi=c.selhi, clamp=False)
                want_clamp = probe.expected_pk_int16(op, c.src0, c.src1, sel=c.sel, selhi=c.selhi, clamp=True)
                with self.subTest(op=op, case=c.name):
                    got_plain, _ = _run_v2(_ps_case_lines(op, c, False))
                    got_clamp, _ = _run_v2(_ps_case_lines(op, c, True))
                    self.assertEqual(got_plain, want_plain)
                    self.assertEqual(got_clamp, want_clamp)
                    if op in probe.TARGET_OPS:
                        self.assertEqual(got_clamp, got_plain)
                    elif got_clamp != got_plain:
                        ctrl_diffs += 1
        self.assertGreater(ctrl_diffs, 0)

    def test_spirv_emits_no_saturation_on_target_ops(self):
        for op in probe.TARGET_OPS:
            with self.subTest(op=op):
                got_plain, spv_plain = _run_v2([
                    "v_mov_b32 v0, 0x02000100",
                    "v_mov_b32 v1, 0x01000200",
                    f"{op} v2, v0, v1",
                ])
                got_clamp, spv_clamp = _run_v2([
                    "v_mov_b32 v0, 0x02000100",
                    "v_mov_b32 v1, 0x01000200",
                    f"{op} v2, v0, v1 clamp",
                ])
                self.assertEqual(
                    got_clamp,
                    probe.expected_pk_int16(op, 0x02000100, 0x01000200, clamp=True),
                )
                self.assertEqual(got_clamp, got_plain)
                self.assertEqual(spv_clamp, spv_plain)
                self.assertNotIn("SClamp", spv_clamp)
                if op != "v_pk_min_u16":
                    self.assertNotIn("UMin", spv_clamp)
                else:
                    self.assertFalse(any("UMin " in ln and "%cffff" in ln for ln in spv_clamp.splitlines()))

        _, ctrl_spv = _run_v2([
            "v_mov_b32 v0, 0xffffffff",
            "v_mov_b32 v1, 0x00020002",
            "v_pk_add_u16 v2, v0, v1 clamp",
        ])
        self.assertTrue(any("UMin " in ln and "%cffff" in ln for ln in ctrl_spv.splitlines()))

    def test_inactive_exec_preserves_dst(self):
        sentinel = 0xA5A55A5A
        for op in probe.TARGET_OPS:
            with self.subTest(op=op):
                got, _ = _run_v2([
                    "v_mov_b32 v0, 0xffffffff",
                    "v_mov_b32 v1, 0x00020003",
                    f"v_mov_b32 v2, {sentinel:#010x}",
                    "s_mov_b32 exec_lo, 0",
                    f"{op} v2, v0, v1 clamp",
                    "s_mov_b32 exec_lo, -1",
                ])
                self.assertEqual(got, sentinel)

    def test_rejected_invalid_modifiers(self):
        for op in probe.TARGET_OPS:
            for bad in (
                f"{op} v2, v0, v1 neg_lo:[1,0]",
                f"{op} v2, v0, v1 neg_hi:[0,1]",
                f"{op} v2, v0, v1 mul:2",
                f"{op} v2, v0, v1 div:2",
                f"{op} v2, v0, v1 op_sel:[0]",
                f"{op} v2, v0, v1 clamp neg_lo:[1,0]",
            ):
                with self.subTest(line=bad):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate(["v_mov_b32 v0, 1", "v_mov_b32 v1, 2", bad, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"])

    def test_spirv_val_vulkan12(self):
        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not on PATH")
        body = [
            "s_mov_b32 s4, 0x0300fffe",
            "v_mov_b32 v0, 0x80007fff",
            "v_mov_b32 v1, 0x02000400",
            "v_pk_mul_lo_u16 v2, v0, v1 clamp",
            "v_pk_mul_lo_u16 v2, v2, -16 op_sel:[1,0] op_sel_hi:[0,1] clamp",
            "v_pk_max_u16 v3, s4, 0x12345678 op_sel:[0,1] op_sel_hi:[1,0] clamp",
            "v_pk_min_u16 v3, v3, v2 clamp",
            "v_pk_max_i16 v4, v2, v3 clamp",
            "v_pk_min_i16 v2, v4, v0 clamp",
            "v_pk_add_u16 v3, v0, v1 clamp",
            "v_xor_b32 v2, v2, v3",
        ]
        _, spv = _run_v2(body)
        with tempfile.TemporaryDirectory() as td:
            spvasm = os.path.join(td, "shader.spvasm")
            spvbin = os.path.join(td, "shader.spv")
            with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin], check=True)
            subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin], check=True)


if __name__ == "__main__":
    unittest.main()
