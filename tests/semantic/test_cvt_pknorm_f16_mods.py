"""Semantic SPIR-V and interpreter tests for RDNA2 v_cvt_pknorm_i16_f16 / u16_f16 modifiers.

Covers every textual probe case from tools/cvt_pknorm_f16_probe.py across plain, VOP3 `clamp` (hardware no-op),
and destination `op_sel:[s0,s1,1]` (16-bit left shift zeroing D[15:0] and overwriting D), checks byte-identical
SPIR-V emission for `clamp`, verifies `OpShiftLeftLogical` emission on dst `op_sel`, checks inactive EXEC lane
preservation, asserts `Unsupported` on rejected modifiers/forms, and validates SPIR-V with spirv-as/spirv-val.
Standard library only; synthetic inputs only.
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

import cvt_pknorm_f16_probe as probe  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
_TMP = None


def _translate(lines):
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_TMP.name, "disasm"))
        with open(os.path.join(_TMP.name, "ps_cvt_pknorm_f16_mods.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    src = os.path.join(_TMP.name, "disasm", "ps_cvt_pknorm_f16_mods.s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("; ps_cvt_pknorm_f16_mods\n" + "".join(f"\t{ln}\n" for ln in lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def _run_v8(body_lines):
    lines = list(body_lines) + [
        "v_and_b32 v5, 0xffff, v8",
        "v_lshrrev_b32 v6, 16, v8",
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


class CvtPknormF16ModsSemanticTests(unittest.TestCase):
    def test_all_probe_cases_plain_and_clamp(self):
        for op in probe.OPS:
            for c in probe.CASES:
                if c.raw_omod is not None:
                    continue
                want_plain = probe.expected_pknorm_f16(
                    op, c.src0, c.src1, sel=c.sel, dhi=bool(c.dhi), clamp=False,
                    src0_abs=c.src0_abs, src0_neg=c.src0_neg, src1_abs=c.src1_abs, src1_neg=c.src1_neg,
                )
                want_clamp = probe.expected_pknorm_f16(
                    op, c.src0, c.src1, sel=c.sel, dhi=bool(c.dhi), clamp=True,
                    src0_abs=c.src0_abs, src0_neg=c.src0_neg, src1_abs=c.src1_abs, src1_neg=c.src1_neg,
                )
                self.assertEqual(want_plain, want_clamp)
                with self.subTest(op=op, case=c.name):
                    got_plain, spv_plain = _run_v8(_ps_case_lines(op, c, False))
                    got_clamp, spv_clamp = _run_v8(_ps_case_lines(op, c, True))
                    self.assertEqual(got_plain, want_plain)
                    self.assertEqual(got_clamp, want_clamp)
                    self.assertEqual(spv_clamp, spv_plain)

    def test_dst_opsel_shift_and_overwrite(self):
        for op in probe.OPS:
            with self.subTest(op=op):
                got_plain, spv_plain = _run_v8([
                    "v_mov_b32 v1, 0x3c003400",
                    "v_mov_b32 v2, 0xbc003801",
                    "v_mov_b32 v8, 0xdeadbeef",
                    f"{op} v8, v1, v2 op_sel:[0,0,0]",
                ])
                got_dhi, spv_dhi = _run_v8([
                    "v_mov_b32 v1, 0x3c003400",
                    "v_mov_b32 v2, 0xbc003801",
                    "v_mov_b32 v8, 0xdeadbeef",
                    f"{op} v8, v1, v2 op_sel:[0,0,1]",
                ])
                self.assertEqual(got_dhi, (got_plain << 16) & 0xFFFFFFFF)
                self.assertEqual(got_dhi & 0xFFFF, 0)
                shift_count_plain = sum(1 for ln in spv_plain.splitlines() if "OpShiftLeftLogical %uint" in ln)
                shift_count_dhi = sum(1 for ln in spv_dhi.splitlines() if "OpShiftLeftLogical %uint" in ln)
                self.assertEqual(shift_count_dhi, shift_count_plain + 1)

    def test_inactive_exec_preserves_dst(self):
        sentinel = 0xA5A55A5A
        for op in probe.OPS:
            for mod in ("clamp", "op_sel:[0,0,1]", "op_sel:[0,0,1] clamp"):
                with self.subTest(op=op, mod=mod):
                    got, _ = _run_v8([
                        "v_mov_b32 v1, 0x3c003400",
                        "v_mov_b32 v2, 0xbc003801",
                        f"v_mov_b32 v8, {sentinel:#010x}",
                        "s_mov_b32 exec_lo, 0",
                        f"{op} v8, v1, v2 {mod}",
                        "s_mov_b32 exec_lo, -1",
                    ])
                    self.assertEqual(got, sentinel)

    def test_rejected_invalid_modifiers(self):
        for op in probe.OPS:
            for bad in (
                f"{op} v8, v1, v2 mul:2",
                f"{op} v8, v1, v2 mul:4",
                f"{op} v8, v1, v2 div:2",
                f"{op} v8, v1, v2 op_sel_hi:[1,1]",
                f"{op} v8, v1, v2 neg_lo:[1,0]",
                f"{op} v8, v1, v2 neg_hi:[0,1]",
                f"{op} v8, v1, v2 op_sel:[0,0,0,1]",
                f"{op} v8, v1, v2 op_sel:[2,0,0]",
                f"{op} v8, v1, v2 op_sel:[]",
                f"{op} v8, v1, v2 op_sel:[0,0,1]bad",
                f"{op} v8, v1, v2 op_sel:[0,0,0] op_sel:[1,0,0]",
                f"{op}_dpp v8, v1, v2 row_mirror",
                f"{op}_sdwa v8, v1, v2",
                f"{op} v8, v1, v2 foo",
            ):
                with self.subTest(line=bad):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate([
                            "v_mov_b32 v1, 0x3c003400",
                            "v_mov_b32 v2, 0xbc003801",
                            bad,
                            "exp mrt0 v8, v8, v8, v8 done vm",
                            "s_endpgm",
                        ])

    def test_spirv_val_vulkan12(self):
        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not on PATH")
        body = [
            "s_mov_b32 s4, 0x3c003400",
            "v_mov_b32 v1, 0x3c003801",
            "v_mov_b32 v2, 0xbc00b801",
            "v_cvt_pknorm_i16_f16 v8, v1, v2 clamp",
            "v_cvt_pknorm_i16_f16 v8, |s4|, -v2 op_sel:[1,0,1] clamp",
            "v_cvt_pknorm_u16_f16 v9, v8, 0.5 op_sel:[0,0,1]",
            "v_cvt_pknorm_u16_f16 v8, -|v1|, v9 op_sel:[0,1,0] clamp",
            "v_xor_b32 v8, v8, v9",
        ]
        _, spv = _run_v8(body)
        with tempfile.TemporaryDirectory() as td:
            spvasm = os.path.join(td, "shader.spvasm")
            spvbin = os.path.join(td, "shader.spv")
            with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin], check=True)
            subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin], check=True)


if __name__ == "__main__":
    unittest.main()
