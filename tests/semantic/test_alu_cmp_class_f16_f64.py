"""Semantic interpreter + spirv-val tests for RDNA2 v_cmp(x)_class_f16 and v_cmp(x)_class_f64.

Tests all 10 IEEE-754 class mask bits (0: sNaN, 1: qNaN, 2: -inf, 3: -normal, 4: -denormal,
5: -0, 6: +0, 7: +denormal, 8: +normal, 9: +inf), combined masks, op_sel[0] on f16, abs/neg
modifiers (including preserving sNaN without quieting), 32-bit literal upper-word rule on f64,
v_cmpx EXEC narrowing, absence of OpCapability Float64, and spirv-val validation.
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
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

# One representative bit pattern for each class bit 0..9
F16_CLASSES = [
    (0, 0x7C01, "pos_snan"),
    (0, 0xFC01, "neg_snan"),
    (1, 0x7E00, "pos_qnan"),
    (1, 0xFE01, "neg_qnan"),
    (2, 0xFC00, "neg_inf"),
    (3, 0xBC00, "neg_normal"),
    (4, 0x8001, "neg_denorm"),
    (5, 0x8000, "neg_zero"),
    (6, 0x0000, "pos_zero"),
    (7, 0x0001, "pos_denorm"),
    (8, 0x3C00, "pos_normal"),
    (9, 0x7C00, "pos_inf"),
]

F64_CLASSES = [
    (0, 0x7FF0000000000001, "pos_snan_lo"),
    (0, 0xFFF4000000000000, "neg_snan_hi"),
    (1, 0x7FF8000000000000, "pos_qnan"),
    (1, 0xFFF8000000000001, "neg_qnan"),
    (2, 0xFFF0000000000000, "neg_inf"),
    (3, 0xBFF0000000000000, "neg_normal"),
    (4, 0x8000000000000001, "neg_denorm_lo"),
    (5, 0x8000000000000000, "neg_zero"),
    (6, 0x0000000000000000, "pos_zero"),
    (7, 0x0000000000000001, "pos_denorm_lo"),
    (8, 0x3FF0000000000000, "pos_normal"),
    (9, 0x7FF0000000000000, "pos_inf"),
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cmp_class.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cmp_class.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_cmp_class\n" + "".join(f"\t{ln}\n" for ln in lines)
    asm += "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_v0(lines) -> int:
    spv = translate_lines(lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


def validate_spirv(spv: str):
    if not SPIRV_AS or not SPIRV_VAL:
        return
    spv_bin = os.path.join(_TMP_DIR.name, "cmp_class.spv")
    p1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", spv_bin],
                        input=spv.encode("utf-8"), capture_output=True)
    assert p1.returncode == 0, p1.stderr.decode("utf-8", "replace")
    p2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_bin], capture_output=True)
    assert p2.returncode == 0, p2.stderr.decode("utf-8", "replace")


class CmpClassF16Tests(unittest.TestCase):
    def test_all_10_classes(self):
        for cls_bit, val, name in F16_CLASSES:
            exact_mask = 1 << cls_bit
            inv_mask = 0x3FF ^ exact_mask
            with self.subTest(case=name):
                hit = run_v0([
                    f"v_mov_b32 v1, {0xBEEF0000 | val:#x}",
                    f"v_cmp_class_f16_e32 vcc_lo, v1, {exact_mask:#x}",
                    "v_cndmask_b32_e32 v0, 0x3f800000, 0x40000000, vcc_lo",
                ])
                miss = run_v0([
                    f"v_mov_b32 v1, {0xBEEF0000 | val:#x}",
                    f"v_cmp_class_f16_e32 vcc_lo, v1, {inv_mask:#x}",
                    "v_cndmask_b32_e32 v0, 0x3f800000, 0x40000000, vcc_lo",
                ])
                self.assertEqual(hit, 0x40000000)
                self.assertEqual(miss, 0x3F800000)

    def test_opsel_mods_cmpx_and_spirv_val(self):
        # op_sel:[1,0] selects high half (-inf = 0xfc00); abs makes it +inf (bit 9 = 0x200)
        got = run_v0([
            "v_mov_b32 v1, 0xfc003c00",
            "v_cmp_class_f16_e64 s0, |v1|, 0x200 op_sel:[1,0]",
            "v_cndmask_b32_e64 v0, 0x3f800000, 0x40000000, s0",
        ])
        self.assertEqual(got, 0x40000000)
        # neg on +0.0 -> -0.0 (bit 5 = 0x20), inline constant 0.5 -> +normal (bit 8 = 0x100)
        got_neg = run_v0([
            "v_mov_b32 v1, 0x00000000",
            "v_cmp_class_f16_e64 s0, -v1, 0x20",
            "v_cndmask_b32_e64 v0, 0x3f800000, 0x40000000, s0",
        ])
        self.assertEqual(got_neg, 0x40000000)
        # v_cmpx_class_f16 narrows EXEC (false leaves v0 at 0x3f800000, skips 0x40000000)
        spv = translate_lines([
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x7c01",
            "v_cmpx_class_f16_e32 v1, 0x2",   # sNaN vs qNaN-only mask -> false
            "v_mov_b32 v0, 0x40000000",
            "s_mov_b32 exec_lo, -1",
        ])
        validate_spirv(spv)
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], 0x3F800000)

    def test_f16_float_const_mask_uses_src_h(self):
        # Float inline constants on v_cmp(x)_class_f16's mask encode as binary16 over a zero high half
        # (e.g. 0.5=0x3800, 1.0=0x3C00, -2.0=0xC000, 0.15915494=0x3118), not binary32.
        # _sdwa: BYTE_3(0.5)=0 (no match), BYTE_1(0.5)=0x38 (bit 3=-normal matches -1.0), WORD_1(1.0)=0, BYTE_3(-2.0)=0.
        for mask, sel, val, want in (
            ("0.5", "BYTE_3", 0xBC00, 0x3F800000),
            ("0.5", "BYTE_1", 0xBC00, 0x40000000),
            ("1.0", "WORD_1", 0x4000, 0x3F800000),
            ("-2.0", "BYTE_3", 0x0001, 0x3F800000),
        ):
            with self.subTest(mask=mask, sel=sel, val=hex(val)):
                got = run_v0([
                    f"v_mov_b32 v1, {val:#x}",
                    f"v_cmp_class_f16_sdwa vcc_lo, v1, {mask} src0_sel:WORD_0 src1_sel:{sel}",
                    "v_cndmask_b32_e32 v0, 0x3f800000, 0x40000000, vcc_lo",
                ])
                self.assertEqual(got, want)

        # _e64 with 0.15915494 (1/(2*pi)): f16=0x3118 (mask & 0x3FF = 0x118 -> bits 3,4,8; bit 0 sNaN clear),
        # whereas f32=0x3E22F983 has mask & 0x3FF = 0x183 (would falsely match sNaN bit 0).
        got_snan = run_v0([
            "v_mov_b32 v1, 0x7c01",
            "v_cmp_class_f16_e64 s0, v1, 0.15915494",
            "v_cndmask_b32_e64 v0, 0x3f800000, 0x40000000, s0",
        ])
        self.assertEqual(got_snan, 0x3F800000)
        got_norm = run_v0([
            "v_mov_b32 v1, 0xbc00",
            "v_cmp_class_f16_e64 s0, v1, 0.15915494",
            "v_cndmask_b32_e64 v0, 0x3f800000, 0x40000000, s0",
        ])
        self.assertEqual(got_norm, 0x40000000)

        # v_cmpx_class_f16_e64 with 0.15915494 kills sNaN lane (bit 0 clear in 0x3118).
        spv_cmpx = translate_lines([
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x7c01",
            "v_cmpx_class_f16_e64 v1, 0.15915494",
            "v_mov_b32 v0, 0x40000000",
            "s_mov_b32 exec_lo, -1",
        ])
        validate_spirv(spv_cmpx)
        self.assertEqual(si.run_fragment(spv_cmpx)["%out_mrt0"][0], 0x3F800000)


class CmpClassF64Tests(unittest.TestCase):
    def test_all_10_classes_and_no_float64(self):
        for cls_bit, val, name in F64_CLASSES:
            exact_mask = 1 << cls_bit
            inv_mask = 0x3FF ^ exact_mask
            lo, hi = val & 0xFFFFFFFF, (val >> 32) & 0xFFFFFFFF
            with self.subTest(case=name):
                spv_hit = translate_lines([
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    f"v_cmp_class_f64_e32 vcc_lo, v[2:3], {exact_mask:#x}",
                    "v_cndmask_b32_e32 v0, 0x3f800000, 0x40000000, vcc_lo",
                ])
                self.assertNotIn("OpCapability Float64", spv_hit)
                self.assertEqual(si.run_fragment(spv_hit)["%out_mrt0"][0], 0x40000000)
                miss = run_v0([
                    f"v_mov_b32 v2, {lo:#x}",
                    f"v_mov_b32 v3, {hi:#x}",
                    f"v_cmp_class_f64_e32 vcc_lo, v[2:3], {inv_mask:#x}",
                    "v_cndmask_b32_e32 v0, 0x3f800000, 0x40000000, vcc_lo",
                ])
                self.assertEqual(miss, 0x3F800000)

    def test_mods_literal_cmpx_rejected_and_spirv_val(self):
        # 32-bit literal 0xfff00000 -> -inf (upper word, lo=0), abs -> +inf (bit 9 = 0x200)
        spv = translate_lines([
            "v_mov_b32 v0, 0x3f800000",
            "v_cmp_class_f64_e64 s0, |0xfff00000|, 0x200",
            "v_cndmask_b32_e64 v0, 0x3f800000, 0x40000000, s0",
            "v_cmpx_class_f64_e64 -1.0, 0x8",  # -normal -> true, keeps lane active
            "s_mov_b32 exec_lo, -1",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        validate_spirv(spv)
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], 0x40000000)

        for bad in (
            "v_cmp_class_f16_e64 s0, v1, 1 clamp",
            "v_cmp_class_f16_e64 s0, v1, 1 op_sel:[0,1]",
            "v_cmp_class_f64_e64 s0, v[2:3], 1 clamp",
            "v_cmp_class_f64_e64 s0, v[2:3], 1 op_sel:[1,0]",
        ):
            with self.subTest(bad=bad), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines([bad])


if __name__ == "__main__":
    unittest.main()
