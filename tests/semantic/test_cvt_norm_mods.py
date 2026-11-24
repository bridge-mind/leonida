"""Semantic interpreter and SPIR-V validation tests for RDNA2 output modifiers on
v_cvt_norm_{i16,u16}_f16 and v_cvt_pknorm_{i16,u16}_f32.

Covers native RDNA2 (gfx1036, wave32 & wave64) measured hardware no-op rules:
- v_cvt_norm_i16_f16 / v_cvt_norm_u16_f16: VOP3 clamp, mul:2, mul:4, div:2 (and combinations)
  and SDWA clamp are hardware no-ops while still preserving the unwritten destination half;
  unsupported modifiers and invalid form/modifier pairs raise Unsupported.
- v_cvt_pknorm_i16_f32 / v_cvt_pknorm_u16_f32: VOP3 clamp is a hardware no-op; mul:*/div:*/op_sel
  raise Unsupported.
"""
from __future__ import annotations

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
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cvt_norm_mods_probe as cnp  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M16 = 0xFFFF
M32 = 0xFFFFFFFF
SEED_DST = 0xCAFE1234
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_norm_mods.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_norm_mods.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str], export_tail: list[str] | None = None) -> str:
    if _TMP_DIR is None:
        setUpModule()
    if export_tail is None:
        export_tail = [
            "v_and_b32 v5, 0xffff, v0",
            "v_lshrrev_b32 v6, 16, v0",
            "v_or_b32 v5, 0x40000000, v5",
            "v_or_b32 v6, 0x40000000, v6",
            "exp mrt0 v5, v6, v5, v5 done vm",
        ]
    asm_text = "; test_cvt_norm_mods\n" + "".join(f"\t{ln}\n" for ln in list(lines) + export_tail) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_v0(lines: list[str]) -> tuple[str, int]:
    spv = translate_ps(lines)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert (lo >> 16) == 0x4000 and (hi >> 16) == 0x4000, (hex(lo), hex(hi))
    return spv, (lo & M16) | ((hi & M16) << 16)


def validate_spirv(spv_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


class CvtNormOutputModsTests(unittest.TestCase):
    def test_vop3_output_modifiers_match_plain_across_edges_and_ties(self):
        edges = (
            0x0000, 0x8000,           # +0, -0
            0x0001, 0x8001, 0x03FF,   # subnormals
            0x37FF, 0x3800, 0x3801,   # +0.5 single-rounding ties
            0xB7FF, 0xB800, 0xB801,   # -0.5 single-rounding ties
            0x3C00, 0xBC00,           # +-1.0
            0x4000, 0xC000,           # +-2.0
            0x7C00, 0xFC00,           # +-inf
            0x7E00, 0x7C01, 0xFE00,   # qNaN / sNaN
        )
        for op in cnp.NORM_F16_OPS:
            snorm = op == "v_cvt_norm_i16_f16"
            for mod in ("clamp", "mul:2", "mul:4", "div:2", "clamp mul:2", "clamp div:2"):
                for h in edges:
                    want = (SEED_DST & 0xFFFF0000) | cnp.expected_norm_f16(h, snorm)
                    with self.subTest(op=op, mod=mod, h=hex(h)):
                        _, got = run_v0([
                            f"v_mov_b32 v1, {(0xDEAD0000 | h):#x}",
                            f"v_mov_b32 v0, {SEED_DST:#x}",
                            f"{op}_e64 v0, v1 {mod}",
                        ])
                        self.assertEqual(got, want)

    def test_vop3_abs_neg_and_opsel_with_output_modifiers(self):
        # v1 = {hi=+0.75 (0x3A00), lo=-0.5 (0xB800)}
        packed = 0x3A00B800
        cases = (
            ("v_cvt_norm_i16_f16_e64 v0, |v1| clamp", (SEED_DST & 0xFFFF0000) | cnp.expected_norm_f16(0x3800, True)),
            ("v_cvt_norm_i16_f16_e64 v0, -v1 mul:2", (SEED_DST & 0xFFFF0000) | cnp.expected_norm_f16(0x3800, True)),
            ("v_cvt_norm_u16_f16_e64 v0, -|v1| div:2", (SEED_DST & 0xFFFF0000) | cnp.expected_norm_f16(0xB800, False)),
            ("v_cvt_norm_i16_f16_e64 v0, v1 op_sel:[1,0] clamp", (SEED_DST & 0xFFFF0000) | cnp.expected_norm_f16(0x3A00, True)),
            ("v_cvt_norm_u16_f16_e64 v0, |v1| op_sel:[0,1] mul:2", (cnp.expected_norm_f16(0x3800, False) << 16) | (SEED_DST & M16)),
            ("v_cvt_norm_i16_f16_e64 v0, -v1 op_sel:[1,1] div:2", (cnp.expected_norm_f16(0xBA00, True) << 16) | (SEED_DST & M16)),
        )
        for inst, want in cases:
            with self.subTest(inst=inst):
                _, got = run_v0([f"v_mov_b32 v1, {packed:#x}", f"v_mov_b32 v0, {SEED_DST:#x}", inst])
                self.assertEqual(got, want)

    def test_sdwa_clamp_across_word_and_dst_unused_forms(self):
        for op in cnp.NORM_F16_OPS:
            want_words = cnp.expected_sdwa_words(op)
            for idx, (label, seed_v0, src_v1, _, src_expr, tail) in enumerate(cnp.SDWA_CASES):
                with self.subTest(op=op, case=label):
                    _, got_plain = run_v0([
                        f"v_mov_b32 v0, {seed_v0:#x}",
                        f"v_mov_b32 v1, {src_v1:#x}",
                        f"{op}_sdwa v0, {src_expr} {tail}",
                    ])
                    _, got_clamp = run_v0([
                        f"v_mov_b32 v0, {seed_v0:#x}",
                        f"v_mov_b32 v1, {src_v1:#x}",
                        f"{op}_sdwa v0, {src_expr} clamp {tail}",
                    ])
                    self.assertEqual(got_plain, want_words[idx])
                    self.assertEqual(got_clamp, want_words[idx])

    def test_pknorm_clamp_matches_plain_across_edges_nearties_and_aliasing(self):
        pairs = (
            (0x00000000, 0x80000000),   # +0, -0
            (0x00000001, 0x807FFFFF),   # subnormals
            (0x38400180, 0x3C014103),   # single-rounding near-ties (i16)
            (0x37C000C0, 0x3B028083),   # single-rounding near-ties (u16)
            (0x3F000000, 0xBF000000),   # +-0.5
            (0x3F800000, 0xBF800000),   # +-1.0
            (0x40000000, 0xC0000000),   # +-2.0 (clamped to +-1.0)
            (0x7F800000, 0xFF800000),   # +-inf
            (0x7FC00000, 0xFF800001),   # qNaN / sNaN
        )
        for op in cnp.PKNORM_F32_OPS:
            for a, b in pairs:
                want = cnp.expected_pknorm_f32(op, a, b)
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    _, got = run_v0([
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        "v_mov_b32 v0, 0xdeadbeef",
                        f"{op} v0, v1, v2 clamp",
                    ])
                    self.assertEqual(got, want)
            # Source modifiers + destination aliasing (v0 == src0) with clamp
            want_alias = cnp.expected_pknorm_f32(op, 0xBF000000, 0x3F000000)
            _, got_alias = run_v0([
                "v_mov_b32 v0, 0x3f000000",   # +0.5 -> negated to -0.5
                "v_mov_b32 v1, 0xbf000000",   # -0.5 -> abs to +0.5
                f"{op} v0, -v0, |v1| clamp",
            ])
            self.assertEqual(got_alias, want_alias)

    def test_inactive_exec_lane_preserved(self):
        for inst in (
            "v_cvt_norm_i16_f16_e64 v0, v1 clamp",
            "v_cvt_norm_u16_f16_e64 v0, v1 mul:2",
            "v_cvt_norm_i16_f16_sdwa v0, v1 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
            "v_cvt_pknorm_i16_f32 v0, v1, v2 clamp",
            "v_cvt_pknorm_u16_f32 v0, v1, v2 clamp",
        ):
            with self.subTest(inst=inst):
                _, got = run_v0([
                    "v_mov_b32 v0, 0xdeadbeef",
                    "v_mov_b32 v1, 0x38003c00",
                    "v_mov_b32 v2, 0x3f000000",
                    "s_mov_b32 exec_lo, 0",
                    inst,
                    "s_mov_b32 exec_lo, 1",
                ])
                self.assertEqual(got, 0xDEADBEEF)

    def test_spirv_val_combined_shader(self):
        spv = translate_ps([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x3800b800",
            "v_mov_b32 v2, 0x3f000000",
            "v_mov_b32 v3, 0xbf000000",
            "v_cvt_norm_i16_f16_e64 v0, v1 clamp",
            "v_cvt_norm_i16_f16_e64 v0, v1 mul:2",
            "v_cvt_norm_u16_f16_e64 v0, |v1| div:2",
            "v_cvt_norm_u16_f16_e64 v0, v1 clamp mul:4",
            "v_cvt_norm_i16_f16_sdwa v0, v1 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
            "v_cvt_pknorm_i16_f32 v0, v2, v3 clamp",
            "v_cvt_pknorm_u16_f32 v0, -v3, |v2| clamp",
        ])
        validate_spirv(spv)

    def test_rejected_invalid_modifiers_and_forms(self):
        bad_lines = (
            "v_cvt_norm_i16_f16_e64 v0, v1 mul:8",
            "v_cvt_norm_u16_f16_e64 v0, v1 div:4",
            "v_cvt_norm_i16_f16_sdwa v0, v1 mul:2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
            "v_cvt_norm_u16_f16_sdwa v0, v1 op_sel:[1,0] dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0",
            "v_cvt_norm_i16_f16_e32 v0, v1 clamp",
            "v_cvt_norm_u16_f16_dpp v0, v1 clamp quad_perm:[0,1,2,3]",
            "v_cvt_pknorm_i16_f32 v0, v1, v2 mul:2",
            "v_cvt_pknorm_u16_f32 v0, v1, v2 div:2",
            "v_cvt_pknorm_i16_f32 v0, v1, v2 op_sel:[1,0,0]",
        )
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_ps([
                        "v_mov_b32 v1, 0x3800",
                        "v_mov_b32 v2, 0x3f000000",
                        line,
                    ])


if __name__ == "__main__":
    unittest.main()
