"""Translator-driven semantic tests for v_cmp_class_f32, v_cmp_class_f64, and v_cmpx_class_*.

Translates hand-written RDNA2 listings through tools/rdna2spirv.py, validates with spirv-as/spirv-val when on PATH,
and executes them in spirv_mini_interp:
  - exhaustive 10-bit IEEE 754 class matrix for v_cmp_class_f32 and v_cmp_class_f64
    (bit 0 sNaN, 1 qNaN, 2 -Inf, 3 -Normal, 4 -Subnormal, 5 -0.0, 6 +0.0, 7 +Subnormal, 8 +Normal, 9 +Inf);
  - composite masks (NaN, Inf, Zero, Subnormal, Normal, All 0x3FF, None 0x000, high bits 0xFC00 ignored);
  - VOP3 source modifiers (-src0, |src0|, -|src0|) on both f32 and f64;
  - v_cmpx_class_f32_e32 and v_cmpx_class_f64_e32 updating EXEC per lane in a 32-lane compute wave.

    python -I tests/semantic/test_valu_cmp_class.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")

F32_CLASSES = [
    ("snan",      0x7F800001, 0),
    ("qnan",      0x7FC00000, 1),
    ("neg_inf",   0xFF800000, 2),
    ("neg_norm",  0xBF800000, 3),
    ("neg_den",   0x80000001, 4),
    ("neg_zero",  0x80000000, 5),
    ("pos_zero",  0x00000000, 6),
    ("pos_den",   0x00000001, 7),
    ("pos_norm",  0x3F800000, 8),
    ("pos_inf",   0x7F800000, 9),
]

F64_CLASSES = [
    ("snan",      0x7FF0000000000001, 0),
    ("qnan",      0x7FF8000000000000, 1),
    ("neg_inf",   0xFFF0000000000000, 2),
    ("neg_norm",  0xBFF0000000000000, 3),
    ("neg_den",   0x8000000000000001, 4),
    ("neg_zero",  0x8000000000000000, 5),
    ("pos_zero",  0x0000000000000000, 6),
    ("pos_den",   0x0000000000000001, 7),
    ("pos_norm",  0x3FF0000000000000, 8),
    ("pos_inf",   0x7FF0000000000000, 9),
]


def _validate_spv(spvasm):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, "out.spv")
        r = subprocess.run([spirv_as, "--target-env", "vulkan1.2", "-o", spv, "-"],
                           input=spvasm, text=True, capture_output=True)
        if r.returncode:
            raise AssertionError(f"spirv-as failed:\n{r.stderr}")
        r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"spirv-val failed:\n{r.stderr}")


def _w32_fssm():
    if SHADERS not in sys.path:
        sys.path.insert(0, SHADERS)
    from compute_fixture import compute_header
    return bytes(compute_header(os.path.join(SHADERS, "cs_w32_mask_imm.asm")))


def _run_cs_w32(name, lines, n_words=32):
    spvasm = support.translate_lines(name, lines, fssm=_w32_fssm())
    _validate_spv(spvasm)
    mem = {4 * i: 0 for i in range(n_words)}
    si.run_workgroup(
        spvasm, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
        memory=mem,
    )
    return [mem[4 * i] for i in range(n_words)]


class ValuCmpClassTests(unittest.TestCase):
    def test_f32_all_ten_ieee_classes_exact(self):
        # Pack all 10 single-bit class probes into a 10-bit bitmask per input value in one shader batch.
        for tag, bits, expected_bit in F32_CLASSES:
            with self.subTest(f32=tag):
                lines = [f"v_mov_b32_e32 v1, {bits:#x}", "v_mov_b32_e32 v4, 0"]
                for b in range(10):
                    lines += [
                        f"v_cmp_class_f32_e64 vcc_lo, v1, {1 << b:#x}",
                        "v_cndmask_b32_e64 v3, 0, 1, vcc_lo",
                        f"v_lshlrev_b32_e32 v3, {b}, v3",
                        "v_or_b32_e32 v4, v4, v3",
                    ]
                lines += [
                    "v_mov_b32_e32 v2, 0",
                    "v_cmpx_eq_u32_e32 0, v0",
                    "buffer_store_dword v4, v2, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                got = _run_cs_w32(f"cs_cmp_class_f32_{tag}", lines, n_words=1)[0]
                self.assertEqual(got, 1 << expected_bit)

    def test_f64_all_ten_ieee_classes_exact(self):
        for tag, bits, expected_bit in F64_CLASSES:
            with self.subTest(f64=tag):
                lo, hi = bits & 0xFFFFFFFF, (bits >> 32) & 0xFFFFFFFF
                lines = [
                    f"v_mov_b32_e32 v2, {lo:#x}",
                    f"v_mov_b32_e32 v3, {hi:#x}",
                    "v_mov_b32_e32 v4, 0",
                ]
                for b in range(10):
                    lines += [
                        f"v_cmp_class_f64_e64 vcc_lo, v[2:3], {1 << b:#x}",
                        "v_cndmask_b32_e64 v6, 0, 1, vcc_lo",
                        f"v_lshlrev_b32_e32 v6, {b}, v6",
                        "v_or_b32_e32 v4, v4, v6",
                    ]
                lines += [
                    "v_mov_b32_e32 v5, 0",
                    "v_cmpx_eq_u32_e32 0, v0",
                    "buffer_store_dword v4, v5, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                got = _run_cs_w32(f"cs_cmp_class_f64_{tag}", lines, n_words=1)[0]
                self.assertEqual(got, 1 << expected_bit)

    def test_composite_masks_e32_and_ignored_high_mask_bits(self):
        lines = [
            "v_mov_b32_e32 v1, 0x7FC00000",  # qNaN (bit 1)
            "v_mov_b32_e32 v2, 0xFF800000",  # -Inf (bit 2)
            "v_mov_b32_e32 v3, 0x80000000",  # -0.0 (bit 5)
            "v_mov_b32_e32 v10, 0x00000003",  # NaN mask
            "v_cmp_class_f32_e32 vcc_lo, v1, v10",
            "v_cndmask_b32_e64 v20, 0, 1, vcc_lo",
            "v_mov_b32_e32 v11, 0x00000204",  # +/-Inf mask
            "v_cmp_class_f32_e32 vcc_lo, v2, v11",
            "v_cndmask_b32_e64 v21, 0, 1, vcc_lo",
            "v_mov_b32_e32 v12, 0x00000060",  # +/-0 mask
            "v_cmp_class_f32_e32 vcc_lo, v3, v12",
            "v_cndmask_b32_e64 v22, 0, 1, vcc_lo",
            # Only bits >= 10 set -> must return 0 for every class
            "v_mov_b32_e32 v13, 0xFFFFFC00",
            "v_cmp_class_f32_e32 vcc_lo, v3, v13",
            "v_cndmask_b32_e64 v23, 0, 1, vcc_lo",
            "v_mov_b32_e32 v5, 0",
            "v_cmpx_eq_u32_e32 0, v0",
            "buffer_store_dwordx4 v[20:23], v5, s[0:3], 0 offen",
            "s_endpgm",
        ]
        self.assertEqual(_run_cs_w32("cs_cmp_class_composite", lines, n_words=4), [1, 1, 1, 0])

    def test_vop3_abs_and_neg_modifiers_f32_and_f64(self):
        lines = [
            "v_mov_b32_e32 v1, 0x3F800000",  # +1.0
            "v_mov_b32_e32 v2, 0xFF800000",  # -Inf
            "v_mov_b32_e32 v3, 0x00000000",  # +0.0
            "v_mov_b32_e32 v6, 0x00000000",
            "v_mov_b32_e32 v7, 0xFFF00000",  # f64 -Inf
            # -(+1.0) -> neg_normal (bit 3 = 0x008)
            "v_cmp_class_f32_e64 vcc_lo, -v1, 0x008",
            "v_cndmask_b32_e64 v20, 0, 1, vcc_lo",
            # |-Inf| -> pos_inf (bit 9 = 0x200)
            "v_cmp_class_f32_e64 vcc_lo, |v2|, 0x200",
            "v_cndmask_b32_e64 v21, 0, 1, vcc_lo",
            # -|+0.0| -> neg_zero (bit 5 = 0x020)
            "v_cmp_class_f32_e64 vcc_lo, -|v3|, 0x020",
            "v_cndmask_b32_e64 v22, 0, 1, vcc_lo",
            # f64 |-Inf| -> pos_inf (bit 9 = 0x200)
            "v_cmp_class_f64_e64 vcc_lo, |v[6:7]|, 0x200",
            "v_cndmask_b32_e64 v23, 0, 1, vcc_lo",
            "v_mov_b32_e32 v5, 0",
            "v_cmpx_eq_u32_e32 0, v0",
            "buffer_store_dwordx4 v[20:23], v5, s[0:3], 0 offen",
            "s_endpgm",
        ]
        self.assertEqual(_run_cs_w32("cs_cmp_class_mods", lines, n_words=4), [1, 1, 1, 1])

    def test_cmpx_class_f32_and_f64_update_exec_per_lane(self):
        # Even lanes hold +Inf, odd lanes hold +0.0 -> v_cmpx_class_* with pos_inf (0x200) keeps even lanes active.
        lines_f32 = [
            "v_and_b32_e32 v1, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v1",
            "v_cndmask_b32_e64 v2, 0, 0x7F800000, vcc_lo",
            "v_mov_b32_e32 v3, 0x1111",
            "v_mov_b32_e32 v4, 0x200",
            "v_cmpx_class_f32_e32 v2, v4",
            "v_mov_b32_e32 v3, 0x2222",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
            "s_endpgm",
        ]
        want = [0x2222 if (i % 2 == 0) else 0x1111 for i in range(32)]
        self.assertEqual(_run_cs_w32("cs_cmpx_class_f32", lines_f32, n_words=32), want)

        lines_f64 = [
            "v_and_b32_e32 v1, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v1",
            "v_mov_b32_e32 v2, 0",
            "v_cndmask_b32_e64 v3, 0, 0x7FF00000, vcc_lo",
            "v_mov_b32_e32 v4, 0x3333",
            "v_mov_b32_e32 v6, 0x200",
            "v_cmpx_class_f64_e32 v[2:3], v6",
            "v_mov_b32_e32 v4, 0x4444",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
            "s_endpgm",
        ]
        want64 = [0x4444 if (i % 2 == 0) else 0x3333 for i in range(32)]
        self.assertEqual(_run_cs_w32("cs_cmpx_class_f64", lines_f64, n_words=32), want64)


if __name__ == "__main__":
    unittest.main()
