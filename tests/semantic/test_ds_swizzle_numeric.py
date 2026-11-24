"""Semantic tests for ds_swizzle_b32 with raw numeric offset:N and omitted offset (= 0).

RDNA2 ISA DS_SWIZZLE_B32 encodes the swizzle in a 16-bit offset (verified across all 65,536 offsets on gfx1036):
  - 0x8000..0xBFFF (bits 15:14 = 10, QDMode): quad-permute using bits [7:0] (two bits per lane 0..3; bits [13:8]
    ignored; llvm-mc -mcpu=gfx1013 disassembles non-zero bits [13:8] as decimal offset:N, e.g. 0x81e4 -> 33252).
  - 0x0000..0x7FFF (bit 15 = 0, Bitmask mode): within each 32-lane half, src = ((lane & and_mask) | or_mask) ^ xor_mask
    with and_mask = offset[4:0], or_mask = offset[9:5], xor_mask = offset[14:10].
  - 0xC000..0xFFFF (bits 15:14 = 11): separate hardware mode (rotations/interleavings, not quad permute); refused.
  - omitted offset defaults to 0 (broadcast lane 0 / lane 32 in wave64); llvm-mc omits offset:0 on disassembly.

    python -I tests/semantic/test_ds_swizzle_numeric.py -v
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
from support import rdna2spirv  # noqa: E402

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")


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


def _fssm(wave):
    if SHADERS not in sys.path:
        sys.path.insert(0, SHADERS)
    from compute_fixture import compute_header
    asm_name = "cs_w32_mask_imm.asm" if wave == 32 else "cs_wm_bfm64.asm"
    return bytes(compute_header(os.path.join(SHADERS, asm_name)))


def _run_swizzle(name, swizzle_line, wave=32):
    lines = [
        "v_add_nc_u32_e32 v1, 100, v0",
        swizzle_line,
        "v_lshlrev_b32_e32 v3, 2, v0",
        "buffer_store_dword v2, v3, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spvasm = support.translate_lines(name, lines, fssm=_fssm(wave))
    _validate_spv(spvasm)
    mem = {4 * i: 0xEEEEEEEE for i in range(wave)}
    si.run_workgroup(
        spvasm, wave, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
        memory=mem,
    )
    return [mem[4 * i] for i in range(wave)]


def _expected_numeric(off, wave):
    out = []
    for lane in range(wave):
        if off & 0x8000:
            sel = (off >> (2 * (lane & 3))) & 3
            src = (lane & ~3) | sel
        else:
            and_m = off & 0x1F
            or_m = (off >> 5) & 0x1F
            xor_m = (off >> 10) & 0x1F
            src = (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m)
        out.append(100 + src)
    return out


class DsSwizzleNumericTests(unittest.TestCase):
    def test_qdmode_numeric_decimal_and_hex_wave32_and_wave64(self):
        cases = [
            # 0x801f = QUAD_PERM(3, 3, 1, 0)
            ("qd_801f", "ds_swizzle_b32 v2, v1 offset:0x801f", 0x801F),
            # 0x81e4 = decimal 33252 (non-zero bit 8, what llvm-mc prints as numeric): QUAD_PERM(0, 1, 2, 3)
            ("qd_33252", "ds_swizzle_b32 v2, v1 offset:33252", 0x81E4),
            # 0xbf1b = QUAD_PERM(3, 2, 1, 0) with all ignored bits 13:8 set and bit 14 clear
            ("qd_bf1b", "ds_swizzle_b32 v2, v1 offset:0xbf1b", 0xBF1B),
        ]
        for tag, line, off in cases:
            for wave in (32, 64):
                with self.subTest(case=tag, wave=wave):
                    self.assertEqual(_run_swizzle(f"cs_{tag}_w{wave}", line, wave=wave),
                                     _expected_numeric(off, wave))

    def test_bitmask_numeric_and_omitted_offset_wave32_and_wave64(self):
        cases = [
            # Omitted offset == 0 (broadcast lane 0 / lane 32 in wave64)
            ("omitted", "ds_swizzle_b32 v2, v1", 0x0000),
            # Explicit offset:0
            ("zero", "ds_swizzle_b32 v2, v1 offset:0", 0x0000),
            # 0x041f = SWAP(1) (and=31, or=0, xor=1)
            ("swap1", "ds_swizzle_b32 v2, v1 offset:0x041f", 0x041F),
            # 0x401f = SWAP(16) (and=31, or=0, xor=16)
            ("swap16", "ds_swizzle_b32 v2, v1 offset:0x401f", 0x401F),
            # 0x7c1f = REVERSE(32) (and=31, or=0, xor=31)
            ("rev32", "ds_swizzle_b32 v2, v1 offset:0x7c1f", 0x7C1F),
            # 0x00a0 = BROADCAST(32, 5) (and=0, or=5, xor=0)
            ("bcast5", "ds_swizzle_b32 v2, v1 offset:0x00a0", 0x00A0),
            # Mixed and/or/xor: and=0x18 (keep bits 4,3), or=0x03 (set bits 1,0), xor=0x05 (flip bits 2,0)
            ("mixed", "ds_swizzle_b32 v2, v1 offset:0x1478", 0x1478),
        ]
        for tag, line, off in cases:
            for wave in (32, 64):
                with self.subTest(case=tag, wave=wave):
                    self.assertEqual(_run_swizzle(f"cs_{tag}_w{wave}", line, wave=wave),
                                     _expected_numeric(off, wave))

    def test_numeric_matches_symbolic_forms(self):
        pairs = [
            ("ds_swizzle_b32 v2, v1 offset:0x801f", 'ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,3,3,1,0)'),
            ("ds_swizzle_b32 v2, v1 offset:0x041f", 'ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)'),
            ("ds_swizzle_b32 v2, v1 offset:0x7c1f", 'ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,32)'),
            ("ds_swizzle_b32 v2, v1 offset:0x00a0", 'ds_swizzle_b32 v2, v1 offset:swizzle(BROADCAST,32,5)'),
            ("ds_swizzle_b32 v2, v1", 'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"00000")'),
        ]
        for idx, (num_line, sym_line) in enumerate(pairs):
            with self.subTest(pair=idx):
                self.assertEqual(
                    _run_swizzle(f"cs_num_{idx}", num_line, wave=64),
                    _run_swizzle(f"cs_sym_{idx}", sym_line, wave=64),
                )

    def test_wave64_emulate_validates_with_spirv_val(self):
        old_emulate = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            for tag, swz in (("qd", "ds_swizzle_b32 v2, v1 offset:33252"),
                             ("bm", "ds_swizzle_b32 v2, v1 offset:0x1478"),
                             ("zero", "ds_swizzle_b32 v2, v1")):
                lines = [
                    "v_add_nc_u32_e32 v1, 100, v0",
                    swz,
                    "v_lshlrev_b32_e32 v3, 2, v0",
                    "buffer_store_dword v2, v3, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                spvasm = support.translate_lines(f"cs_emul_{tag}", lines, fssm=_fssm(64))
                _validate_spv(spvasm)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emulate

    def test_invalid_offsets_and_modifiers_stay_refused(self):
        for bad in (
            "ds_swizzle_b32 v2, v1 offset:0xc000",
            "ds_swizzle_b32 v2, v1 offset:0xff1b",
            "ds_swizzle_b32 v2, v1 offset:0xffff",
            "ds_swizzle_b32 v2, v1 offset:65536",
            "ds_swizzle_b32 v2, v1 offset:-1",
            "ds_swizzle_b32 v2, v1 offset:notanumber",
            "ds_swizzle_b32 v2, v1 offset:swizzle(UNKNOWN,1)",
            "ds_swizzle_b32 v2, v1 offset:0x801f gds",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("cs_swz_bad", [bad, "s_endpgm"], fssm=_fssm(32))


if __name__ == "__main__":
    unittest.main()
