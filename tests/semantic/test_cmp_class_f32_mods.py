"""Semantic tests for v_cmp_class_f32 / v_cmpx_class_f32 with VOP3 neg/abs modifiers.

Hardware rule (measured natively on AMD RDNA2 gfx1036, wave32 & wave64, IEEE=0/1,
denorm32 keep/flush, 672/672 exact across all 21 values x 4 modifiers x 10 class bits):
  v_cmp_class_f32 and v_cmpx_class_f32 apply VOP3 abs (& 0x7FFFFFFF) then neg (^ 0x80000000)
  as raw sign-bit moves on the 32-bit operand before testing the IEEE-754 class bits.
  A signalling NaN stays signalling (class bit 0, never quieted to bit 1) and subnormals
  stay subnormal (never flushed to zero), regardless of MODE.IEEE or MODE.FP32_DENORM.

Before this fix, Gen.cmp_class loaded src0 through src_f -> modify(), emitting SPIR-V
FAbs / OpFNegate, which SPIR-V leaves free to quiet NaNs and which spirv_mini_interp.py
quieted from sNaN (bit 0) to qNaN (bit 1).

No game files, llvm-mc, or GPU required; spirv-val check runs when spirv-as and spirv-val
are on PATH.
"""
import gc
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
COMPUTE_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CLASS_SNAN = 1 << 0
CLASS_QNAN = 1 << 1
CLASS_NINF = 1 << 2
CLASS_NNORM = 1 << 3
CLASS_NSUB = 1 << 4
CLASS_NZERO = 1 << 5
CLASS_PZERO = 1 << 6
CLASS_PSUB = 1 << 7
CLASS_PNORM = 1 << 8
CLASS_PINF = 1 << 9

# All 21 representative IEEE-754 binary32 values measured natively on gfx1036.
VALUES = [
    ("+sNaN_min",   0x7F800001),
    ("+sNaN_max",   0x7FBFFFFF),
    ("-sNaN_min",   0xFF800001),
    ("-sNaN_mid",   0xFFA00000),
    ("-sNaN_max",   0xFFBFFFFF),
    ("+qNaN_canon", 0x7FC00000),
    ("+qNaN_max",   0x7FFFFFFF),
    ("-qNaN_canon", 0xFFC00000),
    ("-qNaN_max",   0xFFFFFFFF),
    ("-inf",        0xFF800000),
    ("-1.0",        0xBF800000),
    ("-FLT_MIN",    0x80800000),
    ("-denorm_min", 0x80000001),
    ("-denorm_max", 0x807FFFFF),
    ("-0.0",        0x80000000),
    ("+0.0",        0x00000000),
    ("+denorm_min", 0x00000001),
    ("+denorm_max", 0x007FFFFF),
    ("+FLT_MIN",    0x00800000),
    ("+1.0",        0x3F800000),
    ("+inf",        0x7F800000),
]

MODS = [
    ("plain", "v1",    lambda u: u),
    ("neg",   "-v1",   lambda u: u ^ 0x80000000),
    ("abs",   "|v1|",  lambda u: u & 0x7FFFFFFF),
    ("nabs",  "-|v1|", lambda u: (u & 0x7FFFFFFF) ^ 0x80000000),
]


def ref_class_bit(u: int) -> int:
    """Return the single 10-bit class mask bit for raw binary32 word u."""
    sign = (u >> 31) & 1
    absu = u & 0x7FFFFFFF
    exp = absu >> 23
    man = absu & 0x7FFFFF
    if exp == 255:
        if man == 0:
            return CLASS_NINF if sign else CLASS_PINF
        return CLASS_QNAN if man >= 0x400000 else CLASS_SNAN
    if absu == 0:
        return CLASS_NZERO if sign else CLASS_PZERO
    if exp == 0:
        return CLASS_NSUB if sign else CLASS_PSUB
    return CLASS_NNORM if sign else CLASS_PNORM


def translate_listing(body_lines, *, stage="ps", wave=32, group=(1, 1, 1)):
    """Translate a hand-written RDNA2 listing through rdna2spirv.translate."""
    with tempfile.TemporaryDirectory() as td:
        dis = os.path.join(td, "disasm")
        fssm = os.path.join(td, "fssm")
        os.makedirs(dis)
        os.makedirs(fssm)
        name = "ps_test" if stage == "ps" else "cs_test"
        with open(os.path.join(td, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC if stage == "ps" else COMPUTE_DXBC)
        # Minimal FSSM header: flags word at 0xD4 bit 5 (0x20) = wave32, 0 = wave64;
        # compute rsrc2 + group size at 0x138..0x147.
        hdr = bytearray(0x148)
        hdr[0:4] = b"FSSM"
        struct.pack_into("<I", hdr, 0xD4, 0x20 if wave == 32 else 0x00)
        if stage == "cs":
            struct.pack_into("<4I", hdr, 0x138, 0x0088, *group)
        with open(os.path.join(fssm, f"{name}.fssm"), "wb") as f:
            f.write(hdr)
        src_path = os.path.join(dis, f"{name}.s")
        text = f"; synthetic {name}\n" + "\n".join(f"\t{ln}" for ln in body_lines) + "\n"
        with open(src_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src_path)
            gc.collect()
        return spv


def run_ps(body_lines, exports=("v2", "v4", "v5", "v6"), wave=32):
    full = list(body_lines) + [f"exp mrt0 {', '.join(exports)} done vm", "s_endpgm"]
    spv = translate_listing(full, stage="ps", wave=wave)
    out = si.run_fragment(spv)
    return tuple(out["%out_mrt0"]), spv


class CmpClassF32ModsTests(unittest.TestCase):
    def test_snan_modifiers_keep_signalling_class_and_emit_bitwise_ops(self):
        """Regression for @Vonsek's finding: -v, |v|, -|v| on sNaNs keep bit 0 (sNaN) and clear bit 1 (qNaN)."""
        snans = [0x7F800001, 0x7FBFFFFF, 0xFF800001, 0xFFA00000, 0xFFBFFFFF]
        for bits in snans:
            for mname, opnd, _ in MODS:
                for wave in (32, 64):
                    vcc = "vcc_lo" if wave == 32 else "vcc"
                    sreg = "s4" if wave == 32 else "s[4:5]"
                    with self.subTest(bits=f"0x{bits:08x}", mod=mname, wave=wave):
                        lines = [
                            f"v_mov_b32 v1, 0x{bits:08x}",
                            # bit 0 = sNaN (must be 1), bit 1 = qNaN (must be 0)
                            f"v_cmp_class_f32_e64 {vcc}, {opnd}, 1",
                            f"v_cndmask_b32_e64 v2, 0, 1, {vcc}",
                            f"v_cmp_class_f32_e64 {sreg}, {opnd}, 2",
                            f"v_cndmask_b32_e64 v4, 0, 1, {sreg}",
                            # v_cmpx_class_f32_e64 on sNaN (bit 0) must keep lane active and preserve VCC
                            "v_mov_b32 v5, 0",
                            f"v_cmpx_class_f32_e64 {opnd}, 1",
                            "v_mov_b32 v5, 1",
                            f"v_cndmask_b32_e64 v6, 0, 1, {vcc}",
                        ]
                        got, spv = run_ps(lines, wave=wave)
                        self.assertEqual(got, (1, 0, 1, 1))
                        self.assertNotIn("FAbs", spv)
                        self.assertNotIn("OpFNegate", spv)

    def test_all_21_values_4_modifiers_10_classes_cmp_and_cmpx(self):
        """Exhaustive 21 values x 4 modifiers x 10 class bits matching gfx1036 hardware (wave32 and wave64)."""
        for wave in (32, 64):
            vcc = "vcc_lo" if wave == 32 else "vcc"
            exec_save = "s_mov_b32 s4, exec_lo" if wave == 32 else "s_mov_b64 s[4:5], exec"
            exec_rest = "s_mov_b32 exec_lo, s4" if wave == 32 else "s_mov_b64 exec, s[4:5]"
            for vname, bits in VALUES:
                for mname, opnd, fn in MODS:
                    want = ref_class_bit(fn(bits))
                    lines = [
                        f"v_mov_b32 v1, 0x{bits:08x}",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v4, 0",
                    ]
                    for b in range(10):
                        lines += [
                            f"v_cmp_class_f32_e64 {vcc}, {opnd}, {1 << b}",
                            f"v_cndmask_b32_e64 v3, 0, {1 << b}, {vcc}",
                            "v_or_b32 v2, v2, v3",
                            exec_save,
                            f"v_cmpx_class_f32_e64 {opnd}, {1 << b}",
                            f"v_or_b32 v4, v4, {1 << b}",
                            exec_rest,
                        ]
                    lines += ["v_mov_b32 v5, 0", "v_mov_b32 v6, 0"]
                    with self.subTest(wave=wave, val=vname, mod=mname):
                        got, _ = run_ps(lines, wave=wave)
                        self.assertEqual(got[0], want, f"v_cmp_class_f32 {vname} {mname} w{wave}")
                        self.assertEqual(got[1], want, f"v_cmpx_class_f32 {vname} {mname} w{wave}")

    def test_constant_operands_with_neg_abs(self):
        """VOP3 neg/abs on literal/inline constants in v_cmp_class_f32 preserve sNaN and subnormal bits."""
        got, spv = run_ps([
            "v_cmp_class_f32_e64 s4, neg(0x7f800001), 1",   # -sNaN -> bit 0 (sNaN)
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_class_f32_e64 s5, |0xffa00000|, 1",      # |(-sNaN)| -> bit 0 (sNaN)
            "v_cndmask_b32_e64 v4, 0, 1, s5",
            "v_cmp_class_f32_e64 s6, -|0x7fbfffff|, 3",     # -|(+sNaN)| -> bit 0, check bits 0|1
            "v_cndmask_b32_e64 v5, 0, 1, s6",
            "v_cmp_class_f32_e64 s7, -|0x7fbfffff|, 2",     # must NOT match bit 1 (qNaN)
            "v_cndmask_b32_e64 v6, 0, 1, s7",
        ])
        self.assertEqual(got, (1, 1, 1, 0))
        self.assertNotIn("FAbs", spv)
        self.assertNotIn("OpFNegate", spv)

    def test_multilane_cmpx_snan_mods_and_inactive_lanes(self):
        """Workgroup test: v_cmpx_class_f32 with -v/|v|/-|v| updates EXEC per lane and keeps disabled lanes off."""
        # 4 lanes:
        #   lane 0: 0x7f800001 (+sNaN) -> -v is -sNaN (bit 0 = 1) -> active
        #   lane 1: 0x7fc00000 (+qNaN) -> -v is -qNaN (bit 0 = 0) -> deactivated
        #   lane 2: 0xffa00000 (-sNaN) -> |v| is +sNaN (bit 0 = 1), but pre-disabled -> stays off
        #   lane 3: 0xffbfffff (-sNaN) -> -|v| is -sNaN (bit 0 = 1) -> active
        body = [
            "s_mov_b32 s8, 0",
            "s_mov_b32 s9, 0",
            "s_mov_b32 s10, 64",
            "s_mov_b32 s11, 0x31016fac",
            # Load per-lane test value from buffer offset 4*lane
            "v_lshlrev_b32 v10, 2, v0",
            "buffer_load_dword v1, v10, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
            # Pre-disable lane 2 via EXEC
            "v_cmp_ne_u32_e32 vcc_lo, 2, v0",
            "s_and_b32 exec_lo, exec_lo, vcc_lo",
            # Intersect EXEC with sNaN check through -v, |v|, and -|v| sequentially
            "v_cmpx_class_f32_e64 -v1, 1",
            "v_cmpx_class_f32_e64 |v1|, 1",
            "v_cmpx_class_f32_e64 -|v1|, 1",
            # Only lanes 0 and 3 remain active and write 0xCAFE0000 | lane to offset 32 + 4*lane
            "v_or_b32 v2, 0xcafe0000, v0",
            "v_add_nc_u32 v11, 32, v10",
            "buffer_store_dword v2, v11, s[8:11], 0 offen",
            "s_endpgm",
        ]
        spv = translate_listing(body, stage="cs", wave=32, group=(4, 1, 1))
        memory = {
            0: 0x7F800001, 4: 0x7FC00000, 8: 0xFFA00000, 12: 0xFFBFFFFF,
            32: 0xDEAD0000, 36: 0xDEAD0001, 40: 0xDEAD0002, 44: 0xDEAD0003,
        }
        si.run_workgroup(
            spv, 4, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
            push_constants=[0] * 32,
            memory=memory,
        )
        got = (memory[32], memory[36], memory[40], memory[44])
        self.assertEqual(got, (0xCAFE0000, 0xDEAD0001, 0xDEAD0002, 0xCAFE0003))

    def test_spirv_val_accepts_modified_cmp_class(self):
        """Translated SPIR-V with -v, |v|, -|v| passes spirv-as and spirv-val when installed."""
        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not on PATH")
        _, spv = run_ps([
            "v_mov_b32 v1, 0x7f800001",
            "v_cmp_class_f32_e64 vcc_lo, -v1, 1",
            "v_cndmask_b32_e64 v2, 0, 1, vcc_lo",
            "v_cmp_class_f32_e64 s4, |v1|, 1",
            "v_cndmask_b32_e64 v4, 0, 1, s4",
            "v_cmp_class_f32_e64 s5, -|v1|, 1",
            "v_cndmask_b32_e64 v5, 0, 1, s5",
            "v_mov_b32 v6, 0",
            "v_cmpx_class_f32_e64 -|v1|, 1",
            "v_mov_b32 v6, 1",
        ])
        with tempfile.TemporaryDirectory() as td:
            asm_p = os.path.join(td, "m.spvasm")
            bin_p = os.path.join(td, "m.spv")
            with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], check=True)
            subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], check=True)


if __name__ == "__main__":
    unittest.main()
