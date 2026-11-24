"""Value tests for wave64 s_*_saveexec_b32 and s_andn{1,2}_wrexec_b32.

On RDNA2 in wave64 mode, the 32-bit SOP1 forms operate strictly on EXEC_LO (lanes 0..31):
  - s_<op>_saveexec_b32 D, S0: D = EXEC_LO; EXEC_LO = f(S0, EXEC_LO); SCC = (EXEC_LO != 0)
  - s_andn{1,2}_wrexec_b32 D, S0: EXEC_LO = f(S0, EXEC_LO); D = EXEC_LO; SCC = (EXEC_LO != 0)
  - EXEC_HI (lanes 32..63) is untouched.
  - D receives a MASK_LO value (lanes 0..31 only), so whole-wave consumers like s_bcnt1_i32_b32
    count only the low half.
  - Cross-half operands (exec_hi / vcc_hi / MASK_HI sources, or exec_hi / vcc_hi destinations)
    transfer the 32-bit word across halves.

Verified through rdna2spirv -> spirv_mini_interp.run_workgroup (64 lanes) and spirv-as/spirv-val
(native wave64 and --wave64-emulate). Standard library only; no GPU or game data.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import lane_masks as lm  # noqa: E402

rdna2spirv = support.rdna2spirv
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

FSSM_64 = bytes(0x138) + struct.pack("<4I", 0x88, 64, 1, 1)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
M32 = 0xFFFFFFFF

EXEC_FUNCS = {
    "and": lambda s, e: (s & e) & M32,
    "or": lambda s, e: (s | e) & M32,
    "xor": lambda s, e: (s ^ e) & M32,
    "andn1": lambda s, e: ((~s) & e) & M32,
    "andn2": lambda s, e: (s & (~e)) & M32,
    "orn1": lambda s, e: ((~s) | e) & M32,
    "orn2": lambda s, e: (s | (~e)) & M32,
    "nand": lambda s, e: (~(s & e)) & M32,
    "nor": lambda s, e: (~(s | e)) & M32,
    "xnor": lambda s, e: (~(s ^ e)) & M32,
}


def validate_spirv(spv):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm = os.path.join(tmp, "s.spvasm")
        bin_ = os.path.join(tmp, "s.spv")
        with open(asm, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.3", asm, "-o", bin_],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", bin_],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def run_w64(lines, *, validate_emulate=True):
    """Translate as native wave64, validate with spirv-val (and optionally --wave64-emulate),
    and run all 64 lanes in spirv_mini_interp.run_workgroup."""
    prev = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = False
        spv = support.translate_lines("cs_w64_saveexec_b32", lines + ["s_endpgm"], fssm=FSSM_64)
        assert rdna2spirv.translate.last_gen.wave64
        validate_spirv(spv)
        if validate_emulate:
            rdna2spirv.WAVE64_EMULATE = True
            spv_emu = support.translate_lines("cs_w64_saveexec_b32", lines + ["s_endpgm"], fssm=FSSM_64)
            assert rdna2spirv.translate.last_gen.wave64
            validate_spirv(spv_emu)
    finally:
        rdna2spirv.WAVE64_EMULATE = prev
    return si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


def read_exec_and_state(lines, init_lo, init_hi):
    """Setup EXEC = (init_hi << 32) | init_lo, run `lines`, then record:
    out[0] = resulting EXEC_LO, out[1] = resulting EXEC_HI, out[2] = SCC (0/1),
    out[3] = popcount(s8) via s_bcnt1_i32_b32, out[4] = s8 restored to EXEC_LO popcount."""
    prog = [
        "s_mov_b64 exec, -1",                 # ensure wave64 classification
        f"s_mov_b32 exec_lo, {init_lo:#x}",
        f"s_mov_b32 exec_hi, {init_hi:#x}",
        *lines,
        # Capture SCC before touching EXEC
        "s_cselect_b32 s10, 1, 0",
        # Capture current 64-bit EXEC into per-lane flag, then ballot via s_bcnt1 / per-lane stores
        "v_mov_b32_e32 v10, 0",
        "s_mov_b64 s[12:13], exec",
        "s_mov_b64 exec, -1",
        "v_cndmask_b32_e64 v10, 0, 1, s[12:13]",
        # Store each lane's exec bit at byte 64 + 4*lane
        "v_lshlrev_b32_e32 v11, 2, v0",
        "v_add_nc_u32_e32 v11, 64, v11",
        "buffer_store_dword v10, v11, s[0:3], 0 offen",
        # Compute s_bcnt1_i32_b32 on s8 (the saved/written destination) to check MASK_LO tagging
        "s_bcnt1_i32_b32 s11, s8",
        # Store SCC (s10) at byte 0, bcnt(s8) (s11) at byte 4 from lane 0
        "v_cmpx_eq_u32_e32 0, v0",
        "v_mov_b32_e32 v1, s10",
        "v_mov_b32_e32 v2, s11",
        "v_mov_b32_e32 v12, 0", "buffer_store_dword v1, v12, s[0:3], 0 offen",
        "v_mov_b32_e32 v12, 4", "buffer_store_dword v2, v12, s[0:3], 0 offen",
    ]
    mem = run_w64(prog)
    exec_lo = sum(((mem.get(64 + 4 * i, 0) & 1) << i) for i in range(32))
    exec_hi = sum(((mem.get(64 + 4 * (32 + i), 0) & 1) << i) for i in range(32))
    return exec_lo, exec_hi, mem.get(0), mem.get(4)


class Wave64SaveexecB32Tests(unittest.TestCase):
    def test_all_ten_saveexec_b32_preserve_exec_hi_and_scc_low_half(self):
        # Crucially: init_lo = 0x0000000F, s0 = 0x000000F0, init_hi = 0x000000FF.
        # For s_and_saveexec_b32: new EXEC_LO = 0 while EXEC_HI stays 0x000000FF (non-zero!) -> SCC MUST be 0!
        # Previously on main, set_exec(r) clobbered EXEC_HI to 0xF0 and scc_from_mask voted 1 over all 64 lanes.
        init_lo, init_hi, s0 = 0x0000000F, 0x000000FF, 0x000000F0
        for kind, fn in EXEC_FUNCS.items():
            op = f"s_{kind}_saveexec_b32"
            with self.subTest(op=op):
                got_lo, got_hi, got_scc, got_bcnt = read_exec_and_state(
                    [f"{op} s8, {s0:#x}"], init_lo, init_hi
                )
                want_lo = fn(s0, init_lo)
                self.assertEqual(got_lo, want_lo)
                self.assertEqual(got_hi, init_hi)                         # EXEC_HI untouched
                self.assertEqual(got_scc, 1 if want_lo != 0 else 0)       # SCC = (EXEC_LO != 0) only
                self.assertEqual(got_bcnt, bin(init_lo).count("1"))       # s8 holds MASK_LO (4, not 4+8=12)

    def test_wrexec_b32_preserves_exec_hi_and_scc_low_half(self):
        # Case where new EXEC_LO is 0 while EXEC_HI is non-zero, and case where new EXEC_LO is non-zero.
        for op, kind in (("s_andn2_wrexec_b32", "andn2"), ("s_andn1_wrexec_b32", "andn1")):
            for init_lo, init_hi, s0 in (
                (0x000000FF, 0xA5A50000, 0x0000000F),   # andn2 -> 0 (while EXEC_HI != 0); andn1 -> 0xF0
                (0x0000000F, 0x000000FF, 0x000000FF),   # andn2 -> 0xF0; andn1 -> 0 (while EXEC_HI != 0)
            ):
                with self.subTest(op=op, init_lo=hex(init_lo), s0=hex(s0)):
                    got_lo, got_hi, got_scc, got_bcnt = read_exec_and_state(
                        [f"{op} s8, {s0:#x}"], init_lo, init_hi
                    )
                    want_lo = EXEC_FUNCS[kind](s0, init_lo)
                    self.assertEqual(got_lo, want_lo)
                    self.assertEqual(got_hi, init_hi)
                    self.assertEqual(got_scc, 1 if want_lo != 0 else 0)
                    self.assertEqual(got_bcnt, bin(want_lo).count("1"))

    def test_cross_half_hi_source_and_exec_hi_dest(self):
        # 1) Source is exec_hi (lanes 32..63 = 0x000000F0) applied to EXEC_LO (0x000000FF):
        #    new EXEC_LO = 0xF0 & 0xFF = 0xF0, EXEC_HI stays 0x000000F0.
        got_lo, got_hi, got_scc, got_bcnt = read_exec_and_state(
            ["s_and_saveexec_b32 s8, exec_hi"], init_lo=0x000000FF, init_hi=0x000000F0
        )
        self.assertEqual((got_lo, got_hi, got_scc, got_bcnt), (0x000000F0, 0x000000F0, 1, 8))

        # 2) Destination is exec_hi: old EXEC_LO (0x00000033) copied into EXEC_HI,
        #    while EXEC_LO becomes (0x0F & 0x33) = 0x03.
        got_lo2, got_hi2, got_scc2, _ = read_exec_and_state(
            ["s_mov_b32 s8, 0", "s_and_saveexec_b32 exec_hi, 0x0000000f"],
            init_lo=0x00000033, init_hi=0xFFFF0000,
        )
        self.assertEqual((got_lo2, got_hi2, got_scc2), (0x00000003, 0x00000033, 1))

    def test_vcc_lo_dest_keeps_vcc_hi_intact(self):
        # Initialize VCC as a lane mask with vcc_hi = 0x000000AA (distinct from later exec_hi = 0x000000F0).
        # s_and_saveexec_b32 vcc_lo, 0x0f must update only vcc_lo (lanes 0..31) to old exec_lo (0x55) while
        # keeping vcc_hi = 0xAA (old mask_dst overwrote lanes 32..63 of %vcc with exec_hi = 0xF0).
        prog = [
            "s_mov_b64 exec, -1",
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, 0x000000aa",
            "v_cmp_lt_u32_e32 vcc, v0, 64",       # vcc_lo = 0, vcc_hi = 0xAA (real lane mask)
            "s_mov_b32 exec_lo, 0x00000055",
            "s_mov_b32 exec_hi, 0x000000f0",
            "s_and_saveexec_b32 vcc_lo, 0x0000000f",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v10, 0, 1, vcc",
            "v_lshlrev_b32_e32 v11, 2, v0",
            "buffer_store_dword v10, v11, s[0:3], 0 offen",
        ]
        mem = run_w64(prog)
        vcc_lo = sum(((mem.get(4 * i, 0) & 1) << i) for i in range(32))
        vcc_hi = sum(((mem.get(4 * (32 + i), 0) & 1) << i) for i in range(32))
        self.assertEqual(vcc_lo, 0x00000055)
        self.assertEqual(vcc_hi, 0x000000AA)

    def test_lane_masks_tags_b32_saveexec_and_wrexec_as_mask_lo(self):
        s = lambda r, n=1: {"k": "sgpr", "r": r, "n": n}
        use = {"op": "s_nop", "args": []}
        ins = lambda op, *a: {"op": op, "args": list(a)}
        for op in ("s_and_saveexec_b32", "s_andn2_wrexec_b32"):
            masks, _ = lm.analyse([[ins(op, s(4), {"k": "imm", "v": 0xF}), use]], [[]])
            self.assertEqual(masks[id(use)], {4: lm.MASK_LO})
        masks64, _ = lm.analyse([[ins("s_and_saveexec_b64", s(4, 2), {"k": "imm", "v": -1}), use]], [[]])
        self.assertEqual(masks64[id(use)], {4: lm.MASK_LO, 5: lm.MASK_HI})


if __name__ == "__main__":
    unittest.main()
