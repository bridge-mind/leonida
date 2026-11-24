"""Probe gfx1036 silicon across wave32 and wave64 for 16-bit vector compares:
  - v_cmp and v_cmpx (EXEC update and inactive-lane 0 write in sdst).
  - Inline int/float constants and 32-bit literals with op_sel=0/1.
  - SGPR high halves, f16 abs/neg/ordered-vs-unordered NaN.

Usage:
  powershell -File C:\\Leonida-worktrees\\with-gpu-lock.ps1 "cd C:\\Leonida-worktrees\\w4-cmp16-opsel; python scripts/probe_gfx1036_vcmp16.py"
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from support import STUB_DXBC, rdna2spirv
import spirv_mini_interp as si

VK_COMPUTE_TEST = os.path.join(ROOT, "runtime", "build", "vk_compute_test.exe")

# Half-precision bit constants
H_ZERO = 0x0000      # +0.0
H_NZERO = 0x8000     # -0.0
H_HALF = 0x3800      # +0.5
H_ONE = 0x3C00       # +1.0
H_TWO = 0x4000       # +2.0
H_THREE = 0x4200     # +3.0
H_FOUR = 0x4400      # +4.0
H_MONE = 0xBC00      # -1.0
H_MTWO = 0xC000      # -2.0
H_MFOUR = 0xC400     # -4.0
H_INF = 0x7C00       # +inf
H_NINF = 0xFC00      # -inf
H_QNAN = 0x7E00      # qNaN
H_SNAN = 0x7D00      # sNaN

SENTINEL = 0xDEADBEEF

def pack16(lo: int, hi: int) -> int:
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)

def run_on_silicon(name, asm_lines, wave_size=32, user_sgprs=None):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n")
            for ln in asm_lines:
                f.write(f"\t{ln}\n")
            f.write("\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        rsrc2 = 0x8090 if user_sgprs else 0x8088
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", rsrc2, wave_size, 1, 1))
        
        spv_text = rdna2spirv.translate(s_path)
        sa_path = os.path.join(tmp, name + ".spvasm")
        spv_path = os.path.join(tmp, name + ".spv")
        with open(sa_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        rdna2spirv.assemble_spirv(sa_path, spv_path)
        
        dump_path = os.path.join(tmp, "dump.bin")
        env = os.environ.copy()
        env["LEONIDA_VK_DEVICE"] = "radeon"
        env["VKC_DUMP"] = dump_path
        env["VKC_OUT_WORDS"] = str(wave_size)
        if user_sgprs:
            env["VKC_SGPRS"] = ",".join(hex(w) for w in user_sgprs)
        
        extra_args = ["--require-wave64"] if wave_size == 64 else []
        res = subprocess.run([VK_COMPUTE_TEST, spv_path, "1", "none"] + extra_args,
                             env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {name} (rc={res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            return None
        
        with open(dump_path, "rb") as f:
            data = f.read()
        return struct.unpack(f"<{len(data)//4}I", data)[:wave_size]


def test_vcmp_inactive_lanes(wave_size):
    """Verify v_cmp writes 0 for inactive lanes into sdst (wave32 and wave64)."""
    name = f"cs_probe_vcmp_inactive_w{wave_size}"
    # Inactive lanes: odd lanes killed in EXEC before compare.
    # Active lanes: v1 == v2 (true).
    # Then re-enable EXEC so all lanes can store their bit from sdst.
    if wave_size == 32:
        exec_init = "s_mov_b32 exec_lo, 0x55555555" # even lanes active
        cmp_line = "v_cmp_eq_u16_e64 s4, v1, v2 op_sel:[0,0]"
        exec_restore = "s_mov_b32 exec_lo, -1"
        read_bit = [
            "v_lshrrev_b32_e32 v3, v0, s4",
            "v_and_b32_e32 v3, 1, v3",
        ]
    else:
        exec_init = "s_mov_b64 exec, 0x5555555555555555"
        cmp_line = "v_cmp_eq_u16_e64 s[4:5], v1, v2 op_sel:[0,0]"
        exec_restore = "s_mov_b64 exec, -1"
        read_bit = [
            "v_cmp_ge_u32_e64 s6, v0, 32",
            "v_cndmask_b32_e64 v4, s4, s5, s6",
            "v_and_b32_e32 v5, 31, v0",
            "v_lshrrev_b32_e32 v3, v5, v4",
            "v_and_b32_e32 v3, 1, v3",
        ]
    
    body = [
        "v_mov_b32 v1, 0x1234",
        "v_mov_b32 v2, 0x1234",
        exec_init,
        cmp_line,
        exec_restore,
        *read_bit,
        "v_lshlrev_b32_e32 v6, 2, v0",
        "buffer_store_dword v3, v6, s[0:3], 0 offen",
    ]
    out = run_on_silicon(name, body, wave_size=wave_size)
    assert out is not None, f"Silicon run failed for {name}"
    for lane in range(wave_size):
        expected = 1 if (lane % 2 == 0) else 0
        if out[lane] != expected:
            raise AssertionError(f"{name} lane {lane}: got {out[lane]}, expected {expected}")
    print(f"[{name}] PASSED on gfx1036 silicon (inactive lanes set to 0 in sdst)")


def test_vcmpx_exec_update(wave_size):
    """Verify v_cmpx kills inactive/failing lanes in EXEC across wave32 and wave64."""
    name = f"cs_probe_vcmpx_exec_w{wave_size}"
    # Even lanes have v1.hi < v2.lo (1.0 < 2.0 -> True).
    # Odd lanes have v1.hi > v2.lo (4.0 > 2.0 -> False).
    body = [
        # Pre-fill with SENTINEL
        "v_lshlrev_b32_e32 v4, 2, v0",
        f"v_mov_b32 v5, {SENTINEL:#x}",
        "buffer_store_dword v5, v4, s[0:3], 0 offen",
        # Even lanes get v1 = pack16(0, 1.0); odd lanes get v1 = pack16(0, 4.0)
        "v_and_b32_e32 v3, 1, v0",
        f"v_cmp_eq_u32_e64 {'vcc' if wave_size == 64 else 'vcc_lo'}, v3, 0",
        f"v_mov_b32 v6, {pack16(0, H_ONE):#x}",
        f"v_mov_b32 v7, {pack16(0, H_FOUR):#x}",
        f"v_cndmask_b32_e64 v1, v7, v6, {'vcc' if wave_size == 64 else 'vcc_lo'}",
        f"v_mov_b32 v2, {pack16(H_TWO, 0):#x}",
        # v_cmpx with op_sel:[1, 0]: reads hi half of v1, lo half of v2
        "v_cmpx_lt_f16_e64 v1, v2 op_sel:[1,0]",
        # Only surviving lanes write lane | 0x100
        "v_or_b32_e32 v5, 0x100, v0",
        "buffer_store_dword v5, v4, s[0:3], 0 offen",
    ]
    out = run_on_silicon(name, body, wave_size=wave_size)
    assert out is not None, f"Silicon run failed for {name}"
    for lane in range(wave_size):
        expected = (0x100 | lane) if (lane % 2 == 0) else SENTINEL
        if out[lane] != expected:
            raise AssertionError(f"{name} lane {lane}: got {hex(out[lane])}, expected {hex(expected)}")
    print(f"[{name}] PASSED on gfx1036 silicon (EXEC properly updated by v_cmpx)")


def test_constants_and_literals(wave_size=32):
    """Probe inline float/int constants and 32-bit literals with op_sel=0 and 1."""
    name = f"cs_probe_consts_lits_w{wave_size}"
    # Test cases: (description, cmp_line, v1_word, expected_bit)
    tests = [
        # Inline float constants: 1.0 (0x3C00 in lo, 0.0 in hi)
        # v1 = pack16(1.0, 2.0). 1.0 == 1.0 -> 1; 2.0 == 1.0 -> 0.
        ("f16_float_const_opsel0", "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,0]", pack16(H_ONE, H_TWO), 1),
        ("f16_float_const_opsel0_diff", "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[1,0]", pack16(H_ONE, H_TWO), 0),
        # op_sel=1 on 1.0 reads +0.0:
        ("f16_float_const_opsel1_reads_zero", "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,1]", pack16(H_ZERO, H_TWO), 1),
        ("f16_float_const_opsel1_nonzero", "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,1]", pack16(H_ONE, H_TWO), 0),
        # Inline float constant -2.0:
        ("f16_float_const_neg2", "v_cmp_eq_f16_e64 s4, v1, -2.0 op_sel:[0,0]", pack16(H_MTWO, H_ONE), 1),
        ("f16_float_const_neg2_hi_zero", "v_cmp_eq_f16_e64 s4, v1, -2.0 op_sel:[0,1]", pack16(H_ZERO, H_ONE), 1),
        
        # 32-bit literal constant in f16: 0x40003C00 (lo = 1.0, hi = 2.0)
        ("f16_lit_opsel0", "v_cmp_eq_f16_e64 s4, v1, 0x40003C00 op_sel:[0,0]", pack16(H_ONE, H_FOUR), 1),
        ("f16_lit_opsel1", "v_cmp_eq_f16_e64 s4, v1, 0x40003C00 op_sel:[1,1]", pack16(H_FOUR, H_TWO), 1),
        
        # Inline integer constants in u16: 1
        ("u16_int_const1_opsel0", "v_cmp_eq_u16_e64 s4, v1, 1 op_sel:[0,0]", pack16(1, 2), 1),
        ("u16_int_const1_opsel1_reads_0", "v_cmp_eq_u16_e64 s4, v1, 1 op_sel:[0,1]", pack16(0, 2), 1),
        # Inline integer constant -1 (0xFFFFFFFF) in u16:
        ("u16_int_const_neg1_opsel0", "v_cmp_eq_u16_e64 s4, v1, -1 op_sel:[0,0]", pack16(0xFFFF, 0), 1),
        ("u16_int_const_neg1_opsel1", "v_cmp_eq_u16_e64 s4, v1, -1 op_sel:[0,1]", pack16(0xFFFF, 0), 1),
        
        # Inline integer constants in i16:
        ("i16_int_const_neg1_opsel0", "v_cmp_eq_i16_e64 s4, v1, -1 op_sel:[0,0]", pack16(0xFFFF, 0), 1),
        ("i16_int_const_neg1_opsel1", "v_cmp_eq_i16_e64 s4, v1, -1 op_sel:[0,1]", pack16(0xFFFF, 0), 1),
        ("i16_int_const1_opsel0", "v_cmp_eq_i16_e64 s4, v1, 1 op_sel:[0,0]", pack16(1, 2), 1),
        ("i16_int_const1_opsel1_reads_0", "v_cmp_eq_i16_e64 s4, v1, 1 op_sel:[0,1]", pack16(0, 2), 1),
        
        # 32-bit literals in u16 and i16: 0x80007FFF (lo = 0x7FFF = 32767, hi = 0x8000 = -32768)
        ("u16_lit_opsel0", "v_cmp_eq_u16_e64 s4, v1, 0x80007FFF op_sel:[0,0]", pack16(0x7FFF, 0), 1),
        ("u16_lit_opsel1", "v_cmp_eq_u16_e64 s4, v1, 0x80007FFF op_sel:[0,1]", pack16(0x8000, 0), 1),
        ("i16_lit_opsel0_signed", "v_cmp_eq_i16_e64 s4, v1, 0x80007FFF op_sel:[0,0]", pack16(0x7FFF, 0), 1),
        ("i16_lit_opsel1_signed", "v_cmp_eq_i16_e64 s4, v1, 0x80007FFF op_sel:[1,1]", pack16(0, 0x8000), 1),
        ("i16_lit_lt_signed", "v_cmp_lt_i16_e64 s4, v1, 0x80007FFF op_sel:[0,1]", pack16(0, 0), 0), # 0 < -32768 is False!
    ]
    
    # For simplicity, run each individually or batch them into slots
    for idx, (tag, cmp_line, v1_w, want) in enumerate(tests):
        single_body = [
            f"v_mov_b32 v1, {v1_w:#x}",
            cmp_line,
            "v_and_b32_e32 v2, 1, s4",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        out = run_on_silicon(f"cs_const_{idx}", single_body, wave_size=wave_size)
        assert out is not None, f"Failed {tag}"
        got = out[0]
        if got != want:
            raise AssertionError(f"Test '{tag}' FAILED on silicon: got {got}, want {want}")
        print(f"  [{tag}] -> {got} (matched expected {want})")
    print(f"[{name}] ALL {len(tests)} CONSTANT/LITERAL TESTS PASSED on gfx1036 silicon!")


def test_sgpr_high_halves(wave_size=32):
    """Probe SGPR high halves as src0 and src1 with op_sel and modifiers."""
    name = f"cs_probe_sgpr_hi_w{wave_size}"
    # User SGPR s4 is passed via VKC_SGPRS:
    # Set s4 = 0x44003C00 (lo = 1.0, hi = 4.0)
    user_sgpr_s4 = pack16(H_ONE, H_FOUR)
    
    tests = [
        # (tag, cmp_line, v1_word, want)
        # s4 as src1:
        ("f16_sgpr_src1_opsel0", "v_cmp_eq_f16_e64 s5, v1, s4 op_sel:[0,0]", pack16(H_ONE, 0), 1),
        ("f16_sgpr_src1_opsel1", "v_cmp_eq_f16_e64 s5, v1, s4 op_sel:[0,1]", pack16(H_FOUR, 0), 1),
        # s4 as src0:
        ("f16_sgpr_src0_opsel0", "v_cmp_eq_f16_e64 s5, s4, v1 op_sel:[0,0]", pack16(H_ONE, 0), 1),
        ("f16_sgpr_src0_opsel1", "v_cmp_eq_f16_e64 s5, s4, v1 op_sel:[1,0]", pack16(H_FOUR, 0), 1),
        # s4 with neg / abs on high half:
        ("f16_sgpr_neg_hi", "v_cmp_eq_f16_e64 s5, v1, -s4 op_sel:[0,1]", pack16(H_MFOUR, 0), 1),
        ("f16_sgpr_abs_hi", "v_cmp_eq_f16_e64 s5, v1, |s4| op_sel:[0,1]", pack16(H_FOUR, 0), 1),
        
        # u16 / i16 with SGPR:
        # s4 = pack16(0x7FFF, 0x8000) (lo = +32767, hi = -32768 in i16 / 32768 in u16)
    ]
    for idx, (tag, cmp_line, v1_w, want) in enumerate(tests):
        single_body = [
            f"v_mov_b32 v1, {v1_w:#x}",
            cmp_line,
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        out = run_on_silicon(f"cs_sgpr_{idx}", single_body, wave_size=wave_size, user_sgprs=[user_sgpr_s4])
        assert out is not None, f"Failed {tag}"
        got = out[0]
        if got != want:
            raise AssertionError(f"Test '{tag}' FAILED on silicon: got {got}, want {want}")
        print(f"  [{tag}] -> {got} (matched expected {want})")
    
    # Also test integer SGPR with s4 = pack16(0x7FFF, 0xFFFF) (lo = 32767, hi = -1 in i16)
    int_sgpr = pack16(0x7FFF, 0xFFFF)
    int_tests = [
        ("u16_sgpr_opsel0", "v_cmp_eq_u16_e64 s5, v1, s4 op_sel:[0,0]", pack16(0x7FFF, 0), 1),
        ("u16_sgpr_opsel1", "v_cmp_eq_u16_e64 s5, v1, s4 op_sel:[0,1]", pack16(0xFFFF, 0), 1),
        ("i16_sgpr_opsel1_lt", "v_cmp_lt_i16_e64 s5, s4, v1 op_sel:[1,0]", pack16(0, 0), 1), # -1 < 0 is True
        ("u16_sgpr_opsel1_lt", "v_cmp_lt_u16_e64 s5, s4, v1 op_sel:[1,0]", pack16(0, 0), 0), # 65535 < 0 is False
    ]
    for idx, (tag, cmp_line, v1_w, want) in enumerate(int_tests, len(tests)):
        single_body = [
            f"v_mov_b32 v1, {v1_w:#x}",
            cmp_line,
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        out = run_on_silicon(f"cs_sgpr_{idx}", single_body, wave_size=wave_size, user_sgprs=[int_sgpr])
        assert out is not None, f"Failed {tag}"
        got = out[0]
        if got != want:
            raise AssertionError(f"Test '{tag}' FAILED on silicon: got {got}, want {want}")
        print(f"  [{tag}] -> {got} (matched expected {want})")
    print(f"[{name}] ALL SGPR HIGH-HALF TESTS PASSED on gfx1036 silicon!")


def test_f16_nan_and_modifiers(wave_size=32):
    """Probe ordered vs unordered, qNaN, sNaN, abs and neg modifiers on silicon."""
    name = f"cs_probe_nan_mods_w{wave_size}"
    # Nan pairs:
    # v1 = pack16(1.0, qNaN), v2 = pack16(qNaN, 2.0)
    xw = pack16(H_ONE, H_QNAN)
    yw = pack16(H_QNAN, H_TWO)
    
    tests = [
        # (0, 0): 1.0 vs qNaN -> unordered
        ("f16_ord_00", "v_cmp_o_f16_e64 s4, v1, v2 op_sel:[0,0]", 0),
        ("f16_unord_00", "v_cmp_u_f16_e64 s4, v1, v2 op_sel:[0,0]", 1),
        ("f16_eq_nan_00", "v_cmp_eq_f16_e64 s4, v1, v2 op_sel:[0,0]", 0),
        ("f16_neq_nan_00", "v_cmp_neq_f16_e64 s4, v1, v2 op_sel:[0,0]", 1),
        
        # (0, 1): 1.0 vs 2.0 -> ordered
        ("f16_ord_01", "v_cmp_o_f16_e64 s4, v1, v2 op_sel:[0,1]", 1),
        ("f16_unord_01", "v_cmp_u_f16_e64 s4, v1, v2 op_sel:[0,1]", 0),
        ("f16_lt_01", "v_cmp_lt_f16_e64 s4, v1, v2 op_sel:[0,1]", 1),
        ("f16_gt_01", "v_cmp_gt_f16_e64 s4, v1, v2 op_sel:[0,1]", 0),
        
        # (1, 0): qNaN vs qNaN -> unordered
        ("f16_ord_10", "v_cmp_o_f16_e64 s4, v1, v2 op_sel:[1,0]", 0),
        ("f16_unord_10", "v_cmp_u_f16_e64 s4, v1, v2 op_sel:[1,0]", 1),
        
        # Modifiers on high half:
        # v1.hi is qNaN: |qNaN| is still NaN!
        ("f16_abs_nan_ord", "v_cmp_o_f16_e64 s4, |v1|, v2 op_sel:[1,1]", 0),
        ("f16_neg_nan_unord", "v_cmp_u_f16_e64 s4, -v1, v2 op_sel:[1,1]", 1),
    ]
    for idx, (tag, cmp_line, want) in enumerate(tests):
        single_body = [
            f"v_mov_b32 v1, {xw:#x}",
            f"v_mov_b32 v2, {yw:#x}",
            cmp_line,
            "v_and_b32_e32 v3, 1, s4",
            "v_lshlrev_b32_e32 v4, 2, v0",
            "buffer_store_dword v3, v4, s[0:3], 0 offen",
        ]
        out = run_on_silicon(f"cs_nan_{idx}", single_body, wave_size=wave_size)
        assert out is not None, f"Failed {tag}"
        got = out[0]
        if got != want:
            raise AssertionError(f"Test '{tag}' FAILED on silicon: got {got}, want {want}")
        print(f"  [{tag}] -> {got} (matched expected {want})")
    print(f"[{name}] ALL NaN & MODIFIER TESTS PASSED on gfx1036 silicon!")


def main():
    print("=" * 70)
    print("PROBING GFX1036 SILICON FOR V_CMP(X) 16-BIT OP_SEL & CONSTANTS")
    print("=" * 70)
    
    print("\n--- 1. Testing v_cmp inactive-lane 0 write in sdst (wave32 & wave64) ---")
    test_vcmp_inactive_lanes(32)
    test_vcmp_inactive_lanes(64)
    
    print("\n--- 2. Testing v_cmpx EXEC update (wave32 & wave64) ---")
    test_vcmpx_exec_update(32)
    test_vcmpx_exec_update(64)
    
    print("\n--- 3. Testing inline float/int constants & 32-bit literals with op_sel=0/1 (wave32 & wave64) ---")
    test_constants_and_literals(32)
    test_constants_and_literals(64)
    
    print("\n--- 4. Testing SGPR high halves with op_sel & modifiers (wave32 & wave64) ---")
    test_sgpr_high_halves(32)
    test_sgpr_high_halves(64)
    
    print("\n--- 5. Testing f16 NaN (ordered vs unordered) & modifiers (wave32 & wave64) ---")
    test_f16_nan_and_modifiers(32)
    test_f16_nan_and_modifiers(64)
    
    print("\n" + "=" * 70)
    print("ALL GFX1036 SILICON PROBES (WAVE32 & WAVE64) PASSED WITH ZERO DIVERGENCES!")
    print("=" * 70)

if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    main()
