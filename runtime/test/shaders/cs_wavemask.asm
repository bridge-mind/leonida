; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 compute: instructions that read a lane mask as a whole, on a partly active wave. EXEC = lanes 16-47
; (exec_lo = 0xffff0000, exec_hi = 0xffff), saved in s[6:7]. s_bcnt1 of EXEC and of the saved pair = 32 each,
; s_ff1 of EXEC = 16, s_bcnt1 of a compare result (lane < 24: lanes 16-23, VCC is clear outside EXEC) = 8,
; v_mbcnt_lo/hi of the saved pair = set lanes below this one = t - 16. Scalars are wave-wide: every lane stores them.
; s_cmp_lg_u32 of the low half with 0 is true for the wave (+1), s_cmp_eq_u64 of the pair with 0 is false (+0).
; Then the saved pair is parked in lanes 0 and 1 of v10 (v_writelane, as a shader out of SGPRs does), fetched back
; with all lanes on, and moved into EXEC: lanes 16-47 again, which set bit 7 of the top byte. A second copy goes
; through the vector unit (v_mov_b32 / v_or3_b32 / v_readfirstlane_b32, the way three SGPRs get ORed) into s[12:13]
; and into EXEC: the same lanes set bit 6.
; Same FSSM header as cs_lds. Run: vk_compute_test cs_wavemask.spv 1 wavemask --require-wave64
;   -> out[t] = 65 | 16 << 8 | 8 << 16 | (0xc0 + t - 16 for t in 16..47, else 0) << 24
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshlrev_b32_e32 v3, 2, v1
v_mov_b32_e32 v4, 0
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff
s_mov_b64 s[6:7], exec
s_bcnt1_i32_b64 s8, exec
s_bcnt1_i32_b64 s10, s[6:7]
s_add_i32 s8, s8, s10
s_cmp_lg_u32 s6, 0
s_cselect_b32 s14, 1, 0
s_add_i32 s8, s8, s14
s_cmp_eq_u64 s[6:7], 0
s_cselect_b32 s14, 2, 0
s_add_i32 s8, s8, s14
s_ff1_i32_b64 s9, exec
v_cmp_gt_u32_e32 vcc_lo, 24, v1
s_bcnt1_i32_b64 s11, vcc
v_mbcnt_lo_u32_b32 v4, s6, 0
v_mbcnt_hi_u32_b32 v4, s7, v4
v_mov_b32_e32 v6, s6
v_or3_b32 v6, s6, s6, v6
v_readfirstlane_b32 s12, v6
v_mov_b32_e32 v6, s7
v_readfirstlane_b32 s13, v6
v_writelane_b32 v10, s6, 0
v_writelane_b32 v10, s7, 1
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
s_mov_b32 s6, 0
s_mov_b32 s7, 0
v_mov_b32_e32 v5, 0
v_readlane_b32 s6, v10, 0
v_readlane_b32 s7, v10, 1
s_mov_b64 exec, s[6:7]
v_mov_b32_e32 v5, 0x80
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
v_mov_b32_e32 v7, 0
s_mov_b64 exec, s[12:13]
v_mov_b32_e32 v7, 0x40
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
s_lshl_b32 s9, s9, 8
s_lshl_b32 s11, s11, 16
s_or_b32 s8, s8, s9
s_or_b32 s8, s8, s11
v_or3_b32 v4, v4, v5, v7
v_lshlrev_b32_e32 v4, 24, v4
v_or_b32_e32 v2, s8, v4
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm
