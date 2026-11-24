; More on exchanging a lane mask once per value (--wave64-emulate), see cs_w64x_mask_reuse.asm:
;  1. EXEC read twice without a change in between (s_bcnt1 exec): both give 10 (lanes < 10), s6 + s7 = 20.
;  2. s[4:5] = lanes with (lane & 7) == 5 (8 lanes): s9 = s_bcnt1 = 8; a buffer_store (it opens labels in the translation)
;     and the same mask read again: s10 = 8.
;  3. One half of the pair rewritten from another mask, the other kept: s5 = s5 & s13 (s[12:13] = lanes < 40), then
;     v_mbcnt_lo(s4) + v_mbcnt_hi(s5). s4 selects lanes 5, 13, 21, 29, s5 now only lane 37: lane 63 -> 4 + 1, lane 20 -> 2.
;  4. A block that ends in a voted branch (s_cbranch_vccnz, not taken) and a successor whose first instruction is a
;     load of the OTHER half's store from before the vote: out[lane] = 100 + lane, then v9 = out[lane ^ 32].
;     lane 0 -> 132, lane 63 -> 131.
; Stores: out[0] s6, out[1] s6 + s7, out[2] s9, out[3] s10, out[4] lane 63 v5, out[5] lane 20 v5, out[6] lane 0 v9,
;   out[7] lane 63 v9.
; Run: vk_compute_test cs_w64x_mask_reuse2.spv 1 dump --require-wave64
;   -> out[0..7] = 10 20 8 8 5 2 132 131
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_cmpx_gt_u32_e32 10, v1
s_bcnt1_i32_b64 s6, exec
s_bcnt1_i32_b64 s7, exec
s_mov_b64 exec, -1
s_add_u32 s8, s6, s7
v_and_b32_e32 v2, 7, v1
v_cmp_eq_u32_e64 s[4:5], 5, v2
s_bcnt1_i32_b64 s9, s[4:5]
v_lshlrev_b32_e32 v7, 2, v1
buffer_store_dword v1, v7, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_bcnt1_i32_b64 s10, s[4:5]
v_cmp_gt_u32_e64 s[12:13], 40, v1
s_and_b32 s5, s5, s13
v_mbcnt_lo_u32_b32 v5, s4, 0
v_mbcnt_hi_u32_b32 v5, s5, v5
v_add_nc_u32_e32 v6, 100, v1
buffer_store_dword v6, v7, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
v_xor_b32_e32 v8, 32, v1
v_lshlrev_b32_e32 v8, 2, v8
v_cmp_lt_u32_e32 vcc, 100, v1
s_cbranch_vccnz 2
buffer_load_dword v9, v8, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
