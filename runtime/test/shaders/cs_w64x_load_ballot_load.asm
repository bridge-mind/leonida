; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; buffer_load, then a ballot + s_bcnt1, then a second buffer_load, then a store, all in one basic block (this
; sequence once produced invalid SPIR-V under --wave64-emulate), with a checked result. The loads read the other
; half's stores of the same wave (out[lane] = 10 + lane, then the stores are waited for).
;   v4 = out[lane ^ 32] = 10 + (lane ^ 32)         ballot (v4 > 41), i.e. lane ^ 32 >= 32, i.e. lanes 0-31: s6 = 32
;   v5 = out[(lane + 1) & 63] = 10 + ((lane + 1) & 63)       v6 = v5 + s6
;   v6: lane 0 -> 10 + 1 + 32 = 43, lane 31 -> 10 + 32 + 32 = 74, lane 32 -> 10 + 33 + 32 = 75,
;       lane 63 -> 10 + 0 + 32 = 42.   v4: lane 0 -> 42, lane 40 -> 10 + 8 = 18, lane 63 -> 10 + 31 = 41.   s6 = 32.
; Stores: out[0] lane 0 v6, out[1] lane 31 v6, out[2] lane 32 v6, out[3] lane 63 v6, out[4] lane 0 v4,
;   out[5] lane 40 v4, out[6] lane 63 v4, out[7] lane 5 s6.
; Run: vk_compute_test cs_w64x_load_ballot_load.spv 1 dump --require-wave64
;   -> out[0..7] = 43 74 75 42 42 18 41 32
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 10, v1
v_lshlrev_b32_e32 v3, 2, v1
buffer_store_dword v2, v3, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_waitcnt vmcnt(0)
v_xor_b32_e32 v7, 32, v1
v_lshlrev_b32_e32 v7, 2, v7
buffer_load_dword v4, v7, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_cmp_gt_u32_e64 s[4:5], v4, 41
s_bcnt1_i32_b64 s6, s[4:5]
v_add_nc_u32_e32 v8, 1, v1
v_and_b32_e32 v8, 63, v8
v_lshlrev_b32_e32 v8, 2, v8
buffer_load_dword v5, v8, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v6, s6, v5
buffer_store_dword v6, v3, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 31, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v6
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 40, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 5, v1
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
