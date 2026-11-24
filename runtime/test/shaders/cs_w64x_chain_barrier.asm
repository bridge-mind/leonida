; 128 threads = two wave64s. Every thread writes LDS[tid] = 500 + tid and waits for it, then the block ends in a voted
; branch (s_cbranch_vccnz, not taken) whose successor starts with s_barrier (--wave64-emulate runs that successor in the
; round that reads the vote). After the barrier every thread reads LDS[tid ^ 64], the other wave's word.
;   t0 -> 500 + 64 = 564, t127 -> 500 + 63 = 563, t64 -> 500, t63 -> 500 + 127 = 627.
; Stores: out[0] t0, out[1] t127, out[2] t64, out[3] t63.
; Run: vk_compute_test cs_w64x_chain_barrier.spv 1 dump --require-wave64
;   -> out[0..7] = 564 563 500 627 and out[4..7] untouched (0xeeeeeeee)
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 128,1,1
v_add_nc_u32_e32 v6, 500, v0
v_lshlrev_b32_e32 v7, 2, v0
ds_write_b32 v7, v6
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v8, 64, v0
v_lshlrev_b32_e32 v8, 2, v8
v_cmp_lt_u32_e32 vcc, 1000, v0
s_cbranch_vccnz 1
s_barrier
ds_read_b32 v9, v8
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 127, v0
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 64, v0
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v0
v_mov_b32_e32 v20, v9
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
