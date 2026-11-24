; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_cmpst_f32 / ds_cmpst_rtn_f32 and buffer_atomic_fcmpswap (#3153): a float compare, so -0 matches +0 and a NaN
; never matches, not even its own bits. Lane 0 of a wave32 group stores eight dwords:
;   out[0] = ds_cmpst_rtn_f32 old value (LDS +0, CMP -0, DATA 1.0)                       = 0x00000000
;   out[1] = the LDS word after it: -0 == +0, so DATA is stored                          = 0x3F800000
;   out[2] = the LDS word after ds_cmpst_rtn_f32 (LDS NaN, CMP the same NaN, DATA 1.0)    = 0x7FC00000
;   out[3] = the LDS word after ds_cmpst_f32 (LDS 1.0, CMP 2.0, DATA 3.0): no match       = 0x3F800000
;   out[4] = buffer_atomic_fcmpswap glc old value (M -0 at byte 0x40, DATA 3.0, CMP +0)   = 0x80000000
;   out[5] = the word at byte 0x40 after it: -0 == +0                                     = 0x40400000
;   out[6] = the word at byte 0x44 after fcmpswap (M 1.0, DATA 2.0, CMP 0x3F800001)       = 0x3F800000
;   out[7] = buffer_atomic_fcmpswap glc old value (M NaN at byte 0x48, DATA 1.0, CMP NaN) = 0x7FC00000
; The atomic words are seeded with buffer_store_dword first. EXEC is cut to lane 0 before any memory op (one address
; per op, #1850).
; group: 32,1,1
; Run: vk_compute_test cs_w32_fcmpswap.spv 1 dump -> out[0..7] = 0 1065353216 2143289344 1065353216 2147483648 1077936128 1065353216 2143289344
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0x40
v_mov_b32_e32 v11, 0
ds_write_b32 v10, v11
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v12, 0x80000000
v_mov_b32_e32 v13, 1.0
ds_cmpst_rtn_f32 v14, v10, v12, v13
s_waitcnt lgkmcnt(0)
ds_read_b32 v15, v10
v_mov_b32_e32 v16, 0x80
v_mov_b32_e32 v17, 0x7fc00000
ds_write_b32 v16, v17
s_waitcnt lgkmcnt(0)
ds_cmpst_rtn_f32 v18, v16, v17, v13
s_waitcnt lgkmcnt(0)
ds_read_b32 v18, v16
v_mov_b32_e32 v19, 0xc0
ds_write_b32 v19, v13
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v12, 2.0
v_mov_b32_e32 v11, 0x40400000
ds_cmpst_f32 v19, v12, v11
s_waitcnt lgkmcnt(0)
ds_read_b32 v21, v19
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v22, 0x80000000
buffer_store_dword v22, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x44
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x48
buffer_store_dword v17, v20, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v2, 0x40400000
v_mov_b32_e32 v3, 0
buffer_atomic_fcmpswap v[2:3], v20, s[0:3], 0 offen glc
v_mov_b32_e32 v20, 0x44
v_mov_b32_e32 v4, 2.0
v_mov_b32_e32 v5, 0x3f800001
buffer_atomic_fcmpswap v[4:5], v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x48
v_mov_b32_e32 v6, 1.0
v_mov_b32_e32 v7, 0x7fc00000
buffer_atomic_fcmpswap v[6:7], v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0x40
buffer_load_dword v8, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v20, 0x44
buffer_load_dword v9, v20, s[0:3], 0 offen glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0
buffer_store_dword v14, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v18, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v2, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_store_dword v9, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
buffer_store_dword v6, v20, s[0:3], 0 offen
s_endpgm
