; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Integer ops added with CPU-only checks (#445 #448 #449 #440 #544 #577), run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords:
;   out[0] = s_absdiff_i32(0x7FFFFFFF, 0x80000000)    = 1           (the 32-bit difference wraps to -1 first)
;   out[1] = s_lshl2_add_u32(5, 7)                     = 27          ((5 << 2) + 7)
;   out[2] = SCC of s_lshl1_add_u32(0x80000000, 0)     = 1           (the bit shifted out of S0 is a carry)
;   out[3] = s_pack_lh_b32_b16(0x11112222, 0x33334444) = 0x33332222  (S0 low half, S1 high half)
;   out[4] = s_mul_hi_i32(0x80000000, 0x7FFFFFFF)     = 0xC0000000  (-2^31 * (2^31-1), high word)
;   out[5] = v_mul_hi_i32_i24(0x800000, 0x800000)      = 0x4000      ((-2^23)^2 = 2^46, high word)
;   out[6] = v_sad_u8(0x01020304, 0x04030201, 10)      = 18          (10 + 3 + 1 + 1 + 3)
;   out[7] = v_lerp_u8(0x10203040, 0x30405060, 0x01010101) = 0x20304050  (per byte (a + b + 1) >> 1)
; group: 32,1,1
; Run: vk_compute_test cs_w32_int_ext.spv 1 dump -> out[0..7] = 1 27 1 858989090 3221225472 16384 18 540033104
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 0x7fffffff
s_mov_b32 s7, 0x80000000
s_absdiff_i32 s8, s6, s7
s_mov_b32 s9, 5
s_lshl2_add_u32 s9, s9, 7
s_lshl1_add_u32 s10, s7, 0
s_cselect_b32 s10, 1, 0
s_mov_b32 s11, 0x11112222
s_mov_b32 s12, 0x33334444
s_pack_lh_b32_b16 s11, s11, s12
s_mul_hi_i32 s12, s7, s6
v_mov_b32_e32 v2, 0x800000
v_mul_hi_i32_i24_e64 v3, v2, v2
v_mov_b32_e32 v4, 0x1020304
v_mov_b32_e32 v5, 0x4030201
v_sad_u8 v4, v4, v5, 10
v_mov_b32_e32 v5, 0x10203040
v_mov_b32_e32 v6, 0x30405060
v_mov_b32_e32 v7, 0x1010101
v_lerp_u8 v5, v5, v6, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v5, v21, s[0:3], 0 offen
s_endpgm
