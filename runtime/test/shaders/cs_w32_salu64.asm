; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 64-bit scalar data ops and v_mad_u64_u32 (#723 #726 #729 #733 #549 #613 #546), run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords:
;   out[0] = low word of s_brev_b64(0x12345678_0000000F)          = 0x1E6A2C48  (reverse(S0.hi))
;   out[1] = low word of s_bfe_u64(0x00000123_45678000, w16 o28)  = 0x1234      (bits 43..28)
;   out[2] = high word of s_bfe_i64(0x00000001_80000000, w2 o31)  = 0xFFFFFFFF  (field 0b11 = -1, sign-extended)
;   out[3] = high word of s_bitset1_b64(0, 40)                    = 0x100       (bit 40)
;   out[4] = s_flbit_i32_b64(0x00000000_00010000)                 = 47          (zeros above bit 16)
;   out[5] = s_flbit_i32_i64(0xFFFFFFFF_FFFF0000)                 = 48          (first bit unlike the sign: 15)
;   out[6] = s_cmov_b32 (SCC = 1) of s_sext_i32_i16(0x00018000)   = 0xFFFF8000
;   out[7] = high word of v_mad_u64_u32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF_FFFFFFFF) = 0xFFFFFFFE (carry dropped)
; group: 32,1,1
; Run: vk_compute_test cs_w32_salu64.spv 1 dump -> out[0..7] = 510274632 4660 4294967295 256 47 48 4294934528 4294967294
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 15
s_mov_b32 s7, 0x12345678
s_brev_b64 s[8:9], s[6:7]
s_mov_b32 s6, 0x45678000
s_mov_b32 s7, 0x123
s_mov_b32 s10, 0x10001c
s_bfe_u64 s[10:11], s[6:7], s10
s_mov_b32 s6, 0x80000000
s_mov_b32 s7, 1
s_mov_b32 s12, 0x2001f
s_bfe_i64 s[12:13], s[6:7], s12
s_mov_b32 s14, 0
s_mov_b32 s15, 0
s_bitset1_b64 s[14:15], 40
s_mov_b32 s6, 0x10000
s_mov_b32 s7, 0
s_flbit_i32_b64 s16, s[6:7]
s_mov_b32 s6, 0xffff0000
s_mov_b32 s7, -1
s_flbit_i32_i64 s17, s[6:7]
s_mov_b32 s19, 0x18000
s_sext_i32_i16 s18, s19
s_mov_b32 s20, 0x5eed
s_cmp_eq_u32 s19, 0x18000
s_cmov_b32 s20, s18
v_mov_b32_e32 v4, -1
v_mov_b32_e32 v6, -1
v_mov_b32_e32 v7, -1
v_mad_u64_u32 v[2:3], s22, v4, v4, v[6:7]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s16
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s20
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v3, v21, s[0:3], 0 offen
s_endpgm
