; buffer_store_format encoder: rgba8 unorm (56), 2_10_10_10 unorm (50), two r16f (13) halves of one dword (atomic merge),
; d16 rgba16f (71), BGRA UNORM stores (56, 50). Run: vk_compute_test cs_store.spv 1 dump -> out[0..7] =
; 0x400080ff 0xe00003ff 0x3c004000 0xeeeeeeee (untouched) 0x40003c00 0xbc003800 0x40ff8000 0x3ff00000. Same FSSM header as cs_lds. OOB_SELECT 3 in word3 (the output V# is raw).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 1.0
v_mov_b32_e32 v2, 0.5
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0x3e800000
v_mov_b32_e32 v5, 0
s_mov_b32 s3, 0x31038fac
buffer_store_format_xyzw v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v6, 1.0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0.5
v_mov_b32_e32 v9, 1.0
v_mov_b32_e32 v5, 4
s_mov_b32 s3, 0x31032fac
buffer_store_format_xyzw v[6:9], v5, s[0:3], 0 offen
s_mov_b32 s3, 0x3100d000
v_mov_b32_e32 v10, 2.0
v_mov_b32_e32 v5, 8
buffer_store_format_x v10, v5, s[0:3], 0 offen
v_mov_b32_e32 v10, 1.0
v_mov_b32_e32 v5, 10
buffer_store_format_x v10, v5, s[0:3], 0 offen
s_mov_b32 s3, 0x31047000
v_mov_b32_e32 v11, 0x40003c00
v_mov_b32_e32 v12, 0xbc003800
v_mov_b32_e32 v5, 16
buffer_store_format_d16_xyzw v[11:12], v5, s[0:3], 0 offen
; The first store's values (1, .5, 0, .25), now with Z,Y,X,W selectors.
s_mov_b32 s3, 0x31038f2e
v_mov_b32_e32 v5, 24
buffer_store_format_xyzw v[1:4], v5, s[0:3], 0 offen
; The low half contains 1.0. A single red component fills the third 10-bit field.
s_mov_b32 s3, 0x31032f2e
v_mov_b32_e32 v11, 0x3c00
v_mov_b32_e32 v5, 28
buffer_store_format_d16_x v11, v5, s[0:3], 0 offen
s_endpgm
