; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Ops merged with CPU-only checks (#1471 #825 #828 #814 #810), run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords:
;   out[0] = high word of s_nand_b64(0xF0F0F0F0_12345678, 0xFF00FF00_FFFFFFFF) = 0x0FFF0FFF  (~(S0 & S1))
;   out[1] = low word of s_xnor_b64(0xF0F0F0F0_12345678, 0xFF00FF00_0F0F0F0F)  = 0xE2C4A688  (~(S0 ^ S1))
;   out[2] = v_cvt_pk_u16_u32(0x12345, 7)                 = 0x0007FFFF  (low half saturated to 0xFFFF)
;   out[3] = v_cvt_pk_i16_i32(-40000, 40000)              = 0x7FFF8000  (each half saturated to i16)
;   out[4] = v_mad_u32_u16(0xFFFF, 0xFFFF, 1)             = 0xFFFE0002
;   out[5] = v_mad_i32_i16(0x8000, 2, 5)                  = 0xFFFF0005  (-32768 * 2 + 5)
;   out[6] = (ds_cmpst_rtn_b32 old << 8) | LDS after      = 0x509       (LDS 5, compare 5, store 9)
;   out[7] = high dword of the second ds_read2_b64 element = 0xCAFEF00D (written by ds_write2_b64 offset1:3)
; EXEC is cut to lane 0 before the LDS ops: 32 lanes on one LDS address would race (which lane's
; ds_cmpst / ds_wrxchg2 runs first is up to the hardware), as #1850 showed for ds_xor_b64.
; group: 32,1,1
; Run: vk_compute_test cs_w32_misc_ext.spv 1 dump -> out[0..7] = 268374015 3804538504 524287 2147450880 4294836226 4294901765 1289 3405705229
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 0x12345678
s_mov_b32 s7, 0xf0f0f0f0
s_mov_b32 s10, -1
s_mov_b32 s11, 0xff00ff00
s_nand_b64 s[8:9], s[6:7], s[10:11]
s_mov_b32 s10, 0xf0f0f0f
s_xnor_b64 s[12:13], s[6:7], s[10:11]
v_mov_b32_e32 v1, 0x12345
v_mov_b32_e32 v2, 7
v_cvt_pk_u16_u32 v3, v1, v2
v_mov_b32_e32 v1, 0xffff63c0
v_mov_b32_e32 v2, 0x9c40
v_cvt_pk_i16_i32 v4, v1, v2
v_mov_b32_e32 v1, 0xffff
v_mad_u32_u16 v5, v1, v1, 1
v_mov_b32_e32 v1, 0x8000
v_mad_i32_i16 v6, v1, 2, 5
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0x10
v_mov_b32_e32 v11, 5
v_mov_b32_e32 v12, 9
ds_write_b32 v10, v11
s_waitcnt lgkmcnt(0)
ds_cmpst_rtn_b32 v13, v10, v11, v12
s_waitcnt lgkmcnt(0)
ds_read_b32 v14, v10
s_waitcnt lgkmcnt(0)
v_lshl_or_b32 v7, v13, 8, v14
v_mov_b32_e32 v10, 0x40
v_mov_b32_e32 v16, 0x11111111
v_mov_b32_e32 v17, 0x22222222
v_mov_b32_e32 v18, 0x33333333
v_mov_b32_e32 v19, 0xcafef00d
ds_write2_b64 v10, v[16:17], v[18:19] offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[20:23], v10 offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v24, s9
v_mov_b32_e32 v25, 0
buffer_store_dword v24, v25, s[0:3], 0 offen
v_mov_b32_e32 v24, s12
v_mov_b32_e32 v25, 4
buffer_store_dword v24, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v4, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v5, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v6, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v7, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v23, v25, s[0:3], 0 offen
s_endpgm
