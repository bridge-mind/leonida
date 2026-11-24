; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Ops merged with CPU-only checks (#1479 #1496 #1510 #1514), run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords:
;   out[0] = s_quadmask_b32(0x01020408)                   = 0x55        (quads 0, 2, 4, 6 non-zero)
;   out[1] = low word of s_quadmask_b64(0x80000000_00000001) = 0x8001   (quads 0 and 15)
;   out[2] = high word of s_bitreplicate_b64_b32(0x80000001) = 0xC0000000 (bit 31 -> bits 63:62)
;   out[3] = v_cvt_pknorm_i16_f32(0.5, -1.0)              = 0x80014000  (16383.5 -> 16384 even; -32767)
;   out[4] = v_cvt_pknorm_u16_f32(1.0, 0.25)              = 0x4000FFFF  (65535; 16383.75 -> 16384)
;   out[5] = ds_wrxchg2_rtn_b32 old word at offset0:2     = 0x77        (LDS seeded 0x77)
;   out[6] = LDS word at offset1:5 after the exchange     = 0x99        (DATA2)
;   out[7] = v7 after ds_nop and v_clrexcp                = 0x1234      (no effect)
; EXEC is cut to lane 0 before the LDS ops: 32 lanes on one LDS address would race (which lane's
; ds_cmpst / ds_wrxchg2 runs first is up to the hardware), as #1850 showed for ds_xor_b64.
; group: 32,1,1
; Run: vk_compute_test cs_w32_new_ops.spv 1 dump -> out[0..7] = 85 32769 3221225472 2147565568 1073807359 119 153 4660
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 0x1020408
s_quadmask_b32 s8, s6
s_mov_b32 s6, 1
s_mov_b32 s7, 0x80000000
s_quadmask_b64 s[10:11], s[6:7]
s_mov_b32 s6, 0x80000001
s_bitreplicate_b64_b32 s[12:13], s6
v_mov_b32_e32 v1, 0.5
v_mov_b32_e32 v2, -1.0
v_cvt_pknorm_i16_f32 v3, v1, v2
v_mov_b32_e32 v1, 1.0
v_mov_b32_e32 v2, 0.25
v_cvt_pknorm_u16_f32 v4, v1, v2
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0x20
v_mov_b32_e32 v11, 0x28
v_mov_b32_e32 v12, 0x34
v_mov_b32_e32 v13, 0x77
v_mov_b32_e32 v14, 0x66
ds_write_b32 v11, v13
ds_write_b32 v12, v14
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v15, 0x88
v_mov_b32_e32 v16, 0x99
ds_wrxchg2_rtn_b32 v[5:6], v10, v15, v16 offset0:2 offset1:5
s_waitcnt lgkmcnt(0)
ds_read_b32 v6, v12
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v7, 0x1234
ds_nop
v_clrexcp
v_mov_b32_e32 v24, s8
v_mov_b32_e32 v25, 0
buffer_store_dword v24, v25, s[0:3], 0 offen
v_mov_b32_e32 v24, s10
v_mov_b32_e32 v25, 4
buffer_store_dword v24, v25, s[0:3], 0 offen
v_mov_b32_e32 v24, s13
v_mov_b32_e32 v25, 8
buffer_store_dword v24, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v4, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v5, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v6, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v7, v25, s[0:3], 0 offen
s_endpgm
