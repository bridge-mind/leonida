; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; v_cvt_pk_u8_f32 (VOP3 opcode 0x15e): lane i converts one binary32 value into byte (i & 3) of 0x11223344.
; Inputs by lane: 12.5, 13.5, 0.5, 254.5, 12.75, NaN (0x7fc00000), 300.0, -1.0.
; Measured natively on RDNA2 (gfx1036, Ryzen 7 9800X3D iGPU, AMD driver 25.10): halves round to even,
; NaN and negative values give 0, values above 255 give 255, and the other three bytes are kept.
; Run: vk_compute_test cs_cvt_pk_u8.spv 1 dump
; out[0..7] = 1122330C 11220E44 11003344 FE223344 1122330D 11220044 11FF3344 00223344 (hex)
v_mov_b32_e32 v1, 0x41480000
v_mov_b32_e32 v2, 0x41580000
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x3f000000
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x437e8000
v_cmp_eq_u32_e32 vcc_lo, 3, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x414c0000
v_cmp_eq_u32_e32 vcc_lo, 4, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x7fc00000
v_cmp_eq_u32_e32 vcc_lo, 5, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0x43960000
v_cmp_eq_u32_e32 vcc_lo, 6, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_mov_b32_e32 v2, 0xbf800000
v_cmp_eq_u32_e32 vcc_lo, 7, v0
v_cndmask_b32_e32 v1, v1, v2, vcc_lo
v_and_b32_e32 v3, 3, v0
v_mov_b32_e32 v4, 0x11223344
v_cvt_pk_u8_f32 v5, v1, v3, v4
v_lshlrev_b32_e32 v6, 2, v0
buffer_store_dword v5, v6, s[0:3], 0 offen
s_endpgm
