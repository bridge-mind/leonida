; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 10:10:10:2 typed buffers. Format names list the channel widths from the top bit: 2_10_10_10 (50) is R10G10B10A2
; with x in bits 9:0, and 10_10_10_2 (44) has a 2-bit x in bits 1:0. out[12..13] = 0xc00ffc00 0x003ff003 (raw),
; out[0..3] = out[12] read as 50, out[4..7] = out[13] read as 44, out[8..11] = out[12] read by tbuffer as 50 through
; a V# that says 44, out[14..15] = (1, 0, 0.5, 1) stored as 50 and 44 = 0xe00003ff 0xffe00003.
; Run: vk_compute_test cs_fmt_1010102.spv 1 dump -> out[0..7] = 0 0x3f800000 0 0x3f800000 0x3f800000 0 0x3f800000 0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0xc00ffc00
v_mov_b32_e32 v2, 0x3ff003
v_mov_b32_e32 v3, 48
buffer_store_dwordx2 v[1:2], v3, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, s2
s_mov_b32 s7, 0x31032fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
s_mov_b32 s7, 0x3102cfac
v_mov_b32_e32 v9, 52
buffer_load_format_xyzw v[4:7], v9, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 16
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
tbuffer_load_format_xyzw v[4:7], v3, s[4:7], 0 format:[BUF_FMT_2_10_10_10_UNORM] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 32
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v10, 1.0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0.5
v_mov_b32_e32 v13, 1.0
s_mov_b32 s7, 0x31032fac
v_mov_b32_e32 v9, 56
buffer_store_format_xyzw v[10:13], v9, s[4:7], 0 offen
s_mov_b32 s7, 0x3102cfac
v_mov_b32_e32 v9, 60
buffer_store_format_xyzw v[10:13], v9, s[4:7], 0 offen
s_endpgm
