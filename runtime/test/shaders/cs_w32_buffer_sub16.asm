; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 8/16-bit MUBUF d16 loads and stores (#1864, BUFFER_SUB16), on the output V# (base 0, num_records 0xFFFFFFFF). Lane 0
; seeds dword 0x40 = 0x8899AABB, 0x50 = 0x11223344, 0x60 = 0x55667788, and loads into a register holding 0xBEEF1234, so
; each d16 load shows the half it writes and the half it keeps. Eight dwords:
;   out[0] = buffer_load_ubyte_d16 of byte 0x41 (0xAA): low half 0x00AA, high kept       = 0xBEEF00AA
;   out[1] = buffer_load_ubyte_d16_hi of byte 0x41: high half 0x00AA, low kept            = 0x00AA1234
;   out[2] = buffer_load_sbyte_d16 of byte 0x41: 0xAA sign-extended to 16 bits            = 0xBEEFFFAA
;   out[3] = buffer_load_sbyte_d16_hi of byte 0x41                                        = 0xFFAA1234
;   out[4] = buffer_load_short_d16 of bytes 0x42..0x43 (0x8899)                           = 0xBEEF8899
;   out[5] = buffer_load_short_d16_hi of bytes 0x42..0x43                                 = 0x88991234
;   out[6] = dword 0x50 after buffer_store_byte_d16_hi of 0x00CD0000 to byte 0x51: bits 23:16 = 0x1122CD44
;   out[7] = dword 0x60 after buffer_store_short_d16_hi of 0xABCD0000 to byte 0x62: bits 31:16 = 0xABCD7788
; EXEC is cut to lane 0 first (one address per op).
; group: 32,1,1
; Run: vk_compute_test cs_w32_buffer_sub16.spv 1 dump -> out[0..7] = 3203334314 11145780 3203399594 4289335860 3203369113 2291733044 287493444 2882369416
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v21, 0x8899aabb
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x50
v_mov_b32_e32 v21, 0x11223344
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x60
v_mov_b32_e32 v21, 0x55667788
buffer_store_dword v21, v20, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v1, 0x41
v_mov_b32_e32 v2, 0x42
v_mov_b32_e32 v10, 0xbeef1234
v_mov_b32_e32 v11, 0xbeef1234
v_mov_b32_e32 v12, 0xbeef1234
v_mov_b32_e32 v13, 0xbeef1234
v_mov_b32_e32 v14, 0xbeef1234
v_mov_b32_e32 v15, 0xbeef1234
buffer_load_ubyte_d16 v10, v1, s[0:3], 0 offen
buffer_load_ubyte_d16_hi v11, v1, s[0:3], 0 offen
buffer_load_sbyte_d16 v12, v1, s[0:3], 0 offen
buffer_load_sbyte_d16_hi v13, v1, s[0:3], 0 offen
buffer_load_short_d16 v14, v2, s[0:3], 0 offen
buffer_load_short_d16_hi v15, v2, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x51
v_mov_b32_e32 v21, 0xcd0000
buffer_store_byte_d16_hi v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x62
v_mov_b32_e32 v21, 0xabcd0000
buffer_store_short_d16_hi v21, v20, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v20, 0x50
buffer_load_dword v16, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x60
buffer_load_dword v17, v20, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v20, 0
buffer_store_dword v10, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v11, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v12, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v14, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_store_dword v16, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
buffer_store_dword v17, v20, s[0:3], 0 offen
s_endpgm
