; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Plain 8/16-bit MUBUF stores (#1864): buffer_store_byte / buffer_store_short, the forms docs/embedded_shaders.md names
; as Box16Rehydration's blocker (cs_w32_buffer_sub16 covers the d16 forms; @YaBulochka on #1864). On the output V#
; (base 0, num_records 0xFFFFFFFF) lane 0 seeds dwords 0x40..0x54, stores into them and reads them back. A store writes
; only its own bytes; a short at byte 3 of a dword writes its high byte into the next one. Eight dwords:
;   out[0] = dword 0x40 (0x11223344) after buffer_store_byte 0xAB to byte 0x41                   = 0x1122AB44
;   out[1] = dword 0x44 (0x55667788) after buffer_store_short 0xCDEF to byte 0x46                 = 0xCDEF7788
;   out[2] = dword 0x48 (0x99AABBCC) after buffer_store_short 0x1234 to byte 0x4B: its low byte   = 0x34AABBCC
;   out[3] = dword 0x4C (0xDDDDDDDD) after the same store: the high byte crossed into byte 0x4C   = 0xDDDDDD12
;   out[4] = dword 0x50 (0x01020304) after buffer_store_byte of 0xFFFFFF5A to byte 0x50 (low byte) = 0x0102035A
;   out[5] = dword 0x54 (0x05060708) after buffer_store_short of 0xABCD1234 to byte 0x54 (low half) = 0x05061234
;   out[6] = buffer_load_ubyte of byte 0x41 afterwards                                           = 0x000000AB
;   out[7] = buffer_load_sshort of bytes 0x46..0x47 afterwards (0xCDEF, sign-extended)            = 0xFFFFCDEF
; EXEC is cut to lane 0 first (one address per op).
; group: 32,1,1
; Run: vk_compute_test cs_w32_buffer_store_sub.spv 1 dump -> out[0..7] = 287484740 3455022984 883604428 3722304786 16909146 84283956 171 4294954479
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0x40
v_mov_b32_e32 v21, 0x11223344
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x44
v_mov_b32_e32 v21, 0x55667788
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x48
v_mov_b32_e32 v21, 0x99aabbcc
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x4c
v_mov_b32_e32 v21, 0xdddddddd
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x50
v_mov_b32_e32 v21, 0x1020304
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x54
v_mov_b32_e32 v21, 0x5060708
buffer_store_dword v21, v20, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v20, 0x41
v_mov_b32_e32 v21, 0xab
buffer_store_byte v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x46
v_mov_b32_e32 v21, 0xcdef
buffer_store_short v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x4b
v_mov_b32_e32 v21, 0x1234
buffer_store_short v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x50
v_mov_b32_e32 v21, 0xffffff5a
buffer_store_byte v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x54
v_mov_b32_e32 v21, 0xabcd1234
buffer_store_short v21, v20, s[0:3], 0 offen
s_waitcnt_vscnt null, 0
v_mov_b32_e32 v20, 0x40
buffer_load_dword v10, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x44
buffer_load_dword v11, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x48
buffer_load_dword v12, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x4c
buffer_load_dword v13, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x50
buffer_load_dword v14, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x54
buffer_load_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x41
buffer_load_ubyte v16, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 0x46
buffer_load_sshort v17, v20, s[0:3], 0 offen
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
