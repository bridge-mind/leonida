; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-85: ds_read_u8_d16 / ds_read_u8_d16_hi / ds_read_i8_d16 / ds_read_i8_d16_hi (RDNA2 ISA, DS opcodes 162-165).
; An 8-bit LDS read lands in ONE 16-bit half of the destination VGPR and the other half is kept:
;   u8_d16    D[15:0]  = zext16(byte)   D[31:16] kept     u8_d16_hi  D[31:16] = zext16(byte)  D[15:0] kept
;   i8_d16    D[15:0]  = sext16(byte)   D[31:16] kept     i8_d16_hi  D[31:16] = sext16(byte)  D[15:0] kept
; LDS dword 0 = 0x7F80FF01, so the bytes at byte address 0..3 are 01 FF 80 7F (512 B LDS from the default header).
; Every destination starts as 0xBEEF1234; the address is v1 (0, or 1 for the last read) plus the offset: field.
;   out[0] u8_d16    offset:1 (0xFF) -> 0xBEEF00FF   out[1] u8_d16_hi offset:1 (0xFF) -> 0x00FF1234
;   out[2] i8_d16    offset:2 (0x80) -> 0xBEEFFF80   out[3] i8_d16_hi offset:2 (0x80) -> 0xFF801234
;   out[4] i8_d16    offset:3 (0x7F) -> 0xBEEF007F   out[5] i8_d16_hi offset:3 (0x7F) -> 0x007F1234
;   out[6] u8_d16    offset:2 (0x80) -> 0xBEEF0080   out[7] i8_d16_hi v1=1, no offset (byte 1, 0xFF) -> 0xFFFF1234
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_u8_d16.spv 1 dump
;   -> out[0..7] = 0xBEEF00FF 0x00FF1234 0xBEEFFF80 0xFF801234 0xBEEF007F 0x007F1234 0xBEEF0080 0xFFFF1234
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0x7f80ff01
ds_write_b32 v2, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v10, 0xbeef1234
v_mov_b32_e32 v11, 0xbeef1234
v_mov_b32_e32 v12, 0xbeef1234
v_mov_b32_e32 v13, 0xbeef1234
v_mov_b32_e32 v14, 0xbeef1234
v_mov_b32_e32 v15, 0xbeef1234
v_mov_b32_e32 v16, 0xbeef1234
v_mov_b32_e32 v17, 0xbeef1234
ds_read_u8_d16 v10, v1 offset:1
ds_read_u8_d16_hi v11, v1 offset:1
ds_read_i8_d16 v12, v1 offset:2
ds_read_i8_d16_hi v13, v1 offset:2
ds_read_i8_d16 v14, v1 offset:3
ds_read_i8_d16_hi v15, v1 offset:3
ds_read_u8_d16 v16, v1 offset:2
v_mov_b32_e32 v1, 1
ds_read_i8_d16_hi v17, v1
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
