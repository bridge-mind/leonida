; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_write_b8 / ds_write_b8_d16_hi / ds_write_b16 / ds_write_b16_d16_hi and ds_read_u8 / ds_read_i8 / ds_read_u16 /
; ds_read_i16 (RDNA2 ISA, "Data Share Instructions"). ADDR = vaddr + offset: (bytes).
;   ds_write_b8   MEM[ADDR][7:0]  = DATA[7:0]       ds_write_b8_d16_hi   MEM[ADDR][7:0]  = DATA[23:16]
;   ds_write_b16  MEM[ADDR][15:0] = DATA[15:0]      ds_write_b16_d16_hi  MEM[ADDR][15:0] = DATA[31:16]
;   ds_read_u8/i8   D = zero/sign-extend 8 -> 32 of MEM[ADDR][7:0]
;   ds_read_u16/i16 D = zero/sign-extend 16 -> 32 of MEM[ADDR][15:0]
; LDS dword 0 = 0x11223344, dword 1 = 0x55667788; DATA = 0xA1B2C3D4. The narrow writes give
;   dword 0 = 0x11B2D444 (byte 1 = 0xD4 from b8, byte 2 = 0xB2 from b8_d16_hi; bytes 0 and 3 kept)
;   dword 1 = 0xA1B2C3D4 (low half 0xC3D4 from b16, high half 0xA1B2 from b16_d16_hi)
;   out[0] ds_read_b32  offset:0 -> 0x11B2D444   out[1] ds_read_b32  offset:4 -> 0xA1B2C3D4
;   out[2] ds_read_i8   offset:2 -> 0xFFFFFFB2   out[3] ds_read_u8   offset:2 -> 0x000000B2
;   out[4] ds_read_i8   offset:3 -> 0x00000011   out[5] ds_read_i16  offset:6 -> 0xFFFFA1B2
;   out[6] ds_read_u16  offset:6 -> 0x0000A1B2   out[7] ds_read_i16  offset:4 -> 0xFFFFC3D4
; Every lane writes the same bytes, so the result does not depend on lane order.
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_subdword.spv 1 dump
;   -> out[0..7] = 0x11B2D444 0xA1B2C3D4 0xFFFFFFB2 0x000000B2 0x00000011 0xFFFFA1B2 0x0000A1B2 0xFFFFC3D4
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v4, 0x11223344
v_mov_b32_e32 v5, 0x55667788
ds_write_b32 v2, v4
ds_write_b32 v2, v5 offset:4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v3, 0xa1b2c3d4
ds_write_b8 v2, v3 offset:1
ds_write_b8_d16_hi v2, v3 offset:2
ds_write_b16 v2, v3 offset:4
ds_write_b16_d16_hi v2, v3 offset:6
s_waitcnt lgkmcnt(0)
ds_read_b32 v10, v2
ds_read_b32 v11, v2 offset:4
ds_read_i8 v12, v2 offset:2
ds_read_u8 v13, v2 offset:2
ds_read_i8 v14, v2 offset:3
ds_read_i16 v15, v2 offset:6
ds_read_u16 v16, v2 offset:6
ds_read_i16 v17, v2 offset:4
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
