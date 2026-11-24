; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mul_hi_u32 / v_mul_hi_i32, v_bcnt_u32_b32 with an accumulator, v_ffbl_b32 / v_ffbh_u32 (and v_ffbh_u32 of 0
; = 0xffffffff), v_xad_u32 and v_add3_u32 wrapping.
; Expected mrt0 bits: 0x80000000, 0xbfffffff, 0x000f100e, 0x0f0f0f1f
; XOR bit 30 before export to encode the integer result as a finite RGBA32F value.
v_mov_b32_e32 v0, 0x80000001
v_mov_b32_e32 v1, -1
v_mul_hi_u32 v2, v0, v1
v_mov_b32_e32 v3, -2
v_mov_b32_e32 v4, 0x40000000
v_mul_hi_i32 v5, v3, v4
v_mov_b32_e32 v6, 0xf0f00001
v_bcnt_u32_b32 v7, v6, 5
v_mov_b32_e32 v8, 0x00010000
v_ffbl_b32_e32 v9, v8
v_ffbh_u32_e32 v10, v8
v_lshl_add_u32 v11, v9, 8, v7
v_lshl_add_u32 v11, v10, 16, v11
v_mov_b32_e32 v12, 0xf0f0f0f0
v_xad_u32 v13, v12, -1, 1
v_mov_b32_e32 v14, 0
v_ffbh_u32_e32 v15, v14
v_add3_u32 v16, v13, 16, v15
v_xor_b32_e32 v5, 0x40000000, v5
exp mrt0 v2, v5, v11, v16 done vm
s_endpgm
