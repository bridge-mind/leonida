; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Packed f16 FMA, integer min3/max3 and signed bitfield extract. Expected mrt0 bits: 0x48804300, 5, 7, 0xbffffffc
; XOR bit 30 before export to encode the integer result as a finite RGBA32F value.
v_mov_b32 v0, 0x40003c00
v_mov_b32 v1, 0x44004200
v_mov_b32 v2, 0x3c003800
v_pk_fma_f16 v3, v0, v1, v2
v_mov_b32 v5, 7
v_mov_b32 v6, 0xfffffff0
v_mov_b32 v7, 5
v_min3_u32 v4, v5, v6, v7
v_max3_i32 v8, v5, v6, v7
v_bfe_i32 v9, v6, 2, 4
v_xor_b32_e32 v9, 0x40000000, v9
exp mrt0 v3, v4, v8, v9 done vm
s_endpgm
