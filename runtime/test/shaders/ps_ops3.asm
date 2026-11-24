; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mul_i32_i24, v_alignbyte_b32, v_xnor_b32, v_ffbh_i32, v_swap_b32. Expected mrt0 bits: 0xbffffffa, 0x44660066, 0x44aabbcc, 16
; XOR bit 30 before export to encode the integer result as a finite RGBA32F value.
v_mov_b32 v0, 0x00fffffe
v_mov_b32 v1, 3
v_mul_i32_i24_e32 v2, v0, v1
v_mov_b32 v3, 0x11223344
v_mov_b32 v4, 0xaabbccdd
v_alignbyte_b32 v5, v3, v4, 1
v_xnor_b32_e32 v6, v3, v4
v_mov_b32 v7, 0x0000ffff
v_ffbh_i32_e32 v8, v7
v_swap_b32 v5, v6
v_xor_b32_e32 v2, 0x40000000, v2
exp mrt0 v2, v5, v6, v8 done vm
s_endpgm
