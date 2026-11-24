; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-46: s_ff1_i32_b32 / s_ff0_i32_b32 on an inline constant or literal read it as a value, not as a lane mask.
; Lowest set (ff1) or clear (ff0) bit of S0, -1 if none. 4 -> ff1 2; 61 = 0b111101 -> ff0 1; 0x00100000 -> ff1 20;
; 0x0000ffff -> ff0 16. Reading each source as a lane mask (all ones or zero from bit 0) gives -1 in every word.
; Expected mrt0 bits: 2, 1, 20, 16
s_ff1_i32_b32 s0, 4
s_ff0_i32_b32 s1, 61
s_ff1_i32_b32 s2, 0x00100000
s_ff0_i32_b32 s3, 0x0000ffff
v_mov_b32_e32 v0, s0
v_mov_b32_e32 v1, s1
v_mov_b32_e32 v2, s2
v_mov_b32_e32 v3, s3
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
