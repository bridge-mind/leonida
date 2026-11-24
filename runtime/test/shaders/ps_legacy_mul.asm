; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mul_legacy_f32: 0 * anything (inf, -inf, NaN) == 0, finite * finite == normal multiply.
; Expected mrt0 bits: 0, 0, 0, 0x40c00000
v_mov_b32 v0, 0
v_mov_b32 v1, 0x80000000
v_mov_b32 v2, 0x7f800000
v_mov_b32 v3, 0xff800000
v_mov_b32 v4, 0x7fc00000
v_mov_b32 v5, 2.0
v_mov_b32 v6, 3.0
v_mul_legacy_f32 v7, v0, v2
v_mul_legacy_f32 v8, v0, v4
v_mul_legacy_f32 v9, v1, v3
v_mul_legacy_f32 v10, v5, v6
exp mrt0 v7, v8, v9, v10 done vm
s_endpgm
