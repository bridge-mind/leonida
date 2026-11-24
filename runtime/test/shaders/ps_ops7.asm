; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_perm_b32 (byte selects 0-7, 12 = 0x00, >= 13 = 0xff, 8-11 = sign of bytes 1/3/5/7 of {S0, S1}),
; v_alignbit_b32 (shift taken mod 32), v_med3_i32 vs v_med3_u32 on the same operands.
; Expected mrt0 bits: 0x00ff8077, 0xff0000ff, 0x89abcdef, 0x00070003
v_mov_b32_e32 v0, 0x80112233
v_mov_b32_e32 v1, 0x4455e677
v_mov_b32_e32 v2, 0x0c0d0700
v_perm_b32 v3, v0, v1, v2
v_mov_b32_e32 v2, 0x0b0a0908
v_perm_b32 v4, v0, v1, v2
v_mov_b32_e32 v5, 0x12345678
v_mov_b32_e32 v6, 0x9abcdef0
v_alignbit_b32 v7, v5, v6, 36
v_mov_b32_e32 v8, -5
v_med3_i32 v9, v8, 7, 3
v_med3_u32 v10, v8, 7, 3
v_lshl_add_u32 v11, v10, 16, v9
exp mrt0 v3, v4, v7, v11 done vm
s_endpgm
