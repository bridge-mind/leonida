; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_pk_lshlrev_b16 / v_pk_lshrrev_b16 (per-half shift amounts), v_add_nc_u16 (keeps dst high half), v_sin_f16 (turns).
; Expected mrt0 bits: 0x34000010, 0x00120800, 0xbeef0000, 0x12343c00
v_mov_b32_e32 v0, 0x12348001
v_mov_b32_e32 v2, 0x80004
v_pk_lshlrev_b16 v1, v2, v0
v_pk_lshrrev_b16 v3, v2, v0
v_mov_b32_e32 v4, 0xbeef0000
v_mov_b32_e32 v5, 0xaaaa0001
v_mov_b32_e32 v6, 0xffff
v_add_nc_u16 v4, v5, v6
v_mov_b32_e32 v7, 0x12340000
v_mov_b32_e32 v8, 0x3400
v_sin_f16_e32 v7, v8
exp mrt0 v1, v3, v4, v7 done vm
s_endpgm
