; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Packed b16 shifts with op_sel: independent per-half shift amounts and source halves.
; Expected mrt0 bits: 0xf800ffff, 0x01000010, 0x00800800, 0x8000fff0
v_mov_b32_e32 v0, 0x00040004
v_mov_b32_e32 v1, 0x8000fff0
v_pk_ashrrev_i16 v2, v0, v1
v_mov_b32_e32 v3, 0x00040008
v_mov_b32_e32 v4, 0x12348001
v_pk_lshlrev_b16 v5, v3, v4 op_sel:[1,0,0] op_sel_hi:[0,0,0]
v_mov_b32_e32 v6, 0x00040008
v_mov_b32_e32 v7, 0xbeef8001
v_pk_lshrrev_b16 v8, v6, v7 op_sel:[1,0,0] op_sel_hi:[0,0,0]
v_mov_b32_e32 v9, 0
v_pk_ashrrev_i16 v10, v9, v1
exp mrt0 v2, v5, v8, v10 done vm
s_endpgm
