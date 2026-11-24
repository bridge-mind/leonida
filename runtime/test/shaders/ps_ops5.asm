; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Scalar b16 shifts (VOP3, high half kept) and v_cos_f16 (turns).
; Expected mrt0 bits: 0x12340010, 0xbeef0800, 0xaaaaffff, 0xdeadbc00
v_mov_b32_e32 v0, 0x12348001
v_mov_b32_e32 v1, 0x12340000
v_mov_b32_e32 v2, 4
v_lshlrev_b16_e64 v1, v2, v0
v_mov_b32_e32 v3, 0xbeef8001
v_mov_b32_e32 v4, 0xbeef0000
v_lshrrev_b16_e64 v4, v2, v3
v_mov_b32_e32 v5, 0xfff0
v_mov_b32_e32 v6, 4
v_mov_b32_e32 v7, 0xaaaa0000
v_ashrrev_i16_e64 v7, v6, v5
v_mov_b32_e32 v8, 0x3800
v_mov_b32_e32 v9, 0xdead0000
v_cos_f16_e32 v9, v8
exp mrt0 v1, v4, v7, v9 done vm
s_endpgm
