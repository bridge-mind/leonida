; group: 1,1,1
; Store binary64 (0.75 + 0.5) * 2 = 2.5 as low/high words: 0x00000000, 0x40040000.
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3fe80000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x3fe00000
v_add_f64 v[4:5], v[0:1], v[2:3] mul:2
v_mov_b32 v6, 0
buffer_store_dwordx2 v[4:5], v6, s[0:3], 0 offen
s_endpgm
