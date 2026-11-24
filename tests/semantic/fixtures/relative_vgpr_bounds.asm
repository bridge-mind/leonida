; The synthetic VGPR file has 256 entries. An out-of-range read uses v0.
; An out-of-range write must preserve v0; a swap with either destination invalid is a no-op.
v_mov_b32 v0, 0x3f800000
v_mov_b32 v1, 0x40000000
v_mov_b32 v4, 0x40800000
v_mov_b32 v255, 0x41100000
s_mov_b32 m0, 5
v_movrels_b32 v6, v250
s_mov_b32 m0, 6
v_movrels_b32 v7, v250
v_movreld_b32 v250, v1
v_movrelsd_b32 v251, v249
s_mov_b32 m0, 0x106
v_movrelsd_2_b32 v8, v1
s_mov_b32 m0, 0x1000000
v_swaprel_b32 v0, v4
s_mov_b32 m0, 0x106
v_swaprel_b32 v4, v0
exp mrt0 v6, v7, v8, v4 done vm
s_endpgm
