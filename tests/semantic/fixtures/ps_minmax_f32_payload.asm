; Read two runtime values, then run each binary min/max with both operand orders.
v_mov_b32_e32 v4, s0
v_mov_b32_e32 v5, s1
v_min_f32_e32 v0, v4, v5
v_max_f32_e32 v1, v4, v5
v_min_f32_e64 v2, v5, v4
v_max_f32_e64 v3, v5, v4
; A quiet NaN becomes a finite color near +/-0.5, retaining all payload bits.
v_xor_b32_e32 v0, 0x40c00000, v0
v_xor_b32_e32 v1, 0x40c00000, v1
v_xor_b32_e32 v2, 0x40c00000, v2
v_xor_b32_e32 v3, 0x40c00000, v3
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
