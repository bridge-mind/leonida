; v_med3_f32 of (2.0, 0.5, 1.5) yields 1.5. Each channel uses a different output modifier.
; Expected mrt0 bits: 0x3f800000, 0x40400000, 0x40c00000, 0x3f400000 (1, 3, 6, 0.75).
; Alpha divides by two before clamping. Reversing that order gives 0.5.
v_mov_b32 v10, 0x40000000
v_mov_b32 v11, 0x3f000000
v_mov_b32 v12, 0x3fc00000
v_med3_f32 v0, v10, v11, v12 clamp
v_med3_f32 v1, v10, v11, v12 mul:2
v_med3_f32 v2, v10, v11, v12 mul:4
v_med3_f32 v3, v10, v11, v12 clamp div:2
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
