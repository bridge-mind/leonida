; v_mul_legacy_f32 in the inferred f32 denorm-mode 0 (flush input and output).
; The two input denormals become signed zero, then DX9 0*x overrides the sign to +0.
; A finite, nonzero product that underflows keeps its sign when flushed.
; Expected mrt0 bits: 0x00000000 0x00000000 0x00000000 0x80000000
v_mov_b32 v0, 0x00000001
v_mov_b32 v1, 0x4b800000
v_mul_legacy_f32 v4, v0, v1
v_mov_b32 v0, 0x80000001
v_mul_legacy_f32 v5, v0, v1
v_mov_b32 v0, 0x00800000
v_mov_b32 v1, 0x3f000000
v_mul_legacy_f32 v6, v0, v1
v_mov_b32 v0, 0x80800000
v_mul_legacy_f32 v7, v0, v1
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
