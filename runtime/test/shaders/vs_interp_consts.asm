; Vertex shader for ps_interp_f32_mods and ps_interp_mov_mods: the vs_ids full-screen triangle, with the same constant
; in each param channel at all three vertices, so interpolation gives that constant at every pixel:
; param0 = (1.5, -0.25, 0.375, 0.625), param1 = (1.5, -0.5, 0.25, 4.0), param2 = (0.25, 3.0, 0, 0).
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
v_mov_b32_e32 v7, 0x3fc00000
v_mov_b32_e32 v8, 0xbe800000
v_mov_b32_e32 v9, 0x3ec00000
v_mov_b32_e32 v10, 0x3f200000
exp param0 v7, v8, v9, v10
v_mov_b32_e32 v8, -0.5
v_mov_b32_e32 v9, 0.25
v_mov_b32_e32 v10, 4.0
exp param1 v7, v8, v9, v10
v_mov_b32_e32 v7, 0.25
v_mov_b32_e32 v8, 0x40400000
exp param2 v7, v8, v5, v5
s_endpgm
