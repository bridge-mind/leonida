; Vertex shader for the pixel tests that need more than one input: a full-screen triangle from the vertex index
; (v0), param0 = (u, v, 0, 1) with u and v running 0..2 over the triangle (0..1 over the image), and
; param1 = the integer 0x101 + vertex index in all four components, as raw bits.
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
exp param0 v1, v2, v5, v6
v_add_nc_u32_e32 v7, 0x101, v0
exp param1 v7, v7, v7, v7
s_endpgm
