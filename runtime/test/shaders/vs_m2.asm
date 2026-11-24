; M2 vertex shader: a quad from the vertex index, no vertex input. Run: d3d12_m2_textured_test (found by the SHA-256
; of its FSSM program, as vs_m15 is).
; v0 = VertexIndex, 0..3 (the test draws indices 0 1 2 2 1 3 with DrawIndexedInstanced and base vertex 0):
;   u = v0 & 1, v = (v0 >> 1) & 1;   pos0 = (2u - 1, 1 - 2v, 0, 1) (clip y up: v = 0 is the top row);   param0 = (u, v, 0, 1)
; fssm-at: 0x10
; fssm-data: 0x32534d56 0x20535620
v_and_b32_e32 v1, 1, v0
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_mul_f32_e32 v2, 0.5, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, -2.0, 1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
exp param0 v1, v2, v5, v6
s_endpgm
