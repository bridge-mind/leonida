; Depth-compare and clamped/biased image_gather4 variants on the 8x8 RGBA gradient texture (R = 32x/255, G = 32y/255).
; Run: vk_texture_test <vs_ids.spv> <ps_gather4_c_l.spv> out.raw 0 92 c00000 14500000 0 -> RGBA32F pixels, all identical.
; At u = v = 0.5 the 2x2 gather footprint is (3,4) (4,4) (4,3) (3,3) in gather order.
;   r: image_gather4_c_l, literal S# DEPTH_COMPARE_FUNC=1 (LESS, 0x1092), ref=0.45:
;      R = (3,4,4,3)*32/255 = (0.3765,0.5020,0.5020,0.3765); 0.45 < R -> (0,1,1,0) -> base 2 = 6.0
;   g: image_gather4_c_l_o, run-time S# DEPTH_COMPARE_FUNC=6 (GREATER_EQUAL, 0x6092), offset x=+2, ref=0.7:
;      R = (5,6,6,5)*32/255 = (0.6275,0.7529,0.7529,0.6275); 0.7 >= R -> (1,0,0,1) -> base 2 = 9.0
;   b: image_gather4_cl, dmask:0x2 (G), clamp=0.0:
;      G = (4,4,3,3)*32/255 -> scaled by 255/32 and rounded -> (4,4,3,3) -> base 16 = 17459.0
;   a: image_gather4_c_b_cl_o d16, literal S# DEPTH_COMPARE_FUNC=4 (GREATER, 0x4092), offset x=-1, bias=0, ref=0.3, clamp=0:
;      R = (2,3,3,2)*32/255 = (0.2510,0.3765,0.3765,0.2510); 0.3 > R -> (1,0,0,1) packed f16 -> base 2 = 9.0
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
; r: image_gather4_c_l (literal S#, LESS)
s_mov_b32 s16, 0x1092
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_mov_b32_e32 v20, 0x3ee66666
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0.5
v_mov_b32_e32 v23, 0
image_gather4_c_l v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_fma_f32 v16, v8, 2.0, v9
v_fma_f32 v16, v16, 2.0, v10
v_fma_f32 v4, v16, 2.0, v11
; g: image_gather4_c_l_o (run-time S# word 0 loaded from descriptor heap, GREATER_EQUAL, offset x=+2)
s_load_dword s16, s[0:1], 0x20
s_waitcnt lgkmcnt(0)
s_or_b32 s16, s16, 0x6000
v_mov_b32_e32 v19, 2
v_mov_b32_e32 v20, 0x3f333333
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0.5
v_mov_b32_e32 v23, 0
image_gather4_c_l_o v[8:11], v[19:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_fma_f32 v16, v8, 2.0, v9
v_fma_f32 v16, v16, 2.0, v10
v_fma_f32 v5, v16, 2.0, v11
; b: image_gather4_cl (dmask:0x2 G channel, clamp=0.0)
s_mov_b32 s16, 0x92
v_mov_b32_e32 v20, 0.5
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0
image_gather4_cl v[8:11], v[20:22], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v10, 0x40ff0000, v10
v_rndne_f32 v10, v10
v_mul_f32 v11, 0x40ff0000, v11
v_rndne_f32 v11, v11
v_fma_f32 v16, v8, 0x41800000, v9
v_fma_f32 v16, v16, 0x41800000, v10
v_fma_f32 v6, v16, 0x41800000, v11
; a: image_gather4_c_b_cl_o d16 (literal S#, GREATER, offset x=-1, bias=0, ref=0.3, clamp=0)
s_mov_b32 s16, 0x4092
v_mov_b32_e32 v18, 0x3f
v_mov_b32_e32 v19, 0
v_mov_b32_e32 v20, 0x3e99999a
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0.5
v_mov_b32_e32 v23, 0
image_gather4_c_b_cl_o v[8:9], v[18:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v12, v8
v_lshrrev_b32_e32 v8, 16, v8
v_cvt_f32_f16_e32 v13, v8
v_cvt_f32_f16_e32 v14, v9
v_lshrrev_b32_e32 v9, 16, v9
v_cvt_f32_f16_e32 v15, v9
v_fma_f32 v16, v12, 2.0, v13
v_fma_f32 v16, v16, 2.0, v14
v_fma_f32 v7, v16, 2.0, v15
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
