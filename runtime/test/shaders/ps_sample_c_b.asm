; Depth-compare sampling with a LOD bias: image_sample_c_b, as ps_sample_c with the bias 1.0 in front of the z-compare
; (address v[4:7] = bias, z-compare 0.5, u, v). A bias read as the reference (1.0 compares false everywhere) or the
; reference as the bias would change every result; on the one-level texture the bias itself has no other effect.
; mrt0 = (f32 result, widened d16 result, u, v).
; Run: vk_texture_test dref_c_b -> every pixel's first two components are 1.0 where its texel has x >= 4, else 0.0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 0.5
v_mov_b32_e32 v6, v2
v_mov_b32_e32 v7, v3
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x3092
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample_c_b v9, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_b v10, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v11, v10
exp mrt0 v9, v11, v2, v3 done vm
s_endpgm
