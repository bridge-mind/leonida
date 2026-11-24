; image_sample_b_cl: as ps_texture_cl with a LOD bias of 0.0 in front of the coordinates (address v[3:6] = bias,
; u / 4, v / 4, clamp), so the bias and the clamp each have to come from their own VGPR. Left half mip 0 (red), right
; half mip 1 (cyan). A bias read from the clamp's register (1.0) would still give cyan on the right but also move the
; left half to mip 1; the clamp read as the bias would never clamp.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_mul_f32_e32 v4, 0.25, v2
v_mul_f32_e32 v5, 0.25, v3
v_mov_b32_e32 v3, 0
v_cndmask_b32_e64 v6, 0, 1.0, vcc_lo
image_sample_b_cl v[8:11], v[3:6], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
