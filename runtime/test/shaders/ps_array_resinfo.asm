; vk_texture_test exposes an 8x8 array view of two cubes (12 layers) with one mip.
; The 2D image at the same descriptor index has one layer, so selecting it changes B.
s_load_dwordx8 s[8:15], s[0:1], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, 0
image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
v_cvt_f32_u32_e32 v4, v4
v_cvt_f32_u32_e32 v5, v5
v_cvt_f32_u32_e32 v6, v6
v_cvt_f32_u32_e32 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
