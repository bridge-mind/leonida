; Explicit mip-zero sampling, paired with the implicit ps_texture_sample test.
; vk_texture_test --two-mip draws 64x64 pixels from a 128x128 base level and 64x64 mip 1.
; This shader must yield red, green, blue and transparent yellow in the four base-level quadrants.
; The implicit pass must yield opaque cyan across mip 1.
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
image_sample_lz v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
