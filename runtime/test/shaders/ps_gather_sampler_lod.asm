; In --two-mip mode the 128x128 base level has four colored quadrants and mip 1 is opaque cyan.
; Request LOD U across the draw. With max LOD 0.25 every pixel must use the base;
; with min LOD 0.75 every pixel must use mip 1. The misplaced half of the frame reveals a dropped bound.
; Each 2x2 gather footprint stays within its quadrant. Export the first tap of R, G, B and A.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, v2
v_mov_b32_e32 v1, 0 ; zero offset for the _l_o path used by the alpha gather
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
image_gather4_l v[8:11], v[2:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l v[12:15], v[2:4], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
image_gather4_l v[16:19], v[2:4], s[8:15], s[16:19] dmask:0x4 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[20:23], v[1:4], s[8:15], s[16:19] dmask:0x8 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v8, v12, v16, v20 done vm
s_endpgm
