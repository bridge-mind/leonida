; image_get_lod on a 2D array (dim:SQ_RSRC_IMG_2D_ARRAY: s, t and the slice; ps_get_lod.asm has the plain 2D form and
; the rules). The slice takes no part in the LOD.
; Run: vk_texture_test vs_ids.spv ps_get_lod_array.spv out.raw 0 92 c00000 14500000 0
; The tool's 2D array (set 0 binding 3) is 8x8 with one level, drawn over 64x64 pixels with u and v from 0 to 1: a
; pixel step is 1/8 texel, so the calculated LOD is log2(1/8) = -3. The first query scales u by 32: four texels a
; pixel along x, log2(4) = 2. With a single level the clamped LOD is 0 for both.
;   mrt0 = (clamped, calculated LOD at (32u, v); clamped, calculated LOD at (u, v)) = 0.0 2.0 0.0 -3.0
; on every pixel, within 1/16 (a device may keep as few as four fraction bits of LOD).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v4, 2.0
s_load_dwordx8 s[8:15], s[0:1], 0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mul_f32_e32 v8, 0x42000000, v2
v_mov_b32_e32 v9, v3
v_mov_b32_e32 v10, v4
image_get_lod v[12:13], v[8:10], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY
image_get_lod v[14:15], v[2:4], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
