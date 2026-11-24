; Texel offsets on the image_sample*_o forms ps_texoff does not run (SHADER-337): image_sample_o (implicit LOD),
; image_sample_b_o (LOD bias) and image_gather4_l_o (explicit LOD). docs/design.md, "Texel offsets": the first address
; dword packs signed 6-bit offsets, x = [5:0], y = [13:8]; the bias (_b) or the LOD (_l) follows the coordinates as
; the MIMG address order puts them: _b_o = {offset, bias, s, t}, _l_o = {offset, s, t, lod}.
; Run: vk_texture_test <vs> ps_texoff_forms.spv out.raw 2 92 0 0 0   (VKT_NEED_MAINTENANCE8=1: exit 77 without it)
; Same set-up as ps_texoff.asm: an 8x8 RGBA8 texture whose texel (x, y) is r = 32x/255, g = 32y/255, one mip, a
; point sampler that clamps to the edge (S# 0x92), 64x64 pixels with attr0.xy = ((px + 0.5) / 64, (py + 0.5) / 64).
; Each sampled channel c reads back as its texel coordinate through rndne(c * 255/32).
;
; Every sample uses the per-pixel offset of ps_texoff's .x: ox = bx - 4, oy = by - 4 (bx = px >> 3, by = py >> 3),
; all 64 pairs in -4..3, none clamped by the sample. Expected mrt0 words, derived from the rule and the texture:
;   .x  image_sample_o at the centre of texel (4, 4); the coordinates are the same in every pixel, so the implicit
;       derivatives are 0 and the one mip is sampled: texel (bx, by), word = bx | by << 4.
;   .y  image_sample_b_o with bias 0.0 at the same point: word = bx | by << 4.
;   .z  image_gather4_l_o of red (dmask 1, LOD 0.0) at uv (0.5, 0.5), the corner of texels 3..4: X0 = max(bx - 1, 0),
;       X1 = bx; gather order (i0, j1), (i1, j1), (i1, j0), (i0, j0): word = X0 | X1 << 4 | X1 << 8 | X0 << 12.
;   .w  the same gather of green: word = Y1 | Y1 << 4 | Y0 << 8 | Y0 << 12 (Y0 = max(by - 1, 0), Y1 = by).
; .z and .w are ps_texoff's gather words (check.py texoff_words). Dropping the offset gives 0x44 in .x and .y; x and y
; swapped give by | bx << 4.
s_load_dwordx8 s[8:15], s[2:3], 0
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x41000000, v2
v_mul_f32_e32 v3, 0x41000000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_add_nc_u32_e32 v2, -4, v2
v_add_nc_u32_e32 v3, -4, v3
v_and_b32_e32 v2, 63, v2
v_and_b32_e32 v3, 63, v3
v_lshl_or_b32 v10, v3, 8, v2
v_mov_b32_e32 v11, 0x3f100000
v_mov_b32_e32 v12, 0x3f100000
s_waitcnt lgkmcnt(0)
image_sample_o v[4:5], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v24, v10
v_mov_b32_e32 v25, 0
v_mov_b32_e32 v26, 0x3f100000
v_mov_b32_e32 v27, 0x3f100000
image_sample_b_o v[6:7], v[24:27], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v13, v10
v_mov_b32_e32 v14, 0.5
v_mov_b32_e32 v15, 0.5
v_mov_b32_e32 v28, 0
v_mov_b32_e32 v29, v10
v_mov_b32_e32 v30, 0.5
v_mov_b32_e32 v31, 0.5
v_mov_b32_e32 v32, 0
image_gather4_l_o v[16:19], v[29:32], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[20:23], v[29:32], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32_e32 v4, 0x40ff0000, v4
v_rndne_f32_e32 v4, v4
v_cvt_u32_f32_e32 v4, v4
v_mul_f32_e32 v5, 0x40ff0000, v5
v_rndne_f32_e32 v5, v5
v_cvt_u32_f32_e32 v5, v5
v_mul_f32_e32 v6, 0x40ff0000, v6
v_rndne_f32_e32 v6, v6
v_cvt_u32_f32_e32 v6, v6
v_mul_f32_e32 v7, 0x40ff0000, v7
v_rndne_f32_e32 v7, v7
v_cvt_u32_f32_e32 v7, v7
v_mul_f32_e32 v16, 0x40ff0000, v16
v_rndne_f32_e32 v16, v16
v_cvt_u32_f32_e32 v16, v16
v_mul_f32_e32 v17, 0x40ff0000, v17
v_rndne_f32_e32 v17, v17
v_cvt_u32_f32_e32 v17, v17
v_mul_f32_e32 v18, 0x40ff0000, v18
v_rndne_f32_e32 v18, v18
v_cvt_u32_f32_e32 v18, v18
v_mul_f32_e32 v19, 0x40ff0000, v19
v_rndne_f32_e32 v19, v19
v_cvt_u32_f32_e32 v19, v19
v_mul_f32_e32 v20, 0x40ff0000, v20
v_rndne_f32_e32 v20, v20
v_cvt_u32_f32_e32 v20, v20
v_mul_f32_e32 v21, 0x40ff0000, v21
v_rndne_f32_e32 v21, v21
v_cvt_u32_f32_e32 v21, v21
v_mul_f32_e32 v22, 0x40ff0000, v22
v_rndne_f32_e32 v22, v22
v_cvt_u32_f32_e32 v22, v22
v_mul_f32_e32 v23, 0x40ff0000, v23
v_rndne_f32_e32 v23, v23
v_cvt_u32_f32_e32 v23, v23
v_lshl_or_b32 v40, v5, 4, v4
v_lshl_or_b32 v41, v7, 4, v6
v_lshl_or_b32 v42, v17, 4, v16
v_lshl_or_b32 v42, v18, 8, v42
v_lshl_or_b32 v42, v19, 12, v42
v_lshl_or_b32 v43, v21, 4, v20
v_lshl_or_b32 v43, v22, 8, v43
v_lshl_or_b32 v43, v23, 12, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
