; image_gather4_l_o under the S# border address modes (SHADER-348): 6 clamp border, 7 mirror once border. The 2x2 footprint
; is four OpImageFetch, so a tap outside the image takes the S# border colour's channel instead of a texel:
; BORDER_COLOR_TYPE = S# word 3 [31:30], 0 transparent black, 1 opaque black, 2 opaque white (Mesa gfx10-rsrc.json for
; the field, SQ_TEX_BORDER_COLOR_* in gfx6.json for the values). Word 3 keeps BORDER_COLOR_PTR = 0x100 so it stays an
; integer literal (0x80000000 alone is the inline float -2.0, which the translator reads at run time).
; Cases r and g use a literal S#; b and a add 0 to s16 and s19 first, so mode and border type are read at run time.
; Run: vk_texture_test <vs_ids.spv> <ps_gather_border.spv> out.raw 0 92 c00000 14500000 0 -> RGBA32F pixels, all the same.
; The harness image is the 8x8 gradient R = 32x/255, G = 32y/255 (as in ps_gather_wrap). u = v = 0.5 puts the footprint's
; lower corner at texel (3, 3), moved by the signed 6-bit texel offset in the first address dword (x [5:0], y [13:8]).
; Each tap is scaled by 255/32 and rounded, so a texel gives its column (R) or row (G) and a border of 1.0 gives 8; the
; four taps, in gather order (x0,y1) (x1,y1) (x1,y0) (x0,y0), are packed in base 16 (exact in f32).
;   r: modes 6,6, opaque white, R, offset x -4: x0 = -1 (out: 8), x1 = 0          -> (8,0,0,8) = 32776
;   g: modes 6,6, opaque white, G, offset y +4: y0 = 7, y1 = 8 (out: 8)            -> (8,8,7,7) = 34935
;   b: modes 7,7 at run time, opaque white, R, offset x -12: x0 = -9 mirrors to 8 (out: 8), x1 = -8 mirrors to 7
;                                                                                -> (8,7,7,8) = 34680
;   a: modes 6,6 at run time, transparent black, R, offset x +4: x0 = 7, x1 = 8 (out: 0) -> (7,0,0,7) = 28679
; Clamp to edge (the run-time treatment of modes 6 and 7 before) gives (0,0,0,0), (7,7,7,7), (0,0,0,0) and (7,7,7,7):
; every case differs.
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0.5
v_mov_b32_e32 v23, 0
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x36
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0x80000100
v_mov_b32 v20, 0x3c
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
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
v_fma_f32 v4, v16, 0x41800000, v11
s_mov_b32 s16, 0x36
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0x80000100
v_mov_b32 v20, 0x400
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
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
v_fma_f32 v5, v16, 0x41800000, v11
s_mov_b32 s16, 0x3f
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0x80000100
s_add_u32 s16, s16, 0
s_add_u32 s19, s19, 0
v_mov_b32 v20, 0x34
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
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
s_mov_b32 s16, 0x36
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0x100
s_add_u32 s16, s16, 0
s_add_u32 s19, s19, 0
v_mov_b32 v20, 0x4
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
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
v_fma_f32 v7, v16, 0x41800000, v11
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
