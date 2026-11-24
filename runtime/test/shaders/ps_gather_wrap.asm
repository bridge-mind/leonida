; image_gather4_l_o at the image border under each S# address mode (SHADER-36). The 2x2 gather footprint is built from four
; OpImageFetch (core Vulkan has no explicit-LOD gather), so the translator runs every texel coordinate through the S#'s own
; CLAMP_X/CLAMP_Y. S# word0: CLAMP_X = [2:0], CLAMP_Y = [5:3]; 0 wrap, 1 mirror, 2 clamp to last texel, 3 mirror once
; (Mesa src/amd/registers/gfx10-rsrc.json for the fields, SQ_TEX_* in gfx6.json for the values, as RADV's radv_tex_wrap maps Vulkan address modes onto them). Cases 1, 3, 5 use a literal S#
; (the translator reads the mode at translation time); cases 2, 4, 6 add 0 to s16 first, so the mode is read from the SGPR
; at run time.
; Run: vk_texture_test <vs_ids.spv> <ps_gather_wrap.spv> out.raw 0 92 c00000 14500000 0 -> RGBA32F pixels, all the same.
; The T# is loaded from the heap as in ps_texture_sample; the harness image is the 8x8 gradient R = 32x/255, G = 32y/255,
; so the R gather returns texel columns and the G gather texel rows (x, y = 0..7). u = v = 0.5 is texel position (4.0, 4.0),
; the footprint's lower corner is (3, 3), moved by the signed 6-bit texel offset in the first address dword (x [5:0], y [13:8]).
; Gather order: (x0,y1) (x1,y1) (x1,y0) (x0,y0). Texel indices are scaled back by 255/32 and rounded, then packed in base 8,
; so every channel is an exact integer below 2^24.
;   out.r: interior case x0=2 y0=5 (S# 0x12). R gather = 2 3 3 2 -> 1242, G gather = 6 6 5 5 -> 3501;
;          1242*4096 + 3501 = 5090733 (0x4dadad)
;   out.g, out.b, out.a: two cases each as first*4096 + second; a case is ((x0'*8 + x1')*8 + y0')*8 + y1', x0' x1' y0' y1'
;   being the texel indices the coordinates map to:
;   case 1 W,M  x0=-1  y0=7  : x -1->7  0->0 ; y 7->7  8->7 (mirror)          -> (7,0,7,7) = 3647
;   case 2 M,W  x0=7   y0=-12: x 7->7   8->7 ; y -12->4 -11->5 (wrap)         -> (7,7,4,5) = 4069
;   case 3 C,O  x0=-3  y0=-12: x 0,0 ; y once(-12)=11->7, once(-11)=10->7      -> (0,0,7,7) = 63
;   case 4 O,C  x0=-2  y0=7  : x once(-2)=1, once(-1)=0 ; y clamp 7,8->7       -> (1,0,7,7) = 575
;   case 5 M,O  x0=20  y0=-1 : x 20 mod 16=4->4, 21->5 ; y once(-1)=0, 0->0    -> (4,5,0,0) = 2368
;   case 6 W,M  x0=13  y0=-12: x 13->5, 14->6 ; y -12 mod 16=4->4, -11->5      -> (5,6,4,5) = 2981
;   out.g = 3647*4096 + 4069 = 14942181 (0xe3ffe5); out.b = 63*4096 + 575 = 258623 (0x3f23f);
;   out.a = 2368*4096 + 2981 = 9702309 (0x940ba5)
; A clamp-to-edge gather gives 0,0,7,7 for case 1 (x: -1,0 -> 0,0), so the cases tell wrap, mirror and mirror-once from clamp.
v_mov_b32_e32 v21, 0.5
v_mov_b32_e32 v22, 0.5
v_mov_b32_e32 v23, 0
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x12
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_mov_b32 v20, 0x23f
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v10, 0x40ff0000, v10
v_rndne_f32 v10, v10
v_mul_f32 v11, 0x40ff0000, v11
v_rndne_f32 v11, v11
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v13, 0x40ff0000, v13
v_rndne_f32 v13, v13
v_mul_f32 v14, 0x40ff0000, v14
v_rndne_f32 v14, v14
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v16, v8, 8.0, v9
v_fma_f32 v16, v16, 8.0, v10
v_fma_f32 v16, v16, 8.0, v11
v_fma_f32 v17, v12, 8.0, v13
v_fma_f32 v17, v17, 8.0, v14
v_fma_f32 v17, v17, 8.0, v15
v_fma_f32 v4, v16, 0x45800000, v17
s_mov_b32 s16, 0x8
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_mov_b32 v20, 0x43c
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v19, v18
s_mov_b32 s16, 0x1
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
s_add_u32 s16, s16, 0
v_mov_b32 v20, 0x3104
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v24, v18
v_fma_f32 v5, v19, 0x45800000, v24
s_mov_b32 s16, 0x1a
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_mov_b32 v20, 0x313a
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v19, v18
s_mov_b32 s16, 0x13
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
s_add_u32 s16, s16, 0
v_mov_b32 v20, 0x43b
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v24, v18
v_fma_f32 v6, v19, 0x45800000, v24
s_mov_b32 s16, 0x19
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
v_mov_b32 v20, 0x3c11
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v19, v18
s_mov_b32 s16, 0x8
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
s_add_u32 s16, s16, 0
v_mov_b32 v20, 0x310a
image_gather4_l_o v[8:11], v[20:23], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_l_o v[12:15], v[20:23], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mul_f32 v8, 0x40ff0000, v8
v_rndne_f32 v8, v8
v_mul_f32 v9, 0x40ff0000, v9
v_rndne_f32 v9, v9
v_mul_f32 v12, 0x40ff0000, v12
v_rndne_f32 v12, v12
v_mul_f32 v15, 0x40ff0000, v15
v_rndne_f32 v15, v15
v_fma_f32 v18, v8, 8.0, v9
v_fma_f32 v18, v18, 8.0, v15
v_fma_f32 v18, v18, 8.0, v12
v_mov_b32 v24, v18
v_fma_f32 v7, v19, 0x45800000, v24
; At UV (0.5,0.5), gather red from x=3,4 at LOD zero: the four bytes are (96,128,128,96).
; Gather green with offset (0,-1): (96,96,64,64). As binary16, 64/255 = 0x3404,
; 96/255 = 0x3606 and 128/255 = 0x3804. Each gather must write just two packed VGPRs.
v_mov_b32 v19, 0x3f00
v_mov_b32 v20, 0.5
v_mov_b32 v21, 0.5
v_mov_b32 v22, 0
v_mov_b32 v10, 0
v_mov_b32 v14, 0
image_gather4_l v[8:9], v[20:22], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
image_gather4_l_o v[12:13], v[19:22], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_xor_b32 v8, 0x38043606, v8
v_xor_b32 v9, 0x36063804, v9
v_or_b32 v8, v8, v9
v_or_b32 v8, v8, v10
v_cmp_eq_u32_e32 vcc_lo, 0, v8
v_cndmask_b32_e32 v4, 0, v4, vcc_lo
v_xor_b32 v12, 0x36063606, v12
v_xor_b32 v13, 0x34043404, v13
v_or_b32 v12, v12, v13
v_or_b32 v12, v12, v14
v_cmp_eq_u32_e32 vcc_lo, 0, v12
v_cndmask_b32_e32 v5, 0, v5, vcc_lo
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
