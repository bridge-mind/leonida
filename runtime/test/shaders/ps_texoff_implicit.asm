; ps_texoff with its per-pixel-offset sample (.x) as image_sample_o, the implicit-LOD form the corpus uses (all
; 31 texel-offset samples in the 4 PS that need VK_KHR_maintenance8 are image_sample_o). Same texture, sampler
; and expected words as ps_texoff below; checked by the same texoff_problem. With --texel-offset-fallback it
; covers the implicit path: Grad with the original coordinate's derivatives, level from OpImageQueryLod.
; The texture has one mip, so the level is 0 here; uv is constant, so the derivatives are 0.
;
; Texel offsets (image_sample*_o, image_gather4_lz_o) on a real texture. docs/design.md, "Texel offsets": the first
; address dword packs signed 6-bit offsets, x = [5:0], y = [13:8]; the translator turns them into the SPIR-V Offset
; operand (always a run-time value there, also when the shader loads the dword from an immediate: VK_KHR_maintenance8
; outside gather, shaderImageGatherExtended for gather).
; Run: vk_texture_test <vs> ps_texoff.spv out.raw 2 92 0 0 0   (VKT_NEED_MAINTENANCE8=1: exit 77 without the extension)
; The tool puts a T# at heap VA 256 (s[2:3] points at it), binds an 8x8 RGBA8 texture whose texel (x, y) is
; r = 32x/255, g = 32y/255, with a point (nearest) sampler that clamps to the edge (S# 0x92, 0, 0, 0: clamp x/y/z,
; point filters), and draws 64x64 pixels with attr0.xy = ((px + 0.5) / 64, (py + 0.5) / 64).
; Each sampled channel c is turned back into its texel coordinate with rndne(c * 255/32) (255/32 = 7.96875 exactly),
; so a sample of texel (x, y) reads as the integers x, y. The 1x1 fallback texture (a descriptor miss) reads as 8, 0.
;
; Expected mrt0 words for pixel (px, py), bx = px >> 3, by = py >> 3 (0..7), derived from the rule above and the
; texture, not from a run:
;   .x  image_sample_lz_o, offset computed per pixel: ox = bx - 4, oy = by - 4 (-4..3, so the dword is
;       (ox & 63) | (oy & 63) << 8, e.g. -4 -> 0x3c, -1 -> 0x3f), at uv = (4.5/8, 4.5/8) = the centre of texel
;       (4, 4). Texel read = (4 + ox, 4 + oy) = (bx, by): all 64 offset pairs, each a different texel, none clamped.
;       word = bx | by << 4.        Offset dropped: 0x44 everywhere. x and y swapped: by | bx << 4.
;   .y  two samples with the offset dword loaded from an immediate, both at the centre of texel (4, 4):
;       A = image_sample_l_o (LOD 0.0), dword 0x023d: x = 0x3d = -3, y = 0x02 = +2 -> texel (1, 6)
;       B = image_sample_lz_o, dword 0x2010: x = 0x10 = +16, y = 0x20 = -32 -> texel (20, -28), which the sampler
;           clamps to (7, 0). This one tells the field widths apart: a 4-bit field gives (4, 4), a 5-bit signed
;           field (0, 4). It is only defined on a device whose minTexelOffset..maxTexelOffset covers -32..16
;           (Vulkan guarantees -8..7); check.py and run_gpu_tests.py judge B only there.
;       word = A.x | A.y << 4 | B.x << 8 | B.y << 12 = 1 | 6 << 4 | 7 << 8 | 0 << 12 = 0x0761
;   .z  image_gather4_lz_o of red (dmask 1), the per-pixel offset dword of .x, at uv = (0.5, 0.5) = the corner shared
;       by texels (3..4, 3..4): the footprint is i0 = 3, i1 = 4 in both axes, moved by the offset and clamped:
;       X0 = max(bx - 1, 0), X1 = bx, Y0 = max(by - 1, 0), Y1 = by. Gather order (Vulkan and D3D, component k ->
;       v[d + k]): (i0, j1), (i1, j1), (i1, j0), (i0, j0); red holds x.
;       word = X0 | X1 << 4 | X1 << 8 | X0 << 12
;   .w  the same gather of green (dmask 2), which holds y: word = Y1 | Y1 << 4 | Y0 << 8 | Y0 << 12
; Examples: pixel (0, 0): 0x00, 0x0761, 0x0000, 0x0000. Pixel (20, 50) (bx 2, by 6): 0x62, 0x0761, 0x1221, 0x5566.
;           pixel (63, 63): 0x77, 0x0761, 0x6776, 0x6677.
; Not seen by this test on an AMD Radeon RX 6800 (driver 26.11.2): fields read unsigned. In theory that reads 7 where
; an offset is negative and (7, 7) for B, but there Offset 60..63 sampled as -4..-1 and +32 as -32, so the same words
; came back. The sign extension is checked on the SPIR-V text by tests/test_texel_offsets.py. The gather offsets stay
; in -4..3, so the gather field width is not tested here either (4-bit and 5-bit gather fields pass).
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
image_sample_o v[4:5], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v24, 0x23d
v_mov_b32_e32 v25, 0x3f100000
v_mov_b32_e32 v26, 0x3f100000
v_mov_b32_e32 v27, 0
image_sample_l_o v[6:7], v[24:27], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v28, 0x2010
v_mov_b32_e32 v29, 0x3f100000
v_mov_b32_e32 v30, 0x3f100000
image_sample_lz_o v[8:9], v[28:30], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mov_b32_e32 v13, v10
v_mov_b32_e32 v14, 0.5
v_mov_b32_e32 v15, 0.5
image_gather4_lz_o v[16:19], v[13:15], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_gather4_lz_o v[20:23], v[13:15], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
; A16: preserve a whole offset dword and unpack U from the low and V from the high half.
; From texel (3,5), offsets (bx-3, by-5) reach the same (bx,by) as the full-precision sample in v4:5.
v_add_nc_u32_e32 v44, 1, v2
v_add_nc_u32_e32 v45, -1, v3
v_and_b32_e32 v44, 63, v44
v_and_b32_e32 v45, 63, v45
v_lshl_or_b32 v44, v45, 8, v44
v_mov_b32_e32 v45, 0x39803700
image_sample_lz_o v[46:47], v[44:45], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
v_xor_b32_e32 v46, v46, v4
v_xor_b32_e32 v47, v47, v5
v_or_b32_e32 v46, v46, v47
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
v_mul_f32_e32 v8, 0x40ff0000, v8
v_rndne_f32_e32 v8, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_f32_e32 v9, 0x40ff0000, v9
v_rndne_f32_e32 v9, v9
v_cvt_u32_f32_e32 v9, v9
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
v_lshl_or_b32 v41, v8, 8, v41
v_lshl_or_b32 v41, v9, 12, v41
v_lshl_or_b32 v42, v17, 4, v16
v_lshl_or_b32 v42, v18, 8, v42
v_lshl_or_b32 v42, v19, 12, v42
v_lshl_or_b32 v43, v21, 4, v20
v_lshl_or_b32 v43, v22, 8, v43
v_lshl_or_b32 v43, v23, 12, v43
; A16 mismatch marks .x outside its valid 0x00..0x77 range.
v_cmp_eq_u32_e32 vcc_lo, 0, v46
v_cndmask_b32_e32 v40, -1, v40, vcc_lo
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
