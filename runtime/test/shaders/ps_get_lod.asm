; image_get_lod: the LOD an implicit sample at these coordinates would use. RDNA2 ISA (MIMG): the result is
; {rawLOD, clampedLOD} in VGPRs 1..0, so the first register gets the clamped LOD (after the sampler's and the image's
; limits) and the second the calculated one before any clamp; dmask keeps one or both, packed.
; Run: vk_texture_test vs_ids.spv ps_get_lod.spv out.raw 0 92 <S# word 1> 14500000 0 --two-mip, twice.
; --two-mip draws 64x64 pixels over a 128x128 texture that has two levels, and vs_ids runs u and v from 0 to 1 over
; the image, so a pixel step is two texels each way: the calculated LOD is log2(2) = 1. The second and third query
; scale u by 4: eight texels a pixel along x, two along y, and the larger one counts: log2(8) = 3, which no level
; of this image reaches, so its clamped LOD stops at the last level, 1. The S# is read from the heap behind the T#;
; its word 1 holds MIN_LOD [11:0] and MAX_LOD [23:12] in 4.8 fixed point, and its mip filter is point (word 2).
;   mrt0 = (clamped LOD, calculated LOD at (u, v); calculated LOD alone at (4u, v); clamped LOD alone at (4u, v))
;   S# word 1 = 0xc00000 (LOD 0..12):   1.0 1.0 3.0 1.0
;   S# word 1 = 0x40000  (LOD 0..0.25): 0.0 1.0 3.0 0.0    MAX_LOD limits the clamped LOD to 0.25, and a point-filtered
;                                                          sample at 0.25 uses level 0; the calculated LOD is not limited
; Every pixel must hold these within 1/16 (a device may keep as few as four fraction bits of LOD).
; The second range is not 0..0: Mesa llvmpipe 26.3.1 does not apply a sampler whose minimum and maximum LOD are equal
; to an LOD query (it gives 1.0 there, where RADV gives 0.0), which says nothing about the translation.
; Not shown by this test: u and v taken in the wrong order. The texture is square, where the LOD is the same either way.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
image_get_lod v[4:5], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
v_mul_f32_e32 v8, 4.0, v2
v_mov_b32_e32 v9, v3
image_get_lod v6, v[8:9], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
image_get_lod v7, v[8:9], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
