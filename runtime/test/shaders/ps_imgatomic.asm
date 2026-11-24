; Image atomics in a pixel shader, the forms the game uses (1,018 of its 1,038 image-atomic shaders are pixel shaders:
; image_atomic_or / umin / umax on a 2D image and umax on a 3D one), through the descriptor table and the set-0
; r32ui storage arrays (binding 4 for 2D, 7 for 3D). User data s2:s3 is the console VA of a table of four T#s:
; +0x00 the 2D umax image, +0x20 the 2D umin image, +0x40 the 2D or image, +0x60 the 3D umax image (any 8 dwords:
; only the key matters). Each of the 64x64 pixels computes px, py from attr0 (vs_ids runs it 0..1 over the target),
; its id = px + 64 * py, and its texel tx = px >> 3, ty = py >> 3, so 64 pixels contend for each texel of the 8x8
; images:
;   umax2d[ty][tx] = max(id), umin2d[ty][tx] = min(id), or2d[ty][tx] |= 1 << (id & 31),
;   umax3d[px & 3][ty >> 1][tx >> 1] = max(id)          (a 4x4x4 image)
; After its atomics, every odd px is discarded (v_cmpx). The atomics of a pixel that is discarded afterwards still
; count, as on hardware: the expected values above are over all 4096 pixels. Even pixels export 1.0.
; Run: vk_bindless_test vs_ids.spv ps_imgatomic.spv   (images cleared to 0, umin to 0xffffffff; checked per texel)
s_load_dwordx8 s[8:15], s[2:3], 0x0
s_load_dwordx8 s[16:23], s[2:3], 0x20
s_load_dwordx8 s[24:31], s[2:3], 0x40
s_load_dwordx8 s[32:39], s[2:3], 0x60
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v4, v2
v_cvt_u32_f32_e32 v5, v3
v_lshlrev_b32_e32 v6, 6, v5
v_add_nc_u32_e32 v6, v6, v4
v_lshrrev_b32_e32 v7, 3, v4
v_lshrrev_b32_e32 v8, 3, v5
image_atomic_umax v6, v[7:8], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_atomic_umin v6, v[7:8], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_and_b32_e32 v9, 31, v6
v_lshlrev_b32_e64 v9, v9, 1
image_atomic_or v9, v[7:8], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_lshrrev_b32_e32 v10, 1, v7
v_lshrrev_b32_e32 v11, 1, v8
v_and_b32_e32 v12, 3, v4
image_atomic_umax v6, v[10:12], s[32:39] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_waitcnt vmcnt(0)
v_and_b32_e32 v13, 1, v4
v_cmpx_eq_u32_e32 0, v13
v_mov_b32_e32 v14, 1.0
exp mrt0 v14, v14, v14, v14 done vm
s_endpgm
