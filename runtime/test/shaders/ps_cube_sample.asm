; Cube sampling test: direction (2u - 1, 2v - 1, -0.6) from attr0, cube face coordinates the way the hardware takes
; them (s, t = sc, tc / |cubema| + 1.5; face + 8 * slice for slice 1 of a cube array), then image_sample dim:CUBE.
; T# at s[8:15] loaded via s[0:1], literal S# (nearest, clamp to edge) at s[16:19].
; Run: vk_texture_test cube -> every pixel is the texel a cube sample of its direction reads from the second cube
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_fma_f32 v2, v2, 2.0, -1.0
v_fma_f32 v3, v3, 2.0, -1.0
v_mov_b32_e32 v4, 0xbf19999a
v_cubema_f32 v5, v2, v3, v4
v_rcp_f32_e64 v5, |v5|
v_cubesc_f32 v6, v2, v3, v4
v_cubetc_f32 v7, v2, v3, v4
v_fmaak_f32 v6, v6, v5, 0x3fc00000
v_fmaak_f32 v7, v7, v5, 0x3fc00000
v_cubeid_f32 v8, v2, v3, v4
v_add_f32_e32 v8, 0x41000000, v8
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
image_sample v[9:12], v[6:8], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_CUBE
s_waitcnt vmcnt(0)
exp mrt0 v9, v10, v11, v12 done vm
s_endpgm
