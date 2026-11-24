; Texture sampling test: samples a 2D texture (T# at s[8:15] loaded via s[0:1], S# at s[16:19])
; using normalized UV coordinates from attr0, outputs sampled color to mrt0.
; Run: vk_texture_test (autonomously, or with vs.spv fs.spv out.raw ptr_sgpr s0 s1 s2 s3)
;   -> verifies center pixel = gradient color (0.5, 0.5, 0.5, 1.0) sampled via descriptor table
; s[0:1] = 64-bit pointer to T# in GPU heap
; s[8:15] = 8-dword T# (image descriptor)
; s[16:19] = 4-dword S# (sampler descriptor)
; v[0:1] = barycentric UV coordinates
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
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
