; A texture found through a descriptor table. User data s2:s3 is the console VA of the table: a T# (8 dwords) at
; +0 and an S# (4 dwords) at +32, read at heap base + VA. The translator hashes each and looks it up in the
; descriptor hash table (set 0, binding 2); the indices pick the image (binding 0) and the sampler (binding 1).
; The texture is sampled at attr0.xy, which vs_ids runs 0..1 over the image.
; Run: vk_bindless_test vs_ids.spv ps_tex.spv
;   -> mrt0 = the texel under each pixel: an 8x8 gradient (x / 8, y / 8, 0.5, 1) through a nearest sampler, also
;      after the table has grown; the fallback image at index 0 (zeros) when the table does not hold the T#.
s_load_dwordx8 s[8:15], s[2:3], 0x0
s_load_dwordx4 s[16:19], s[2:3], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
