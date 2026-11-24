; M2 pixel shader: samples a texture found through two root descriptor tables. Run: d3d12_m2_textured_test.
; User data from the root signature ([0] a table holding one SRV, [1] a table holding one sampler):
;   s[0:1] = console VA of the SRV table's first slot: the 32 bytes of the T# CreateShaderResourceView wrote
;   s[2:3] = console VA of the sampler table's first slot: the 16 bytes of the S# CreateSampler wrote
; The translator hashes the T# and the S# the shader loads and looks both up in the descriptor hash table (set 0,
; binding 2); the indices pick the image (binding 0) and the sampler (binding 1) in the bindless set.
; attr0.xy = (u, v), 0..1 over the quad.   mrt0 = the texel under the pixel.
; fssm-at: 0x10
; fssm-data: 0x324d5350 0x20535020
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[2:3], 0x0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
