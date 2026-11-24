; M3 pixel shader, pass 2: samples the first render target of pass 1 (read through an SRV in a descriptor table) and
; swaps red and blue. Run: d3d12_m3_frame_test.
; User data from the root signature ([0] a table of one SRV, [1] a table of one sampler, [2] one 32-bit constant):
;   s[0:1] = console VA of the SRV table's first slot (the T#);   s[2:3] = console VA of the sampler table's slot (the S#)
;   s4 = the output's alpha, a float (changes every frame)
; attr0.xy = (u, v), 0..1 over the target.   mrt0 = (texel.b, texel.g, texel.r, s4)
; fssm-at: 0x10
; fssm-data: 0x33505350 0x20535020
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[2:3], 0x0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, s4
exp mrt0 v6, v5, v4, v8 done vm
s_endpgm
