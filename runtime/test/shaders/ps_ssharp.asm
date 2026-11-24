; The S# and T# mip fields through the descriptor path. s0:s1 is the console VA of a T# (8 dwords) with the S#
; (4 dwords) 32 bytes after it; the translator hashes both and looks them up in the descriptor hash table.
; Run: vk_sampler_test vs_ids.spv ps_ssharp.spv out.raw <S# words> [<base level> <last level>]  (check.py: SAMPLER_CASES)
;   vs_ids runs UV 0..1 over the 64x64 target and the texture is 128x128 with two levels, so the implicit sample
;   sees LOD 1 and image_sample_lz LOD 0, before the S# and the T# clamp them.
;   -> mrt0 = (lz.r, lz.g, implicit.r, implicit.g). Level 0 is four quadrants, red (1,0), green (0,1), blue (0,0) and
;      yellow (1,1) in r,g; level 1 is grey (128/255, 128/255).
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[6:7], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_sample_lz v[4:5], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
