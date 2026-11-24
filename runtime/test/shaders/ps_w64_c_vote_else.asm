; SHADER-10 tier C: the "no" side of a vote has code of its own. Only the pixel (5, 7) votes yes (v4 == 0).
; s_cbranch_vccnz goes to the run side (EXEC narrowed to the vote, v6 = 7) when some lane of the WAVE voted yes; the
; fall-through (v6 = 5) runs when none did. On a wave64 the other 63 pixels of the pixel's wave keep v6 = 0; on two
; 32-lane subgroups the half without the pixel runs the fall-through and gets 5. r0 = 7 for the pixel, else
; partition dependent; r1 = the pixel.
; Run: vk_harness vs_ids.spv ps_w64_c_vote_else.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs
; in the pixels of the other half (the detector must call this shader tier C).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
s_mov_b64 s[10:11], exec
v_xor_b32_e32 v4, 5, v2
v_xor_b32_e32 v5, 7, v3
v_or_b32_e32 v4, v4, v5
v_mov_b32_e32 v6, 0
v_cmp_eq_u32_e32 vcc, 0, v4
v_cndmask_b32_e64 v7, 0, 1, vcc
s_cbranch_vccnz 2
v_mov_b32_e32 v6, 5
s_branch 3
s_and_saveexec_b64 s[16:17], vcc
v_mov_b32_e32 v6, 7
s_or_b64 exec, exec, s[16:17]
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
