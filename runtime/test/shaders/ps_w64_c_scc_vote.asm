; SHADER-10 tier C: the SCC of a wave-wide vote read as data. Only the pixel (5, 7) is "on" (v4 == 0).
; s_cmp_lg_u64 vcc, 0 sets SCC when any lane of the wave voted; s_cselect_b32 turns it into a number: on a wave64
; every pixel of the pixel's wave reads 1, on two 32-lane subgroups only the half that holds the pixel does.
; r0 = SCC (partition dependent, 1 for the pixel), r1 = the pixel.
; Run: vk_harness vs_ids.spv ps_w64_c_scc_vote.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs
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
v_cmp_eq_u32_e32 vcc, 0, v4
s_cmp_lg_u64 vcc, 0
s_cselect_b32 s5, 1, 0
v_mov_b32_e32 v6, s5
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
