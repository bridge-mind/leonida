; SHADER-10 tier C: scalar state defined behind a wave-wide vote and read afterwards (the wave64 form of
; ps_uniform_branch). Only the pixel (0, 0) (x = floor(u * 64) = 0, y = floor(v * 64) = 0 on the 64x64 target of
; vs_ids) sets VCC. s_cbranch_vccz is a vote over the wave: on a wave64 the 63 other pixels of that pixel's wave
; run `s_add_u32 s5, s5, 1` as well and read 8; on two 32-lane subgroups the half without pixel (0, 0) votes "no" and
; keeps s5 = 7. r0 = s5, r1 = pixel (0, 0).
; Run: vk_harness vs_ids.spv ps_w64_c_vote_stale.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs
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
v_or_b32_e32 v4, v2, v3
v_cmp_eq_u32_e32 vcc, 0, v4
s_mov_b32 s5, 7
s_cbranch_vccz 1
s_add_u32 s5, s5, 1
v_mov_b32_e32 v6, s5
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
