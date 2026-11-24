; SHADER-10 tier C: the EXEC-zero side of an s_cbranch_execz writes an SGPR that is exported afterwards. Only the pixel
; (5, 7) is "on" (v4 == 0). s_cbranch_execz is taken when no lane of the WAVE is active: on a wave64 the pixel's wave
; runs the active side and every pixel of it reads s5 = 7; the 63 other waves take the branch and read 9. On two 32-lane
; subgroups the half without the pixel takes it as well and reads 9. r0 = s5 (7 for the pixel, else partition
; dependent), r1 = the pixel.
; Run: vk_harness vs_ids.spv ps_w64_c_execz_sgpr.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs
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
s_mov_b32 s5, 7
v_cmp_eq_u32_e32 vcc, 0, v4
v_cndmask_b32_e64 v7, 0, 1, vcc
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 2
v_mov_b32_e32 v8, 5
s_branch 1
s_mov_b32 s5, 9
s_or_b64 exec, exec, s[12:13]
v_mov_b32_e32 v6, s5
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
