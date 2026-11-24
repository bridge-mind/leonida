; SHADER-10 tier B: a waterfall loop in a wave64 pixel shader. a = (x % 4) + 4 * (y % 4) is a per-pixel index into 16
; distinct values (x = floor(u * 64), y = floor(v * 64), pixel coordinates on the 64x64 target of vs_ids). Each trip
; reads the first active lane's a (v_readfirstlane_b32), keeps the lanes whose a equals it, runs the body for them
; with the scalar (s21 = 17 * a, v6 = s21 + x) and takes them out of EXEC; the loop ends when EXEC is empty. The
; scalars are only used inside the body, so a lane's result is 17 * its own a + x however the lanes are grouped
; (one wave of 64 or two of 32). r0 = 17 * a + x; r1 = a; r2 = r3 = 0.
; Run: vk_harness vs_ids.spv ps_w64_b_waterfall.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: equal, and equal to
; the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_and_b32_e32 v4, 3, v2
v_and_b32_e32 v5, 3, v3
v_lshl_add_u32 v4, v5, 2, v4
v_mov_b32_e32 v6, 0
s_mov_b64 s[10:11], exec
v_readfirstlane_b32 s20, v4
v_cmp_eq_u32_e32 vcc, s20, v4
s_and_saveexec_b64 s[12:13], vcc
s_mul_i32 s21, s20, 17
v_add_nc_u32_e32 v6, s21, v2
s_xor_b64 exec, exec, s[12:13]
s_cbranch_execnz -7
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, 0
exp mrt0 v6, v4, v7, v7 done vm
s_endpgm
