; SHADER-10 tier C, the negative twin of ps_w64_b_waterfall: the same waterfall loop over a = (x % 8) + 8 * (y % 8)
; (64 distinct values per 8x8 tile of pixels, x = floor(u * 64), y = floor(v * 64) on the 64x64 target of vs_ids),
; but it also counts the trips in s22 and exports the count after the loop. On a wave64 the loop runs once per
; distinct a in the whole wave, on two 32-lane subgroups once per distinct a in each half, so the exported count
; changes with the partition: a wave of 64 pixels from one 8x8 tile gives 64, a half of it 32 or fewer.
; r0 = 17 * a + x (exact either way), r1 = the trip count s22 (not exact on 32-wide subgroups).
; Run: vk_harness vs_ids.spv ps_w64_c_waterfall_count.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r0 equal, r1
; differs in some pixel (the detector must call this shader tier C).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_and_b32_e32 v4, 7, v2
v_and_b32_e32 v5, 7, v3
v_lshl_add_u32 v4, v5, 3, v4
v_mov_b32_e32 v6, 0
s_mov_b32 s22, 0
s_mov_b64 s[10:11], exec
v_readfirstlane_b32 s20, v4
v_cmp_eq_u32_e32 vcc, s20, v4
s_and_saveexec_b64 s[12:13], vcc
s_mul_i32 s21, s20, 17
v_add_nc_u32_e32 v6, s21, v2
s_add_u32 s22, s22, 1
s_xor_b64 exec, exec, s[12:13]
s_cbranch_execnz -8
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, s22
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
