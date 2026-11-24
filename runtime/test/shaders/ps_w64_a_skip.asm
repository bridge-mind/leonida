; SHADER-10 tier A (docs/design.md, "Wave64 graphics: partition invariance"): a wave64 pixel shader that gives the same
; exports when its 64 lanes are cut into two 32-lane subgroups. x = floor(u * 64), y = floor(v * 64) are the pixel's
; coordinates on the 64x64 target (u, v = attr0.xy of vs_ids), so every expected word is a function of (x, y) and not
; of the lane the pixel landed on.
;  (1) a skip region behind s_cbranch_execz: only lanes with x % 4 == 1 run it (EXEC narrowed by s_and_saveexec_b64);
;      it defines VGPRs only. r0 = x % 4 == 1 ? 7 * y + x : 1000 + x.
;  (2) s_wqm_b64 of the mask x % 4 == 1: a quad is 4 consecutive lanes and never crosses lane 32, so the whole-quad
;      mask is the same inside a half. r1 = x % 4 < 2 (the 2x2 quad of pixel x holds columns x & ~1 and (x & ~1) + 1).
;  (3) s_cbranch_vccz whose region starts by narrowing EXEC to the very VCC it voted on (y % 4 == 2); VGPR defs only.
;      r2 = 2000 + y + (y % 4 == 2 ? 9 : 0).
;  (4) a divergent loop closed by s_cbranch_execnz, (x % 4) + 1 iterations of v9 += 1, no scalar state carried.
;      r3 = x % 4 + 1.
; Run: vk_harness vs_ids.spv ps_w64_a_skip.spv out.raw, once with VKH_SUBGROUP_SIZE=64 and once with 32: the two images
; must be equal (and equal to the expected words). Output is raw bits (r0, r1, r2, r3) per pixel.
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
v_add_nc_u32_e32 v6, 0x3e8, v2
v_and_b32_e32 v4, 3, v2
v_cmp_eq_u32_e32 vcc, 1, v4
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 2
v_mul_u32_u24_e32 v6, 7, v3
v_add_nc_u32_e32 v6, v6, v2
s_or_b64 exec, exec, s[12:13]
v_cmp_eq_u32_e32 vcc, 1, v4
s_wqm_b64 s[14:15], vcc
v_cndmask_b32_e64 v7, 0, 1, s[14:15]
v_and_b32_e32 v5, 3, v3
v_add_nc_u32_e32 v8, 0x7d0, v3
v_cmp_eq_u32_e32 vcc, 2, v5
s_cbranch_vccz 3
s_and_saveexec_b64 s[16:17], vcc
v_add_nc_u32_e32 v8, 9, v8
s_or_b64 exec, exec, s[16:17]
v_add_nc_u32_e32 v10, 1, v4
v_mov_b32_e32 v9, 0
v_add_nc_u32_e32 v9, 1, v9
v_cmp_lt_u32_e32 vcc, v9, v10
s_and_b64 exec, exec, vcc
s_cbranch_execnz -4
s_mov_b64 exec, s[10:11]
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
