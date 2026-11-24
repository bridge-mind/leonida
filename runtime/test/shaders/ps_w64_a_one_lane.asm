; SHADER-10 tier A, the case where one half votes and the other does not: only the pixel (5, 7) (x = floor(u * 64),
; y = floor(v * 64) on the 64x64 target of vs_ids) is "on" (v4 == 0). Every other wave, and on 32-lane subgroups every
; half that does not hold the pixel, votes "no". The three regions are skips of lane-masked work whose "no" side is
; the join itself, so a half that votes no does exactly what its lanes did in the 64-lane wave:
;  (1) s_cbranch_execz over a VGPR def: r0 = 7777 for the pixel, 1000 + x elsewhere.
;  (2) s_cbranch_vccz whose run side starts by narrowing EXEC to the vote's VCC: r1 = 3000 + 11 for the pixel.
;  (3) s_cbranch_vccnz to the run side, whose fall-through is only a jump to the join: r2 = 4000 + y + 13 for the pixel.
; r3 = x % 4. Run: vk_harness vs_ids.spv ps_w64_a_one_lane.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: equal images,
; equal to the derived words.
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
v_add_nc_u32_e32 v6, 0x3e8, v2
v_cmp_eq_u32_e32 vcc, 0, v4
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 2
v_mov_b32_e32 v6, 0x1e61
s_or_b64 exec, exec, s[12:13]
v_mov_b32_e32 v7, 0xbb8
v_cmp_eq_u32_e32 vcc, 0, v4
s_cbranch_vccz 3
s_and_saveexec_b64 s[16:17], vcc
v_add_nc_u32_e32 v7, 11, v7
s_or_b64 exec, exec, s[16:17]
v_add_nc_u32_e32 v8, 0xfa0, v3
v_cmp_eq_u32_e32 vcc, 0, v4
s_cbranch_vccnz 1
s_branch 3
s_and_saveexec_b64 s[16:17], vcc
v_add_nc_u32_e32 v8, 13, v8
s_or_b64 exec, exec, s[16:17]
v_and_b32_e32 v9, 3, v2
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
