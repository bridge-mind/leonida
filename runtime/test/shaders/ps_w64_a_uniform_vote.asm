; SHADER-10 tier A: votes on a wave-uniform compare (docs/design.md, "Wave64 graphics: partition invariance"). s4 is a
; user SGPR (the test sets it to 9 with VKH_SGPRS=0,0,0,0,9), copied to a VGPR while EXEC is the launch mask, so every
; lane holds 9 and a v_cmp on it gives every launch lane or none: each 32-lane half votes as the whole wave does, and
; both sides of each branch may have code of their own, scalars included. x = floor(u * 64) on the 64x64 target of
; vs_ids.
;  (1) s_cbranch_vccz if/else on 5 < s4 (yes): the yes side sets v6 = 7 and s20 = 11, the no side v6 = 5 and s20 = 22;
;      both are read after the join. r0 = x + 7, r1 = 11.
;  (2) the LLVM form v_cmp_e64 s[14:15] + s_and_b64 vcc, exec, s[14:15] + s_cbranch_vccnz on 50 < s4 (no): the no
;      side sets v8 = 3 and s21 = 33, the yes side v8 = 4 and s21 = 44. r2 = s21 + v8 = 36.
;  (3) a loop whose back edge is s_cbranch_vccnz on s22 < s4 (s22 the trip count, an SGPR carried across the back
;      edge): s4 = 9 iterations of v9 += 2, then v9 += s22. r3 = 2 * 9 + 9 = 27.
; Run: VKH_SGPRS=0,0,0,0,9 vk_harness vs_ids.spv ps_w64_a_uniform_vote.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32:
; equal, and equal to the expected words (x + 7, 11, 36, 27).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_cmp_lt_u32_e32 vcc, 5, v5
s_cbranch_vccz 3
v_mov_b32_e32 v6, 7
s_mov_b32 s20, 11
s_branch 2
v_mov_b32_e32 v6, 5
s_mov_b32 s20, 22
v_add_nc_u32_e32 v6, v6, v2
v_mov_b32_e32 v7, s20
v_cmp_lt_u32_e64 s[14:15], 50, v5
s_and_b64 vcc, exec, s[14:15]
s_cbranch_vccnz 3
v_mov_b32_e32 v8, 3
s_mov_b32 s21, 33
s_branch 2
v_mov_b32_e32 v8, 4
s_mov_b32 s21, 44
v_add_nc_u32_e32 v8, s21, v8
s_mov_b32 s22, 0
v_mov_b32_e32 v9, 0
s_add_u32 s22, s22, 1
v_add_nc_u32_e32 v9, 2, v9
v_mov_b32_e32 v11, s22
v_cmp_gt_u32_e32 vcc, s4, v11
s_cbranch_vccnz -5
v_add_nc_u32_e32 v9, s22, v9
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
