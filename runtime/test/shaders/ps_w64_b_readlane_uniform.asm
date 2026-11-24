; SHADER-10 tier B: v_readlane_b32 of a value that is the same in every lane, at lanes that a 32-lane subgroup does not
; have. s4 is a user SGPR (the test sets it to 123 with VKH_SGPRS=0,0,0,0,123); v5 = s4 and v7 = 77 are written while
; EXEC is the launch mask, so every lane holds them. Reading lane 40 of v5 and lane 63 of v7 gives 123 and 77 on a
; wave64 and, since the value is the same everywhere, the lane's own copy is the exact answer on a 32-lane subgroup
; (a shuffle from lane 40 of a 32-wide subgroup is undefined: the translator rewrites the read, docs/design.md).
; x = floor(u * 64) on the 64x64 target of vs_ids. r0 = x + 123, r1 = 123 + 77 = 200, r2 = r3 = 0.
; Run: VKH_SGPRS=0,0,0,0,123 vk_harness vs_ids.spv ps_w64_b_readlane_uniform.spv out.raw with VKH_SUBGROUP_SIZE=64
; and 32: equal, and equal to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_mov_b32_e32 v7, 77
v_readlane_b32 s20, v5, 40
v_readlane_b32 s21, v7, 63
v_add_nc_u32_e32 v6, s20, v2
s_add_u32 s22, s20, s21
v_mov_b32_e32 v7, s22
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
