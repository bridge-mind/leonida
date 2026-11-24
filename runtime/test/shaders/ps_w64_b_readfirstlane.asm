; SHADER-10 tier B: v_readfirstlane_b32 of a wave-uniform value in a wave64 pixel shader. s4 is a user SGPR (the test
; sets it to 123 with VKH_SGPRS=0,0,0,0,123), copied to a VGPR and read back; so is a constant. The scalar is the
; same whatever lane is first in the wave or in a 32-lane half. x = floor(u * 64) on the 64x64 target of vs_ids.
; r0 = x + 123, r1 = 77 * 2 + 123 (= 277), r2 = r3 = 0.
; Run: VKH_SGPRS=0,0,0,0,123 vk_harness vs_ids.spv ps_w64_b_readfirstlane.spv out.raw with VKH_SUBGROUP_SIZE=64 and
; 32: equal, and equal to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_readfirstlane_b32 s20, v5
v_add_nc_u32_e32 v6, s20, v2
v_mov_b32_e32 v7, 77
v_readfirstlane_b32 s21, v7
s_lshl_b32 s22, s21, 1
s_add_u32 s22, s22, s20
v_mov_b32_e32 v7, s22
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
