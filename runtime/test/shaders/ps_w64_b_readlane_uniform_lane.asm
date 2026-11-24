; SHADER-314 tier B: v_readlane_b32 of a uniform VGPR (same value in every lane) with a tainted lane selector
; (s_ff1_i32_b64 / s_bcnt1_i32_b64 of EXEC). Because v5 = s4 and v7 = 77 are uniform across all lanes, the
; readlane result is invariant across 32-lane halves regardless of which lane index s6/s7 selects.
; x = floor(u * 64) on the 64x64 target of vs_ids. r0 = x + 123, r1 = 123 + 77 = 200, r2 = r3 = 0.
; Run: VKH_SGPRS=0,0,0,0,123 vk_harness vs_ids.spv ps_w64_b_readlane_uniform_lane.spv out.raw.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_mov_b32_e32 v7, 77
s_ff1_i32_b64 s6, exec
s_bcnt1_i32_b64 s7, exec
v_readlane_b32 s20, v5, s6
v_readlane_b32 s21, v7, s7
v_add_nc_u32_e32 v6, s20, v2
s_add_u32 s22, s20, s21
v_mov_b32_e32 v7, s22
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
