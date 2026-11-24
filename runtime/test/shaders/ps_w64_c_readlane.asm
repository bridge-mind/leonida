; SHADER-10 tier C: v_readlane_b32 of a varying value. p = y * 64 + x is the pixel number (x = floor(u * 64),
; y = floor(v * 64) on the 64x64 target of vs_ids), different for every lane. s6 = lane 40's p: lane 40 does not exist
; in a 32-lane subgroup, so it cannot be the same there. s7 = lane 7's p: it looks harmless (lane 7 exists in a
; half) but it is not: on a wave64 every lane gets the value of hardware lane 7, in two 32-lane halves the upper half
; gets the value of its own lane 7 (hardware lane 39). Only a readlane of a value that is the same in every lane is
; invariant. r0 = s6, r1 = the own p (equal everywhere), r2 = s7.
; Run: vk_harness vs_ids.spv ps_w64_c_readlane.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 and r2
; differ (the detector must call this shader tier C, whichever of the two reads it looks at).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_lshl_add_u32 v4, v3, 6, v2
s_mov_b64 s[10:11], exec
v_readlane_b32 s6, v4, 40
v_readlane_b32 s7, v4, 7
v_mov_b32_e32 v6, s6
v_mov_b32_e32 v7, s7
v_mov_b32_e32 v8, 0
exp mrt0 v6, v4, v7, v8 done vm
s_endpgm
