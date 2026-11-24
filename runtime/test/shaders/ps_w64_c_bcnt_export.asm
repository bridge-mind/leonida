; SHADER-10 tier C: a lane count exported as data. Pixels with x % 4 == 1 (x = floor(u * 64) on the 64x64 target of
; vs_ids) set VCC; s_bcnt1_i32_b64 of it counts them in the whole wave (16 for a full wave of 64 pixels in which a
; quarter of the columns match) and in one 32-lane half only half as many. r0 = the count; r1 = x % 4 == 1.
; Run: vk_harness vs_ids.spv ps_w64_c_bcnt_export.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0
; differs (the detector must call this shader tier C).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
v_and_b32_e32 v4, 3, v2
v_cmp_eq_u32_e32 vcc, 1, v4
s_bcnt1_i32_b64 s6, vcc
v_mov_b32_e32 v6, s6
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
