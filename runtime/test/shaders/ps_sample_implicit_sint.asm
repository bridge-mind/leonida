; The 128x128 signed view minifies to mip 1 on the 64x64 target. First clamp the sampler to mip 0,
; then permit mip 1. Every input byte is 0 or 128, so a signed D16 sample must produce 0 or -128.
; Divide each signed channel by -128 to compare all output components with the shared 0/1 color reference.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
image_sample v[4:5], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D d16
s_waitcnt vmcnt(0)
v_bfe_i32 v7, v5, 16, 16
v_bfe_i32 v6, v5, 0, 16
v_bfe_i32 v5, v4, 16, 16
v_bfe_i32 v4, v4, 0, 16
v_sub_nc_u32_e32 v4, 0, v4
v_cvt_f32_i32_e32 v4, v4
v_sub_nc_u32_e32 v5, 0, v5
v_cvt_f32_i32_e32 v5, v5
v_sub_nc_u32_e32 v6, 0, v6
v_cvt_f32_i32_e32 v6, v6
v_sub_nc_u32_e32 v7, 0, v7
v_cvt_f32_i32_e32 v7, v7
v_mul_f32_e32 v4, 0x3c000000, v4
v_mul_f32_e32 v5, 0x3c000000, v5
v_mul_f32_e32 v6, 0x3c000000, v6
v_mul_f32_e32 v7, 0x3c000000, v7
exp mrt0 v4,v5,v6,v7 done vm
s_endpgm
