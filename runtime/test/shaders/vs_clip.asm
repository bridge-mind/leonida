; Full-screen triangle. At pixel centers the two signed distances are positive
; left of column 32 and above row 16 in a 64x64 target. Adding 4 via s0 or s1
; makes that plane positive everywhere. Expected masks: 4096, 2048, 1024, 512.
; The varying location changes between builds; pos1 supplies the clip values.
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
v_sub_f32_e32 v7, 0.5, v1
v_sub_f32_e32 v8, 0x3e800000, v2
v_add_f32_e32 v7, s0, v7
v_add_f32_e32 v8, s1, v8
exp pos0 v3, v4, v5, v6
exp pos1 v7, v8, v5, v6 done
.ifndef PARAM_SLOT
.set PARAM_SLOT, 1
.endif
.if PARAM_SLOT == 4
exp param4 v7, v8, v5, v6
.else
exp param1 v7, v8, v5, v6
.endif
s_endpgm
