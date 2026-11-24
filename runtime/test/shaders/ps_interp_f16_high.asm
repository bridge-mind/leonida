; 16-bit interpolation of both halves of a packed channel, drawn with vs_f16_pack (param0.x = u | v << 16,
; param0.y = v | u << 16). Without `high` v_interp_p1ll_f16 / v_interp_p2_f16 read the low half, with it the high
; half (Mesa ACO emit_interp_instr passes high_16bits to both). The low halves of mrt0.xyz must be f16 of u, v and
; u at the pixel centre, and mrt0.w the f32 v interpolated from attr1.y (control).
v_interp_p1ll_f16 v2, v0, attr0.x
v_interp_p2_f16 v2, v1, attr0.x, v2
v_interp_p1ll_f16 v3, v0, attr0.x high
v_interp_p2_f16 v3, v1, attr0.x, v3 high
v_interp_p1ll_f16 v4, v0, attr0.y high
v_interp_p2_f16 v4, v1, attr0.y, v4 high
v_interp_p1_f32_e32 v5, v0, attr1.y
v_interp_p2_f32_e32 v5, v1, attr1.y
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
