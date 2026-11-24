; Real binary64 inputs from high-dword literals: 16, 16, 0.25, 6, 6.
; Export (1/16, 1/sqrt(16), sqrt(0.25), frexp(6).mantissa * frexp(6).exponent / 4).
v_rcp_f64_e32 v[8:9], 0x40300000
v_cvt_f32_f64 v0, v[8:9]
v_rsq_f64_e32 v[8:9], 0x40300000
v_cvt_f32_f64 v1, v[8:9]
v_sqrt_f64_e32 v[8:9], 0x3fd00000
v_cvt_f32_f64 v2, v[8:9]
v_frexp_mant_f64_e32 v[8:9], 0x40180000
v_cvt_f32_f64 v3, v[8:9]
v_frexp_exp_i32_f64_e32 v7, 0x40180000
v_cvt_f32_i32 v7, v7
v_mul_f32 v7, v7, 0.25
v_mul_f32 v3, v3, v7
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
