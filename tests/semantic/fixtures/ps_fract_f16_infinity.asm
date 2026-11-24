; Both infinities live in the same register. The other destination half must survive.
v_mov_b32_e32 v4, 0xfc007c00
v_mov_b32_e32 v0, 0x13570000
v_mov_b32_e32 v1, 0x24680000
v_fract_f16_e32 v0, v4
v_lshrrev_b32_e32 v5, 16, v4
v_fract_f16_e64 v1, v5
; Turn the sign bit of the low-half result into a visible 0.0 or 1.0.
v_bfe_u32 v2, v0, 15, 1
v_cvt_f32_u32_e32 v2, v2
; The test supplies the clamped channel in s0.
v_mov_b32_e32 v6, s0
v_mov_b32_e32 v3, 0x1234ffff
v_fract_f16_e64 v3, v6 clamp
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
