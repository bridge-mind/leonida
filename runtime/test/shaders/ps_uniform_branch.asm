; Wave-uniform s_cbranch_vccz: lanes with i > 0.27 set VCC; a wave with any set lane runs the scalar add for
; all its lanes. Expected: bit=1 pixels export s5 = 8; some bit=0 pixels (same wave) also 8; the rest 7.
v_cmp_lt_f32 vcc_lo, 0x3e8a3d71, v0
s_mov_b32 s5, 7
s_cbranch_vccz 1
s_add_u32 s5, s5, 1
v_mov_b32 v3, s5
v_cndmask_b32_e64 v4, 0, 1, vcc_lo
exp mrt0 v3, v4, v0, v0 done vm
s_endpgm
