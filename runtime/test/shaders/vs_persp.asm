; Vertex shader for ps_persp_bary: one triangle with a different clip w per vertex, so interpolation is
; perspective-correct only if it divides by w. Vertex k (VertexIndex v0) lands at screen (5,8), (57,16), (21,55) of
; the 64x64 viewport with clip w = 1, 2, 4, and exports param0 = attr0 = (0.25,-0.5), (2.25,0.75), (-1,3), 0, 1.
; Clip position = ((sx/32 - 1) w, (sy/32 - 1) w, 0, w). Checked by check.py's bary_persp_problem (suggested in #794).
v_mov_b32_e32 v1, 0xbf580000
v_mov_b32_e32 v2, 0xbf400000
v_mov_b32_e32 v3, 0x3f800000
v_mov_b32_e32 v4, 0x3e800000
v_mov_b32_e32 v5, 0xbf000000
v_cmp_eq_u32_e32 vcc_lo, 1, v0
s_mov_b32 s10, 0x3fc80000
v_cndmask_b32_e64 v1, v1, s10, vcc_lo
s_mov_b32 s10, 0xbf800000
v_cndmask_b32_e64 v2, v2, s10, vcc_lo
s_mov_b32 s10, 0x40000000
v_cndmask_b32_e64 v3, v3, s10, vcc_lo
s_mov_b32 s10, 0x40100000
v_cndmask_b32_e64 v4, v4, s10, vcc_lo
s_mov_b32 s10, 0x3f400000
v_cndmask_b32_e64 v5, v5, s10, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
s_mov_b32 s10, 0xbfb00000
v_cndmask_b32_e64 v1, v1, s10, vcc_lo
s_mov_b32 s10, 0x40380000
v_cndmask_b32_e64 v2, v2, s10, vcc_lo
s_mov_b32 s10, 0x40800000
v_cndmask_b32_e64 v3, v3, s10, vcc_lo
s_mov_b32 s10, 0xbf800000
v_cndmask_b32_e64 v4, v4, s10, vcc_lo
s_mov_b32 s10, 0x40400000
v_cndmask_b32_e64 v5, v5, s10, vcc_lo
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
exp pos0 v1, v2, v6, v3 done
exp param0 v4, v5, v6, v7
s_endpgm
