; Wave32 SCC from a 32-bit lane-mask op: lanes with i > 0.27 set VCC, and s_and_b32 of VCC_LO with EXEC_LO sets
; SCC = (the wave's mask != 0), one value for the whole wave. A wave with any set lane runs the scalar add in all
; its lanes. Expected, as for ps_uniform_branch: bit=1 pixels export s5 = 8; some bit=0 pixels (same wave) also 8;
; the rest 7. Before 2026-11-22 each lane took its own bit as SCC, so no bit=0 pixel got 8 (found by FUZZ-8,
; tools/cf_diff.py).
v_cmp_lt_f32 vcc_lo, 0x3e8a3d71, v0
s_mov_b32 s5, 7
s_and_b32 s6, vcc_lo, exec_lo
s_cbranch_scc0 1
s_add_u32 s5, s5, 1
v_mov_b32 v3, s5
v_cndmask_b32_e64 v4, 0, 1, vcc_lo
exp mrt0 v3, v4, v0, v0 done vm
s_endpgm
