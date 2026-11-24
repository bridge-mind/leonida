; SCC from s_and_b32 (zero -> 0), s_add_i32 (signed overflow -> 1), s_max_u32 (S0 > S1 -> 1), each followed by a branch.
; Expected mrt0 bits: 1, 1, 3, 0x80000000
s_mov_b32 s0, 0x80
s_and_b32 s1, s0, 0x7f
s_mov_b32 s2, 1
s_cbranch_scc0 1
s_mov_b32 s2, 2
s_mov_b32 s7, 0x7fffffff
s_add_i32 s3, s7, 1
s_mov_b32 s4, 1
s_cbranch_scc1 1
s_mov_b32 s4, 2
s_max_u32 s5, 5, 3
s_mov_b32 s6, 1
s_cbranch_scc0 1
s_mov_b32 s6, 3
v_mov_b32_e32 v0, s2
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s6
v_mov_b32_e32 v3, s3
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
