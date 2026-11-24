; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_fma_f16 op_sel (source and destination halves), v_subrev_co_ci_u32 with borrow, 64-bit v_cmp/v_cmpx.
; Expected mrt0 bits: 0xabcd4400, 0x46001234, 0xbffffffa, 7
; XOR bit 30 before export to encode the integer result as a finite RGBA32F value.
v_mov_b32 v0, 0x40003c00
v_mov_b32 v1, 0x44004200
v_mov_b32 v2, 0xabcd0000
v_mov_b32 v3, 0x00001234
v_fma_f16 v2, v0, v1, 0 op_sel:[0,1,0,0]
v_fma_f16 v3, v0, v1, 0 op_sel:[1,0,0,1]
v_mov_b32 v4, 5
s_mov_b32 vcc_lo, -1
v_subrev_co_ci_u32_e64 v5, s0, 10, v4, vcc_lo
v_mov_b32 v6, 0
v_mov_b32 v7, 0
v_cmp_eq_u64_e64 s1, 0, v[6:7]
v_cndmask_b32_e64 v8, 0, 7, s1
v_cmpx_eq_u64_e32 0, v[6:7]
v_xor_b32_e32 v5, 0x40000000, v5
exp mrt0 v2, v3, v5, v8 done vm
s_endpgm
