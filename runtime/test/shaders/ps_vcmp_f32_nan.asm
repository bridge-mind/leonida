; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; f32 VOPC compares with a quiet NaN, inputs read from memory at run time (not inline constants, which the driver
; could fold): v_cmp_{ge,lg,neq,nge}_f32 (e32, VCC), v_cmp_{ngt,nle,nlt,tru}_f32 (e64, SGPR destination) and
; v_cmpx_{ge,lg,nge,nlt,neq,o,tru,f}_f32 (EXEC is copied out and put back after each one). None of these had a test.
; The V# in s[0:3] points at 0x1000 (num_records 16); the test seeds 0x1000..0x1008 with 0x7fc10001 0x3f800000
; 0x40000000, so v1 = NaN, v2 = 1.0, v3 = 2.0.
; Each compare runs on four pairs; bit 4*k + i of a word is op k of that word on pair i:
;   P0 = (NaN, 1.0)   P1 = (1.0, 2.0)   P2 = (2.0, 1.0)   P3 = (1.0, 1.0)
; RDNA2 ISA VOPC definitions: GE = S0 >= S1, LG = S0 <> S1 (ordered: false on NaN), O = !isNan(S0) && !isNan(S1),
; F = 0, TRU = 1; the N forms are the negations, so true on NaN: NGE = !(S0 >= S1), NGT = !(S0 > S1),
; NLE = !(S0 <= S1), NLT = !(S0 < S1), NEQ = !(S0 == S1). Nibble per op, bits P3 P2 P1 P0:
;   ge  P2 P3                = 0b1100 = 0xc     ngt  P0 P1 P3 = 0b1011 = 0xb
;   lg  P1 P2                = 0b0110 = 0x6     nle  P0 P2    = 0b0101 = 0x5
;   neq P0 P1 P2             = 0b0111 = 0x7     nlt  P0 P2 P3 = 0b1101 = 0xd
;   nge P0 P1                = 0b0011 = 0x3     tru  all      = 0xf
;   o   P1 P2 P3             = 0b1110 = 0xe     f    none     = 0x0
; word 0 = ge | lg<<4 | neq<<8 | nge<<12            = 0xc | 0x60 | 0x700 | 0x3000 = 0x0000376c
; word 1 = ngt | nle<<4 | nlt<<8 | tru<<12          = 0xb | 0x50 | 0xd00 | 0xf000 = 0x0000fd5b
; word 2 = cmpx ge | lg<<4 | nge<<8 | nlt<<12       = 0xc | 0x60 | 0x300 | 0xd000 = 0x0000d36c
; word 3 = cmpx neq | o<<4 | tru<<8 | f<<12         = 0x7 | 0xe0 | 0xf00 | 0x0000 = 0x00000fe7
; Expected mrt0 bits: 0x0000376c, 0x0000fd5b, 0x0000d36c, 0x00000fe7
s_buffer_load_dwordx4 s[4:7], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
s_mov_b32 s20, exec_lo
; word 0: ge, lg, neq, nge
v_mov_b32_e32 v40, 0
v_cmp_ge_f32_e32 vcc_lo, v1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_ge_f32_e32 vcc_lo, v2, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_ge_f32_e32 vcc_lo, v3, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_ge_f32_e32 vcc_lo, v2, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_lg_f32_e32 vcc_lo, v1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_lg_f32_e32 vcc_lo, v2, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_lg_f32_e32 vcc_lo, v3, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_lg_f32_e32 vcc_lo, v2, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_neq_f32_e32 vcc_lo, v1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_neq_f32_e32 vcc_lo, v2, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_neq_f32_e32 vcc_lo, v3, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_neq_f32_e32 vcc_lo, v2, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_nge_f32_e32 vcc_lo, v1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_nge_f32_e32 vcc_lo, v2, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_nge_f32_e32 vcc_lo, v3, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_nge_f32_e32 vcc_lo, v2, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 15, v40
; word 1: ngt, nle, nlt, tru
v_mov_b32_e32 v41, 0
v_cmp_ngt_f32_e64 s22, v1, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_ngt_f32_e64 s22, v2, v3
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_ngt_f32_e64 s22, v3, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_ngt_f32_e64 s22, v2, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_nle_f32_e64 s22, v1, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_nle_f32_e64 s22, v2, v3
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_nle_f32_e64 s22, v3, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_nle_f32_e64 s22, v2, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_nlt_f32_e64 s22, v1, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_nlt_f32_e64 s22, v2, v3
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_nlt_f32_e64 s22, v3, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_nlt_f32_e64 s22, v2, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_tru_f32_e64 s22, v1, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_tru_f32_e64 s22, v2, v3
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_tru_f32_e64 s22, v3, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_tru_f32_e64 s22, v2, v2
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 15, v41
; word 2: x:ge, x:lg, x:nge, x:nlt
v_mov_b32_e32 v42, 0
v_cmpx_ge_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 0, v42
v_cmpx_ge_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 1, v42
v_cmpx_ge_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 2, v42
v_cmpx_ge_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 3, v42
v_cmpx_lg_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 4, v42
v_cmpx_lg_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 5, v42
v_cmpx_lg_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 6, v42
v_cmpx_lg_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 7, v42
v_cmpx_nge_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 8, v42
v_cmpx_nge_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 9, v42
v_cmpx_nge_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 10, v42
v_cmpx_nge_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 11, v42
v_cmpx_nlt_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 12, v42
v_cmpx_nlt_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 13, v42
v_cmpx_nlt_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 14, v42
v_cmpx_nlt_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 15, v42
; word 3: x:neq, x:o, x:tru, x:f
v_mov_b32_e32 v43, 0
v_cmpx_neq_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 0, v43
v_cmpx_neq_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 1, v43
v_cmpx_neq_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 2, v43
v_cmpx_neq_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 3, v43
v_cmpx_o_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 4, v43
v_cmpx_o_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 5, v43
v_cmpx_o_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 6, v43
v_cmpx_o_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 7, v43
v_cmpx_tru_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 8, v43
v_cmpx_tru_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 9, v43
v_cmpx_tru_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 10, v43
v_cmpx_tru_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 11, v43
v_cmpx_f_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 12, v43
v_cmpx_f_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 13, v43
v_cmpx_f_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 14, v43
v_cmpx_f_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 15, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
