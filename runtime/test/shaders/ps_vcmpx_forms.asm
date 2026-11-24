; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The 14 v_cmpx forms no test ran: v_cmpx_{eq,gt,le,lt,ngt,nle}_f32, v_cmpx_f_i32, v_cmpx_t_u32,
; v_cmpx_{eq,gt,le}_i64 and v_cmpx_{ge,lt,ne}_u64. RDNA2 ISA: EXEC = EXEC & compare(S0, S1), per lane; a v_cmpx
; writes EXEC only, so VCC is kept. Each one runs alone: EXEC is copied out and put back after it (s20 holds the
; EXEC the shader started with). Inputs are read from memory at run time, not inline constants, which the driver
; could fold. The V# in s[0:3] points at 0x1000 (num_records 96); the test seeds 24 dwords:
;   0x1000: 0x7fc10001 0x3f800000 0x40000000 0x80000000 0x00000000 0xff800000 (then two unused zeros)
;           v1 = NaN   v2 = 1.0   v3 = 2.0   v4 = -0.0  v5 = +0.0  v6 = -infinity
;   0x1020: four 64-bit pairs (A, B) as A.lo A.hi B.lo B.hi, in v[8:9], v[10:11] ... v[20:21], v[22:23]:
;     Q0: A = 0x00000000_80000000  B = 0x00000000_00000001  equal high words, low words differ in bit 31
;     Q1: A = 0x80000000_00000007  B = 0x00000000_00000007  equal low words; A > B unsigned, A < B signed
;     Q2: A = 0x00000005_00000001  B = 0x00000005_80000000  the low words of Q0 swapped, under a non-zero high word
;     Q3: A = 0x00000005_00000007  B = 0x00000005_00000007  equal
; ISA compare definitions: EQ = S0 == S1, GT = S0 > S1, LE = S0 <= S1, LT = S0 < S1 (ordered: false when a source
; is NaN; -0 == +0), NGT = !(S0 > S1), NLE = !(S0 <= S1) (true when a source is NaN), F = 0, T = 1. _U64 / _I64
; compare the whole 64-bit values unsigned / signed (the low word is never signed).
; Bit 4*k + i of words 0..2 is op k of that word on pair i.
; word 0, e32, f32 pairs P0 = (NaN, 1.0)  P1 = (1.0, 2.0)  P2 = (2.0, 1.0)  P3 = (1.0, 1.0). Nibbles, bits P3..P0:
;   eq  P3       = 0x8    gt  P2    = 0x4    le    P1 P3 = 0xa    lt    P1  = 0x2
;   ngt P0 P1 P3 = 0xb    nle P0 P2 = 0x5    f_i32 none  = 0x0    t_u32 all = 0xf
;   0x8 | 0x40 | 0xa00 | 0x2000 | 0xb0000 | 0x500000 | 0x0000000 | 0xf0000000 = 0xf05b2a48
; word 1, e64 (VOP3), f32 pairs P4 = (2.0, 1.0)  P5 = (-0.0, +0.0)  P6 = (NaN, NaN)  P7 = (-infinity, 1.0):
;   eq  P5       = 0x2    gt  P4    = 0x1    le    P5 P7 = 0xa    lt    P7  = 0x8
;   ngt P5 P6 P7 = 0xe    nle P4 P6 = 0x5    f_i32 none  = 0x0    t_u32 all = 0xf
;   0x2 | 0x10 | 0xa00 | 0x8000 | 0xe0000 | 0x500000 | 0x0000000 | 0xf0000000 = 0xf05e8a12
; word 2, 64-bit pairs Q0..Q3 (Q0: A > B both ways, Q1: A > B unsigned and A < B signed, Q2: A < B both ways):
;   eq_i64 Q3       = 0x8    gt_i64 Q0 = 0x1    le_i64 Q1 Q2 Q3 = 0xe
;   ge_u64 Q0 Q1 Q3 = 0xb    lt_u64 Q2 = 0x4    ne_u64 Q0 Q1 Q2 = 0x7    then e64: gt_i64 = 0x1, ge_u64 = 0xb
;   0x8 | 0x10 | 0xe00 | 0xb000 | 0x40000 | 0x700000 | 0x1000000 | 0xb0000000 = 0xb174be18
;   A translation that compared only the low words would call Q1 equal; one that compared only the high words
;   would call Q0 and Q2 equal; one that took the low word as signed would swap Q0 and Q2.
; word 3, the 14 ops in the order above (k = 0..13), all e32:
;   bit k: EXEC is emptied first (v_cmpx_ne_u32 of equal values), then op k runs with operands that make it true
;          (f_i32 never is). v_cmpx only ANDs into EXEC, so no lane comes back: 0 for every k.
;   bit 16 + k: VCC is kept. Even k: VCC = 1, then op k with operands that make it false; the bit is VCC = 1.
;          Odd k: VCC = 0, then op k with operands that make it true; the bit is VCC = 0. A v_cmpx that also wrote
;          its result to VCC would flip every one of these bits.
;   bits 16, 18, ... 28 = 0x15550000
; Expected mrt0 bits: 0xf05b2a48, 0xf05e8a12, 0xb174be18, 0x15550000
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s9
s_buffer_load_dwordx16 s[4:19], s[0:3], 0x20
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v8, s4
v_mov_b32_e32 v9, s5
v_mov_b32_e32 v10, s6
v_mov_b32_e32 v11, s7
v_mov_b32_e32 v12, s8
v_mov_b32_e32 v13, s9
v_mov_b32_e32 v14, s10
v_mov_b32_e32 v15, s11
v_mov_b32_e32 v16, s12
v_mov_b32_e32 v17, s13
v_mov_b32_e32 v18, s14
v_mov_b32_e32 v19, s15
v_mov_b32_e32 v20, s16
v_mov_b32_e32 v21, s17
v_mov_b32_e32 v22, s18
v_mov_b32_e32 v23, s19
s_mov_b32 s20, exec_lo
; word 0: e32 eq, gt, le, lt, ngt, nle _f32 on P0..P3, then f_i32 and t_u32
v_mov_b32_e32 v40, 0
v_cmpx_eq_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 0, v40
v_cmpx_eq_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 1, v40
v_cmpx_eq_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 2, v40
v_cmpx_eq_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 3, v40
v_cmpx_gt_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 4, v40
v_cmpx_gt_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 5, v40
v_cmpx_gt_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 6, v40
v_cmpx_gt_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 7, v40
v_cmpx_le_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 8, v40
v_cmpx_le_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 9, v40
v_cmpx_le_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 10, v40
v_cmpx_le_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 11, v40
v_cmpx_lt_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 12, v40
v_cmpx_lt_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 13, v40
v_cmpx_lt_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 14, v40
v_cmpx_lt_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 15, v40
v_cmpx_ngt_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 16, v40
v_cmpx_ngt_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 17, v40
v_cmpx_ngt_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 18, v40
v_cmpx_ngt_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 19, v40
v_cmpx_nle_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 20, v40
v_cmpx_nle_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 21, v40
v_cmpx_nle_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 22, v40
v_cmpx_nle_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 23, v40
v_cmpx_f_i32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 24, v40
v_cmpx_f_i32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 25, v40
v_cmpx_f_i32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 26, v40
v_cmpx_f_i32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 27, v40
v_cmpx_t_u32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 28, v40
v_cmpx_t_u32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 29, v40
v_cmpx_t_u32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 30, v40
v_cmpx_t_u32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 31, v40
; word 1: the same eight in the e64 (VOP3) encoding, on P4..P7
v_mov_b32_e32 v41, 0
v_cmpx_eq_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 0, v41
v_cmpx_eq_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 1, v41
v_cmpx_eq_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 2, v41
v_cmpx_eq_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 3, v41
v_cmpx_gt_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 4, v41
v_cmpx_gt_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 5, v41
v_cmpx_gt_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 6, v41
v_cmpx_gt_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 7, v41
v_cmpx_le_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 8, v41
v_cmpx_le_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 9, v41
v_cmpx_le_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 10, v41
v_cmpx_le_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 11, v41
v_cmpx_lt_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 12, v41
v_cmpx_lt_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 13, v41
v_cmpx_lt_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 14, v41
v_cmpx_lt_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 15, v41
v_cmpx_ngt_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 16, v41
v_cmpx_ngt_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 17, v41
v_cmpx_ngt_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 18, v41
v_cmpx_ngt_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 19, v41
v_cmpx_nle_f32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 20, v41
v_cmpx_nle_f32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 21, v41
v_cmpx_nle_f32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 22, v41
v_cmpx_nle_f32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 23, v41
v_cmpx_f_i32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 24, v41
v_cmpx_f_i32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 25, v41
v_cmpx_f_i32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 26, v41
v_cmpx_f_i32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 27, v41
v_cmpx_t_u32_e64 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 28, v41
v_cmpx_t_u32_e64 v4, v5
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 29, v41
v_cmpx_t_u32_e64 v1, v1
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 30, v41
v_cmpx_t_u32_e64 v6, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 31, v41
; word 2: e32 eq, gt, le _i64 and ge, lt, ne _u64 on Q0..Q3, then e64 gt_i64 and ge_u64
v_mov_b32_e32 v42, 0
v_cmpx_eq_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 0, v42
v_cmpx_eq_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 1, v42
v_cmpx_eq_i64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 2, v42
v_cmpx_eq_i64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 3, v42
v_cmpx_gt_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 4, v42
v_cmpx_gt_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 5, v42
v_cmpx_gt_i64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 6, v42
v_cmpx_gt_i64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 7, v42
v_cmpx_le_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 8, v42
v_cmpx_le_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 9, v42
v_cmpx_le_i64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 10, v42
v_cmpx_le_i64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 11, v42
v_cmpx_ge_u64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 12, v42
v_cmpx_ge_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 13, v42
v_cmpx_ge_u64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 14, v42
v_cmpx_ge_u64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 15, v42
v_cmpx_lt_u64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 16, v42
v_cmpx_lt_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 17, v42
v_cmpx_lt_u64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 18, v42
v_cmpx_lt_u64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 19, v42
v_cmpx_ne_u64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 20, v42
v_cmpx_ne_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 21, v42
v_cmpx_ne_u64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 22, v42
v_cmpx_ne_u64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 23, v42
v_cmpx_gt_i64_e64 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 24, v42
v_cmpx_gt_i64_e64 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 25, v42
v_cmpx_gt_i64_e64 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 26, v42
v_cmpx_gt_i64_e64 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 27, v42
v_cmpx_ge_u64_e64 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 28, v42
v_cmpx_ge_u64_e64 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 29, v42
v_cmpx_ge_u64_e64 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 30, v42
v_cmpx_ge_u64_e64 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 31, v42
; word 3, bits 0..13: EXEC emptied first, then op k with operands that make it true (f_i32 never is)
v_mov_b32_e32 v43, 0
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_eq_f32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 0, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_gt_f32_e32 v3, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 1, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_le_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 2, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_lt_f32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 3, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_ngt_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 4, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_nle_f32_e32 v1, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 5, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_f_i32_e32 v2, v2
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 6, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_t_u32_e32 v2, v3
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 7, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_eq_i64_e32 v[20:21], v[22:23]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 8, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_gt_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 9, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_le_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 10, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_ge_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 11, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_lt_u64_e32 v[16:17], v[18:19]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 12, v43
v_cmpx_ne_u32_e32 v2, v2
v_cmpx_ne_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 13, v43
; word 3, bits 16..29: VCC is kept. Even k: VCC = 1, op k false; odd k: VCC = 0, op k true
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_eq_f32_e32 v1, v1
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 16, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_gt_f32_e32 v3, v2
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 17, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_le_f32_e32 v3, v2
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 18, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_lt_f32_e32 v2, v3
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 19, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_ngt_f32_e32 v3, v2
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 20, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_nle_f32_e32 v1, v2
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 21, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_f_i32_e32 v2, v2
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 22, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_t_u32_e32 v2, v3
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 23, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_eq_i64_e32 v[12:13], v[14:15]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 24, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_gt_i64_e32 v[8:9], v[10:11]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 25, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_le_i64_e32 v[8:9], v[10:11]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 26, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_ge_u64_e32 v[12:13], v[14:15]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 27, v43
v_cmp_eq_u32_e32 vcc_lo, v2, v2
v_cmpx_lt_u64_e32 v[12:13], v[14:15]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 28, v43
v_cmp_ne_u32_e32 vcc_lo, v2, v2
v_cmpx_ne_u64_e32 v[12:13], v[14:15]
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 29, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
