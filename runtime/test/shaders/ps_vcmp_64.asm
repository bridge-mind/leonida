; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 64-bit integer VOPC compares: v_cmp_{lt,le,gt,ge}_{u64,i64}, v_cmp_ne_u64, v_cmp_{eq,ne}_i64 (e32 VCC and e64
; SGPR destination) and v_cmpx_{lt,ge,ne}_i64 / v_cmpx_{gt,le}_u64 (EXEC copied out and put back after each one).
; Only v_cmp_eq_u64 / v_cmpx_eq_u64 had a test. Inputs read from memory at run time: the V# in s[0:3] points at
; 0x1000 (num_records 64) and the test seeds 16 dwords, four pairs (A, B) as A.lo A.hi B.lo B.hi:
;   Q0: A = 0x00000000_80000000  B = 0x00000000_00000001   (equal high words; low words differ in bit 31)
;   Q1: A = 0xffffffff_ffffffff  B = 0x00000000_00000001   (high words decide; -1 < 1 signed, > unsigned)
;   Q2: A = 0x00000000_00000001  B = 0x00000000_80000000   (Q0 swapped)
;   Q3: A = 0x00000005_00000007  B = 0x00000005_00000007   (equal)
; RDNA2 ISA: _U64 compares S0.u64 with S1.u64, _I64 compares S0.i64 with S1.i64 (whole 64-bit values; the low word
; is never signed, so Q0 is A > B for both). Bit 4*k + i of a word is op k of that word on pair Qi:
;   Q0: u A>B, s A>B   Q1: u A>B, s A<B   Q2: u A<B, s A<B   Q3: A == B
;   lt_u64 Q2          = 0x4    lt_i64 Q1 Q2       = 0x6
;   le_u64 Q2 Q3       = 0xc    le_i64 Q1 Q2 Q3    = 0xe
;   gt_u64 Q0 Q1       = 0x3    gt_i64 Q0          = 0x1
;   ge_u64 Q0 Q1 Q3    = 0xb    ge_i64 Q0 Q3       = 0x9
;   ne Q0 Q1 Q2        = 0x7    eq Q3              = 0x8
; word 0 = lt_u64 | le_u64<<4 | gt_u64<<8 | ge_u64<<12          = 0x4 | 0xc0 | 0x300 | 0xb000 = 0x0000b3c4
; word 1 = lt_i64 | le_i64<<4 | gt_i64<<8 | ge_i64<<12          = 0x6 | 0xe0 | 0x100 | 0x9000 = 0x000091e6
; word 2 = ne_u64 | eq_i64<<4 | ne_i64<<8 | cmpx lt_i64<<12     = 0x7 | 0x80 | 0x700 | 0x6000 = 0x00006787
; word 3 = cmpx gt_u64 | ge_i64<<4 | le_u64<<8 | ne_i64<<12     = 0x3 | 0x90 | 0xc00 | 0x7000 = 0x00007c93
; A translation that compared the low words signed would set Q0's lt bits (0x80000000 < 1) and clear Q2's.
; Expected mrt0 bits: 0x0000b3c4, 0x000091e6, 0x00006787, 0x00007c93
s_buffer_load_dwordx16 s[4:19], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, s4
v_mov_b32_e32 v1, s5
v_mov_b32_e32 v2, s6
v_mov_b32_e32 v3, s7
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s9
v_mov_b32_e32 v6, s10
v_mov_b32_e32 v7, s11
v_mov_b32_e32 v8, s12
v_mov_b32_e32 v9, s13
v_mov_b32_e32 v10, s14
v_mov_b32_e32 v11, s15
v_mov_b32_e32 v12, s16
v_mov_b32_e32 v13, s17
v_mov_b32_e32 v14, s18
v_mov_b32_e32 v15, s19
s_mov_b32 s20, exec_lo
; word 0: lt_u64, le_u64, gt_u64, ge_u64
v_mov_b32_e32 v40, 0
v_cmp_lt_u64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_lt_u64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_lt_u64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_lt_u64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_le_u64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_le_u64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_le_u64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_le_u64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_gt_u64_e64 s22, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_gt_u64_e64 s22, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_gt_u64_e64 s22, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_gt_u64_e64 s22, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_ge_u64_e64 s22, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_ge_u64_e64 s22, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_ge_u64_e64 s22, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_ge_u64_e64 s22, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v40, v20, 15, v40
; word 1: lt_i64, le_i64, gt_i64, ge_i64
v_mov_b32_e32 v41, 0
v_cmp_lt_i64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_lt_i64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_lt_i64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_lt_i64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_le_i64_e64 s22, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_le_i64_e64 s22, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_le_i64_e64 s22, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_le_i64_e64 s22, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_gt_i64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_gt_i64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_gt_i64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_gt_i64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_ge_i64_e64 s22, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_ge_i64_e64 s22, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_ge_i64_e64 s22, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_ge_i64_e64 s22, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v41, v20, 15, v41
; word 2: ne_u64, eq_i64, ne_i64, x:lt_i64
v_mov_b32_e32 v42, 0
v_cmp_ne_u64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 0, v42
v_cmp_ne_u64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 1, v42
v_cmp_ne_u64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 2, v42
v_cmp_ne_u64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 3, v42
v_cmp_eq_i64_e64 s22, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 4, v42
v_cmp_eq_i64_e64 s22, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 5, v42
v_cmp_eq_i64_e64 s22, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 6, v42
v_cmp_eq_i64_e64 s22, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 7, v42
v_cmp_ne_i64_e32 vcc_lo, v[0:1], v[2:3]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 8, v42
v_cmp_ne_i64_e32 vcc_lo, v[4:5], v[6:7]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 9, v42
v_cmp_ne_i64_e32 vcc_lo, v[8:9], v[10:11]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 10, v42
v_cmp_ne_i64_e32 vcc_lo, v[12:13], v[14:15]
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 11, v42
v_cmpx_lt_i64_e32 v[0:1], v[2:3]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 12, v42
v_cmpx_lt_i64_e32 v[4:5], v[6:7]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 13, v42
v_cmpx_lt_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 14, v42
v_cmpx_lt_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v42, v20, 15, v42
; word 3: x:gt_u64, x:ge_i64, x:le_u64, x:ne_i64
v_mov_b32_e32 v43, 0
v_cmpx_gt_u64_e32 v[0:1], v[2:3]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 0, v43
v_cmpx_gt_u64_e32 v[4:5], v[6:7]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 1, v43
v_cmpx_gt_u64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 2, v43
v_cmpx_gt_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 3, v43
v_cmpx_ge_i64_e32 v[0:1], v[2:3]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 4, v43
v_cmpx_ge_i64_e32 v[4:5], v[6:7]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 5, v43
v_cmpx_ge_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 6, v43
v_cmpx_ge_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 7, v43
v_cmpx_le_u64_e32 v[0:1], v[2:3]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 8, v43
v_cmpx_le_u64_e32 v[4:5], v[6:7]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 9, v43
v_cmpx_le_u64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 10, v43
v_cmpx_le_u64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 11, v43
v_cmpx_ne_i64_e32 v[0:1], v[2:3]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 12, v43
v_cmpx_ne_i64_e32 v[4:5], v[6:7]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 13, v43
v_cmpx_ne_i64_e32 v[8:9], v[10:11]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 14, v43
v_cmpx_ne_i64_e32 v[12:13], v[14:15]
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v20, 0, 1, s21
v_lshl_or_b32 v43, v20, 15, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
