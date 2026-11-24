; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 32-bit integer VOPC compares: v_cmp_{eq,ne,lt,le,gt,ge,f,t}_{u32,i32}, in the e32 form (VCC) and the e64 form
; (SGPR destination), with inline constants and literals as operands, on values where signed and unsigned disagree.
; The cmpx forms are in ps_vcmpx_forms; 64-bit and 16-bit compares are not covered here.
; RDNA2 ISA: _U32 compares S0.u32 with S1.u32, _I32 compares S0.i32 with S1.i32 (two's complement), per lane;
; EQ = S0 == S1, NE = S0 != S1, LT = S0 < S1, LE = S0 <= S1, GT = S0 > S1, GE = S0 >= S1, F = 0, T = 1. The result
; goes to VCC (e32) or to the SGPR pair named by VDST (e64). Inputs: the V# in s[0:3] points at 0x1000 (num_records
; 32), the test seeds 8 dwords that become v1..v6: 1, 0, 0x80000000, 0x12345678, 3, 5 (then two unused zeros).
; Five pairs (A, B) with A as src0 in e32; A is a constant or VGPR, B is a VGPR:
;   P0: A = literal 0x80000000, B = v1 = 1           unsigned A > B, signed A < B
;   P1: A = inline -1,          B = v2 = 0           unsigned A > B (0xffffffff), signed A < B
;   P2: A = inline 1,           B = v3 = 0x80000000  unsigned A < B, signed A > B
;   P3: A = literal 0x12345678, B = v4 = 0x12345678  equal
;   P4: A = v5 = 3,             B = v6 = 5           A < B both ways
; Result bit 5*k + i of a word is op k (eq, ne, lt, le, gt, ge = 0..5) on pair Pi; bit 30 is v_cmp_f on P3 (must be
; 0) and bit 31 is v_cmp_t on P0 (must be 1). The five-bit masks (bit i = Pi):
;   unsigned: eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P2 P4 = 0x14   le P2 P3 P4 = 0x1c
;             gt P0 P1 = 0x03   ge P0 P1 P3 = 0x0b
;   signed:   eq P3 = 0x08   ne P0 P1 P2 P4 = 0x17   lt P0 P1 P4 = 0x13   le P0 P1 P3 P4 = 0x1b
;             gt P2 = 0x04   ge P2 P3 = 0x0c
; word 0 (e32, u32) = 0x08 | 0x17<<5 | 0x14<<10 | 0x1c<<15 | 0x03<<20 | 0x0b<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x5000 + 0xe0000 + 0x300000 + 0x16000000 + 0x80000000 = 0x963e52e8
; word 1 (e32, i32) = 0x08 | 0x17<<5 | 0x13<<10 | 0x1b<<15 | 0x04<<20 | 0x0c<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x4c00 + 0xd8000 + 0x400000 + 0x18000000 + 0x80000000 = 0x984dcee8
; words 2 and 3 are the e64 forms with the operands swapped: src0 is the VGPR B, src1 is the constant (or v5 for
; P4), so "op(B, A)" is the mirrored relation of op(A, B): lt <-> gt and le <-> ge, eq and ne are unchanged. The
; masks by position k = eq, ne, lt, le, gt, ge are therefore the unsigned (resp. signed) masks of eq, ne, gt, ge, lt, le:
; word 2 (e64, u32) = 0x08 | 0x17<<5 | 0x03<<10 | 0x0b<<15 | 0x14<<20 | 0x1c<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0xc00 + 0x58000 + 0x1400000 + 0x38000000 + 0x80000000 = 0xb9458ee8
; word 3 (e64, i32) = 0x08 | 0x17<<5 | 0x04<<10 | 0x0c<<15 | 0x13<<20 | 0x1b<<25 | 1<<31
;                   = 0x8 + 0x2e0 + 0x1000 + 0x60000 + 0x1300000 + 0x36000000 + 0x80000000 = 0xb73612e8
; A translation that compared one of the _u32 forms signed would clear the P0/P1 ge and gt bits and set the P2 ones;
; one that compared an _i32 form unsigned would do the opposite; a v_cmp_f that returned true sets bit 30.
; Expected mrt0 bits: 0x963e52e8, 0x984dcee8, 0xb9458ee8, 0xb73612e8
s_buffer_load_dwordx8 s[4:11], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, s8
v_mov_b32_e32 v6, s9
; word 0: v_cmp_*_u32_e32
v_mov_b32_e32 v40, 0
v_cmp_eq_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_eq_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_eq_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_eq_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_eq_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_ne_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_ne_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_ne_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_ne_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_ne_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_lt_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_lt_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_lt_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_lt_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_lt_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_le_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 15, v40
v_cmp_le_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 16, v40
v_cmp_le_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 17, v40
v_cmp_le_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 18, v40
v_cmp_le_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 19, v40
v_cmp_gt_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 20, v40
v_cmp_gt_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 21, v40
v_cmp_gt_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 22, v40
v_cmp_gt_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 23, v40
v_cmp_gt_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 24, v40
v_cmp_ge_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 25, v40
v_cmp_ge_u32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 26, v40
v_cmp_ge_u32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 27, v40
v_cmp_ge_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 28, v40
v_cmp_ge_u32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 29, v40
v_cmp_f_u32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 30, v40
v_cmp_t_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 31, v40
; word 1: v_cmp_*_i32_e32
v_mov_b32_e32 v41, 0
v_cmp_eq_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_eq_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_eq_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_eq_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_eq_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_ne_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_ne_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_ne_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_ne_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_ne_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_lt_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_lt_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_lt_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_lt_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_lt_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_le_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 15, v41
v_cmp_le_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 16, v41
v_cmp_le_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 17, v41
v_cmp_le_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 18, v41
v_cmp_le_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 19, v41
v_cmp_gt_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 20, v41
v_cmp_gt_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 21, v41
v_cmp_gt_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 22, v41
v_cmp_gt_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 23, v41
v_cmp_gt_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 24, v41
v_cmp_ge_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 25, v41
v_cmp_ge_i32_e32 vcc_lo, -1, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 26, v41
v_cmp_ge_i32_e32 vcc_lo, 1, v3
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 27, v41
v_cmp_ge_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 28, v41
v_cmp_ge_i32_e32 vcc_lo, v5, v6
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 29, v41
v_cmp_f_i32_e32 vcc_lo, 0x12345678, v4
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 30, v41
v_cmp_t_i32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 31, v41
; word 2: v_cmp_*_u32_e64
v_mov_b32_e32 v42, 0
v_cmp_eq_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 0, v42
v_cmp_eq_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 1, v42
v_cmp_eq_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 2, v42
v_cmp_eq_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 3, v42
v_cmp_eq_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 4, v42
v_cmp_ne_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 5, v42
v_cmp_ne_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 6, v42
v_cmp_ne_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 7, v42
v_cmp_ne_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 8, v42
v_cmp_ne_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 9, v42
v_cmp_lt_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 10, v42
v_cmp_lt_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 11, v42
v_cmp_lt_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 12, v42
v_cmp_lt_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 13, v42
v_cmp_lt_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 14, v42
v_cmp_le_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 15, v42
v_cmp_le_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 16, v42
v_cmp_le_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 17, v42
v_cmp_le_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 18, v42
v_cmp_le_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 19, v42
v_cmp_gt_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 20, v42
v_cmp_gt_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 21, v42
v_cmp_gt_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 22, v42
v_cmp_gt_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 23, v42
v_cmp_gt_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 24, v42
v_cmp_ge_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 25, v42
v_cmp_ge_u32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 26, v42
v_cmp_ge_u32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 27, v42
v_cmp_ge_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 28, v42
v_cmp_ge_u32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 29, v42
v_cmp_f_u32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 30, v42
v_cmp_t_u32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v42, v20, 31, v42
; word 3: v_cmp_*_i32_e64
v_mov_b32_e32 v43, 0
v_cmp_eq_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 0, v43
v_cmp_eq_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 1, v43
v_cmp_eq_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 2, v43
v_cmp_eq_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 3, v43
v_cmp_eq_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 4, v43
v_cmp_ne_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 5, v43
v_cmp_ne_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 6, v43
v_cmp_ne_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 7, v43
v_cmp_ne_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 8, v43
v_cmp_ne_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 9, v43
v_cmp_lt_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 10, v43
v_cmp_lt_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 11, v43
v_cmp_lt_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 12, v43
v_cmp_lt_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 13, v43
v_cmp_lt_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 14, v43
v_cmp_le_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 15, v43
v_cmp_le_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 16, v43
v_cmp_le_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 17, v43
v_cmp_le_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 18, v43
v_cmp_le_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 19, v43
v_cmp_gt_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 20, v43
v_cmp_gt_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 21, v43
v_cmp_gt_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 22, v43
v_cmp_gt_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 23, v43
v_cmp_gt_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 24, v43
v_cmp_ge_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 25, v43
v_cmp_ge_i32_e64 s22, v2, -1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 26, v43
v_cmp_ge_i32_e64 s22, v3, 1
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 27, v43
v_cmp_ge_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 28, v43
v_cmp_ge_i32_e64 s22, v6, v5
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 29, v43
v_cmp_f_i32_e64 s22, v4, 0x12345678
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 30, v43
v_cmp_t_i32_e64 s22, v1, 0x80000000
v_cndmask_b32_e64 v20, 0, 1, s22
v_lshl_or_b32 v43, v20, 31, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
