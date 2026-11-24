; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 16-bit float VOPC compares, e32 form (VCC destination): v_cmp_{f,lt,eq,le,gt,lg,ge,o,u,nge,nlg,ngt,nle,neq,nlt,tru}_f16
; (all 16 IEEE predicates) on seven operand pairs covering the four relations less, greater, equal and unordered, with
; +-0, +-inf, quiet and signalling NaN and f16 denormals, and junk in the high halves of the register operands.
; RDNA2 ISA (VOPC, _F16): the compare reads S0.f16 and S1.f16 (bits 15:0 of the operands only) and writes one bit per lane
; to VCC. Per the ISA pseudocode, for the relation of S0 to S1:
;   F = 0, TRU = 1; LT: S0 < S1; EQ: S0 == S1; LE: S0 <= S1; GT: S0 > S1; LG: S0 <> S1 (less or greater);
;   GE: S0 >= S1; O: !isNAN(S0) && !isNAN(S1); U: isNAN(S0) || isNAN(S1); NGE = !(S0 >= S1); NLG = !(S0 <> S1);
;   NGT = !(S0 > S1); NLE = !(S0 <= S1); NEQ = !(S0 == S1); NLT = !(S0 < S1).
; Every ordered compare with a NaN operand is false, so the N* forms (negations) are true for an unordered pair; +0 == -0;
; the f16 denormals 0x0001 and 0x0002 are kept (they are different numbers, 0x0001 < 0x0002). Cases (A = src0, B = src1,
; the VGPR operand is shown as junk:value, junk = high half that must be ignored):
;   k0: A = inline 1.0 (0x3c00),   B = 0x1234:0x4000 (2.0)            less
;   k1: A = inline 2.0 (0x4000),   B = 0xabcd:0x3c00 (1.0)            greater
;   k2: A = inline 0 (+0),         B = 0xffff:0x8000 (-0)             equal
;   k3: A = 0xffff:0x7e00 (qNaN),  B = 0x0001:0x3c00 (1.0)            unordered
;   k4: A = 0x5555:0x3c00 (1.0),   B = 0x7c00:0x7c01 (sNaN)           unordered
;   k5: A = 0x0001:0x7c00 (+inf),  B = 0xbeef:0xfc00 (-inf)           greater
;   k6: A = 0xdead:0x0001 (denormal), B = 0x8000:0x0002 (denormal)    less
; Relation sets: LESS = {k0,k6} = 0x41, GREATER = {k1,k5} = 0x22, EQUAL = {k2} = 0x04, UNORDERED = {k3,k4} = 0x18.
; Mask of an op (bit k = result on case k) is the union of the sets it is true for:
;   f 0x00   lt 0x41   eq 0x04   le 0x45   gt 0x22   lg 0x63   ge 0x26   o 0x67
;   u 0x18   nge 0x59  nlg 0x1c  ngt 0x5d  nle 0x3a  neq 0x7b  nlt 0x3e  tru 0x7f
; Word w holds ops 4w..4w+3 (the order above), op j of the word at bit 7*j + k, plus bit 30 so the float export never
; sees a denormal or NaN pattern:
; word 0 = 0x40000000 | 0x00 | 0x41<<7 | 0x04<<14 | 0x45<<21 = 0x40000000 + 0x2080 + 0x10000 + 0x8a00000 = 0x48a12080
; word 1 = 0x40000000 | 0x22 | 0x63<<7 | 0x26<<14 | 0x67<<21 = 0x40000000 + 0x31a2 + 0x98000 + 0xce00000 = 0x4ce9b1a2
; word 2 = 0x40000000 | 0x18 | 0x59<<7 | 0x1c<<14 | 0x5d<<21 = 0x40000000 + 0x2c98 + 0x70000 + 0xba00000 = 0x4ba72c98
; word 3 = 0x40000000 | 0x3a | 0x7b<<7 | 0x3e<<14 | 0x7f<<21 = 0x40000000 + 0x3dba + 0xf8000 + 0xfe00000 = 0x4fefbdba
; A translation that compared the whole 32-bit registers, flushed the denormals (k6 would become equal), ordered a NaN
; (k3, k4 true for lt/gt/le/ge...), or took -0 below +0 (k2) moves bits in the words.
; Expected mrt0 bits: 0x48a12080, 0x4ce9b1a2, 0x4ba72c98, 0x4fefbdba
v_mov_b32_e32 v8, 0x12344000
v_mov_b32_e32 v9, 0xabcd3c00
v_mov_b32_e32 v10, 0xffff8000
v_mov_b32_e32 v4, 0xffff7e00
v_mov_b32_e32 v11, 0x00013c00
v_mov_b32_e32 v5, 0x55553c00
v_mov_b32_e32 v12, 0x7c007c01
v_mov_b32_e32 v6, 0x00017c00
v_mov_b32_e32 v13, 0xbeeffc00
v_mov_b32_e32 v7, 0xdead0001
v_mov_b32_e32 v14, 0x80000002
v_mov_b32_e32 v40, 0x40000000
v_cmp_f_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_f_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_f_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_f_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_f_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_f_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_f_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_lt_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_lt_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_lt_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_lt_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_lt_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_lt_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_lt_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_eq_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_eq_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 15, v40
v_cmp_eq_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 16, v40
v_cmp_eq_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 17, v40
v_cmp_eq_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 18, v40
v_cmp_eq_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 19, v40
v_cmp_eq_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 20, v40
v_cmp_le_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 21, v40
v_cmp_le_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 22, v40
v_cmp_le_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 23, v40
v_cmp_le_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 24, v40
v_cmp_le_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 25, v40
v_cmp_le_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 26, v40
v_cmp_le_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v40, v20, 27, v40
v_mov_b32_e32 v41, 0x40000000
v_cmp_gt_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_gt_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_gt_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_gt_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_gt_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_gt_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_gt_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_lg_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_lg_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_lg_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_lg_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_lg_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_lg_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_lg_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_ge_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_ge_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 15, v41
v_cmp_ge_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 16, v41
v_cmp_ge_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 17, v41
v_cmp_ge_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 18, v41
v_cmp_ge_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 19, v41
v_cmp_ge_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 20, v41
v_cmp_o_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 21, v41
v_cmp_o_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 22, v41
v_cmp_o_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 23, v41
v_cmp_o_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 24, v41
v_cmp_o_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 25, v41
v_cmp_o_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 26, v41
v_cmp_o_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v41, v20, 27, v41
v_mov_b32_e32 v42, 0x40000000
v_cmp_u_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 0, v42
v_cmp_u_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 1, v42
v_cmp_u_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 2, v42
v_cmp_u_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 3, v42
v_cmp_u_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 4, v42
v_cmp_u_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 5, v42
v_cmp_u_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 6, v42
v_cmp_nge_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 7, v42
v_cmp_nge_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 8, v42
v_cmp_nge_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 9, v42
v_cmp_nge_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 10, v42
v_cmp_nge_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 11, v42
v_cmp_nge_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 12, v42
v_cmp_nge_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 13, v42
v_cmp_nlg_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 14, v42
v_cmp_nlg_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 15, v42
v_cmp_nlg_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 16, v42
v_cmp_nlg_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 17, v42
v_cmp_nlg_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 18, v42
v_cmp_nlg_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 19, v42
v_cmp_nlg_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 20, v42
v_cmp_ngt_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 21, v42
v_cmp_ngt_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 22, v42
v_cmp_ngt_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 23, v42
v_cmp_ngt_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 24, v42
v_cmp_ngt_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 25, v42
v_cmp_ngt_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 26, v42
v_cmp_ngt_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v42, v20, 27, v42
v_mov_b32_e32 v43, 0x40000000
v_cmp_nle_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 0, v43
v_cmp_nle_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 1, v43
v_cmp_nle_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 2, v43
v_cmp_nle_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 3, v43
v_cmp_nle_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 4, v43
v_cmp_nle_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 5, v43
v_cmp_nle_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 6, v43
v_cmp_neq_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 7, v43
v_cmp_neq_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 8, v43
v_cmp_neq_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 9, v43
v_cmp_neq_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 10, v43
v_cmp_neq_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 11, v43
v_cmp_neq_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 12, v43
v_cmp_neq_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 13, v43
v_cmp_nlt_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 14, v43
v_cmp_nlt_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 15, v43
v_cmp_nlt_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 16, v43
v_cmp_nlt_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 17, v43
v_cmp_nlt_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 18, v43
v_cmp_nlt_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 19, v43
v_cmp_nlt_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 20, v43
v_cmp_tru_f16_e32 vcc_lo, 1.0, v8
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 21, v43
v_cmp_tru_f16_e32 vcc_lo, 2.0, v9
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 22, v43
v_cmp_tru_f16_e32 vcc_lo, 0, v10
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 23, v43
v_cmp_tru_f16_e32 vcc_lo, v4, v11
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 24, v43
v_cmp_tru_f16_e32 vcc_lo, v5, v12
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 25, v43
v_cmp_tru_f16_e32 vcc_lo, v6, v13
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 26, v43
v_cmp_tru_f16_e32 vcc_lo, v7, v14
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_lshl_or_b32 v43, v20, 27, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
