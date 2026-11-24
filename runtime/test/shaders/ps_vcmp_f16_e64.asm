; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 16-bit float VOPC compares, e64 form (SGPR destination) with swapped operands: v_cmp_{f,lt,eq,le,gt,lg,ge,o,u,nge,nlg,
; ngt,nle,neq,nlt,tru}_f16_e64 s4, B, A. Same seven operand pairs as ps_vcmp_f16 (see its header for the ISA rules and
; the case list), but src0 is the VGPR B and src1 is A (the inline constant for k0..k2, the VGPR otherwise), so each
; op sees the mirrored relation: less <-> greater, equal and unordered unchanged.
; Cases (A to B as in ps_vcmp_f16): k0 1.0/2.0, k1 2.0/1.0, k2 +0/-0, k3 qNaN/1.0, k4 1.0/sNaN,
; k5 +inf/-inf, k6 denormal 0x0001/0x0002; the high halves of the VGPR operands hold junk.
; Relation of B (src0) to A (src1): LESS = {k1,k5} = 0x22, GREATER = {k0,k6} = 0x41,
; EQUAL = {k2} = 0x04, UNORDERED = {k3,k4} = 0x18. Masks (bit k = result on case k):
;   f 0x00   lt 0x22   eq 0x04   le 0x26   gt 0x41   lg 0x63   ge 0x45   o 0x67
;   u 0x18   nge 0x3a  nlg 0x1c  ngt 0x3e  nle 0x59  neq 0x7b  nlt 0x5d  tru 0x7f
; Word w holds ops 4w..4w+3, op j at bit 7*j + k, plus bit 30 (no denormal or NaN export pattern):
; word 0 = 0x40000000 | 0x00 | 0x22<<7 | 0x04<<14 | 0x26<<21 = 0x40000000 + 0x1100 + 0x10000 + 0x4c00000 = 0x44c11100
; word 1 = 0x40000000 | 0x41 | 0x63<<7 | 0x45<<14 | 0x67<<21 = 0x40000000 + 0x31c1 + 0x114000 + 0xce00000 = 0x4cf171c1
; word 2 = 0x40000000 | 0x18 | 0x3a<<7 | 0x1c<<14 | 0x3e<<21 = 0x40000000 + 0x1d18 + 0x70000 + 0x7c00000 = 0x47c71d18
; word 3 = 0x40000000 | 0x59 | 0x7b<<7 | 0x5d<<14 | 0x7f<<21 = 0x40000000 + 0x3dd9 + 0x174000 + 0xfe00000 = 0x4ff77dd9
; Expected mrt0 bits: 0x44c11100, 0x4cf171c1, 0x47c71d18, 0x4ff77dd9
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
v_cmp_f_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 0, v40
v_cmp_f_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 1, v40
v_cmp_f_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 2, v40
v_cmp_f_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 3, v40
v_cmp_f_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 4, v40
v_cmp_f_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 5, v40
v_cmp_f_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 6, v40
v_cmp_lt_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 7, v40
v_cmp_lt_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 8, v40
v_cmp_lt_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 9, v40
v_cmp_lt_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 10, v40
v_cmp_lt_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 11, v40
v_cmp_lt_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 12, v40
v_cmp_lt_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 13, v40
v_cmp_eq_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 14, v40
v_cmp_eq_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 15, v40
v_cmp_eq_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 16, v40
v_cmp_eq_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 17, v40
v_cmp_eq_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 18, v40
v_cmp_eq_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 19, v40
v_cmp_eq_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 20, v40
v_cmp_le_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 21, v40
v_cmp_le_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 22, v40
v_cmp_le_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 23, v40
v_cmp_le_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 24, v40
v_cmp_le_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 25, v40
v_cmp_le_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 26, v40
v_cmp_le_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v40, v20, 27, v40
v_mov_b32_e32 v41, 0x40000000
v_cmp_gt_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 0, v41
v_cmp_gt_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 1, v41
v_cmp_gt_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 2, v41
v_cmp_gt_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 3, v41
v_cmp_gt_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 4, v41
v_cmp_gt_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 5, v41
v_cmp_gt_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 6, v41
v_cmp_lg_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 7, v41
v_cmp_lg_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 8, v41
v_cmp_lg_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 9, v41
v_cmp_lg_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 10, v41
v_cmp_lg_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 11, v41
v_cmp_lg_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 12, v41
v_cmp_lg_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 13, v41
v_cmp_ge_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 14, v41
v_cmp_ge_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 15, v41
v_cmp_ge_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 16, v41
v_cmp_ge_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 17, v41
v_cmp_ge_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 18, v41
v_cmp_ge_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 19, v41
v_cmp_ge_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 20, v41
v_cmp_o_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 21, v41
v_cmp_o_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 22, v41
v_cmp_o_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 23, v41
v_cmp_o_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 24, v41
v_cmp_o_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 25, v41
v_cmp_o_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 26, v41
v_cmp_o_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v41, v20, 27, v41
v_mov_b32_e32 v42, 0x40000000
v_cmp_u_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 0, v42
v_cmp_u_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 1, v42
v_cmp_u_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 2, v42
v_cmp_u_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 3, v42
v_cmp_u_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 4, v42
v_cmp_u_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 5, v42
v_cmp_u_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 6, v42
v_cmp_nge_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 7, v42
v_cmp_nge_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 8, v42
v_cmp_nge_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 9, v42
v_cmp_nge_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 10, v42
v_cmp_nge_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 11, v42
v_cmp_nge_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 12, v42
v_cmp_nge_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 13, v42
v_cmp_nlg_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 14, v42
v_cmp_nlg_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 15, v42
v_cmp_nlg_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 16, v42
v_cmp_nlg_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 17, v42
v_cmp_nlg_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 18, v42
v_cmp_nlg_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 19, v42
v_cmp_nlg_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 20, v42
v_cmp_ngt_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 21, v42
v_cmp_ngt_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 22, v42
v_cmp_ngt_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 23, v42
v_cmp_ngt_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 24, v42
v_cmp_ngt_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 25, v42
v_cmp_ngt_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 26, v42
v_cmp_ngt_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v42, v20, 27, v42
v_mov_b32_e32 v43, 0x40000000
v_cmp_nle_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 0, v43
v_cmp_nle_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 1, v43
v_cmp_nle_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 2, v43
v_cmp_nle_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 3, v43
v_cmp_nle_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 4, v43
v_cmp_nle_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 5, v43
v_cmp_nle_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 6, v43
v_cmp_neq_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 7, v43
v_cmp_neq_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 8, v43
v_cmp_neq_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 9, v43
v_cmp_neq_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 10, v43
v_cmp_neq_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 11, v43
v_cmp_neq_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 12, v43
v_cmp_neq_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 13, v43
v_cmp_nlt_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 14, v43
v_cmp_nlt_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 15, v43
v_cmp_nlt_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 16, v43
v_cmp_nlt_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 17, v43
v_cmp_nlt_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 18, v43
v_cmp_nlt_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 19, v43
v_cmp_nlt_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 20, v43
v_cmp_tru_f16_e64 s4, v8, 1.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 21, v43
v_cmp_tru_f16_e64 s4, v9, 2.0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 22, v43
v_cmp_tru_f16_e64 s4, v10, 0
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 23, v43
v_cmp_tru_f16_e64 s4, v11, v4
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 24, v43
v_cmp_tru_f16_e64 s4, v12, v5
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 25, v43
v_cmp_tru_f16_e64 s4, v13, v6
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 26, v43
v_cmp_tru_f16_e64 s4, v14, v7
v_cndmask_b32_e64 v20, 0, 1, s4
v_lshl_or_b32 v43, v20, 27, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
