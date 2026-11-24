; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; 16-bit float v_cmpx, e32 form: v_cmpx_{f,lt,eq,le,gt,lg,ge,o,u,nge,nlg,ngt,nle,neq,nlt,tru}_f16 on the seven operand
; pairs of ps_vcmp_f16 (see that header for the case list and the ISA rules of the 16 predicates: bits 15:0 of the
; operands only, ordered compares false for a NaN, +0 == -0, denormals kept). A v_cmpx writes the result to EXEC
; (EXEC = EXEC & compare, per lane) and to nothing else, so VCC is kept. Each compare runs alone: EXEC is copied out
; and put back after it (s20 holds the EXEC the shader started with, s21 the copy), so a lane that was true shows as 1
; and every lane has the same operands. Cases (A = src0, B = src1; the junk is the high half that must be ignored):
;   k0: A = inline 1.0, B = 0x1234:2.0       less           k4: A = 0x5555:1.0, B = 0x7c00:sNaN 0x7c01   unordered
;   k1: A = inline 2.0, B = 0xabcd:1.0       greater        k5: A = 0x0001:+inf, B = 0xbeef:-inf         greater
;   k2: A = inline 0,   B = 0xffff:-0        equal          k6: A = 0xdead:0x0001, B = 0x8000:0x0002     less (denormals)
;   k3: A = 0xffff:qNaN 0x7e00, B = 0x0001:1.0   unordered
; Relation sets: LESS = {k0,k6} = 0x41, GREATER = {k1,k5} = 0x22, EQUAL = {k2} = 0x04, UNORDERED = {k3,k4} = 0x18; the
; mask of an op (bit k = result on case k) is the union of the sets it is true for:
;   f 0x00   lt 0x41   eq 0x04   le 0x45   gt 0x22   lg 0x63   ge 0x26   o 0x67
;   u 0x18   nge 0x59  nlg 0x1c  ngt 0x5d  nle 0x3a  neq 0x7b  nlt 0x3e  tru 0x7f
; Word w holds ops 4w..4w+3 (the order above), op j at bit 7*j + k. Three more bits per word, from one op of the form:
;   bit 30: always 1 (keeps the float export away from denormal and NaN patterns).
;   bit 29: VCC = 1 in every lane (v_cmp_tru_f16 v8, v8), then v_cmpx_f_f16 empties EXEC; after EXEC is restored VCC
;           must still be 1, because a v_cmpx writes EXEC only: the bit is 1.
;   bit 28: v_cmpx_f_f16 empties EXEC, the v_cmpx_tru_f16 after it is true in every lane, but v_cmpx only ANDs into
;           EXEC, so no lane comes back: the bit is 0.
; word 0 = 0x60000000 | 0x00 | 0x41<<7 | 0x04<<14 | 0x45<<21 = 0x60000000 + 0x2080 + 0x10000 + 0x8a00000 = 0x68a12080
; word 1 = 0x60000000 | 0x22 | 0x63<<7 | 0x26<<14 | 0x67<<21 = 0x60000000 + 0x31a2 + 0x98000 + 0xce00000 = 0x6ce9b1a2
; word 2 = 0x60000000 | 0x18 | 0x59<<7 | 0x1c<<14 | 0x5d<<21 = 0x60000000 + 0x2c98 + 0x70000 + 0xba00000 = 0x6ba72c98
; word 3 = 0x60000000 | 0x3a | 0x7b<<7 | 0x3e<<14 | 0x7f<<21 = 0x60000000 + 0x3dba + 0xf8000 + 0xfe00000 = 0x6fefbdba
; A v_cmpx that wrote VCC, or did not AND with the old EXEC, flips bit 29 or bit 28; one that ordered a NaN, flushed a
; denormal, took -0 below +0 or read the high halves moves bits in the words.
; Expected mrt0 bits: 0x68a12080, 0x6ce9b1a2, 0x6ba72c98, 0x6fefbdba
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
s_mov_b32 s20, exec_lo
v_mov_b32_e32 v40, 0x40000000
v_cmp_tru_f16_e32 vcc_lo, v8, v8
v_cmpx_f_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v40, v30, 29, v40
v_cmpx_f_f16_e32 v8, v8
v_cmpx_tru_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 28, v40
v_cmpx_f_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 0, v40
v_cmpx_f_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 1, v40
v_cmpx_f_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 2, v40
v_cmpx_f_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 3, v40
v_cmpx_f_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 4, v40
v_cmpx_f_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 5, v40
v_cmpx_f_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 6, v40
v_cmpx_lt_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 7, v40
v_cmpx_lt_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 8, v40
v_cmpx_lt_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 9, v40
v_cmpx_lt_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 10, v40
v_cmpx_lt_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 11, v40
v_cmpx_lt_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 12, v40
v_cmpx_lt_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 13, v40
v_cmpx_eq_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 14, v40
v_cmpx_eq_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 15, v40
v_cmpx_eq_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 16, v40
v_cmpx_eq_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 17, v40
v_cmpx_eq_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 18, v40
v_cmpx_eq_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 19, v40
v_cmpx_eq_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 20, v40
v_cmpx_le_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 21, v40
v_cmpx_le_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 22, v40
v_cmpx_le_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 23, v40
v_cmpx_le_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 24, v40
v_cmpx_le_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 25, v40
v_cmpx_le_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 26, v40
v_cmpx_le_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v40, v30, 27, v40
v_mov_b32_e32 v41, 0x40000000
v_cmp_tru_f16_e32 vcc_lo, v8, v8
v_cmpx_f_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v41, v30, 29, v41
v_cmpx_f_f16_e32 v8, v8
v_cmpx_tru_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 28, v41
v_cmpx_gt_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 0, v41
v_cmpx_gt_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 1, v41
v_cmpx_gt_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 2, v41
v_cmpx_gt_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 3, v41
v_cmpx_gt_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 4, v41
v_cmpx_gt_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 5, v41
v_cmpx_gt_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 6, v41
v_cmpx_lg_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 7, v41
v_cmpx_lg_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 8, v41
v_cmpx_lg_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 9, v41
v_cmpx_lg_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 10, v41
v_cmpx_lg_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 11, v41
v_cmpx_lg_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 12, v41
v_cmpx_lg_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 13, v41
v_cmpx_ge_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 14, v41
v_cmpx_ge_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 15, v41
v_cmpx_ge_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 16, v41
v_cmpx_ge_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 17, v41
v_cmpx_ge_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 18, v41
v_cmpx_ge_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 19, v41
v_cmpx_ge_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 20, v41
v_cmpx_o_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 21, v41
v_cmpx_o_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 22, v41
v_cmpx_o_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 23, v41
v_cmpx_o_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 24, v41
v_cmpx_o_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 25, v41
v_cmpx_o_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 26, v41
v_cmpx_o_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v41, v30, 27, v41
v_mov_b32_e32 v42, 0x40000000
v_cmp_tru_f16_e32 vcc_lo, v8, v8
v_cmpx_f_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v42, v30, 29, v42
v_cmpx_f_f16_e32 v8, v8
v_cmpx_tru_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 28, v42
v_cmpx_u_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 0, v42
v_cmpx_u_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 1, v42
v_cmpx_u_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 2, v42
v_cmpx_u_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 3, v42
v_cmpx_u_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 4, v42
v_cmpx_u_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 5, v42
v_cmpx_u_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 6, v42
v_cmpx_nge_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 7, v42
v_cmpx_nge_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 8, v42
v_cmpx_nge_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 9, v42
v_cmpx_nge_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 10, v42
v_cmpx_nge_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 11, v42
v_cmpx_nge_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 12, v42
v_cmpx_nge_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 13, v42
v_cmpx_nlg_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 14, v42
v_cmpx_nlg_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 15, v42
v_cmpx_nlg_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 16, v42
v_cmpx_nlg_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 17, v42
v_cmpx_nlg_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 18, v42
v_cmpx_nlg_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 19, v42
v_cmpx_nlg_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 20, v42
v_cmpx_ngt_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 21, v42
v_cmpx_ngt_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 22, v42
v_cmpx_ngt_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 23, v42
v_cmpx_ngt_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 24, v42
v_cmpx_ngt_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 25, v42
v_cmpx_ngt_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 26, v42
v_cmpx_ngt_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v42, v30, 27, v42
v_mov_b32_e32 v43, 0x40000000
v_cmp_tru_f16_e32 vcc_lo, v8, v8
v_cmpx_f_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, vcc_lo
v_lshl_or_b32 v43, v30, 29, v43
v_cmpx_f_f16_e32 v8, v8
v_cmpx_tru_f16_e32 v8, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 28, v43
v_cmpx_nle_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 0, v43
v_cmpx_nle_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 1, v43
v_cmpx_nle_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 2, v43
v_cmpx_nle_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 3, v43
v_cmpx_nle_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 4, v43
v_cmpx_nle_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 5, v43
v_cmpx_nle_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 6, v43
v_cmpx_neq_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 7, v43
v_cmpx_neq_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 8, v43
v_cmpx_neq_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 9, v43
v_cmpx_neq_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 10, v43
v_cmpx_neq_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 11, v43
v_cmpx_neq_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 12, v43
v_cmpx_neq_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 13, v43
v_cmpx_nlt_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 14, v43
v_cmpx_nlt_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 15, v43
v_cmpx_nlt_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 16, v43
v_cmpx_nlt_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 17, v43
v_cmpx_nlt_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 18, v43
v_cmpx_nlt_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 19, v43
v_cmpx_nlt_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 20, v43
v_cmpx_tru_f16_e32 1.0, v8
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 21, v43
v_cmpx_tru_f16_e32 2.0, v9
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 22, v43
v_cmpx_tru_f16_e32 0, v10
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 23, v43
v_cmpx_tru_f16_e32 v4, v11
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 24, v43
v_cmpx_tru_f16_e32 v5, v12
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 25, v43
v_cmpx_tru_f16_e32 v6, v13
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 26, v43
v_cmpx_tru_f16_e32 v7, v14
s_mov_b32 s21, exec_lo
s_mov_b32 exec_lo, s20
v_cndmask_b32_e64 v30, 0, 1, s21
v_lshl_or_b32 v43, v30, 27, v43
exp mrt0 v40, v41, v42, v43 done vm
s_endpgm
