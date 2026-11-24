; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_cmp_nlg / v_cmpx_nlg (docs/design.md "Constant and ordered VOPC"): nlg ("not less or greater") is true when
; x == y or either source is NaN (unordered equal). Each case ORs its bit into a mask; channels are mask | 0x40000000.
;   x, f32 cases: bit0 1.0 vs 1.0 (1), bit1 +0 vs -0 (1), bit2 quiet NaN vs 1.0 (1), bit3 1.0 vs 2.5 (0),
;      bit4 signalling NaN 0x7f800001 vs 1.0 (1), bit5 +inf vs +inf (1), bit6 -inf vs +inf (0): mask 0x37
;   y, f16 cases (high halves 0xABCD / 0x1234 must be ignored): bit0 1.0h vs 1.0h (1), bit1 +0h vs -0h (1),
;      bit2 NaN 0x7e00 vs 1.0h (1), bit3 1.0h vs 2.5h (0): mask 0x7
;   z, the e64 form with source modifiers: bit0 nlg(-|-1.0|, -1.0) (1), bit1 nlg(|x|, 2.5) with x = -1.0 (0): mask 0x1
;   w: v_cmpx_nlg_f32 and _f16 on equal, +-0 and NaN operands keep every lane, so the pixel is still written: 1.0
; Expected mrt0 bits: 0x40000037, 0x40000007, 0x40000001, 0x3f800000
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v0, 1.0
v_cmp_nlg_f32_e32 vcc_lo, 1.0, v0
v_cndmask_b32_e64 v1, 0, 1, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v0, 0x80000000
v_cmp_nlg_f32_e32 vcc_lo, 0, v0
v_cndmask_b32_e64 v1, 0, 2, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v0, 0x7fc00000
v_cmp_nlg_f32_e32 vcc_lo, 1.0, v0
v_cndmask_b32_e64 v1, 0, 4, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v0, 0x40200000
v_cmp_nlg_f32_e32 vcc_lo, 1.0, v0
v_cndmask_b32_e64 v1, 0, 8, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v0, 0x7f800001
v_cmp_nlg_f32_e32 vcc_lo, 1.0, v0
v_cndmask_b32_e64 v1, 0, 16, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v0, 0x7f800000
v_mov_b32_e32 v2, 0x7f800000
v_cmp_nlg_f32_e32 vcc_lo, v2, v0
v_cndmask_b32_e64 v1, 0, 32, vcc_lo
v_or_b32_e32 v10, v10, v1
v_mov_b32_e32 v2, 0xff800000
v_cmp_nlg_f32_e32 vcc_lo, v2, v0
v_cndmask_b32_e64 v1, 0, 64, vcc_lo
v_or_b32_e32 v10, v10, v1
v_or_b32_e32 v10, 0x40000000, v10
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v0, 0xabcd3c00
v_mov_b32_e32 v2, 0x12343c00
v_cmp_nlg_f16_e32 vcc_lo, v0, v2
v_cndmask_b32_e64 v1, 0, 1, vcc_lo
v_or_b32_e32 v11, v11, v1
v_mov_b32_e32 v0, 0xabcd0000
v_mov_b32_e32 v2, 0x12348000
v_cmp_nlg_f16_e32 vcc_lo, v0, v2
v_cndmask_b32_e64 v1, 0, 2, vcc_lo
v_or_b32_e32 v11, v11, v1
v_mov_b32_e32 v0, 0xabcd7e00
v_mov_b32_e32 v2, 0x12343c00
v_cmp_nlg_f16_e32 vcc_lo, v0, v2
v_cndmask_b32_e64 v1, 0, 4, vcc_lo
v_or_b32_e32 v11, v11, v1
v_mov_b32_e32 v0, 0xabcd3c00
v_mov_b32_e32 v2, 0x12344100
v_cmp_nlg_f16_e32 vcc_lo, v0, v2
v_cndmask_b32_e64 v1, 0, 8, vcc_lo
v_or_b32_e32 v11, v11, v1
v_or_b32_e32 v11, 0x40000000, v11
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v0, -1.0
v_cmp_nlg_f32_e64 s1, -|v0|, v0
v_cndmask_b32_e64 v1, 0, 1, s1
v_or_b32_e32 v12, v12, v1
v_mov_b32_e32 v2, 0x40200000
v_cmp_nlg_f32_e64 s1, |v0|, v2
v_cndmask_b32_e64 v1, 0, 2, s1
v_or_b32_e32 v12, v12, v1
v_or_b32_e32 v12, 0x40000000, v12
v_mov_b32_e32 v0, 1.0
v_cmpx_nlg_f32_e32 1.0, v0
v_mov_b32_e32 v0, 0x80000000
v_cmpx_nlg_f32_e32 0, v0
v_mov_b32_e32 v0, 0x7fc00000
v_cmpx_nlg_f32_e32 1.0, v0
v_mov_b32_e32 v0, 0x7e00
v_mov_b32_e32 v2, 0x3c00
v_cmpx_nlg_f16_e32 v0, v2
v_mov_b32_e32 v13, 1.0
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
