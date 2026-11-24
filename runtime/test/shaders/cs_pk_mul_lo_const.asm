; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; group: 32,1,1
; v_pk_mul_lo_u16 with a constant operand (an SGPR, a literal, an inline constant), one half of the constant 0 or
; small, the other operand computed at run time from the output buffer's 0xEEEEEEEE fill (so no compiler can fold it).
; Each 16-bit half of D is the low 16 bits of the product of the selected source halves (RDNA2 ISA, V_PK_MUL_LO_U16;
; op_sel picks the half of each source for the low result, op_sel_hi for the high one; defaults [0,0] and [1,1]):
;   0: v_pk_mul_lo_u16 v30, v3, s8 op_sel:[0,1] op_sel_hi:[0,0]   v3.lo = 0x1234, s8 = 0xaaaa0000
;        lo = 0x1234 * 0xaaaa & 0xffff = 0x9e88, hi = 0x1234 * 0x0000 = 0                             -> 0x00009E88
;   1: v_pk_mul_lo_u16 v31, s7, v15 op_sel_hi:[1,0]               s7 = 0x0003ffff, v15.lo = 0xabcd
;        lo = 0xffff * 0xabcd & 0xffff, hi = 0x0003 * 0xabcd & 0xffff                                -> 0x03675433
;   2: v_pk_mul_lo_u16 v32, v2, 0x00050000                        v2 = 0x0007beef: lo = 0xbeef * 0 = 0,
;        hi = 0x0007 * 0x0005 = 0x23                                                                 -> 0x00230000
;   3: v_pk_mul_lo_u16 v33, v3, 7 op_sel_hi:[1,0]                 v3 = 0x00011234: lo = 0x1234 * 7,
;        hi = v3.hi * 7.lo (op_sel_hi[1] = 0: the low half of the constant 7)                        -> 0x00077F6C
; Measured natively on RDNA2 (gfx1036, Ryzen 9 9950X3D iGPU, driver 26.10.1) through tools/hw_oracle (raw HIP kernel).
; On the AMD Windows driver the translated (x * y) & 0xffff of case 0 came out 0x0c229e88: the mask was dropped when the
; other half was a constant 0. The translator now computes the low 16 bits as ((x << 16) * y) >> 16.
; Run: vk_compute_test cs_pk_mul_lo_const.spv 1 dump
; out[0..7] = 00009E88 03675433 00230000 00077F6C 00009E88 03675433 00230000 00077F6C (hex)
s_mov_b32 exec_lo, -1
v_mov_b32_e32 v20, v0
v_lshlrev_b32_e32 v21, 4, v20
buffer_load_dword v22, v21, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_xor_b32_e32 v3, 0xeeeffcda, v22
v_xor_b32_e32 v15, 0xfcda4523, v22
v_xor_b32_e32 v2, 0xeee95001, v22
s_mov_b32 s8, 0xaaaa0000
s_mov_b32 s7, 0x0003ffff
v_pk_mul_lo_u16 v30, v3, s8 op_sel:[0,1] op_sel_hi:[0,0]
v_pk_mul_lo_u16 v31, s7, v15 op_sel_hi:[1,0]
v_pk_mul_lo_u16 v32, v2, 0x00050000
v_pk_mul_lo_u16 v33, v3, 7 op_sel_hi:[1,0]
buffer_store_dword v30, v21, s[0:3], 0 offen
buffer_store_dword v31, v21, s[0:3], 0 offen offset:4
buffer_store_dword v32, v21, s[0:3], 0 offen offset:8
buffer_store_dword v33, v21, s[0:3], 0 offen offset:12
s_endpgm
