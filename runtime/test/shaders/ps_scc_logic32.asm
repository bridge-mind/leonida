; s_nand_b32, s_nor_b32, s_xnor_b32: D = ~(S0 op S1), SCC = (D != 0). Values by hand from the RDNA2 ISA SOP2 text.
; s2 = 0xf0f0f0f0, s3 = 0xff00ff00: and = 0xf000f000, or = 0xfff0fff0, xor = 0x0ff00ff0.
; Expected mrt0 bits: nand 0x0fff0fff, nor 0x000f000f, xnor 0xf00ff00f, and an SCC word, bit n = SCC after case n:
;   0 nand(-1, -1) = 0 -> 0     1 nor(0, 0) = -1 -> 1     2 xnor(x, x) = -1 -> 1
;   3 xnor(x, ~x) = 0 -> 0      4 nor(0xfffffff0, 0) = 0xf -> 1     5 nand(0x80000000, -1) = 0x7fffffff -> 1
; so the SCC word is 0x36.
s_mov_b32 s2, 0xf0f0f0f0
s_mov_b32 s3, 0xff00ff00
s_nand_b32 s4, s2, s3
s_nor_b32 s5, s2, s3
s_xnor_b32 s6, s2, s3
s_mov_b32 s8, -1
s_mov_b32 s9, 0
s_mov_b32 s10, 0x0f0f0f0f
s_mov_b32 s11, 0xfffffff0
s_mov_b32 s12, 0x80000000
s_nand_b32 s13, s8, s8
s_cselect_b32 s20, 1, 0
s_nor_b32 s13, s9, s9
s_cselect_b32 s21, 1, 0
s_xnor_b32 s13, s2, s2
s_cselect_b32 s22, 1, 0
s_xnor_b32 s13, s2, s10
s_cselect_b32 s23, 1, 0
s_nor_b32 s13, s11, s9
s_cselect_b32 s24, 1, 0
s_nand_b32 s13, s12, s8
s_cselect_b32 s25, 1, 0
s_lshl_b32 s21, s21, 1
s_lshl_b32 s22, s22, 2
s_lshl_b32 s23, s23, 3
s_lshl_b32 s24, s24, 4
s_lshl_b32 s25, s25, 5
s_or_b32 s20, s20, s21
s_or_b32 s20, s20, s22
s_or_b32 s20, s20, s23
s_or_b32 s20, s20, s24
s_or_b32 s20, s20, s25
v_mov_b32_e32 v0, s4
v_mov_b32_e32 v1, s5
v_mov_b32_e32 v2, s6
v_mov_b32_e32 v3, s20
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
