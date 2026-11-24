; s_nand_b64, s_nor_b64, s_xnor_b64 on plain SGPR pairs: D = ~(S0 op S1), SCC = ((D.lo | D.hi) != 0). By hand from the RDNA2 ISA.
; A = s[2:3] = (lo 0xf0f0f0f0, hi 0x0000ffff), B = s[4:5] = (lo 0xff00ff00, hi 0x00ff00ff).
; Expected mrt0 bits: nand.lo 0x0fff0fff, nor.hi 0xff000000 (A.hi | B.hi = 0x00ffffff), xnor.hi 0xff0000ff (A.hi ^ B.hi = 0x00ffff00),
; and an SCC word, bit n = SCC after case n (X = (-1, 0), Y = (0, -1)):
;   0 nand(ones, ones) = 0 -> 0              1 nor(X, 0) = (0, -1), only the high dword set -> 1
;   2 xnor(X, X) = all ones -> 1             3 xnor(X, Y) = 0 -> 0
;   4 nor(Y, (0xfffffff0, 0)) = (0xf, 0), only the low dword set -> 1
; so the SCC word is 0x16.
s_mov_b32 s2, 0xf0f0f0f0
s_mov_b32 s3, 0x0000ffff
s_mov_b32 s4, 0xff00ff00
s_mov_b32 s5, 0x00ff00ff
s_nand_b64 s[6:7], s[2:3], s[4:5]
s_nor_b64 s[8:9], s[2:3], s[4:5]
s_xnor_b64 s[10:11], s[2:3], s[4:5]
s_mov_b32 s12, -1
s_mov_b32 s13, -1
s_mov_b32 s14, -1
s_mov_b32 s15, 0
s_mov_b32 s16, 0
s_mov_b32 s17, -1
s_mov_b32 s18, 0
s_mov_b32 s19, 0
s_mov_b32 s24, 0xfffffff0
s_mov_b32 s25, 0
s_nand_b64 s[30:31], s[12:13], s[12:13]
s_cselect_b32 s40, 1, 0
s_nor_b64 s[30:31], s[14:15], s[18:19]
s_cselect_b32 s41, 1, 0
s_xnor_b64 s[30:31], s[14:15], s[14:15]
s_cselect_b32 s42, 1, 0
s_xnor_b64 s[30:31], s[14:15], s[16:17]
s_cselect_b32 s43, 1, 0
s_nor_b64 s[30:31], s[16:17], s[24:25]
s_cselect_b32 s44, 1, 0
s_lshl_b32 s41, s41, 1
s_lshl_b32 s42, s42, 2
s_lshl_b32 s43, s43, 3
s_lshl_b32 s44, s44, 4
s_or_b32 s40, s40, s41
s_or_b32 s40, s40, s42
s_or_b32 s40, s40, s43
s_or_b32 s40, s40, s44
v_mov_b32_e32 v0, s6
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v2, s11
v_mov_b32_e32 v3, s40
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
