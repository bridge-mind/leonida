; Far branches: the idiom s_getpc_b64 s[a:a+1]; s_add_u32 sa, sa, K; s_addc_u32 sa+1, sa+1, 0|-1; s_setpc_b64 s[a:a+1] (and the
; s_sub_u32/s_subb_u32 form), which jumps beyond the +-32K dwords an s_branch reaches. docs/design.md "Far branches": the
; translator folds the four instructions into one branch. RDNA2 ISA: S_GETPC_B64 returns the address of the next
; instruction; S_ADD_U32 / S_ADDC_U32 add K to that 64-bit address (carry through SCC; the high word is 0 for a forward
; K and 0xffffffff, as -1, with K = 2^32 - |K| for a backward one; S_SUB_U32 / S_SUBB_U32 subtract |K|); S_SETPC_B64
; sets PC. K is in bytes from the instruction after s_getpc. The block between a far branch and its target holds
; "poison" ORs that must never run, and 40 s_nop (4 bytes each, 160 bytes) so that K > 64 is a 32-bit literal.
; Sizes (bytes): s_nop 4, s_or_b32 with an inline constant 4, with a literal 8, s_add_u32/s_sub_u32/s_cmp with a literal 8.
; x (s10 = 1 to start): two forward far branches, one through s[4:5], one through s[8:9]:
;   K = add (8) + addc (4) + setpc (4) + poison s_or literal (8) + 40 nops (160) = 184 = 0xb8, landing on "s_or s10, 2";
;   the poison ORs 0x100 / 0x200 are skipped, the targets OR 2 and 4: x = 1 | 2 | 4 = 0x00000007.
; y (s11 = 0): a backward loop with s_sub_u32 / s_subb_u32 (hi = 0). The body is s_add_u32 s11, 1 (4) + 40 nops (160) +
;   s_cmp_ge_u32 s11, 3 (4) + s_cbranch_scc1 (4) + s_getpc (4) = 176 bytes from the loop top to the instruction after
;   s_getpc, so K = 176 = 0xb0. The far branch back is taken after iterations 1 and 2 and not after iteration 3
;   (scc = 1 jumps over it), so s11 counts 3; then s_or s11, 0x10: y = 0x00000013.
; z (s12 = 0): a backward loop with s_add_u32 K / s_addc_u32 -1, K = -180 = 0xffffff4c: body s_add_u32 s12, 0x10 (4) +
;   40 nops (160) + s_cmp_ge_u32 s12, 0x50 (literal, 8) + s_cbranch_scc1 (4) + s_getpc (4) = 180. s12 steps by 0x10 and
;   the loop leaves when s12 >= 0x50, after 5 iterations (the far branch back taken 4 times): s12 = 0x50, then
;   s_or s12, 1: z = 0x00000051.
; w (s13 = 0): a far forward branch behind a conditional branch that skips it (s_cbranch_scc0 over the 5 dwords of the
;   sequence), run twice. First scc = (s13 == 1) = 0: the sequence is skipped, the code after it runs: s13 |= 1, then
;   the target ORs 2 (K = 8+4+4 + 4 + 160 = 180 = 0xb4), s13 = 3. Second scc = (s13 == 3) = 1: the sequence runs, it
;   jumps over "s_or s13, 4" and the nops to the target: s13 |= 8: w = 0x0000000b.
; A translation that folded a far branch to the wrong target runs a poison OR (x gains 0x300), repeats or skips a
; loop iteration (y, z), or mishandles the conditional case (w). The loops count on s11/s12, so a wrong target shows
; as a different count, not just a hang.
; Expected mrt0 bits: 0x00000007, 0x00000013, 0x00000051, 0x0000000b
s_mov_b32 s10, 1
s_getpc_b64 s[4:5]
s_add_u32 s4, s4, 184
s_addc_u32 s5, s5, 0
s_setpc_b64 s[4:5]
s_or_b32 s10, s10, 0x100
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_or_b32 s10, s10, 2
s_getpc_b64 s[8:9]
s_add_u32 s8, s8, 184
s_addc_u32 s9, s9, 0
s_setpc_b64 s[8:9]
s_or_b32 s10, s10, 0x200
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_or_b32 s10, s10, 4
s_mov_b32 s11, 0
s_add_u32 s11, s11, 1
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_cmp_ge_u32 s11, 3
s_cbranch_scc1 5
s_getpc_b64 s[6:7]
s_sub_u32 s6, s6, 176
s_subb_u32 s7, s7, 0
s_setpc_b64 s[6:7]
s_or_b32 s11, s11, 0x10
s_mov_b32 s12, 0
s_add_u32 s12, s12, 0x10
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_cmp_ge_u32 s12, 0x50
s_cbranch_scc1 5
s_getpc_b64 s[8:9]
s_add_u32 s8, s8, 0xffffff4c
s_addc_u32 s9, s9, -1
s_setpc_b64 s[8:9]
s_or_b32 s12, s12, 1
s_mov_b32 s13, 0
s_cmp_eq_u32 s13, 1
s_cbranch_scc0 5
s_getpc_b64 s[4:5]
s_add_u32 s4, s4, 180
s_addc_u32 s5, s5, 0
s_setpc_b64 s[4:5]
s_or_b32 s13, s13, 1
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_or_b32 s13, s13, 2
s_cmp_eq_u32 s13, 3
s_cbranch_scc0 5
s_getpc_b64 s[4:5]
s_add_u32 s4, s4, 180
s_addc_u32 s5, s5, 0
s_setpc_b64 s[4:5]
s_or_b32 s13, s13, 4
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_or_b32 s13, s13, 8
v_mov_b32_e32 v10, s10
v_mov_b32_e32 v11, s11
v_mov_b32_e32 v12, s12
v_mov_b32_e32 v13, s13
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
