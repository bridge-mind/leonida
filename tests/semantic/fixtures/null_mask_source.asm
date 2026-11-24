; Every lane exports (1, 64, 0xffffffff, 7).
; Bit 42 of null is zero, all 64 bits are zero, and no set bit exists.
s_bitcmp0_b64 null, 42
s_cselect_b32 s8, 1, 0
s_bcnt0_i32_b64 s9, null
s_ff1_i32_b64 s10, null
v_mov_b32_e32 v0, s8
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v2, s10
v_mov_b32_e32 v3, 7
; No lane may write 9 while EXEC is ANDed with null.
s_and_saveexec_b64 s[4:5], null
v_mov_b32_e32 v3, 9
s_mov_b64 exec, s[4:5]
