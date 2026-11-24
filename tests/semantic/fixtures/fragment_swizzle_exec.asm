; Each quad swaps adjacent pixel values while its second lane is disabled.
v_mov_b32 v2, 0.75
v_mov_b32 v4, 0.5
s_mov_b32 exec_lo, 0xdddddddd
ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,1,0,3,2)
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
; Store each lane's exported component for the subgroup interpreter to read back.
v_mbcnt_lo_u32_b32 v5, -1, 0
v_lshlrev_b32 v5, 2, v5
buffer_store_dword v4, v5, s[0:3], 0 offen
exp mrt0 v4, v4, v4, v4 done vm
s_endpgm
