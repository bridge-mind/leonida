; group: 32,1,1
; Part-2 test (SHADER-21 follow-up): a lane mask written to a VGPR under partial EXEC.
; Hardware: inactive lanes keep their old VGPR value.
; If the translator's write_all_lanes ignores EXEC, inactive lanes get the mask word instead.
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mov_b32_e32 v2, 0xdeadbeef
v_mov_b32_e32 v5, 1
v_mov_b32_e32 v6, 2
v_cmp_lt_u32_e64 vcc_lo, v5, v6
s_mov_b32 exec_lo, 0x0000000f
v_mov_b32_e32 v2, vcc_lo
s_mov_b32 exec_lo, -1
v_lshlrev_b32_e32 v3, 2, v1
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm
