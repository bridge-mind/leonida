; Initial LDS at byte 248 + 4*t is 1000 + t, and at 1036 + 4*t is 3000.
; Full read returns 1000 + wave lane, including in a second wave.
; Masked read keeps 99 on excluded lanes. Masked write updates each even t below 16.
v_lshlrev_b32 v1, 2, v0
v_add_nc_u32 v2, 1000, v0
v_mov_b32 v8, 3000
ds_write_b32 v1, v2 offset:248
ds_write_b32 v1, v8 offset:1036
s_waitcnt lgkmcnt(0)
s_barrier
s_mov_b32 m0, 0x123400f0
ds_read_addtid_b32 v3 offset:8
v_mov_b32 v4, 99
v_and_b32 v6, 1, v0
v_cmpx_eq_u32_e32 0, v6
v_cmpx_gt_u32_e32 16, v0
ds_read_addtid_b32 v4 offset:8
s_mov_b32 m0, 0x56780400
ds_write_addtid_b32 v2 offset:12
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
s_barrier
ds_read_b32 v5, v1 offset:1036
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v7, 4, v0
buffer_store_dword v3, v7, s[0:3], 0 offen
buffer_store_dword v4, v7, s[0:3], 0 offen offset:4
buffer_store_dword v5, v7, s[0:3], 0 offen offset:8
s_endpgm
