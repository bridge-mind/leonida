; Compute: workgroup id in s4 (RSRC2 0x8088: 4 user SGPRs, TGID_X, 512 B LDS; group 64x1x1), thread id in v0,
; LDS exchange across the two waves of a group behind s_barrier, buffer store through V# s[0:3] (base 0 = heap).
; Run: vk_compute_test cs_lds.spv 4 mirror64  (out[g*64+t] == g*64 + 63 - t)
v_lshl_add_u32 v1, s4, 6, v0
v_lshlrev_b32_e32 v2, 2, v0
ds_write_b32 v2, v1
s_waitcnt lgkmcnt(0)
s_barrier
v_sub_nc_u32_e32 v3, 0xfc, v2
ds_read_b32 v4, v3
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v5, 2, v1
buffer_store_dword v4, v5, s[0:3], 0 offen
s_endpgm
