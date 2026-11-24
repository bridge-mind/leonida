; Wave64 compute: constant EXEC halves (exec_lo = 0xffff0000 -> lanes 16-31, exec_hi = 0xffff -> lanes 32-47), lane from
; v_mbcnt_lo/hi. Run: vk_compute_test cs_exec64.spv 1 exec64 -> out[t] = 7 for t in 16..47, else 0.
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mov_b32_e32 v2, 0
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff
v_mov_b32_e32 v2, 7
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, -1
v_lshlrev_b32_e32 v3, 2, v1
buffer_store_dword v2, v3, s[0:3], 0 offen
s_endpgm
