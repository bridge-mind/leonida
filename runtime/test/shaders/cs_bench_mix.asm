; Benchmark (not a pass/fail test): a realistic wave64 mix, timed by scripts/bench_wave64.py. Per lane: a hash, a ballot +
; v_mbcnt_lo/hi stream-compaction slot and s_bcnt1 count, v_readlane of lane 37, three wave branches (vcc / exec), an LDS
; write then a read of a neighbour's slot with no barrier in between (same wave), then one store of the result.
; Every dispatch writes the same bytes: out[g*64 + lane], so the output of any configuration can be compared with another.
; Run: vk_compute_test cs_bench_mix.spv 16384 dump   (VKC_BENCH=N adds N timed dispatches)
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_lshl_add_u32 v10, s4, 6, v1
v_mul_lo_u32 v2, v10, 0x9e3779b1
v_xor_b32_e32 v2, v2, v10
v_and_b32_e32 v3, 3, v2
v_cmp_eq_u32_e64 s[6:7], 1, v3
s_bcnt1_i32_b64 s8, s[6:7]
v_mbcnt_lo_u32_b32 v4, s6, 0
v_mbcnt_hi_u32_b32 v4, s7, v4
v_readlane_b32 s9, v2, 37
v_add_nc_u32_e32 v5, v4, v2
v_add_nc_u32_e32 v5, s9, v5
v_add_nc_u32_e32 v5, s8, v5
v_lshlrev_b32_e32 v6, 2, v1
ds_write_b32 v6, v5
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v8, 1, v1
v_lshlrev_b32_e32 v8, 2, v8
ds_read_b32 v7, v8
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v5, v5, v7
v_cmp_gt_u32_e32 vcc, 0x80000000, v2
s_cbranch_vccz 2
v_add_nc_u32_e32 v5, 1, v5
v_add_nc_u32_e32 v5, 2, v5
v_cmp_gt_u32_e32 vcc, 0x7fffffff, v5
s_cbranch_vccnz 1
v_xor_b32_e32 v5, 5, v5
v_cmp_lt_u32_e64 s[10:11], 0x40000000, v2
s_and_saveexec_b64 s[12:13], s[10:11]
s_cbranch_execz 2
v_add_nc_u32_e32 v5, 7, v5
v_xor_b32_e32 v5, v5, v10
s_mov_b64 exec, s[12:13]
v_lshlrev_b32_e32 v9, 2, v10
buffer_store_dword v5, v9, s[0:3], 0 offen
s_endpgm
