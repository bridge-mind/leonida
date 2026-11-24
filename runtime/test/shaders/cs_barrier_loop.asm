; group: 128,1,1
; SHADER-91: s_barrier in a loop whose trip count differs per wave. Wave32, 128 threads = 4 waves; wave w runs
; w + 1 iterations. Iteration k adds 1 to LDS[k] from each of the wave's 32 lanes, waits at s_barrier, then
; reads LDS[k] and stores it to out[4 * (3 - w) + k]. On RDNA every s_barrier counts against the same barrier
; and a wave that has ended no longer counts, so the k-th barrier releases once every wave still running has
; added to LDS[k]: LDS[k] = 32 * (4 - k) when read. Wave 3 stores out[0..3] and wave 2 out[4..6]; out[7] would
; be wave 2's fourth iteration, which it doesn't run, so it keeps the sentinel 0xeeeeeeee.
; Run: vk_compute_test cs_barrier_loop.spv 1 dump
; Expect: out[0..7] = 128 96 64 32 128 96 64 4008636142
v_lshlrev_b32_e32 v2, 2, v0
v_mov_b32_e32 v3, 0
v_cmpx_gt_u32_e32 4, v0
ds_write_b32 v2, v3
s_mov_b32 exec_lo, -1
s_waitcnt lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s5, v0
s_lshr_b32 s5, s5, 5
s_add_u32 s6, s5, 1
s_sub_u32 s9, 3, s5
s_lshl_b32 s9, s9, 4
s_mov_b32 s7, 0
v_mov_b32_e32 v5, 1
v_mov_b32_e32 v1, s7
ds_add_u32 v1, v5
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v4, v1
s_waitcnt lgkmcnt(0)
s_add_u32 s10, s9, s7
v_mov_b32_e32 v6, s10
buffer_store_dword v4, v6, s[0:3], 0 offen
s_add_u32 s7, s7, 4
s_sub_u32 s6, s6, 1
s_cmp_lg_u32 s6, 0
s_cbranch_scc1 -16
s_endpgm
