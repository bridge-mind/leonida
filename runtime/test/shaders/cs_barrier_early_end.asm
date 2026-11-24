; group: 128,1,1
; SHADER-91: a wave that has ended no longer counts at s_barrier. Wave32, 128 threads = 4 waves. Every thread
; writes its id to LDS[id]; waves 1-3 then end without an s_barrier, and wave 0 waits at one. On RDNA the barrier
; releases once the other waves have ended, and wave 0 reads what wave 3 wrote: out[i] = LDS[127 - i] = 127 - i.
; Translated, the s_barrier sits in a branch that only wave 0 takes: Intel's driver loses the device on that
; (an OpControlBarrier outside uniform control flow); barrier rounds keep the barrier at the top level.
; Run: vk_compute_test cs_barrier_early_end.spv 1 dump
; Expect: out[0..7] = 127 126 125 124 123 122 121 120
v_lshlrev_b32_e32 v2, 2, v0
ds_write_b32 v2, v0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s5, v0
s_lshr_b32 s5, s5, 5
s_cmp_lg_u32 s5, 0
s_cbranch_scc1 9
s_barrier
v_sub_nc_u32_e32 v3, 0x1fc, v2
ds_read_b32 v4, v3
s_waitcnt lgkmcnt(0)
v_cmpx_gt_u32_e32 8, v0
buffer_store_dword v4, v2, s[0:3], 0 offen
s_endpgm
