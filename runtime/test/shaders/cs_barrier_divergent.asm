; group: 128,1,1
; SHADER-91: s_barrier in divergent (but wave-uniform) control flow. Wave32, 128 threads = 4 waves. Every thread
; writes its id to LDS[id]; wave 0 then takes one branch with an s_barrier and waves 1-3 another branch with their
; own s_barrier. On RDNA every s_barrier counts against the same workgroup barrier, so the two sites meet and wave 0
; reads what wave 3 wrote: out[i] = LDS[127 - i] = 127 - i for i = 0..7. In SPIR-V, an OpControlBarrier that not
; every invocation of the workgroup reaches is undefined, which is what the barrier-only scheduler avoids.
; Branch: s_cbranch_scc1 skips the 10 dwords of wave 0's path (checked with llvm-mc -show-encoding).
; Run: vk_compute_test cs_barrier_divergent.spv 1 dump
; Expect: out[0..7] = 127 126 125 124 123 122 121 120
v_lshlrev_b32_e32 v2, 2, v0
ds_write_b32 v2, v0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s5, v0
s_lshr_b32 s5, s5, 5
s_cmp_lg_u32 s5, 0
s_cbranch_scc1 10
s_barrier
v_sub_nc_u32_e32 v3, 0x1fc, v2
ds_read_b32 v4, v3
s_waitcnt lgkmcnt(0)
v_cmpx_gt_u32_e32 8, v0
buffer_store_dword v4, v2, s[0:3], 0 offen
s_endpgm
s_barrier
s_endpgm
