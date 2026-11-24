; SHADER-10 tier B: a wave-aggregated atomic in a wave64 pixel shader, the compaction idiom of a ballot, a count, one
; atomic from an elected lane and a rank. Pixels with x % 4 == 1 (x = floor(u * 64), y = floor(v * 64) on the 64x64
; target of vs_ids) are the "wanted" lanes. EXEC = the wanted lanes; n = s_bcnt1_i32_b64 exec; rank = v_mbcnt_lo/hi of
; EXEC; the lane of rank 0 does buffer_atomic_add of n to the counter at heap offset 0 and gets the old value back;
; v_readfirstlane (EXEC is the wanted lanes again, and the first of them is the elected lane) broadcasts it as the
; base; every wanted lane stores its pixel number (y * 64 + x + 1) at heap offset 0x100 + 4 * (base + rank).
; On one wave of 64 or on two of 32 every wanted lane gets a slot of its own and the counter ends at the number of
; wanted pixels, 4096 / 4 = 1024. Which pixel gets which slot depends on the partition (allocation order), so the
; expected result is a set: the counter word is 1024 and the 1024 slot words are exactly the numbers y * 64 + x + 1 of
; the pixels with x % 4 == 1, each once.
; Run: VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness vs_ids.spv ps_w64_b_ballot_atomic.spv
; out.raw, with VKH_SUBGROUP_SIZE=64 and 32. Image: (r0 = x % 4 == 1, 0, 0, 0) per pixel, equal in both runs.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_and_b32_e32 v4, 3, v2
s_mov_b64 s[10:11], exec
v_cmp_eq_u32_e32 vcc, 1, v4
v_cndmask_b32_e64 v15, 0, 1, vcc
s_and_saveexec_b64 s[12:13], vcc
s_bcnt1_i32_b64 s6, exec
v_mbcnt_lo_u32_b32 v10, exec_lo, 0
v_mbcnt_hi_u32_b32 v10, exec_hi, v10
v_cmp_eq_u32_e32 vcc, 0, v10
s_and_saveexec_b64 s[14:15], vcc
v_mov_b32_e32 v11, s6
v_mov_b32_e32 v12, 0
buffer_atomic_add v11, v12, s[0:3], 0 offen glc
s_or_b64 exec, exec, s[14:15]
v_readfirstlane_b32 s7, v11
v_add_nc_u32_e32 v13, s7, v10
v_lshlrev_b32_e32 v13, 2, v13
v_lshl_add_u32 v14, v3, 6, v2
v_add_nc_u32_e32 v14, 1, v14
buffer_store_dword v14, v13, s[0:3], 0 offen offset:256
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v16, 0
exp mrt0 v15, v16, v16, v16 done vm
s_endpgm
