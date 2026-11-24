; TEST-251 (SHADER-10 tier B, vertex stage): the wave-aggregated atomic of ps_w64_b_ballot_atomic in a wave64 vertex
; shader. Vertex v (v0 = vertex index) has x = v & 63; vertices with x % 4 == 1 are the "wanted" lanes. EXEC = the
; wanted lanes; n = s_bcnt1_i32_b64 exec; rank = v_mbcnt_lo/hi of EXEC; the lane of rank 0 does buffer_atomic_add of n
; to the counter at heap offset 0 and gets the old value back; v_readfirstlane broadcasts it as the base; every wanted
; vertex stores v + 1 at heap offset 0x100 + 4 * (base + rank). Whatever the wave size and the packing, every wanted
; vertex gets a slot of its own and the counter ends at the number of wanted vertices (240 of 960); which vertex gets
; which slot depends on the packing, so the slots are compared as a set. Its record at heap 0x1000 + 16 * v holds
; (x % 4 == 1, 0, 0, 0).
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_b_ballot_atomic.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32.
; mattr: +wavefrontsize64,-wavefrontsize32
v_and_b32_e32 v2, 63, v0
v_and_b32_e32 v4, 3, v2
s_mov_b64 s[10:11], exec
v_cmp_eq_u32_e32 vcc, 1, v4
v_cndmask_b32_e64 v6, 0, 1, vcc
s_and_saveexec_b64 s[12:13], vcc
s_bcnt1_i32_b64 s6, exec
v_mbcnt_lo_u32_b32 v10, exec_lo, 0
v_mbcnt_hi_u32_b32 v10, exec_hi, v10
v_cmp_eq_u32_e32 vcc, 0, v10
s_and_saveexec_b64 s[14:15], vcc
v_mov_b32_e32 v11, s6
v_mov_b32_e32 v13, 0
buffer_atomic_add v11, v13, s[0:3], 0 offen glc
s_or_b64 exec, exec, s[14:15]
v_readfirstlane_b32 s7, v11
v_add_nc_u32_e32 v13, s7, v10
v_lshlrev_b32_e32 v13, 2, v13
v_add_nc_u32_e32 v14, 1, v0
buffer_store_dword v14, v13, s[0:3], 0 offen offset:256
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
