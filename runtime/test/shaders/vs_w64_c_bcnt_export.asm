; TEST-251 (SHADER-10 tier C, vertex stage): a lane count used as data. Vertex v (v0 = vertex index) has x = v & 63;
; s_bcnt1_i32_b64 counts the lanes of its wave with x % 4 == 1, which depends on the wave size and on how many
; vertices the driver put in the wave. r0 = that count (partition dependent), r1 = x % 4 == 1, r2 = r3 = 0, stored to
; its record at heap 0x1000 + 16 * v through the V# in s[0:3].
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_c_bcnt_export.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: r1 equal, r0
; differs.
; mattr: +wavefrontsize64,-wavefrontsize32
v_and_b32_e32 v2, 63, v0
v_and_b32_e32 v4, 3, v2
v_cmp_eq_u32_e32 vcc, 1, v4
s_bcnt1_i32_b64 s6, vcc
v_mov_b32_e32 v6, s6
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
