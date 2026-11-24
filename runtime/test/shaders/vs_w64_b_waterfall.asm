; TEST-251 (SHADER-10 tier B, vertex stage): the waterfall loop of ps_w64_b_waterfall in a wave64 vertex shader.
; Vertex v (v0 = vertex index) has x = v & 63, y = v >> 6 and the key a = x % 4 + 4 * (y % 4). The loop takes
; v_readfirstlane of the key, runs the lanes holding it (v_cmp_eq, s_and_saveexec), retires them (s_xor exec) and
; repeats while lanes are left (s_cbranch_execnz); the scalar is used only inside the body. It runs once per distinct
; key of the wave or of the 32-lane subgroup, and every vertex ends with r0 = 17 * a + x, r1 = a, r2 = r3 = 0, stored
; to its record at heap 0x1000 + 16 * v through the V# in s[0:3].
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_b_waterfall.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: equal, and equal to
; the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_and_b32_e32 v2, 63, v0
v_lshrrev_b32_e32 v3, 6, v0
v_and_b32_e32 v4, 3, v2
v_and_b32_e32 v5, 3, v3
v_lshl_add_u32 v4, v5, 2, v4
v_mov_b32_e32 v6, 0
s_mov_b64 s[10:11], exec
v_readfirstlane_b32 s20, v4
v_cmp_eq_u32_e32 vcc, s20, v4
s_and_saveexec_b64 s[12:13], vcc
s_mul_i32 s21, s20, 17
v_add_nc_u32_e32 v6, s21, v2
s_xor_b64 exec, exec, s[12:13]
s_cbranch_execnz -7
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, v4
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
