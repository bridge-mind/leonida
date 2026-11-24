; TEST-251 (SHADER-10 tier A, vertex stage): the skip regions and the loop of ps_w64_a_skip in a wave64 vertex shader,
; without its whole-quad mask (vertex quads are lanes, not pixels). Vertex v (v0 = vertex index) has x = v & 63 and
; y = v >> 6, so every word is a function of v and not of the lane the vertex landed on. It stores (r0, r1, r2, r3)
; to its record at heap 0x1000 + 16 * v through the V# in s[0:3].
;  (1) a skip region behind s_cbranch_execz: only lanes with x % 4 == 1 run it. r0 = x % 4 == 1 ? 7 * y + x : 1000 + x.
;  (2) r1 = x % 4 == 1 (the VCC of (1), per lane).
;  (3) s_cbranch_vccz over a region that narrows EXEC to that VCC. r2 = 2000 + y + (y % 4 == 2 ? 9 : 0).
;  (4) a loop closed by s_cbranch_execnz, no scalar state carried. r3 = x % 4 + 1.
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_a_skip.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: equal records, and equal
; to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_and_b32_e32 v2, 63, v0
v_lshrrev_b32_e32 v3, 6, v0
s_mov_b64 s[10:11], exec
v_add_nc_u32_e32 v6, 0x3e8, v2
v_and_b32_e32 v4, 3, v2
v_cmp_eq_u32_e32 vcc, 1, v4
v_cndmask_b32_e64 v7, 0, 1, vcc
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 2
v_mul_u32_u24_e32 v6, 7, v3
v_add_nc_u32_e32 v6, v6, v2
s_or_b64 exec, exec, s[12:13]
v_and_b32_e32 v5, 3, v3
v_add_nc_u32_e32 v8, 0x7d0, v3
v_cmp_eq_u32_e32 vcc, 2, v5
s_cbranch_vccz 3
s_and_saveexec_b64 s[16:17], vcc
v_add_nc_u32_e32 v8, 9, v8
s_or_b64 exec, exec, s[16:17]
v_add_nc_u32_e32 v10, 1, v4
v_mov_b32_e32 v9, 0
v_add_nc_u32_e32 v9, 1, v9
v_cmp_lt_u32_e32 vcc, v9, v10
s_and_b64 exec, exec, vcc
s_cbranch_execnz -4
s_mov_b64 exec, s[10:11]
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
