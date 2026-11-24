; TEST-251 (SHADER-10 tier C, vertex stage): v_readlane of a varying value (the vertex index) at lanes 40 and 7. In a
; wave64 VS every lane gets the vertex in hardware lane 40 (or 7) of its wave; in a wave32 VS lane 40 wraps or is
; undefined and lane 7 is the vertex of another wave. Vertex v (v0 = vertex index): r0 = readlane 40, r1 = v,
; r2 = readlane 7 (both partition dependent), r3 = 0, stored to its record at heap 0x1000 + 16 * v through the V# in
; s[0:3].
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_c_readlane.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: r1 equal, r2 differs.
; mattr: +wavefrontsize64,-wavefrontsize32
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v4, v0
v_readlane_b32 s6, v4, 40
v_readlane_b32 s7, v4, 7
v_mov_b32_e32 v6, s6
v_mov_b32_e32 v7, v4
v_mov_b32_e32 v8, s7
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
