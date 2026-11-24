; TEST-251 (SHADER-10 tier C, vertex stage): the raw lane number (v_mbcnt_lo/hi of -1) used as data. Each vertex v
; (v0 = vertex index) stores (lane id, lane id & 31, v, 0) to its record at heap 0x1000 + 16 * v through the V# in
; s[0:3]. In a wave64 VS some vertices get lanes 32..63; in a wave32 VS none do, so r0 and r1 depend on how the driver
; packed the vertices into waves and r2 does not. w64_graphics.py also uses it as its probe: the largest r0 shows the
; wave size the driver picked (63 by default on RADV GFX12, 31 with RADV_PERFTEST=gewave32).
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_c_lane_id.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32 (the triangles have
; zero area, so nothing is drawn).
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v6, -1, 0
v_mbcnt_hi_u32_b32 v6, -1, v6
v_and_b32_e32 v7, 31, v6
v_mov_b32_e32 v8, v0
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
