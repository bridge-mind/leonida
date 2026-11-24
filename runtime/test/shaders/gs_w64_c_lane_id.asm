; TEST-251 (SHADER-10 tier C, geometry stage): the raw lane number (v_mbcnt_lo/hi of -1) used as data, in a legacy
; (LDS-ring) wave64 geometry shader with gs_tri's EXEC prologue. Drawn with vs_ids, whose param1 is the integer
; 0x101 + vertex index: the GS reads it from its first input vertex (signature register 1, TEXCOORD0.x, dword 4),
; so triangle p = (that - 0x101) / 3. Each GS invocation stores (lane id, lane id & 31, p, 0) to its record at heap
; 0x1000 + 16 * p through the V# in s[4:7] (the translator sets s3 for the prologue), then emits one vertex, which
; makes no primitive. In a wave64 GS some invocations get lanes 32..63, in a wave32 one none do, so r0 and r1 depend on
; the wave size and r2 does not. w64_graphics.py uses it as the probe of the GS wave size.
; Needs a DXBC with ISG1/OSG1 = {SV_Position reg 0, TEXCOORD0 reg 1} (build.py makes it for gs_*).
; Run: VKH_GS=gs_w64_c_lane_id.spv VKH_VERTICES=960 VKH_SGPRS=0,0,0,0,0,0,65536,0x30000000 VKH_HEAP=0=0
; VKH_HEAP_DUMP=heap.raw vk_harness vs_ids.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32.
; mattr: +wavefrontsize64,-wavefrontsize32
s_bfe_i32 s0, s3, 0x80008
s_bfm_b64 exec, s0, 0
s_cmp_eq_u32 s0, 64
s_cselect_b64 exec, -1, exec
v_lshlrev_b32_e32 v3, 2, v0
v_and_b32_e32 v10, 0x3fffc, v3
ds_read_b32 v20, v10 offset:16
s_waitcnt lgkmcnt(0)
v_subrev_nc_u32_e32 v20, 0x101, v20
v_mul_hi_u32 v21, v20, 0xaaaaaaab
v_lshrrev_b32_e32 v21, 1, v21
v_mbcnt_lo_u32_b32 v6, -1, 0
v_mbcnt_hi_u32_b32 v6, -1, v6
v_and_b32_e32 v7, 31, v6
v_mov_b32_e32 v8, v21
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v21
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[4:7], 0 offen
v_mov_b32_e32 v13, 0
ds_write2_b32 v13, v13, v13 offset0:0 offset1:1
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
