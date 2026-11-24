; TEST-251 (SHADER-10 tier A, geometry stage): the lane number forms only the GSVS address. A legacy GS writes each
; emitted vertex to its lane's GSVS area; here that area's base is lane id * 96 (v_mbcnt_lo/hi of -1), the shape a
; hardware GS has. The translator maps an LDS write to an output slot from its offset alone and drops the VGPR address
; (rdna2spirv.py, gs_ds), so the lane id reaches no output and the program is partition invariant: after spirv-opt -O
; its SPIR-V is the same as without the base. Drawn with vs_ids (param1 = 0x101 + vertex index): triangle
; p = (TEXCOORD0.x of input vertex 0 - 0x101) / 3. Each invocation stores (p, 3 * p + 1, 77, 0) to its record at heap
; 0x1000 + 16 * p through the V# in s[4:7], and emits one vertex (no primitive) through the lane-based ring address.
; Needs a DXBC with ISG1/OSG1 = {SV_Position reg 0, TEXCOORD0 reg 1} (build.py makes it for gs_*).
; Run: VKH_GS=gs_w64_a_ring.spv VKH_VERTICES=960 VKH_SGPRS=0,0,0,0,0,0,65536,0x30000000 VKH_HEAP=0=0
; VKH_HEAP_DUMP=heap.raw vk_harness vs_ids.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32:
; equal records, and equal to the expected words.
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
v_mov_b32_e32 v6, v21
v_lshl_add_u32 v7, v21, 1, v21
v_add_nc_u32_e32 v7, 1, v7
v_mov_b32_e32 v8, 0x4d
v_mov_b32_e32 v9, 0
v_lshlrev_b32_e32 v12, 4, v21
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[4:7], 0 offen
v_mbcnt_lo_u32_b32 v13, -1, 0
v_mbcnt_hi_u32_b32 v13, -1, v13
v_mul_u32_u24_e32 v13, 0x60, v13
v_mov_b32_e32 v14, 0
ds_write2_b32 v13, v14, v14 offset0:0 offset1:1
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
