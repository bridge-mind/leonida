; Mesh-shader producer for ps_interp_f16_high: ms_tri's triangle (NGG, wave64, 128 threads, threads 0-2 export
; the vertices, thread 0 the primitive) with vs_f16_pack's params, two f16 packed per 32-bit channel:
; param0.x = f16(u) | f16(v) << 16, param0.y = f16(v) | f16(u) << 16, param1 = (u, v, 0.5, 1) as f32.
; Run: VKH_MESH=1 vk_harness ms_f16_pack.spv ps_interp_f16_high.spv out.raw; checked as ps_interp_f16_high is.
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s4, s3, 0x40018
v_lshl_add_u32 v0, s4, 6, v0
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc_lo
v_cndmask_b32_e64 v3, 0, 2.0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc_lo
v_cndmask_b32_e64 v4, 0, 2.0, vcc_lo
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v8, 0.5
v_cvt_pkrtz_f16_f32_e32 v10, v3, v4
v_cvt_pkrtz_f16_f32_e32 v11, v4, v3
s_mov_b32 m0, 0x1003
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7 done
exp param0 v10, v11, v8, v7
exp param1 v3, v4, v8, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
