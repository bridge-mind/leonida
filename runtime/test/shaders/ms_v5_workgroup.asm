; One full-screen triangle per mesh workgroup. RGB records incoming v5 from local lanes 0, 1 and 31.
; With --mesh-group-readback 2 3 4, pixel g = x + 2*y + 6*z must be (32g, 32g+1, 32g+31, 1): v5 = group index *
; NumThreads + lane (#209; the game's mesh entry shifts v5 right by 7 on lane 0 with 128 threads).
; Stencil selects the workgroup independently of its color. PSV0: 32 threads, 128 bytes LDS, 3 vertices, 1 triangle.
v_mbcnt_lo_u32_b32 v0, -1, 0
; Exchange through group memory so the probe also works when a host subgroup has fewer than 32 lanes.
v_lshlrev_b32_e32 v11, 2, v0
ds_write_b32 v11, v5
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v11, 0
ds_read_b32 v8, v11
ds_read_b32 v9, v11 offset:4
ds_read_b32 v10, v11 offset:124
s_waitcnt lgkmcnt(0)
v_cvt_f32_u32_e32 v8, v8
v_cvt_f32_u32_e32 v9, v9
v_cvt_f32_u32_e32 v10, v10
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc_lo
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
s_mov_b32 m0, 0x1003
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7 done
exp param0 v8, v9, v10, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
