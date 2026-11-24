; Mesh shader (NGG, wave64, 128 threads): thread id = s3[27:24]*64 + mbcnt; GS_ALLOC_REQ m0 = 1 prim << 12 | 3 verts;
; threads 0-2 export the vk_harness triangle and attr0, thread 0 exports prim (0,1,2). Needs a DXBC with PSV0
; (128 threads, 64/64 outputs, triangles). Run: VKH_MESH=1 vk_harness ms_tri.spv fs_passthru.spv out.raw ->
; every pixel = ((x+0.5)/64, (y+0.5)/64, 0.5, 1).
; Both waves send the same allocation. Its counts must survive the barrier before the exports.
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
s_mov_b32 m0, 0x1003
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
s_barrier
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7 done
exp param0 v3, v4, v8, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
