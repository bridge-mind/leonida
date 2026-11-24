; Mesh shader (NGG, wave64, 128 threads = 2 waves): verifies s3[27:24] == local_index / 64 across all 128 lanes.
; Every lane extracts wave_id = s_bfe_u32(s3, 0x40018) and atomically increments LDS[min(wave_id, 2)]:
;   LDS[0] = count of lanes seeing wave_id == 0 (must be 64)
;   LDS[1] = count of lanes seeing wave_id == 1 (must be 64)
;   LDS[2] = count of lanes seeing wave_id >= 2 (must be 0)
; Only wave 1 sends the mesh allocation. Both waves meet at the barrier before wave 0 exports the triangle.
; Threads 0..2 export ms_tri's fullscreen triangle with param0.b = 0.5 iff LDS[0]==64 && LDS[1]==64 && LDS[2]==0,
; else 0.0. Needs DXBC with PSV0 (128x1x1 threads, 64/64 outputs, triangles; see scripts/run_gpu_tests.py):
;   VKH_MESH=1 vk_harness ms_wave_info_s3.spv fs_passthru.spv out.raw
;   -> every pixel = ((x+0.5)/64, (y+0.5)/64, 0.5, 1.0).
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s4, s3, 0x40018
v_lshl_add_u32 v0, s4, 6, v0
v_mov_b32_e32 v10, 0
v_cmpx_eq_u32_e32 0, v0
ds_write_b32 v10, v10 offset:0
ds_write_b32 v10, v10 offset:4
ds_write_b32 v10, v10 offset:8
s_mov_b64 exec, -1
s_barrier
s_min_u32 s5, s4, 2
s_lshl_b32 s5, s5, 2
v_mov_b32_e32 v11, s5
v_mov_b32_e32 v12, 1
ds_add_u32 v11, v12
s_barrier
ds_read_b32 v13, v10 offset:0
ds_read_b32 v14, v10 offset:4
ds_read_b32 v15, v10 offset:8
v_xor_b32_e32 v13, 64, v13
v_xor_b32_e32 v14, 64, v14
v_or3_b32 v13, v13, v14, v15
v_cmp_eq_u32_e32 vcc, 0, v13
v_cndmask_b32_e64 v8, 0, 0.5, vcc
v_cmp_eq_u32_e32 vcc, 1, v0
v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v3, 0, 2.0, vcc
v_cmp_eq_u32_e32 vcc, 2, v0
v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v4, 0, 2.0, vcc
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
s_mov_b32 m0, 0x1003
s_cmp_eq_u32 s4, 0
s_cbranch_scc1 1
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
s_barrier
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7 done
exp param0 v3, v4, v8, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
