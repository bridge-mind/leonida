; Mesh shader (NGG, wave64, 128 threads): ms_tri's triangle, but the blue channel of param0 is 0.5 only when a
; ballot + v_mbcnt compaction over BOTH halves of the wave is exact, 0 otherwise. Selected lanes: (lane & 7) == 5
; (lanes 5, 13, 21, 29 | 37, 45, 53, 61). s10 = s_bcnt1 of the ballot = 8 (one half would see 4). Each lane's
; compaction slot = v_mbcnt_lo(s6) + v_mbcnt_hi(s7) must equal lane >> 3 (upper lanes count the 4 low-half
; lanes); s11 = bcnt of the selected lanes whose slot is right = 8. s10 + s11 == 16 -> blue = 0.5, else 0.
; Run like ms_tri (needs a DXBC with PSV0, 128 threads, 64/64 outputs, triangles; see scripts/run_gpu_tests.py):
;   VKH_MESH=1 vk_harness ms_w64x_ballot.spv fs_passthru.spv out.raw
;   -> every pixel = ((x+0.5)/64, (y+0.5)/64, 0.5, 1), identical to ms_tri.
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
v_mov_b32_e32 v11, v0
s_bfe_u32 s4, s3, 0x40018
v_lshl_add_u32 v0, s4, 6, v0
v_and_b32_e32 v12, 7, v11
v_cmp_eq_u32_e64 s[6:7], 5, v12
s_bcnt1_i32_b64 s10, s[6:7]
v_mbcnt_lo_u32_b32 v13, s6, 0
v_mbcnt_hi_u32_b32 v13, s7, v13
v_lshrrev_b32_e32 v14, 3, v11
v_cmp_eq_u32_e64 s[8:9], v14, v13
s_and_b64 s[8:9], s[8:9], s[6:7]
s_bcnt1_i32_b64 s11, s[8:9]
s_add_u32 s12, s10, s11
s_cmp_eq_u32 s12, 16
s_cselect_b32 s13, 1, 0
v_cvt_f32_u32_e32 v15, s13
v_mul_f32_e32 v15, 0.5, v15
v_cmp_eq_u32_e32 vcc, 1, v0
v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v3, 0, 2.0, vcc
v_cmp_eq_u32_e32 vcc, 2, v0
v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v4, 0, 2.0, vcc
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v8, v15
s_mov_b32 m0, 0x1003
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7 done
exp param0 v3, v4, v8, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
