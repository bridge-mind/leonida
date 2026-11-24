; Mesh shader (NGG, wave64, 128 threads): fullscreen triangle like ms_tri, plus SV_ClipDistance[0] = NDC x
; exported in pos1.x (`exp pos1 v1, off, off, off done`). Clips the left half of the 64x64 viewport (x < 0):
; expected covered pixels = 2048/4096 (right half x in [32..63], left half untouched clear color -1).
; Needs a DXBC with PSV0 (128 threads, 64/64 outputs, triangles) and OSG1 declaring SV_ClipDistance (mask 0x1).
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
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v6, v7
exp pos1 v1, off, off, off done
exp param0 v3, v4, v8, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
