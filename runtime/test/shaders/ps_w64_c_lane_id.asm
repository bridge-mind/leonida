; SHADER-10 tier C: the raw lane number 0..63 (v_mbcnt_lo/hi of -1) used as data. On a 64-lane subgroup the pixels of a
; wave get 0..63 in lane order; on 32-lane subgroups 0..31 twice. r0 = lane id, r1 = lane id & 31 (inside a half the
; same everywhere), r2 = x = floor(u * 64) on the 64x64 target of vs_ids.
; Run: vk_harness vs_ids.spv ps_w64_c_lane_id.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 and r2 equal, r0
; differs (the detector must call this shader tier C).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mul_f32_e32 v2, 0x42800000, v2
v_cvt_u32_f32_e32 v2, v2
s_mov_b64 s[10:11], exec
v_mbcnt_lo_u32_b32 v6, -1, 0
v_mbcnt_hi_u32_b32 v6, -1, v6
v_and_b32_e32 v7, 31, v6
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v2, v8 done vm
s_endpgm
