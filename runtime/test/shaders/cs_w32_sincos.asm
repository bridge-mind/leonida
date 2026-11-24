; SHADER-88: v_sin_f32 / v_cos_f32 take the angle in turns and must reduce it in turns before scaling by 2*pi.
; Six large-angle cases (254.25, 254.5, -200.75, 100.3, 254.51, -254.51 turns); each stores 1 when
; |result - reference| < 1e-3 (the reference is sin/cos(2*pi*x) in double; a Vulkan Sin/Cos is only guaranteed to
; 2^-11 = 4.9e-4 in [-pi, pi], so the shader compares with a tolerance and stores a flag, not the float).
; Lane 0 stores out[0..5] = 1 1 1 1 1 1. (Expected references, f32 bits: 1.0, -1.0, 1.0, 0xbe9e39de,
; 0xbd808642, 0xbf7f7ed2.)
; group: 32,1,1
; Run: vk_compute_test cs_w32_sincos.spv 1 dump -> out[0..5] = 1 1 1 1 1 1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v9, 0x3a83126f
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x437e4000
v_sin_f32_e32 v2, v1
v_sub_f32_e32 v2, 0x3f800000, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x437e8000
v_cos_f32_e32 v2, v1
v_sub_f32_e32 v2, 0xbf800000, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xc348c000
v_sin_f32_e32 v2, v1
v_sub_f32_e32 v2, 0x3f800000, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x42c8999a
v_cos_f32_e32 v2, v1
v_sub_f32_e32 v2, 0xbe9e39de, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x437e828f
v_sin_f32_e32 v2, v1
v_sub_f32_e32 v2, 0xbd808642, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xc37e828f
v_cos_f32_e32 v2, v1
v_sub_f32_e32 v2, 0xbf7f7ed2, v2
v_and_b32_e32 v2, 0x7fffffff, v2
v_cmp_gt_f32_e32 vcc_lo, v9, v2
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
