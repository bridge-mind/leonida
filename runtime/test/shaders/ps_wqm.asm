; Whole-quad mode in a pixel shader. x = floor(u * 64), y = floor(v * 64) are the pixel's coordinates on the 64x64
; target (u, v = attr0.xy of vs_ids, 0..1 over the image). Own mask: x % 4 == 1 and y % 4 == 2, one pixel in 16.
; s_wqm_b32 of it must light the whole 2x2 quad that pixel belongs to: x % 4 in {0,1} and y % 4 in {2,3}, so pixel
; (x, y) is in the WQM mask iff ((x >> 1) & 1) == 0 and ((y >> 1) & 1) == 1, and each pixel's own bit is contained
; in it. The target is fully covered, so no quad has a helper lane. Assumption: a hardware quad (4 consecutive lanes
; of the pixel wave) is the 2x2 pixel quad Vulkan's quad operations use.
; Run: vk_harness vs_ids.spv ps_wqm.spv out.raw. Per pixel (x, y) the words are (wqm, own, wqm, own) with
; wqm = 1 iff x % 4 < 2 and y % 4 >= 2, own = 1 iff x % 4 == 1 and y % 4 == 2.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_and_b32_e32 v4, 3, v2
v_and_b32_e32 v5, 3, v3
v_cmp_eq_u32_e32 vcc_lo, 1, v4
v_cmp_eq_u32_e64 s6, 2, v5
s_and_b32 s7, vcc_lo, s6
s_wqm_b32 s8, s7
v_cndmask_b32_e64 v6, 0, 1, s8
v_cndmask_b32_e64 v7, 0, 1, s7
exp mrt0 v6, v7, v6, v7 done vm
s_endpgm
