; SHADER-10 tier A (docs/design.md, "Wave64 graphics: partition invariance"): the early exit of a pixel kill. A wave64
; pixel shader whose 32-lane halves can lose every pixel while the other half lives on gives the same image when its
; 64 lanes are cut into two 32-lane subgroups. x = floor(u * 64), y = floor(v * 64) are the pixel's coordinates on
; the 64x64 target (u, v = attr0.xy of vs_ids), so every expected word is a function of (x, y) and not of the lane
; the pixel landed on.
;  Three kills, each `s_andn2_b64 live, live, <kill mask>` then `s_cbranch_scc0` to its own exit block, which only
;  sets EXEC from the (empty) live mask, exports null with done vm and ends (an exit shared by several kills reads a
;  mask redefined behind the earlier votes: tier C tainted-exec, a stale scalar). The run side narrows EXEC into the
;  whole quads of the live mask (s_wqm_b64, s_and_b64 exec) and never widens it.
;  (1) x < 30, through v_cmpx (EXEC = the pixels to kill); (2) y >= 50, through VCC; (3) x % 4 == 3 and y odd
;      ((y & 1) * 4 + (x & 3) == 7), through VCC.
;      The first two discard large regions: whole waves and, at x = 30 and y = 50, halves whose pixels are all dead
;      next to halves that live (for a wave tiled as 8x8 cut into 8x4 or 4x8 halves, among others).
;  Before the export EXEC is narrowed to the live mask itself, so the helper lanes of the quads are discarded.
;  Live pixel: r0 = 7 * y + x, r1 = 1, r2 = 2000 + y, r3 = x % 4 + 1. A discarded pixel keeps the harness clear
;  value, -1.0f in every word (0xbf800000).
; Run: vk_harness vs_ids.spv ps_w64_a_discard.spv out.raw, once with VKH_SUBGROUP_SIZE=64 and once with 32: the two
; images must be equal (and equal to the expected words). Output is raw bits (r0, r1, r2, r3) per pixel.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
s_mov_b64 s[10:11], exec
s_mov_b64 s[20:21], exec
v_cmpx_gt_u32_e32 30, v2
s_andn2_b64 s[20:21], s[20:21], exec
s_cbranch_scc0 34
s_wqm_b64 s[22:23], s[20:21]
s_and_b64 exec, s[10:11], s[22:23]
v_cmp_lt_u32_e32 vcc, 49, v3
s_andn2_b64 s[20:21], s[20:21], vcc
s_cbranch_scc0 25
s_wqm_b64 s[22:23], s[20:21]
s_and_b64 exec, exec, s[22:23]
v_and_b32_e32 v4, 3, v2
v_and_b32_e32 v5, 1, v3
v_lshlrev_b32_e32 v5, 2, v5
v_add_nc_u32_e32 v5, v5, v4
v_cmp_eq_u32_e32 vcc, 7, v5
s_andn2_b64 s[20:21], s[20:21], vcc
s_cbranch_scc0 12
s_wqm_b64 s[22:23], s[20:21]
s_and_b64 exec, exec, s[22:23]
v_mul_u32_u24_e32 v6, 7, v3
v_add_nc_u32_e32 v6, v6, v2
v_mov_b32_e32 v7, 1
v_add_nc_u32_e32 v8, 0x7d0, v3
v_add_nc_u32_e32 v9, 1, v4
s_and_b64 exec, exec, s[20:21]
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
s_mov_b64 exec, s[20:21]
exp null off, off, off, off done vm
s_endpgm
s_mov_b64 exec, s[20:21]
exp null off, off, off, off done vm
s_endpgm
s_mov_b64 exec, s[20:21]
exp null off, off, off, off done vm
s_endpgm
