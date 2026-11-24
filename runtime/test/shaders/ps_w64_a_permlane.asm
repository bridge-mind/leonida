; SHADER-10 tier A, GPU check of v_permlane16_b32 and v_permlanex16_b32 in a wave64 pixel shader (they never leave a
; 32-lane half: permlane16 reads within the lane's row of 16, permlanex16 from the other row of the same half).
; p = y * 64 + x is the pixel number (x = floor(u * 64), y = floor(v * 64) on the 64x64 target of vs_ids). The words
; must not depend on how the rasterizer packs pixels into lanes, only on the 2x2 quad layout of fragment lanes (lane
; 4k+0 top-left, +1 top-right, +2 bottom-left, +3 bottom-right), so every read stays in the lane's quad or comes back:
;  r0 = permlane16 with selects lane ^ 1 (0x67452301, 0xefcdab89): the right/left neighbour, (x ^ 1) + 64 * y.
;  r1 = permlane16 with selects lane ^ 2 (0x54761032, 0xdcfe98ba): the pixel below/above, x + 64 * (y ^ 1).
;  r2 = permlanex16 with the identity selects (0x76543210, 0xfedcba98) twice: lane L reads lane L ^ 16 of its half,
;       then back, so r2 = p.
;  r3 = permlanex16 with selects lane ^ 3 (0x45670123, 0xcdef89ab) of a value already read from lane ^ 3 of the other
;       row (the same selects with permlanex16), so r3 = p again, through a non-identity mapping.
; The reads that go out and come back also need the partner lane to hold a pixel: vs_ids' full-screen triangle on
; the 64x64 target fills every wave.
; Run: vk_harness vs_ids.spv ps_w64_a_permlane.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: equal images, equal to
; the derived words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
v_lshl_add_u32 v4, v3, 6, v2
s_mov_b64 s[10:11], exec
s_mov_b32 s20, 0x67452301
s_mov_b32 s21, 0xefcdab89
v_permlane16_b32 v6, v4, s20, s21
s_mov_b32 s22, 0x54761032
s_mov_b32 s23, 0xdcfe98ba
v_permlane16_b32 v7, v4, s22, s23
s_mov_b32 s24, 0x76543210
s_mov_b32 s25, 0xfedcba98
v_permlanex16_b32 v5, v4, s24, s25
v_permlanex16_b32 v8, v5, s24, s25
s_mov_b32 s26, 0x45670123
s_mov_b32 s27, 0xcdef89ab
v_permlanex16_b32 v5, v4, s26, s27
v_permlanex16_b32 v9, v5, s26, s27
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
