; SHADER-10 tier A, GPU check of ds_swizzle_b32 in a wave64 pixel shader (the swizzle works on groups of 32 lanes, so
; it never leaves a half). p = y * 64 + x is the pixel number (x = floor(u * 64), y = floor(v * 64) on the 64x64 target
; of vs_ids). The words depend only on the 2x2 quad layout of fragment lanes (lane 4k+0 top-left, +1 top-right,
; +2 bottom-left, +3 bottom-right), not on how pixels are packed into lanes:
;  r0 = quad-permute mode, selects 3, 2, 1, 0 (offset 0x801b, swizzle(QUAD_PERM,3,2,1,0)): the diagonal pixel,
;       (x ^ 1) + 64 * (y ^ 1).
;  r1 = bitmask mode, and 0x1f, xor 1 (offset 0x041f, swizzle(SWAP,1)): the right/left neighbour, (x ^ 1) + 64 * y.
;  r2 = bitmask mode, and 0x1f, xor 0x10 (offset 0x401f, swizzle(SWAP,16)) twice: lane L reads lane L ^ 16, then
;       back, so r2 = p.
;  r3 = bitmask mode, and 0x1c (offset 0x001c, swizzle(BROADCAST,4,0)): every lane reads its quad's first lane,
;       (x & ~1) + 64 * (y & ~1).
; The reads that go out and come back also need the partner lane to hold a pixel: vs_ids' full-screen triangle on
; the 64x64 target fills every wave.
; Run: vk_harness vs_ids.spv ps_w64_a_swizzle.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: equal images, equal to
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
ds_swizzle_b32 v6, v4 offset:0x801b
ds_swizzle_b32 v7, v4 offset:0x41f
ds_swizzle_b32 v5, v4 offset:0x401f
s_waitcnt lgkmcnt(0)
ds_swizzle_b32 v8, v5 offset:0x401f
ds_swizzle_b32 v9, v4 offset:0x1c
s_waitcnt lgkmcnt(0)
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
