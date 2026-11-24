; SHADER-10, GPU check of ds_bpermute_b32 in a wave64 pixel shader: a wave64 ds_bpermute runs as two wave32 halves
; (ISA 10.4.4; tests/semantic/test_wave64_cross_half.py), so an address taken from the lane id reads the same lane
; whether the lane id is counted in the wave (0..63) or in a half (0..31): only address bits 2..6 are used.
; p = y * 64 + x (x = floor(u * 64), y = floor(v * 64) on the 64x64 target of vs_ids). The words depend only on the
; 2x2 quad layout of fragment lanes (lane 4k+0 top-left, +1 top-right, +2 bottom-left, +3 bottom-right):
;  r0 = ds_bpermute from lane ^ 1: the right/left neighbour, (x ^ 1) + 64 * y.
;  r1 = ds_bpermute from lane ^ 3: the diagonal pixel, (x ^ 1) + 64 * (y ^ 1).
;  r2 = ds_bpermute from lane ^ 16, twice: back to the own pixel, p.
;  r3 = ds_bpermute from lane & ~3 + 2 (the quad's bottom-left lane): (x & ~1) + 64 * (y | 1).
; The reads that go out and come back also need the partner lane to hold a pixel: vs_ids' full-screen triangle on
; the 64x64 target fills every wave.
; Run: vk_harness vs_ids.spv ps_w64_a_bpermute.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: equal images, equal to
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
v_mbcnt_lo_u32_b32 v10, -1, 0
v_mbcnt_hi_u32_b32 v10, -1, v10
v_xor_b32_e32 v11, 1, v10
v_lshlrev_b32_e32 v11, 2, v11
ds_bpermute_b32 v6, v11, v4
v_xor_b32_e32 v11, 3, v10
v_lshlrev_b32_e32 v11, 2, v11
ds_bpermute_b32 v7, v11, v4
v_xor_b32_e32 v11, 16, v10
v_lshlrev_b32_e32 v11, 2, v11
ds_bpermute_b32 v5, v11, v4
s_waitcnt lgkmcnt(0)
ds_bpermute_b32 v8, v11, v5
v_and_b32_e32 v11, -4, v10
v_or_b32_e32 v11, 2, v11
v_lshlrev_b32_e32 v11, 2, v11
ds_bpermute_b32 v9, v11, v4
s_waitcnt lgkmcnt(0)
exp mrt0 v6, v7, v8, v9 done vm
s_endpgm
