; Sample vk_texture_test's two-mip texture (--two-mip: 128x128 quadrants, solid cyan mip 1) with
; image_sample_d_cl ... a16: the four derivatives stay f32 (1/128, so the unclamped LOD is 0), then the two f16
; coordinates and the f16 min-LOD clamp are packed, with an f16 pad of 1.0 after the clamp. The clamp is 0.0 left of
; x = 32 and 1.0 from there on, so the left half must read mip 0 (red over blue) and the right half mip 1 (cyan).
; Taken from the pad the clamp would be 1.0 everywhere, and read as pairs the derivatives and coordinates would come
; from the wrong words.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mov_b32_e32 v8, 0x3c000000  ; du/dx = 1/128
v_mov_b32_e32 v9, 0           ; dv/dx
v_mov_b32_e32 v10, 0          ; du/dy
v_mov_b32_e32 v11, 0x3c000000 ; dv/dy = 1/128
v_cvt_pkrtz_f16_f32 v12, v2, v3
v_mov_b32_e32 v14, 0x3c000000 ; clamp 0.0, pad 1.0
v_mov_b32_e32 v15, 0x3c003c00 ; clamp 1.0, pad 1.0
v_cmp_le_f32_e32 vcc_lo, 0.5, v2
v_cndmask_b32_e32 v13, v14, v15, vcc_lo
image_sample_d_cl v[4:7], v[8:13], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
