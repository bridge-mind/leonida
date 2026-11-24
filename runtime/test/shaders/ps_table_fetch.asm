; vk_texture_test fetch-sint / fetch-uint / fetch-miss. Texel (0,0) holds bytes
; 82 22 f8 4d: expect (-126,34,-8,77) or (130,34,248,77). A key mismatch
; fetches magenta UNORM at index zero. Convert raw integer words for RGBA32F output.
s_load_dwordx8 s[8:15], s[0:1], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v1, 0
image_load v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_cvt_f32_i32_e32 v4, v4
v_cvt_f32_i32_e32 v5, v5
v_cvt_f32_i32_e32 v6, v6
v_cvt_f32_i32_e32 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
