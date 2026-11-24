; Zero, full, shifted contiguous SGPR and sparse literal masks.
; Expected MRT0 words: 3f123456 4055aa7e 3f553456 3f523476.
	v_mov_b32 v0, 0x4055aa7e
	v_mov_b32 v1, 0x3f123456
	v_bfi_b32 v2, 0, v0, v1
	v_bfi_b32 v3, -1, v0, v1
	s_mov_b32 s8, 0x00ff0000
	v_bfi_b32 v4, s8, v0, v1
	v_bfi_b32 v5, 0x00f000f0, v0, v1
	exp mrt0 v2, v3, v4, v5 done vm
	s_endpgm
