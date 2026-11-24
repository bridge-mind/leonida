; Known and partly known operands exercise the two low-five-bit masks.
; OR with the bits of 1.0 to return normal floats even for the zero mask.
	v_mov_b32 v1, 35
	v_bfm_b32 v2, 33, 35
	s_mov_b32 s8, 35
	v_bfm_b32 v3, s8, 31
	v_bfm_b32 v4, 32, v1
	v_bfm_b32 v5, 3, v1
	v_or_b32 v2, 0x3f800000, v2
	v_or_b32 v3, 0x3f800000, v3
	v_or_b32 v4, 0x3f800000, v4
	v_or_b32 v5, 0x3f800000, v5
	exp mrt0 v2, v3, v4, v5 done vm
	s_endpgm
