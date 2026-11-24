; R,G = binary64(-2147483647), B,A = binary64(2147483649), low word first.
; Expected MRT0 words: ffc00000 c1dfffff 00200000 41e00000. The signed conversion overwrites its input.
	v_mov_b32_e32 v4, 0x80000001
	v_cvt_f64_u32_e32 v[6:7], v4
	v_cvt_f64_i32_e32 v[4:5], v4
	exp mrt0 v4, v5, v6, v7 done vm
	s_endpgm
