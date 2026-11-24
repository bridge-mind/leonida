	v_mov_b32 v0, 0
	v_mov_b32 v1, 0xbffc0000
	s_mov_b32 s0, 0
	s_mov_b32 s1, 0xc0040000
	v_trunc_f64_e32 v[0:1], v[0:1]
	v_ceil_f64_e32 v[2:3], s[0:1]
	v_floor_f64_e32 v[4:5], 0xbff40000
	v_rndne_f64_e32 v[6:7], s[0:1]
	exp mrt0 v0, v1, v2, v3
	exp mrt1 v4, v5, v6, v7 done vm
	s_endpgm
