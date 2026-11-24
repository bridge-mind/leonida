; Each source has four unsigned bytes. Only bit 0 of each S2 byte rounds.
; Expected MRT0 words: 40ff7f80 40807f80 41800080 41123456.
	v_mov_b32 v0, 0x40ff00ff
	v_mov_b32 v1, 0x40ffff01
	v_mov_b32 v2, 0x01ff00ff
	v_lerp_u8 v4, v0, v1, v2
	v_mov_b32 v0, 0x40807fff
	v_mov_b32 v1, 0x407f8001
	v_mov_b32 v2, 0xfe01fe00
	v_lerp_u8 v5, v0, v1, v2
	v_mov_b32 v0, 0x41ff0000
	v_mov_b32 v1, 0x400000ff
	v_mov_b32 v2, 0x01010101
	v_lerp_u8 v6, v0, v1, v2
	v_mov_b32 v0, 0x42123456
	v_mov_b32 v1, 0x40123456
	v_mov_b32 v2, 0xfefefefe
	v_lerp_u8 v7, v0, v1, v2
	exp mrt0 v4, v5, v6, v7 done vm
	s_endpgm
