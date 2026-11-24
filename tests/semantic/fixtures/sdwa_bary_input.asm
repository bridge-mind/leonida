; For I=0.25 and J=0.5, expect MRT0 words 3e8000ab 3f00abcd 3e8000ab 3f00abcd.
	v_mov_b32 v4, 0xab
	v_mov_b32 v5, 0xabcd
	v_mov_b32_sdwa v0, v4 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
	v_mov_b32_sdwa v1, v5 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
	exp mrt0 v0, v1, v0, v1 done vm
	s_endpgm
