; The T# comes from user SGPRs and the S# words are literal. Both array indices still
; come from the runtime hash table. The sample uses bindings 0 and 1 and needs capability 29.
s_mov_b32 s8, 0x92
s_mov_b32 s9, 0xc00000
s_mov_b32 s10, 0x14500000
s_mov_b32 s11, 0
v_mov_b32_e32 v4, 0.5
v_mov_b32_e32 v5, 0.5
image_sample_lz v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
