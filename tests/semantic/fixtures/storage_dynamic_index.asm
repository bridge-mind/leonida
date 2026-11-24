; A literal 2D T# still goes through the runtime hash table. At its returned index,
; the UINT alias (binding 9) writes 42 to (0, 0). All three typed aliases are emitted.
s_mov_b32 s0, 1
s_mov_b32 s1, 0x01400000
s_mov_b32 s2, 0
s_mov_b32 s3, 0x90000000
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_mov_b32 s6, 0
s_mov_b32 s7, 0
v_mov_b32_e32 v0, 42
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
image_store v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
