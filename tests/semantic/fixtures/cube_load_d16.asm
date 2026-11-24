; Float R=0.5 at (1,2) on layer 9. With a one-component D16 load the
; exported register must be 0x3f803800 (old upper half, binary16 red).
v_mov_b32 v0, 0x3f800000
v_mov_b32 v4, 1
v_mov_b32 v5, 2
v_mov_b32 v6, 9
image_load v0, v[4:6], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm d16
s_waitcnt vmcnt(0)
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
