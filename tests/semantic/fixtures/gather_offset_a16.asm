; Two 2D mips (8x8, 4x4), with green texel value 100*mip + 10*y + x.
; Expected taps at mip 1: (131,132,122,121). The mip-zero form gives (43,44,34,33).
; Full-word offset (-1,+1), packed coordinates (0.625,0.375), LOD 1 and padding.
s_mov_b32 s8, 0x12
s_mov_b32 s9, 0x100000
v_mov_b32 v4, 0x013f
v_mov_b32 v5, 0x36003900
v_mov_b32 v6, 0x48003c00
image_gather4_l_o v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
