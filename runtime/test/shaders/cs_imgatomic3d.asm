; Image atomics on a 3D r32 uint image (dim:SQ_RSRC_IMG_3D, three coordinates): the same RDNA2 ISA rules as in
; cs_imgatomic.asm, which runs every op on a 2D image; these six cases cover the 3D addressing with an arithmetic op,
; an unsigned and a signed compare, cmpswap both ways and an op without glc.
; Run: vk_store_test cs_imgatomic3d.spv words
; The tool fills a 4x4x4 volume (T# 0x5000...) with V = 200 + i at texel i = 16z + 4y + x. The three coordinates of
; a case all differ, so a swapped pair lands on another texel. Case c stores what its data register holds
; afterwards into texel c of the sint image (T# 0x3000...). A = 0xfffffff0 is -16 signed.
;    c  op        data            (x, y, z)   i    V     volume word i afterwards           sint word c
;    0  add       0x10            (1, 2, 3)   57   257   257 + 16 = 0x00000111              0x00000101
;    1  umin      3               (3, 0, 1)   19   219   3 < 219: 0x00000003                0x000000db
;    2  smax      A               (0, 3, 2)   44   244   244 > -16: 0x000000f4              0x000000f4
;    3  cmpswap   0xd00d, 206     (2, 1, 0)    6   206   equal, swapped: 0x0000d00d         0x000000ce
;    4  cmpswap   0xbad, 0        (2, 3, 1)   30   230   not equal, kept: 0x000000e6        0x000000e6
;    5  xor       0xff, no glc    (3, 1, 2)   39   239   0xef ^ 0xff = 0x00000010           0x000000ff (the data)
; Every other word keeps its fill: volume word i = 200 + i, sint word i = -2i for i = 6..63, uint word i = 100 + i.
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s16, 0x5000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x55
s_mov_b32 s19, 0x55
s_mov_b32 s20, 0x55
s_mov_b32 s21, 0x55
s_mov_b32 s22, 0x55
s_mov_b32 s23, 0x55
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 3
v_mov_b32_e32 v4, 0x10
image_atomic_add v4, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 0
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 1
v_mov_b32_e32 v4, 3
image_atomic_umin v4, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 1
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 3
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v4, 0xfffffff0
image_atomic_smax v4, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 2
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v7, 0xd00d
v_mov_b32_e32 v8, 206
image_atomic_cmpswap v[7:8], v[1:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 3
image_store v7, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v2, 3
v_mov_b32_e32 v3, 1
v_mov_b32_e32 v7, 0xbad
v_mov_b32_e32 v8, 0
image_atomic_cmpswap v[7:8], v[1:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
s_waitcnt vmcnt(0)
v_mov_b32_e32 v5, 4
image_store v7, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v4, 0xff
image_atomic_xor v4, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
v_mov_b32_e32 v5, 5
image_store v4, v[5:6], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
