; Image atomics on a 2D r32 uint image: image_atomic_add, sub, and, or, xor, smin, smax, umin, umax and cmpswap.
; RDNA2 ISA (MIMG): each one reads the texel, writes OP(texel, data) back and, with glc, returns the previous texel in
; the data register; without glc the data register is left alone. cmpswap takes two data registers, the new value
; and the comparand, and writes the new value only where the texel equals the comparand. A lane that EXEC turns off
; does nothing: only thread 0 is on for cases 0 to 15, so an op the other 63 lanes also ran would show (an add
; applied 64 times). Case 16 runs before that, with all 64 threads on, each adding 1 to the same texel.
; Run: vk_store_test cs_imgatomic.spv words
; The tool fills the uint image (T# 0x2000..., what the atomics address) with M = 100 + i at texel i = 8y + x and
; the sint image (T# 0x3000...) with -2i. Case c uses texel c, and stores what its data register holds afterwards
; into texel c of the sint image. A = 0xfffffff0 is -16 signed and 4294967280 unsigned, so it tells the signed
; min and max from the unsigned ones; each of those four also has a case that goes the other way.
;    c  op        data            M     uint word c afterwards               sint word c (data register)
;    0  add       A               100   100 + A mod 2^32 = 0x00000054        0x00000064
;    1  sub       200             101   101 - 200 mod 2^32 = 0xffffff9d      0x00000065
;    2  and       0x0f0f0f0f      102   0x66 & 0x0f0f0f0f = 0x00000006       0x00000066
;    3  or        0xf000          103   0x67 | 0xf000 = 0x0000f067           0x00000067
;    4  xor       0xff            104   0x68 ^ 0xff = 0x00000097             0x00000068
;    5  smin      A               105   -16 < 105: 0xfffffff0                0x00000069
;    6  smin      200             106   106 < 200: 0x0000006a                0x0000006a
;    7  smax      A               107   107 > -16: 0x0000006b                0x0000006b
;    8  smax      200             108   200 > 108: 0x000000c8                0x0000006c
;    9  umin      A               109   109 < 4294967280: 0x0000006d         0x0000006d
;   10  umin      5               110   5 < 110: 0x00000005                  0x0000006e
;   11  umax      A               111   4294967280 > 111: 0xfffffff0         0x0000006f
;   12  umax      5               112   112 > 5: 0x00000070                  0x00000070
;   13  cmpswap   0xc0ffee, 113   113   equal, swapped: 0x00c0ffee           0x00000071
;   14  cmpswap   0xbad, 115      114   not equal, kept: 0x00000072          0x00000072
;   15  add       7, no glc       115   115 + 7 = 0x0000007a                 0x00000007 (the data, not the old texel)
;   16  add       1, 64 threads   116   116 + 64 = 0x000000b4                (nothing stored: -2 * 16 = 0xffffffe0)
; Every other texel keeps its fill: uint word i = 100 + i for i = 17..63 and sint word i = -2i for i = 16..63.
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v3, 1
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 0xfffffff0
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v3, 200
image_atomic_sub v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v3, 0x0f0f0f0f
image_atomic_and v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v3, 0xf000
image_atomic_or v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v3, 0xff
image_atomic_xor v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v3, 0xfffffff0
image_atomic_smin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
v_mov_b32_e32 v3, 200
image_atomic_smin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 7
v_mov_b32_e32 v3, 0xfffffff0
image_atomic_smax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 200
image_atomic_smax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v3, 0xfffffff0
image_atomic_umin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v3, 5
image_atomic_umin v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
v_mov_b32_e32 v3, 0xfffffff0
image_atomic_umax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v3, 5
image_atomic_umax v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v3, 0xc0ffee
v_mov_b32_e32 v4, 113
image_atomic_cmpswap v[3:4], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
v_mov_b32_e32 v3, 0xbad
v_mov_b32_e32 v4, 115
image_atomic_cmpswap v[3:4], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc
s_waitcnt vmcnt(0)
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 7
v_mov_b32_e32 v3, 7
image_atomic_add v3, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_store v3, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
