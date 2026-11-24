; image_get_resinfo on a 16x4 image with five mip levels. RDNA2 ISA (MIMG): the address register holds a mip level,
; there is no sampler, and the result is four integers, {num_mip_levels, depth, height, width} in VGPRs 3..0, of
; which dmask keeps some and packs them into consecutive registers. Width and height are those of the level asked
; for (halved per level, never below 1), the depth word is 0 for a 2D image (not the T#'s DEPTH field + 1), and the
; level count is that of the view; past the last level width, height and depth are 0 (both measured on RDNA2 gfx1036
; with tools/image_query_probe.py, SHADER-388).
; Run: vk_store_test cs_resinfo.spv words
; The T# in s[8:15] is the one of that image, as Mesa's gfx10 register database lays it out (and as
; runtime/src/gfx10_descriptors.h writes it): word 1 = FORMAT 56 (rgba8 unorm) << 20 | (width - 1 = 15) low two bits
; << 30, word 2 = (15 >> 2) | (height - 1 = 3) << 14, word 3 = DST_SEL x, y, z, w (0xfac) | LAST_LEVEL 4 << 16 |
; TYPE 9 (2D) << 28, word 5 = MAX_MIP 4 << 4. vk_store_test binds a 16x4 rgba8 image with five levels behind exactly
; these eight words. Thread 0 stores each result word into its own texel of the uint image (T# 0x2000...), texel
; i = 8y + x:
;   level 0, dmask 0xf  -> words  0..3  = 16 4 0 5
;   level 1, dmask 0xf  -> words  4..7  =  8 2 0 5
;   level 3, dmask 0xf  -> words  8..11 =  2 1 0 5     (4 >> 3 would be 0: the height stays 1)
;   level 2, dmask 0x3  -> words 12..13 =  4 1         width and height
;                          word  14     = 0xabcd       the register after them is not written
;   level 1, dmask 0xa  -> words 16..17 =  2 5         height and level count, packed into two registers
;   level 4, dmask 0x8  -> word  18     =  5           the level count alone
;   level 5, dmask 0xf  -> words 20..23 =  0 0 0 5     past the last level
; Every other word keeps the tool's fill, 100 + i, and the sint image its -2i.
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s8, 0x4000
s_mov_b32 s9, 0xc3800000
s_mov_b32 s10, 0xc003
s_mov_b32 s11, 0x90040fac
s_mov_b32 s12, 0
s_mov_b32 s13, 0x40
s_mov_b32 s14, 0
s_mov_b32 s15, 0
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
v_mov_b32_e32 v3, 0
image_get_resinfo v[4:7], v3, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v1, 0
image_store v4, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
image_store v5, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
image_store v6, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
image_store v7, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 1
image_get_resinfo v[4:7], v3, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v1, 4
image_store v4, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
image_store v5, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
image_store v6, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 7
image_store v7, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 3
image_get_resinfo v[4:7], v3, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v1, 0
image_store v4, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
image_store v5, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 2
image_store v6, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 3
image_store v7, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 2
v_mov_b32_e32 v10, 0xabcd
image_get_resinfo v[8:9], v3, s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v1, 4
image_store v8, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
image_store v9, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
image_store v10, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 1
image_get_resinfo v[11:12], v3, s[8:15] dmask:0xa dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v1, 0
image_store v11, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 1
image_store v12, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 4
image_get_resinfo v13, v3, s[8:15] dmask:0x8 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v1, 2
image_store v13, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v3, 5
image_get_resinfo v[4:7], v3, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v2, 2
v_mov_b32_e32 v1, 4
image_store v4, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 5
image_store v5, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 6
image_store v6, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_mov_b32_e32 v1, 7
image_store v7, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
