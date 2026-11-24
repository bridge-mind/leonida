; Typed buffer formats added for the SCALED, SINT, 10_11_11 / 11_11_10 and 2_10_10_10 / 10_10_10_2 families, and the older
; ones' stores that now follow the hardware (tools/buffer_format_fuzz.py, docs/buffer_formats.md). One thread. Words 0..127: 32 loads, 4 words each,
; of the 16-byte inputs written at 0x200 (words 128..255): 29 through a V# whose FORMAT is the code, 3 through tbuffer's
; own format; a channel the format lacks repeats component (c mod n) through the V# and is 0 through tbuffer. Words 256..335:
; 20 buffer_store_format_xyzw into 0x400.., the rest of each 16 bytes left at 0xeeeeeeee. A two-component format takes
; channels x and w through a V#; 16-bit float stores round toward zero and saturate at 65504; NORM stores round the exact product.
; The expected words are the MODEL of tools/buffer_format_fuzz.py, which the Radeon 880M (gfx1150, RDNA3.5) and a real RDNA2 (gfx1036,
; @AwkGrep256, #2022) both reproduced: 1,330,469 elements of the 63 formats GFX11 has, and 199,144 of the 14 codes only GFX10 has.
; Run: VKC_OUT_WORDS=336 VKC_DUMP=<file> vk_compute_test cs_fmt_all.spv 1 dump; check.py compares all 336 words.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v8, 0xff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x200
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x80
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x210
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xc8
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x220
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x9c
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x230
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x8000
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x240
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xea60
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x250
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xfc18
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x260
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x807f
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x270
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xff01
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x280
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x7f80ff01
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x290
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xff800100
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2a0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xff800100
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2b0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x80007fff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2c0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xffff0001
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2d0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x7fff8000
v_mov_b32_e32 v9, 0xffff0001
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2e0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x702003c0
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x2f0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x7e0fc0
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x300
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xffc1ffbf
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x310
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xf7f001e0
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x320
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x805ff801
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x330
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xffdff802
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x340
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x6017fe00
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x350
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xe00003ff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x360
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xbff7fe00
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x370
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xbff7fe00
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x380
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xfff181e0
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x390
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x7f80ff01
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3a0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x80007fff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3b0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xfffff800
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3c0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0x800ffe00
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3d0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xffc00fff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3e0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
v_mov_b32_e32 v8, 0xfff003ff
v_mov_b32_e32 v9, 0x0
v_mov_b32_e32 v10, 0x0
v_mov_b32_e32 v11, 0x0
v_mov_b32_e32 v2, 0x3f0
buffer_store_dwordx4 v[8:11], v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, s2
v_mov_b32_e32 v3, 0x200
s_mov_b32 s7, 0x31005fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x210
s_mov_b32 s7, 0x31006fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x10
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x220
s_mov_b32 s7, 0x31003fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x20
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x230
s_mov_b32 s7, 0x31004fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x30
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x240
s_mov_b32 s7, 0x3100cfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x40
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x250
s_mov_b32 s7, 0x31009fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x50
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x260
s_mov_b32 s7, 0x3100afac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x60
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x270
s_mov_b32 s7, 0x31013fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x70
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x280
s_mov_b32 s7, 0x31010fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x80
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x290
s_mov_b32 s7, 0x3103dfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x90
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2a0
s_mov_b32 s7, 0x3103afac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xa0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2b0
s_mov_b32 s7, 0x3103bfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xb0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2c0
s_mov_b32 s7, 0x3101cfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xc0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2d0
s_mov_b32 s7, 0x31019fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xd0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2e0
s_mov_b32 s7, 0x31044fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xe0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x2f0
s_mov_b32 s7, 0x31024fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0xf0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x300
s_mov_b32 s7, 0x31024fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x100
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x310
s_mov_b32 s7, 0x31024fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x110
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x320
s_mov_b32 s7, 0x3102bfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x120
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x330
s_mov_b32 s7, 0x3102dfac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x130
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x340
s_mov_b32 s7, 0x31031fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x140
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x350
s_mov_b32 s7, 0x31033fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x150
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x360
s_mov_b32 s7, 0x31034fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x160
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x370
s_mov_b32 s7, 0x31035fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x170
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x380
s_mov_b32 s7, 0x31037fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x180
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x390
s_mov_b32 s7, 0x31016fac
tbuffer_load_format_xyzw v[4:7], v3, s[4:7], 0 format:[BUF_FMT_11_11_10_FLOAT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x190
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3a0
s_mov_b32 s7, 0x31016fac
tbuffer_load_format_xyzw v[4:7], v3, s[4:7], 0 format:[BUF_FMT_8_8_8_8_SINT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1a0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3b0
s_mov_b32 s7, 0x31016fac
tbuffer_load_format_xyzw v[4:7], v3, s[4:7], 0 format:[BUF_FMT_16_16_SINT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1b0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3c0
s_mov_b32 s7, 0x3101efac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1c0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3d0
s_mov_b32 s7, 0x31026fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1d0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3e0
s_mov_b32 s7, 0x31022fac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1e0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v3, 0x3f0
s_mov_b32 s7, 0x3102afac
buffer_load_format_xyzw v[4:7], v3, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v8, 0x1f0
buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
v_mov_b32_e32 v4, 0x100
v_mov_b32_e32 v5, 0x0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31006fac
v_mov_b32_e32 v9, 0x400
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0xffffff00
v_mov_b32_e32 v5, 0x0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31006fac
v_mov_b32_e32 v9, 0x410
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x1ff
v_mov_b32_e32 v5, 0x0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31005fac
v_mov_b32_e32 v9, 0x420
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0x0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31003fac
v_mov_b32_e32 v9, 0x430
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x8000
v_mov_b32_e32 v5, 0x0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x3100cfac
v_mov_b32_e32 v9, 0x440
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x12345
v_mov_b32_e32 v5, 0xffff0000
v_mov_b32_e32 v6, 0x7
v_mov_b32_e32 v7, 0xffff8000
s_mov_b32 s7, 0x3101cfac
v_mov_b32_e32 v9, 0x450
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0x3f000000
v_mov_b32_e32 v6, 0x3e800000
v_mov_b32_e32 v7, 0x3f400000
s_mov_b32 s7, 0x31010fac
v_mov_b32_e32 v9, 0x460
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0xbf800000
v_mov_b32_e32 v5, 0x3f000000
v_mov_b32_e32 v6, 0x3f800000
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x3101ffac
v_mov_b32_e32 v9, 0x470
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x7ff
v_mov_b32_e32 v5, 0x5
v_mov_b32_e32 v6, 0xffffffff
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31029fac
v_mov_b32_e32 v9, 0x480
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x7149f2ca
v_mov_b32_e32 v5, 0x3effffff
v_mov_b32_e32 v6, 0x7f800001
v_mov_b32_e32 v7, 0x80000000
s_mov_b32 s7, 0x31047fac
v_mov_b32_e32 v9, 0x490
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3e99999a
v_mov_b32_e32 v5, 0x37c000c0
v_mov_b32_e32 v6, 0x0
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31017fac
v_mov_b32_e32 v9, 0x4a0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0xbf800000
v_mov_b32_e32 v6, 0x3f000000
v_mov_b32_e32 v7, 0xbf000000
s_mov_b32 s7, 0x3103bfac
v_mov_b32_e32 v9, 0x4b0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0x40000000
v_mov_b32_e32 v6, 0x3f000000
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31024fac
v_mov_b32_e32 v9, 0x4c0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0xbf800000
v_mov_b32_e32 v5, 0x7f800000
v_mov_b32_e32 v6, 0x47ffe000
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x31024fac
v_mov_b32_e32 v9, 0x4d0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0x477fe000
v_mov_b32_e32 v6, 0x7f800001
v_mov_b32_e32 v7, 0x0
s_mov_b32 s7, 0x3102bfac
v_mov_b32_e32 v9, 0x4e0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f000000
v_mov_b32_e32 v5, 0xbf800000
v_mov_b32_e32 v6, 0x3f800000
v_mov_b32_e32 v7, 0x3e99999a
s_mov_b32 s7, 0x3102dfac
v_mov_b32_e32 v9, 0x4f0
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3f800000
v_mov_b32_e32 v5, 0x3f000000
v_mov_b32_e32 v6, 0x3e800000
v_mov_b32_e32 v7, 0x3f800000
s_mov_b32 s7, 0x31034fac
v_mov_b32_e32 v9, 0x500
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x7fffffff
v_mov_b32_e32 v5, 0x80000000
v_mov_b32_e32 v6, 0x5
v_mov_b32_e32 v7, 0x5
s_mov_b32 s7, 0x31037fac
v_mov_b32_e32 v9, 0x510
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0x3ff
v_mov_b32_e32 v5, 0x5
v_mov_b32_e32 v6, 0x7d0
v_mov_b32_e32 v7, 0x9
s_mov_b32 s7, 0x31030fac
v_mov_b32_e32 v9, 0x520
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
v_mov_b32_e32 v4, 0xffff8000
v_mov_b32_e32 v5, 0x8000
v_mov_b32_e32 v6, 0x7fff
v_mov_b32_e32 v7, 0xffffffff
s_mov_b32 s7, 0x31046fac
v_mov_b32_e32 v9, 0x530
buffer_store_format_xyzw v[4:7], v9, s[4:7], 0 offen
s_endpgm
