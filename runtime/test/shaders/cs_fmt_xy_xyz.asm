; buffer_load_format_xy/xyz, tbuffer_load_format_xy/xyz and buffer_store_format_xy/xyz (TEST-185). Every expected
; word in the original fixture was measured by running its body natively on a Radeon 680M (gfx1035, RDNA2) through tools/hw_oracle's kernel
; wrapper, with s[0:3] set to the V# vk_compute_test passes; wave32 and wave64 gave the same 84 words. One thread runs.
; The updated UINT store at 0x2a0 and all 84 current words were checked on gfx1036 in both wave sizes.
; - Loads write only their 2 or 3 VGPRs; the next ones keep 0x5a5a00xx. A missing channel repeats one the format has
;   (component c reads c mod n, SHADER-86): 32_32 xyz gives (x, y, x), 32 xy gives (x, x).
; - tbuffer loads take the instruction's format, not the V#'s.
; - Bounds: the whole element must end inside num_records (OOB_SELECT 3), even when the loaded components do: with
;   NR 0x208, an xy load of a 16-byte element at 0x200 reads 0, 0. A 32_32_32_32 element that fits reads normally.
; - Stores write the whole element: x, y (or x, y, z) and 0 for the format's other components (xy into 32_32_32_32
;   leaves z = w = 0, xyz into 16_16_16_16 leaves w = 0). Into a one-channel format only x is written. A store whose
;   element ends past num_records writes nothing.
; Inputs: RGBA32F (1, 2, 3, 4) at 0x200, RGBA16F (1, 2, 3, 4) at 0x210. Store regions 0x240..0x2cf start as
; 0xa5a50000 | address. out[0..47] are the 12 loads, out[48..83] the 9 store regions read back, in source order.
; Run: VKC_OUT_WORDS=192 VKC_DUMP=<file> vk_compute_test cs_fmt_xy_xyz.spv 1 dump; check.py compares words 0..127 of
; the dump with FMT_XY_XYZ_WORDS and 44 untouched 0xeeeeeeee words.
v_cmpx_eq_u32_e32 0, v0
; inputs: RGBA32F (1, 2, 3, 4) at 0x200, RGBA16F (1, 2, 3, 4) at 0x210
v_mov_b32_e32 v1, 1.0
v_mov_b32_e32 v2, 2.0
v_mov_b32_e32 v3, 0x40400000
v_mov_b32_e32 v4, 4.0
v_mov_b32_e32 v5, 0x200
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x40003c00
v_mov_b32_e32 v2, 0x44004200
v_mov_b32_e32 v5, 0x210
buffer_store_dwordx2 v[1:2], v5, s[0:3], 0 offen
; the store regions 0x240..0x2cf hold 0xa5a50000 | their address
v_mov_b32_e32 v1, 0xa5a50240
v_mov_b32_e32 v2, 0xa5a50244
v_mov_b32_e32 v3, 0xa5a50248
v_mov_b32_e32 v4, 0xa5a5024c
v_mov_b32_e32 v5, 0x240
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50250
v_mov_b32_e32 v2, 0xa5a50254
v_mov_b32_e32 v3, 0xa5a50258
v_mov_b32_e32 v4, 0xa5a5025c
v_mov_b32_e32 v5, 0x250
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50260
v_mov_b32_e32 v2, 0xa5a50264
v_mov_b32_e32 v3, 0xa5a50268
v_mov_b32_e32 v4, 0xa5a5026c
v_mov_b32_e32 v5, 0x260
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50270
v_mov_b32_e32 v2, 0xa5a50274
v_mov_b32_e32 v3, 0xa5a50278
v_mov_b32_e32 v4, 0xa5a5027c
v_mov_b32_e32 v5, 0x270
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50280
v_mov_b32_e32 v2, 0xa5a50284
v_mov_b32_e32 v3, 0xa5a50288
v_mov_b32_e32 v4, 0xa5a5028c
v_mov_b32_e32 v5, 0x280
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50290
v_mov_b32_e32 v2, 0xa5a50294
v_mov_b32_e32 v3, 0xa5a50298
v_mov_b32_e32 v4, 0xa5a5029c
v_mov_b32_e32 v5, 0x290
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a502a0
v_mov_b32_e32 v2, 0xa5a502a4
v_mov_b32_e32 v3, 0xa5a502a8
v_mov_b32_e32 v4, 0xa5a502ac
v_mov_b32_e32 v5, 0x2a0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a502b0
v_mov_b32_e32 v2, 0xa5a502b4
v_mov_b32_e32 v3, 0xa5a502b8
v_mov_b32_e32 v4, 0xa5a502bc
v_mov_b32_e32 v5, 0x2b0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a502c0
v_mov_b32_e32 v2, 0xa5a502c4
v_mov_b32_e32 v3, 0xa5a502c8
v_mov_b32_e32 v4, 0xa5a502cc
v_mov_b32_e32 v5, 0x2c0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
; out[0..3]: buffer_load_format_xy, 32_32_32_32_FLOAT
v_mov_b32_e32 v10, 0x5a5a0000
v_mov_b32_e32 v11, 0x5a5a0004
v_mov_b32_e32 v12, 0x5a5a0008
v_mov_b32_e32 v13, 0x5a5a000c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xy v[10:11], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[4..7]: buffer_load_format_xyz, 32_32_32_32_FLOAT
v_mov_b32_e32 v10, 0x5a5a0010
v_mov_b32_e32 v11, 0x5a5a0014
v_mov_b32_e32 v12, 0x5a5a0018
v_mov_b32_e32 v13, 0x5a5a001c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xyz v[10:12], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x10
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[8..11]: buffer_load_format_xy, 16_16_16_16_FLOAT
v_mov_b32_e32 v10, 0x5a5a0020
v_mov_b32_e32 v11, 0x5a5a0024
v_mov_b32_e32 v12, 0x5a5a0028
v_mov_b32_e32 v13, 0x5a5a002c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31047fac
v_mov_b32_e32 v5, 0x210
buffer_load_format_xy v[10:11], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x20
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[12..15]: buffer_load_format_xyz, 16_16_16_16_FLOAT
v_mov_b32_e32 v10, 0x5a5a0030
v_mov_b32_e32 v11, 0x5a5a0034
v_mov_b32_e32 v12, 0x5a5a0038
v_mov_b32_e32 v13, 0x5a5a003c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31047fac
v_mov_b32_e32 v5, 0x210
buffer_load_format_xyz v[10:12], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x30
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[16..19]: buffer_load_format_xyz, 32_32_FLOAT (z repeats x)
v_mov_b32_e32 v10, 0x5a5a0040
v_mov_b32_e32 v11, 0x5a5a0044
v_mov_b32_e32 v12, 0x5a5a0048
v_mov_b32_e32 v13, 0x5a5a004c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31040fac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xyz v[10:12], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x40
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[20..23]: buffer_load_format_xy, 32_FLOAT (y repeats x)
v_mov_b32_e32 v10, 0x5a5a0050
v_mov_b32_e32 v11, 0x5a5a0054
v_mov_b32_e32 v12, 0x5a5a0058
v_mov_b32_e32 v13, 0x5a5a005c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xy v[10:11], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x50
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[24..27]: tbuffer_load_format_xy, instruction 32_32_32_32_FLOAT over a V# that says 32_FLOAT
v_mov_b32_e32 v10, 0x5a5a0060
v_mov_b32_e32 v11, 0x5a5a0064
v_mov_b32_e32 v12, 0x5a5a0068
v_mov_b32_e32 v13, 0x5a5a006c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x200
tbuffer_load_format_xy v[10:11], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x60
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[28..31]: tbuffer_load_format_xyz, instruction 16_16_16_16_FLOAT over a V# that says 32_FLOAT
v_mov_b32_e32 v10, 0x5a5a0070
v_mov_b32_e32 v11, 0x5a5a0074
v_mov_b32_e32 v12, 0x5a5a0078
v_mov_b32_e32 v13, 0x5a5a007c
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x210
tbuffer_load_format_xyz v[10:12], v5, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x70
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[32..35]: buffer_load_format_xy, 32_32_32_32_FLOAT, NR 0x208: x,y end at 0x208, the element at 0x210
v_mov_b32_e32 v10, 0x5a5a0080
v_mov_b32_e32 v11, 0x5a5a0084
v_mov_b32_e32 v12, 0x5a5a0088
v_mov_b32_e32 v13, 0x5a5a008c
s_mov_b32 s6, 0x208
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xy v[10:11], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x80
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[36..39]: buffer_load_format_xyz, 32_32_32_32_FLOAT, NR 0x20c: x,y,z end at 0x20c, the element at 0x210
v_mov_b32_e32 v10, 0x5a5a0090
v_mov_b32_e32 v11, 0x5a5a0094
v_mov_b32_e32 v12, 0x5a5a0098
v_mov_b32_e32 v13, 0x5a5a009c
s_mov_b32 s6, 0x20c
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xyz v[10:12], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x90
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[40..43]: buffer_load_format_xy, 32_32_32_32_FLOAT, NR 0x210: the whole element in range
v_mov_b32_e32 v10, 0x5a5a00a0
v_mov_b32_e32 v11, 0x5a5a00a4
v_mov_b32_e32 v12, 0x5a5a00a8
v_mov_b32_e32 v13, 0x5a5a00ac
s_mov_b32 s6, 0x210
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x200
buffer_load_format_xy v[10:11], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xa0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[44..47]: tbuffer_load_format_xyz, 16_16_16_16_FLOAT, NR 0x216: x,y,z end at 0x216, the element at 0x218
v_mov_b32_e32 v10, 0x5a5a00b0
v_mov_b32_e32 v11, 0x5a5a00b4
v_mov_b32_e32 v12, 0x5a5a00b8
v_mov_b32_e32 v13, 0x5a5a00bc
s_mov_b32 s6, 0x216
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x210
tbuffer_load_format_xyz v[10:12], v5, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xb0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; 0x240: buffer_store_format_xy, 32_32_32_32_FLOAT
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x240
buffer_store_format_xy v[20:21], v5, s[4:7], 0 offen
; 0x250: buffer_store_format_xyz, 32_32_32_32_FLOAT
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x250
buffer_store_format_xyz v[20:22], v5, s[4:7], 0 offen
; 0x260: buffer_store_format_xy, 16_16_16_16_FLOAT
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31047fac
v_mov_b32_e32 v5, 0x260
buffer_store_format_xy v[20:21], v5, s[4:7], 0 offen
; 0x270: buffer_store_format_xyz, 16_16_16_16_FLOAT
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31047fac
v_mov_b32_e32 v5, 0x270
buffer_store_format_xyz v[20:22], v5, s[4:7], 0 offen
; 0x280: buffer_store_format_xy, 32_FLOAT (one channel)
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x280
buffer_store_format_xy v[20:21], v5, s[4:7], 0 offen
; 0x290: buffer_store_format_xy, 8_8_8_8_UNORM, y = 2.0 clamps
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31038fac
v_mov_b32_e32 v5, 0x290
buffer_store_format_xy v[20:21], v5, s[4:7], 0 offen
; 0x2a0: buffer_store_format_xyz, 8_8_8_8_UINT; y = 258 saturates to 255
v_mov_b32_e32 v20, 0x1
v_mov_b32_e32 v21, 0x102
v_mov_b32_e32 v22, 0x3
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3103cfac
v_mov_b32_e32 v5, 0x2a0
buffer_store_format_xyz v[20:22], v5, s[4:7], 0 offen
; 0x2b0: buffer_store_format_xy, 32_32_32_32_FLOAT, NR 0x2b8: x,y end at 0x2b8, the element at 0x2c0
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0x2b8
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x2b0
buffer_store_format_xy v[20:21], v5, s[4:7], 0 offen
; 0x2c0: buffer_store_format_xyz, 32_32_32_32_FLOAT, NR 0x2cc: x,y,z end at 0x2cc, the element at 0x2d0
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
s_mov_b32 s6, 0x2cc
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x2c0
buffer_store_format_xyz v[20:22], v5, s[4:7], 0 offen
s_waitcnt_vscnt null, 0x0
; out[48..51]: the 16 bytes at 0x240
v_mov_b32_e32 v5, 0x240
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xc0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[52..55]: the 16 bytes at 0x250
v_mov_b32_e32 v5, 0x250
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xd0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[56..59]: the 16 bytes at 0x260
v_mov_b32_e32 v5, 0x260
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xe0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[60..63]: the 16 bytes at 0x270
v_mov_b32_e32 v5, 0x270
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xf0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[64..67]: the 16 bytes at 0x280
v_mov_b32_e32 v5, 0x280
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x100
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[68..71]: the 16 bytes at 0x290
v_mov_b32_e32 v5, 0x290
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x110
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[72..75]: the 16 bytes at 0x2a0
v_mov_b32_e32 v5, 0x2a0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x120
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[76..79]: the 16 bytes at 0x2b0
v_mov_b32_e32 v5, 0x2b0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x130
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[80..83]: the 16 bytes at 0x2c0
v_mov_b32_e32 v5, 0x2c0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x140
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
s_endpgm
