; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; tbuffer_store_format_* and its d16 forms. Every expected word was measured by running this body natively on a Radeon
; 680M (gfx1035, RDNA2) through tools/hw_oracle's kernel wrapper, with s[0:3] set to the V# vk_compute_test passes;
; wave32 and wave64 gave the same 52 words. One thread runs.
; - The instruction's format replaces the V#'s, for the encoding, the element size and the bounds check alike. No
;   format modifier means GFX10's default, BUF_FMT_8_UNORM: one byte, its neighbours kept.
; - A store writes its format's whole element: components the format has but the instruction doesn't give are 0
;   (xy into 16_16_16_16 leaves z = w = 0); components the format doesn't have are dropped (xyzw into 32_32 writes
;   x, y; into 32 only x).
; - d16 data is two 16-bit values per VGPR, low half first.
; - Bounds (OOB_SELECT 3): the instruction's element must end inside num_records. With NR 0x298, an xy store of a
;   16-byte element at 0x290 writes nothing, though a V# that says 32_FLOAT has a 4-byte element; with NR 0x2a1, an
;   8_UNORM byte at 0x2a0 is written, though the V#'s 16-byte element would not fit.
; - A V# whose FORMAT is 0 (INVALID) disables the store, though the instruction brings its own format.
; Store regions 0x200..0x2cf start as 0xa5a50000 | address; out[0..51] are the 13 regions read back, in source order.
; Run: VKC_OUT_WORDS=192 VKC_DUMP=<file> vk_compute_test cs_tbuffer_store.spv 1 dump; check.py compares words 0..127 of
; the dump with TBUFFER_STORE_WORDS and 76 untouched 0xeeeeeeee words.
v_cmpx_eq_u32_e32 0, v0
; the store regions 0x200..0x2cf hold 0xa5a50000 | their address
v_mov_b32_e32 v1, 0xa5a50200
v_mov_b32_e32 v2, 0xa5a50204
v_mov_b32_e32 v3, 0xa5a50208
v_mov_b32_e32 v4, 0xa5a5020c
v_mov_b32_e32 v5, 0x200
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50210
v_mov_b32_e32 v2, 0xa5a50214
v_mov_b32_e32 v3, 0xa5a50218
v_mov_b32_e32 v4, 0xa5a5021c
v_mov_b32_e32 v5, 0x210
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50220
v_mov_b32_e32 v2, 0xa5a50224
v_mov_b32_e32 v3, 0xa5a50228
v_mov_b32_e32 v4, 0xa5a5022c
v_mov_b32_e32 v5, 0x220
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0xa5a50230
v_mov_b32_e32 v2, 0xa5a50234
v_mov_b32_e32 v3, 0xa5a50238
v_mov_b32_e32 v4, 0xa5a5023c
v_mov_b32_e32 v5, 0x230
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
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
; 0x200: tbuffer_store_format_xyzw, xyzw, 32_32_32_32_FLOAT over a V# that says 32_FLOAT: all four dwords
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
v_mov_b32_e32 v23, 0x40800000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x200
tbuffer_store_format_xyzw v[20:23], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen
; 0x210: tbuffer_store_format_xy, xy, 16_16_16_16_FLOAT over a V# that says 32_32_32_32_FLOAT: the 8-byte element, z = w = 0
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x210
tbuffer_store_format_xy v[20:21], v5, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen
; 0x220: tbuffer_store_format_xyz, xyz, 8_8_8_8_UINT over a V# that says 32_FLOAT
v_mov_b32_e32 v20, 0x1
v_mov_b32_e32 v21, 0x2
v_mov_b32_e32 v22, 0x3
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x220
tbuffer_store_format_xyz v[20:22], v5, s[4:7], 0 format:[BUF_FMT_8_8_8_8_UINT] offen
; 0x230: tbuffer_store_format_x, x, no format modifier (GFX10 default 8_UNORM) at byte 1, over a V# that says 32_32_32_32_FLOAT
v_mov_b32_e32 v20, 0x3f800000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x231
tbuffer_store_format_x v20, v5, s[4:7], 0 offen
; 0x240: tbuffer_store_format_x, x, 16_FLOAT at byte 2 (the high half)
v_mov_b32_e32 v20, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x242
tbuffer_store_format_x v20, v5, s[4:7], 0 format:[BUF_FMT_16_FLOAT] offen
; 0x250: tbuffer_store_format_xyzw, xyzw, 2_10_10_10_UNORM (1.0, 0.0, 0.25, 1.0)
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x0
v_mov_b32_e32 v22, 0x3e800000
v_mov_b32_e32 v23, 0x3f800000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x250
tbuffer_store_format_xyzw v[20:23], v5, s[4:7], 0 format:[BUF_FMT_2_10_10_10_UNORM] offen
; 0x260: tbuffer_store_format_xyzw, xyzw into 32_32_FLOAT: only x, y
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
v_mov_b32_e32 v23, 0x40800000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x260
tbuffer_store_format_xyzw v[20:23], v5, s[4:7], 0 format:[BUF_FMT_32_32_FLOAT] offen
; 0x270: tbuffer_store_format_d16_xy, d16_xy, 16_16_FLOAT (packed 1.0, 2.0)
v_mov_b32_e32 v20, 0x40003c00
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x270
tbuffer_store_format_d16_xy v20, v5, s[4:7], 0 format:[BUF_FMT_16_16_FLOAT] offen
; 0x280: tbuffer_store_format_d16_xyzw, d16_xyzw, 8_8_8_8_UNORM (packed 1.0, 0.5, 0.0, 1.0)
v_mov_b32_e32 v20, 0x38003c00
v_mov_b32_e32 v21, 0x3c000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x280
tbuffer_store_format_d16_xyzw v[20:21], v5, s[4:7], 0 format:[BUF_FMT_8_8_8_8_UNORM] offen
; 0x290: tbuffer_store_format_xy, xy, 32_32_32_32_FLOAT, NR 0x298 over a V# that says 32_FLOAT: x, y end at 0x298, the element at 0x2a0
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0x298
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x290
tbuffer_store_format_xy v[20:21], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen
; 0x2a0: tbuffer_store_format_x, x, 8_UNORM, NR 0x2a1 over a V# that says 32_32_32_32_FLOAT: the byte fits, the V#'s element would not
v_mov_b32_e32 v20, 0x3f800000
s_mov_b32 s6, 0x2a1
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x2a0
tbuffer_store_format_x v20, v5, s[4:7], 0 format:[BUF_FMT_8_UNORM] offen
; 0x2b0: tbuffer_store_format_xyzw, xyzw into 32_FLOAT (one channel) over a V# that says 32_32_32_32_FLOAT: only x
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
v_mov_b32_e32 v22, 0x40400000
v_mov_b32_e32 v23, 0x40800000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x3104dfac
v_mov_b32_e32 v5, 0x2b0
tbuffer_store_format_xyzw v[20:23], v5, s[4:7], 0 format:[BUF_FMT_32_FLOAT] offen
; 0x2c0: tbuffer_store_format_xy, xy, 32_32_FLOAT over a V# whose FORMAT is 0 (INVALID): nothing
v_mov_b32_e32 v20, 0x3f800000
v_mov_b32_e32 v21, 0x40000000
s_mov_b32 s6, 0xffffffff
s_mov_b32 s7, 0x31000fac
v_mov_b32_e32 v5, 0x2c0
tbuffer_store_format_xy v[20:21], v5, s[4:7], 0 format:[BUF_FMT_32_32_FLOAT] offen
s_waitcnt_vscnt null, 0x0
; out[0..3]: the 16 bytes at 0x200
v_mov_b32_e32 v5, 0x200
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[4..7]: the 16 bytes at 0x210
v_mov_b32_e32 v5, 0x210
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x10
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[8..11]: the 16 bytes at 0x220
v_mov_b32_e32 v5, 0x220
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x20
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[12..15]: the 16 bytes at 0x230
v_mov_b32_e32 v5, 0x230
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x30
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[16..19]: the 16 bytes at 0x240
v_mov_b32_e32 v5, 0x240
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x40
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[20..23]: the 16 bytes at 0x250
v_mov_b32_e32 v5, 0x250
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x50
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[24..27]: the 16 bytes at 0x260
v_mov_b32_e32 v5, 0x260
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x60
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[28..31]: the 16 bytes at 0x270
v_mov_b32_e32 v5, 0x270
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x70
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[32..35]: the 16 bytes at 0x280
v_mov_b32_e32 v5, 0x280
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x80
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[36..39]: the 16 bytes at 0x290
v_mov_b32_e32 v5, 0x290
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0x90
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[40..43]: the 16 bytes at 0x2a0
v_mov_b32_e32 v5, 0x2a0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xa0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[44..47]: the 16 bytes at 0x2b0
v_mov_b32_e32 v5, 0x2b0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xb0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
; out[48..51]: the 16 bytes at 0x2c0
v_mov_b32_e32 v5, 0x2c0
buffer_load_dwordx4 v[10:13], v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0xc0
buffer_store_dwordx4 v[10:13], v6, s[0:3], 0 offen
s_endpgm
