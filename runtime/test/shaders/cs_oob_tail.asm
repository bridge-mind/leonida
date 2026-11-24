; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The end of an access decides, not its start (docs/design.md, Buffer bounds checking; measured on a Radeon 880M
; (gfx1150), a valid proxy for RDNA2: gfx1036 follows the same column, docs/buffer_bounds.md). Mesa's GFX10 column, which
; compares only the start, gives different words for out[0], out[1] and out[6]. Thread 0 seeds heap bytes
; 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X; every V# has base 0x40, the heap is 0xee-filled.
;   0 mode 3 (RAW), NR 10, b32 at off 8:              ends at 12 > 10                  -> 0 (a start check reads X 8)
;   1 mode 0, stride 10, NR 4, idx 0, b32 at off 8:   8 + 4 > 10                       -> 0 (a start check reads X 8)
;   2 mode 3, NR 20, x4 load at off 8, dword 2:        ends at 20 <= 20                -> X 0x10: 0x5eed0010
;   3 same load, dword 3:                              ends at 24 > 20                 -> 0 (per dword, not per load)
;   4 mode 3, NR 32, x2 store at off 0x1c, read back X 0x1c (through a V# with NR 256): first dword ends at 32    -> 0xbeef0000
;   5 same store, read X 0x20 (its second dword):      ends at 36 > 32, dropped        -> 0xeeeeeeee
;   6 mode 3, NR 32, buffer_load_format_xyzw 32_32_32_32_UINT at off 0x14, x: element ends at 36 > 32 -> 0 (all or nothing)
;   7 same load at off 0x10, x:                        ends at 32 <= 32                -> X 0x10: 0x5eed0010
; Run: vk_compute_test cs_oob_tail.spv 1 dump -> out[0..7] = 0x00000000 0x00000000 0x5eed0010 0x00000000 0xbeef0000
; 0xeeeeeeee 0x00000000 0x5eed0010 (0 0 1592590352 0 3203334144 4008636142 0 1592590352).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0000
v_mov_b32 v2, 0x5eed0004
v_mov_b32 v3, 0x5eed0008
v_mov_b32 v4, 0x5eed000c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
v_mov_b32 v1, 0x5eed0010
v_mov_b32 v2, 0x5eed0014
v_mov_b32 v3, 0x5eed0018
v_mov_b32 v4, 0x5eed001c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:80
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 10
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0xa0000
s_mov_b32 s10, 4
s_mov_b32 s11, 0x01016fac
s_mov_b32 s12, 0x40
s_mov_b32 s13, 0
s_mov_b32 s14, 20
s_mov_b32 s15, 0x31016fac
s_mov_b32 s16, 0x40
s_mov_b32 s17, 0
s_mov_b32 s18, 32
s_mov_b32 s19, 0x31016fac
s_mov_b32 s28, 0x40
s_mov_b32 s29, 0
s_mov_b32 s30, 0x100
s_mov_b32 s31, 0x31016fac
s_mov_b32 s24, 0x40
s_mov_b32 s25, 0
s_mov_b32 s26, 32
s_mov_b32 s27, 0x3104bfac
v_mov_b32 v1, 8
v_mov_b32 v5, 0x1c
v_mov_b32 v6, 0x20
v_mov_b32 v7, 0x14
v_mov_b32 v8, 0x10
v_mov_b32 v24, 0xbeef0000
v_mov_b32 v25, 0xbeef0001
buffer_load_dword v10, v1, s[4:7], 0 offen
buffer_load_dword v11, v1, s[8:11], 0 offen
buffer_load_dwordx4 v[20:23], v1, s[12:15], 0 offen
buffer_store_dwordx2 v[24:25], v5, s[16:19], 0 offen
s_waitcnt vmcnt(0)
buffer_load_dword v26, v5, s[28:31], 0 offen
buffer_load_dword v27, v6, s[28:31], 0 offen
buffer_load_format_xyzw v[28:31], v7, s[24:27], 0 offen
buffer_load_format_xyzw v[32:35], v8, s[24:27], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v10, off, s[0:3], 0
buffer_store_dword v11, off, s[0:3], 0 offset:4
buffer_store_dword v22, off, s[0:3], 0 offset:8
buffer_store_dword v23, off, s[0:3], 0 offset:12
buffer_store_dword v26, off, s[0:3], 0 offset:16
buffer_store_dword v27, off, s[0:3], 0 offset:20
buffer_store_dword v28, off, s[0:3], 0 offset:24
buffer_store_dword v32, off, s[0:3], 0 offset:28
s_endpgm
