; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Memory ops merged with CPU-only checks (#1633's global_atomic_*, #1643 buffer_atomic_*_x2, #1666 8/16-bit global
; stores and d16 loads, global_atomic_*_x2), run once on a GPU. As cs_global_neg_offset: 40 groups make a 0x2800-byte
; heap, only group 0 thread 0 runs, global addresses are heap offsets (s[20:21] = P = 0x2000, v30 = 0).
; Seeds (MUBUF stores through the V# in s[0:3], base 0):
;   0x2000 5   0x2004 9   0x2008 0x0000000F 0x200c 0x00F00000   0x2010 0x12345678 0x2014 0x9ABCDEF0
;   0x2018 0x00000080   0x2020 0x44332211 0x2024 0x88776655
; Operations and the words stored to out[0..7]:
;   out[0] global_atomic_add glc at P+0, DATA 7: the returned old value       = 5
;   out[1] the dword at P+0 after it                                          = 12
;   out[2] global_atomic_cmpswap glc at P+4 {DATA 0x77, CMP 9}: (old << 8) | the dword after = 0x977
;   out[3] global_atomic_or_x2 at P+8 with 0xFF000000_000000F0: high dword after = 0xFFF00000
;   out[4] buffer_atomic_xor_x2 glc at 0x2010 with 0x0000FFFF_FFFF0000: low dword after = 0xEDCB5678
;   out[5] global_store_short 0xBEEF at P+0x23 (crosses into the next dword): the dword at 0x2020 = 0xEF332211
;   out[6] the dword at 0x2024 after it                                       = 0x887766BE
;   out[7] global_load_sbyte_d16_hi of the byte 0x80 at P+0x18 into 0x00001234 = 0xFF801234
; Run: vk_compute_test cs_global_mem_ext.spv 40 dump -> out[0..7] = 0x5 0xc 0x977 0xfff00000 0xedcb5678
; 0xef332211 0x887766be 0xff801234. Same FSSM header as cs_global_neg_offset.
v_or_b32_e32 v1, s4, v0
v_cmpx_eq_u32_e32 0, v1
v_mov_b32 v1, 5
v_mov_b32 v2, 0x2000
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 9
v_mov_b32 v2, 0x2004
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 15
v_mov_b32 v2, 0x2008
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0xf00000
v_mov_b32 v2, 0x200c
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x12345678
v_mov_b32 v2, 0x2010
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x9abcdef0
v_mov_b32 v2, 0x2014
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x80
v_mov_b32 v2, 0x2018
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x44332211
v_mov_b32 v2, 0x2020
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x88776655
v_mov_b32 v2, 0x2024
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s20, 0x2000
s_mov_b32 s21, 0
v_mov_b32 v30, 0
v_mov_b32 v3, 7
global_atomic_add v10, v30, v3, s[20:21] glc
s_waitcnt vmcnt(0)
global_load_dword v11, v30, s[20:21]
v_mov_b32 v4, 0x77
v_mov_b32 v5, 9
global_atomic_cmpswap v12, v30, v[4:5], s[20:21] offset:4 glc
s_waitcnt vmcnt(0)
global_load_dword v13, v30, s[20:21] offset:4
v_mov_b32 v6, 0xf0
v_mov_b32 v7, 0xff000000
global_atomic_or_x2 v30, v[6:7], s[20:21] offset:8
v_mov_b32 v8, 0xffff0000
v_mov_b32 v9, 0xffff
v_mov_b32 v2, 0x2010
buffer_atomic_xor_x2 v[8:9], v2, s[0:3], 0 offen glc
v_mov_b32 v14, 0xbeef
global_store_short v30, v14, s[20:21] offset:35
v_mov_b32 v15, 0x1234
global_load_sbyte_d16_hi v15, v30, s[20:21] offset:24
s_waitcnt vmcnt(0)
global_load_dword v16, v30, s[20:21] offset:12
global_load_dword v17, v30, s[20:21] offset:16
global_load_dword v18, v30, s[20:21] offset:32
global_load_dword v19, v30, s[20:21] offset:36
s_waitcnt vmcnt(0)
v_lshl_or_b32 v12, v12, 8, v13
v_mov_b32 v20, 0
buffer_store_dword v10, v20, s[0:3], 0 offen
v_mov_b32 v20, 4
buffer_store_dword v11, v20, s[0:3], 0 offen
v_mov_b32 v20, 8
buffer_store_dword v12, v20, s[0:3], 0 offen
v_mov_b32 v20, 12
buffer_store_dword v16, v20, s[0:3], 0 offen
v_mov_b32 v20, 16
buffer_store_dword v17, v20, s[0:3], 0 offen
v_mov_b32 v20, 20
buffer_store_dword v18, v20, s[0:3], 0 offen
v_mov_b32 v20, 24
buffer_store_dword v19, v20, s[0:3], 0 offen
v_mov_b32 v20, 28
buffer_store_dword v15, v20, s[0:3], 0 offen
s_endpgm
