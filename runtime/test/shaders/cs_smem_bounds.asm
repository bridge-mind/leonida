; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_buffer_load NUM_RECORDS bound, per dword, on the 4-truncated offset. Rule measured on an AMD Radeon RX 6800
; (gfx1030) as a raw compute kernel (docs/design.md, Scalar/buffer memory): off = (soffset & ~3) + (imm & ~3);
; dword i at o = off + 4*i is 0 when o + 4 > limit (limit = NUM_RECORDS bytes for stride 0), else the dword at
; (base & ~3) + o.
; Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X, then loads through
; V#s it builds itself (base 0x40, stride 0, word3 0x31016fac = OOB_SELECT 3 as measured) with four NUM_RECORDS:
;   s_buffer_load_dword   s24,      NR 0x12, 0x10            -> 0x10 + 4 > 0x12: 0           (no bound: 0x5eed0010)
;   s_buffer_load_dword   s25,      NR 0x15, 0x13            -> off 0x10, 0x14 <= 0x15: 0x5eed0010
;   s_buffer_load_dword   s26,      NR 0x10, s20=0xe offset:0x2 -> off 0xc + 0, 0x10 <= 0x10: 0x5eed000c
;                                                              (truncating the sum 0x10 would give 0)
;   s_buffer_load_dword   s27,      NR 0x14, s21=0x10 offset:0x4 -> off 0x14, 0x18 > 0x14: 0
;   s_buffer_load_dwordx4 s[28:31], NR 0x15, 0xc             -> 0xc, 0x10 in; 0x14, 0x18 out: data, data, 0, 0
; and writes s24..s31 to out[0..7].
; Run: vk_compute_test cs_smem_bounds.spv 1 dump -> out[0..7] = 0x00000000 0x5eed0010 0x5eed000c 0x00000000
; 0x5eed000c 0x5eed0010 0x00000000 0x00000000 (0 1592590352 1592590348 0 1592590348 1592590352 0 0).
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
s_mov_b32 s6, 0x12
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0
s_mov_b32 s10, 0x15
s_mov_b32 s11, 0x31016fac
s_mov_b32 s12, 0x40
s_mov_b32 s13, 0
s_mov_b32 s14, 0x10
s_mov_b32 s15, 0x31016fac
s_mov_b32 s16, 0x40
s_mov_b32 s17, 0
s_mov_b32 s18, 0x14
s_mov_b32 s19, 0x31016fac
s_mov_b32 s20, 0xe
s_mov_b32 s21, 0x10
s_buffer_load_dword s24, s[4:7], 0x10
s_buffer_load_dword s25, s[8:11], 0x13
s_buffer_load_dword s26, s[12:15], s20 offset:0x2
s_buffer_load_dword s27, s[16:19], s21 offset:0x4
s_buffer_load_dwordx4 s[28:31], s[8:11], 0xc
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, s24
v_mov_b32 v2, s25
v_mov_b32 v3, s26
v_mov_b32 v4, s27
buffer_store_dwordx4 v[1:4], off, s[0:3], 0
v_mov_b32 v1, s28
v_mov_b32 v2, s29
v_mov_b32 v3, s30
v_mov_b32 v4, s31
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:16
s_endpgm
