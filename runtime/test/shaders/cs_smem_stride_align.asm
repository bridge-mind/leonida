; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_buffer_load with a stride (limit = NUM_RECORDS * stride bytes) and SMEM address truncation: base, soffset and
; imm are each truncated to a multiple of 4 before they are added. Rules measured on an AMD Radeon RX 6800 (gfx1030)
; as a raw compute kernel (docs/design.md, Scalar/buffer memory).
; Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X, then:
;   s_buffer_load_dword s24, stride 4, NR 5, 0x10   -> limit 20, 0x14 <= 20: 0x5eed0010 (NR as bytes would give 0)
;   s_buffer_load_dword s25, stride 4, NR 5, 0x14   -> 0x18 > 20: 0
;   s_buffer_load_dword s26, stride 1, NR 17, 0x10  -> limit 17, 0x14 > 17: 0 (index 16 < 17 would read)
;   s_buffer_load_dword s27, base 0x41, 0x3         -> (0x41 & ~3) + 0: 0x5eed0000 (unaligned sum 0x44: 0x5eed0004)
;   s_load_dword        s28, ptr 0x41, 0x1          -> 0x40: 0x5eed0000
;   s_buffer_load_dword s29, base 0x41, s20=3 offset:0x3 -> 0 + 0: 0x5eed0000 (sum 6 truncated would be 4)
;   s_load_dword        s30, ptr 0x40, s20=3 offset:0x3  -> 0x5eed0000
;   s_buffer_load_dword s31, base 0x41, s21=2 offset:0x6 -> 0 + 4: 0x5eed0004 (sum 8 would be 8)
; (V#s with base 0x41 have NR 0x100, word3 0x31016fac = OOB_SELECT 3 everywhere) and writes s24..s31 to out[0..7].
; Run: vk_compute_test cs_smem_stride_align.spv 1 dump -> out[0..7] = 0x5eed0010 0x00000000 0x00000000 0x5eed0000
; 0x5eed0000 0x5eed0000 0x5eed0000 0x5eed0004 (1592590352 0 0 1592590336 1592590336 1592590336 1592590336
; 1592590340).
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
s_mov_b32 s5, 0x40000
s_mov_b32 s6, 5
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0x10000
s_mov_b32 s10, 17
s_mov_b32 s11, 0x31016fac
s_mov_b32 s12, 0x41
s_mov_b32 s13, 0
s_mov_b32 s14, 0x100
s_mov_b32 s15, 0x31016fac
s_mov_b32 s16, 0x41
s_mov_b32 s17, 0
s_mov_b32 s18, 0x40
s_mov_b32 s19, 0
s_mov_b32 s20, 3
s_mov_b32 s21, 2
s_buffer_load_dword s24, s[4:7], 0x10
s_buffer_load_dword s25, s[4:7], 0x14
s_buffer_load_dword s26, s[8:11], 0x10
s_buffer_load_dword s27, s[12:15], 0x3
s_load_dword s28, s[16:17], 0x1
s_buffer_load_dword s29, s[12:15], s20 offset:0x3
s_load_dword s30, s[18:19], s20 offset:0x3
s_buffer_load_dword s31, s[12:15], s21 offset:0x6
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
