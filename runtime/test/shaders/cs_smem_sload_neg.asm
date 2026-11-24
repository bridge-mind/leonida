; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_load with a soffset and a negative immediate: the immediate is sign-extended (21 bits) and truncated to a
; multiple of 4 on its own, soffset likewise, and both are added to the 4-truncated 64-bit pointer. Measured on an
; AMD Radeon RX 6800 (gfx1030) as a raw compute kernel, only with soffset + imm >= 0 (docs/design.md, Scalar/buffer
; memory); every row here keeps that sum non-negative.
; Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X, then (ptr 0x40 in
; s[4:5], ptr 0x43 in s[6:7]):
;   s_load_dword   s24,      s[4:5], s20=8 offset:-0x4    -> 8 - 4: 0x5eed0004 (zero-extended imm: +4 GiB)
;   s_load_dword   s25,      s[4:5], s20=8 offset:-0x8    -> 0: 0x5eed0000
;   s_load_dword   s26,      s[4:5], s21=0xb offset:-0x1  -> 8 + -4: 0x5eed0004 (truncating the sum 0xa: +8)
;   s_load_dword   s27,      s[6:7], s21=0xb offset:-0x1  -> 0x40 + 4: 0x5eed0004
;   s_load_dwordx2 s[28:29], s[4:5], s22=0x10 offset:-0x8 -> 8, 0xc: 0x5eed0008 0x5eed000c
; and writes s24..s29 to out[0..5]; out[6..7] keep the 0xee fill.
; With a translator that zero-extends the immediate the first four reads go 4 GiB past the heap: do not run this
; test on a GPU with such a translator (it can fault the device).
; Run: vk_compute_test cs_smem_sload_neg.spv 1 dump -> out[0..7] = 0x5eed0004 0x5eed0000 0x5eed0004 0x5eed0004
; 0x5eed0008 0x5eed000c 0xeeeeeeee 0xeeeeeeee (1592590340 1592590336 1592590340 1592590340 1592590344 1592590348
; 4008636142 4008636142).
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
s_mov_b32 s6, 0x43
s_mov_b32 s7, 0
s_mov_b32 s20, 8
s_mov_b32 s21, 0xb
s_mov_b32 s22, 0x10
s_load_dword s24, s[4:5], s20 offset:-0x4
s_load_dword s25, s[4:5], s20 offset:-0x8
s_load_dword s26, s[4:5], s21 offset:-0x1
s_load_dword s27, s[6:7], s21 offset:-0x1
s_load_dwordx2 s[28:29], s[4:5], s22 offset:-0x8
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, s24
v_mov_b32 v2, s25
v_mov_b32 v3, s26
v_mov_b32 v4, s27
buffer_store_dwordx4 v[1:4], off, s[0:3], 0
v_mov_b32 v1, s28
v_mov_b32 v2, s29
buffer_store_dwordx2 v[1:2], off, s[0:3], 0 offset:16
s_endpgm
