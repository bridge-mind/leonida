; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Raw MUBUF loads (stride 0) check every dword against NUM_RECORDS on its own: dword i of a load at byte offset
; O = inst_offset + soffset + vaddr is data when O + 4*i + 4 <= NUM_RECORDS, else 0. Measured on an AMD Radeon
; RX 6800 (gfx1030) as a raw compute kernel with V# word3 OOB_SELECT 3 (docs/design.md, Buffer bounds checking).
; Only dword-aligned offsets here: the unaligned rows of the measurement are in tests/semantic.
; Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X, then loads through
; V#s it builds itself (base 0x40, stride 0, word3 0x31016fac):
;   buffer_load_dword   v10,      v5=0x10 offen, NR 0x12   -> 0x14 > 0x12: 0      (start-only check: 0x5eed0010)
;   buffer_load_dword   v11,      v6=0xc offen,  NR 0x12   -> 0x10 <= 0x12: 0x5eed000c
;   buffer_load_dwordx4 v[12:15], v7=8 offen,    NR 0x12   -> 8, 0xc in; 0x10, 0x14 out: data, data, 0, 0
;   buffer_load_dwordx2 v[16:17], off, s20=4 offset:0xc, NR 0x17 -> O 0x10: 0x14 <= 0x17 data, 0x18 > 0x17: 0
; and writes v10..v17 to out[0..7].
; Run: vk_compute_test cs_buf_tail.spv 1 dump -> out[0..7] = 0x00000000 0x5eed000c 0x5eed0008 0x5eed000c
; 0x00000000 0x00000000 0x5eed0010 0x00000000 (0 1592590348 1592590344 1592590348 0 0 1592590352 0).
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
s_mov_b32 s10, 0x17
s_mov_b32 s11, 0x31016fac
s_mov_b32 s20, 4
v_mov_b32 v5, 0x10
v_mov_b32 v6, 0xc
v_mov_b32 v7, 8
buffer_load_dword v10, v5, s[4:7], 0 offen
buffer_load_dword v11, v6, s[4:7], 0 offen
buffer_load_dwordx4 v[12:15], v7, s[4:7], 0 offen
buffer_load_dwordx2 v[16:17], off, s[8:11], s20 offset:12
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16
s_endpgm
