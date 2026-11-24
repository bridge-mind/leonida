; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; V# OOB_SELECT (word3 bits 29:28) picks the MUBUF bounds check (docs/design.md, Buffer bounds checking). RDNA2 rules
; (measured on gfx1036, docs/buffer_bounds.md; Mesa's GFX11 column): idx = vindex (0 without idxen), off =
; inst_offset + vaddr offset, offs = off + soffset, a dword access:
;   0 STRUCTURED_WITH_OFFSET: out of range if idx >= NUM_RECORDS || off + 4 > STRIDE
;   1 STRUCTURED:             out of range if idx >= NUM_RECORDS
;   2 DISABLED:               out of range if NUM_RECORDS == 0
;   3 RAW:                    out of range if offs + 4 > NUM_RECORDS (idx * stride is in the address, not in the check)
; Out of range reads 0. Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X;
; every V# has base 0x40. The heap is 0xee-filled, so an access that is in range but beyond the seed reads 0xeeeeeeee.
; Literal V#s (s_mov_b32 of word3) fold the mode at translation time; the three with an s_or_b32 of word3 are run-time
; V#s and select it with OpSelect. One thread, no wave64 requirement. Words, in out[0..7]:
;   0 mode 0 literal, stride 8, NR 4, idx 1 + off 4:   idx < 4, 4 + 4 <= 8 -> X = 8 + 4 = 0xc: 0x5eed000c
;   1 mode 0 literal, idx 1 + off 8:                   8 + 4 > 8         -> 0   (mode 1 or the stride check alone reads X 0x10)
;   2 mode 1 literal, stride 8, NR 4, idx 1 + off 12:  idx < 4 only       -> X = 8 + 12 = 0x14: 0x5eed0014
;   3 mode 1 literal, idx 4 + off 0:                   4 >= 4             -> 0
;   4 mode 2 run-time, stride 0, NR 4, off 0x18:       NR != 0            -> X 0x18: 0x5eed0018 (RAW would give 0)
;   5 mode 3 run-time, stride 8, NR 0x10, idx 7 + off 4: offs 4 < 0x10    -> X = 56 + 4 = 0x3c, beyond the seed: 0xeeeeeeee
;                                                       (a check of idx against NUM_RECORDS would give 0)
;   6 mode 3 literal, stride 8, NR 0x10, off 0x10 (offen only): 0x10 + 4 > 0x10 -> 0   (mode 1 reads X 0x10)
;   7 mode 0 run-time, stride 8, NR 4, off 0x10 (offen only):   0x10 + 4 > 8 -> 0
; The expected words are derived from those rules; the probes avoid an access that straddles an edge, so a start-based
; check predicts the same words; cs_oob_tail.asm is the one that tells the two apart.
; Run: vk_compute_test cs_oob_select.spv 1 dump -> out[0..7] = 0x5eed000c 0x00000000 0x5eed0014 0x00000000
; 0x5eed0018 0xeeeeeeee 0x00000000 0x00000000 (1592590348 0 1592590356 0 1592590360 4008636142 0 0).
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
s_mov_b32 s5, 0x80000
s_mov_b32 s6, 4
s_mov_b32 s7, 0x01016fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0x80000
s_mov_b32 s10, 4
s_mov_b32 s11, 0x11016fac
s_mov_b32 s12, 0x40
s_mov_b32 s13, 0
s_mov_b32 s14, 4
s_mov_b32 s15, 0x01016fac
s_mov_b32 s20, 0x20000000
s_or_b32 s15, s15, s20
s_mov_b32 s16, 0x40
s_mov_b32 s17, 0x80000
s_mov_b32 s18, 0x10
s_mov_b32 s19, 0x01016fac
s_mov_b32 s21, 0x30000000
s_or_b32 s19, s19, s21
s_mov_b32 s24, 0x40
s_mov_b32 s25, 0x80000
s_mov_b32 s26, 0x10
s_mov_b32 s27, 0x31016fac
s_mov_b32 s28, 0x40
s_mov_b32 s29, 0x80000
s_mov_b32 s30, 4
s_mov_b32 s31, 0x01016fac
s_mov_b32 s22, 0
s_or_b32 s31, s31, s22
v_mov_b32 v20, 1
v_mov_b32 v21, 4
v_mov_b32 v22, 1
v_mov_b32 v23, 8
v_mov_b32 v24, 1
v_mov_b32 v25, 12
v_mov_b32 v26, 4
v_mov_b32 v27, 0
v_mov_b32 v28, 0x18
v_mov_b32 v29, 7
v_mov_b32 v30, 4
v_mov_b32 v31, 0x10
buffer_load_dword v10, v[20:21], s[4:7], 0 idxen offen
buffer_load_dword v11, v[22:23], s[4:7], 0 idxen offen
buffer_load_dword v12, v[24:25], s[8:11], 0 idxen offen
buffer_load_dword v13, v[26:27], s[8:11], 0 idxen offen
buffer_load_dword v14, v28, s[12:15], 0 offen
buffer_load_dword v15, v[29:30], s[16:19], 0 idxen offen
buffer_load_dword v16, v31, s[24:27], 0 offen
buffer_load_dword v17, v31, s[28:31], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16
s_endpgm
