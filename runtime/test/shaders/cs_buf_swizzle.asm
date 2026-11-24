; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Swizzled MUBUF addressing and ADD_TID_ENABLE (docs/design.md, Swizzled buffer addressing). Measured natively on
; RDNA2 (gfx1036, wave32 & wave64): SWIZZLE_ENABLE = word1 bit 31, STRIDE = word1 [29:16], INDEX_STRIDE = word3 [22:21]
; (IS = 8 << n), ADD_TID_ENABLE = word3 bit 23:
;   index = idxen ? vindex : (ADD_TID_ENABLE ? lane : 0), offset = uint32((offen ? vaddr : 0) + inst_offset)
;   buffer_offset = (index / IS * stride + (offset & ~3)) * IS + (index % IS) * 4 + (offset & 3), address = base +
;   soffset + buffer_offset (IS = index_stride). With idxen=1, multi-dword accesses step linearly by +4 bytes; with
;   idxen=0 and ADD_TID_ENABLE=1 (scratch swizzle), raw multi-dword accesses step by 4 * IS. OOB_SELECT 3 with active
;   swizzle and stride != 0 checks (index + soffset) >= NUM_RECORDS or offset + payload > stride.
; Thread 0 seeds heap bytes 0x40..0x8f (region offset X = byte 0x40 + X) with 0x5eed0000 + X; the heap is 0xee-filled,
; so an in-range access past the seed reads 0xeeeeeeee. V# A (s[4:7], literal): base 0x40, swizzled, stride 8, NR 16,
; IS 8, OOB_SELECT 3. V# B (s[8:11], words 1 and 3 or-ed at run time): base 0x40, swizzled, stride 8, NR 8, IS 16,
; OOB_SELECT 0. V# C (s[12:15], literal): base 0x18 (out[6]), linear, stride 4, NR 2, ADD_TID, OOB_SELECT 1.
; Words, in out[0..7] (in brackets what an unswizzled translation reads):
;   0 A, idx 1, off 4:          (0 * 8 + 4) * 8 + 1 * 4 = X 0x24: 0x5eed0024           [X 0xc]
;   1 A, idx 9, off 0:          (1 * 8 + 0) * 8 + 1 * 4 = X 0x44: 0x5eed0044           [X 0x48]
;   2 B, idx 3, off 4:          (0 * 8 + 4) * 16 + 3 * 4 = X 0x4c: 0x5eed004c          [X 0x1c]
;   3 A x2, idx 2, off 0, dword 0: 0 * 8 + 2 * 4 = X 0x8: 0x5eed0008                   [X 0x10]
;   4 A x2, idx 2, dword 1 (idxen=1 steps linearly +4): X 0x8 + 4 = X 0xc: 0x5eed000c  [X 0x14]
;   5 A, idx 16, off 0:         16 >= NR 16 -> 0     (the raw check, 4 <= 16, reads X 0x80 past the seed: 0xeeeeeeee)
;   6, 7 lanes 0 and 1 store 0xbeef0000 + lane through C: address 0x18 + lane * 4 -> 0xbeef0000 0xbeef0001
;        (without ADD_TID both lanes store to out[6], and out[7] keeps 0xeeeeeeee)
; Lanes 0 and 1 only store (no lane reads what another wrote); thread 0 alone seeds and loads. No wave64 requirement.
; Run: vk_compute_test cs_buf_swizzle.spv 1 dump -> out[0..7] = 0x5eed0024 0x5eed0044 0x5eed004c 0x5eed0008
; 0x5eed000c 0x00000000 0xbeef0000 0xbeef0001 (1592590372 1592590404 1592590412 1592590344 1592590348 0 3203334144
; 3203334145).
v_cmpx_gt_u32_e32 2, v0
s_mov_b32 s12, 0x18
s_mov_b32 s13, 0x40000
s_mov_b32 s14, 2
s_mov_b32 s15, 0x11816fac
v_or_b32_e32 v40, 0xbeef0000, v0
buffer_store_dword v40, off, s[12:15], 0
s_waitcnt vmcnt(0)
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
v_mov_b32 v1, 0x5eed0020
v_mov_b32 v2, 0x5eed0024
v_mov_b32 v3, 0x5eed0028
v_mov_b32 v4, 0x5eed002c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:96
v_mov_b32 v1, 0x5eed0030
v_mov_b32 v2, 0x5eed0034
v_mov_b32 v3, 0x5eed0038
v_mov_b32 v4, 0x5eed003c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:112
v_mov_b32 v1, 0x5eed0040
v_mov_b32 v2, 0x5eed0044
v_mov_b32 v3, 0x5eed0048
v_mov_b32 v4, 0x5eed004c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:128
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0x80080000
s_mov_b32 s6, 16
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0x80000
s_mov_b32 s20, 0x80000000
s_or_b32 s9, s9, s20
s_mov_b32 s10, 8
s_mov_b32 s11, 0x01216fac
s_mov_b32 s21, 0
s_or_b32 s11, s11, s21
v_mov_b32 v20, 1
v_mov_b32 v21, 4
v_mov_b32 v22, 9
v_mov_b32 v23, 0
v_mov_b32 v24, 3
v_mov_b32 v25, 4
v_mov_b32 v26, 2
v_mov_b32 v27, 16
buffer_load_dword v10, v[20:21], s[4:7], 0 idxen offen
buffer_load_dword v11, v[22:23], s[4:7], 0 idxen offen
buffer_load_dword v12, v[24:25], s[8:11], 0 idxen offen
buffer_load_dwordx2 v[13:14], v26, s[4:7], 0 idxen
buffer_load_dword v15, v27, s[4:7], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx2 v[14:15], off, s[0:3], 0 offset:16
s_endpgm
