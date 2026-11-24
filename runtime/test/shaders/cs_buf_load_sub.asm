; TEST-510: sub-dword MUBUF loads on the GPU (docs/design.md, "Sub-dword MUBUF loads and GFX10_FORMAT_8_UINT";
; CPU-checked by tests/semantic/test_buffer_load_sub.py). Thread 0 seeds heap bytes 0x40..0x47 with 7f 80 ff 01 ab cd
; 00 fe; the heap is 0xee-filled past them. The V# has base 0x40, stride 0, NUM_RECORDS 8 and OOB_SELECT 3 (RAW: an
; element is in range when offset + size <= NUM_RECORDS; out of range reads 0 without touching memory). Words, in
; out[0..7]:
;   0 buffer_load_ubyte  at 1:  0x80                       zero-extended
;   1 buffer_load_sbyte  at 1:  0x80 -> 0xffffff80         sign-extended
;   2 buffer_load_ushort at 2:  ff 01 -> 0x01ff
;   3 buffer_load_sshort at 0:  7f 80 -> 0xffff807f
;   4 buffer_load_ushort at 3:  01 ab -> 0xab01            crosses into the next dword (3 + 2 <= 8)
;   5 buffer_load_sshort at 5:  cd 00 -> 0x000000cd        positive, no sign bits
;   6 buffer_load_ubyte  at 8:  8 + 1 > 8 -> 0
;   7 buffer_load_ushort at 7:  7 + 2 > 8 -> 0             a start-only check would read fe ee -> 0xeefe
; One thread, no wave64 requirement.
; Run: vk_compute_test cs_buf_load_sub.spv 1 dump -> out[0..7] = 0x00000080 0xffffff80 0x000001ff 0xffff807f
; 0x0000ab01 0x000000cd 0x00000000 0x00000000 (128 4294967168 511 4294934655 43777 205 0 0).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x01ff807f
v_mov_b32 v2, 0xfe00cdab
buffer_store_dwordx2 v[1:2], off, s[0:3], 0 offset:64
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 8
s_mov_b32 s7, 0x31016fac
v_mov_b32 v20, 1
v_mov_b32 v21, 2
v_mov_b32 v22, 0
v_mov_b32 v23, 3
v_mov_b32 v24, 5
v_mov_b32 v25, 8
v_mov_b32 v26, 7
buffer_load_ubyte v10, v20, s[4:7], 0 offen
buffer_load_sbyte v11, v20, s[4:7], 0 offen
buffer_load_ushort v12, v21, s[4:7], 0 offen
buffer_load_sshort v13, v22, s[4:7], 0 offen
buffer_load_ushort v14, v23, s[4:7], 0 offen
buffer_load_sshort v15, v24, s[4:7], 0 offen
buffer_load_ubyte v16, v25, s[4:7], 0 offen
buffer_load_ushort v17, v26, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16
s_endpgm
