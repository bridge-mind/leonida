; GPU test for the two LDS families that docs/design.md lists as not GPU-verified:
;   ds_cmpst_b32 (the non-returning compare-and-store) and the b64 2-address forms
;   (ds_read2_b64 / ds_write2_b64 and their st64 variants).
;
; LDS 4 KiB (RSRC2 0x40088), group 32x1x1, only lane 0 runs. The output words are
; derived from the RDNA2 ISA's 2-address and DS_CMPST_B32 pseudocode:
;
;   out[0..3]  ds_write2_b64 then ds_read2_b64 at base 0x20, offset0:1 offset1:3,
;              writes A at 0x28 and B at 0x38 (unit 8), reads the same two elements:
;              A.lo A.hi B.lo B.hi
;   out[4]     ds_read2st64_b64 at base 0x100, offset0:0 offset1:1 (unit 512):
;              XOR of the four dwords seeded at 0x100 and 0x300
;   out[5]     ds_write2st64_b64 at base 0x100, offset0:0 offset1:1, then plain
;              ds_read_b32 at 0x100/0x104/0x300/0x304: XOR of the four dwords
;   out[6]     ds_read2_b64 v[2:5], v2 at base 0x200 (vdst overlaps vaddr): the
;              second element's low dword, 0xBBBB000B, read before v2 is overwritten
;   out[7]     ds_cmpst_b32 hit (mem 0x13572468 -> 0x2468ABCD) XOR the miss case's
;              unchanged word (0x11111111): 0x3579BADC
;
; Run: vk_compute_test cs_lds_b64_cmpst.spv 1 dump
;   -> out[0..7] = 0x11110001 0x22220002 0x33330003 0x44440004
;                  0xCCCC000C 0x44440004 0xBBBB000B 0x3579BADC
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0

; --- ds_write2_b64 / ds_read2_b64, unit 8 bytes ---
v_mov_b32_e32 v2, 0x20
v_mov_b32_e32 v4, 0x11110001
v_mov_b32_e32 v5, 0x22220002
v_mov_b32_e32 v6, 0x33330003
v_mov_b32_e32 v7, 0x44440004
ds_write2_b64 v2, v[4:5], v[6:7] offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[10:13], v2 offset0:1 offset1:3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen

; --- ds_read2st64_b64, unit 512 bytes: seed 0x100 and 0x300 ---
v_mov_b32_e32 v1, 0x100
v_mov_b32_e32 v3, 0x55550005
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x104
v_mov_b32_e32 v3, 0x66660006
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x300
v_mov_b32_e32 v3, 0x77770007
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x304
v_mov_b32_e32 v3, 0x88880008
ds_write_b32 v1, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, 0x100
ds_read2st64_b64 v[10:13], v2 offset0:0 offset1:1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v14, v10, v11
v_xor_b32_e32 v14, v14, v12
v_xor_b32_e32 v14, v14, v13
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen

; --- ds_write2st64_b64, unit 512 bytes: A at 0x100, B at 0x300 ---
v_mov_b32_e32 v2, 0x100
ds_write2st64_b64 v2, v[4:5], v[6:7] offset0:0 offset1:1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x100
ds_read_b32 v10, v1
v_mov_b32_e32 v1, 0x104
ds_read_b32 v11, v1
v_mov_b32_e32 v1, 0x300
ds_read_b32 v12, v1
v_mov_b32_e32 v1, 0x304
ds_read_b32 v13, v1
s_waitcnt lgkmcnt(0)
v_xor_b32_e32 v15, v10, v11
v_xor_b32_e32 v15, v15, v12
v_xor_b32_e32 v15, v15, v13
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen

; --- ds_read2_b64 with vdst overlapping vaddr: v[2:5] <- base v2 ---
v_mov_b32_e32 v1, 0x200
v_mov_b32_e32 v3, 0x99990009
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x204
v_mov_b32_e32 v3, 0xAAAA000A
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x208
v_mov_b32_e32 v3, 0xBBBB000B
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x20C
v_mov_b32_e32 v3, 0xCCCC000C
ds_write_b32 v1, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, 0x200
ds_read2_b64 v[2:5], v2 offset0:0 offset1:1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 24
buffer_store_dword v4, v21, s[0:3], 0 offen

; --- ds_cmpst_b32: hit at 0x400, miss at 0x404 ---
v_mov_b32_e32 v1, 0x400
v_mov_b32_e32 v3, 0x13572468
ds_write_b32 v1, v3
v_mov_b32_e32 v1, 0x404
v_mov_b32_e32 v3, 0x11111111
ds_write_b32 v1, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x400
v_mov_b32_e32 v3, 0x13572468
v_mov_b32_e32 v4, 0x2468ABCD
ds_cmpst_b32 v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_read_b32 v10, v1
v_mov_b32_e32 v1, 0x404
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0xFFFFFFFF
ds_cmpst_b32 v1, v3, v4
s_waitcnt lgkmcnt(0)
ds_read_b32 v11, v1
v_xor_b32_e32 v16, v10, v11
v_mov_b32_e32 v21, 28
buffer_store_dword v16, v21, s[0:3], 0 offen
s_endpgm
