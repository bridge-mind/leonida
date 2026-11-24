; global_load/global_store with a negative instruction offset: address = base + offset, the offset a signed 12-bit
; value as llvm-mc prints it (offset:-4 is the field 0xffc). Measured on an AMD Radeon RX 6800 (gfx1030) as a raw
; compute kernel, dword at byte n = 0x10000000+n, pointer 0x1000: offset:-4 reads 0x10000ffc, -2048 0x10000800,
; 2044 0x100017fc. A translator that subtracts 0x1000 again lands 4096 bytes lower.
; Run with 40 groups so the heap is 40*256 = 0x2800 bytes: both the right addresses and the 4096-lower ones are
; inside it. Only group 0 thread 0 runs (s4 is the workgroup id: 4 user SGPRs, then TGID_X). Pointer P = 0x2000.
; Seeds (MUBUF stores through the V# in s[0:3], base 0), right addresses then the 4096-lower ones:
;   0x1ff0 0x5eed1ff0  0x1ff4 0x5eed1ff4  0x1ffc 0x5eed1ffc  0x1800 0x8bad1800  0x27fc 0x9e5e27fc
;   0x0ff0 0xbad00ff0  0x0ff4 0xbad00ff4  0x0ffc 0xbad00ffc  0x0800 0xbad00800
; Loads and the store:
;   out[0]   global_load_dword   v10, v30(0), s[20:21](P)          offset:-4    byte 0x1ffc -> 0x5eed1ffc
;   out[1]   global_load_dword   v11, v31(0x800), s[22:23](0x1800) offset:-2048 byte 0x1800 -> 0x8bad1800
;   out[2,3] global_load_dwordx2 v[12:13], v[2:3](P), off          offset:-16   bytes 0x1ff0, 0x1ff4
;   out[4]   global_load_ubyte   v14, v[2:3](P), off               offset:2047  byte 0x27ff = top byte of 0x9e5e27fc -> 0x9e
;   out[5]   global_load_sshort  v15, v30(0), s[20:21](P)          offset:-2046 bytes 0x1802..3 = 0x8bad -> 0xffff8bad
;   global_store_dword v[4:5](0x2400), 0x57023ffc, off offset:-4 -> byte 0x23fc; out[6] reads 0x23fc back and
;   out[7] reads 0x13fc (4096 lower, never written: the 0xee fill) through MUBUF.
; Run: vk_compute_test cs_global_neg_offset.spv 40 dump -> out[0..7] = 0x5eed1ffc 0x8bad1800 0x5eed1ff0 0x5eed1ff4
; 0x0000009e 0xffff8bad 0x57023ffc 0xeeeeeeee (1592598524 2343376896 1592598512 1592598516 158 4294937517 1459765244
; 4008636142). Same FSSM header as cs_lds.
v_or_b32_e32 v1, s4, v0
v_cmpx_eq_u32_e32 0, v1
v_mov_b32 v1, 0x5eed1ff0
v_mov_b32 v2, 0x1ff0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed1ff4
v_mov_b32 v2, 0x1ff4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed1ffc
v_mov_b32 v2, 0x1ffc
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x8bad1800
v_mov_b32 v2, 0x1800
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x9e5e27fc
v_mov_b32 v2, 0x27fc
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0xbad00ff0
v_mov_b32 v2, 0xff0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0xbad00ff4
v_mov_b32 v2, 0xff4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0xbad00ffc
v_mov_b32 v2, 0xffc
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0xbad00800
v_mov_b32 v2, 0x800
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt_vscnt null, 0x0
s_mov_b32 s20, 0x2000
s_mov_b32 s21, 0
s_mov_b32 s22, 0x1800
s_mov_b32 s23, 0
v_mov_b32 v30, 0
v_mov_b32 v31, 0x800
v_mov_b32 v2, 0x2000
v_mov_b32 v3, 0
v_mov_b32 v4, 0x2400
v_mov_b32 v5, 0
global_load_dword v10, v30, s[20:21] offset:-4
global_load_dword v11, v31, s[22:23] offset:-2048
global_load_dwordx2 v[12:13], v[2:3], off offset:-16
global_load_ubyte v14, v[2:3], off offset:2047
global_load_sshort v15, v30, s[20:21] offset:-2046
v_mov_b32 v1, 0x57023ffc
global_store_dword v[4:5], v1, off offset:-4
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
v_mov_b32 v2, 0x23fc
buffer_load_dword v16, v2, s[0:3], 0 offen
v_mov_b32 v2, 0x13fc
buffer_load_dword v17, v2, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32 v2, 0
buffer_store_dword v10, v2, s[0:3], 0 offen
v_mov_b32 v2, 4
buffer_store_dword v11, v2, s[0:3], 0 offen
v_mov_b32 v2, 8
buffer_store_dword v12, v2, s[0:3], 0 offen
v_mov_b32 v2, 12
buffer_store_dword v13, v2, s[0:3], 0 offen
v_mov_b32 v2, 16
buffer_store_dword v14, v2, s[0:3], 0 offen
v_mov_b32 v2, 20
buffer_store_dword v15, v2, s[0:3], 0 offen
v_mov_b32 v2, 24
buffer_store_dword v16, v2, s[0:3], 0 offen
v_mov_b32 v2, 28
buffer_store_dword v17, v2, s[0:3], 0 offen
s_endpgm
