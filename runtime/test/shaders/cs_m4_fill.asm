; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x4c4c4946 0x3d344d5f
; M4 compute shader 1: fills a structured UAV buffer and a typed UAV image from a raw input buffer, and writes the
; arguments of the next (indirect) dispatch. Run: d3d12_m4_compute_test (a compute PSO found by the SHA-256 of its
; synthetic FSSM header, as for the other cs_ shaders).
; User data, from the root signature: [0] four 32-bit constants, [1] a descriptor table of five UAVs. RSRC2 0x808c =
; D3D12 build: runtime/CMakeLists.txt builds this with LEONIDA_USER_DATA_SHIFT=4, so in the PSO every SGPR named
; here is 4 higher (the console compute layout starts the root arguments at s4, D3D-448). The table pointer
; then reads s[8:9]: s8 is the table's 32-bit VA and s9 a user dword the runtime leaves zero.
; six user SGPRs and TGID_X, so the group index is s6:
;   s0 = seed   s1 = multiplier   s2 = group count of the indirect dispatch   s3 = unused
;   s[4:5] = console VA of the table's first slot. A slot is 32 bytes: a V# in its first 16 bytes or a T# in all 32.
;   slot 0 (+0x00): V#, structured buffer, stride 16, 256 records (written)
;   slot 1 (+0x20): V#, raw buffer, the arguments of the indirect dispatch (written)
;   slot 2 (+0x40): T#, 16x16 R32_UINT image (written)
;   slot 3 (+0x60): V#, raw buffer, 256 input words (read)
;   slot 4 (+0x80): V#, raw buffer, the second shader's output (not used here)
; g = 64 * group + lane, 0..255:  in = input[g]
;   record g = { in + seed, g * mul, in ^ (g << 8), rotr(in, 13) }
;   image(x = g & 15, y = g >> 4) = in * 3 + x * 0x10001 + y
;   g == 0 also writes the arguments { s2, 1, 1, 0xc0de0001 }
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_load_dwordx4 s[12:15], s[4:5], 0x20
s_load_dwordx8 s[16:23], s[4:5], 0x40
s_load_dwordx4 s[24:27], s[4:5], 0x60
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
v_lshlrev_b32_e32 v2, 2, v1
buffer_load_dword v3, v2, s[24:27], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v4, s0, v3
v_mul_lo_u32 v5, v1, s1
v_lshlrev_b32_e32 v6, 8, v1
v_xor_b32_e32 v6, v3, v6
v_alignbit_b32 v7, v3, v3, 13
buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 idxen
v_and_b32_e32 v8, 15, v1
v_lshrrev_b32_e32 v9, 4, v1
v_mul_u32_u24_e32 v10, 0x10001, v8
v_mul_lo_u32 v11, v3, 3
v_add_nc_u32_e32 v10, v10, v11
v_add_nc_u32_e32 v10, v10, v9
image_store v10, v[8:9], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, s2
v_mov_b32_e32 v14, 1
v_mov_b32_e32 v15, 1
v_mov_b32_e32 v16, 0xc0de0001
buffer_store_dwordx4 v[13:16], v12, s[12:15], 0 offen
s_endpgm
