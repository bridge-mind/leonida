; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x44505954 0x3d424655
; Typed buffer UAVs (D3D12 CreateUnorderedAccessView with a Format on a BUFFER view): hand-written RDNA2 the way a
; translated `RWBuffer<T>` shader reads and writes them. Run: d3d12_buffer_uav_test (a compute PSO found by the
; SHA-256 of its synthetic FSSM header, as for the other cs_ shaders).
; User data, from the root signature: [0] four 32-bit constants, [1] a descriptor table of six UAVs. RSRC2 0x808c =
; D3D12 build: runtime/CMakeLists.txt builds this with LEONIDA_USER_DATA_SHIFT=4, so in the PSO every SGPR named
; here is 4 higher (the console compute layout starts the root arguments at s4, D3D-448). The table pointer
; then reads s[8:9]: s8 is the table's 32-bit VA and s9 a user dword the runtime leaves zero.
; six user SGPRs and TGID_X, so the group index is s6:
;   s0 = seed   s1 = multiplier   s[4:5] = console VA of the table's first slot (a slot is 32 bytes, a V# in its first 16)
;   slot 0 (+0x00): V#, typed R32_UINT, 240 elements of a 256-element buffer (written, then atomically added to)
;   slot 1 (+0x20): V#, typed R16G16B16A16_FLOAT, 256 elements (written)
;   slot 2 (+0x40): V#, typed R8G8B8A8_UNORM, 256 elements from element 16 of the buffer (written)
;   slot 3 (+0x60): V#, typed R32G32B32A32_UINT, 256 elements (written)
;   slot 4 (+0x80): V#, typed R32_FLOAT, 250 elements of a 256-element buffer (read)
;   slot 5 (+0xa0): V#, typed R32_SINT, 256 elements (written)
; g = 64 * group + lane, 0..255 (an index past a view's NUM_RECORDS is out of range: a load gives 0, a store or an
; atomic touches nothing):
;   slot 0: element g = g * mul + seed, then += g
;   slot 1: element g = { f16(g), f16(g + 0.5), f16(-1 - g), f16(0.25) }
;   slot 2: element g = unorm8 { (g & 255) / 255, (255 - (g & 255)) / 255, 0, 1 }
;   slot 3: element g = { g, g ^ 0xffff, g * 7, 0xa5 }
;   slot 5: element g = (int)(2 * input[g]), input[g] = g - 100.5
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_load_dwordx4 s[12:15], s[4:5], 0x20
s_load_dwordx4 s[16:19], s[4:5], 0x40
s_load_dwordx4 s[20:23], s[4:5], 0x60
s_load_dwordx4 s[24:27], s[4:5], 0x80
s_load_dwordx4 s[28:31], s[4:5], 0xa0
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
v_mul_lo_u32 v2, v1, s1
v_add_nc_u32_e32 v2, s0, v2
buffer_store_format_x v2, v1, s[8:11], 0 idxen
v_mov_b32_e32 v3, v1
buffer_atomic_add v3, v1, s[8:11], 0 idxen
v_cvt_f32_u32_e32 v4, v1
v_add_f32_e32 v5, 0.5, v4
v_sub_f32_e32 v6, -1.0, v4
v_mov_b32_e32 v7, 0x3e800000
buffer_store_format_xyzw v[4:7], v1, s[12:15], 0 idxen
v_and_b32_e32 v8, 255, v1
v_cvt_f32_ubyte0_e32 v9, v8
v_mov_b32_e32 v10, 0x3b808081
v_mul_f32_e32 v11, v9, v10
v_sub_nc_u32_e32 v12, 255, v8
v_cvt_f32_ubyte0_e32 v12, v12
v_mul_f32_e32 v12, v12, v10
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 1.0
buffer_store_format_xyzw v[11:14], v1, s[16:19], 0 idxen
v_mov_b32_e32 v15, v1
v_xor_b32_e32 v16, 0xffff, v1
v_mul_lo_u32 v17, v1, 7
v_mov_b32_e32 v18, 0xa5
buffer_store_format_xyzw v[15:18], v1, s[20:23], 0 idxen
buffer_load_format_x v19, v1, s[24:27], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32_e32 v19, v19, v19
v_cvt_i32_f32_e32 v19, v19
buffer_store_format_x v19, v1, s[28:31], 0 idxen
s_endpgm
