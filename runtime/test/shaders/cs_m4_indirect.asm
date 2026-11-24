; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x44444e49 0x3d344d5f
; M4 compute shader 2: runs under ExecuteIndirect with the arguments cs_m4_fill wrote ({ groups, 1, 1 }), reads the
; structured buffer and the image the first dispatch wrote, and writes a raw output buffer. Run:
; d3d12_m4_compute_test. Same root signature, user data and table as cs_m4_fill (see there):
;   slot 0: structured buffer (read)   slot 2: R32_UINT image (read, image_load)   slot 4: raw output (written)
; D3D12 build: runtime/CMakeLists.txt builds this with LEONIDA_USER_DATA_SHIFT=4, so in the PSO every SGPR named
; here is 4 higher (the console compute layout starts the root arguments at s4, D3D-448). The table pointer
; then reads s[8:9]: s8 is the table's 32-bit VA and s9 a user dword the runtime leaves zero.
; g = 64 * group + lane; t = image(g & 15, g >> 4); record g = { w0, w1, w2, w3 }
;   out[g] = { (w0 + t) ^ w3, (w2 ^ t) + w1 } at byte offset 8 * g
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_load_dwordx8 s[16:23], s[4:5], 0x40
s_load_dwordx4 s[12:15], s[4:5], 0x80
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
buffer_load_dwordx4 v[4:7], v1, s[8:11], 0 idxen
v_and_b32_e32 v8, 15, v1
v_lshrrev_b32_e32 v9, 4, v1
image_load v10, v[8:9], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v11, v4, v10
v_xor_b32_e32 v11, v11, v7
v_xor_b32_e32 v12, v6, v10
v_add_nc_u32_e32 v12, v12, v5
v_lshlrev_b32_e32 v13, 3, v1
buffer_store_dwordx2 v[11:12], v13, s[12:15], 0 offen
s_endpgm
