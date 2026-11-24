; v_movrelsd_2_b32: VGPR[D + M0[25:16]] = VGPR[S0 + M0[9:0]] (RDNA2 ISA, VOP1 opcode 72). Unlike v_movrelsd_b32,
; each side has its own 10-bit M0 field, and M0[15:10] and M0[31:26] are ignored.
; v1..v8 = 11, 22, ..., 88; v15 = 1515, v17 = 1717, v25 = 0.
; 1. m0 = 0xFC03FC02 (dst 3, src 2, ignored bits set): v_movrelsd_2_b32 v1, v1 -> v4 = v3 = 33.
; 2. m0 = (20 << 16) | 10 (the ISA's example): v_movrelsd_2_b32 v5, v7 -> v25 = v17 = 1717.
;    v15 (v5 + the source offset) keeps 1515: a translator that offset both sides by M0[9:0] would write it.
; Lane 0 stores: out[0..7] = v1 v3 v4 v5 v25 v7 v15 v17 = 11 33 33 55 1717 77 1515 1717.
; group: 32,1,1
; Run: vk_compute_test cs_movrelsd_2.spv 1 dump -> out[0..7] = 11 33 33 55 1717 77 1515 1717
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 11
v_mov_b32_e32 v2, 22
v_mov_b32_e32 v3, 33
v_mov_b32_e32 v4, 44
v_mov_b32_e32 v5, 55
v_mov_b32_e32 v6, 0x42
v_mov_b32_e32 v7, 0x4d
v_mov_b32_e32 v8, 0x58
v_mov_b32_e32 v15, 0x5eb
v_mov_b32_e32 v17, 0x6b5
v_mov_b32_e32 v25, 0
s_mov_b32 m0, 0xfc03fc02
v_movrelsd_2_b32 v1, v1
s_mov_b32 m0, 0x14000a
v_movrelsd_2_b32 v5, v7
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v5, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v25, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v7, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
