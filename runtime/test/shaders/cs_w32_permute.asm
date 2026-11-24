; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-90: ds_permute_b32 vD, vAddr, vData [offset:N], the forward (scatter) permute (RDNA2 ISA, DS opcode 178).
; Every lane that is on in EXEC sends vData to lane ((vAddr + offset) / 4) mod 32; a destination lane nobody sends
; to gets 0 (the ISA's tmp[0..63] = 0), and only lanes that are on in EXEC are written (VDST of the others is kept).
; v0 = lane (LocalInvocationId.x, 0..31), v2 = 7 * lane + 1 (data). Cases with ONE source per destination only:
; (A) addr = (lane + 1) * 4, a rotation: lane l writes lane l + 1, so lane d holds data[d - 1].
;     lane 0 <- lane 31 = 218, lane 5 <- lane 4 = 29.
; (B) addr = (31 - lane) * 4, offset:8, a reversal shifted by two lanes: lane l writes lane (33 - l) mod 32.
;     lane 0 <- lane 1 = 8, lane 3 <- lane 30 = 211.
; (C) EXEC = lanes 0-15 (v12 preset to 99), addr = lane * 8: lane l writes lane 2 * l, 0 <= l < 16.
;     lane 3 is on but no source targets it: 0 (not its old 99). Lane 4 <- lane 2 = 15. Lane 20 is off in EXEC
;     and keeps 99, although lane 10 targets it (data would be 71).
; (D) EXEC = even lanes (v13 preset to 99): an even lane l writes l + 2, an odd lane l writes l + 1 (but is off,
;     so it sends nothing). Lane 4 <- lane 2 = 15; the off lane 3, which also targets lane 4, must not win (22).
; Lane 0 and a few others store: out[0..7] = A0 A5 B0 B3 C3 C4 C20 D4 = 218 29 8 211 0 15 99 15.
; group: 32,1,1
; Run: vk_compute_test cs_w32_permute.spv 1 dump -> out[0..7] = 218 29 8 211 0 15 99 15
; mattr: +wavefrontsize32,-wavefrontsize64
v_mad_u32_u24 v2, v0, 7, 1
v_add_nc_u32_e32 v3, 1, v0
v_lshlrev_b32_e32 v3, 2, v3
ds_permute_b32 v10, v3, v2
v_sub_nc_u32_e32 v4, 31, v0
v_lshlrev_b32_e32 v4, 2, v4
ds_permute_b32 v11, v4, v2 offset:8
v_mov_b32_e32 v12, 99
v_lshlrev_b32_e32 v5, 3, v0
s_mov_b32 exec_lo, 0xffff
ds_permute_b32 v12, v5, v2
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
v_and_b32_e32 v6, 1, v0
v_sub_nc_u32_e32 v6, 2, v6
v_add_nc_u32_e32 v6, v0, v6
v_lshlrev_b32_e32 v6, 2, v6
v_mov_b32_e32 v13, 99
s_mov_b32 exec_lo, 0x55555555
ds_permute_b32 v13, v6, v2
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 5, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 3, v0
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 3, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 4, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 20, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 4, v0
v_mov_b32_e32 v20, v13
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
