; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-90: wave64 ds_permute_b32. Like ds_bpermute_b32 it runs as two independent wave32 halves (RDNA2 ISA 10.4.4): a
; source lane writes lane ((vAddr + offset) / 4) mod 32 of ITS OWN half, a destination nobody writes gets 0, and
; EXEC gates source and destination. v0 = lane 0..63, v2 = 7 * lane + 1 (data), v10..v13 start at 99.
; (A) addr = (lane + 1) * 4: a rotation inside each half. Lane 0 <- lane 31 = 218. Lane 32 <- lane 63 = 442 (NOT lane
;     31: a half never writes the other). Lane 63 <- lane 62 = 435.
; (B) addr = (31 - (lane & 31)) * 4, offset:8: lane l writes rel (33 - rel) mod 32 of its half. Lane 32 <- lane 33 = 232.
; (C) EXEC = lanes 0-15 and 32-47, addr = lane * 8: lane l writes rel 2 * rel. Lane 36 <- lane 34 = 239. Lane 35 is on
;     and nobody writes it: 0. Lane 52 is off: 99.
; (D) EXEC = even lanes, an even lane l writes l + 2, an odd lane l writes l + 1: lane 36 <- lane 34 = 239; the off
;     lane 35, which also targets lane 36, must not win (it would give 246).
; out[0..7] = A0 A32 A63 C36 C35 C52 D36 B32 = 218 442 435 239 0 99 239 232
; Run: vk_compute_test cs_w64x_permute.spv 1 dump --require-wave64 -> out[0..7] = 218 442 435 239 0 99 239 232
; mattr: +wavefrontsize64,-wavefrontsize32
v_mad_u32_u24 v2, v0, 7, 1
v_add_nc_u32_e32 v3, 1, v0
v_lshlrev_b32_e32 v3, 2, v3
ds_permute_b32 v10, v3, v2
v_and_b32_e32 v4, 31, v0
v_sub_nc_u32_e32 v4, 31, v4
v_lshlrev_b32_e32 v4, 2, v4
ds_permute_b32 v11, v4, v2 offset:8
v_mov_b32_e32 v12, 99
v_lshlrev_b32_e32 v5, 3, v0
s_mov_b32 exec_lo, 0xffff
s_mov_b32 exec_hi, 0xffff
ds_permute_b32 v12, v5, v2
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_and_b32_e32 v6, 1, v0
v_sub_nc_u32_e32 v6, 2, v6
v_add_nc_u32_e32 v6, v0, v6
v_lshlrev_b32_e32 v6, 2, v6
v_mov_b32_e32 v13, 99
s_mov_b32 exec_lo, 0x55555555
s_mov_b32 exec_hi, 0x55555555
ds_permute_b32 v13, v6, v2
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 63, v0
v_mov_b32_e32 v20, v10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 36, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 35, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 52, v0
v_mov_b32_e32 v20, v12
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 36, v0
v_mov_b32_e32 v20, v13
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v0
v_mov_b32_e32 v20, v11
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
