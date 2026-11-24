; Constant EXEC masks in wave32: s_mov_b32 exec_lo, <imm> turns on exactly the lanes whose bit is set.
; v0 = lane (LocalInvocationId.x, 0..31). For each mask a VALU write marks the lanes that ran it; a later full-EXEC
; v_cmp turns the marks back into a lane mask, and lane 0 stores its lane count (s_bcnt1) and lowest lane (s_ff1).
; Three constants fit in 16 bits (0xffff, 0xff, 0x8001), which is the case Intel's Windows driver
; (UHD 770 / Xe-LP, 32.0.101.6129) got wrong for a per-lane OpBitFieldUExtract of the constant: it took the lane
; index mod 16, so lanes 16-31 copied lanes 0-15 (0xffff ran all 32 lanes). 0xffff0000 does not fit and was right.
; Lane 0 stores: out[0..3] = set lanes of 0xffff 0xff 0x8001 0xffff0000 = 16 8 2 16; out[4..5] = lowest lane of
; 0x8001 and of 0xffff0000 = 0 16. With the lane taken mod 16 (the Intel bug) it was 32 16 4 16 0 16.
; group: 32,1,1
; Run: vk_compute_test cs_exec_const.spv 1 dump -> out[0..5] = 16 8 2 16 0 16
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
s_mov_b32 exec_lo, 0xffff
v_mov_b32_e32 v10, 1
s_mov_b32 exec_lo, 0xff
v_mov_b32_e32 v11, 1
s_mov_b32 exec_lo, 0x8001
v_mov_b32_e32 v12, 1
s_mov_b32 exec_lo, 0xffff0000
v_mov_b32_e32 v13, 1
s_mov_b32 exec_lo, -1
v_cmp_ne_u32_e32 vcc_lo, 0, v10
s_bcnt1_i32_b32 s6, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v11
s_bcnt1_i32_b32 s7, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v12
s_bcnt1_i32_b32 s8, vcc_lo
s_ff1_i32_b32 s10, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v13
s_bcnt1_i32_b32 s9, vcc_lo
s_ff1_i32_b32 s11, vcc_lo
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s6
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s7
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s11
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
