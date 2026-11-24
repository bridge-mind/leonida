; group: 1,1,1
; rsrc2: 0x308
; Run: vk_compute_test cs_workgroup_yz.spv 2x3x4 workgroup_yz
; One invocation per group. RSRC2 0x308 enables Y and Z only, so s4 = group Y, s5 = group Z.
; Each (Y, Z) bin must receive two increments, one per X group. Overflow bins must stay zero.
; A single group dispatched at a known base must increment only that base's (Y, Z) bin.
s_min_u32 s6, s4, 3
s_min_u32 s7, s5, 4
s_lshl_b32 s7, s7, 2
s_add_u32 s6, s7, s6
s_lshl_b32 s6, s6, 2
v_mov_b32_e32 v0, s6
v_mov_b32_e32 v1, 1
buffer_atomic_add v1, v0, s[0:3], 0 offen
s_endpgm
