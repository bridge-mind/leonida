; v_add_f64 under MODE.FP_ROUND[3:2] (3 = toward zero, 0 = nearest even) with extreme magnitudes near +-DBL_MAX.
; Expected mrt0 bits: 0xc12af665, 0x3fe7ffff, 0xbfffffff, 0xc12af666
;
; mrt0.x: 0x7fdd9bb47daa1333 + 0xffefffffffffffff (RTZ) -> 0xffe13225:c12af665; low word 0xc12af665 (#2891; RNE 0xc12af666)
; mrt0.y: 0xffcffffffffffffe + 0x7fefffffffffffff (RTZ) -> 0x7fe7ffff:ffffffff (borrow across lo=0); hi ^ 0x40000000 = 0x3fe7ffff
; mrt0.z: 0x7fefffffffffffff + 0x7fefffffffffffff (RTZ) -> 0x7fefffff:ffffffff (finite overflow -> +DBL_MAX); lo ^ 0x40000000 = 0xbfffffff
; mrt0.w: 0x7fdd9bb47daa1333 + 0xffefffffffffffff after switching back to RNE (0) -> low word 0xc12af666
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
v_mov_b32_e32 v0, 0x7daa1333
v_mov_b32_e32 v1, 0x7fdd9bb4
v_mov_b32_e32 v2, 0xffffffff
v_mov_b32_e32 v3, 0xffefffff
v_add_f64 v[4:5], v[0:1], v[2:3]
v_mov_b32_e32 v6, 0xfffffffe
v_mov_b32_e32 v7, 0xffcfffff
v_mov_b32_e32 v8, 0xffffffff
v_mov_b32_e32 v9, 0x7fefffff
v_add_f64 v[10:11], v[6:7], v[8:9]
v_xor_b32_e32 v11, 0x40000000, v11
v_add_f64 v[12:13], v[8:9], v[8:9]
v_xor_b32_e32 v12, 0x40000000, v12
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
v_add_f64 v[14:15], v[0:1], v[2:3]
exp mrt0 v4, v11, v12, v14 done vm
s_endpgm
