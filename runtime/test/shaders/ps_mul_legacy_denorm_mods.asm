; The product is flushed before VOP3 output modifiers, and the modified result is flushed again.
; A subnormal scaled to a normal number by mul:2 or mul:4 must already be zero.
; On native RDNA2 with IEEE off an omod changes -0 into +0. An ordinary normal product can still
; be modified. Legacy -0 * -inf gives +0.
; Expected mrt0 bits: 0x00000000 0x00000000 0x00000000 0x40400000
v_mov_b32 v0, 0x00800000
v_mov_b32 v1, 0x3f000000
v_mul_legacy_f32_e64 v4, v0, v1 mul:2
v_mov_b32 v2, 0x80800000
v_mul_legacy_f32_e64 v5, v2, v1 mul:4
v_mov_b32 v3, 0xff800000
v_mov_b32 v6, 0x80000000
v_mul_legacy_f32 v7, v6, v3
v_mov_b32 v9, 0x40000000
v_mov_b32 v10, 0x40400000
v_mul_legacy_f32_e64 v8, v9, v10 div:2
exp mrt0 v4, v5, v7, v8 done vm
s_endpgm
