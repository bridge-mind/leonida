; Synthetic f64 input 1.5, low word first.
; Expected f32 result words when the test appends output modifiers:
; none        0x3fc00000
; mul:2       0x40400000
; mul:4       0x40c00000
; div:2       0x3f400000
; clamp       0x3f800000
; clamp div:2 0x3f400000 (clamp before scaling would give 0x3f000000)
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3ff80000
v_cvt_f32_f64_e64 v3, v[0:1]
