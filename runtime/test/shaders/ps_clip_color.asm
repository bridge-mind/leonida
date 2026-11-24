; Dedicated fragment shader for vertex clipping tests.
; Export finite, exactly representable RGBA32F values so
; NaN payload canonicalization cannot affect clip validation.
; Expected mrt0 bits: 0x3e800000, 0x3f000000,
;                     0x3f400000, 0x3f800000

v_mov_b32 v0, 0x3e800000
v_mov_b32 v1, 0x3f000000
v_mov_b32 v2, 0x3f400000
v_mov_b32 v3, 0x3f800000
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
