; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; VOP3 output modifiers on v_cvt_f16_f32, in the order an RX 6800 applies them (@shelbykb2, #718): round to f16, then
; omod, then clamp. With an omod a zero or denormal f16 gives +0. The destination's high half is kept.
;   1.5 clamp -> 1.0 (0x3c00), high half 0xaaaa kept    0.375 mul:4 -> 1.5 (0x3e00)
;   2^-15 mul:2 -> +0 (the f16 is a denormal)          65520 div:2 -> inf (65520 rounds to inf first)
;   -0 mul:2 -> +0                                      2.0 mul:2 clamp -> 1.0 (omod before clamp)
; Words 2 and 3 hold two results each, low half first. On the RX 6800 1.5 clamp gave 0x3c00 in every float mode, and
; 0.375 mul:4, 2^-15 mul:2, 65520 div:2 and -0 mul:2 gave these halves with IEEE off and f16 denormals flushed, the
; mode an omod implies (with f16 denormals kept the omod is ignored). 2.0 mul:2 clamp wasn't measured as such.
; With the modifiers dropped the words are 0xaaaa3e00, 0xaaaa3600, 0x7c000200, 0x40008000; scaled before the
; rounding, as the f16 ALU results are, words 2 and 3 are 0x78000400 and 0x3c008000.
; Expected mrt0 bits: 0xaaaa3c00, 0xaaaa3e00, 0x7c000000, 0x3c000000
v_mov_b32 v0, 1.5
v_mov_b32 v1, 0x3ec00000
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_cvt_f16_f32_e64 v2, v0 clamp
v_cvt_f16_f32_e64 v3, v1 mul:4
v_mov_b32 v8, 0x38000000
v_mov_b32 v9, 0x477ff000
v_mov_b32 v10, 0x80000000
v_mov_b32 v11, 2.0
v_mov_b32 v4, 0
v_mov_b32 v5, 0
v_mov_b32 v6, 0
v_mov_b32 v7, 0
v_cvt_f16_f32_e64 v4, v8 mul:2
v_cvt_f16_f32_e64 v5, v9 div:2
v_cvt_f16_f32_e64 v6, v10 mul:2
v_cvt_f16_f32_e64 v7, v11 clamp mul:2
v_lshl_or_b32 v4, v5, 16, v4
v_lshl_or_b32 v5, v7, 16, v6
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
