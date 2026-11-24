; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; VOP3 FP source modifiers on constants (docs/design.md "VOP3 FP modifiers on constants"): on an inline constant or a
; literal, abs clears the sign bit first and neg flips it second; neg on a literal is not an integer negation.
;   x: v_add_f32_e64 0 + -|-2.0| = -2.0 (folding - into the constant first gave +2.0)
;   y: v_add_f32_e64 0 + neg(0x3f800000) = -1.0 (two's-complement negation gave 0xc0800000, -4.0)
;   z: v_add_f32_e64 0 + |0xbf800000| = 1.0 (dropping abs gave -1.0)
;   w: v_add_f16_e64 0 + -|0x3c00| = -1.0h in the low half, the destination's high half 0xAAAA kept
; Expected mrt0 bits: 0xc0000000, 0xbf800000, 0x3f800000, 0xaaaabc00
v_add_f32_e64 v2, 0, -|-2.0|
v_add_f32_e64 v3, 0, neg(0x3f800000)
v_add_f32_e64 v4, 0, |0xbf800000|
v_mov_b32_e32 v5, 0xaaaa1234
v_add_f16_e64 v5, 0, -|0x3c00|
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
