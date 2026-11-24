; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_cvt_off_f32_i4 (docs/design.md "Constant and ordered VOPC, ..."): D = sext(S0[3:0]) / 16, the ISA's CVT_OFF table
; (0 .. 0.4375, then -0.5 .. -0.0625); bits 4-31 are ignored and the VOP3 output modifiers apply, omod before clamp.
;   x: S0 = 0x12345678, low nibble 8: -0.5
;   y: S0 = 7: 0.4375
;   z: S0 = 0xF with mul:4: -0.0625 * 4 = -0.25
;   w: S0 = 7 with mul:4 clamp: 1.75 clamped to 1.0 (clamping first would give 1.75)
; Expected mrt0 bits: 0xbf000000, 0x3ee00000, 0xbe800000, 0x3f800000
v_mov_b32_e32 v0, 0x12345678
v_cvt_off_f32_i4_e32 v2, v0
v_mov_b32_e32 v0, 7
v_cvt_off_f32_i4_e32 v3, v0
v_mov_b32_e32 v0, 15
v_cvt_off_f32_i4_e64 v4, v0 mul:4
v_mov_b32_e32 v0, 7
v_cvt_off_f32_i4_e64 v5, v0 clamp mul:4
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
