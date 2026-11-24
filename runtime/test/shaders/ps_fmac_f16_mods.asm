; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; VOP3 output modifiers on v_fmac_f16_e64 (docs/design.md "v_fmac_f16, v_interp_p2/mov_f32 output modifiers"):
; On RDNA2 with denorms kept (default MODE), omod (mul:2 / mul:4 / div:2) is ignored on f16;
; clamp to [0, 1] applies; the destination's high half (0xAAAA) is kept.
;   x: 1.0 + 1.0 * 1.0 = 2.0, clamp: 1.0 (0x3c00)
;   y: 0.5 + 0.5 * 0.5 = 0.75, mul:2 ignored: 0.75 (0x3a00)
;   z: -1.0 + 0.5 * 0.5 = -0.75, clamp: 0.0 (0x0000)
;   w: 2.0 + 1.0 * 2.0 = 4.0, div:2 ignored: 4.0 (0x4400)
; Expected mrt0 bits: 0xaaaa3c00, 0xaaaa3a00, 0xaaaa0000, 0xaaaa4400
v_mov_b32_e32 v2, 0xaaaa3c00
v_fmac_f16_e64 v2, 1.0, 1.0 clamp
v_mov_b32_e32 v3, 0xaaaa3800
v_fmac_f16_e64 v3, 0.5, 0.5 mul:2
v_mov_b32_e32 v4, 0xaaaabc00
v_fmac_f16_e64 v4, 0.5, 0.5 clamp
v_mov_b32_e32 v5, 0xaaaa4000
v_fmac_f16_e64 v5, 1.0, 2.0 div:2
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
