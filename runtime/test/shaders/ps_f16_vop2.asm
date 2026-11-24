; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; f16 VOP2 ops read the low halves and are partial writes: the high half of the destination is kept. The VOP3
; (_e64) form adds neg/abs. The accumulator of v_fmac_f16 is the low half of its destination.
; v0 = 1.5 | 4.0 << 16, v1 = 0.5 | 2.0 << 16 (low | high half); destinations start as 0xaaaa5555, v5 as 1.0 | 3.0 << 16.
;   v_add_f16 1.5 + 0.5 = 2.0    v_sub_f16_e64 -v0, |v1|: -1.5 - 0.5 = -2.0    v_mul_f16 1.5 * 0.5 = 0.75
;   v_fmac_f16 1.5 * 0.5 + 1.0 = 1.75
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa4000, 0xaaaac000, 0xaaaa3a00, 0x42003f00
v_mov_b32 v0, 0x44003e00
v_mov_b32 v1, 0x40003800
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0x42003c00
v_add_f16 v2, v0, v1
v_sub_f16_e64 v3, -v0, |v1|
v_mul_f16 v4, v0, v1
v_fmac_f16 v5, v0, v1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
