; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_subrev_f16 (VOP2 0x34 on GFX10): D.f16 = S1.f16 - S0.f16. Like the other f16 VOP2 ops it is a partial write
; (docs/design.md, "16-bit results", from Mesa ACO): the low halves are read, the result goes to the low half and the
; high half of the destination is kept, in the VOP2 (_e32) and the VOP3 (_e64) form. neg/clamp in the _e64 form.
; v0 = 1.5 | 4.0 << 16 (0x44003e00), v1 = 0.5 | 2.0 << 16 (0x40003800); v2..v4 start as 0xaaaa5555,
; v5 as 1.0 | 3.0 << 16 (0x42003c00).
;   x: v_subrev_f16 v2, v0, v1:               0.5 - 1.5 = -1.0 = 0xbc00 -> 0xaaaabc00 (v_sub order would give 0x3c00)
;   y: v_subrev_f16_e64 v3, v0, -v1:          -0.5 - 1.5 = -2.0 = 0xc000 -> 0xaaaac000
;   z: v_subrev_f16_e64 v4, -v0, v1 clamp:    0.5 - (-1.5) = 2.0, clamped to 1.0 = 0x3c00 -> 0xaaaa3c00
;   w: v_subrev_f16 v5, v5, v0 (dst = src0):  1.5 - 1.0 = 0.5 = 0x3800, high half 3.0 kept -> 0x42003800
; Expected mrt0 bits: 0xaaaabc00, 0xaaaac000, 0xaaaa3c00, 0x42003800
v_mov_b32 v0, 0x44003e00
v_mov_b32 v1, 0x40003800
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0x42003c00
v_subrev_f16 v2, v0, v1
v_subrev_f16_e64 v3, v0, -v1
v_subrev_f16_e64 v4, -v0, v1 clamp
v_subrev_f16 v5, v5, v0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
