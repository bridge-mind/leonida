; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_pack_b32_f16 with NEG and ABS on a source: each acts on the selected 16-bit half as a plain bit move
; (ABS clears bit 15, NEG flips it, ABS first), before the half is packed. The rule and its words were measured
; on a Radeon 680M (gfx1036, RDNA2 iGPU, AMD Windows driver 32.0.21044.64) with raw kernels: -v0 turned 0x7c01
; into 0xfe01 under IEEE mode 1 and 0xfc01 under mode 0, in either half and with op_sel (#1548 SHADER-260).
; Quieting an sNaN half under IEEE mode 1 is not modelled (SHADER-190); the values below are not NaNs.
; v0 = 0.25 | 3.0 << 16, v1 = 4.0 | 0.25 << 16 (low | high half).
;   -v0, |v1|: -0x3400 | 0x4400 << 16
;   v0, -v1: 0x3400 | -0x4400 << 16
;   -|v0|, v1 op_sel:[1,0,0]: -(0x4200) | 0x4400 << 16
;   -v0, -|v1| op_sel:[0,1,0]: -0x3400 | -(0x3400) << 16
; Expected mrt0 bits: 0x4400b400, 0xc4003400, 0x4400c200, 0xb400b400
v_mov_b32 v0, 0x42003400
v_mov_b32 v1, 0x34004400
v_pack_b32_f16 v2, -v0, |v1|
v_pack_b32_f16 v3, v0, -v1
v_pack_b32_f16 v4, -|v0|, v1 op_sel:[1,0,0]
v_pack_b32_f16 v5, -v0, -|v1| op_sel:[0,1,0]
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
