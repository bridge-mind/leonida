; More f16 VOP2 ops: low halves in, high half of the destination kept; neg/abs and clamp in the VOP3 (_e64) form.
; v0 = 1.5 | 4.0 << 16, v1 = 0.5 | 2.0 << 16 (low | high half); destinations start as 0xaaaa5555, v5 as 1.0 | 3.0 << 16.
;   v_max_f16 max(1.5, 0.5) = 1.5    v_min_f16_e64 v0, -v1: min(1.5, -0.5) = -0.5
;   v_add_f16_e64 clamp: 2.0 clamped to 1.0    v_fmac_f16_e64 -v0, |v1|: -1.5 * 0.5 + 1.0 = 0.25
; The words are the ones an RX 6800 (gfx1030) gave for this body run as a raw kernel, wave32 and wave64.
; Expected mrt0 bits: 0xaaaa3e00, 0xaaaab800, 0xaaaa3c00, 0x42003400
v_mov_b32 v0, 0x44003e00
v_mov_b32 v1, 0x40003800
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v3, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v5, 0x42003c00
v_max_f16 v2, v0, v1
v_min_f16_e64 v3, v0, -v1
v_add_f16_e64 v4, v0, v1 clamp
v_fmac_f16_e64 v5, -v0, |v1|
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
