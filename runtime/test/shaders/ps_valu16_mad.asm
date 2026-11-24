; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_mad_u32_u16 / v_mad_i32_i16 (VOP3-only on GFX10): D = S0.half * S1.half + S2, wrapping at 32 bits. The halves
; are zero-extended (u16) or sign-extended (i16); op_sel bit 0 / 1 picks the high half of S0 / S1. S2 and D are whole
; dwords, so D is written in full: the destinations start as 0xaaaa5555 and none of that may survive. The source
; high halves hold other values, so a read of the wrong half shows up. RDNA2 ISA pseudocode:
;   v_mad_u32_u16: D.u32 = S0.u16 * S1.u16 + S2.u32      v_mad_i32_i16: D.i32 = S0.i16 * S1.i16 + S2.i32
; v0 = 0x1234ffff (low 0xffff = -1 as i16), v1 = 0x5678fffe (low 0xfffe = -2), v3 = 0x00030000
; v5 = 0x00018000 (low 0x8000 = -32768, high 0x0001), v6 = 0xffff7fff (low 0x7fff = 32767, high 0xffff = -1)
; v7 = 0x40000000
;   x: v_mad_u32_u16 v2, v0, v1, v3: 0xffff * 0xfffe = 0xfffd0002; + 0x00030000 = 0x1_00000002, wraps to 0x00000002
;   y: v_mad_i32_i16 v4, v0, v1, v3: (-1) * (-2) = 2; + 0x00030000 = 0x00030002 (zero-extending would give 0x00000002)
;   z: v_mad_i32_i16 v8, v5, v6, v7: -32768 * 32767 = -1073709056 = 0xc0008000; + 0x40000000 = 0x00008000
;      (zero-extending: 0x3fff8000 + 0x40000000 = 0x7fff8000)
;   w: v_mad_i32_i16 v9, v5, v6, 5 op_sel:[1,1,0,0]: high halves 1 * (-1) = -1; + 5 = 0x00000004
;      (zero-extended high halves: 0xffff + 5 = 0x00010004; low halves: 0xc0008005)
; Expected mrt0 bits: 0x00000002, 0x00030002, 0x00008000, 0x00000004
v_mov_b32 v0, 0x1234ffff
v_mov_b32 v1, 0x5678fffe
v_mov_b32 v3, 0x00030000
v_mov_b32 v5, 0x00018000
v_mov_b32 v6, 0xffff7fff
v_mov_b32 v7, 0x40000000
v_mov_b32 v2, 0xaaaa5555
v_mov_b32 v4, 0xaaaa5555
v_mov_b32 v8, 0xaaaa5555
v_mov_b32 v9, 0xaaaa5555
v_mad_u32_u16 v2, v0, v1, v3
v_mad_i32_i16 v4, v0, v1, v3
v_mad_i32_i16 v8, v5, v6, v7
v_mad_i32_i16 v9, v5, v6, 5 op_sel:[1,1,0,0]
exp mrt0 v2, v4, v8, v9 done vm
s_endpgm
