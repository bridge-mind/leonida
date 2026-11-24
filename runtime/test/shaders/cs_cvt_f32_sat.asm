; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s[0:3] is the output V#, s4 is one binary32 input as a raw word.
; RDNA 2 ISA, VOP1 opcodes 7 and 8: truncate toward zero, saturate to
; the destination range, and return zero for NaN.
; At offset 0: unsigned result. At offset 4: signed result, as raw bits.
; Example s4 = 0x4f000000 (+2^31): out = 0x80000000, 0x7fffffff.
; group: 1,1,1
; rsrc2: 0x808a
v_mov_b32_e32 v0, s4
v_mov_b32_e32 v1, 0
v_cvt_u32_f32_e32 v2, v0
v_cvt_i32_f32_e32 v3, v0
buffer_store_dwordx2 v[2:3], v1, s[0:3], 0 offen
s_endpgm
