; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_buffer_load_dwordx2 into VCC: the low dword lands in vcc_lo, the high dword in vcc_hi.
; The V# in s[0:3] points at 0x1000; the test seeds memory with 0x11223344 and 0x55667788 there.
; Expected mrt0 bits: 0x11223344, 0x55667788, 0, 0
s_buffer_load_dwordx2 vcc, s[0:3], 0
s_waitcnt vmcnt(0)
v_mov_b32_e32 v1, vcc_lo
v_mov_b32_e32 v2, vcc_hi
v_mov_b32_e32 v3, 0
exp mrt0 v1, v2, v3, v3 done vm
s_endpgm
