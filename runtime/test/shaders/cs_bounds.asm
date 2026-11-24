; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; num_records bounds checking: s[0:3] has base 0, stride 0, num_records = 16 (bytes 0..15), word3 0x31016fac (OOB_SELECT RAW).
; Thread 0 writes 0x12345678 to offset 0 (in-bounds), attempts write 0xdeadbeef to offset 16 (OOB, ignored),
; loads offset 0 (0x12345678) into offset 4, and loads offset 16 (OOB -> 0) into offset 8.
; Run: vk_compute_test cs_bounds.spv 1 dump -> out[0..3] = 0x12345678 0x12345678 0x00000000 0xeeeeeeee
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_mov_b32 s6, 16
s_mov_b32 s7, 0x31016fac
v_mov_b32 v1, 0x12345678
v_mov_b32 v2, 0
buffer_store_dword v1, v2, s[4:7], 0 offen
v_mov_b32 v3, 0xdeadbeef
v_mov_b32 v4, 16
buffer_store_dword v3, v4, s[4:7], 0 offen
s_waitcnt vmcnt(0)
buffer_load_dword v5, v2, s[4:7], 0 offen
buffer_load_dword v6, v4, s[4:7], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32 v7, 4
v_mov_b32 v8, 8
buffer_store_dword v5, v7, s[0:3], 0 offen
buffer_store_dword v6, v8, s[0:3], 0 offen
s_endpgm
