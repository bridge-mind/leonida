; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_write_b64 / ds_write_b96 / ds_write_b128, ds_read_b64 / ds_read_b96 / ds_read_b128 and ds_read2st64_b32
; (RDNA2 ISA, "Data Share Instructions"). ADDR = vaddr + offset: (bytes); an N-dword access covers
; MEM[ADDR .. ADDR + 4N - 1], lowest register first. ds_read2st64_b32 reads MEM[ADDR + offset0 * 256] and
; MEM[ADDR + offset1 * 256] (offsets count 64 dwords).
; v[30:33] = A B C D, v[34:35] = E F (values below). Writes:
;   b128 v[30:33] at byte 16, b32 0 at byte 60 then b96 v[30:32] at byte 48 (byte 60 stays 0),
;   b64 v[34:35] at byte 64, b32 G at byte 0 and b32 H at byte 256 (LDS is 512 B, the default header).
;   out[0] b128 at 16, first reg  -> A 0xCAFEF00D   out[1] b128 at 16, last reg  -> D 0xFEEDFACE
;   out[2] b96 at 48, last reg    -> C 0x600DD00D   out[3] b32 at 60 (after b96) -> 0x00000000
;   out[4] b64 at 64, low reg     -> E 0x8BADF00D   out[5] b64 at 64, high reg   -> F 0xDEADC0DE
;   out[6] read2st64 offset0:0    -> G 0x12345678   out[7] read2st64 offset1:1   -> H 0x9ABCDEF0
; Every lane writes the same dwords, so the result does not depend on lane order.
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_wide.spv 1 dump
;   -> out[0..7] = 0xCAFEF00D 0xFEEDFACE 0x600DD00D 0x00000000 0x8BADF00D 0xDEADC0DE 0x12345678 0x9ABCDEF0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v30, 0xcafef00d
v_mov_b32_e32 v31, 0x0badbeef
v_mov_b32_e32 v32, 0x600dd00d
v_mov_b32_e32 v33, 0xfeedface
v_mov_b32_e32 v34, 0x8badf00d
v_mov_b32_e32 v35, 0xdeadc0de
v_mov_b32_e32 v4, 0x12345678
v_mov_b32_e32 v5, 0x9abcdef0
ds_write_b128 v2, v[30:33] offset:16
ds_write_b32 v2, v2 offset:60
s_waitcnt lgkmcnt(0)
ds_write_b96 v2, v[30:32] offset:48
ds_write_b64 v2, v[34:35] offset:64
ds_write_b32 v2, v4
ds_write_b32 v2, v5 offset:256
s_waitcnt lgkmcnt(0)
ds_read_b128 v[40:43], v2 offset:16
ds_read_b96 v[44:46], v2 offset:48
ds_read_b32 v47, v2 offset:60
ds_read_b64 v[48:49], v2 offset:64
ds_read2st64_b32 v[50:51], v2 offset1:1
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v40, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v43, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v46, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v47, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v48, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v49, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v50, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v51, v21, s[0:3], 0 offen
s_endpgm
