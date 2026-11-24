; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-328: an s_buffer_load SGPR offset the translator can see as a constant gives what the same offset gives at run
; time. The rule (docs/design.md, Scalar/buffer memory): off = zext(soffset & ~3) + (imm & ~3) in 64 bits, and the
; dword is 0 when off + 4 > limit. #1686 folded a constant soffset as a 32-bit sum, so 0xfffffffc + 20 wrapped to 16
; and read data; gfx1036 (tools/hw_oracle, #1686) reads 0.
; Thread 0 seeds heap bytes 0x40..0x5f (region offset X = byte 0x40 + X) with 0x5eed0000 + X and byte 0x60 with
; 4, then loads through a V# it builds itself (base 0x40, stride 0, NUM_RECORDS 0x40, word3 0x31016fac):
;   s_buffer_load_dword s24, s8=-4 offset:20 (constant)        -> 0xfffffffc + 0x14 > 0x40: 0   (32-bit wrap: 0x5eed0010)
;   s_buffer_load_dword s25, s9=4 offset:20 (from heap 0x60)      -> run time, in range: 0x5eed0018, as s26
;   s_buffer_load_dword s26, s10=4 offset:20 (constant)        -> 0x18: 0x5eed0018
;   s_buffer_load_dword s27, s11=0x10 offset:4 (constant)      -> 0x14: 0x5eed0014
; and writes s24..s27 to out[0..3]. The run-time load stays in range: an out-of-range run-time sOFF (-4) made llvmpipe
; (Mesa 26.0.8) do the guarded load anyway and segfault (#1938 review); the CPU tests cover that path.
; Run: vk_compute_test cs_smem_sconst_off.spv 1 dump -> out[0..3] = 0x00000000 0x5eed0018 0x5eed0018 0x5eed0014.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0000
v_mov_b32 v2, 0x5eed0004
v_mov_b32 v3, 0x5eed0008
v_mov_b32 v4, 0x5eed000c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
v_mov_b32 v1, 0x5eed0010
v_mov_b32 v2, 0x5eed0014
v_mov_b32 v3, 0x5eed0018
v_mov_b32 v4, 0x5eed001c
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:80
v_mov_b32 v5, 4
buffer_store_dword v5, off, s[0:3], 0 offset:96
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 0x40
s_mov_b32 s7, 0x31016fac
s_mov_b32 s8, -4
s_buffer_load_dword s9, s[0:3], 0x60
s_mov_b32 s10, 4
s_mov_b32 s11, 0x10
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s24, s[4:7], s8 offset:0x14
s_buffer_load_dword s25, s[4:7], s9 offset:0x14
s_buffer_load_dword s26, s[4:7], s10 offset:0x14
s_buffer_load_dword s27, s[4:7], s11 offset:0x4
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, s24
v_mov_b32 v2, s25
v_mov_b32 v3, s26
v_mov_b32 v4, s27
buffer_store_dwordx4 v[1:4], off, s[0:3], 0
s_endpgm
