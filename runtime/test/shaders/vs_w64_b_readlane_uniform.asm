; TEST-251 (SHADER-10 tier B, vertex stage): v_readfirstlane and v_readlane (lanes 40 and 63) of wave-uniform VGPRs
; in a wave64 vertex shader, the vertex twin of ps_w64_b_readfirstlane and ps_w64_b_readlane_uniform. s4 is a user
; SGPR (VKH_SGPRS sets it to 123), copied to a VGPR and read back; so is the constant 77. Every lane holds the same
; value, so the scalar does not depend on which lane is read, on the wave size or on how many vertices a wave got.
; Vertex v (v0 = vertex index) has x = v & 63. r0 = x + 123 (readfirstlane), r1 = 123 + 77 (readlane 40 and 63),
; r2 = x + 123 (readlane 40), r3 = 77 * 2 + 123 (readfirstlane), stored to its record at heap 0x1000 + 16 * v
; through the V# in s[0:3].
; Run: VKH_VERTICES=960 VKH_SGPRS=0,0,65536,0x30000000,123 VKH_HEAP=0=0 VKH_HEAP_DUMP=heap.raw vk_harness
; vs_w64_b_readlane_uniform.spv <any fragment shader> out.raw, with and without RADV_PERFTEST=gewave32: equal, and
; equal to the expected words.
; mattr: +wavefrontsize64,-wavefrontsize32
v_and_b32_e32 v2, 63, v0
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v5, s4
v_mov_b32_e32 v4, 77
v_readfirstlane_b32 s20, v5
v_add_nc_u32_e32 v6, s20, v2
v_readlane_b32 s21, v5, 40
v_readlane_b32 s22, v4, 63
s_add_u32 s23, s21, s22
v_mov_b32_e32 v7, s23
v_add_nc_u32_e32 v8, s21, v2
v_readfirstlane_b32 s24, v4
s_lshl_b32 s24, s24, 1
s_add_u32 s24, s24, s20
v_mov_b32_e32 v9, s24
v_lshlrev_b32_e32 v12, 4, v0
v_add_nc_u32_e32 v12, 0x1000, v12
buffer_store_dwordx4 v[6:9], v12, s[0:3], 0 offen
v_mov_b32_e32 v5, 0
exp pos0 v5, v5, v5, v5 done
exp param0 v5, v5, v5, v5
s_endpgm
