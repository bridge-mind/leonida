; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 32-bit scalar compare on a zero low half / non-zero high half.
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_half_cmp.spv 1 dump
; Expect: out[0..3] = 1 0 1 1
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0xffff
s_mov_b64 s[6:7], exec
s_cmp_eq_u32 s6, 0
s_cselect_b32 s8, 1, 0
s_cmp_lg_u32 s6, 0
s_cselect_b32 s9, 1, 0
s_cmp_lg_u32 s7, 0
s_cselect_b32 s10, 1, 0
s_cmp_eq_u32 exec_lo, 0
s_cselect_b32 s11, 1, 0
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s11
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
