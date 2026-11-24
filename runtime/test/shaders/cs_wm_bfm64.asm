; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 s_bfm_b64 lane-mask tracking (low half and high half bitfields).
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_bfm64.spv 1 dump
; Expect: out[0..3] = 16 8 8 40
s_mov_b64 exec, -1
s_bfm_b64 s[6:7], 16, 8
s_bcnt1_i32_b64 s8, s[6:7]
s_ff1_i32_b64 s9, s[6:7]
s_bfm_b64 s[10:11], 8, 40
s_bcnt1_i32_b64 s12, s[10:11]
s_ff1_i32_b64 s13, s[10:11]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s12
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s13
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
