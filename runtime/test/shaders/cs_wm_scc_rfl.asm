; Wave64 uniform SCC after s_and_saveexec_b64 and v_readfirstlane_b32 broadcast.
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_scc_rfl.spv 1 dump
; Expect: out[0..1] = 55 48
s_mov_b32 s14, 0
v_cmp_gt_u32_e32 vcc_lo, 16, v0
s_and_saveexec_b64 s[6:7], vcc
s_cbranch_scc0 1                ; skip the one-dword s_mov_b32 s14, 55
s_mov_b32 s14, 55
s_mov_b64 exec, -1
v_add_nc_u32_e32 v4, 32, v0
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff
v_readfirstlane_b32 s15, v4
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 32, v0
v_mov_b32_e32 v1, s14
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s15
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
