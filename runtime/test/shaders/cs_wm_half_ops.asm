; Wave64 half-mask bcnt1/ff1 and cross-BB mask survival.
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wm_half_ops.spv 1 dump
; Expect: out[0..4] = 32 16 16 0 16
s_mov_b32 exec_lo, 0xffff0000
s_mov_b32 exec_hi, 0xffff
s_mov_b64 s[6:7], exec
s_branch 0                       ; to the next instruction: starts a new basic block
s_bcnt1_i32_b64 s8, s[6:7]
s_ff1_i32_b64 s9, s[6:7]
s_bcnt1_i32_b32 s10, exec_lo
s_ff1_i32_b32 s11, exec_hi
s_bcnt1_i32_b32 s12, s7
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
v_mov_b32_e32 v1, s12
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
