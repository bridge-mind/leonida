; Wave64 s_wqm_b32 on one half of EXEC: it changes only its own 32 lanes and SCC is the any over that half only.
; Bits are lanes. exec_lo = 0x80000240 (lanes 6, 9, 31), exec_hi = 0x00000102 (lanes 33, 40).
;   WQM of lo = 0xF0000FF0 (12 lanes, see cs_wqm_w64.asm), WQM of hi = 0x00000F0F (8 lanes).
;   Case 1: s_wqm_b32 exec_hi, exec_hi -> exec = lo unchanged (3 lanes) + hi 8 lanes:
;     out0 = s_bcnt1_i32_b64 exec = 11, out4 = s_ff1_i32_b64 exec = 6 (lane 6, lo untouched), out5 = s_bcnt1_i32_b32 exec_hi = 8
;   Case 2: exec as at the start, s_wqm_b32 exec_lo, exec_lo -> lo 12 lanes + hi unchanged (2 lanes):
;     out1 = s_bcnt1_i32_b64 exec = 14, out6 = s_bcnt1_i32_b32 exec_lo = 12, out7 = s_bcnt1_i32_b32 exec_hi = 2
;   SCC (read right after the instruction), exec_lo = 0 and exec_hi = 0x100 (lane 40 only):
;     out2 = SCC after s_wqm_b32 of exec_lo (the low half is empty; a wave-wide any would say 1) = 0
;     out3 = SCC after s_wqm_b32 of exec_hi = 1
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wqm_w64_half.spv 1 dump
; Expect: out[0..7] = 11 14 0 1 6 8 12 2
s_mov_b32 exec_lo, 0x80000240
s_mov_b32 exec_hi, 0x102
s_wqm_b32 exec_hi, exec_hi
s_bcnt1_i32_b64 s8, exec
s_ff1_i32_b64 s13, exec
s_bcnt1_i32_b32 s14, exec_hi
s_mov_b32 exec_lo, 0x80000240
s_mov_b32 exec_hi, 0x102
s_wqm_b32 exec_lo, exec_lo
s_bcnt1_i32_b64 s9, exec
s_bcnt1_i32_b32 s15, exec_lo
s_bcnt1_i32_b32 s16, exec_hi
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0x100
s_wqm_b32 s11, exec_lo
s_cselect_b32 s10, 1, 0
s_wqm_b32 s12, exec_hi
s_cselect_b32 s17, 1, 0
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
v_mov_b32_e32 v1, s17
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s13
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s14
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s15
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s16
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
