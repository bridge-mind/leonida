; Wave32 whole-quad mode (s_wqm_b32) on EXEC, on VCC, on data. A quad is 4 aligned lanes. Derivation (bits are lanes):
;   exec_lo = 0x80000240 (lanes 6, 9, 31): WQM = quad 1 0xF0 | quad 2 0xF00 | quad 7 0xF0000000 = 0xF0000FF0
;   out0 = s_bcnt1_i32_b32 exec_lo after s_wqm_b32 exec_lo, exec_lo (EXEC as destination) = 12
;   out1 = SCC after it (cselect) = 1
;   out2 = VCC = lane 5 only (v_cmp_eq) -> s_wqm_b32 vcc_lo, vcc_lo -> quad 1 (lanes 4-7) -> s_bcnt1_i32_b32 = 4
;   out3 = SCC after s_wqm_b32 of an empty mask (exec_lo = 0) = 0
;   out4 = exec_lo = 0x18 (lanes 3 and 4, two different quads, neighbours across the quad edge): WQM = 0xFF -> bcnt 8
;   out5 = data (not a mask): s20 = 0x00800010 (bit 4, bit 23) -> 0x00F000F0 = 15728880
;   out6 = s_ff1_i32_b32 of the exec result of out0 (lowest lane of 0xF0000FF0) = 4
;   out7 = s_wqm_b32 of an all-ones mask -> s_bcnt1_i32_b32 = 32
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave32: only exec_lo and 32-bit operations)
; Run: vk_compute_test cs_wqm_w32.spv 1 dump
; Expect: out[0..7] = 12 1 4 0 8 15728880 4 32
s_mov_b32 exec_lo, 0x80000240
s_wqm_b32 exec_lo, exec_lo
s_cselect_b32 s9, 1, 0
s_bcnt1_i32_b32 s8, exec_lo
s_ff1_i32_b32 s14, exec_lo
s_mov_b32 exec_lo, -1
v_cmp_eq_u32_e32 vcc_lo, 5, v0
s_wqm_b32 vcc_lo, vcc_lo
s_bcnt1_i32_b32 s10, vcc_lo
s_mov_b32 exec_lo, 0
s_wqm_b32 s11, exec_lo
s_cselect_b32 s12, 1, 0
s_mov_b32 exec_lo, 0x18
s_wqm_b32 s16, exec_lo
s_bcnt1_i32_b32 s13, s16
s_mov_b32 s20, 0x00800010
s_wqm_b32 s21, s20
s_mov_b32 exec_lo, -1
s_wqm_b32 s17, exec_lo
s_bcnt1_i32_b32 s15, s17
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
v_mov_b32_e32 v1, s12
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s13
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s21
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s14
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s15
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
