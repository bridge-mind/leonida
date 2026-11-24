; Wave64 whole-quad mode (s_wqm_b64 / s_wqm_b32) on lane masks and on data. A quad is 4 aligned lanes (4n..4n+3), so it
; never crosses lane 31/32. Derivation (bits are lanes):
;   mask lo = 0x80000240 (lanes 6, 9, 31), hi = 0x00000102 (lanes 33, 40)
;   WQM lo: lane 6 -> quad 1 (4-7) 0xF0, lane 9 -> quad 2 (8-11) 0xF00, lane 31 -> quad 7 (28-31) 0xF0000000 = 0xF0000FF0
;   WQM hi: lane 33 = bit 1 -> quad 0 0xF, lane 40 = bit 8 -> quad 2 0xF00 = 0x00000F0F
;   out0 = s_bcnt1_i32_b64 of the b64 result = 12 + 8 = 20, out1 = s_ff1_i32_b64 = 4 (lane 4)
;   out2 = s_bcnt1_i32_b32 of s_wqm_b32 on the low half = 12, out3 = same on the high half = 8
;   out4 = SCC after s_wqm_b64 of a mask with only lane 40 set, read by lane 0 whose own quad is empty = 1 (wave-wide any)
;   out5 = SCC after s_wqm_b64 of an empty mask (vcc from a v_cmp no lane satisfies) = 0
;   data (not a lane mask): s20 = 0x00800010 (bit 4, bit 23), s21 = 1: WQM = 0x00F000F0 and 0xF -> out6 = s22 + s23 = 15728895
;   out7 = s_bcnt1_i32_b64 after s_wqm_b64 exec, exec (EXEC as destination) = 20
; Assemble (as scripts/run_gpu_tests.py / build.py do): llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding (wave64 is inferred by the translator from exec_hi/_b64 EXEC use)
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_wqm_w64.spv 1 dump
; Expect: out[0..7] = 20 4 12 8 1 0 15728895 20
s_mov_b32 exec_lo, 0x80000240
s_mov_b32 exec_hi, 0x102
s_mov_b64 s[6:7], exec
s_wqm_b64 s[8:9], s[6:7]
s_bcnt1_i32_b64 s10, s[8:9]
s_ff1_i32_b64 s11, s[8:9]
s_mov_b32 s16, exec_lo
s_mov_b32 s17, exec_hi
s_wqm_b32 s18, s16
s_bcnt1_i32_b32 s12, s18
s_wqm_b32 s19, s17
s_bcnt1_i32_b32 s13, s19
s_mov_b32 exec_lo, 0
s_mov_b32 exec_hi, 0x100
s_mov_b64 s[24:25], exec
s_wqm_b64 s[26:27], s[24:25]
s_cselect_b32 s14, 1, 0
v_cmp_lt_u32_e32 vcc_lo, 99, v0
s_wqm_b64 s[30:31], vcc
s_cselect_b32 s15, 1, 0
s_mov_b32 s20, 0x00800010
s_mov_b32 s21, 1
s_wqm_b64 s[22:23], s[20:21]
s_add_u32 s32, s22, s23
s_mov_b32 exec_lo, 0x80000240
s_mov_b32 exec_hi, 0x102
s_wqm_b64 exec, exec
s_bcnt1_i32_b64 s33, exec
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s11
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s12
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s13
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s14
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s15
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s32
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s33
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
