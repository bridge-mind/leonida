; group: 1,1,1
; rsrc2: 0x8090
; Eight user SGPRs: s[0:3] are the output V#, s4..s7 are input words seeded by vk_compute_test's VKC_SGPRS (both
; runners pass 0x4f000000,0xdeadbeef,7,0xffffffff). One thread stores them back, so a compute test can read an
; input word (#300's cs_cvt_f32_sat takes its float in s4). Without VKC_SGPRS they read 0.
; Run: VKC_SGPRS=0x4f000000,0xdeadbeef,7,0xffffffff vk_compute_test cs_user_sgprs.spv 1 dump
; out[0..7] = 4F000000 DEADBEEF 00000007 FFFFFFFF EEEEEEEE EEEEEEEE EEEEEEEE EEEEEEEE (hex)
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, s4
v_mov_b32_e32 v3, s5
v_mov_b32_e32 v4, s6
v_mov_b32_e32 v5, s7
buffer_store_dwordx4 v[2:5], v1, s[0:3], 0 offen
s_endpgm
