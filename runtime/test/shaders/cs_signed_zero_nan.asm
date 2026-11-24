; group: 8,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; NaN, infinity and -0 come out of RDNA2 float ops as computed (IEEE). The translator declares
; SignedZeroInfNanPreserve so the driver may not assume they don't occur. Without it, on an RX 9070 XT (AMD
; 26.11.2) this test gets 0 in out[4..7]: the x != x NaN checks fold to false and the NaN conversion is lost. The
; f16 fuzzer there also saw -0 products come back +0, and NaN results narrowed with integer ops never stored.
; Apple GPUs (MoltenVK) return +0 in out[0], out[2] and out[3] with or without the mode (#1203, SHADER-182), so
; check.py and run_gpu_tests.py don't judge those three words on an Apple device; the other five still count.
; Mesa llvmpipe returns 0 in out[5] (the 0 * inf check) with or without the mode, while its other NaN checks give 1
; (#664), so out[5] isn't judged there; the other seven still count (UNJUDGED_WORDS in check.py).
; Thread 0 stores eight words:
;   out[0] v_mul_f32 -0.0 * 1.0                             = 0x80000000
;   out[1] v_add_f32 -0.0 + -0.0                            = 0x80000000
;   out[2] v_pk_mul_f16 (-0, -0) * (1.0, 1.0)               = 0x80008000
;   out[3] v_mul_f16 -0 * 1.0, high half kept (0)           = 0x00008000
;   out[4] v_cmp_neq_f32 x, x with x = inf - inf (NaN)      = 0x00000001
;   out[5] v_cmp_neq_f32 x, x with x = 0 * inf (NaN)        = 0x00000001
;   out[6] v_cmp_neq_f16 x, x with x = f16 inf - inf (NaN)  = 0x00000001
;   out[7] v_cvt_f16_f32 of NaN, exponent bits (& 0x7c00)   = 0x00007c00
; Run: vk_compute_test cs_signed_zero_nan.spv 1 dump
; out[0..7] = 80000000 80000000 80008000 00008000 00000001 00000001 00000001 00007c00 (hex)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x80000000
v_mov_b32_e32 v2, 1.0
v_mul_f32_e32 v10, v1, v2
v_add_f32_e32 v11, v1, v1
v_mov_b32_e32 v3, 0x80008000
v_mov_b32_e32 v4, 0x3c003c00
v_pk_mul_f16 v12, v3, v4
v_mov_b32_e32 v13, 0
v_mul_f16_e32 v13, v3, v4
v_mov_b32_e32 v5, 0x7f800000
v_sub_f32_e32 v6, v5, v5
v_cmp_neq_f32_e32 vcc_lo, v6, v6
v_cndmask_b32_e64 v14, 0, 1, vcc_lo
v_mov_b32_e32 v7, 0
v_mul_f32_e32 v8, v7, v5
v_cmp_neq_f32_e32 vcc_lo, v8, v8
v_cndmask_b32_e64 v15, 0, 1, vcc_lo
v_mov_b32_e32 v9, 0x7c00
v_sub_f16_e32 v16, v9, v9
v_cmp_neq_f16_e32 vcc_lo, v16, v16
v_cndmask_b32_e64 v17, 0, 1, vcc_lo
v_mov_b32_e32 v18, 0
v_cvt_f16_f32_e32 v18, v6
v_and_b32_e32 v18, 0x7c00, v18
v_mov_b32_e32 v30, 0
buffer_store_dword v10, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 4
buffer_store_dword v11, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 8
buffer_store_dword v12, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 12
buffer_store_dword v13, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 16
buffer_store_dword v14, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 20
buffer_store_dword v15, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 24
buffer_store_dword v17, v30, s[0:3], 0 offen
v_mov_b32_e32 v30, 28
buffer_store_dword v18, v30, s[0:3], 0 offen
s_endpgm
