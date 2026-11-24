; s[0:3] is the output V#. One thread, eight words: each is an f32 multiply-add whose single-rounded (fused) result
; differs from a multiply followed by an add. RDNA 2 v_fma_f32, v_fmac_f32, v_fmaak_f32 and v_fmamk_f32 round once,
; so every word below is the exact fused result (derived by hand in the comments; measured on gfx1150 by
; tools/f32_fuzz.py --oracle hw, 72 of 72 cancel probes fused). A translation whose multiply-add rounds twice
; (a driver may split GLSL.std.450 Fma: the Vulkan environment says "inherited from OpFMul followed by OpFAdd")
; gives the "split" value instead. rdna2spirv.py --fma khr emits OpFmaKHR (SPV_KHR_fma, correctly rounded).
;
; a = 1 + 2^-12 (0x3f800800), a*a = 1 + 2^-11 + 2^-24 exactly. That is a tie between 1 + 2^-11 and
; 1 + 2^-11 + 2^-23; ties-to-even picks p = 1 + 2^-11 (0x3f801000), so a*a - p = 2^-24 is the product's rounding error.
;   out[0] v_fma_f32 a, a, -p           2^-24         = 0x33800000   split 0x00000000
;   out[1] v_fma_f32 (1+2^-23), (1-2^-23), -1:
;          the product is 1 - 2^-46      -2^-46        = 0xa8800000   split 0x00000000
;   out[2] v_fma_f32 a, a, -1.0          2^-11 + 2^-24 = 0x3a000400   split 0x3a000000
;   out[3] v_fmaak_f32 a, a, K = -p      2^-24         = 0x33800000   split 0x00000000
;   out[4] v_fmamk_f32 a * K + (-p), K = a  2^-24      = 0x33800000   split 0x00000000
;   out[5] v_fmac_f32 d = -p; d += a*a   2^-24         = 0x33800000   split 0x00000000
;   out[6] v_fma_f32 (1+2^-23)^2, -(1+2^-22): (1+2^-23)^2 = 1 + 2^-22 + 2^-46, so 2^-46 = 0x28800000  split 0x00000000
;   out[7] v_fma_f16 0x6ce1 * 0x4200 + 0x8004 (the f16 triple of tests/semantic/test_f16_fma_rounding.py where
;          rounding the exact result to f32 and then to f16 is one step off): 0x00007351; f32-then-f16 gives 0x7352.
;          The f16 form does not use Fma at all (round-to-odd, Gen.fma_f16_odd), so it must be the same with
;          --fma glsl and --fma khr.
; The operands come in through s4..s7 (VKC_SGPRS=0x3f800800,0x3f800001,0x3f7ffffe,0xbf800002: a, 1+2^-23, 1-2^-23,
; -(1+2^-22)), so a driver cannot fold the multiply-adds at compile time; only the fmaak/fmamk constants are literals.
; Run: VKC_SGPRS=0x3f800800,0x3f800001,0x3f7ffffe,0xbf800002 vk_compute_test cs_fma_fused.spv 1 dump
; out[0..7] = 33800000 a8800000 3a000400 33800000 33800000 33800000 28800000 00007351 (hex)
; group: 1,1,1
; rsrc2: 0x8090
v_mov_b32_e32 v0, s4
v_mul_f32_e32 v3, v0, v0
v_mov_b32_e32 v20, 0
v_fma_f32 v4, v0, v0, -v3
buffer_store_dword v4, v20, s[0:3], 0 offen
v_mov_b32_e32 v5, s5
v_mov_b32_e32 v6, s6
v_fma_f32 v7, v5, v6, -1.0
buffer_store_dword v7, v20, s[0:3], 0 offen offset:4
v_fma_f32 v8, v0, v0, -1.0
buffer_store_dword v8, v20, s[0:3], 0 offen offset:8
v_fmaak_f32 v9, v0, v0, 0xbf801000
buffer_store_dword v9, v20, s[0:3], 0 offen offset:12
v_sub_f32_e32 v10, 0, v3
v_fmamk_f32 v11, v0, 0x3f800800, v10
buffer_store_dword v11, v20, s[0:3], 0 offen offset:16
v_mov_b32_e32 v12, v10
v_fmac_f32_e32 v12, v0, v0
buffer_store_dword v12, v20, s[0:3], 0 offen offset:20
v_mov_b32_e32 v13, s7
v_fma_f32 v14, v5, v5, v13
buffer_store_dword v14, v20, s[0:3], 0 offen offset:24
v_mov_b32_e32 v15, 0x6ce1
v_mov_b32_e32 v16, 0x4200
v_mov_b32_e32 v17, 0x8004
v_fma_f16 v18, v15, v16, v17
v_and_b32_e32 v18, 0xffff, v18
buffer_store_dword v18, v20, s[0:3], 0 offen offset:28
s_endpgm
