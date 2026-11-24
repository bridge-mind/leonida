; group: 1,1,1
; v_div_fmas_f64 with VCC = 0 is a binary64 multiply-add with ONE rounding (RDNA2 ISA, VOP3). Low word first.
; a = 1 + 2^-30 (0x3ff00000_00400000), c = -(1 + 2^-29) (0xbff00000_00800000): a*a = 1 + 2^-29 + 2^-60 exactly, so
; fma(a, a, c) = 2^-60 = 0x3c300000_00000000 fused; rounding a*a first gives 1 + 2^-29 and a result of 0 ("split").
; out[0..1] = 0x00000000 0x3c300000 (split: 0x00000000 0x00000000). Under --fma khr (LEONIDA_FMA=khr) the translator
; emits OpFmaKHR on doubles (feature shaderFmaFloat64) and only the fused words pass; under the default GLSL Fma a
; driver may split (NVIDIA RTX 4090, 621.24, does).
s_mov_b64 vcc, 0
v_mov_b32 v0, 0x00400000
v_mov_b32 v1, 0x3ff00000
v_mov_b32 v2, 0x00800000
v_mov_b32 v3, 0xbff00000
v_div_fmas_f64 v[4:5], v[0:1], v[0:1], v[2:3]
v_mov_b32 v6, 0
buffer_store_dwordx2 v[4:5], v6, s[0:3], 0 offen
s_endpgm
