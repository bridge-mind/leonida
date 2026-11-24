; M1.5 pixel shader: interpolates param0 (a colour), scales it by four floats that sit in the shader's own code
; blob and are read through the code base (s_getpc_b64), and adds a root constant. Run: d3d12_translated_draw_test.
;
; The first instruction is s_getpc_b64, so the address it returns is the blob's code base + 4 (the push-constant
; code base, dwords 34-35, is the console VA of the FSSM program the layer uploaded). The table of four floats is at
; blob offset 0x100: 0x100 - 4 = 0xfc. build.py writes that blob (fssm-at / fssm-data below) next to the SPIR-V, and
; adds its SHA-256 to the index, as the translator's --fssm-dir does for the game's programs.
;   scale = (0.5, 2.0, 1.0, 1.0)      s4 = bias, a float from the root constants (s2, s3 are the VS offset)
;   mrt0  = (r * 0.5 + bias, g * 2.0 + bias, b * 1.0 + bias, a * 1.0)
; The test draws colour (0.25, 0.5, 0.75, 1) with bias 0 and 0.125, and colour (1, 0, 0.5, 1) with bias 0.25. Every
; value is a short binary fraction, so the float render target holds the exact result, bit for bit:
;   (0.25, 0.5, 0.75, 1) bias 0     -> (0.125, 1.0,   0.75,  1)
;   (0.25, 0.5, 0.75, 1) bias 0.125 -> (0.25,  1.125, 0.875, 1)
;   (1.0,  0.0, 0.5,  1) bias 0.25  -> (0.75,  0.25,  0.75,  1)
; The header of vs_m15.asm describes the vertex data; runtime/test/d3d12_translated_draw_test.cpp the three draws.
; fssm-at: 0x100
; fssm-data: 0x3f000000 0x40000000 0x3f800000 0x3f800000
s_getpc_b64 s[6:7]
s_add_u32 s6, s6, 0xfc
s_addc_u32 s7, s7, 0
s_load_dwordx4 s[8:11], s[6:7], 0x0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_interp_p1_f32_e32 v4, v0, attr0.z
v_interp_p2_f32_e32 v4, v1, attr0.z
v_interp_p1_f32_e32 v5, v0, attr0.w
v_interp_p2_f32_e32 v5, v1, attr0.w
s_waitcnt lgkmcnt(0)
v_fma_f32 v2, v2, s8, s4
v_fma_f32 v3, v3, s9, s4
v_fma_f32 v4, v4, s10, s4
v_mul_f32_e32 v5, s11, v5
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
