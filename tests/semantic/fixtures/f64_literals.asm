; Each f64 literal supplies bits 63:32. The low word is zero.
; The first three conversions produce f32 +1.5, -infinity, and +0.
v_cvt_f32_f64_e32 v0, 0x3ff80000
v_cvt_f32_f64_e64 v1, 0xfff00000
v_cvt_f32_f64_e32 v2, 64
; LLVM disassembles a literal word of 1 as 0x1 and an inline integer as 1.
; The addends below are 0x0000000100000000 and 0x0000000000000001.
v_add_f64 v[4:5], 0x1, 1 ; encoding: [0x04,0x00,0x64,0xd5,0xff,0x02,0x01,0x00,0x01,0x00,0x00,0x00]
; The inline float is binary64 1.0. The literal is -1.5, giving -0.5.
v_add_f64 v[6:7], 1.0, 0xbff80000
; The inline integer -1 is sign-extended to all 64 bits.
v_cvt_f32_f64_e32 v3, -1
