; M3 pixel shader, pass 1: two render targets from root constants. Run: d3d12_m3_frame_test.
; User data (the root signature of vs_m3.asm, [0], eight 32-bit constants): s5, s6, s7 = c0, c1, c2 (floats, k/255).
;   mrt0 = (c0, c1, c2, 1.0)   an R8G8B8A8_UNORM target
;   mrt1 = (c1 + c2)           an R32_FLOAT target: one float add, so the CPU gets the same bits
; fssm-at: 0x10
; fssm-data: 0x33535350 0x20535020
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, 1.0
v_add_f32_e32 v6, s6, v4
exp mrt0 v2, v3, v4, v5
exp mrt1 v6, off, off, off done vm
s_endpgm
