; Mandelbrot set: a divergent backward loop with EXEC masking. Each lane iterates z = z^2 + c until |z|^2 > 4 or
; 64 iterations, with c from attr0.xy (u, v 0..1 over the image) mapped to [-2.25, 0.75] x [-1.5, 1.5]. A lane that
; escapes drops out of EXEC (s_and_b32 exec_lo, exec_lo, vcc_lo); s_cbranch_execz leaves the loop once every lane has
; escaped, else the uniform counter s10 ends it after 64 rounds. EXEC is restored from s20 before the export.
; Only v_mul/v_add/v_sub (no FMA), so a float32 model on the CPU gives the same counts.
; Branch offsets are dwords from the next instruction: execz +9 forward to the restore, scc1 -15 back to the loop top.
; Run: vk_harness vs_ids.spv ps_mandelbrot.spv out.raw
;   -> mrt0 = (sqrt(t), t*t, 4t(1-t), iteration count as raw bits) with t = count / 64
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v4, 0x40400000, v2
v_add_f32_e32 v4, 0xc0100000, v4
v_mul_f32_e32 v5, 0x40400000, v3
v_add_f32_e32 v5, 0xbfc00000, v5
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
s_mov_b32 s10, 0
s_mov_b32 s20, exec_lo
v_mul_f32_e32 v9, v6, v6
v_mul_f32_e32 v10, v7, v7
v_add_f32_e32 v11, v9, v10
v_cmp_ge_f32_e32 vcc_lo, 4.0, v11
s_and_b32 exec_lo, exec_lo, vcc_lo
s_cbranch_execz 9
v_sub_f32_e32 v12, v9, v10
v_mul_f32_e32 v13, v6, v7
v_add_f32_e32 v13, v13, v13
v_add_f32_e32 v6, v12, v4
v_add_f32_e32 v7, v13, v5
v_add_nc_u32_e32 v8, 1, v8
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, 64
s_cbranch_scc1 -15
s_mov_b32 exec_lo, s20
v_cvt_f32_u32_e32 v14, v8
v_mul_f32_e32 v14, 0x3c800000, v14
v_sqrt_f32_e32 v15, v14
v_mul_f32_e32 v16, v14, v14
v_sub_f32_e32 v17, 1.0, v14
v_mul_f32_e32 v17, v14, v17
v_mul_f32_e32 v17, 4.0, v17
exp mrt0 v15, v16, v17, v8 done vm
s_endpgm
