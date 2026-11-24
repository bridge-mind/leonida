; Vertex shader for ps_interp_f16_high: the vs_ids triangle (u and v run 0..1 over the image) with two f16 values
; packed in each 32-bit param channel, as Mesa's prerast export packs 16-bit varyings (low half, high half):
; param0.x = f16(u) | f16(v) << 16, param0.y = f16(v) | f16(u) << 16, and param1 = (u, v, 0, 1) as f32.
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
v_cvt_pkrtz_f16_f32_e32 v7, v1, v2
v_cvt_pkrtz_f16_f32_e32 v8, v2, v1
exp param0 v7, v8, v5, v6
exp param1 v1, v2, v5, v6
s_endpgm
