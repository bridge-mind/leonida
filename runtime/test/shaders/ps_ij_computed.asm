; SHADER-177: interpolation through a barycentric pair the program computed. Drawn with vs_ids
; (attr0 = (u, v, 0, 1), u = 2i and v = 2j running 0..1 over the image).
; Hardware weights v_interp with the values the operand registers hold (Vonsek's #1224 rows: zeroed
; sources interpolate to P0, swapped copies to (v,u)); the translator now models the computed pair:
;   mrt0.x = attr0.x with I' = J' = 0.25  -> the snapped point 0.5 * P0 + 0.25 * P1 + 0.25 * P2 = 0.5
;   mrt0.y = attr0.y with I' = J' = 0     -> the provoking vertex's value, 0.0
;   mrt0.z = attr0.y with I' = i, J' = j/2 -> v / 2 = (y + 0.5) / 128 (evaluate-at-half-offset in j)
;   mrt0.w = mrt0.x again (control)
; On main (operand ignored) this rendered (u, v, v, 0.5): x, y and z wrong on 4,096 of 4,096 pixels.
; No integer-bits row here: attr1's words are f32 denormals, which the weighting arithmetic may flush
; (the mov path keeps them exact; that is ps_flat's). Run: vk_harness vs_ids.spv ps_ij_computed.spv out.raw
v_mov_b32_e32 v10, v0
v_mov_b32_e32 v11, v1
v_mov_b32_e32 v0, 0.25
v_mov_b32_e32 v1, 0.25
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v1, 0
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v11, 0.5, v11
v_mov_b32_e32 v0, v10
v_mov_b32_e32 v1, v11
v_interp_p1_f32_e32 v4, v0, attr0.y
v_interp_p2_f32_e32 v4, v1, attr0.y
exp mrt0 v2, v3, v4, v2 done vm
s_endpgm
