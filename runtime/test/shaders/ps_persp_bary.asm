; ps_bary's code, drawn with vs_persp (clip w = 1, 2, 4) instead of vs_ids (w = 1): attr0.xy and the raw i/j
; (v0, v1) must follow the perspective-correct weights. At a pixel centre with screen-space weights l0, l1, l2 of the
; three vertices, p_k = (l_k / w_k) / sum(l_m / w_m); mrt0 = (sum p_k attr0_k.x, sum p_k attr0_k.y, p1, p2). The
; affine weights (l1, l2) differ by at least 0.007 on every covered pixel, so NoPerspective interpolation fails it.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32 v4, v0
v_mov_b32 v5, v1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
