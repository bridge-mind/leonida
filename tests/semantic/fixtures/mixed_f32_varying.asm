; Read attr0.x both from P0 and from the perspective-correct P1/P2 pair.
v_interp_mov_f32 v2, p0, attr0.x
v_interp_p1_f32 v3, v0, attr0.x
v_interp_p2_f32 v3, v1, attr0.x
exp mrt0 v2, v3, v2, v3 done vm
s_endpgm
