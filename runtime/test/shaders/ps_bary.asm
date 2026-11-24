; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Raw barycentric reads (v0 = i, v1 = j) next to v_interp. With vk_harness geometry, raw i/j must equal
; the interpolated attr0.xy / 2 on every pixel (attr0 = (0,0), (2,0), (0,2) at the three vertices).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32 v4, v0
v_mov_b32 v5, v1
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
