; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Flat inputs: attr1.x is read only with v_interp_mov (the provoking vertex's value), so its location is declared
; Flat. vs_ids gives the three vertices 0x101, 0x102 and 0x103 there; every pixel must see vertex 0's, 0x101, and
; integer maths on it must work (0x101 + 1). Interpolated, those bits come back as something else. attr0 next to
; it is interpolated as usual. Run: vk_harness vs_ids.spv ps_flat.spv out.raw
;   -> mrt0 bits = 0x101, 0x102 on every pixel (0, 1 without Flat); mrt0.zw = ((x + 0.5) / 64, (y + 0.5) / 64) as floats
v_interp_mov_f32_e32 v2, p0, attr1.x
v_add_nc_u32_e32 v3, 1, v2
v_interp_p1_f32_e32 v4, v0, attr0.x
v_interp_p2_f32_e32 v4, v1, attr0.x
v_interp_p1_f32_e32 v5, v0, attr0.y
v_interp_p2_f32_e32 v5, v1, attr0.y
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
