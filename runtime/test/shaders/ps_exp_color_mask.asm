; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; A pixel shader's color export has its own lane mask: EXEC at that export. This program exports mrt0 from every
; lane with `done` and no `vm`, waits for the export, and only then turns every lane off. Measured on RDNA2
; (gfx1036, ps_oracle.py of #3061): every pixel is written with the exported words. A path without a `vm` export
; goes by EXEC at the color export, not by EXEC at the end; until 2026-11-23 the translator discarded every pixel
; here. (The opposite order, EXEC off at the export and back on afterwards, leaves the pixel unwritten on
; hardware; that half is in tests/semantic/test_export_valid_mask.py, where a discard can be asserted.)
; The s_waitcnt matters on hardware: an export is asynchronous, and without it the EXEC write that follows is
; what the hardware sees.
; Expected mrt0 bits, every pixel: 0x40000000 (2.0), 0x40800000 (4.0), 0x40000000, 0x40800000.
v_mov_b32 v2, 2.0
v_mov_b32 v3, 4.0
v_mov_b32 v1, 1
exp mrt0 v2, v3, v2, v3 done
s_waitcnt expcnt(0)
v_cmpx_eq_u32 0, v1
s_endpgm
