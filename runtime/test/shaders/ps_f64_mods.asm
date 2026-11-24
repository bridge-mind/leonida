; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Output modifiers on v_add_f64 (VOP3_Profile<VOP_F64_F64_F64>: clamp and omod): omod scales, then clamp to [0, 1].
; a = 0.75, b = 0.5 as doubles (low word first); each export is the high word of one double result (low words are 0).
; mul:2 (a+b)*2 = 2.5, clamp min(a+b, 1) = 1.0, div:2 clamp (a+b)/2 = 0.625 (clamp before scaling would give 0.5),
; mul:4 (a-b)*4 = 1.0. Without the modifiers every word would be 0x3ff40000 (1.25) or 0x3fd00000 (0.25).
; Needs shaderFloat64; vk_harness exits 77 (SKIP) on a device without it.
; Expected mrt0 bits: 0x40040000, 0x3ff00000, 0x3fe40000, 0x3ff00000
v_mov_b32 v0, 0
v_mov_b32 v1, 0x3fe80000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x3fe00000
v_add_f64 v[4:5], v[0:1], v[2:3] mul:2
v_add_f64 v[6:7], v[0:1], v[2:3] clamp
v_add_f64 v[8:9], v[0:1], v[2:3] clamp div:2
v_add_f64 v[10:11], v[0:1], -v[2:3] mul:4
exp mrt0 v5, v7, v9, v11 done vm
s_endpgm
