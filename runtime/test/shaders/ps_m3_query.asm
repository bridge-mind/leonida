; M3 occlusion probe: one color export, with all framebuffer writes disabled by the PSO.
; Same root constants as ps_m3.asm; no depth export, discard or memory side effects.
; Vulkan permits occlusion counts multiplied by the number of written color outputs, even with PRECISE.
; Keep the probe single-output so its count can be compared exactly with the CPU's visible rectangle area.
; fssm-at: 0x10
; fssm-data: 0x33515350 0x20535020
v_mov_b32_e32 v2, s5
v_mov_b32_e32 v3, s6
v_mov_b32_e32 v4, s7
v_mov_b32_e32 v5, 1.0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
