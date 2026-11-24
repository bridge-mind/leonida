; The live width of the pixel stage: how many invocations of a fragment subgroup are alive, as the shader's own EXEC
; shows it (the translator makes EXEC the subgroup ballot of the live invocations). Every pixel writes
; popcount(exec_lo) to all four words of mrt0, so the largest word of the frame is the width a full subgroup has.
; It is not gl_SubgroupSize: a driver can report 32 and run 16 invocations (Intel UHD 770: reported 32, 16 live,
; lane ids 0..15; a read of lane L ^ 16 returns garbage), and w64_graphics.py skips the pixel fixtures that need
; 32 live lanes on a device that measures less.
; Run: vk_harness vs_ids.spv ps_live_width.spv out.raw; it is a probe, no expected words.
s_bcnt1_i32_b32 s4, exec_lo
v_mov_b32_e32 v2, s4
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
