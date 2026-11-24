; Wave64 pixel shader with v_readlane_b32 across both halves of the wave (RDNA2 ISA: lane = S1 % 64, EXEC ignored).
; The translator's pixel path takes the lane from SubgroupLocalInvocationId, which is the hardware lane only in a
; 64-wide subgroup; in a 32-wide one the lane select wraps (lane 40 reads lane 8), so the words below differ.
; v2 = 3 * lane + 1. Every lane of a full wave reads the same scalars:
;   s6 = v2[40] = 121, s7 = v2[5] = 16, s8 = v2[33] = 100, s9 = v2[63] = 190
; Each is or'ed with 0x40000000 so the export is a normal float (a float render target keeps bit patterns, but a
; denormal could be flushed). Export mrt0 = (0x40000079, 0x40000010, 0x40000064, 0x400000be) for every covered
; pixel of a full wave; draw a triangle that covers the whole target so every wave is full.
; Run: d3d12_subgroup_test (the PS through a D3D12 graphics PSO, R32G32B32A32_FLOAT target).
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_mad_u32_u24 v2, v1, 3, 1
v_readlane_b32 s6, v2, 40
v_readlane_b32 s7, v2, 5
v_readlane_b32 s8, v2, 33
v_readlane_b32 s9, v2, 63
s_or_b32 s6, s6, 0x40000000
s_or_b32 s7, s7, 0x40000000
s_or_b32 s8, s8, 0x40000000
s_or_b32 s9, s9, 0x40000000
v_mov_b32 v3, s6
v_mov_b32 v4, s7
v_mov_b32 v5, s8
v_mov_b32 v6, s9
exp mrt0 v3, v4, v5, v6 done vm
s_endpgm
