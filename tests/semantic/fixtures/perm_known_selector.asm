; Hand-written pixel shader. The four words cover a mixed literal selector, a known SGPR,
; the same SGPR after a non-folded write, and the source-1 identity selector.
v_mov_b32 v0, 0x8c7a06f5
v_mov_b32 v1, 0x13e49b80
s_mov_b32 s8, 0x0b0a0908
v_perm_b32 v4, v0, v1, 0x800c0700
v_perm_b32 v5, v0, v1, s8
s_xor_b32 s8, s8, 0x0b0a0908
v_perm_b32 v6, v0, v1, s8
v_perm_b32 v7, v0, v1, 0x03020100
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
