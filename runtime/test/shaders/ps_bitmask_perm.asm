; ds_swizzle BITMASK_PERM: v1 must equal lane ^ 9 ("pippi"), v2 must equal ((lane & 5) | 16) ^ 1 ("10p0i") on every pixel.
v_mbcnt_lo_u32_b32 v0, -1, 0
ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,"pippi")
ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM,"10p0i")
v_mov_b32_e32 v3, 0
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
