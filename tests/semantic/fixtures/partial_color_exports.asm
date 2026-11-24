; MRT0: sparse uncompressed components. MRT1: compressed half pairs.
v_mov_b32_e32 v0, 0x3f800000
v_mov_b32_e32 v1, 0x40000000
v_mov_b32_e32 v2, 0x40400000
v_mov_b32_e32 v3, 0x40800000
v_mov_b32_e32 v4, 0x46004500
v_mov_b32_e32 v5, 0x48004700
exp mrt0 v0, off, v2, off
exp mrt1 v4, v4, off, off compr
exp mrt0 off, v1, off, v3
exp mrt1 off, off, v5, v5 done compr vm
s_endpgm
