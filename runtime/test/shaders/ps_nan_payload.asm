; Export two non-canonical f32 quiet NaNs and two finite controls to RGBA32F.
; Expected mrt0 words: 0x7fc12345, 0xffc54321, 0x3f800000, 0x3f000000.
; Only KosmicKrisp may return the canonical 0x7fc00000 for the NaNs (#3125).

v_mov_b32 v0, 0x7fc12345
v_mov_b32 v1, 0xffc54321
v_mov_b32 v2, 0x3f800000
v_mov_b32 v3, 0x3f000000
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
