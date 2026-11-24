; Both coordinates share one VGPR, low half first. The outputs remain f32.
; 2D: (0.5, -0.25) in v10. Array: (-1.0, 2.0) in v12, layer in v21.
v_mov_b32_e32 v10, 0xb4003800
v_mov_b32_e32 v12, 0x4000bc00
v_mov_b32_e32 v21, 0x4700
image_get_lod v[4:5], v10, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D a16
image_get_lod v6, [v12, v21], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY a16
v_mov_b32_e32 v3, v6
