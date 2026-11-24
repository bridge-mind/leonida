; dPdx = v10..v12, dPdy = v13..v15. Full addresses use v16..v18
; for XYZ and v19 for the clamp. A16 uses v16.lo/hi and v17.lo for
; XYZ, and v17.hi for the clamp, keeping all six derivatives as f32.
image_sample_d_cl v3, v[10:19], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D
image_sample_d v3, v[10:18], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D
image_sample_d_cl v3, v[10:17], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D a16
image_sample_d v3, v[10:17], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D a16
