; Packed address words contain Y:X and padding:mip, low half first.
image_load_mip v[0:3], v[8:9], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D a16
image_load_mip v0, [v18,v5], s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_2D a16 d16
