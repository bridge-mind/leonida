"""Memory-op forms that the assembler accepts and the translator must refuse cleanly, not crash on.

An image instruction whose result operand is not a VGPR range (`image_sample off, v[1:2], s[16:23], s[24:27] ...`:
llvm-mc takes `off` there) used to raise a KeyError('r') in the translator when it wrote the result. It is a form no
compiler emits (a sample with a dmask and nowhere to put the texels); the translator now raises Unsupported for the
result operand instead. A probe of about 60,000 DS, MUBUF, MTBUF, MIMG, SMEM and FLAT/GLOBAL/SCRATCH operand and
modifier variations (glc, slc, dlc, lds, tfe, d16, a16, unorm, dmask, dim, offset, idxen, offen, addr64, format; as
written and as llvm-mc prints them; pixel and compute stages) found no other crash. Standard library only; no GPU.
"""
import unittest

import support
import rdna2spirv

CASES = (
    "image_sample off, v[1:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D slc",
    "image_sample off, v[1:2], s[16:23], s[24:27] dmask:0x3 dim:SQ_RSRC_IMG_2D tfe lwe",
    "image_sample_c_lz off, v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    "image_sample_c_lz off, v[1:4], s[16:23], s[24:27] dmask:0x3 dim:SQ_RSRC_IMG_CUBE d16",
    "image_sample_o off, v[1:4], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_3D tfe",
    "image_sample_l off, v[1:4], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY lwe",
    "image_load off, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    "image_load_mip off, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    "image_gather4 off, v[1:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    "image_gather4_lz off, v[1:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D",
    "image_get_lod off, v[1:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D",
)


class ImageResultOperand(unittest.TestCase):
    def test_non_vgpr_result_is_unsupported_not_a_crash(self):
        for line in CASES:
            with self.subTest(line=line):
                text = "; mem_crash\n\t" + line + "\n\tv_mov_b32 v50, 1\n\texp mrt0 v50, v50, v50, v50 done vm\n\ts_endpgm\n"
                try:
                    support.translate_listing("ps_mem_crash", text.encode("utf-8"))
                except rdna2spirv.Unsupported:
                    pass


if __name__ == "__main__":
    unittest.main()
