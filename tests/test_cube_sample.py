"""Cube samples in tools/rdna2spirv.py.

The hardware takes a cube sample's s and t in [1, 2] (sc / |cubema| + 1.5) and its third coordinate as 8 * slice + face
(Mesa ac_nir_lower_tex_coords.c, AMD LLPC ImageBuilder.cpp). The translator samples a 6-layer 2D-array view instead,
so it passes s - 1, t - 1 and layer 6 * slice + face. These tests evaluate the coordinate the SPIR-V hands to the
sample; runtime/test/shaders/ps_cube_sample.asm checks the result on a GPU.
"""
import math
import re
import struct

import pytest

from listing import translate_listing


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def sample_coords(spv, vgprs):
    """The coordinate of the shader's only sample or gather, given the float in each VGPR it reads."""
    d = {}
    for line in spv.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+) (.*)", line)
        if m:
            d[m.group(1)] = (m.group(2), m.group(3).split())

    def value(x):
        op, a = d[x]
        if op == "OpConstant":
            return float(a[1]) if a[0] == "%float" else int(a[1])
        if op == "OpBitcast" and a[0] == "%float":
            return struct.unpack("<f", struct.pack("<I", value(a[1])))[0]
        if op == "OpLoad":
            chain = d[a[1]][1]
            assert chain[1] == "%vgpr"
            return bits(vgprs[value(chain[2])])
        if op == "OpFSub":
            return f32(value(a[1]) - value(a[2]))
        if op == "OpFMul":
            return f32(value(a[1]) * value(a[2]))
        if op == "OpExtInst" and a[2] == "Floor":
            return float(math.floor(value(a[3])))
        raise AssertionError(f"unexpected {op} in the coordinate")

    sample = re.search(r"= (?:OpImageSample\w+|OpImageGather) %v4float %\w+ (%\w+)", spv)
    return [value(c) for c in d[sample.group(1)][1][1:]]


def listing(instruction, s, t, z, lod=None):
    regs = [s, t, z] + ([lod] if lod is not None else [])
    movs = "\n".join(f"v_mov_b32 v{4 + i}, 0x{bits(x):08x}" for i, x in enumerate(regs))
    return movs + "\n" + instruction + "\ns_waitcnt vmcnt(0)"


CUBE = {
    "sample": "image_sample v[0:3], v[4:6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE",
    "gather4_lz": "image_gather4_lz v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
    "sample_l": "image_sample_l v[0:3], v[4:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE",
}


@pytest.mark.parametrize("name", sorted(CUBE))
@pytest.mark.parametrize("face, slice_", [(0, 0), (3, 0), (5, 0), (3, 2), (0, 7)])
def test_cube_coordinates_map_onto_the_2d_array_view(name, face, slice_, tmp_path):
    # Direction (1, 0.5, -0.25) on +X: sc = 0.25, tc = -0.5, cubema = 2, so s = 1.625 and t = 1.25 on the hardware.
    lod = 2.0 if name == "sample_l" else None
    spv = translate_listing(tmp_path, listing(CUBE[name], 1.625, 1.25, 8.0 * slice_ + face, lod))
    vgprs = {4: 1.625, 5: 1.25, 6: 8.0 * slice_ + face}
    assert sample_coords(spv, vgprs) == [0.625, 0.25, float(6 * slice_ + face)]


def test_face_coordinate_edges_stay_inside_the_face(tmp_path):
    spv = translate_listing(tmp_path, listing(CUBE["sample"], 1.0, 2.0, 4.0))
    assert sample_coords(spv, {4: 1.0, 5: 2.0, 6: 4.0}) == [0.0, 1.0, 4.0]


def test_2d_array_coordinates_are_unchanged(tmp_path):
    asm = "image_sample v[0:3], v[4:6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY"
    spv = translate_listing(tmp_path, listing(asm, 0.625, 0.25, 9.0))
    assert "Floor" not in spv
    assert sample_coords(spv, {4: 0.625, 5: 0.25, 6: 9.0}) == [0.625, 0.25, 9.0]


def test_cube_loads_keep_their_integer_layer(tmp_path):
    # Integer coordinates address a cube array as layer 6 * slice + face already (LLPC combineCubeArrayFaceAndSlice).
    asm = "image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm"
    spv = translate_listing(tmp_path, "v_mov_b32 v4, 3\nv_mov_b32 v5, 4\nv_mov_b32 v6, 15\n" + asm + "\ns_waitcnt vmcnt(0)")
    assert "Floor" not in spv and "OpFSub" not in spv
