"""d16 image_load reads integer-format images through their uint/sint views, like the 32-bit image_load.

The runtime binds each image's view in its own format in the sampled array of its dimension, and the translator reads
UINT/SINT formats through uint/sint aliases of that array (docs/design.md, Typed image loads; SHADER-151): a float
fetch of an integer view is undefined in Vulkan. A d16 load used to skip the T# FORMAT branch and always read the float
sampled view, then pack f16: a Texture2D<min16uint> Load came back as garbage.
"""
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import Shader, requires_tools
from test_image_load_typed import Module

pytestmark = requires_tools


def test_cube_load_d16_selects_the_format_at_the_array_binding(make_shader: Callable[[str], Shader]) -> None:
    fixture = Path(__file__).parent / "semantic" / "fixtures" / "cube_load_d16.asm"
    shader = make_shader(fixture.read_text(encoding="utf-8"))
    mod = Module(shader.translate())
    shader.validate(mod.text)
    assert {image[0] for _, _, image in mod.fetches()} == {"%uint", "%int", "%float"}
    for _, var, image in mod.fetches():
        assert (mod.binding[var], image[1:4]) == (3, ["2D", "0", "1"])
    for fmt, kind in ((20, "uint"), (21, "sint"), (22, "float")):  # GFX10 32_UINT/SINT/FLOAT
        assert mod.branch(fmt << 20, 5) == kind
        assert mod.branch(fmt << 20, 0) == "float"


@pytest.mark.parametrize("mask,dst", [("0x1", "v3"), ("0x3", "v3"), ("0x7", "v[3:4]")])
def test_d16_image_load_reads_integer_views(make_shader, mask, dst):
    sh = make_shader("v_mov_b32 v1, 0\nv_mov_b32 v2, 0\n"
                     f"image_load {dst}, v[1:2], s[8:15] dmask:{mask} dim:SQ_RSRC_IMG_2D unorm d16\n"
                     "s_waitcnt vmcnt(0)\nexp mrt0 v3, v4, v3, v4 done vm\ns_endpgm\n", name="ps_d16_load")
    spv = sh.translate()
    fetches = re.findall(r"OpImageFetch %(\w+) ", spv)
    assert "v4uint_st" in fetches and "v4int_st" in fetches, "d16 image_load never reads the uint/sint views"
    assert "OpImageRead" not in spv and "StorageImageReadWithoutFormat" not in spv   # sampled aliases, not storage
    assert "PackHalf2x16" not in spv      # the float branch keeps the integer RTE f16 pack (#489)
    sh.validate(spv)


@pytest.mark.parametrize("load,addr,dim", [("image_load", "v[1:2]", "2D"), ("image_load_mip", "v[4:7]", "3D")])
def test_d16_integer_fetch_saturates_before_packing(make_shader, load, addr, dim):
    shader = make_shader(f"{load} v3, {addr}, s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_{dim} d16\n"
                         "s_waitcnt vmcnt(0)\nexp mrt0 v3, v3, v3, v3 done vm\ns_endpgm\n", name="ps_d16_sat")
    text = shader.translate()
    # R and A are both narrowed on each integer path. Full-width and float paths have different conversions.
    for op, bounds in (("UMin", [65535]), ("SClamp", [0xffff8000, 32767])):
        clamps = re.findall(rf"OpExtInst %\w+ %glsl {op} (%\w+(?: %\w+)+)", text)
        assert len(clamps) == 2
        for args in clamps:
            bound_ids = args.split()[1:]
            assert len(bound_ids) == len(bounds)
            for bound, expected in zip(bound_ids, bounds):
                cast = re.search(rf"{bound} = OpBitcast %int (%\w+)", text)
                const = cast[1] if cast else bound
                assert re.search(rf"{const} = OpConstant %uint {expected}\b", text)
    shader.validate(text)
