"""GPU regression for translated GS strip cuts. Set LEONIDA_VK_HARNESS to the built vk_harness."""
import os
from pathlib import Path
import struct
import subprocess

import pytest

from conftest import GS_TRI_DXBC, Shader, requires_tools


@requires_tools
def test_offscreen_geometry_strip_cuts(tmp_path):
    harness = os.environ.get("LEONIDA_VK_HARNESS")
    if not harness:
        pytest.skip("set LEONIDA_VK_HARNESS to the built vk_harness")

    # Form a nondegenerate triangle with z=-1, w=1, entirely behind Vulkan's near clip plane.
    vs = Shader(tmp_path, "vs_clipped", """
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_mov_b32_e32 v3, -1.0
v_mov_b32_e32 v4, 1.0
exp pos0 v1, v2, v3, v4 done
exp param1 v1, v2, v3, v4
s_endpgm
""")
    ps = Shader(tmp_path, "ps_white", """
v_mov_b32_e32 v0, 1.0
exp mrt0 v0, v0, v0, v0 done
s_endpgm
""")
    for shader in (vs, ps):
        shader.validate(shader.translate())

    fixture = Path(__file__).resolve().parents[1] / "runtime/test/shaders/gs_tri.asm"
    triangle = fixture.read_text(encoding="utf-8")
    # Three single-vertex strips produce zero triangles. One three-vertex strip produces one.
    singles = triangle.replace("GS_OP_EMIT,", "GS_OP_EMIT_CUT,")
    env = {key: value for key, value in os.environ.items() if not key.startswith("VKH_")}
    for expected, asm in [(0, singles), (1, triangle)]:
        gs = Shader(tmp_path, f"gs_strip_{expected}", asm, dxbc=GS_TRI_DXBC)
        gs.validate(gs.translate())
        raw = tmp_path / f"{gs.name}.raw"
        result = subprocess.run(
            [harness, tmp_path / "vs_clipped.spv", tmp_path / "ps_white.spv", raw, "8", "8"],
            env=dict(env, VKH_GS=str(tmp_path / f"{gs.name}.spv"), VKH_GS_PRIMITIVES="1"),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 77:
            pytest.skip(result.stderr)
        assert result.returncode == 0, result.stderr
        assert f"gs_primitives={expected}" in result.stderr.splitlines(), result.stderr
        assert raw.read_bytes() == struct.pack("<4f", -1, -1, -1, -1) * 64
