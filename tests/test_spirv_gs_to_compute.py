"""tools/spirv_gs_to_compute.py: a translated geometry shader as a compute capture pass (SHADER-159)."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from conftest import HAVE_SPIRV, make_dxbc, requires_tools, signature_chunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spirv_gs_to_compute as gsc  # noqa: E402

# gs_tri's signature, as scripts/run_gpu_tests.py builds it: SV_Position in register 0, TEXCOORD0 in register 1.
GS_DXBC = make_dxbc([signature_chunk(b"ISG1", [(0, 1, 0xF), (1, 0, 0xF)]),
                     signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)])])


@pytest.fixture
def gs_tri(make_shader):
    asm = open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8").read()
    shader = make_shader(asm, name="gs_tri", dxbc=GS_DXBC)
    spvasm = shader.translate()
    assert "OpEntryPoint Geometry" in spvasm
    return shader, spvasm


@requires_tools
def test_layout(gs_tri):
    _, spvasm = gs_tri
    _, lay = gsc.rewrite(spvasm)
    assert lay == {"verts_in": 3, "max_verts": 3, "topology": "triangle_strip",
                   "inputs": ["position", "location 1"], "outputs": ["position", "location 1"], "local_size": 64}


@requires_tools
def test_compute_module(gs_tri):
    shader, spvasm = gs_tri
    out, _ = gsc.rewrite(spvasm)
    assert "OpCapability Geometry" not in out
    assert re.search(r"OpEntryPoint GLCompute %\w+ \"main\"", out)
    assert re.search(r"OpExecutionMode %\w+ LocalSize 64 1 1", out)
    assert "OpEmitVertex" not in out and "OpEndPrimitive" not in out
    # no outputs are left, and the only inputs are GlobalInvocationId and the subgroup lane the GS already read
    assert not re.search(r"OpVariable %\w+ Output\b", out)
    inputs = re.findall(r"(%\w+) = OpVariable %\w+ Input\b", out)
    builtins = {v: re.search(rf"OpDecorate {v} BuiltIn (\w+)", out).group(1) for v in inputs}
    assert sorted(builtins.values()) == ["GlobalInvocationId", "SubgroupLocalInvocationId"]
    # three emits, two outputs each, go to the record buffer
    assert len(re.findall(r"OpAccessChain %gsc_ptr_sb_v4 %gsc_rec ", out)) == 6
    for b in (0, 1, 2):
        assert f"Binding {b}" in out
    shader.validate(out)    # spirv-as + spirv-val, as for the translator's own output


@requires_tools
def test_rejects_non_geometry(make_shader):
    shader = make_shader("v_mov_b32 v1, 1.0\ns_endpgm", name="cs_not_gs")
    with pytest.raises(gsc.Unsupported):
        gsc.rewrite(shader.translate())


@requires_tools
def test_cli_writes_module_and_layout(gs_tri, tmp_path):
    shader, spvasm = gs_tri
    sa, sv = tmp_path / "gs.spvasm", tmp_path / "gs.spv"
    sa.write_text(spvasm)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    out = tmp_path / "cs.spv"
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"), str(sv),
                        str(out), "--layout", str(tmp_path / "lay.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "lay.json").exists()
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(out)], check=True)


# A hand-built GS whose entry block branches, with an OpPhi naming the entry label as a predecessor (the translator
# emits those for guarded loads: its block_label is %entry until the first new label). The primitive-count check
# splits the entry block, so the OpPhi must name the block that now ends the entry code (review of #1331).
PHI_GS = """\
OpCapability Shader
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %in_pos %out_pos
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %in_pos BuiltIn Position
OpDecorate %out_pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u3 = OpConstant %uint 3
%arr = OpTypeArray %v4float %u3
%ptr_in_arr = OpTypePointer Input %arr
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%in_pos = OpVariable %ptr_in_arr Input
%out_pos = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpIEqual %bool %u0 %u1
OpSelectionMerge %m None
OpBranchConditional %c %t %m
%t = OpLabel
OpBranch %m
%m = OpLabel
%x = OpPhi %uint %u1 %t %u0 %entry
%p = OpAccessChain %ptr_in_v4 %in_pos %x
%v = OpLoad %v4float %p
OpStore %out_pos %v
OpEmitVertex
OpEmitVertex
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
"""


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
def test_phi_from_entry_block_names_the_split_block(tmp_path):
    out, _ = gsc.rewrite(PHI_GS)
    assert "%x = OpPhi %uint %u1 %t %u0 %gsc_body" in out
    sa, sv = tmp_path / "cs.spvasm", tmp_path / "cs.spv"
    sa.write_text(out)
    subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"], check=True)
    r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(sv)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not HAVE_SPIRV or not shutil.which("spirv-dis"), reason="needs SPIRV-Tools on PATH")
@pytest.mark.parametrize("target", ["vulkan1.0", "vulkan1.1", "vulkan1.2"])
def test_cli_preserves_valid_capture_across_spirv_versions(tmp_path: Path, target: str) -> None:
    source, output = tmp_path / "gs.spv", tmp_path / "cs.spv"
    subprocess.run(["spirv-as", "--target-env", target, "-", "-o", source], input=PHI_GS, text=True, check=True)
    subprocess.run(["spirv-val", "--target-env", target, source], check=True)
    tool = os.path.join(ROOT, "tools", "spirv_gs_to_compute.py")
    result = subprocess.run([sys.executable, "-I", tool, source, output], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(["spirv-val", "--target-env", target, output], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes()[4:8] == source.read_bytes()[4:8]


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
def test_lowers_in_bounds_geometry_interface_access_chains(tmp_path: Path) -> None:
    geometry = PHI_GS.replace(
        "%p = OpAccessChain %ptr_in_v4 %in_pos %x",
        "%p = OpInBoundsAccessChain %ptr_in_v4 %in_pos %x",
    ).replace(
        "%out_pos = OpVariable %ptr_out_v4 Output",
        "%ptr_out_float = OpTypePointer Output %float\n%out_pos = OpVariable %ptr_out_v4 Output",
    ).replace(
        "OpStore %out_pos %v",
        "OpStore %out_pos %v\n"
        "%out_x = OpInBoundsAccessChain %ptr_out_float %out_pos %u0\n"
        "%x_value = OpCompositeExtract %float %v 0\n"
        "OpStore %out_x %x_value",
    )

    def validate(name: str, assembly: str) -> None:
        module = tmp_path / f"{name}.spv"
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(module)],
                       input=assembly, text=True, check=True)
        result = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    validate("geometry", geometry)
    lowered, _ = gsc.rewrite(geometry)
    assert "%p = OpAccessChain %gsc_ptr_sb_v4 %gsc_in " in lowered
    assert "%out_x = OpAccessChain %gsc_ptr_pv_f %gsc_out0 %u0" in lowered
    validate("compute", lowered)


@pytest.fixture
def gs_module(tmp_path):
    if not (HAVE_SPIRV and shutil.which("spirv-dis")):
        pytest.skip("needs SPIRV-Tools on PATH")
    source = tmp_path / "gs.spv"
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(source)],
                   input=PHI_GS, text=True, check=True)
    return source


@pytest.mark.parametrize("overlap", ["input-output", "input-layout", "output-layout", "hardlinked-layout"])
def test_cli_rejects_overlapping_paths_without_writing(tmp_path, gs_module, overlap):
    source = gs_module
    output = tmp_path / "cs.spv"
    layout = tmp_path / "layout.json"
    original = source.read_bytes()
    if overlap == "input-output":
        output = source
    elif overlap == "input-layout":
        layout = source
    elif overlap == "output-layout":
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        layout = subdir / ".." / output.name
    else:
        try:
            os.link(source, layout)
        except OSError as e:
            pytest.skip(f"hard links unavailable: {e}")

    result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"),
                             str(source), str(output), "--layout", str(layout)], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert "same file" in result.stderr
    assert source.read_bytes() == original
    assert not (tmp_path / "cs.spv").exists()
    if overlap == "hardlinked-layout":
        assert layout.read_bytes() == original


def test_cli_accepts_distinct_module_and_layout_paths(tmp_path, gs_module):
    output = tmp_path / "cs.spv"
    layout = tmp_path / "layout.json"
    original = gs_module.read_bytes()
    result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "spirv_gs_to_compute.py"),
                             str(gs_module), str(output), "--layout", str(layout)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert gs_module.read_bytes() == original
    assert json.loads(layout.read_text(encoding="utf-8"))["topology"] == "triangle_strip"
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(output)], check=True)


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("controls", [
    (),
    ("SignedZeroInfNanPreserve 32", "DenormPreserve 16"),
    ("DenormFlushToZero 32", "RoundingModeRTE 32", "RoundingModeRTZ 16"),
], ids=["opt-out", "translator-default", "explicit"])
def test_capture_preserves_float_control_modes(tmp_path: Path, controls: tuple[str, ...]) -> None:
    capabilities = "".join(f"OpCapability {control.split()[0]}\n" for control in controls)
    modes = [f"OpExecutionMode %main {control}" for control in controls]
    geometry = PHI_GS.replace(
        "OpMemoryModel", capabilities + "OpMemoryModel", 1,
    ).replace(
        "OpDecorate %in_pos", "".join(f"{mode}\n" for mode in modes) + "OpDecorate %in_pos", 1,
    ).replace(
        "OpStore %out_pos %v", "%neg = OpFNegate %v4float %v\nOpStore %out_pos %neg",
    )

    def validate(name: str, assembly: str) -> None:
        module = tmp_path / f"{name}.spv"
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(module)],
                       input=assembly, text=True, check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)], check=True)

    validate("geometry", geometry)
    lowered, _ = gsc.rewrite(geometry)
    validate("compute", lowered)
    execution_modes = [line for line in lowered.splitlines() if line.startswith("OpExecutionMode ")]
    assert execution_modes == ["OpExecutionMode %main LocalSize 64 1 1", *modes]


def test_help_is_a_clean_exit_zero(capsys):
    for flag in ("-h", "--help"):
        assert gsc.main(["spirv_gs_to_compute.py", flag]) == 0
        out = capsys.readouterr().out
        assert "spirv_gs_to_compute.py in.spv out.spv" in out
        assert "SPIRV-Tools" in out


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("targets,expected_targets", [
    ("%in_pos", ""),
    ("%out_pos", ""),
    ("%in_pos %v %out_pos %x", "%v %x"),
    ("%v %x", "%v %x"),
], ids=["removed-input", "removed-output", "mixed", "live-only"])
def test_group_decorate_keeps_only_live_capture_targets(tmp_path: Path, targets: str,
                                                        expected_targets: str) -> None:
    geometry = PHI_GS.replace(
        "OpDecorate %in_pos BuiltIn Position", "OpDecorate %in_pos Location 0",
    ).replace(
        "%void = OpTypeVoid",
        "OpDecorate %relaxed RelaxedPrecision\n"
        "%relaxed = OpDecorationGroup\n"
        f"OpGroupDecorate %relaxed {targets} ; keep only module IDs\n"
        "%void = OpTypeVoid",
    )

    lowered, _ = gsc.rewrite(geometry)
    for name, assembly in (("geometry", geometry), ("compute", lowered)):
        module = tmp_path / f"{name}.spv"
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(module)],
                       input=assembly, text=True, check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)], check=True)

    group_uses = [line.partition(";")[0].strip() for line in lowered.splitlines()
                  if line.strip().startswith("OpGroupDecorate ")]
    expected = [f"OpGroupDecorate %relaxed {expected_targets}"] if expected_targets else []
    assert group_uses == expected


def test_builtin_load_and_output_store_with_memory_operands():
    gs = PHI_GS.replace(
        'OpEntryPoint Geometry %main "main" %in_pos %out_pos',
        'OpEntryPoint Geometry %main "main" %in_pos %out_pos %prim_id %inv_id',
    ).replace(
        "OpDecorate %out_pos BuiltIn Position",
        "OpDecorate %out_pos BuiltIn Position\n"
        "OpDecorate %prim_id BuiltIn PrimitiveId\n"
        "OpDecorate %inv_id BuiltIn InvocationId",
    ).replace(
        "%out_pos = OpVariable %ptr_out_v4 Output",
        "%ptr_in_u = OpTypePointer Input %uint\n"
        "%out_pos = OpVariable %ptr_out_v4 Output\n"
        "%prim_id = OpVariable %ptr_in_u Input\n"
        "%inv_id = OpVariable %ptr_in_u Input",
    ).replace(
        "%v = OpLoad %v4float %p\nOpStore %out_pos %v",
        "%pid = OpLoad %uint %prim_id Aligned 4\n"
        "%iid = OpLoad %uint %inv_id Volatile\n"
        "%v = OpLoad %v4float %p\n"
        "OpStore %out_pos %v Aligned 16",
    )
    out, _ = gsc.rewrite(gs)
    assert "%prim_id" not in out and "%inv_id" not in out
    assert "%pid = OpCopyObject %uint %gsc_prim" in out
    assert "%iid = OpCopyObject %uint " in out
    assert "OpStore %gsc_out0 %v" in out
    if HAVE_SPIRV:
        res = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", os.devnull],
                             input=out, text=True, capture_output=True)
        assert res.returncode == 0, res.stderr

    bad_gs = gs.replace("%pid = OpLoad %uint %prim_id Aligned 4", "OpCopyMemory %prim_id %inv_id")
    with pytest.raises(gsc.Unsupported, match="%prim_id"):
        gsc.rewrite(bad_gs)


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("before_first,before_second", [
    ("", ""),
    ("OpLine %source 1 1\n", ""),
    ("", "OpNoLine\n"),
    ("OpLine %source 1 1\n", "OpNoLine\n"),
], ids=["no-debug", "line", "no-line", "mixed"])
def test_entry_debug_lines_keep_locals_before_capture_setup(tmp_path: Path, before_first: str,
                                                            before_second: str) -> None:
    geometry = PHI_GS.replace(
        "OpDecorate %in_pos", '%source = OpString "synthetic.glsl"\nOpDecorate %in_pos', 1,
    ).replace(
        "%out_pos = OpVariable", "%ptr_local = OpTypePointer Function %float\n%out_pos = OpVariable", 1,
    ).replace(
        "%entry = OpLabel",
        f"%entry = OpLabel\n{before_first}%first = OpVariable %ptr_local Function\n"
        f"{before_second}%second = OpVariable %ptr_local Function",
    ).replace(
        "OpStore %out_pos %v",
        "%x_value = OpCompositeExtract %float %v 0\n"
        "OpStore %first %x_value\n"
        "%first_value = OpLoad %float %first\n"
        "OpStore %second %first_value\n"
        "%second_value = OpLoad %float %second\n"
        "%position = OpCompositeInsert %v4float %second_value %v 0\n"
        "OpStore %out_pos %position",
    )

    lowered, _ = gsc.rewrite(geometry)
    for name, assembly in (("geometry", geometry), ("compute", lowered)):
        module = tmp_path / f"{name}.spv"
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", str(module)],
                       input=assembly, text=True, check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)], check=True)

    assert lowered.index("%first = OpVariable") < lowered.index("%gsc_gidv = OpLoad")
    assert lowered.index("%second = OpVariable") < lowered.index("%gsc_gidv = OpLoad")
