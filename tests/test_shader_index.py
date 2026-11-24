"""Index-identity metadata for tools/rdna2spirv.py (no game data, no GPU).

Run from the repository root:  python -m pytest tests/test_shader_index.py
Needs nothing but the standard library: Gen objects are built by hand and the
hashes are computed over temp files, so these run on every CI leg including
Windows without llvm-mc.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import rdna2spirv  # noqa: E402
from rdna2spirv import Gen, Unsupported, shader_meta


def _gen(stage="ps", wave64=False):
    g = Gen(stage)
    g.wave64 = wave64
    return g


def test_basic_fields_and_subgroup_size():
    m = shader_meta(_gen("ps"), "")
    assert m["stage"] == "ps"
    assert m["wave64"] is False
    assert m["subgroup_size"] == 32
    assert m["wave64_emulated"] is False
    assert m["push_constant_bytes"] == 4 * rdna2spirv.USER_SGPRS + 16


def test_wave64_compute_requests_64():
    m = shader_meta(_gen("cs", wave64=True), "")
    assert m["wave64"] is True
    assert m["subgroup_size"] == 64
    assert m["wave64_emulated"] is False


def test_wave64_graphics_is_emulated():
    # 64-wide subgroups can only be requested for compute/mesh; VS/PS/GS
    # wave64 stays emulated as per-32-lane halves (docs/design.md).
    for stage in ("vs", "ps", "gs"):
        assert shader_meta(_gen(stage, wave64=True), "")["wave64_emulated"] is True
    assert shader_meta(_gen("ms", wave64=True), "OpCapability MeshShadingEXT")["wave64_emulated"] is False


def test_vertex_signature_passthrough(tmp_path):
    g = _gen("vs")
    g.vertex_inputs = [(0, 3)]
    g.vertex_signature = [{"register": 0, "mask": 7, "system_value": 0, "component_type": 3}]
    m = shader_meta(g, "")
    assert m["vertex_inputs"] == [{"location": 0, "type": "float"}]
    assert m["vertex_input_signature"] == [
        {"register": 0, "mask": 7, "system_value": 0, "component_type": 3}]


def test_dynamic_and_fixed_format_flags():
    m = shader_meta(_gen("cs"), "")
    assert "dynamic_format" not in m
    assert "fixed_format_codes" not in m
    g = _gen("cs")
    g.dynamic_format = True
    g.fixed_format_codes = {22, 44}
    m = shader_meta(g, "")
    assert m["dynamic_format"] is True
    assert m["fixed_format_codes"] == [22, 44]


def test_src_and_fssm_hashes(tmp_path):
    src = tmp_path / "t.s"
    fssm = tmp_path / "t.fssm"
    src.write_bytes(b"s_endpgm\n")
    fssm.write_bytes(b"XBCH fake\n")
    m = shader_meta(_gen("cs"), "", src_path=os.fspath(src), fssm_path=os.fspath(fssm))
    assert m["src_sha256"] == hashlib.sha256(b"s_endpgm\n").hexdigest()
    assert m["fssm_sha256"] == hashlib.sha256(b"XBCH fake\n").hexdigest()


def test_missing_sidecars_omitted():
    m = shader_meta(_gen("cs"), "", src_path="/no/such.s", fssm_path="/no/such.fssm")
    assert "src_sha256" not in m
    assert "fssm_sha256" not in m


def test_decode_unknown_raises_with_code():
    g = _gen("cs")
    try:
        g.decode_format(["%x"], 511)
    except Unsupported as e:
        assert "511" in str(e)
    else:
        raise AssertionError("decode_format(511) did not raise")


def test_encode_unknown_raises_with_code():
    g = _gen("cs")
    try:
        g.encode_format(["%x"], ["%y"], 511)
    except Unsupported as e:
        assert "511" in str(e)
    else:
        raise AssertionError("encode_format(511) did not raise")


def test_known_format_still_works():
    # 22 = 4-byte f32: covered by FORMATS, must not raise.
    assert len(_gen("cs").decode_format(["%w0"], 22)) == 4
