import pytest
"""SHADER-15: the translator uses the FSSM pixel-input table (interpolant count, fp16 mask).

All blobs and shaders here are built by hand; no game data is used, and nothing needs
llvm-mc, spirv-as/val or a GPU: the .s listings are written directly in the disassembler
listing format, and values are checked with tests/semantic/spirv_mini_interp.py.

Rule under test (tools/rdna2spirv.py, PS input path):
  * When a sidecar fssm/<name>.fssm next to the disasm directory holds a non-empty
    pixel-input table (tools/fssm_header.py), every table position 0..count-1 is declared
    as a fragment input at that Location, including positions the code never reads (the
    table overcounts the code's distinct v_interp attrs in 775 of 12,101 corpus PS).
    Position in the table is the v_interp attr number, not the record's slot byte.
  * A 32-bit v_interp read (p1/p2/mov) of a position flagged 16-bit (flag bit 4) sees the
    value rounded to f16 (round to nearest, f16 denormals flushed to signed zero,
    and back to f32). Reads through the *_f16
    opcodes already narrow to f16 and are unchanged.
  * Without a sidecar, or with a sidecar holding no table, the output is byte-identical
    to the old code-driven path (no new declarations, no quantization, no new metadata).
"""

import gc
import os
import struct
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import rdna2spirv  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

# An empty DXBC container (no ISG1): pixel shaders that only read barycentrics through
# v_interp need no input signature.
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

FP16_FLAG = 0x10
# Read method, flags bits 0-1 (SHADER-181): every corpus record carries 1 (v_interp_mov) or 2 (v_interp_p1/p2).
FLAT = 1
LINEAR = 2


def fssm_blob(records):
    """A synthetic FSSM header holding `records` = [(semantic, mask, index, slot, flags)]."""
    data = bytearray(max(0x140, 0x138 + 8 * len(records)))
    data[0:4] = b"FSSM"
    for i, (semantic, mask, index, slot, flags) in enumerate(records):
        struct.pack_into("<8B", data, 0x138 + 8 * i, 6, semantic, mask, index, 0, 0, slot, flags)
    return bytes(data)


def write_case(tmp_path, name, body, fssm=None):
    """Write a synthetic shader case: disasm/<name>.s, <name>.dxbc and optionally fssm/<name>.fssm."""
    disasm = tmp_path / "disasm"
    disasm.mkdir(exist_ok=True)
    (tmp_path / f"{name}.dxbc").write_bytes(STUB_DXBC)
    if fssm is not None:
        (tmp_path / "fssm").mkdir(exist_ok=True)
        (tmp_path / "fssm" / f"{name}.fssm").write_bytes(fssm)
    src = disasm / f"{name}.s"
    src.write_text(f"; {name} synthetic\n" + body)
    return str(src)


def translate(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)  # translate() leaves its .dxbc handle to the GC
        result = rdna2spirv.translate(path)
        gc.collect()
    return result


def mrt0_words(spvasm, **run_options):
    out = run_fragment(spvasm, **run_options)
    assert list(out) == ["%out_mrt0"], f"expected exactly one output, got {sorted(out)}"
    return tuple(out["%out_mrt0"])


def f16_roundtrip_bits(word):
    """Independent oracle: f32 -> f16 (round to nearest even, f16 denormals flushed) -> f32, as bits.

    OpQuantizeToF16 returns +0/-0 for a magnitude below the smallest normal f16, so the oracle
    flushes exactly like the interpreter and the hardware are supposed to.
    """
    as_float = struct.unpack("<f", struct.pack("<I", word))[0]
    half = struct.unpack("<H", struct.pack("<e", as_float))[0]
    if half & 0x7C00 == 0:
        half &= 0x8000
    back = struct.unpack("<e", struct.pack("<H", half))[0]
    return struct.unpack("<I", struct.pack("<f", back))[0]


READ_ATTR0 = (
    "\tv_interp_p1_f32_e32 v2, v0, attr0.x\n"
    "\tv_interp_p2_f32_e32 v2, v1, attr0.x\n"
    "\texp mrt0 v2, v2, v2, v2 done vm\n"
    "\ts_endpgm\n"
)

READ_ATTR1 = (
    "\tv_interp_p1_f32_e32 v2, v0, attr1.x\n"
    "\tv_interp_p2_f32_e32 v2, v1, attr1.x\n"
    "\texp mrt0 v2, v2, v2, v2 done vm\n"
    "\ts_endpgm\n"
)

READ_ATTR0_THEN_ATTR1 = (
    "\tv_interp_p1_f32_e32 v2, v0, attr0.x\n"
    "\tv_interp_p2_f32_e32 v2, v1, attr0.x\n"
    "\tv_interp_p1_f32_e32 v3, v0, attr1.y\n"
    "\tv_interp_p2_f32_e32 v3, v1, attr1.y\n"
    "\texp mrt0 v2, v3, v2, v3 done vm\n"
    "\ts_endpgm\n"
)

MOV_ATTR1 = (
    "\tv_interp_mov_f32_e32 v2, p0, attr1.x\n"
    "\texp mrt0 v2, v2, v2, v2 done vm\n"
    "\ts_endpgm\n"
)


def test_table_predeclares_unread_interpolants(tmp_path):
    """Positions the code never reads are still declared (Location = table position)."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR), (4, 0xF, 2, 2, LINEAR)])
    spv = translate(write_case(tmp_path, "ps_pix_table", READ_ATTR0, fssm=table))
    lines = [line.strip() for line in spv.splitlines()]
    for n in (0, 1, 2):
        assert f"OpDecorate %in_attr{n} Location {n}" in lines
        assert f"%in_attr{n} = OpVariable %_ptr_Input_v4 Input" in lines
    assert "OpDecorate %in_attr0 Flat" not in lines  # read through p1/p2: interpolated
    assert "OpDecorate %in_attr1 Flat" in lines  # never read: Flat and unused
    assert "OpDecorate %in_attr2 Flat" in lines


def test_table_position_not_slot_is_the_attr_number(tmp_path):
    """The record's position is the v_interp attr number even when the slot bytes disagree."""
    table = fssm_blob([(4, 0xF, 0, 7, LINEAR), (4, 0xF, 1, 3, LINEAR | FP16_FLAG)])
    spv = translate(write_case(tmp_path, "ps_pix_slots", READ_ATTR1, fssm=table))
    assert "OpQuantizeToF16" in spv  # attr1 = table position 1 = flagged, whatever its slot says


def test_fp16_read_rounds_to_nearest(tmp_path):
    """A 32-bit v_interp read of a flagged position sees the f16-rounded value."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR | FP16_FLAG)])
    spv = translate(write_case(tmp_path, "ps_pix_fp16", READ_ATTR1, fssm=table))
    assert "OpQuantizeToF16" in spv  # single quantize, not a pack/unpack round-trip (folded by drivers)
    assert "PackHalf2x16" not in spv
    f32_one_point_one = struct.unpack("<I", struct.pack("<f", 1.1))[0]
    want = f16_roundtrip_bits(f32_one_point_one)
    assert want != f32_one_point_one  # the fixture value is not f16-exact, so the check is meaningful
    got = mrt0_words(spv, inputs={"%in_attr1": (1.1, 0.0, 0.0, 0.0)})
    assert got == (want,) * 4


def test_fp16_subnormal_read_flushes_to_zero(tmp_path):
    """A magnitude too small for a normal f16 reads as +0/-0, not as an f16 subnormal.

    SPIR-V's OpQuantizeToF16 returns signed zero below the smallest normal f16 (6.1e-5);
    keeping the subnormal (struct.pack('<e') does) would be more precise than a driver.
    """
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR | FP16_FLAG)])
    spv = translate(write_case(tmp_path, "ps_pix_subnormal", READ_ATTR1, fssm=table))
    zero = struct.unpack("<I", struct.pack("<f", 0.0))[0]
    neg_zero = struct.unpack("<I", struct.pack("<f", -0.0))[0]
    assert f16_roundtrip_bits(struct.unpack("<I", struct.pack("<f", 1e-6))[0]) == zero
    assert mrt0_words(spv, inputs={"%in_attr1": (1e-6, 0.0, 0.0, 0.0)}) == (zero,) * 4
    assert mrt0_words(spv, inputs={"%in_attr1": (-1e-6, 0.0, 0.0, 0.0)}) == (neg_zero,) * 4


def test_fp16_mov_read_also_rounds(tmp_path):
    """v_interp_mov_f32 of a flagged position rounds too, and stays Flat."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, FLAT | FP16_FLAG)])
    spv = translate(write_case(tmp_path, "ps_pix_fmov", MOV_ATTR1, fssm=table))
    lines = [line.strip() for line in spv.splitlines()]
    assert "OpDecorate %in_attr1 Flat" in lines  # mov-only: Flat, as before
    f32_one_point_one = struct.unpack("<I", struct.pack("<f", 1.1))[0]
    got = mrt0_words(spv, inputs={"%in_attr1": (1.1, 0.0, 0.0, 0.0)})
    assert got == (f16_roundtrip_bits(f32_one_point_one),) * 4


def test_no_table_leaves_values_and_declarations_unchanged(tmp_path):
    """Control: without a sidecar there is no quantization and only code-read inputs exist."""
    src = write_case(tmp_path, "ps_pix_ctrl", READ_ATTR1)
    spv = translate(src)
    assert "PackHalf2x16" not in spv
    assert "%in_attr0 = OpVariable" not in spv
    assert "%in_attr2 = OpVariable" not in spv
    f32_one_point_one = struct.unpack("<I", struct.pack("<f", 1.1))[0]
    got = mrt0_words(spv, inputs={"%in_attr1": (1.1, 0.0, 0.0, 0.0)})
    assert got == (f32_one_point_one,) * 4  # exact f32, not rounded


def test_empty_table_is_the_old_path(tmp_path):
    """Control: a sidecar whose first byte is not 6 holds no table and changes nothing."""
    import pathlib

    src = write_case(tmp_path, "ps_pix_empty", READ_ATTR1, fssm=b"FSSM" + bytes(0x140 - 4))
    with tempfile.TemporaryDirectory() as other:
        plain = translate(write_case(pathlib.Path(other), "ps_pix_empty", READ_ATTR1))
    assert translate(src) == plain


def test_agreeing_shader_is_byte_identical_with_table(tmp_path):
    """A shader reading exactly the table's attrs with no fp16 flags translates as before."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR)])
    with_table = translate(write_case(tmp_path, "ps_pix_same", READ_ATTR0_THEN_ATTR1, fssm=table))
    import pathlib

    with tempfile.TemporaryDirectory() as other:
        other = pathlib.Path(other)
        without_table = translate(write_case(other, "ps_pix_same", READ_ATTR0_THEN_ATTR1))
    assert with_table == without_table


def test_out_of_table_read_still_declares(tmp_path):
    """A read past the table declares an input instead of failing (no new rejections)."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR)])
    body = (
        "\tv_interp_p1_f32_e32 v2, v0, attr5.x\n"
        "\tv_interp_p2_f32_e32 v2, v1, attr5.x\n"
        "\texp mrt0 v2, v2, v2, v2 done vm\n"
        "\ts_endpgm\n"
    )
    spv = translate(write_case(tmp_path, "ps_pix_beyond", body, fssm=table))
    assert "OpDecorate %in_attr5 Location 5" in [line.strip() for line in spv.splitlines()]


def test_metadata_records_table(tmp_path):
    """shader_meta carries the interpolant count and the flagged positions; absent without a table."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR | FP16_FLAG), (4, 0xF, 2, 2, LINEAR)])
    spv = translate(write_case(tmp_path, "ps_pix_meta", READ_ATTR0, fssm=table))
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    assert meta["ps_interpolants"] == 3
    assert meta["ps_fp16_interpolants"] == [1]
    assert meta["ps_flat_interpolants"] == []

    import pathlib

    with tempfile.TemporaryDirectory() as other:
        other = pathlib.Path(other)
        spv_plain = translate(write_case(other, "ps_pix_meta", READ_ATTR0))
    meta_plain = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv_plain)
    assert "ps_interpolants" not in meta_plain
    assert "ps_fp16_interpolants" not in meta_plain


# SHADER-181: the FSSM read-method bits must agree with how the code reads each position.

FLAT_THEN_INTERP = (
    "\tv_interp_mov_f32_e32 v2, p0, attr0.x\n"
    "\tv_interp_p1_f32_e32 v3, v0, attr1.y\n"
    "\tv_interp_p2_f32_e32 v3, v1, attr1.y\n"
    "\texp mrt0 v2, v3, v2, v3 done vm\n"
    "\ts_endpgm\n"
)

F16_READ = (
    "\tv_interp_p1ll_f16 v2, v0, attr0.x\n"
    "\tv_interp_p2_f16 v2, v1, attr0.x, v2\n"
    "\tv_cvt_f32_f16_e32 v2, v2\n"
    "\texp mrt0 v2, v2, v2, v2 done vm\n"
    "\ts_endpgm\n"
)


def test_read_methods_agreeing_with_the_code_translate(tmp_path):
    """Mode 1 read by v_interp_mov and mode 2 read by p1/p2 is the corpus case (12,101 of 12,101 PS)."""
    table = fssm_blob([(4, 0xF, 0, 0, FLAT), (4, 0xF, 1, 1, LINEAR)])
    spv = translate(write_case(tmp_path, "ps_mode_ok", FLAT_THEN_INTERP, fssm=table))
    lines = [line.strip() for line in spv.splitlines()]
    assert "OpDecorate %in_attr0 Flat" in lines
    assert "OpDecorate %in_attr1 Flat" not in lines
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    assert meta["ps_flat_interpolants"] == [0]


def test_f16_interp_of_a_linear_position_translates(tmp_path):
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR | FP16_FLAG)])
    translate(write_case(tmp_path, "ps_mode_f16", F16_READ, fssm=table))


@pytest.mark.parametrize("flags, body", [
    (LINEAR, MOV_ATTR1),            # flat read of a linear position
    (FLAT, READ_ATTR1),             # interpolated read of a flat position
    (0, READ_ATTR1),                # mode 0: never seen, meaning unknown
    (3, READ_ATTR1),                # mode 3: never seen, meaning unknown
    (3, MOV_ATTR1),
])
def test_disagreeing_or_unknown_read_method_is_refused(tmp_path, flags, body):
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, flags)])
    with pytest.raises(rdna2spirv.Unsupported, match="PS input 1 read method"):
        translate(write_case(tmp_path, "ps_mode_bad", body, fssm=table))


def test_unknown_read_method_on_an_unread_position_is_not_refused(tmp_path):
    """Only positions the code reads are checked; an unread record with mode 0 changes nothing."""
    table = fssm_blob([(4, 0xF, 0, 0, LINEAR), (4, 0xF, 1, 1, LINEAR), (4, 0xF, 2, 2, 0)])
    translate(write_case(tmp_path, "ps_mode_unread", READ_ATTR1, fssm=table))


def test_read_method_check_unit():
    def ins(op, a):
        return {"op": op, "args": [{"k": "vgpr", "r": 2}, {"k": "attr", "a": a, "c": 0}], "mods": {}}
    rdna2spirv.check_read_methods([1, 2], [ins("v_interp_mov_f32_e32", 0), ins("v_interp_p1_f32_e32", 1),
                                                  ins("v_interp_p1ll_f16", 1), ins("v_interp_p1_f32_e32", 7)])
    rdna2spirv.check_read_methods([], [ins("v_interp_p1_f32_e32", 0)])
    with pytest.raises(rdna2spirv.Unsupported):
        rdna2spirv.check_read_methods([1, 2], [ins("v_interp_p2_f16", 0)])
