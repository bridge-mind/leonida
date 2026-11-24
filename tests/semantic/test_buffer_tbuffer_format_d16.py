"""buffer_load_format_d16_* and tbuffer_load_format_d16_* (GFX10 16-bit packed format loads) in compute shaders.

Covers all 8 instructions (buffer_load_format_d16_{x,xy,xyz,xyzw} and tbuffer_load_format_d16_{x,xy,xyz,xyzw})
through rdna2spirv.translate and spirv_mini_interp.run_fragment:
- GFX10 16-bit packing (_x preserves high 16 bits, _xy packs x | (y << 16), _xyz/_xyzw pack into v[a:a+1] with
  _xyz preserving v[a+1][31:16]),
- float/UNORM formats (32_32_32_32_FLOAT, 16_16_16_16_FLOAT, 8_8_8_8_UNORM) and integer formats
  (16_16_16_16_UINT, 16_16_16_16_SINT including negative 16-bit values),
- tbuffer format:[...] modifier precedence over the V#'s FORMAT field,
- out-of-bounds (num_records) behavior, and
- negative controls on the generated SPIR-V text (shift-16 packing, 0xFFFF/0xFFFF0000 masks).
"""
import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import support     # noqa: E402


def fssm(rsrc2, group):
    """Build a minimal FSSM compute header with COMPUTE_PGM_RSRC2 and thread-group size."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# 8 user SGPRs (s[0:3] = input V#, s[4:7] = output V#), workgroup id in s8, 64x1x1 threads.
CS_FSSM = fssm(0x8090, (64, 1, 1))

# Format codes from rdna2spirv.FORMATS / GFX10_FORMATS.
FMT_32_32_32_32_FLOAT = 77
FMT_16_16_16_16_FLOAT = 71
FMT_8_8_8_8_UNORM = 56
FMT_16_16_16_16_UINT = 69
FMT_16_16_16_16_SINT = 70
FMT_32_FLOAT = 22


def make_vsharp(base, num_records, fmt_code, dst_sel=0xFAC, oob_select=3):
    """Build a 4-dword GFX10 buffer descriptor (V#) with RAW OOB_SELECT=3 and identity swizzle."""
    w0 = base & 0xFFFFFFFF
    w1 = (base >> 32) & 0xFFFF
    w2 = num_records & 0xFFFFFFFF
    w3 = (oob_select << 28) | (1 << 24) | ((fmt_code & 0x7F) << 12) | (dst_sel & 0xFFF)
    return [w0, w1, w2, w3]


# Output V# in s[4:7]: base 0, num_records 0xFFFFFFFF, 32_FLOAT, OOB_SELECT 3.
OUT_VSHARP = make_vsharp(0, 0xFFFFFFFF, FMT_32_FLOAT)


def f32_bits(val):
    """IEEE-754 binary32 bit pattern for a Python float."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def f16_bits(val):
    """IEEE-754 binary16 bit pattern for a Python float."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def pack_u16(lo, hi):
    """Pack two 16-bit values into one 32-bit word: (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)."""
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_d16_load(op, fmt_mod, vsharp, memory, addr=0x100, init_v20=0xABCD1234, init_v21=0xDEAD5678):
    """Translate and run a single (t)buffer_load_format_d16_* instruction in compute, returning (spv, out_words)."""
    n_comp = len(op.rsplit("_", 1)[1])
    vdata = "v20" if n_comp <= 2 else "v[20:21]"
    mod_str = f" format:[{fmt_mod}]" if fmt_mod else ""
    lines = [
        f"v_mov_b32_e32 v20, {init_v20:#x}",
        f"v_mov_b32_e32 v21, {init_v21:#x}",
        f"v_mov_b32_e32 v2, {addr:#x}",
        f"{op} {vdata}, v2, s[0:3], 0{mod_str} offen",
        "v_mov_b32_e32 v5, 0x0",
        ("buffer_store_dword v20, v5, s[4:7], 0 offen" if n_comp <= 2
         else "buffer_store_dwordx2 v[20:21], v5, s[4:7], 0 offen"),
        "s_endpgm",
    ]
    spv = support.translate_lines(f"cs_{op}", lines, fssm=CS_FSSM)
    mem = dict(memory)
    support.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=vsharp + OUT_VSHARP + [0] * 24,
        memory=mem,
    )
    out = [mem.get(0, 0)] if n_comp <= 2 else [mem.get(0, 0), mem.get(4, 0)]
    return spv, out


class BufferFormatD16Test(unittest.TestCase):
    """Semantic tests for buffer_load_format_d16_{x,xy,xyz,xyzw} and tbuffer_load_format_d16_{x,xy,xyz,xyzw}."""

    def test_buffer_load_d16_all_widths_32f(self):
        mem = {
            0x100: f32_bits(1.5),
            0x104: f32_bits(-2.25),
            0x108: f32_bits(0.5),
            0x10C: f32_bits(4.0),
        }
        hx, hy, hz, hw = f16_bits(1.5), f16_bits(-2.25), f16_bits(0.5), f16_bits(4.0)
        vsharp = make_vsharp(0, 0x200, FMT_32_32_32_32_FLOAT)

        _, out_x = run_d16_load("buffer_load_format_d16_x", None, vsharp, mem)
        self.assertEqual(out_x, [pack_u16(hx, 0xABCD)])

        _, out_xy = run_d16_load("buffer_load_format_d16_xy", None, vsharp, mem)
        self.assertEqual(out_xy, [pack_u16(hx, hy)])

        _, out_xyz = run_d16_load("buffer_load_format_d16_xyz", None, vsharp, mem)
        self.assertEqual(out_xyz, [pack_u16(hx, hy), pack_u16(hz, 0xDEAD)])

        _, out_xyzw = run_d16_load("buffer_load_format_d16_xyzw", None, vsharp, mem)
        self.assertEqual(out_xyzw, [pack_u16(hx, hy), pack_u16(hz, hw)])

    def test_tbuffer_load_d16_all_widths_16f(self):
        hx, hy, hz, hw = f16_bits(1.0), f16_bits(-1.0), f16_bits(2.0), f16_bits(3.0)
        mem = {
            0x100: pack_u16(hx, hy),
            0x104: pack_u16(hz, hw),
        }
        vsharp = make_vsharp(0, 0x200, FMT_16_16_16_16_FLOAT)

        _, out_x = run_d16_load("tbuffer_load_format_d16_x", "BUF_FMT_16_16_16_16_FLOAT", vsharp, mem)
        self.assertEqual(out_x, [pack_u16(hx, 0xABCD)])

        _, out_xy = run_d16_load("tbuffer_load_format_d16_xy", "BUF_FMT_16_16_16_16_FLOAT", vsharp, mem)
        self.assertEqual(out_xy, [pack_u16(hx, hy)])

        _, out_xyz = run_d16_load("tbuffer_load_format_d16_xyz", "BUF_FMT_16_16_16_16_FLOAT", vsharp, mem)
        self.assertEqual(out_xyz, [pack_u16(hx, hy), pack_u16(hz, 0xDEAD)])

        _, out_xyzw = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_16_16_16_16_FLOAT", vsharp, mem)
        self.assertEqual(out_xyzw, [pack_u16(hx, hy), pack_u16(hz, hw)])

    def test_unorm8_d16_conversion(self):
        # 0x00 -> 0.0 (0x0000), 0xFF -> 1.0 (0x3C00), 0x80 -> 128/255 (~0.50196 -> 0x3804),
        # 0x40 -> 64/255 (~0.25098 -> 0x3404).
        mem = {0x100: 0x4080FF00}
        vsharp = make_vsharp(0, 0x200, FMT_8_8_8_8_UNORM)

        _, out_b = run_d16_load("buffer_load_format_d16_xyzw", None, vsharp, mem)
        self.assertEqual(out_b, [pack_u16(0x0000, 0x3C00), pack_u16(0x3804, 0x3404)])

        _, out_t = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_8_8_8_8_UNORM", vsharp, mem)
        self.assertEqual(out_t, out_b)

    def test_uint16_and_sint16_formats(self):
        # UINT: 16-bit unsigned values remain intact (including 0x8000 and 0xFFFF).
        mem_u = {
            0x100: pack_u16(0x0001, 0x1234),
            0x104: pack_u16(0x8000, 0xFFFF),
        }
        vsharp_u = make_vsharp(0, 0x200, FMT_16_16_16_16_UINT)
        _, out_bu = run_d16_load("buffer_load_format_d16_xyzw", None, vsharp_u, mem_u)
        self.assertEqual(out_bu, [pack_u16(0x0001, 0x1234), pack_u16(0x8000, 0xFFFF)])

        _, out_tu = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_16_16_16_16_UINT", vsharp_u, mem_u)
        self.assertEqual(out_tu, out_bu)

        # SINT: negative 16-bit values (-1 -> 0xFFFF, -32768 -> 0x8000, -123 -> 0xFF85, 32767 -> 0x7FFF)
        # remain intact at 16 bits.
        mem_s = {
            0x100: pack_u16(0xFFFF, 0x8000),
            0x104: pack_u16(0xFF85, 0x7FFF),
        }
        vsharp_s = make_vsharp(0, 0x200, FMT_16_16_16_16_SINT)
        _, out_bs = run_d16_load("buffer_load_format_d16_xyzw", None, vsharp_s, mem_s)
        self.assertEqual(out_bs, [pack_u16(0xFFFF, 0x8000), pack_u16(0xFF85, 0x7FFF)])

        _, out_ts = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_16_16_16_16_SINT", vsharp_s, mem_s)
        self.assertEqual(out_ts, out_bs)

    def test_tbuffer_format_modifier_overrides_vsharp_format(self):
        # V# declares 32_32_32_32_FLOAT (16-byte element), but tbuffer specifies BUF_FMT_16_16_16_16_UINT (8-byte).
        # If V#'s format were used, 0xFFFF/0x8000 would be interpreted as f32 denorm/NaN and flushed/converted;
        # with format:[BUF_FMT_16_16_16_16_UINT] taking precedence, the raw 16-bit integers are preserved.
        mem = {
            0x100: pack_u16(0x0007, 0x8000),
            0x104: pack_u16(0xFFFE, 0xFFFF),
        }
        vsharp_wrong_fmt = make_vsharp(0, 0x200, FMT_32_32_32_32_FLOAT)
        _, out_t = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_16_16_16_16_UINT", vsharp_wrong_fmt, mem)
        self.assertEqual(out_t, [pack_u16(0x0007, 0x8000), pack_u16(0xFFFE, 0xFFFF)])

    def test_out_of_bounds_returns_zero_and_preserves_high_half(self):
        # Element size of 16_16_16_16_UINT is 8 bytes. At addr=0x100, end is 0x108; num_records=0x104 is OOB.
        mem = {
            0x100: pack_u16(0x1111, 0x2222),
            0x104: pack_u16(0x3333, 0x4444),
        }
        vsharp_oob = make_vsharp(0, 0x104, FMT_16_16_16_16_UINT)

        _, out_x = run_d16_load("buffer_load_format_d16_x", None, vsharp_oob, mem)
        self.assertEqual(out_x, [pack_u16(0x0000, 0xABCD)])

        _, out_xyzw = run_d16_load("buffer_load_format_d16_xyzw", None, vsharp_oob, mem)
        self.assertEqual(out_xyzw, [0, 0])

        _, out_tx = run_d16_load("tbuffer_load_format_d16_x", "BUF_FMT_16_16_16_16_UINT", vsharp_oob, mem)
        self.assertEqual(out_tx, [pack_u16(0x0000, 0xABCD)])

        _, out_txyzw = run_d16_load("tbuffer_load_format_d16_xyzw", "BUF_FMT_16_16_16_16_UINT", vsharp_oob, mem)
        self.assertEqual(out_txyzw, [0, 0])

    def test_spirv_negative_controls(self):
        vsharp = make_vsharp(0, 0x200, FMT_16_16_16_16_UINT)
        mem = {0x100: 0x00020001, 0x104: 0x00040003}

        spv_x, _ = run_d16_load("buffer_load_format_d16_x", None, vsharp, mem)
        # write_lo16 must mask the previous high half with 0xFFFF0000 (4294901760) and OR in the new low half.
        self.assertIn("OpConstant %uint 4294901760", spv_x)
        self.assertIn("OpBitwiseAnd", spv_x)
        self.assertIn("OpBitwiseOr", spv_x)

        spv_xy, _ = run_d16_load("buffer_load_format_d16_xy", None, vsharp, mem)
        # Packing two 16-bit components shifts the second component left by 16 and masks/saturates to 0xFFFF (65535).
        self.assertIn("OpConstant %uint 16", spv_xy)
        self.assertIn("OpShiftLeftLogical", spv_xy)
        self.assertIn("OpConstant %uint 65535", spv_xy)
        self.assertIn("SClamp", spv_xy)
        self.assertIn("UMin", spv_xy)

        spv_t, _ = run_d16_load("tbuffer_load_format_d16_xy", "BUF_FMT_16_16_16_16_SINT", vsharp, mem)
        self.assertIn("OpConstant %uint 16", spv_t)
        self.assertIn("OpShiftLeftLogical", spv_t)
        self.assertIn("SClamp", spv_t)


if __name__ == "__main__":
    unittest.main()
