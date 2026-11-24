"""The multi-component `_d16_hi_` MUBUF format spellings are not RDNA instructions and stay Unsupported.

`buffer_load_format_d16_hi_xy`, `buffer_store_format_d16_hi_xy` and their `_xyz`/`_xyzw` forms name a high-half
version of the packed 16-bit format load/store. RDNA has no such instruction: the only format D16 high-half forms
are the single-component `BUFFER_LOAD_FORMAT_D16_HI_X` and `BUFFER_STORE_FORMAT_D16_HI_X`, the `x` forms with the
one component in bits 31:16 instead of 15:0 (AMD's machine-readable RDNA1 and RDNA2 ISA XML, MUBUF opcodes 0x026
and 0x027; the multi-component D16 formats are 0x080-0x087 with no high-half counterpart). LLVM's gfx10 through
gfx12 MUBUF tables define the same two, and the console's gfx1013 disassembles through the gfx10 table, so no
listing can carry a multi-component `_d16_hi_` form (the 25,735-shader corpus has none, reported in #4084).

The translator refuses an unknown opcode by name, so these must keep raising Unsupported rather than inventing a
placement for component 1; #1885 (SHADER-320) adds the `_x` forms, which do exist. CPU only, hand-written listings,
no GPU and no game files.
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
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# 4 user SGPRs (s[0:3] = the buffer V#), 64x1x1 threads.
CS_FSSM = fssm(0x8090, (64, 1, 1))
VSHARP = ["s_mov_b32 s0, 0", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000", "s_mov_b32 s3, 0x31016FAC"]

HIGH_HALF = ("buffer_load_format_d16_hi_xy", "buffer_load_format_d16_hi_xyz",
             "buffer_load_format_d16_hi_xyzw", "buffer_store_format_d16_hi_xy",
             "buffer_store_format_d16_hi_xyz", "buffer_store_format_d16_hi_xyzw")
PACKED = ("buffer_load_format_d16_xy", "buffer_load_format_d16_xyz", "buffer_load_format_d16_xyzw",
          "buffer_store_format_d16_xy", "buffer_store_format_d16_xyz", "buffer_store_format_d16_xyzw")


def translate(op):
    return support.translate_lines("cs_d16_hi_absent", VSHARP + [
        "v_mov_b32_e32 v1, 0x100",              # vaddr
        "v_mov_b32_e32 v5, 0xBEEF1234",         # vdata
        op + " v5, v1, s[0:3], 0 offen",
    ], fssm=CS_FSSM)


class MultiComponentD16Hi(unittest.TestCase):
    def test_multi_component_hi_spellings_are_refused(self):
        for op in HIGH_HALF:
            with self.subTest(op=op), self.assertRaisesRegex(rdna2spirv.Unsupported, op):
                translate(op)

    def test_the_packed_forms_without_hi_translate(self):
        # Control: the same widths without `_hi_` are real instructions, so the refusal above is about the form.
        for op in PACKED:
            with self.subTest(op=op):
                self.assertTrue(translate(op))


if __name__ == "__main__":
    unittest.main()
