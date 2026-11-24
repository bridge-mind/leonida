"""image_store dmask and format encoding: channel packing, sparse writes, and descriptor format decoding.

RDNA2 ISA, MIMG DMASK: for writes, which components are written with data from VGPRs; enabled components come
from consecutive VGPRs (DMASK = 1001: red in VGPRn, alpha in VGPRn+1). docs/design.md "Typed image stores":
"writes the dmask components (missing ones = 0)".

Covers:
  - 256-bit AMD RDNA2 / GFX10 Image Resource Descriptor (T#) decoding and format bitfield [28:20]
  - Reference oracle for 4-bit dmask component assignment and VGPR popcount packing
  - Sparse target memory writes where unmasked channels retain their previous state
  - Numeric format classification: *_UINT -> uint view, *_SINT -> sint view, otherwise -> float view
  - Verification of SPIR-V code generation across all 16 dmask permutations
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools")
sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402

MASK4 = 0xF
MASK9 = 0x1FF
MASK13 = 0x1FFF
MASK14 = 0x3FFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


class ImageResourceDescriptor:
    """256-bit AMD RDNA2 / GFX10 Image Resource Descriptor (T#, 8 SGPRs)."""

    def __init__(
        self,
        d0: int = 0,
        d1: int = 0,
        d2: int = 0,
        d3: int = 0,
        d4: int = 0,
        d5: int = 0,
        d6: int = 0,
        d7: int = 0,
    ) -> None:
        self.d0 = d0 & MASK32
        self.d1 = d1 & MASK32
        self.d2 = d2 & MASK32
        self.d3 = d3 & MASK32
        self.d4 = d4 & MASK32
        self.d5 = d5 & MASK32
        self.d6 = d6 & MASK32
        self.d7 = d7 & MASK32

    @classmethod
    def from_fields(
        cls,
        base_address: int = 0,
        width: int = 1,
        height: int = 1,
        depth: int = 1,
        format_code: int = 0,
        min_lod: int = 0,
    ) -> ImageResourceDescriptor:
        d0 = base_address & MASK32
        base_hi = (base_address >> 32) & MASK16
        lod_field = (min_lod & 0xF) << 16
        fmt_field = (format_code & MASK9) << 20
        d1 = base_hi | lod_field | fmt_field
        w_enc = max(0, width - 1) & MASK14
        h_enc = max(0, height - 1) & MASK14
        d2 = w_enc | (h_enc << 16)
        d_enc = max(0, depth - 1) & MASK13
        d3 = d_enc
        return cls(d0, d1, d2, d3, 0, 0, 0, 0)

    @property
    def base_lo(self) -> int:
        return self.d0

    @property
    def base_hi(self) -> int:
        return self.d1 & MASK16

    @property
    def format_code(self) -> int:
        return (self.d1 >> 20) & MASK9

    @property
    def format_name(self) -> str:
        return rdna2spirv.GFX10_FORMATS.get(self.format_code, "UNKNOWN")

    @property
    def is_uint(self) -> bool:
        return self.format_name.endswith("_UINT")

    @property
    def is_sint(self) -> bool:
        return self.format_name.endswith("_SINT")

    @property
    def is_float(self) -> bool:
        return not (self.is_uint or self.is_sint)

    @property
    def numeric_class(self) -> int:
        if self.is_sint:
            return 2
        if self.is_uint:
            return 1
        return 0

    @property
    def width(self) -> int:
        return (self.d2 & MASK14) + 1

    @property
    def height(self) -> int:
        return ((self.d2 >> 16) & MASK14) + 1

    @property
    def depth(self) -> int:
        return (self.d3 & MASK13) + 1

    def as_words(self) -> tuple[int, int, int, int, int, int, int, int]:
        return (self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7)


class ImageStoreOracle:
    """Semantic reference oracle for RDNA2 image store operations."""

    def __init__(self, surface: dict[tuple[int, ...], list[int]] | None = None) -> None:
        self.surface = {} if surface is None else dict(surface)

    @staticmethod
    def popcount(dmask: int) -> int:
        return bin(dmask & MASK4).count("1")

    @classmethod
    def unpack_source_vgprs(cls, vdata_base: int, dmask: int, vgprs: dict[int, int] | list[int]) -> list[int]:
        """Maps enabled channels to consecutive source VGPRs, missing channels to 0."""
        reg_idx = vdata_base
        texel = [0, 0, 0, 0]
        for c in range(4):
            if (dmask >> c) & 1:
                val = vgprs[reg_idx] if isinstance(vgprs, (list, tuple)) else vgprs.get(reg_idx, 0)
                texel[c] = val & MASK32
                reg_idx += 1
            else:
                texel[c] = 0
        return texel

    @classmethod
    def get_source_vgpr_indices(cls, vdata_base: int, dmask: int) -> list[int]:
        """Returns the list of source VGPR register numbers read by an image_store instruction."""
        indices = []
        cur = vdata_base
        for c in range(4):
            if (dmask >> c) & 1:
                indices.append(cur)
                cur += 1
        return indices

    def store_sparse(
        self,
        coord: tuple[int, ...],
        vdata_base: int,
        dmask: int,
        vgprs: dict[int, int] | list[int],
        default_texel: list[int] | None = None,
    ) -> list[int]:
        """Applies sparse write: only channels enabled in dmask are updated in target surface."""
        if coord not in self.surface:
            self.surface[coord] = list(default_texel) if default_texel is not None else [0, 0, 0, 0]
        cur = self.surface[coord]
        reg_idx = vdata_base
        for c in range(4):
            if (dmask >> c) & 1:
                val = vgprs[reg_idx] if isinstance(vgprs, (list, tuple)) else vgprs.get(reg_idx, 0)
                cur[c] = val & MASK32
                reg_idx += 1
        return list(cur)


def texels(line: str) -> dict[str, list[int | None]]:
    """Analyzes Gen.image_store output and returns texel components for float, uint, and sint views."""
    ins = parse_line(line)
    g = rdna2spirv.Gen("cs")
    dim = next(m for m in ins["mods"] if m.startswith("dim:"))
    g.image_store(ins["args"], ins["mods"], dim)
    consts = {name: value for value, name in g.consts.items()}
    src: dict[str, tuple[str, int]] = {}
    out: dict[str, list[int | None]] = {}
    for line_ in g.body:
        w = line_.split()
        if len(w) < 5 or w[1] != "=":
            continue
        res, op, ty, args = w[0], w[2], w[3], w[4:]
        if op == "OpAccessChain" and args[0] == "%vgpr":
            src[res] = ("p", consts[args[1]])
        elif op == "OpLoad" and src.get(args[0], (None,))[0] == "p":
            src[res] = ("v", src[args[0]][1])
        elif op == "OpBitcast" and args[0] in src:
            src[res] = src[args[0]]
        elif op == "OpBitcast" and args[0] in consts:
            consts[res] = consts[args[0]]
        elif op == "OpCompositeConstruct" and ty in ("%v4float", "%v4uint_st", "%v4int_st"):
            out[ty] = [src[x][1] if x in src and src[x][0] == "v" else consts.get(x) for x in args]
    return out


class ImageStoreDmask(unittest.TestCase):
    def check(self, line: str, want: list[int | None]) -> None:
        got = texels(line)
        self.assertEqual(len(got), 3)
        for ty, t in got.items():
            with self.subTest(view=ty):
                self.assertEqual(t, want)

    def test_full_mask(self) -> None:
        self.check("image_store v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm", [4, 5, 6, 7])

    def test_low_mask(self) -> None:
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm", [4, 5, 0, 0])

    def test_blue_only(self) -> None:
        self.check("image_store v4, v[0:1], s[8:15] dmask:0x4 dim:SQ_RSRC_IMG_2D unorm", [0, 0, 4, 0])

    def test_red_and_alpha(self) -> None:
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm", [4, 0, 0, 5])

    def test_empty_mask(self) -> None:
        self.check("image_store v4, v[0:1], s[8:15] dmask:0x0 dim:SQ_RSRC_IMG_2D unorm", [0, 0, 0, 0])


class TestImageResourceDescriptor(unittest.TestCase):
    def test_descriptor_fields_packing(self) -> None:
        desc = ImageResourceDescriptor.from_fields(
            base_address=0x123456789ABC,
            width=1920,
            height=1080,
            depth=64,
            format_code=5,  # GFX10_FORMAT_8_UINT
            min_lod=2,
        )
        self.assertEqual(desc.width, 1920)
        self.assertEqual(desc.height, 1080)
        self.assertEqual(desc.depth, 64)
        self.assertEqual(desc.format_code, 5)
        self.assertEqual(desc.format_name, "GFX10_FORMAT_8_UINT")
        self.assertTrue(desc.is_uint)
        self.assertFalse(desc.is_sint)
        self.assertFalse(desc.is_float)
        self.assertEqual(desc.numeric_class, 1)

    def test_descriptor_sint_format(self) -> None:
        desc = ImageResourceDescriptor.from_fields(format_code=6)  # GFX10_FORMAT_8_SINT
        self.assertEqual(desc.format_name, "GFX10_FORMAT_8_SINT")
        self.assertFalse(desc.is_uint)
        self.assertTrue(desc.is_sint)
        self.assertFalse(desc.is_float)
        self.assertEqual(desc.numeric_class, 2)

    def test_descriptor_float_format(self) -> None:
        desc = ImageResourceDescriptor.from_fields(format_code=1)  # GFX10_FORMAT_8_UNORM
        self.assertEqual(desc.format_name, "GFX10_FORMAT_8_UNORM")
        self.assertFalse(desc.is_uint)
        self.assertFalse(desc.is_sint)
        self.assertTrue(desc.is_float)
        self.assertEqual(desc.numeric_class, 0)

    def test_descriptor_words_length(self) -> None:
        desc = ImageResourceDescriptor()
        words = desc.as_words()
        self.assertEqual(len(words), 8)
        self.assertTrue(all(isinstance(w, int) for w in words))


class TestImageStoreChannelMasking(unittest.TestCase):
    def test_all_16_dmask_permutations_oracle(self) -> None:
        oracle = ImageStoreOracle()
        vdata_base = 10
        vgprs = {10: 0x1111, 11: 0x2222, 12: 0x3333, 13: 0x4444}
        for dmask in range(16):
            with self.subTest(dmask=hex(dmask)):
                pop = oracle.popcount(dmask)
                indices = oracle.get_source_vgpr_indices(vdata_base, dmask)
                self.assertEqual(len(indices), pop)
                if pop > 0:
                    self.assertEqual(indices, list(range(vdata_base, vdata_base + pop)))
                texel = oracle.unpack_source_vgprs(vdata_base, dmask, vgprs)
                for c in range(4):
                    if (dmask >> c) & 1:
                        self.assertNotEqual(texel[c], 0)
                    else:
                        self.assertEqual(texel[c], 0)

    def test_sparse_channel_write_retention(self) -> None:
        oracle = ImageStoreOracle()
        coord = (10, 20)
        oracle.surface[coord] = [0xAA, 0xBB, 0xCC, 0xDD]

        # Store only channel G (dmask 0x2)
        oracle.store_sparse(coord, vdata_base=0, dmask=0x2, vgprs={0: 0x99})
        self.assertEqual(oracle.surface[coord], [0xAA, 0x99, 0xCC, 0xDD])

        # Store channels R and A (dmask 0x9)
        oracle.store_sparse(coord, vdata_base=4, dmask=0x9, vgprs={4: 0x11, 5: 0x44})
        self.assertEqual(oracle.surface[coord], [0x11, 0x99, 0xCC, 0x44])

        # Store channels G and B (dmask 0x6)
        oracle.store_sparse(coord, vdata_base=2, dmask=0x6, vgprs={2: 0x77, 3: 0x88})
        self.assertEqual(oracle.surface[coord], [0x11, 0x77, 0x88, 0x44])

    def test_all_16_dmasks_in_spirv_gen(self) -> None:
        for dmask in range(16):
            with self.subTest(dmask=hex(dmask)):
                pop = ImageStoreOracle.popcount(dmask)
                vdata = f"v[{10}:{10 + max(0, pop - 1)}]" if pop > 1 else f"v{10}"
                line = f"image_store {vdata}, v[0:1], s[8:15] dmask:{hex(dmask)} dim:SQ_RSRC_IMG_2D unorm"
                got = texels(line)
                self.assertIn("%v4float", got)
                self.assertIn("%v4uint_st", got)
                self.assertIn("%v4int_st", got)
                reg = 10
                expected = []
                for c in range(4):
                    if (dmask >> c) & 1:
                        expected.append(reg)
                        reg += 1
                    else:
                        expected.append(0)
                for ty in ("%v4float", "%v4uint_st", "%v4int_st"):
                    self.assertEqual(got[ty], expected)


class TestImageStoreFormatEncoding(unittest.TestCase):
    def test_format_masks_constants(self) -> None:
        u_masks, si_masks = rdna2spirv.Gen.int_format_masks()
        self.assertEqual(len(u_masks), 16)
        self.assertEqual(len(si_masks), 16)
        # Check that GFX10_FORMAT_8_UINT (code 5) is recorded in uint mask
        word_idx = 5 >> 5
        bit_idx = 5 & 31
        self.assertTrue(bool((u_masks[word_idx] >> bit_idx) & 1))
        # Check that GFX10_FORMAT_8_SINT (code 6) is recorded in sint mask
        word_idx = 6 >> 5
        bit_idx = 6 & 31
        self.assertTrue(bool((si_masks[word_idx] >> bit_idx) & 1))


if __name__ == "__main__":
    unittest.main()
