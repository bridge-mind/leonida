"""The format table of runtime/src/gfx10_descriptors.h against tools/gfx10_formats.json.

The C++ header keeps its own copy of the GFX10_FORMAT codes it maps to Vulkan (the runtime does not read JSON for
this). This test reads the table out of the header and checks every row: the code has the name its comment says,
in the list the translator uses, and the Vulkan format beside it has that name's channel sizes and numeric type.
A GFX10 format names its channel sizes from the top bit down; a Vulkan format does so only when it is packed.

    python -I tests/test_gfx10_descriptors.py
    python -m pytest tests/test_gfx10_descriptors.py -q
"""
import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "runtime", "src", "gfx10_descriptors.h")
FORMATS = os.path.join(ROOT, "tools", "gfx10_formats.json")

ROW = re.compile(r"^\s*\{(\d+), VK_FORMAT_(\w+), channels_(\w+)\},\s*// GFX10_FORMAT_(\w+)\s*$")
ANY_ROW = re.compile(r"^\s*\{\d+, VK_FORMAT_")

# Depth-stencil formats, which Mesa's table maps by hand (src/amd/common/gfx10_format_table.py, HARDCODED).
BY_HAND = {"D24_UNORM_S8_UINT": "8_24_UNORM", "D32_SFLOAT_S8_UINT": "X24_8_32_FLOAT"}


def table():
    """[(code, Vulkan name without VK_FORMAT_, channel order, GFX10 name without GFX10_FORMAT_)], in file order."""
    rows, loose = [], 0
    with open(HEADER, encoding="utf-8") as f:
        for line in f:
            loose += bool(ANY_ROW.match(line))
            m = ROW.match(line)
            if m:
                rows.append((int(m.group(1)), m.group(2), m.group(3), m.group(4)))
    return rows, loose


def gfx10_name(vulkan):
    """The GFX10 format name a Vulkan format name implies."""
    if vulkan in BY_HAND:
        return BY_HAND[vulkan]
    m = re.fullmatch(r"BC(\d)H?(?:_RGBA?)?_(\w+)_BLOCK", vulkan)
    if m:
        return f"BC{m.group(1)}_{m.group(2)}"
    packed = re.search(r"_PACK\d+$", vulkan)
    layout, numeric = (vulkan[:packed.start()] if packed else vulkan).rsplit("_", 1)
    sizes = re.findall(r"[RGBAEXDS](\d+)", layout)
    if not packed:
        sizes.reverse()  # memory order, lowest byte first -> from the top bit down
    return "_".join(sizes + ["FLOAT" if numeric in ("SFLOAT", "UFLOAT") else numeric])


class FormatTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.loose = table()
        with open(FORMATS, encoding="utf-8") as f:
            cls.names = {int(k): v[len("GFX10_FORMAT_"):] for k, v in json.load(f).items()}

    def test_every_row_is_read(self):
        self.assertGreater(len(self.rows), 80)
        self.assertEqual(len(self.rows), self.loose, "a table row does not have the form this test parses")

    def test_codes_have_the_names_in_the_comments(self):
        for code, vulkan, _order, name in self.rows:
            self.assertEqual(self.names.get(code), name, f"{{{code}, VK_FORMAT_{vulkan}}}")

    def test_vulkan_formats_have_the_layout_of_their_code(self):
        for code, vulkan, _order, name in self.rows:
            self.assertEqual(gfx10_name(vulkan), name, f"{{{code}, VK_FORMAT_{vulkan}}}")

    def test_rows_are_sorted_and_each_vulkan_format_appears_once(self):
        codes = [r[0] for r in self.rows]
        self.assertEqual(codes, sorted(codes))
        formats = [r[1] for r in self.rows]
        self.assertEqual(len(formats), len(set(formats)))

    def test_what_the_translator_and_its_tests_rely_on(self):
        by_vulkan = {r[1]: r[0] for r in self.rows}
        # cs_imgstore.asm: word 1 = FORMAT << 20 with 0x38, 0x14 and 0x15.
        self.assertEqual(by_vulkan["R8G8B8A8_UNORM"], 0x38)
        self.assertEqual(by_vulkan["R32_UINT"], 0x14)
        self.assertEqual(by_vulkan["R32_SINT"], 0x15)
        # The translator picks the uint or sint view from the code's name (rdna2spirv.py, int_format_masks). A
        # depth-stencil format is in the table as its depth plane, whatever its stencil plane's type.
        for code, vulkan, _order, name in self.rows:
            if vulkan in BY_HAND:
                continue
            for suffix in ("_UINT", "_SINT"):
                self.assertEqual(name.endswith(suffix), vulkan.endswith(suffix) or f"{suffix}_PACK" in vulkan,
                                 f"{{{code}, VK_FORMAT_{vulkan}}}")


if __name__ == "__main__":
    unittest.main()
