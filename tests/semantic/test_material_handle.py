"""Regression fixture for a per-draw handle word selecting an eight-word image descriptor."""
import unittest

from spirv_mini_interp import run_fragment
from support import translate_lines


class MaterialHandleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # s0:s1 points to the root table, s2:s3 to the image heap, and s4 is
        # a byte offset into the raw buffer. Export descriptor words as markers.
        body = [
            "s_load_dwordx4 s[16:19], s[0:1], 0x40",
            "s_waitcnt lgkmcnt(0)",
            "s_add_u32 s7, s4, 8",
            "s_buffer_load_dword s8, s[16:19], s7",
            "s_waitcnt lgkmcnt(0)",
            "s_lshl_b32 s9, s8, 5",
            "s_load_dwordx8 s[24:31], s[2:3], s9",
            "s_waitcnt lgkmcnt(0)",
        ]
        body += [f"v_mov_b32_e32 v{i}, s{24+i}" for i in range(8)]
        body += ["exp mrt0 v0, v1, v2, v3", "exp mrt1 v4, v5, v6, v7 done vm", "s_endpgm"]
        cls.shader = translate_lines("ps_material_handle", body)

    def test_effect_word_selects_descriptor_for_each_draw(self):
        root, raw, heap = 0x1000, 0x2000, 0x3000
        # Build two material blocks from the same authored rows. One extra
        # drawable-only parameter shifts the normal map from raw row 3 to effect word 2.
        effect = ("paint", "detail", "normal")
        authored = (("paint", 2), ("extra", 4), ("detail", 6), ("normal", 9))
        handles = dict(authored)
        memory = {}
        for i, word in enumerate((raw, 0, 128, 0)):
            memory[root + 0x40 + i * 4] = word
        descriptors = {}
        for handle in handles.values():
            descriptors[handle] = tuple(0x3f000000 + 0x100 * handle + i for i in range(8))
            for i, word in enumerate(descriptors[handle]):
                memory[heap + 32 * handle + 4 * i] = word

        for base, layout, expected_handle in (
            (0x14, effect, 9),
            (0x38, tuple(handles), 6),
        ):
            with self.subTest(draw_base=base, layout=layout):
                for i, name in enumerate(layout):
                    memory[raw + base + i * 4] = handles[name]
                out = run_fragment(self.shader, push_constants=[root, 0, heap, 0, base], memory=memory)
                words = tuple(out["%out_mrt0"]) + tuple(out["%out_mrt1"])
                self.assertEqual(words, descriptors[expected_handle])


if __name__ == "__main__":
    unittest.main()
