"""tools/sampler_lod_probe.py without a GPU: the descriptor fields sit where the runtime reads them, the measured model
gives the gfx1036 levels, and the kernel builds.

    python -I tests/test_sampler_lod_probe.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hw_oracle"))

import sampler_lod_probe as p  # noqa: E402

# What gfx1036 (Radeon iGPU, AMD 26.10.1) read for every row, wave32 and wave64 alike:
# tools/sampler_lod_probe.py --oracle hw. All 43 are pinned, so a change to rdna2_level shows without a GPU.
MEASURED = {
    "calib_lod0": 0, "calib_lod1": 1, "calib_lod2": 2, "calib_lod3": 3, "calib_lod4": 4, "calib_lod5": 5,
    "calib_lod6": 6, "lz": 0, "s_min2_l0": 2, "s_min2_lz": 2, "s_max1_l4": 1, "s_max0_l3": 0, "s_bias+2_l1": 3,
    "s_bias+2_lz": 2, "s_bias-1_l3": 2, "s_bias+2_max3_l3": 3, "s_bias+0.5_l1": 2, "l1.25": 1, "l1.5": 2,
    "l1.75": 2, "mipnone_l3": 0, "mipnone_min2_l0": 0, "mipnone_t_min3_l0": 3, "mipnone_t_min2.25": 2,
    "mipnone_t_min2.5": 2, "mipnone_base1_tmin3": 3, "t_min3_l1": 3, "t_min3_lz": 3, "t_min3_l5": 5,
    "t_min2.5_l0": 3, "t_min2.5_l2.6": 3, "t_min2.25_l0": 2, "t_min2.75_l0": 3, "t_min2.25_lz": 2, "l0.5": 1,
    "l2.5": 3, "s_min1.5_l0": 2, "s_max1.5_l4": 2, "l1.49": 1, "base1_l0": 1, "base1_s_min1_l0": 2,
    "base1_t_min2_l0": 2, "base1_t_min2_lz": 2,
}


def field(word, lo, width):
    return (word >> lo) & ((1 << width) - 1)


class FieldsTest(unittest.TestCase):
    def test_sampler_words_match_check_py(self):
        sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
        import check
        for s in ((1, 0.0, 15.0, 0.0), (1, 2.0, 15.0, 0.0), (1, 0.0, 1.5, -1.0), (0, 1.0, 15.0, 0.5)):
            with self.subTest(s=s):
                self.assertEqual(p.sampler_words(s), list(check.ssharp_words(s[0], s[1], s[2], s[3])))

    def test_image_mip_fields(self):
        w = p.image_words("gfx1036", (1, 2.5))
        self.assertEqual(field(w[1], 8, 12), round(2.5 * 256))      # MIN_LOD, 4.8 fixed
        self.assertEqual(field(w[3], 12, 4), 1)                     # BASE_LEVEL
        self.assertEqual(field(w[3], 16, 4), p.LEVELS - 1)          # LAST_LEVEL
        self.assertEqual(field(w[5], 4, 4), p.LEVELS - 1)           # MAX_MIP
        self.assertEqual(field(w[3], 28, 4), 9)                     # SQ_RSRC_IMG_2D

    def test_gfx11_is_refused(self):
        with self.assertRaises(ValueError):
            p.image_words("gfx1150", (0, 0.0))


class RowsTest(unittest.TestCase):
    def test_names_are_unique_and_calibration_comes_first(self):
        rows = p.rows()
        self.assertEqual(len({r[0] for r in rows}), len(rows))
        self.assertEqual([r[0] for r in rows[:p.LEVELS]], [f"calib_lod{n}" for n in range(p.LEVELS)])

    def test_model_gives_the_measured_levels(self):
        by_name = {r[0]: r for r in p.rows()}
        self.assertEqual(set(MEASURED), set(by_name), "every row is pinned to its measured level")
        for name, level in MEASURED.items():
            _n, instr, lod, s, t = by_name[name]
            with self.subTest(name):
                self.assertEqual(p.rdna2_level(instr, lod, s, t), level)

    def test_rows_tell_the_hardware_from_vulkan(self):
        differ = [r[0] for r in p.rows() if p.rdna2_level(*r[1:]) != p.vulkan_level(*r[1:])]
        self.assertIn("l1.5", differ)            # an exact .5 rounds up on the hardware, down on Vulkan
        self.assertIn("t_min2.5_l0", differ)     # the T#'s MIN_LOD clamps the fractional LOD, not the level
        self.assertNotIn("l1.49", differ)

    def test_kernel_builds_for_rdna2(self):
        rows = p.rows()
        lines = p.kernel("gfx1036", rows)
        text = "\n".join(lines)
        self.assertEqual(sum("image_sample_l " in ln for ln in lines), sum(r[1] == "l" for r in rows))
        self.assertEqual(sum("image_sample_lz " in ln for ln in lines), sum(r[1] == "lz" for r in rows))
        self.assertEqual(text.count("global_store_b32"), len(rows))
        self.assertTrue(lines[-1].startswith("  .long"))


if __name__ == "__main__":
    unittest.main()
