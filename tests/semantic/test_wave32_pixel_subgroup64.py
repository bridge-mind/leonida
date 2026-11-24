"""Wave32 pixel shaders stay isolated when Vulkan runs them in a 64-lane subgroup.

The inputs are deliberately different in the two physical halves. Each translated PS is run both as one 64-lane
subgroup and as two 32-lane subgroups; outputs must match the independent logical-wave32 expectations.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def run_ps(spv, subgroup_size, inputs, lanes=None, **kwargs):
    lanes = list(range(64)) if lanes is None else lanes
    _, outputs = si.run_workgroup(
        spv, len(lanes), subgroup_size,
        inputs_for=lambda i: {
            "%sg_lane": lanes[i] % subgroup_size,
            "%in_attr0": (inputs[i], 0.0, 0.0, 0.0),
        },
        return_outputs=True, lanes=lanes, **kwargs,
    )
    return [out["%out_mrt0"] for out in outputs]


class Wave32PixelSubgroupTests(unittest.TestCase):
    def test_lane_reads_and_writes_stay_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_lane_ops", [
            "v_interp_mov_f32_e32 v2, p0, attr0.x",
            "v_readfirstlane_b32 s20, v2",
            "v_readlane_b32 s21, v2, 0",
            "v_mov_b32_e32 v6, s20",
            "v_mov_b32_e32 v7, s21",
            "v_mov_b32_e32 v8, 0",
            "v_writelane_b32 v8, s20, 0",
            "v_mov_b32_e32 v3, 0",
            "ds_bpermute_b32 v9, v3, v2",
            "exp mrt0 v6, v7, v8, v9 done vm",
            "s_endpgm",
        ])
        inputs = [float(i + 1) for i in range(64)]
        expected = []
        for i in range(64):
            first = fbits(float((i // 32) * 32 + 1))
            expected.append((first, first, first if i % 32 == 0 else 0, first))
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_ds_permute_scatter_stays_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_ds_permute", [
            "v_interp_mov_f32_e32 v2, p0, attr0.x",
            "v_mbcnt_lo_u32_b32 v5, -1, 0",
            "v_lshlrev_b32_e32 v3, 2, v5",
            "ds_permute_b32 v4, v3, v2",
            "exp mrt0 v4, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        inputs = [float(i + 1) for i in range(64)]
        expected = [(fbits(value),) * 4 for value in inputs]
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_ballots_and_votes_stay_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_ballot", [
            "v_interp_mov_f32_e32 v1, p0, attr0.x",
            "v_cmp_ne_u32_e32 vcc_lo, 0, v1",
            "s_bcnt1_i32_b32 s20, vcc_lo",
            "s_cmp_lg_u32 vcc_lo, 0",
            "s_cselect_b32 s21, 1, 0",
            "s_cmp_eq_u32 vcc_lo, exec_lo",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v6, s20",
            "v_mov_b32_e32 v7, s21",
            "v_mov_b32_e32 v8, s22",
            "exp mrt0 v6, v7, v8, v8 done vm",
            "s_endpgm",
        ])
        inputs = [0.0] * 32 + [1.0] * 32
        expected = [(0, 0, 0, 0)] * 32 + [(32, 1, 1, 1)] * 32
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_descriptor_index_comes_from_a_running_lane_of_each_half(self):
        # SHADER-1016: the image a wave32 pixel shader reads must not depend on which lanes of its half are
        # running. Lanes 0-3 and 32-35 (the first quad of each half) are either not running (a quad with no covered
        # pixel) or helpers that run the shader but that the driver leaves out of subgroup operations; only the
        # other lanes' pixels are judged. Each half reads its own T# (an SGPR offset from its first lane), so a
        # broadcast across the whole 64-lane subgroup is wrong too.
        spv = support.translate_lines("ps_wave32_halves_descriptor", [
            "v_interp_mov_f32_e32 v1, p0, attr0.x",
            "v_cvt_u32_f32_e32 v1, v1",
            "v_readfirstlane_b32 s4, v1",
            "s_lshl_b32 s4, s4, 5",
            "s_load_dwordx8 s[8:15], s[0:1], s4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "image_load v[4:7], v[2:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ])
        # Word 1 (data format) is 0, a float format, so both halves read the float image binding.
        tsharp = {half: [0 if w == 1 else tag + w for w in range(8)]
                  for half, tag in ((0, 0x11110000), (1, 0x22220000))}
        base = 0x1000
        memory = {base + 32 * half + 4 * w: word for half, words in tsharp.items() for w, word in enumerate(words)}
        # The runtime's descriptor hash table: word 0 the slot mask, then 10-word slots (T# words at 1-8, the
        # bindless index at 9). Two slots and mask 1: every T# is found within two probes whatever its hash.
        desc_map = [1, *tsharp[0], 1, 0, *tsharp[1], 2]
        texels = {1: (0.25, 0.5, 0.75, 1.0), 2: (1.0, 0.75, 0.5, 0.25)}
        images = [None] + [(lambda coord, lod, t=texels[k]: t) for k in (1, 2)]
        for first_quad in ("absent", "helpers"):
            lanes = [lane for lane in range(64) if first_quad == "helpers" or lane % 32 >= 4]
            helpers = [i for i, lane in enumerate(lanes) if lane % 32 < 4] if first_quad == "helpers" else []
            inputs = [float(lane // 32) for lane in lanes]
            judged = [i for i in range(len(lanes)) if i not in helpers]
            expected = [tuple(fbits(c) for c in texels[1 + lanes[i] // 32]) for i in judged]
            for subgroup_size in (32, 64):
                with self.subTest(first_quad=first_quad, subgroup_size=subgroup_size):
                    got = run_ps(spv, subgroup_size, inputs, lanes, push_constants=[base, 0], memory=memory,
                                 resources={"%desc_map": desc_map, "%images": images}, helpers=helpers)
                    self.assertEqual([got[i] for i in judged], expected)


if __name__ == "__main__":
    unittest.main()
