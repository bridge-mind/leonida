"""Wave32 lane select of v_readlane_b32 / v_writelane_b32 (SHADER-52), executed across the 32 lanes of the
translated SPIR-V.

The lane is S1[4:0] in wave32 and S1[5:0] in wave64. Until #488 the translator masked with 63 in wave32 too, so
a lane select of 32 or more wrote no lane, and a read shuffled from a lane past the end of a 32-wide subgroup
(undefined in SPIR-V; NVIDIA happens to wrap it). Found by tools/crosslane_fuzz.py (FUZZ-7).

tests/test_translate.py checks which masks appear in the SPIR-V text; this runs the result. The input is the GPU
test runtime/test/shaders/cs_w32_lane_wrap.asm (hand-written, no game data; fixture from regen_fixtures.py),
run by spirv_mini_interp.run_workgroup as one 32-thread workgroup on one 32-wide subgroup. A shuffle from lane
32 or more raises UndefinedResult there, so a translation that masks with 63 fails loudly instead of passing on
a driver's wrap.
"""
import re
import struct
import unittest

import support
import spirv_mini_interp as si

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 32x1x1 threads (the .asm's "; group:").
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
# The .asm header (and check.py): readlane(40) = lane 8 = 25, readlane(5) = 16, writelane(40) lands on lane 8,
# so readlane(8) and readlane(40) both give 777. Lane 0 stores the four words.
EXPECTED = [25, 16, 777, 777]


def translate():
    return support.translate_fixture("cs_w32_lane_wrap", fssm=FSSM)


def run(spv):
    memory = si.run_workgroup(
        spv, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    stray = sorted(a for a in memory if not 0 <= a < 16 or a % 4)
    if stray:
        raise AssertionError(f"stores outside out[0..3]: {stray}")
    return [memory.get(4 * k) for k in range(4)]


class Wave32LaneSelect(unittest.TestCase):
    def test_lane_select_uses_bits_4_0(self):
        self.assertEqual(run(translate()), EXPECTED)

    def test_mask_63_is_undefined(self):
        # Negative control: put the wave64 mask back on the lane selects; the read of lane 40 must then be
        # out of range for the subgroup.
        spv = translate()
        c31 = re.search(r"^(%\S+) = OpConstant %uint 31$", spv, re.M).group(1)
        lines = spv.splitlines()
        sels = [i for i, ln in enumerate(lines)
                if re.search(r"= OpBitwiseAnd %uint \S+ " + re.escape(c31) + "$", ln)]
        # Four reads and one write select a lane, plus the wave lane (LocalInvocationIndex & 31), which a wider
        # mask leaves unchanged in a 32-thread group.
        self.assertGreaterEqual(len(sels), 5)
        for i in sels:
            lines[i] = lines[i][:-len(c31)] + "%neg_c63"
        k = next(i for i, ln in enumerate(lines) if ln.startswith(c31 + " = OpConstant"))
        lines.insert(k + 1, "%neg_c63 = OpConstant %uint 63")
        with self.assertRaises(si.UndefinedResult):
            run("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
