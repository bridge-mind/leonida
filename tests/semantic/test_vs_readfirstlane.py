"""v_readfirstlane_b32 handles first-active lanes without crossing wave32 pixel halves.

    python -I tests/semantic/test_vs_readfirstlane.py

A native (not --wave64-emulate) readfirstlane is a ballot of the EXEC lanes, its lowest set bit and a shuffle from
that lane. A wave32 pixel shader selects the ballot word and physical lane for its own 32-lane half, since a
fragment subgroup may contain two logical waves. MoltenVK lowers OpGroupNonUniformBallotFindLSB, like the SubgroupSize built-in, to Metal's
`threads_per_simdgroup` input, which a vertex function can't declare ("invalid 'threads_per_simdgroup' attribute for
input declaration in a vertex function", Apple M4, MoltenVK 1.4.2), so every VS with a readfirstlane failed to
compile there. A vertex shader takes the lowest set bit with FindILsb over the ballot's four words, which Metal
accepts in a vertex function; wave32 pixel shaders use the same word-wise operation to isolate a logical wave.
Other stages keep BallotFindLSB. A ballot has no bits at or above the subgroup size, so the two give the same lane
(docs/design.md, the v_readfirstlane_b32 row).

Hand-written listings run on the strict wave interpreter (spirv_wave_interp.py) with vertex index = lane, with the
first EXEC lane in each ballot word. No llvm-mc, GPU or game data.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
from support import translate_listing  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

LITERAL_OPS = ("exp", "buffer_", "global_", "ds_", "image_")


def size(text):
    """Byte size for the fake encoding the translator places branch offsets with."""
    op = text.split()[0]
    if op.startswith(LITERAL_OPS) or op.endswith("_e64") or "0x" in text:
        return 8
    return 4


def listing(name, lines):
    out = f"; {name} synthetic\n"
    for text in lines:
        out += f"\t{text} ; encoding: [{','.join(['0x00'] * size(text))}]\n"
    return out.encode()


def first_lane_vs(first, wave64):
    """A VS whose lanes below `first` are off in EXEC around `v_readfirstlane_b32 s20, v4` (v4 = 3 * vertex + 1);
    every lane then exports s20 in param0.x."""
    if wave64:
        save, restore, andsave = (["s_mov_b64 s[10:11], exec"], ["s_mov_b64 exec, s[10:11]"],
                                  ["s_and_saveexec_b64 s[12:13], vcc"])
    else:
        save, restore, andsave = (["s_mov_b32 s10, exec_lo"], ["s_mov_b32 exec_lo, s10"],
                                  ["s_and_saveexec_b32 s12, vcc_lo"])
    return (save + ["v_mul_u32_u24_e32 v4, 3, v0", "v_add_nc_u32_e32 v4, 1, v4",
                    f"v_cmp_lt_u32_e32 vcc, {first - 1}, v0"] + andsave +
            ["v_readfirstlane_b32 s20, v4"] + restore +
            ["v_mov_b32_e32 v5, 0", "v_mov_b32_e32 v6, s20", "exp pos0 v5, v5, v5, v5 done",
             "exp param0 v6, v5, v5, v5", "s_endpgm"])


def param0_x(asm, wave):
    res = wi.run_wave(asm, wave=wave, builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})
    xs = []
    for out in res["outputs"]:
        word = next(v for name, v in out.items() if name.endswith("param0"))
        xs.append(word[0].bits if isinstance(word[0], wi.F) else word[0])
    return xs


class VertexReadFirstLane(unittest.TestCase):
    def check(self, wave, first):
        asm = translate_listing(f"vs_rfl_{wave}_{first}", listing("vs", first_lane_vs(first, wave > 32)))
        self.assertEqual(param0_x(asm, wave), [3 * first + 1] * wave)
        return asm

    def test_first_lane_in_word_0(self):
        for first in (1, 6, 31):
            with self.subTest(first=first):
                self.check(32, first)

    def test_first_lane_in_word_1_of_a_64_lane_subgroup(self):
        for first in (32, 40, 63):
            with self.subTest(first=first):
                self.check(64, first)

    def test_all_four_ballot_words_are_read(self):
        # Vulkan allows subgroups of up to 128 invocations (the interpreter runs 32 and 64), so all four words count.
        asm = self.check(32, 6)
        bal = re.search(r"(%\w+) = OpGroupNonUniformBallot ", asm).group(1)
        words = set(re.findall(rf"= OpCompositeExtract %uint {re.escape(bal)} (\d)\b", asm))
        self.assertEqual(words, {"0", "1", "2", "3"})

    def test_no_ballot_find_lsb_or_subgroup_size_in_a_vertex_shader(self):
        asm = self.check(32, 6)
        self.assertNotIn("OpGroupNonUniformBallotFindLSB", asm)
        self.assertIsNone(re.search(r"BuiltIn SubgroupSize\b", asm))
        self.assertIn("OpGroupNonUniformBallot ", asm)
        self.assertIn("FindILsb", asm)

    def test_wave32_pixel_shader_selects_its_subgroup_half(self):
        lines = ["s_mov_b32 s10, exec_lo", "v_cmp_lt_u32_e32 vcc, 5, v2", "s_and_saveexec_b32 s12, vcc_lo",
                 "v_readfirstlane_b32 s20, v2", "s_mov_b32 exec_lo, s10", "v_mov_b32_e32 v6, s20",
                 "exp mrt0 v6, v6, v6, v6 done vm", "s_endpgm"]
        asm = translate_listing("ps_rfl", listing("ps", lines))
        self.assertNotIn("OpGroupNonUniformBallotFindLSB", asm)
        self.assertIn("FindILsb", asm)
        self.assertRegex(asm, r"OpBitwiseAnd %uint %\w+ %c20")
        self.assertIn("OpGroupNonUniformShuffle", asm)


if __name__ == "__main__":
    unittest.main()
