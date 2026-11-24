"""Wave64 whole-wave mask and vote tests, run across every lane of the translated SPIR-V.

Eight GPU tests from runtime/test/shaders (hand-written, no game data; fixtures from regen_fixtures.py), translated
without --wave64-emulate and run by spirv_mini_interp.run_workgroup the way vk_compute_test runs them: one group
of the size in the .asm header (64 threads, 40 for cs_w64x_partial), a heap filled with 0xEE bytes, s[0:3] = a V#
with base 0 and num_records 0xFFFFFFFF.

* One 64-wide subgroup: every word equals the GPU check's value (check.py CS_DUMPS, W64X_DUMPS, W64X_PARTIAL; for
  cs_wavemask the per-lane formula of vk_compute_test's "wavemask" mode). This covers ballots and s_bcnt1 /
  s_ff1 of 64-bit masks, readfirstlane with only upper lanes on, the votes behind wave-uniform branches, and the
  missing lanes of a partial wave (ds_bpermute uses the ballot to avoid shuffling from a missing lane).
* Two 32-wide subgroups (NVIDIA without emulation): the words differ, and WAVE32_EXPECTED pins them. Measured:
  NVIDIA GeForce RTX 4070 SUPER (subgroup 32), driver 610.88, `vk_compute_test <name>.spv 1 dump` on
  run_gpu_tests.py's non-emulated build, three runs each: every word equals WAVE32_EXPECTED, and cs_wavemask
  reports 64 mismatches with lane 0 = 0x00081021, as here. These follow from SPIR-V (each subgroup votes over its
  own 32 lanes) except cs_w64x_load_ballot_load, whose loads read the other subgroup's stores: the order between
  subgroups is not specified, and the interpreter's (subgroup 0's instruction, then subgroup 1's) is the order
  that GPU showed.

Not covered: the cs_w64x tests with branches (branch, sbarrier, wg128_branches, and cs_wm_half_ops /
cs_wm_scc_rfl), because the translator needs llvm-mc's encodings for branch targets and these fixtures are
unassembled; and the --wave64-emulate output (Workgroup memory and a round loop).
"""
import os
import re
import struct
import sys
import unittest

import support
import spirv_mini_interp as si

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
import check  # noqa: E402  (the GPU check's expected words)
from compute_fixture import compute_header  # noqa: E402

ASM_DIR = os.path.join(support.ROOT, "runtime", "test", "shaders")
GPU_DUMPS = dict(check.CS_DUMPS + check.W64X_DUMPS + check.W64X_PARTIAL)
DUMPS = ("cs_w64x_ballot", "cs_w64x_readfirstlane", "cs_w64x_load_ballot_load", "cs_w64x_partial",
         "cs_wave64_ballot", "cs_wm_bfm64", "cs_wm_half_cmp")
HEAP_FILL = 0xEEEEEEEE
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# Two 32-wide subgroups: each half votes, counts and picks its first lane on its own.
WAVE32_EXPECTED = {
    "cs_w64x_ballot": [1061, 1069, 1077, 1085] + [HEAP_FILL] * 4,
    "cs_w64x_readfirstlane": [100, 140, 100, 132, 132, 133, 100, 48],
    "cs_w64x_load_ballot_load": [43, 74, 43, 10, 42, 18, 41, 32],
    "cs_w64x_partial": [32, 8, 528, 0, 0, 532, 539, 500],
    "cs_wave64_ballot": [16, 16, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF],
    "cs_wm_bfm64": [16, 8, 0, 0xFFFFFFFF],
    "cs_wm_half_cmp": [1, 0, 0, 1],
}


class Heap(dict):
    """vk_compute_test's heap: every byte starts as 0xEE."""

    def get(self, key, default=None):
        return super().get(key, HEAP_FILL)


def wavemask_want(lane):
    # vk_compute_test.cpp "wavemask": lanes 16-47 come back on after the parked mask (0x80) and the one through a
    # VGPR (0x40), plus how many set lanes are below this one.
    top = 0xC0 + lane - 16 if 16 <= lane < 48 else 0
    return 65 | 16 << 8 | 8 << 16 | top << 24


def run(name, subgroup_size):
    header = compute_header(os.path.join(ASM_DIR, name + ".asm"))
    threads = struct.unpack_from("<I", header, 0x13C)[0]
    spv = support.translate_fixture(name, fssm=bytes(header))
    memory = si.run_workgroup(
        spv, threads, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
        push_constants=VSHARP + [0] * 28, memory=Heap())
    return memory


def words(memory, n):
    return [dict.get(memory, 4 * k, HEAP_FILL) for k in range(n)]


class OnSubgroup64(unittest.TestCase):
    def test_dumps_equal_the_gpu_check(self):
        for name in DUMPS:
            with self.subTest(shader=name):
                want = GPU_DUMPS[name]
                self.assertEqual(words(run(name, 64), len(want)), want)

    def test_wavemask_every_lane(self):
        memory = run("cs_wavemask", 64)
        self.assertEqual(words(memory, 64), [wavemask_want(t) for t in range(64)])


class OnTwoSubgroups32(unittest.TestCase):
    def test_dumps_match_the_measured_wave32_words(self):
        for name in DUMPS:
            with self.subTest(shader=name):
                want = WAVE32_EXPECTED[name]
                got = words(run(name, 32), len(want))
                self.assertEqual(got, want)
                self.assertNotEqual(got, GPU_DUMPS[name][:len(want)])

    def test_wavemask_fails_on_every_lane(self):
        got = words(run("cs_wavemask", 32), 64)
        self.assertEqual(got[0], 0x00081021)
        self.assertEqual(sum(g != wavemask_want(t) for t, g in enumerate(got)), 64)


class NegativeControl(unittest.TestCase):
    def test_partial_wave_bpermute_without_its_source_lane_fallback(self):
        # Bypass the safe shuffle index. Reading a missing lane is undefined even if its result is masked.
        header = compute_header(os.path.join(ASM_DIR, "cs_w64x_partial.asm"))
        spv = support.translate_fixture("cs_w64x_partial", fssm=bytes(header))
        bad, replaced = re.subn(
            r"(?P<ballot>(?P<present>%\w+) = OpGroupNonUniformBallotBitExtract[^\n]*\n\s*)"
            r"(?P<index>%\w+) = OpSelect %uint (?P=present) (?P<source>%\w+) %\w+(?=\n)",
            r"\g<ballot>\g<index> = OpCopyObject %uint \g<source>",
            spv,
        )
        self.assertEqual(replaced, 2)
        with self.assertRaisesRegex(si.UnsupportedConstruct, "undefined value"):
            si.run_workgroup(bad, 40, 64,
                             inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                             push_constants=VSHARP + [0] * 28, memory=Heap())


class LockstepMemoryOrder(unittest.TestCase):
    def test_a_load_after_a_store_sees_every_lane(self):
        # cs_w64x_load_ballot_load: every lane stores out[lane], then lane 63 loads out[0]. In a SIMD wave all 64
        # stores happen first and the second load reads 10 + 0, so out[3] = 42; a run that let lane 0 finish its
        # block before lane 63 started would read lane 0's later store instead.
        self.assertEqual(words(run("cs_w64x_load_ballot_load", 64), 8)[3], 42)


if __name__ == "__main__":
    unittest.main()
