"""Wave64 cross-half lane operations, executed across all 64 lanes of the translated SPIR-V.

The inputs are the GPU tests runtime/test/shaders/cs_w64x_readlane, cs_w64x_writelane and cs_w64x_bpermute
(hand-written, no game data; fixtures from regen_fixtures.py). Each is translated without --wave64-emulate and run
by spirv_mini_interp.run_workgroup as one 64-thread workgroup, in two layouts:

* one 64-wide subgroup (AMD with requiredSubgroupSize = 64). The eight output words must equal the values the
  GPU check expects (runtime/test/shaders/check.py W64X_DUMPS, scripts/run_gpu_tests.py), derived in each .asm
  header from the RDNA2 ISA: v_readlane/v_writelane select lane S1 % 64 and ignore EXEC; ds_bpermute_b32 works
  on each 32-lane half as an independent wave32 (ISA 10.4.4) and reads 0 from a lane that is off in EXEC.
* two 32-wide subgroups, invocations 0-31 and 32-63 (NVIDIA, e.g. an RTX 4070 Super, without emulation). This is
  the documented gap in docs/design.md ("Wave64 handling": cross-lane ops between the two halves stay approximate
  on 32-wide subgroups). The tests pin which results stay exact there and which do not:
    - ds_bpermute_b32 never crosses halves, so all eight words stay exact;
    - v_writelane_b32 writes the right lane in both halves (the wave lane comes from LocalInvocationIndex);
    - a v_readlane_b32 from the other half shuffles from lane >= 32 of a 32-wide subgroup, which SPIR-V leaves
      undefined. Strictly that is UndefinedResult. Under the wrap model (lane Id mod 32, the symptom in issue
      #94, an observation and not the specification) the expected words below are what a 32-wide GPU without
      --wave64-emulate shows. Measured: NVIDIA GeForce RTX 4070 SUPER (subgroup 32), driver 610.88, Vulkan SDK
      1.4.363.0, `vk_compute_test <name>.spv 1 dump` on the non-emulated build of run_gpu_tests.py: all 24
      words equal WRAP32_EXPECTED. That is one driver's behaviour, not a requirement.

Limits: --wave64-emulate output uses Workgroup memory and a round loop, which the interpreter rejects, so the
emulated path is covered only by the GPU tests. The subgroup layout (invocation i in subgroup i // 32) is the
usual one for a 1-D workgroup but Vulkan does not guarantee it.
"""
import re
import struct
import unittest

import support
import spirv_mini_interp as si

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 64x1x1 threads: build.py's default header.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM = bytes(FSSM)
# V# in s[0:3]: base 0, stride 0, num_records 0x1000 bytes. out[k] is the dword at byte 4k.
VSHARP = [0, 0, 0x1000, 0x31016FAC]

# Expected on a 64-wide subgroup: the GPU check's values (check.py W64X_DUMPS).
GPU_EXPECTED = {
    "cs_w64x_readlane": [121, 121, 16, 97, 16, 100, 100, 121],
    "cs_w64x_writelane": [4660, 77, 99, 2748, 0, 0, 4660, 99],
    "cs_w64x_bpermute": [36, 316, 253, 218, 435, 225, 0, 442],
}

# Two 32-wide subgroups, wrap model. readlane: v2 = 3 * lane + 1 and each subgroup reads lane (S1 % 64) % 32 of
# its own half. The low half reads lane 8 for lane 40 (25), lane 0 for 32 (1) and lane 1 for 33 (4); the upper
# half reads lane 32 + 5 = 37 for lane 5 (112). Words stored by lanes of the half the source lane is in stay right:
#   out[0] lane 0: s6 = v2[8] = 25        out[1] lane 33: s6 = v2[40] = 121   out[2] lane 63: s7 = v2[37] = 112
#   out[3] lane 0: s8 = v2[0] = 1         out[4] lane 45: s7 = v2[37] = 112   out[5] lane 7: s9 = v2[1] = 4
#   out[6] lane 60: s9 = v2[33] = 100     out[7] lane 63: v2[40] = 121
# writelane: the writes are exact, the two readlanes from lanes 0 and 8 read lanes 8 and 31 of the low half,
# which nobody wrote (0).
WRAP32_EXPECTED = {
    "cs_w64x_readlane": [25, 121, 112, 1, 112, 4, 100, 121],
    "cs_w64x_writelane": [4660, 77, 99, 2748, 0, 0, 0, 0],
    "cs_w64x_bpermute": GPU_EXPECTED["cs_w64x_bpermute"],
}


def translate(name):
    return support.translate_fixture(name, fssm=FSSM)


def run(spv, subgroup_size, out_of_range="raise"):
    """Run one 64-thread workgroup; return out[0..7] and check nothing was written past them."""
    memory = si.run_workgroup(
        spv, 64, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup_size},
        push_constants=VSHARP + [0] * 28, memory={}, out_of_range=out_of_range)
    stray = sorted(a for a in memory if not 0 <= a < 32 or a % 4)
    if stray:
        raise AssertionError(f"stores outside out[0..7]: {stray}")
    return [memory.get(4 * k) for k in range(8)]


class Wave64OnSubgroup64(unittest.TestCase):
    """Exact on a 64-wide subgroup: the values the GPU check expects."""

    def test_each_shader(self):
        for name, want in GPU_EXPECTED.items():
            with self.subTest(shader=name):
                self.assertEqual(run(translate(name), 64), want)


class Wave64OnTwoSubgroups32(unittest.TestCase):
    """The same modules on two 32-wide subgroups, no --wave64-emulate."""

    def test_bpermute_stays_exact(self):
        spv = translate("cs_w64x_bpermute")
        self.assertEqual(run(spv, 32), GPU_EXPECTED["cs_w64x_bpermute"])

    def test_cross_half_readlane_is_undefined(self):
        for name in ("cs_w64x_readlane", "cs_w64x_writelane"):
            with self.subTest(shader=name):
                with self.assertRaisesRegex(si.UndefinedResult, r"lane 40 of a 32-wide subgroup"):
                    run(translate(name), 32)

    def test_wrap_model_prediction(self):
        for name, want in WRAP32_EXPECTED.items():
            with self.subTest(shader=name):
                got = run(translate(name), 32, out_of_range="wrap")
                self.assertEqual(got, want)
                wrong = [k for k in range(8) if got[k] != GPU_EXPECTED[name][k]]
                self.assertEqual(bool(wrong), name != "cs_w64x_bpermute", f"{name}: words {wrong} differ")

    def test_writelane_writes_are_exact(self):
        # out[0..5] are the written lanes and their untouched neighbours; only the readlanes (out[6..7]) cross.
        got = run(translate("cs_w64x_writelane"), 32, out_of_range="wrap")
        self.assertEqual(got[:6], GPU_EXPECTED["cs_w64x_writelane"][:6])


class NegativeControls(unittest.TestCase):
    """Sabotage the translated SPIR-V the way a translator regression would; the 64-wide result must change."""

    def sabotage(self, name, pattern, repl, count):
        spv = translate(name)
        bad, n = re.subn(pattern, repl, spv)
        self.assertEqual(n, count, f"anchor {pattern!r} matched {n} times in {name}")
        got = run(bad, 64)
        self.assertNotEqual(got, GPU_EXPECTED[name])
        return got

    def test_readlane_lane_select_mod_32(self):
        # v_readlane lane = S1 & 31 instead of & 63 (the issue #94 symptom, moved into the translator).
        got = self.sabotage("cs_w64x_readlane",
                            r"(OpBitwiseAnd %uint \S+ )%c3f(\n\s*%\w+ = OpGroupNonUniformShuffle)", r"\1%c1f\2", 5)
        self.assertEqual(got[0], 25)          # lane 40 -> lane 8

    def test_bpermute_upper_half_reads_low_half(self):
        # The half base from the subgroup lane dropped (the bug fixed on 2026-11-21, see docs/design.md "LDS").
        got = self.sabotage("cs_w64x_bpermute",
                            r"(= OpLoad %uint %sg_lane\n\s*%\w+ = OpBitwiseAnd %uint %\w+ )%c20", r"\1%c0", 3)
        self.assertEqual(got[1], 7 * 13 + 1)   # lane 40 reads lane 13 of the low half instead of lane 45


_SHIFT_MODULE = """
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%c1 = OpConstant %uint 1
%c7 = OpConstant %uint 7
%c40 = OpConstant %uint 40
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%main = OpFunction
%entry = OpLabel
%big = OpShiftLeftLogical %uint %c1 %c40
%dec = OpISub %uint %big %c1
%pick = OpSelect %uint {cond} %c7 %dec
OpStore %out %pick
OpReturn
OpFunctionEnd
"""


class InterpreterContract(unittest.TestCase):
    def test_undefined_shift_only_fails_when_used(self):
        # 1 << 40 on a 32-bit type is undefined; the translator's v_mbcnt computes it for lanes >= 32 and selects
        # it away. The unselected side is fine, storing it is not.
        self.assertEqual(si.run_fragment(_SHIFT_MODULE.format(cond="%true")), {"%out": (7,)})
        with self.assertRaisesRegex(si.UnsupportedConstruct, "undefined value"):
            si.run_fragment(_SHIFT_MODULE.format(cond="%false"))

    def test_single_invocation_rejects_a_shuffle(self):
        with self.assertRaisesRegex(si.UnsupportedConstruct, "run_workgroup"):
            si.run_fragment(translate("cs_w64x_readlane"),
                            inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)}, push_constants=VSHARP)


if __name__ == "__main__":
    unittest.main()
