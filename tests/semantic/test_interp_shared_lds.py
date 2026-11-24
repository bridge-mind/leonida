"""spirv_mini_interp.run_workgroup shares Workgroup (LDS) memory between its invocations, and OpControlBarrier.

Until this change every invocation of run_workgroup had its own zero-initialised Workgroup variables, so no CPU test
could check lanes communicating through LDS or racing on an LDS dword (each saw only its own writes), and
OpControlBarrier (s_barrier) was refused. Now the Workgroup boxes are the workgroup's, and a barrier is a point every
invocation must reach together, as a subgroup operation is; run_fragment still refuses it. The programs below are
translated compute shaders (one or two wave32 waves, 512 bytes of LDS) writing their results through the
vk_compute_test V#. Standard library only; no GPU, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # base 0, num_records max, raw


def fssm(invocations):
    b = bytearray(0x148)
    struct.pack_into("<4I", b, 0x138, 0x8088, invocations, 1, 1)   # 4 user SGPRs, wg id in s4, 512 B LDS
    return bytes(b)


def run(name, lines, invocations=32):
    spv = support.translate_lines(name, lines + ["s_endpgm"], fssm=fssm(invocations))
    mem = {}
    si.run_workgroup(spv, invocations, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i % 32},
                     push_constants=VSHARP + [0] * 28, memory=mem)
    return [mem.get(4 * i) for i in range(invocations)]


STORE_V3 = ["v_lshlrev_b32_e32 v4, 2, v0", "buffer_store_dword v3, v4, s[0:3], 0 offen"]   # out[lane] = v3


class SharedLds(unittest.TestCase):
    def test_one_lane_writes_every_lane_reads_after_the_barrier(self):
        lines = ["v_mov_b32_e32 v1, 0x40", "v_cmpx_eq_u32_e32 0, v0", "v_mov_b32_e32 v2, 0x1234",
                 "ds_write_b32 v1, v2", "s_mov_b32 exec_lo, -1", "s_waitcnt lgkmcnt(0)", "s_barrier",
                 "ds_read_b32 v3, v1", "s_waitcnt lgkmcnt(0)"] + STORE_V3
        for n in (32, 64):                                        # one wave, and two waves of one workgroup
            with self.subTest(invocations=n):
                self.assertEqual(run("cs_lds_bcast", lines, n), [0x1234] * n)

    def test_lanes_racing_on_one_dword(self):
        # every lane adds 1 to one LDS dword (zero-initialised); after the barrier each reads the total
        lines = ["v_mov_b32_e32 v1, 0x80", "v_mov_b32_e32 v2, 1", "ds_add_u32 v1, v2", "s_waitcnt lgkmcnt(0)",
                 "s_barrier", "ds_read_b32 v3, v1", "s_waitcnt lgkmcnt(0)"] + STORE_V3
        for n in (32, 64):
            with self.subTest(invocations=n):
                self.assertEqual(run("cs_lds_count", lines, n), [n] * n)

    def test_returned_values_are_each_old_value_once(self):
        lines = ["v_mov_b32_e32 v1, 0x80", "v_mov_b32_e32 v2, 1", "ds_add_rtn_u32 v3, v1, v2",
                 "s_waitcnt lgkmcnt(0)"] + STORE_V3
        self.assertEqual(sorted(run("cs_lds_ticket", lines)), list(range(32)))

    def test_a_cas_loop_on_shared_lds(self):
        # ds_add_f32 is an OpAtomicCompareExchange loop: 32 lanes each add 1.0 to one dword
        lines = ["v_mov_b32_e32 v1, 0x80", "v_mov_b32_e32 v2, 1.0", "ds_add_f32 v1, v2", "s_waitcnt lgkmcnt(0)",
                 "s_barrier", "ds_read_b32 v3, v1", "s_waitcnt lgkmcnt(0)"] + STORE_V3
        self.assertEqual(run("cs_lds_fadd", lines), [struct.unpack("<I", struct.pack("<f", 32.0))[0]] * 32)

    def test_workgroups_do_not_share(self):
        # two run_workgroup calls are two workgroups: the second starts from zeroed LDS again
        lines = ["v_mov_b32_e32 v1, 0x80", "v_mov_b32_e32 v2, 1", "ds_add_rtn_u32 v3, v1, v2",
                 "s_waitcnt lgkmcnt(0)"] + STORE_V3
        self.assertEqual(sorted(run("cs_lds_ticket", lines)), sorted(run("cs_lds_ticket", lines)))


BARRIER_MODULE = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %lid
OpExecutionMode %main LocalSize 2 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%_ptr_Input_uint = OpTypePointer Input %uint
%lid = OpVariable %_ptr_Input_uint Input
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%c108 = OpConstant %uint 264
%main = OpFunction %void None %fn
%entry = OpLabel
%id = OpLoad %uint %lid
%first = OpIEqual %bool %id %c0
OpSelectionMerge %end None
OpBranchConditional %first %do %end
%do = OpLabel
OpControlBarrier %c2 %c2 %c108
OpBranch %end
%end = OpLabel
OpReturn
OpFunctionEnd
"""


class Barrier(unittest.TestCase):
    def test_a_barrier_only_some_invocations_reach_is_refused(self):
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_workgroup(BARRIER_MODULE, 2, 32, inputs_for=lambda i: {"%lid": i})

    def test_run_fragment_still_refuses_it(self):
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(BARRIER_MODULE, inputs={"%lid": 0})
        self.assertIn("OpControlBarrier", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
