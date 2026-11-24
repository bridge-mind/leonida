"""SHADER-579: the barrier-round count no longer pushes a full-LDS wave32 compute shader over the device limit.

A native wave32 compute shader whose s_barrier is not in uniform control flow runs in barrier rounds (SHADER-91,
`bar_scheduler`). The round loop counts the invocations not done in `%x_buf`, three Workgroup dwords (#930). On top of
a full 32 KiB LDS that made 32,780 B, 12 bytes over the 32,768 B `maxComputeSharedMemorySize` of the AMD Windows and
Intel drivers, so the runtime check (SHADER-309) refused those pipelines there. When the 12 bytes would take the total
past 16,384 B (the Vulkan minimum), the count now borrows LDS word 0 between rounds instead: invocation 0 saves the
word, the count runs in it, invocation 0 puts it back, each step behind a workgroup barrier while no invocation runs
shader code. Smaller shaders keep `%x_buf` and the single barrier per round.

The Workgroup bytes are counted from the emitted SPIR-V; the barrier behaviour runs in the workgroup interpreter of
test_loop_ballot_barrier.py, whose LDS starts as 0xDEADBEEF. The programs are written here; no game data.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import test_cs_shared_memory_limit as shl  # noqa: E402  (module import: its TestCase stays out of this module)
import test_loop_ballot_barrier as lbb  # noqa: E402

rdna2spirv = support.rdna2spirv
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]      # raw buffer V# at base 0 in s[0:3]


def ring_shift(threads=64, iters=5):
    """s_barrier in a loop (barrier rounds): each iteration thread t writes its value to lds[t], s_barrier, reads
    lds[(t + 1) % threads] and adds 100, s_barrier. Thread 63 reads LDS word 0, the word the count borrows."""
    return [
        "v_add_nc_u32_e32 v1, 1000, v0",
        "v_lshlrev_b32_e32 v2, 2, v0",
        "v_add_nc_u32_e32 v3, 1, v0",
        f"v_and_b32_e32 v3, {threads - 1}, v3",
        "v_lshlrev_b32_e32 v3, 2, v3",
        "s_mov_b32 s8, 0",
        "loop:",
        "ds_write_b32 v2, v1",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        "ds_read_b32 v1, v3",
        "s_waitcnt lgkmcnt(0)",
        "v_add_nc_u32_e32 v1, 100, v1",
        "s_barrier",
        "s_add_u32 s8, s8, 1",
        f"s_cmp_lt_u32 s8, {iters}",
        "s_cbranch_scc1 loop",
        "buffer_store_dword v1, v2, s[0:3], 0 offen",
        "s_endpgm",
    ]


def early_end():
    """Wave 0 runs 2 iterations of a barrier loop and ends, wave 1 runs 4: a wave that is done no longer counts.
    Each thread keeps its count in its own LDS word, so thread 0's lives in the borrowed word."""
    return [
        "v_lshrrev_b32_e32 v1, 5, v0",
        "v_readfirstlane_b32 s8, v1",
        "s_lshl_b32 s9, s8, 1",
        "s_add_u32 s9, s9, 2",
        "s_mov_b32 s12, 0",
        "v_lshlrev_b32_e32 v2, 2, v0",
        "v_mov_b32_e32 v3, 0",
        "loop:",
        "ds_write_b32 v2, v3",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        "ds_read_b32 v4, v2",
        "s_waitcnt lgkmcnt(0)",
        "v_add_nc_u32_e32 v3, 1, v4",
        "s_add_u32 s12, s12, 1",
        "s_cmp_lt_u32 s12, s9",
        "s_cbranch_scc1 loop",
        "buffer_store_dword v3, v2, s[0:3], 0 offen",
        "s_endpgm",
    ]


def translate(items, lds_units, threads=64, limit=None):
    """Native wave32 compute shader of `threads` invocations with LDS_SIZE `lds_units` (512-byte units): 8 user
    SGPRs and TGID_X in RSRC2, nothing else. Returns (SPIR-V text, generator)."""
    lines = [f"{text} ; encoding: [{','.join(['0x00'] * size)}]" for text, size in lbb._resolve_labels(items)]
    header = bytes(0x138) + struct.pack("<4I", 0x90 | (lds_units << 15), threads, 1, 1)
    prev = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.WAVE64_EMULATE = False
    if limit is not None:
        rdna2spirv.MAX_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("cs_bar_counts", lines, fssm=header)
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    gen = rdna2spirv.translate.last_gen
    assert not gen.wave64, "the test program must translate as wave32"
    return spv, gen


def run(spv, threads=64):
    """Output dwords of thread 0..threads-1, for both wave orders."""
    out = []
    for rev in (False, True):
        mem = lbb.run_compute_workgroup(spv, threads, 32, push_constants=VSHARP, reverse_subgroups=rev)
        out.append([mem.get(4 * t) for t in range(threads)])
    return out


RING = [1000 + (t + 5) % 64 + 500 for t in range(64)]


class BarrierCountsInLdsTests(unittest.TestCase):
    def test_full_32k_lds_is_exactly_32768_bytes(self):
        # the finding: 32 KiB LDS + the 12-byte count was 32,780 B; now the count is in the LDS array
        spv, gen = translate(ring_shift(), 64)
        self.assertTrue(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "barrier rounds")
        wg = shl.workgroup_bytes(spv)
        self.assertEqual(wg, {"%lds": 32768})
        self.assertLessEqual(sum(wg.values()), 32768)
        self.assertEqual(rdna2spirv.shader_meta(gen, spv)["lds_bytes"], 32768)
        self.assertNotIn("%x_buf", spv)

    def test_full_32k_lds_translates_under_a_32k_cap(self):
        # --max-shared-memory 32768 (the AMD Windows / Intel limit): accepted, the array not shrunk
        spv, _ = translate(ring_shift(), 64, limit=32768)
        self.assertEqual(shl.workgroup_bytes(spv), {"%lds": 32768})
        self.assertEqual(shl.lds_index_ops(spv), {"OpBitwiseAnd"})

    def test_full_64k_lds_fits_the_hard_cap(self):
        # before: 65,548 B, shrunk or refused; now exactly the 64 KiB cap with the power-of-two array
        spv, _ = translate(ring_shift(), 128)
        self.assertEqual(shl.workgroup_bytes(spv), {"%lds": 65536})
        self.assertEqual(shl.lds_index_ops(spv), {"OpBitwiseAnd"})

    def test_barrier_rounds_still_exact_with_the_borrowed_word(self):
        # the ring shift reads every LDS word another thread wrote, word 0 included, across 10 barriers
        for units in (32, 64, 128):
            with self.subTest(units=units):
                spv, gen = translate(ring_shift(), units)
                self.assertTrue(gen.bar_counts_in_lds())
                self.assertEqual(run(spv), [RING, RING])

    def test_a_wave_that_ends_no_longer_counts(self):
        spv, gen = translate(early_end(), 64)
        self.assertTrue(gen.bar_counts_in_lds())
        self.assertEqual(run(spv), [[2] * 32 + [4] * 32] * 2)

    def test_small_lds_keeps_x_buf_and_one_barrier_per_round(self):
        # 8 KiB + 12 B fits every device: the three Workgroup counts, the barrier rounds as before
        for units in (1, 16):
            with self.subTest(units=units):
                spv, gen = translate(ring_shift(), units)
                self.assertTrue(gen.bar_rounds)
                self.assertFalse(gen.bar_counts_in_lds())
                wg = shl.workgroup_bytes(spv)
                self.assertEqual(wg["%x_buf"], 12)
                self.assertEqual(rdna2spirv.shader_meta(gen, spv)["lds_bytes"], sum(wg.values()))
                self.assertEqual(spv.count("OpControlBarrier"), 2)
                self.assertEqual(run(spv), [RING, RING])

    def test_no_lds_access_keeps_x_buf(self):
        # a 32 KiB allocation the program never touches: no LDS array to borrow from, the 12 bytes stay
        items = [ln for ln in ring_shift() if not ln.startswith(("ds_", "s_waitcnt"))]
        spv, gen = translate(items, 64)
        self.assertTrue(gen.bar_rounds)
        self.assertFalse(gen.uses_lds)
        self.assertEqual(shl.workgroup_bytes(spv), {"%x_buf": 12})

    def test_top_level_barrier_has_no_count(self):
        # a barrier every invocation reaches stays one OpControlBarrier: no rounds, nothing added to the LDS
        items = ["v_lshlrev_b32_e32 v2, 2, v0", "ds_write_b32 v2, v0", "s_waitcnt lgkmcnt(0)", "s_barrier",
                 "ds_read_b32 v1, v2 offset:4", "s_waitcnt lgkmcnt(0)",
                 "buffer_store_dword v1, v2, s[0:3], 0 offen", "s_endpgm"]
        spv, gen = translate(items, 64)
        self.assertFalse(gen.bar_rounds)
        self.assertEqual(shl.workgroup_bytes(spv), {"%lds": 32768})
        self.assertEqual(spv.count("OpControlBarrier"), 1)


if __name__ == "__main__":
    unittest.main()
