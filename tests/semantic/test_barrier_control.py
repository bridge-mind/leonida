"""Semantic interpreter tests for workgroup execution barriers (s_barrier).

Covers:
  * AMD RDNA2 ISA specification for s_barrier (AMD RDNA 2 Instruction Set Architecture,
    Section 6.2 "Program Flow Control" and Section 12 "Scalar Instructions"):
      - Synchronizes all active wavefronts within a workgroup / threadgroup.
      - Blocks wavefront execution until all waves in the workgroup have reached the barrier.
      - Orders LDS memory operations across waves in conjunction with lgkmcnt counters.
  * SPIR-V 1.5 Specification (Section 3.32.20 "OpControlBarrier"):
      - OpControlBarrier <Execution_Scope> <Memory_Scope> <Memory_Semantics>
      - Execution Scope = 2 (Scope::Workgroup)
      - Memory Scope = 2 (Scope::Workgroup)
      - Memory Semantics = 0x108 (WorkgroupMemory [0x100] | AcquireRelease [0x8])
      - Uniformity: OpControlBarrier must be dynamically uniform across the Execution scope.
  * Translator verification (tools/rdna2spirv.py):
      - In compute shaders (cs) and mesh shaders (ms), s_barrier emits
        'OpControlBarrier %c2 %c2 %c108'.
      - In pixel (ps), vertex (vs), and geometry (gs) shaders, s_barrier raises
        Unsupported("s_barrier outside compute").
  * Multi-thread / multi-wave workgroup execution simulation:
      - Workgroup synchronization state machine (arrival tracking, barrier release, epoch advancement).
      - LDS shared memory coherency across barrier boundaries.
      - Multi-phase parallel tree reduction with successive barriers.
      - Workgroup neighbor / ring-shift data exchange across waves.
      - Asynchronous wave arrival orders (staggered scheduling).
      - Deadlock detection for divergent/non-uniform barrier execution.
      - Data race hazard detection when barriers are omitted or bypassed.
  * Mini SPIR-V interpreter strict rejection:
      - Confirms spirv_mini_interp strictly rejects OpControlBarrier as unsupported for
        single-lane straight-line fragment modules.

Run directly:
    python -I tests/semantic/test_barrier_control.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

# SPIR-V Specification constants (Section 3.27 Scope <id> & Section 3.28 Memory Semantics <id>)
SCOPE_CROSS_DEVICE = 0
SCOPE_DEVICE = 1
SCOPE_WORKGROUP = 2
SCOPE_SUBGROUP = 3
SCOPE_INVOCATION = 4

SEMANTICS_RELAXED = 0x0
SEMANTICS_ACQUIRE = 0x2
SEMANTICS_RELEASE = 0x4
SEMANTICS_ACQUIRE_RELEASE = 0x8
SEMANTICS_SEQUENTIALLY_CONSISTENT = 0x10
SEMANTICS_UNIFORM_MEMORY = 0x40
SEMANTICS_WORKGROUP_MEMORY = 0x100
SEMANTICS_IMAGE_MEMORY = 0x800

# The exact memory semantics emitted by rdna2spirv for s_barrier
EXPECTED_BARRIER_SEMANTICS = SEMANTICS_WORKGROUP_MEMORY | SEMANTICS_ACQUIRE_RELEASE  # 0x108 = 264


class BarrierSynchronizationError(Exception):
    """Raised when an invocation attempts to proceed past an unsynchronized barrier."""


class BarrierDeadlockError(Exception):
    """Raised when a control barrier is executed with divergent/non-uniform workgroup flow."""


class DataRaceHazard(Exception):
    """Raised when un-synchronized cross-thread LDS memory access is detected."""


class WorkgroupBarrierInterpreter:
    """Deterministic semantic interpreter modeling multi-wave workgroup execution barriers.

    Simulates a workgroup consisting of multiple waves and threads sharing LDS memory,
    strictly enforcing barrier synchronization, dynamic uniformity, and memory visibility.
    """

    def __init__(self, num_threads=64, wave_size=32, lds_size_dwords=1024):
        if num_threads % wave_size != 0:
            raise ValueError("num_threads must be a multiple of wave_size")
        self.num_threads = num_threads
        self.wave_size = wave_size
        self.num_waves = num_threads // wave_size
        self.lds_size = lds_size_dwords

        # Shared LDS memory
        self.lds = [0] * self.lds_size

        # Barrier state
        self.current_epoch = 0
        self.barrier_arrivals = set()
        self.thread_epochs = [0] * self.num_threads

        # Memory tracking for data race detection within the current epoch
        # address -> set of thread_ids that wrote in current epoch
        self.epoch_writes = {}
        # address -> set of thread_ids that read in current epoch
        self.epoch_reads = {}

    def wave_for_thread(self, thread_id):
        return thread_id // self.wave_size

    def write_lds(self, thread_id, address, value):
        if not (0 <= address < self.lds_size):
            raise IndexError(f"LDS write out of bounds: address {address} (size {self.lds_size})")

        # Check for un-synchronized conflicting access in the same epoch from another thread
        if address in self.epoch_reads:
            conflicting_readers = self.epoch_reads[address] - {thread_id}
            if conflicting_readers:
                raise DataRaceHazard(
                    f"WAR race hazard: thread {thread_id} writing address {address} "
                    f"read by threads {sorted(conflicting_readers)} in epoch {self.current_epoch} without barrier"
                )

        if address not in self.epoch_writes:
            self.epoch_writes[address] = set()
        self.epoch_writes[address].add(thread_id)

        self.lds[address] = value & 0xFFFFFFFF

    def read_lds(self, thread_id, address):
        if not (0 <= address < self.lds_size):
            raise IndexError(f"LDS read out of bounds: address {address} (size {self.lds_size})")

        # Check for un-synchronized RAW hazard (read of an address written by another thread in the same epoch)
        if address in self.epoch_writes:
            conflicting_writers = self.epoch_writes[address] - {thread_id}
            if conflicting_writers:
                raise DataRaceHazard(
                    f"RAW race hazard: thread {thread_id} reading address {address} "
                    f"written by threads {sorted(conflicting_writers)} in epoch {self.current_epoch} without barrier"
                )

        if address not in self.epoch_reads:
            self.epoch_reads[address] = set()
        self.epoch_reads[address].add(thread_id)

        return self.lds[address]

    def arrive_barrier(self, thread_id):
        """Record thread arrival at the workgroup barrier."""
        if not (0 <= thread_id < self.num_threads):
            raise ValueError(f"Invalid thread_id {thread_id}")

        if thread_id in self.barrier_arrivals:
            raise BarrierSynchronizationError(f"Thread {thread_id} arrived twice at the same barrier epoch")

        self.barrier_arrivals.add(thread_id)

        # When all threads have arrived, the barrier is fulfilled
        if len(self.barrier_arrivals) == self.num_threads:
            self._release_barrier()

    def wave_arrive_barrier(self, wave_id):
        """Simulate RDNA2 hardware wave-level barrier: a wave reaches s_barrier."""
        if not (0 <= wave_id < self.num_waves):
            raise ValueError(f"Invalid wave_id {wave_id}")
        start_thread = wave_id * self.wave_size
        for t in range(start_thread, start_thread + self.wave_size):
            self.arrive_barrier(t)

    def _release_barrier(self):
        """Release the barrier for all threads, advance epoch, and clear access hazard sets."""
        for t in range(self.num_threads):
            self.thread_epochs[t] += 1
        self.current_epoch += 1
        self.barrier_arrivals.clear()
        self.epoch_writes.clear()
        self.epoch_reads.clear()

    def step_past_barrier(self, thread_id):
        """Assert that thread_id is permitted to proceed into the next execution phase."""
        if len(self.barrier_arrivals) > 0:
            if thread_id in self.barrier_arrivals:
                raise BarrierSynchronizationError(
                    f"Thread {thread_id} attempted to proceed before all other workgroup threads reached the barrier"
                )
            raise BarrierSynchronizationError(
                f"Thread {thread_id} attempted to proceed before reaching the barrier"
            )

    def check_dynamic_uniformity(self, active_mask):
        """Verify that the barrier is executed uniformly by all workgroup threads."""
        all_active = (1 << self.num_threads) - 1
        if active_mask != all_active:
            divergent_count = bin(active_mask).count("1")
            raise BarrierDeadlockError(
                f"Barrier divergence detected: only {divergent_count} of {self.num_threads} threads reached s_barrier"
            )


class BarrierSpecificationTests(unittest.TestCase):
    """Verify SPIR-V and RDNA2 specification constants and parameter validity for s_barrier."""

    def test_scope_constants(self):
        self.assertEqual(SCOPE_WORKGROUP, 2)
        self.assertEqual(SCOPE_SUBGROUP, 3)
        self.assertEqual(SCOPE_DEVICE, 1)

    def test_memory_semantics_constants(self):
        self.assertEqual(SEMANTICS_WORKGROUP_MEMORY, 0x100)
        self.assertEqual(SEMANTICS_ACQUIRE_RELEASE, 0x8)
        self.assertEqual(EXPECTED_BARRIER_SEMANTICS, 0x108)

    def test_semantics_contains_required_flags(self):
        # Workgroup barrier must order workgroup memory (LDS) with acquire-release semantics
        self.assertTrue(EXPECTED_BARRIER_SEMANTICS & SEMANTICS_WORKGROUP_MEMORY)
        self.assertTrue(EXPECTED_BARRIER_SEMANTICS & SEMANTICS_ACQUIRE_RELEASE)
        # Should not set device-level or uniform memory scopes unless specified
        self.assertFalse(EXPECTED_BARRIER_SEMANTICS & SEMANTICS_UNIFORM_MEMORY)
        self.assertFalse(EXPECTED_BARRIER_SEMANTICS & SEMANTICS_IMAGE_MEMORY)


class TranslatorBarrierEmissionTests(unittest.TestCase):
    """Verify tools/rdna2spirv.py handling of s_barrier across pipeline stages."""

    def test_compute_stage_emits_opcontrolbarrier(self):
        g = rdna2spirv.Gen(stage="cs")
        g.insn({"op": "s_barrier", "args": [], "mods": []})
        emitted = "\n".join(g.body)
        self.assertIn("OpControlBarrier", emitted)
        # Scope 2 (Workgroup), Scope 2 (Workgroup), Semantics 0x108 (264)
        c2 = g.cu(2)
        c108 = g.cu(0x108)
        expected_line = f"OpControlBarrier {c2} {c2} {c108}"
        self.assertIn(expected_line, emitted)

    def test_mesh_stage_emits_opcontrolbarrier(self):
        g = rdna2spirv.Gen(stage="ms")
        g.insn({"op": "s_barrier", "args": [], "mods": []})
        emitted = "\n".join(g.body)
        self.assertIn("OpControlBarrier", emitted)
        c2 = g.cu(2)
        c108 = g.cu(0x108)
        expected_line = f"OpControlBarrier {c2} {c2} {c108}"
        self.assertIn(expected_line, emitted)

    def test_pixel_stage_rejects_s_barrier(self):
        g = rdna2spirv.Gen(stage="ps")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.insn({"op": "s_barrier", "args": [], "mods": []})
        self.assertIn("s_barrier outside compute", str(ctx.exception))

    def test_vertex_stage_rejects_s_barrier(self):
        g = rdna2spirv.Gen(stage="vs")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.insn({"op": "s_barrier", "args": [], "mods": []})
        self.assertIn("s_barrier outside compute", str(ctx.exception))

    def test_geometry_stage_rejects_s_barrier(self):
        g = rdna2spirv.Gen(stage="gs")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.insn({"op": "s_barrier", "args": [], "mods": []})
        self.assertIn("s_barrier outside compute", str(ctx.exception))

    def test_s_barrier_leaves_registers_unmodified(self):
        g = rdna2spirv.Gen(stage="cs")
        initial_cur_masks = copy.copy(g.cur_masks)
        g.insn({"op": "s_barrier", "args": [], "mods": []})
        self.assertEqual(g.cur_masks, initial_cur_masks)


class WorkgroupBarrierSemanticTests(unittest.TestCase):
    """Semantic verification of workgroup barrier synchronization and memory exchange."""

    def test_neighbor_exchange_with_barrier(self):
        """Each thread writes to LDS, workgroup synchronizes via s_barrier, then reads neighbor value."""
        num_threads = 64
        interp = WorkgroupBarrierInterpreter(num_threads=num_threads, wave_size=32)

        # Phase 1: All threads write unique pattern to their LDS slot
        for t in range(num_threads):
            val = (t * 31 + 7) & 0xFFFFFFFF
            interp.write_lds(t, address=t, value=val)

        # Synchronize workgroup via waves
        interp.wave_arrive_barrier(0)
        interp.wave_arrive_barrier(1)

        # Phase 2: Each thread reads from neighbor (t + 1) % num_threads
        results = []
        for t in range(num_threads):
            interp.step_past_barrier(t)
            neighbor_addr = (t + 1) % num_threads
            results.append(interp.read_lds(t, address=neighbor_addr))

        # Check every result matches the neighbor's initial write
        for t in range(num_threads):
            expected = ((t + 1) % num_threads * 31 + 7) & 0xFFFFFFFF
            self.assertEqual(results[t], expected)

    def test_missing_barrier_triggers_data_race_hazard(self):
        """Attempting to read neighbor data in the same epoch without a barrier raises DataRaceHazard."""
        num_threads = 64
        interp = WorkgroupBarrierInterpreter(num_threads=num_threads, wave_size=32)

        for t in range(num_threads):
            interp.write_lds(t, address=t, value=t * 10)

        # Omit the barrier: thread 0 tries to read LDS[1] which was written by thread 1 in the same epoch
        with self.assertRaises(DataRaceHazard) as ctx:
            interp.read_lds(0, address=1)
        self.assertIn("RAW race hazard", str(ctx.exception))

    def test_parallel_tree_reduction_across_successive_barriers(self):
        """Tree reduction of 64 elements using successive workgroup barriers across 6 steps."""
        num_threads = 64
        interp = WorkgroupBarrierInterpreter(num_threads=num_threads, wave_size=32)

        # Step 0: Initialize LDS[t] = t + 1
        for t in range(num_threads):
            interp.write_lds(t, address=t, value=t + 1)

        interp.wave_arrive_barrier(0)
        interp.wave_arrive_barrier(1)

        # Reduction loop: strides 1, 2, 4, 8, 16, 32
        stride = 1
        while stride < num_threads:
            for t in range(0, num_threads, stride * 2):
                interp.step_past_barrier(t)
                a = interp.read_lds(t, address=t)
                b = interp.read_lds(t, address=t + stride)
                interp.write_lds(t, address=t, value=a + b)

            # Synchronize all threads at the end of this stride
            for t in range(num_threads):
                interp.arrive_barrier(t)

            stride *= 2

        # Final sum in LDS[0] must equal 1 + 2 + ... + 64 = 64 * 65 / 2 = 2080
        final_sum = interp.lds[0]
        self.assertEqual(final_sum, 2080)
        # 6 reduction steps + 1 initial step = 7 epochs
        self.assertEqual(interp.current_epoch, 7)

    def test_asynchronous_wave_arrival_order(self):
        """Waves may arrive in non-sequential order; barrier holds until all waves arrive."""
        interp = WorkgroupBarrierInterpreter(num_threads=128, wave_size=32)  # 4 waves
        self.assertEqual(interp.num_waves, 4)

        for t in range(128):
            interp.write_lds(t, address=t, value=t)

        # Waves arrive in reverse order: 3, 2, 1
        interp.wave_arrive_barrier(3)
        interp.wave_arrive_barrier(2)
        interp.wave_arrive_barrier(1)

        # Barrier must not be released yet (wave 0 has not arrived)
        self.assertEqual(interp.current_epoch, 0)
        with self.assertRaises(BarrierSynchronizationError):
            interp.step_past_barrier(thread_id=96)  # Thread from wave 3

        # Wave 0 arrives last
        interp.wave_arrive_barrier(0)
        self.assertEqual(interp.current_epoch, 1)

        # Now any thread can proceed safely
        for t in range(128):
            interp.step_past_barrier(t)

    def test_divergent_barrier_detected_as_deadlock(self):
        """Conditional branch causing only half the workgroup to execute s_barrier triggers deadlock."""
        num_threads = 64
        interp = WorkgroupBarrierInterpreter(num_threads=num_threads, wave_size=32)

        # Active mask where only thread_id < 32 executes the barrier
        divergent_mask = (1 << 32) - 1
        with self.assertRaises(BarrierDeadlockError) as ctx:
            interp.check_dynamic_uniformity(divergent_mask)
        self.assertIn("Barrier divergence detected", str(ctx.exception))

    def test_duplicate_arrival_error(self):
        """A thread cannot arrive twice at the same barrier epoch."""
        interp = WorkgroupBarrierInterpreter(num_threads=64, wave_size=32)
        interp.arrive_barrier(0)
        with self.assertRaises(BarrierSynchronizationError):
            interp.arrive_barrier(0)

    def test_out_of_bounds_lds_rejected(self):
        """Access beyond declared LDS dword capacity is rejected."""
        interp = WorkgroupBarrierInterpreter(num_threads=64, wave_size=32, lds_size_dwords=128)
        with self.assertRaises(IndexError):
            interp.write_lds(0, address=128, value=1)
        with self.assertRaises(IndexError):
            interp.read_lds(0, address=128)


class MiniInterpreterRejectionTests(unittest.TestCase):
    """Verify that spirv_mini_interp strictly rejects OpControlBarrier as unsupported."""

    HEADER = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c2 = OpConstant %uint 2
%c108 = OpConstant %uint 264
%main = OpFunction %void None %fn
%entry = OpLabel
"""
    FOOTER = "OpReturn\nOpFunctionEnd\n"

    def test_mini_interp_refuses_opcontrolbarrier(self):
        text = self.HEADER + "OpControlBarrier %c2 %c2 %c108\n" + self.FOOTER
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(text)
        self.assertIn("OpControlBarrier", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
