"""tools/lds_fuzz.py without a GPU: the reference model's instruction semantics and the generator's guarantees
(every access in bounds, no two lanes racing on the same bytes)."""
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz as f


class Swizzle(unittest.TestCase):
    # Offsets as llvm-mc 23 (-mcpu=gfx1013) encodes the swizzle() macros of ds_swizzle_b32.
    def test_quad_perm(self):
        offset = 0x801B                                  # swizzle(QUAD_PERM,3,2,1,0)
        self.assertEqual([f.swizzle_src(lane, offset) for lane in range(8)], [3, 2, 1, 0, 7, 6, 5, 4])

    def test_bitmask_or(self):
        self.assertEqual(f.swizzle_src(3, 0x020F), 19)  # swizzle(BITMASK_PERM,"1pppp"): bit 4 set
        self.assertEqual(f.swizzle_src(19, 0x000F), 3)  # swizzle(BITMASK_PERM,"0pppp"): bit 4 cleared

    def test_reverse_and_broadcast(self):
        self.assertEqual([f.swizzle_src(lane, 0x1C1F) for lane in range(8)], [7, 6, 5, 4, 3, 2, 1, 0])  # REVERSE,8
        self.assertEqual({f.swizzle_src(lane, 0x0078) for lane in range(8, 16)}, {11})               # BROADCAST,8,3

    def test_bitmask_stays_in_its_32_lane_group(self):
        self.assertEqual(f.swizzle_src(32 + 3, 0x1C1F), 32 + (3 ^ 7))     # REVERSE,8 in the upper group
        self.assertEqual(f.swizzle_src(32 + 3, 0x7C1F), 32 + (3 ^ 31))    # REVERSE,32: still lanes 32-63


class Atomics(unittest.TestCase):
    def test_signed_and_unsigned_min_max(self):
        neg = 0xFFFFFFFF
        self.assertEqual(f.atomic("min_i32", 1, neg), neg)
        self.assertEqual(f.atomic("min_u32", 1, neg), 1)
        self.assertEqual(f.atomic("max_i32", 1, neg), 1)
        self.assertEqual(f.atomic("max_u32", 1, neg), neg)

    def test_wrapping_add_sub(self):
        self.assertEqual(f.atomic("add", 0xFFFFFFFF, 2), 1)
        self.assertEqual(f.atomic("sub", 1, 2), 0xFFFFFFFF)


class Generator(unittest.TestCase):
    def programs(self):
        return f.programs(seed=11, count=60, waves=(64, 32))

    def test_deterministic(self):
        a, b = f.Program(3, 5, 64), f.Program(3, 5, 64)
        self.assertEqual(a.asm(), b.asm())
        self.assertEqual(a.model(), b.model())

    def test_accesses_in_bounds_and_race_free(self):
        for p in self.programs():
            for op in p.ops:
                if "addr" not in op:
                    continue
                stride, base, off = op["addr"]
                if op["kind"] in ("read2", "write2", "wrxchg2"):
                    unit = 256 if op["st64"] else 4
                    spans = [[(t * stride + base + o * unit, 4) for o in (op["o0"], op["o1"])]
                             for t in range(f.THREADS)]
                else:
                    size = {"read": lambda: f.READS[op["name"]][0], "write": lambda: f.WRITES[op["name"]][0]}.get(
                        op["kind"], lambda: 4)()
                    spans = [[(t * stride + base + off, size)] for t in range(f.THREADS)]
                for ss in spans:
                    for a, n in ss:
                        self.assertTrue(0 <= a and a + n <= f.LDS_BYTES, (p.name, op))
                if op["kind"] in ("write", "write2", "wrxchg2", "atomic_rtn", "wrxchg", "cmpst"):
                    self.assertTrue(p.disjoint(spans), (p.name, op))

    def test_only_order_independent_atomics_share_addresses(self):
        for p in self.programs():
            for op in p.ops:
                if op["kind"] == "atomic":
                    self.assertIn(op["op"], f.SHARED_ATOMICS)

    def test_no_exec_keeps_the_rest_of_the_program(self):
        a, b = f.Program(1, 32, 32), f.Program(1, 32, 32, no_exec=("bpermute",))
        self.assertEqual(len(a.ops), len(b.ops))
        for x, y in zip(a.ops, b.ops):
            if x["kind"] == "bpermute":
                self.assertIsNone(y["exec"])
            else:
                self.assertEqual(x, y)

    def test_permute_is_generated_and_no_exec_clears_only_permute(self):
        self.assertIn("permute", f.EXEC_KINDS)
        found = 0
        for idx in range(40):
            a = f.Program(1, idx, 64, permute=True)
            b = f.Program(1, idx, 64, no_exec=("permute",), permute=True)
            self.assertEqual(len(a.ops), len(b.ops))
            for x, y in zip(a.ops, b.ops):
                if x["kind"] == "permute":
                    if x["exec"] is not None:
                        found += 1
                    self.assertIsNone(y["exec"])
                    self.assertEqual({k: v for k, v in x.items() if k not in ("exec", "hyp")},
                                     {k: v for k, v in y.items() if k not in ("exec", "hyp")})
                else:
                    self.assertEqual(x, y)
        self.assertGreater(found, 0, "expected at least one permute op with non-None exec mask")


    def test_wrxchg2_is_generated_and_may_alias_the_data(self):
        self.assertIn("wrxchg2", f.EXEC_KINDS)
        found = [op for idx in range(200) for op in f.Program(2, idx, 64).ops if op["kind"] == "wrxchg2"]
        self.assertTrue(found, "expected the generator to emit ds_wrxchg2")
        for op in found:
            self.assertLessEqual(op["dst"] + 1, f.HW_VGPRS - 1)
        self.assertTrue(any(op["dst"] in (op["src0"], op["src1"]) for op in found),
                        "expected some ops to alias vdst with the data pair")


class Wrxchg2(unittest.TestCase):
    """The two-dword exchange: element-scaled offsets, a 64-element st64 step, and data0 back on a repeated offset."""

    def model(self, threads=8, **op):
        p = f.Program(1, 0, 32)
        p.threads = threads
        base = {"kind": "wrxchg2", "addr": (0, 0, 0), "dst": 20, "src0": 22, "src1": 23, "exec": None,
                "st64": False, "o0": 0, "o1": 0}
        base.update(op)
        p.ops = [base]
        return p, *p.model()

    @staticmethod
    def init_lds(p, dword):
        return ((dword * p.lds_mul) & f.M32) ^ p.lds_xor

    @staticmethod
    def init_reg(p, t, r):
        k, c = p.pool_init[r]
        return ((t * k) & f.M32) ^ c

    @staticmethod
    def word(lds, dword):
        return struct.unpack_from("<I", lds, 4 * dword)[0]

    def test_offsets_are_elements_and_the_pair_is_returned(self):
        p, regs, lds = self.model(addr=(24, 0, 0), o0=0, o1=4)        # second element 16 bytes on
        for t in range(8):
            a0, a1 = t * 24, t * 24 + 16
            self.assertEqual(self.word(lds, a0 // 4), self.init_reg(p, t, 22))
            self.assertEqual(self.word(lds, a1 // 4), self.init_reg(p, t, 23))
            self.assertEqual(regs[t][20], self.init_lds(p, a0 // 4))
            self.assertEqual(regs[t][21], self.init_lds(p, a1 // 4))

    def test_a_repeated_offset_returns_data0(self):
        p, regs, lds = self.model(addr=(8, 0, 0), o0=0, o1=0)
        for t in range(8):
            d = t * 8 // 4
            self.assertEqual(self.word(lds, d), self.init_reg(p, t, 23))     # data1 wins the dword
            self.assertEqual(regs[t][20], self.init_lds(p, d))               # the old word
            self.assertEqual(regs[t][21], self.init_reg(p, t, 22))           # then data0

    def test_aliasing_the_data_stores_it_and_returns_the_old_words(self):
        p, regs, lds = self.model(addr=(24, 0, 0), o0=0, o1=4, dst=22, src0=22, src1=23)
        for t in range(8):
            a0, a1 = t * 24, t * 24 + 16
            self.assertEqual(self.word(lds, a0 // 4), self.init_reg(p, t, 22))
            self.assertEqual(self.word(lds, a1 // 4), self.init_reg(p, t, 23))
            self.assertEqual(regs[t][22], self.init_lds(p, a0 // 4))
            self.assertEqual(regs[t][23], self.init_lds(p, a1 // 4))

    def test_st64_steps_64_elements(self):
        p, regs, lds = self.model(addr=(264, 0, 0), o0=0, o1=1, st64=True)
        for t in range(8):
            a1 = t * 264 + 256
            self.assertEqual(self.word(lds, a1 // 4), self.init_reg(p, t, 23))
            self.assertEqual(regs[t][21], self.init_lds(p, a1 // 4))


class Model(unittest.TestCase):
    def test_initial_lds_pattern(self):
        p = f.Program(1, 0, 64)
        p.ops = []
        _, lds = p.model()
        for i in (0, 1, 1023):
            self.assertEqual(struct.unpack_from("<I", lds, 4 * i)[0], ((i * p.lds_mul) & f.M32) ^ p.lds_xor)

    def test_bpermute_inactive_source_reads_zero(self):
        p = f.Program(1, 0, 32)
        p.ops = [{"kind": "bpermute", "mul": 0, "add": 8, "offset": 0, "src": 20, "dst": 21,
                  "exec": 0xFFFF00FF}]
        regs, _ = p.model()
        self.assertEqual(regs[4][21], 0)                       # source lane 8 is off
        keep = f.Program(1, 0, 32)
        keep.ops = []
        self.assertEqual(regs[9][21], keep.model()[0][9][21])  # lane 9 is off itself: keeps its value

    def test_permute_collision_highest_active_wins_and_unaddressed_is_zero(self):
        # All active lanes scatter to lane 5 in their own 32-lane half (mul=0, add=5, offset=0).
        # EXEC: low half has lanes 0..7 active (highest active = 7); high half has lanes 32, 35, 40 active (highest = 40).
        mask = 0xFF | (1 << 32) | (1 << 35) | (1 << 40)
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "permute", "mul": 0, "add": 5, "offset": 0, "src": 20, "dst": 21, "exec": mask}]
        base = f.Program(1, 0, 64)
        base.ops = []
        init_regs, _ = base.model()
        regs, _ = p.model()

        # Low half: lane 5 is active and receives highest active source (lane 7).
        self.assertEqual(regs[5][21], init_regs[7][20])
        # Low half: active lanes != 5 receive nobody -> 0.
        for t in (0, 1, 2, 3, 4, 6, 7):
            self.assertEqual(regs[t][21], 0)
        # Low half: inactive lanes keep their original v21.
        for t in (8, 15, 31):
            self.assertEqual(regs[t][21], init_regs[t][21])
        # High half: target lane 32+5=37 is INACTIVE, so it keeps its original v21 even though 32,35,40 targeted it.
        self.assertEqual(regs[37][21], init_regs[37][21])
        # High half: active lanes 32, 35, 40 receive nobody -> 0.
        for t in (32, 35, 40):
            self.assertEqual(regs[t][21], 0)
        self.assertIn("ds_permute_b32 v21, v6, v20 offset:0", p.asm())

        with f.perturbed("permute_lowest_wins"):
            bad_regs, _ = p.model()
            self.assertEqual(bad_regs[5][21], init_regs[0][20])

    def test_permute_half_isolation_and_offset(self):
        # Identity + offset:8 (+2 lanes) mod 32 within each 32-lane half, all lanes active.
        p = f.Program(2, 1, 64)
        p.ops = [{"kind": "permute", "mul": 1, "add": 0, "offset": 8, "src": 20, "dst": 22, "exec": None}]
        base = f.Program(2, 1, 64)
        base.ops = []
        init_regs, _ = base.model()
        regs, _ = p.model()
        for t in range(64):
            src_lane = (t & ~31) | ((t - 2) & 31)
            self.assertEqual(regs[t][22], init_regs[src_lane][20])

    def test_d16_hi_read_keeps_the_low_half(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_u16_d16_hi", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        self.assertEqual(regs[0][20], (old & 0xFFFF) | (struct.unpack_from("<H", lds, 0)[0] << 16))

    def test_u8_d16_read_zero_extends_into_low_half(self):
        # Tests for the four 8-bit d16 reads written by @0xTehuti in #605.
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_u8_d16", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        self.assertEqual(regs[0][20], (old & 0xFFFF0000) | lds[0])

    def test_u8_d16_hi_read_zero_extends_into_high_half(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_u8_d16_hi", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        self.assertEqual(regs[0][20], (old & 0xFFFF) | (lds[0] << 16))

    def test_i8_d16_read_sign_extends_into_low_half(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_i8_d16", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        byte = lds[0]
        sext = byte - 256 if byte & 0x80 else byte
        self.assertEqual(regs[0][20], (old & 0xFFFF0000) | (sext & 0xFFFF))

    def test_i8_d16_hi_read_sign_extends_into_high_half(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_i8_d16_hi", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        byte = lds[0]
        sext = byte - 256 if byte & 0x80 else byte
        self.assertEqual(regs[0][20], (old & 0xFFFF) | ((sext & 0xFFFF) << 16))

    def test_cmpst_stores_data2_when_data_matches(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "cmpst", "addr": (4, 0, 0), "src": 20, "src2": 21, "dst": 22, "exec": None,
                  "cmp_hit": True}]
        regs, lds = p.model()
        for t in (0, 63):
            self.assertEqual(struct.unpack_from("<I", lds, 4 * t)[0], regs[t][21])
            self.assertEqual(regs[t][22], regs[t][20])            # returns the old word, which was the compare value


class Runner(unittest.TestCase):
    """main() with the translator and the GPU run faked: what counts as a match, a skip and a harness error."""

    def run_main(self, gpu, extra=()):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch

        def translate(progs, work, _emulate):
            (work / "spirv").mkdir(parents=True, exist_ok=True)
            for p in progs:
                (work / "spirv" / (p.name + ".spv")).touch()
            return {p.name: {} for p in progs}

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["lds_fuzz.py", "--programs", "2", "--wave", "64", "--work", tmp, *extra]
            out = io.StringIO()
            with (patch.object(sys, "argv", argv), patch.object(f, "translate", side_effect=translate),
                  patch.object(f, "run_gpu", side_effect=gpu), contextlib.redirect_stdout(out)):
                status = f.main()
        return status, out.getvalue()

    @staticmethod
    def correct_dump(p, dump):
        regs, lds = p.model()
        words = [regs[t][r] for t in range(f.THREADS) for r in f.POOL] + list(struct.unpack(f"<{f.LDS_DWORDS}I", lds))
        dump.write_bytes(struct.pack(f"<{len(words)}I", *words))

    def test_matching_dumps_pass(self):
        def gpu(p, _spv, _build, _entry, _emulate, dump):
            self.correct_dump(p, dump)
            return 0, ""
        status, out = self.run_main(gpu)
        self.assertEqual(status, 0, out)
        self.assertIn("2/2 programs that ran match", out)

    def test_all_skipped_is_not_a_pass(self):
        status, out = self.run_main(lambda *_: (77, "full 64-wide compute subgroups unavailable"))
        self.assertEqual(status, 2, out)
        self.assertIn("2 skipped", out)

    def test_failed_run_with_an_old_dump_is_a_harness_error(self):
        def gpu(p, _spv, _build, _entry, _emulate, dump):
            self.assertFalse(dump.exists(), "the previous run's dump must be gone")
            return 1, "FAILED vkCreateComputePipelines -> -3"
        status, out = self.run_main(gpu)
        self.assertEqual(status, 2, out)
        self.assertIn("harness errors", out)

    def test_short_dump_is_a_harness_error(self):
        def gpu(_p, _spv, _build, _entry, _emulate, dump):
            dump.write_bytes(bytes(256))
            return 0, ""
        status, out = self.run_main(gpu)
        self.assertEqual(status, 2, out)
        self.assertIn("256 of", out)

    UNPINNED = "device: llvmpipe (LLVM 20.1.2, 256 bits) (subgroup 8..8)\ncompute subgroup size: 32 not available"

    def wrong_dump(self, msg):
        """A run whose words are all zero, as a narrower subgroup leaves cross-lane results."""
        def gpu(p, _spv, _build, _entry, _emulate, dump):
            dump.write_bytes(bytes(4 * p.out_words))
            return 0, msg
        return gpu

    def test_unpinned_wave32_is_a_skip_not_a_mismatch(self):
        status, out = self.run_main(self.wrong_dump(self.UNPINNED), ("--wave", "32"))
        self.assertEqual(status, 2, out)                      # nothing ran
        self.assertNotIn("MISMATCH", out)
        self.assertIn("2 skipped", out)
        self.assertIn("can't run it at 32 lanes", out)

    def test_unpinned_wave32_warning_does_not_hide_a_later_harness_failure(self) -> None:
        passing = iter((False, True))

        def gpu(p: f.Program, _spv: Path, _build: str, _entry: dict[str, object],
                _emulate: bool, dump: Path) -> tuple[int, str]:
            if next(passing):
                self.correct_dump(p, dump)
                return 0, "compute subgroup size: 32"
            return 1, self.UNPINNED + "\nFAILED vkQueueSubmit -> -4"

        status, out = self.run_main(gpu, ("--wave", "32"))
        self.assertEqual(status, 2, out)
        self.assertIn("FAILED vkQueueSubmit", out)
        self.assertIn("1/1 programs that ran match", out)
        self.assertIn("1 harness errors", out)
        self.assertIn("0 skipped", out)

    def test_pinned_wave32_is_still_compared(self):
        status, out = self.run_main(self.wrong_dump("compute subgroup size: 32"), ("--wave", "32"))
        self.assertEqual(status, 1, out)
        self.assertIn("MISMATCH", out)

    def test_emulated_wave64_is_compared_at_any_subgroup_size(self):
        # --wave64-emulate output uses no subgroup operations, so llvmpipe's 8 lanes are a real test of it.
        status, out = self.run_main(self.wrong_dump(self.UNPINNED), ("--emulate",))
        self.assertEqual(status, 1, out)
        self.assertIn("MISMATCH", out)
        self.assertIn("0 skipped", out)


class UnpinnedSubgroup(unittest.TestCase):
    """compute_fixture.unpinned_subgroup reads vk_compute_test's VKC_SUBGROUP_SIZE line."""

    def test_not_available_gives_the_size(self):
        from compute_fixture import unpinned_subgroup
        self.assertEqual(unpinned_subgroup("device: X (subgroup 8..8)\ncompute subgroup size: 32 not available\n"), 32)

    def test_pinned_or_missing_line_gives_none(self):
        from compute_fixture import unpinned_subgroup
        self.assertIsNone(unpinned_subgroup("device: X (subgroup 32..64)\ncompute subgroup size: 32\n"))
        self.assertIsNone(unpinned_subgroup("device: X (subgroup 32..32)\n1 groups x 64: 0 mismatches\n"))
        self.assertIsNone(unpinned_subgroup(""))


if __name__ == "__main__":
    unittest.main()
