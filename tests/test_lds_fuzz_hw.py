"""tools/lds_fuzz.py --oracle hw without a GPU: the native kernel text (GFX11 mnemonics, operand order, LDS size,
entry EXEC), the op-mix options, the deliberate model errors and the exit-77 contract. The last test runs on the
hardware when the oracle tools are present and is skipped otherwise. Standard library only."""
import contextlib
import io
import os
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import lds_fuzz as f  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


def lines(p, arch="gfx1150"):
    return p.hw_lines(arch)


class DsSpelling(unittest.TestCase):
    def test_gfx11_renames(self):
        for old, new in {"ds_read_b32": "ds_load_b32", "ds_read_u8_d16_hi": "ds_load_u8_d16_hi",
                         "ds_read2_b32": "ds_load_2addr_b32", "ds_read2st64_b32": "ds_load_2addr_stride64_b32",
                         "ds_write_b128": "ds_store_b128", "ds_write_b8_d16_hi": "ds_store_b8_d16_hi",
                         "ds_write2_b32": "ds_store_2addr_b32", "ds_write2st64_b32": "ds_store_2addr_stride64_b32",
                         "ds_wrxchg_rtn_b32": "ds_storexchg_rtn_b32", "ds_cmpst_rtn_b32": "ds_cmpstore_rtn_b32"}.items():
            self.assertEqual(asm.ds_mnemonic(old, "gfx1150"), new)
            self.assertEqual(asm.ds_mnemonic(old, "gfx1201"), new)
            self.assertEqual(asm.ds_mnemonic(old, "gfx1030"), old)

    def test_unrenamed_ops_keep_their_names(self):
        for op in ("ds_add_u32", "ds_max_rtn_i32", "ds_and_b32", "ds_swizzle_b32", "ds_bpermute_b32",
                   "ds_permute_b32", "v_add_nc_u32_e32", "s_barrier"):
            self.assertEqual(asm.ds_mnemonic(op, "gfx1150"), op)

    def test_cmpstore_takes_the_store_value_first_on_gfx11(self):
        line = "ds_cmpst_rtn_b32 v22, v6, v28, v35 offset:12"          # vdst, vaddr, compare, store
        self.assertEqual(asm.ds_line(line, "gfx1150"), "ds_cmpstore_rtn_b32 v22, v6, v35, v28 offset:12")
        self.assertEqual(asm.ds_line(line, "gfx1201"), "ds_cmpstore_rtn_b32 v22, v6, v35, v28 offset:12")
        self.assertEqual(asm.ds_line(line, "gfx1030"), line)
        nortn = "ds_cmpst_b32 v6, v28, v35"                                  # vaddr, compare, store
        for arch in ("gfx1150", "gfx1201"):                                  # measured on gfx1100 and gfx1201
            self.assertEqual(asm.ds_line(nortn, arch), "ds_cmpstore_b32 v6, v35, v28")
        self.assertEqual(asm.ds_line(nortn, "gfx1030"), nortn)
        with self.assertRaises(ValueError):
            asm.ds_line("ds_cmpst_rtn_b32 v22, v6, 5", "gfx1150")

    def test_gfx12_spells_the_barrier_and_the_lds_wait_its_own_way(self):
        self.assertEqual(asm.sync_lines("s_barrier", "gfx1201"), ["s_barrier_signal -1", "s_barrier_wait -1"])
        self.assertEqual(asm.sync_lines("s_waitcnt lgkmcnt(0)", "gfx1201"), ["s_wait_dscnt 0"])
        for arch in ("gfx1030", "gfx1150"):
            for line in ("s_barrier", "s_waitcnt lgkmcnt(0)"):
                self.assertEqual(asm.sync_lines(line, arch), [line])
        for line in ("s_waitcnt vmcnt(0)", "ds_bpermute_b32 v1, v2, v3", "s_endpgm"):
            self.assertEqual(asm.sync_lines(line, "gfx1201"), [line])

    def test_kernel_carries_its_lds_size_and_registers(self):
        k = asm.Kernel(["s_endpgm"], 64, 64, 8, raw=True, lds=4096, vgprs=40)
        text, names, _ = asm.module_text([k, asm.Kernel(["s_endpgm"], 64, 64, 8, raw=True)], "gfx1150", wave=64)
        self.assertIn(".amdhsa_group_segment_fixed_size 4096", text)
        self.assertIn(".amdhsa_group_segment_fixed_size 0", text)
        self.assertIn(".group_segment_fixed_size: 4096", text)
        self.assertIn(".amdhsa_next_free_vgpr 40", text)
        self.assertIn(".vgpr_count: 40", text)
        for bad in ({"lds": 6}, {"lds": 1 << 20}, {"vgprs": 0}):
            with self.assertRaises(ValueError):
                asm.Kernel([], **bad)


class Native(unittest.TestCase):
    def test_gfx11_kernel_body(self):
        p = f.Program(1, 4, 64)
        body = lines(p)
        self.assertEqual(body[0], "s_mov_b64 s[10:11], exec")
        self.assertTrue(any(ln.startswith("ds_load_b32") for ln in body))
        self.assertTrue(any(ln.startswith("ds_store_b32") for ln in body))
        self.assertFalse(any(ln.startswith(("buffer_", "ds_read", "ds_write")) for ln in body), body)
        stores = [ln for ln in body if ln.startswith("global_store_b32")]
        self.assertEqual(len(stores), len(f.POOL) + f.LDS_DWORDS // 64)
        self.assertTrue(all(ln.endswith("s[2:3]") or "s[2:3] offset:" in ln for ln in stores))
        self.assertEqual(body[-1], "s_endpgm")
        self.assertNotIn("s_mov_b64 exec, -1", body)

    def test_wave32_saves_exec_lo_only(self):
        body = lines(f.Program(1, 4, 32))
        self.assertEqual(body[0], "s_mov_b32 s10, exec_lo")
        self.assertNotIn("exec_hi", " ".join(body))
        restore = [ln for ln in body if ln.startswith("s_mov_b32 exec_lo, s10")]
        masked = [op for op in f.Program(1, 4, 32).ops if op["exec"] is not None]
        self.assertEqual(len(restore), len(masked))

    def test_gfx10_spellings(self):
        body = lines(f.Program(1, 4, 64), "gfx1030")
        self.assertTrue(any(ln.startswith("global_store_dword") for ln in body))
        self.assertTrue(any(ln.startswith("ds_read_b32") for ln in body))
        self.assertFalse(any(ln.startswith("ds_load") for ln in body))

    def test_cmpst_text_per_architecture(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "cmpst", "addr": (4, 0, 0), "src": 20, "src2": 21, "dst": 22, "exec": None, "cmp_hit": False}]
        self.assertIn("ds_cmpstore_rtn_b32 v22, v6, v21, v20 offset:0", lines(p))
        self.assertIn("ds_cmpst_rtn_b32 v22, v6, v20, v21 offset:0", lines(p, "gfx1030"))

    def test_translator_text_is_unchanged(self):
        text = f.Program(1, 4, 64).asm()
        self.assertIn("buffer_store_dword v20, v6, s[0:3], 0 offen offset:0", text)
        self.assertIn("s_mov_b64 exec, -1", text)
        self.assertIn("; group: 64,1,1", text)

    def test_partial_wave_exec_masks_stay_inside_the_group(self):
        p = f.Program(1, 0, 64, threads=32)
        self.assertTrue(p.partial_wave)
        self.assertEqual(p.asm_exec(0xFFFFFFFFFFFFFFFF, True), ["s_mov_b32 exec_lo, 0xffffffff", "s_mov_b32 exec_hi, 0x0"])
        p64 = f.Program(1, 0, 64)
        self.assertEqual(p64.asm_exec(0xFFFFFFFFFFFFFFFF, True), ["s_mov_b32 exec_lo, 0xffffffff",
                                                                  "s_mov_b32 exec_hi, 0xffffffff"])


class OpMix(unittest.TestCase):
    def test_defaults_do_not_change_the_programs(self):
        a, b = f.Program(5, 7, 32), f.Program(5, 7, 32, threads=64, permute=False, swizzle_exec=False)
        self.assertEqual((a.name, a.asm(), a.ops), (b.name, b.asm(), b.ops))
        self.assertEqual(a.name, "cs_ldsfuzz_s5_7_w32")
        self.assertFalse(a.hyp_ops)
        self.assertTrue(all(op["kind"] not in ("permute",) for op in a.ops))

    def test_options_change_the_name_and_the_draws(self):
        names = {f.Program(5, 7, 64, **kw).name for kw in ({}, {"permute": True}, {"swizzle_exec": True},
                                                          {"threads": 128})}
        self.assertEqual(len(names), 4)

    def test_permute_programs_carry_a_hypothesis(self):
        progs = f.programs(3, 40, (64, 32), permute=True)
        perm = [p for p in progs if any(op["kind"] == "permute" for op in p.ops)]
        self.assertTrue(perm)
        parities = set()
        for p in perm:
            for i, op in enumerate(p.ops):
                self.assertEqual(i in p.hyp_ops, op["kind"] == "permute")
                if op["kind"] == "permute":
                    parities.add(op["mul"] % 2)
        self.assertEqual(parities, {0, 1})               # generates both colliding and bijective multipliers

    def test_swizzle_exec_hypothesis_only_under_a_mask(self):
        progs = f.programs(3, 60, (32,), swizzle_exec=True)
        swz = [op for p in progs for op in p.ops if op["kind"] == "swizzle"]
        self.assertTrue(any(op["exec"] is not None for op in swz))
        for op in swz:
            self.assertEqual("hyp" in op, op["exec"] is not None)

    def test_group_sizes_accesses_stay_in_bounds_and_race_free(self):
        for threads in (32, 128, 256, "mix"):
            for p in f.programs(4, 25, (64, 32), threads=threads, permute=True, swizzle_exec=True):
                self.assertIn(p.threads, f.GROUP_SIZES)
                self.assertEqual(p.out_words, p.threads * len(f.POOL) + f.LDS_DWORDS)
                for op in p.ops:
                    if "addr" not in op or op["kind"] in ("read", "read2", "atomic"):
                        continue
                    stride, base, off = op["addr"]
                    unit = (256 if op.get("st64") else 4)
                    offs = (op["o0"] * unit, op["o1"] * unit) if op["kind"] == "write2" else (off,)
                    size = f.WRITES[op["name"]][0] if op["kind"] == "write" else 4
                    spans = [[(t * stride + base + o, size) for o in offs] for t in range(p.threads)]
                    self.assertTrue(p.disjoint(spans), (p.name, op))

    def test_bad_group_sizes(self):
        for bad in (0, 48, 100, 2048, "x"):
            with self.assertRaises(ValueError):
                f.Program(1, 0, 64, threads=bad)

    def test_bpermute_and_permute_stay_in_their_32_lane_group_in_big_groups(self):
        p = f.Program(1, 0, 32, threads=128)
        p.ops = [{"kind": "bpermute", "mul": 1, "add": 9, "offset": 0, "src": 20, "dst": 21, "exec": None},
                 {"kind": "permute", "mul": 3, "add": 5, "offset": 0, "src": 22, "dst": 23, "exec": None}]
        regs, _ = p.model()
        init = f.Program(1, 0, 32, threads=128)
        init.ops = []
        r0 = init.model()[0]
        for t in range(128):
            self.assertEqual(regs[t][21], r0[(t & ~31) | ((t + 9) & 31)][20])
        self.assertEqual(sorted(regs[t][23] for t in range(32, 64)), sorted(r0[t][22] for t in range(32, 64)))

    def test_permute_model(self):
        p = f.Program(1, 0, 32)
        p.ops = [{"kind": "permute", "mul": 1, "add": 1, "offset": 0, "src": 20, "dst": 21,
                  "exec": 0x0000FFFF, "hyp": "x"}]
        regs, _ = p.model()
        init = f.Program(1, 0, 32)
        init.ops = []
        r0 = init.model()[0]
        for t in range(64):                       # lane L sends to lane L + 1 of its wave; lanes 16..31 are off
            lane = t & 31
            if lane >= 16:
                self.assertEqual(regs[t][21], r0[t][21])         # off lanes keep their value
            else:
                self.assertEqual(regs[t][21], 0 if lane == 0 else r0[t - 1][20])   # lane 0 gets nothing

    def test_swizzle_under_exec_reads_zero_from_an_off_lane(self):
        p = f.Program(1, 0, 32)
        p.ops = [{"kind": "swizzle", "offset": 0x8000 | 0b00011011, "src": 20, "dst": 21,     # quad perm 3,2,1,0
                  "exec": 0x7, "hyp": "x"}]
        regs, _ = p.model()
        init = f.Program(1, 0, 32)
        init.ops = []
        r0 = init.model()[0]
        self.assertEqual([regs[t][21] for t in range(3)], [0, r0[2][20], r0[1][20]])      # lane 0 reads lane 3 (off)
        self.assertEqual(regs[3][21], r0[3][21])


class Perturbations(unittest.TestCase):
    """Each deliberate error changes the model's result on a program that uses the op, so the comparison can fail."""

    def test_every_perturbation_is_visible_on_some_program(self):
        progs = f.programs(1, 150, (64, 32), permute=True, swizzle_exec=True)
        for name, (fam, _) in f.PERTURBATIONS.items():
            changed = 0
            for p in progs:
                if not any(f.op_family(op) == fam or f.op_family(op).startswith(fam + "_") or
                           f.op_family(op).endswith("_" + fam) for op in p.ops):
                    continue
                want = p.model()
                with f.perturbed(name):
                    got = p.model()
                changed += want[0] != got[0] or want[1] != got[1]
            self.assertGreater(changed, 0, f"{name} changes nothing on 300 programs")

    def test_perturbed_restores_the_model(self):
        p = f.Program(1, 3, 64)
        before = p.model()
        with f.perturbed("cmpst_swap", "max_i32_unsigned"):
            self.assertIn("cmpst_swap", f.PERTURB)
        self.assertFalse(f.PERTURB)
        self.assertEqual(before, p.model())
        with self.assertRaises(ValueError):
            f.perturbed("nonsense")

    def test_atomic_perturbations(self):
        neg = 0xFFFFFFFF
        with f.perturbed("max_i32_unsigned"):
            self.assertEqual(f.atomic("max_i32", 1, neg), neg)
        self.assertEqual(f.atomic("max_i32", 1, neg), 1)
        with f.perturbed("min_u32_signed"):
            self.assertEqual(f.atomic("min_u32", 1, neg), neg)
        self.assertEqual(f.atomic("min_u32", 1, neg), 1)


class FakeOracle:
    """What runner.Oracle gives run_hw: arch, run(kernels) -> one list of words (or a str) per kernel."""

    arch = "gfx1150"
    device = "fake"

    def __init__(self, programs, mutate=None, reject=()):
        self.by_text = {tuple(p.hw_lines(self.arch)): p for p in programs}
        self.mutate, self.reject, self.kernels = mutate, set(reject), []

    def run(self, kernels):
        out = []
        for k in kernels:
            self.kernels.append(k)
            p = self.by_text[tuple(k.lines)]
            if p.name in self.reject:
                out.append("assembler rejected: instruction not supported on this GPU")
                continue
            regs, lds = p.model()
            words = [regs[t][r] for t in range(p.threads) for r in f.POOL] + list(struct.unpack(f"<{f.LDS_DWORDS}I", lds))
            if self.mutate:
                words = self.mutate(p, words)
            out.append(words)
        return out

    def close(self):
        pass


class RunHw(unittest.TestCase):
    def run_hw(self, oracle, progs, **kw):
        a = type("A", (), dict(seed=1, no_exec=(), opts={}, max_report=3, minimize=False, keep=None, **kw))()
        tally, results = {}, []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = f.run_hw(oracle, progs, a, tally, results)
        return rc, out.getvalue(), tally, results

    def test_kernels_are_launched_as_the_program_is(self):
        progs = f.programs(1, 3, (32, 64), threads="mix")
        o = FakeOracle(progs)
        rc, out, tally, _ = self.run_hw(o, progs)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(o.kernels), len(progs))
        for k in o.kernels:
            p = o.by_text[tuple(k.lines)]
            self.assertEqual((k.wave, k.threads, k.words, k.lds, k.raw), (p.wave, p.threads, p.out_words,
                                                                           f.LDS_BYTES, True))
            self.assertGreaterEqual(k.vgprs, max(f.POOL) + 1)

    def test_a_backend_without_lds_support_is_unavailable_not_a_mismatch(self):
        progs = f.programs(1, 2, (32,))

        class NoLds(FakeOracle):
            def run(self, kernels):
                raise RuntimeError("k0: LDS, scratch memory and COMPUTE_PGM_RSRC3 are not supported by the launcher")
        rc, out, *_ = self.run_hw(NoLds(progs), progs)
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", out)

        class Broken(FakeOracle):
            def run(self, kernels):
                raise RuntimeError("hw_oracle exited 3: something else")
        with self.assertRaises(RuntimeError):
            self.run_hw(Broken(progs), progs)

    def test_gfx12_runs_cmpst_programs(self):
        # The ds_cmpstore operand order is measured on gfx1201 too, so these are no longer skipped there.
        with_cmpst = next(p for p in f.programs(1, 60, (32,)) if any(o["kind"] == "cmpst" for o in p.ops))
        for arch in ("gfx1201", "gfx1150", "gfx1030"):
            self.assertIsNone(f.hw_skip_reason(with_cmpst, arch), arch)

    def test_gfx12_skips_only_wave64_programs_with_a_bpermute(self):
        has_bp = lambda p: any(o["kind"] == "bpermute" for o in p.ops)
        w64 = next(p for p in f.programs(1, 60, (64,)) if has_bp(p))
        w32 = next(p for p in f.programs(1, 60, (32,)) if has_bp(p))
        plain = next(p for p in f.programs(1, 60, (64,)) if not has_bp(p))
        self.assertIn("all 64 lanes", f.hw_skip_reason(w64, "gfx1201"))
        self.assertIsNone(f.hw_skip_reason(w32, "gfx1201"))
        self.assertIsNone(f.hw_skip_reason(plain, "gfx1201"))
        for arch in ("gfx1150", "gfx1030"):                   # halves stay separate there (measured)
            self.assertIsNone(f.hw_skip_reason(w64, arch), arch)
        f.PERTURB.add("bpermute_wave64_cross")                 # the model that crosses halves can run them
        try:
            self.assertIsNone(f.hw_skip_reason(w64, "gfx1201"))
        finally:
            f.PERTURB.discard("bpermute_wave64_cross")

    def test_gfx12_kernels_use_its_barrier_and_lds_wait(self):
        p = f.programs(1, 1, (64,))[0]               # every kernel waits and barriers after each op
        g12, g11 = p.hw_lines("gfx1201"), p.hw_lines("gfx1150")
        self.assertIn("s_barrier", g11)
        self.assertNotIn("s_barrier", g12)
        self.assertNotIn("s_waitcnt lgkmcnt(0)", g12)
        self.assertEqual(g12.count("s_barrier_signal -1"), g11.count("s_barrier"))
        self.assertEqual(g12.count("s_barrier_wait -1"), g11.count("s_barrier"))
        self.assertEqual(g12.count("s_wait_dscnt 0"), g11.count("s_waitcnt lgkmcnt(0)"))
        self.assertEqual(p.hw_lines("gfx1030").count("s_barrier"), g11.count("s_barrier"))

    def test_a_flipped_word_is_a_mismatch(self):
        progs = f.programs(1, 4, (32,))

        def flip(p, words):
            words[5] ^= 1
            return words
        rc, out, tally, _ = self.run_hw(FakeOracle(progs, mutate=flip), progs)
        self.assertEqual(rc, 1, out)
        self.assertEqual(set(tally["hw"].values()), {"mismatch"})
        self.assertIn("MISMATCH", out)

    def test_every_word_is_compared(self):
        p = f.Program(1, 1, 32)
        for index in (0, len(f.POOL) * p.threads - 1, len(f.POOL) * p.threads, p.out_words - 1):
            def flip(_p, words, index=index):
                words[index] ^= 0x80000000
                return words
            rc, out, *_ = self.run_hw(FakeOracle([p], mutate=flip), [p])
            self.assertEqual(rc, 1, (index, out))

    def test_rejected_programs_are_skips_not_passes(self):
        progs = f.programs(1, 3, (32,))
        rc, out, tally, _ = self.run_hw(FakeOracle(progs, reject=[progs[1].name]), progs)
        self.assertEqual(rc, 0, out)
        self.assertEqual(tally["hw"][progs[1].name], "rejected")
        self.assertIn("rejected 1", out)
        rc, out, *_ = self.run_hw(FakeOracle(progs, reject=[p.name for p in progs]), progs)
        self.assertEqual(rc, 2, out)               # nothing ran: not a pass

    def test_a_failed_hypothesis_is_a_mismatch(self):
        progs = [p for p in f.programs(2, 40, (32,), permute=True) if p.hyp_ops][:3]
        self.assertTrue(progs)

        def flip(p, words):
            words[0] ^= 1
            return words
        rc, out, tally, _ = self.run_hw(FakeOracle(progs, mutate=flip), progs)
        self.assertEqual(rc, 1)
        self.assertIn("hyp_only 3", out)

    def test_negative_control_detects_each_perturbation(self):
        progs = f.programs(1, 200, (64, 32), permute=True, swizzle_exec=True)
        rc, out, tally, results = self.run_hw(FakeOracle(progs), progs)
        self.assertEqual(rc, 0, out)
        for name, (cases, bad) in f.negative_control(results).items():
            self.assertGreater(cases, 0, name)
            self.assertGreater(bad, 0, f"{name} is not detected")

    def test_family_table_and_attribution(self):
        progs = f.programs(1, 20, (32,))
        _, _, tally, results = self.run_hw(FakeOracle(progs), progs)
        table = f.family_table(results, tally["hw"])
        self.assertTrue(all(row[1] == 0 for row in table.values()))
        self.assertEqual(sum(r[0] for r in table.values()), sum(len(p.ops) for p in progs))
        self.assertEqual(f.attribute(results), {})


class Main(unittest.TestCase):
    def main(self, *argv):
        out = io.StringIO()
        with patch.object(sys, "argv", ["lds_fuzz.py", *argv]), contextlib.redirect_stdout(out):
            return f.main(), out.getvalue()

    def test_exit_77_when_the_oracle_is_unavailable(self):
        with patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no AMD clang")):
            rc, out = self.main("--oracle", "hw", "--programs", "2")
        self.assertEqual(rc, 77)
        self.assertIn("hardware oracle unavailable: no AMD clang", out)

    def test_exit_77_on_an_rdna4_gpu(self):
        oracle = FakeOracle([])
        oracle.arch = "gfx1201"
        with patch.object(runner, "open_oracle", return_value=oracle):
            rc, out = self.main("--oracle", "hw", "--programs", "2")
        self.assertEqual(rc, 77)
        self.assertIn("hardware oracle unavailable: gfx1201", out)

    def test_exit_77_with_a_missing_compiler(self):
        with patch.dict(os.environ, {"LEONIDA_AMD_CLANG": str(TOOLS / "no-such-clang.exe")}):
            rc, out = self.main("--oracle", "translator,hw", "--programs", "1")
        self.assertEqual(rc, 77, out)

    def test_hw_options_need_the_hardware(self):
        for opt in ("--perturb=cmpst_swap", "--negative-control", "--minimize"):
            with patch.object(sys, "argv", ["lds_fuzz.py", opt, "--programs", "1"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as r:
                        f.main()
            self.assertEqual(r.exception.code, 2, opt)

    def test_oracle_names_are_checked(self):
        with patch.object(sys, "argv", ["lds_fuzz.py", "--oracle", "gpu"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as r:
                    f.main()
        self.assertEqual(r.exception.code, 2)

    def test_model_only_with_the_wider_op_mix(self):
        rc, out = self.main("--model-only", "--extended", "--programs", "5")
        self.assertEqual(rc, 0)
        self.assertIn("10 programs generated", out)

    def test_only_filters_by_index(self):
        oracle = FakeOracle([])
        with patch.object(runner, "open_oracle", return_value=oracle):
            oracle.by_text = {tuple(p.hw_lines("gfx1150")): p for p in f.programs(1, 6, (32,))}
            rc, out = self.main("--oracle", "hw", "--programs", "6", "--wave", "32", "--only", "1,4")
        self.assertEqual(rc, 0, out)
        self.assertIn("programs 2,", out)

    def test_show_prints_the_native_body(self):
        rc, out = self.main("--show", "1:2:32", "--hw-arch", "gfx1150", "--extended")
        self.assertEqual(rc, 0)
        self.assertIn("s_mov_b32 s10, exec_lo", out)


class OnHardware(unittest.TestCase):
    """The oracle itself, when its tools and an AMD GPU are present: the programs of seed 1 match the model, and
    the negative control catches a wrong ds_cmpst."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"no hardware oracle: {e}")
        why = asm.rdna23_skip_reason(cls.oracle.arch)
        if why:
            cls.oracle.close()
            raise unittest.SkipTest(why)
        if asm.family(cls.oracle.arch) == "gfx10":
            raise unittest.SkipTest("the cmpstore operand order is measured on GFX11")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_programs_match_and_a_wrong_model_does_not(self):
        progs = f.programs(1, 40, (32, 64), permute=True, swizzle_exec=True, threads="mix")
        a = type("A", (), dict(seed=1, no_exec=(), opts={}, max_report=3, minimize=False, keep=None))()
        tally, results = {}, []
        with contextlib.redirect_stdout(io.StringIO()):
            rc = f.run_hw(self.oracle, progs, a, tally, results)
        self.assertEqual(rc, 0, tally)
        for name in ("cmpst_swap", "max_i32_unsigned", "bpermute_inactive_reads"):
            cases, bad = f.negative_control(results, only=name)[name]
            self.assertTrue(cases and bad, (name, cases, bad))


if __name__ == "__main__":
    unittest.main()
