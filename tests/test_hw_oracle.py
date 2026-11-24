"""tools/hw_oracle: kernel text, literal and opcode filters, error mapping, the exit-77 contract, and the parts of
the Linux backend that need no GPU (reading the assembled object, the launcher's tokens, the display check). The
last tests run on the hardware when the oracle tools are present and are skipped otherwise.
Standard library only."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from hw_oracle import asm, drm, runner  # noqa: E402

PROG = ["s_mov_b32 s4, 0x12345678", "s_mov_b32 s5, 7", "s_add_u32 s6, s4, s5", "s_cselect_b32 s30, 1, 0",
        "v_mov_b32 v8, s4", "v_mov_b32 v9, s5", "v_mov_b32 v10, s6", "v_mov_b32 v11, s30"]


class Text(unittest.TestCase):
    def test_families(self):
        self.assertEqual(asm.family("gfx1150"), "gfx11")
        self.assertEqual(asm.family("gfx1030"), "gfx10")
        self.assertEqual(asm.family("gfx1013"), "gfx10")
        self.assertEqual(asm.family("gfx1201"), "gfx12")
        with self.assertRaises(ValueError):
            asm.family("gfx90a")

    def test_kernel_uses_the_mnemonics_of_its_family(self):
        k11, k10 = asm.kernel_text("k0", PROG, "gfx1150"), asm.kernel_text("k0", PROG, "gfx1030")
        k12 = asm.kernel_text("k0", PROG, "gfx1201")
        for k in (k11, k12):
            self.assertIn("s_load_b64 s[2:3], s[0:1], 0x0", k)
            self.assertIn("global_store_b128 v0, v[8:11], s[2:3]", k)
        self.assertIn("s_load_dwordx2", k10)
        self.assertIn("global_store_dwordx4", k10)
        for k in (k11, k12, k10):
            self.assertIn("  s_add_u32 s6, s4, s5\n", k)
            self.assertIn(".amdhsa_wavefront_size32 1", k)
            # the program runs between the output-pointer load and the store
            self.assertLess(k.index("s_waitcnt lgkmcnt(0)"), k.index("s_add_u32"))
            self.assertLess(k.index("s_add_u32"), k.index("v[8:11]"))

    def test_raw_kernels_get_the_memory_mnemonics_of_their_family(self):
        body = ["v_mov_b32 v20, 7", "global_store_b32 v22, v20, s[2:3] offset:4", "s_endpgm"]
        k10, k11 = asm.raw_kernel_text("k0", body, "gfx1036"), asm.raw_kernel_text("k0", body, "gfx1150")
        self.assertIn("  global_store_dword v22, v20, s[2:3] offset:4\n", k10)     # GFX10 has no global_store_b32
        self.assertNotIn("global_store_b32", k10)
        self.assertIn("  global_store_b32 v22, v20, s[2:3] offset:4\n", k11)
        self.assertEqual(asm.gfx10_spelling("global_load_b128 v[4:7], v0, s[2:3]"), "global_load_dwordx4 v[4:7], v0, s[2:3]")
        self.assertEqual(asm.gfx10_spelling("v_mov_b32 v20, 7"), "v_mov_b32 v20, 7")

    def test_gfx1170_and_later_reject_the_ieee_and_dx10_clamp_mode_fields(self):
        body = ["s_endpgm"]
        for arch in ("gfx1170", "gfx1200", "gfx1201"):
            for field in ("ieee", "dx10_clamp"):
                with self.assertRaisesRegex(asm.ModeUnavailable, field, msg=f"{arch} {field}"):
                    asm.raw_kernel_text("k0", body, arch, mode={field: 0})

    def test_denormal_and_rounding_mode_fields_still_work_on_gfx12(self):
        text = asm.raw_kernel_text("k0", ["s_endpgm"], "gfx1201",
                                   mode={"denorm32": 0, "denorm16_64": 3, "round16_64": 3})
        self.assertIn(".amdhsa_float_denorm_mode_32 0", text)
        self.assertIn(".amdhsa_float_denorm_mode_16_64 3", text)
        self.assertIn(".amdhsa_float_round_mode_16_64 3", text)
        self.assertNotIn(".amdhsa_ieee_mode", text)
        self.assertNotIn(".amdhsa_dx10_clamp", text)

    def test_gfx1150_still_emits_the_ieee_and_dx10_clamp_mode_fields(self):
        text = asm.raw_kernel_text("k0", ["s_endpgm"], "gfx1150", mode={"ieee": 0, "dx10_clamp": 1})
        self.assertIn(".amdhsa_ieee_mode 0", text)
        self.assertIn(".amdhsa_dx10_clamp 1", text)

    def test_the_runner_reports_an_unavailable_mode_as_a_skip(self):
        o = object.__new__(runner.Oracle)      # no tools, no GPU
        o.arch, o.chunk = "gfx1201", 500
        # The kernel is not assembled; it comes back as the reason (#1620), and the rest of the run goes on.
        result, = o.run([asm.Kernel(["s_endpgm"], raw=True, mode={"ieee": 0})])
        self.assertIsInstance(result, str)
        self.assertIn("ieee", result)

    def test_metadata_is_required_and_lists_every_kernel(self):
        text, names, starts = asm.module_text([PROG, PROG, PROG], "gfx1150")
        self.assertEqual(names, ["k0", "k1", "k2"])
        self.assertIn('.amdgcn_target "amdgcn-amd-amdhsa--gfx1150"', text)
        self.assertEqual(text.count(".symbol: k"), 3)
        self.assertIn(".end_amdgpu_metadata", text)
        self.assertEqual(text.count("global_store_b128"), 3)

    def test_error_lines_map_back_to_kernels(self):
        text, names, starts = asm.module_text([PROG, PROG, PROG], "gfx1150")
        lines = text.split("\n")
        for k, start in enumerate(starts):
            self.assertTrue(lines[start - 1].startswith(".globl k"), (k, lines[start - 1]))
        err = f"k.s:{starts[1] + 7}:3: error: invalid operand\n  s_foo\n"
        self.assertEqual(asm.failing_kernels(err, starts), {1})
        err2 = err + f"k.s:{starts[2]}:1: error: x\n"
        self.assertEqual(asm.failing_kernels(err2, starts), {1, 2})


class DetectArch(unittest.TestCase):
    def test_amdgpu_arch_target_features_are_dropped(self):
        for printed, want in (("gfx1010:xnack-\n", "gfx1010"), ("gfx1150\n", "gfx1150"),
                              ("gfx1030:sramecc+:xnack-\ngfx1150\n", "gfx1030")):
            with self.subTest(printed=printed):
                done = subprocess.CompletedProcess([], 0, stdout=printed)
                with mock.patch.dict(os.environ, {}, clear=False), \
                        mock.patch.object(os.path, "isfile", return_value=True), \
                        mock.patch.object(subprocess, "run", return_value=done):
                    os.environ.pop("LEONIDA_HW_ARCH", None)
                    arch = runner.detect_arch(os.path.join("opt", "clang"))
                self.assertEqual(arch, want)
                asm.family(arch)   # what Oracle() does next must accept it


class True16(unittest.TestCase):
    def test_the_low_half_is_named_only_on_the_f16_operand(self):
        for old, new in (
                ("v_cvt_f16_f32_e32 v8, v4", "v_cvt_f16_f32_e32 v8.l, v4"),
                ("  v_cvt_f16_f32_e64 v8, -|v4|", "  v_cvt_f16_f32_e64 v8.l, -|v4|"),
                ("v_cvt_f16_f32 v8, 0x3f800000", "v_cvt_f16_f32 v8.l, 0x3f800000"),
                ("v_cvt_f32_f16_e32 v8, v4", "v_cvt_f32_f16_e32 v8, v4.l"),
                ("v_cvt_f32_f16_e64 v8, -|v14| clamp mul:2", "v_cvt_f32_f16_e64 v8, -|v14.l| clamp mul:2"),
                ("v_cvt_f32_f16 v8, abs(v4)", "v_cvt_f32_f16 v8, abs(v4.l)"),
                ("v_exp_f16 v12, v2", "v_exp_f16 v12.l, v2.l"),              # both operands are f16
                ("v_sqrt_f16_e64 v12, -|v2|", "v_sqrt_f16_e64 v12.l, -|v2.l|"),
                ("v_rcp_f16 v12, 0x3c00", "v_rcp_f16 v12.l, 0x3c00")):
            self.assertEqual(asm.true16_spelling(old), new)
            self.assertEqual(asm.true16_spelling(new), new)        # already spelled that way
        for same in ("v_cvt_pkrtz_f16_f32 v8, v4, v5", "v_cvt_f32_f16 v8, s4", "v_cvt_f32_f16 v8, 0x3c00",
                     "v_cvt_f32_f16 v8, v4.h", "v_mov_b32 v8, v4"):
            self.assertEqual(asm.true16_spelling(same), same)

    def test_op_sel_of_the_16_bit_integer_vop3_ops_is_named_on_their_vgprs(self):
        """Bit i of op_sel picks source i's half, the last bit the destination's. GFX11+ assemblers reject the bare
        form of these eight ops (gfx1100, gfx1150 and gfx1201 checked with AMD LLVM 23); SGPRs and constants keep
        taking their half from the op_sel list, so it stays as written."""
        for old, new in (
                ("v_max_u16_e64 v14, s12, v5 op_sel:[1,0,0]", "v_max_u16_e64 v14.l, s12, v5.l op_sel:[1,0,0]"),
                ("v_mul_lo_u16_e64 v12, s16, v14 op_sel:[0,1,0]",
                 "v_mul_lo_u16_e64 v12.l, s16, v14.h op_sel:[0,1,0]"),
                ("v_max_u16_e64 v18, v18, s9 op_sel:[1,1,0]", "v_max_u16_e64 v18.l, v18.h, s9 op_sel:[1,1,0]"),
                ("v_mul_lo_u16_e64 v16, s19, 0x100 op_sel:[1,1,1]",
                 "v_mul_lo_u16_e64 v16.h, s19, 0x100 op_sel:[1,1,1]"),
                ("  v_lshlrev_b16_e64 v2, v19, 32 op_sel:[0,1,1]",
                 "  v_lshlrev_b16_e64 v2.h, v19.l, 32 op_sel:[0,1,1]"),
                ("v_ashrrev_i16 v0, s13, v18 op_sel:[0,1,0]", "v_ashrrev_i16 v0.l, s13, v18.h op_sel:[0,1,0]")):
            self.assertEqual(asm.opsel16_spelling(old), new)
            self.assertEqual(asm.opsel16_spelling(new), new)        # already spelled that way
        for same in ("v_max_u16_e64 v5, v1, v3",                   # no op_sel: the bare form still assembles
                     "v_add_nc_u16 v1, v2, v3 op_sel:[1,0,0]",     # these ops take the bare form on GFX11+
                     "v_mad_u16 v1, v2, v3, v4 op_sel:[1,0,0,0]",
                     "v_max_u16_e64 v1, v2 op_sel:[1,0,0]"):       # not a destination and two sources
            self.assertEqual(asm.opsel16_spelling(same), same)

    def test_the_non_returning_compare_and_store_is_respelled_and_swapped_on_gfx11(self):
        """GFX11 has no ds_cmpst_b32, and ds_cmpstore_b32 takes the store value before the compare value."""
        self.assertEqual(asm.ds_line("ds_cmpst_b32 v0, v1, v2", "gfx1150"),
                         "ds_cmpstore_b32 v0, v2, v1")
        self.assertEqual(asm.ds_line("ds_cmpst_b32 v0, v1, v2 offset:8", "gfx1100"),
                         "ds_cmpstore_b32 v0, v2, v1 offset:8")
        self.assertEqual(asm.ds_line("ds_cmpst_b32 v0, v1, v2", "gfx1030"),
                         "ds_cmpst_b32 v0, v1, v2")
        # Both changes are needed together: the rename alone would compare v2 and store v1.
        with self.assertRaises(ValueError):
            asm.ds_line("ds_cmpst_b32 v0, v1", "gfx1150")

    def test_error_lines_still_map_back(self):
        text = "v_mov_b32 v1, 0\nv_cvt_f32_f16 v8, 0x3c00\nv_cvt_f16_f32 v8, v1\nv_mov_b32 v9, 0\n"
        out = asm.true16_spelling(text)
        self.assertEqual(out, text.replace("v_cvt_f16_f32 v8,", "v_cvt_f16_f32 v8.l,"))

    def test_true16_binary_keeps_explicit_halves_and_other_instruction_forms(self):
        for old, new in (
                ("v_mul_f16_e32 v0, v1, v2", "v_mul_f16_e32 v0.l, v1.l, v2.l"),
                (" v_max_f16_e64 v8.h, -|v4|, s1 // v9", " v_max_f16_e64 v8.h, -|v4.l|, s1 // v9"),
                ("v_lshlrev_b16 v0, 3, v2", "v_lshlrev_b16 v0.l, 3, v2.l")):
            self.assertEqual(asm.true16_spelling(old), new)
            self.assertEqual(asm.true16_spelling(new), new)
        for text in ("v_mul_lo_u16_e64 v1, v2, v3 op_sel:[0,1,0]",
                     "v_fma_f16_e64 v0, v1, v2, v3 op_sel:[1,0,1,1]",
                     "v_pk_add_f16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[1,1]",
                     ".long 0xd7050014 // v_mul_lo_u16 v20, v21, v22"):
            self.assertEqual(asm.true16_spelling(text), text)


class ToolchainFallbacks(unittest.TestCase):
    """The two probes, with a stand-in for the assembler and the launcher."""

    def oracle(self, arch="gfx1103"):
        o = object.__new__(runner.Oracle)      # no tools, no GPU
        o.arch, o.chunk = arch, 500
        return o

    def probe(self, loads, error="hipModuleLoadData(&mod, img) -> hip error 209"):
        o = self.oracle()
        seen = []

        def launch(co, names, words=4):
            seen.append(o.code_object)
            if o.code_object in loads:
                return 0, "# device=test arch=gfx1103\n", ""
            return 3, "", error
        o._assemble = lambda text, wave=32: ("k.co", "")
        o._launch = launch
        return o, seen

    def test_a_runtime_that_takes_the_default_keeps_it(self):
        o, seen = self.probe(loads={None, 5})
        o._probe()
        self.assertEqual(seen, [None])
        self.assertIsNone(o.code_object)

    def test_error_209_retries_as_v5_and_keeps_it(self):
        o, seen = self.probe(loads={5})
        self.assertEqual(o._probe(), "device=test arch=gfx1103")
        self.assertEqual((seen, o.code_object), ([None, 5], 5))

    def test_error_303_retries_as_v5_and_keeps_it(self):
        o, seen = self.probe(loads={5}, error="hipModuleLoadData(&mod, img) -> hip error 303")
        self.assertEqual(o._probe(), "device=test arch=gfx1103")
        self.assertEqual((seen, o.code_object), ([None, 5], 5))

    def test_a_209_that_stays_and_other_errors_are_unavailable(self):
        o, seen = self.probe(loads=set())
        with self.assertRaisesRegex(runner.Unavailable, "209"):
            o._probe()
        self.assertEqual(seen, [None, 5])
        o, seen = self.probe(loads=set(), error="hipModuleLoadData(&mod, img) -> hip error 303")
        with self.assertRaisesRegex(runner.Unavailable, "303"):
            o._probe()
        self.assertEqual(seen, [None, 5])
        o, seen = self.probe(loads={5}, error="hipMalloc -> hip error 2")
        with self.assertRaisesRegex(runner.Unavailable, "error 2"):
            o._probe()
        self.assertEqual(seen, [None])

    def test_a_launcher_quarantined_by_defender_is_unavailable(self):
        o = self.oracle()
        o.exe, o.dll = "leonida_hw_oracle_0123456789.exe", "amdhip64_7.dll"
        blocked = OSError("Operation did not complete successfully because the file contains a virus")
        blocked.winerror = 225              # only set by Python on Windows
        with unittest.mock.patch.object(runner.subprocess, "run", side_effect=blocked):
            with self.assertRaisesRegex(runner.Unavailable, "Defender.*exclusion"):
                o._launch("k.co", ["k0"])
        with unittest.mock.patch.object(runner.subprocess, "run", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                o._launch("k.co", ["k0"])

    def test_v5_that_does_not_assemble_is_unavailable(self):
        o, _ = self.probe(loads={5})
        o._assemble = lambda text, wave=32: (None, "unknown argument") if o.code_object else ("k.co", "")
        with self.assertRaisesRegex(runner.Unavailable, "unknown argument"):
            o._probe()

    def test_the_chosen_version_reaches_the_assembler(self):
        o = self.oracle()
        o.clang, o.lld = "clang", "ld.lld"
        ok = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(runner.subprocess, "run", return_value=ok) as run:
            o.workdir = tmp
            o._assemble("", 64)
            self.assertFalse([a for a in run.call_args_list[0].args[0] if "code-object" in a])
            o.code_object = 5
            o._assemble("", 64)
            args = run.call_args_list[2].args[0]
        self.assertIn("-mcode-object-version=5", args)
        self.assertIn("-mwavefrontsize64", args)

    def test_the_hip_linker_uses_amdgpu_elf_emulation(self):
        o = self.oracle()
        o.clang, o.lld = "clang", "ld.lld"
        ok = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner.subprocess, "run", return_value=ok) as run:
            o.workdir = tmp
            code, error = o._assemble("")
        self.assertEqual(error, "")
        run.assert_called_with(
            [o.lld, "-m", "elf64_amdgpu", "-shared", os.path.join(tmp, "k.o"), "-o", code],
            capture_output=True, text=True,
        )

    def test_true16_is_used_only_where_the_assembler_needs_it(self):
        bare_only = lambda text, wave=32: (None, "error") if ".l" in text else ("k.co", "")
        half_only = lambda text, wave=32: ("k.co", "") if ".l" in text else (None, "error")
        neither = lambda text, wave=32: (None, "error")
        for arch, assemble, want in (("gfx1150", bare_only, False), ("gfx1103", half_only, True),
                                     ("gfx1103", neither, False), ("gfx1036", half_only, False)):
            o = self.oracle(arch)
            o._assemble = assemble
            self.assertEqual(o._needs_true16(), want, arch)

    def test_a_chunk_is_assembled_with_that_spelling(self):
        for true16 in (False, True):
            o = self.oracle()
            o.true16, texts = true16, []
            o._assemble = lambda text, wave=32: (texts.append(text) or "k.co", "")
            o._launch = lambda co, names, words=4: (0, "k0 00003c00\n", "")
            kernel = asm.Kernel(["v_cvt_f16_f32 v8, v1", "s_endpgm"], words=1, raw=True)
            self.assertEqual(o.run([kernel], words=1), [[0x3C00]])
            self.assertEqual("v8.l" in texts[0], true16)

    def test_op_sel_halves_are_named_on_gfx11_and_later_even_without_true16(self):
        # The assembler that takes a bare v_cvt_f16_f32 (true16 probe: no) still needs v14.l / v5.l on these lines.
        line = "v_max_u16_e64 v14, s12, v5 op_sel:[1,0,0]"
        for arch, want in (("gfx1036", line), ("gfx1100", "v_max_u16_e64 v14.l, s12, v5.l op_sel:[1,0,0]"),
                           ("gfx1201", "v_max_u16_e64 v14.l, s12, v5.l op_sel:[1,0,0]")):
            o = self.oracle(arch)
            o.true16, texts = False, []
            o._assemble = lambda text, wave=32: (texts.append(text) or "k.co", "")
            o._launch = lambda co, names, words=4: (0, "k0 00000001\n", "")
            kernel = asm.Kernel([line, "s_endpgm"], words=1, raw=True)
            self.assertEqual(o.run([kernel], words=1), [[1]])
            self.assertIn(want, texts[0], arch)

    def test_a_mode_the_architecture_cannot_set_is_never_assembled(self):
        # FUZZ-43: gfx1170+ rejects .amdhsa_ieee_mode / .amdhsa_dx10_clamp. Those kernels come back as a reason; the
        # others in the same run are still assembled and launched.
        o = self.oracle("gfx1201")
        o.true16, texts = False, []
        o._assemble = lambda text, wave=32: (texts.append(text) or "k.co", "")
        o._launch = lambda co, names, words=4: (0, "".join(f"{n.split(':')[0]} 0000002a\n" for n in names), "")
        body = ["s_endpgm"]
        kernels = [asm.Kernel(body, words=1, raw=True, mode={"ieee": 0}),
                   asm.Kernel(body, words=1, raw=True, mode={"denorm32": 3}),
                   asm.Kernel(body, words=1, raw=True, mode={"dx10_clamp": 1, "round32": 1})]
        got = o.run(kernels, words=1)
        self.assertIn("MODE ieee cannot be set on gfx1201", got[0])
        self.assertEqual(got[1], [0x2A])
        self.assertIn("MODE dx10_clamp cannot be set on gfx1201", got[2])
        self.assertEqual(len(texts), 1)
        self.assertNotIn(".amdhsa_ieee_mode", texts[0])
        self.assertNotIn(".amdhsa_dx10_clamp", texts[0])

    def test_every_kernel_skipped_is_not_a_broken_chunk(self):
        o = self.oracle("gfx1201")
        o.true16 = False
        o._assemble = lambda text, wave=32: self.fail("nothing should be assembled")
        got = o.run([asm.Kernel(["s_endpgm"], words=1, raw=True, mode={"ieee": 1}) for _ in range(3)], words=1)
        self.assertTrue(all(isinstance(g, str) and "unsupported on gfx1170+" in g for g in got))


class Filters(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(asm.literals("s_add_u32 s4, 0x100, s5"), {0x100})
        self.assertEqual(asm.literals("s_add_u32 s4, 64, -16"), set())
        self.assertEqual(asm.literals("s_and_b32 s4, 0x100, 0x100"), {0x100})
        self.assertEqual(len(asm.literals("s_and_b32 s4, 0x100, 0x200")), 2)
        self.assertEqual(asm.literals("s_and_b32 s4, -1000, 0xfffffc18"), {0xFFFFFC18})   # one 32-bit value
        self.assertEqual(asm.literals("s_cmpk_lt_u32 s4, 0x8000"), set())     # SIMM16, not a literal
        self.assertEqual(asm.literals("s_movk_i32 s4, 0xffff"), set())

    def test_mode_skip_reasons(self):
        for arch in ("gfx1013", "gfx1030", "gfx1100", "gfx1150", "gfx1151"):
            self.assertIsNone(asm.mode_skip_reason({"ieee": 0, "dx10_clamp": 0}, arch), arch)
        for arch in ("gfx1170", "gfx1200", "gfx1201"):
            self.assertIn("ieee", asm.mode_skip_reason({"ieee": 0}, arch))
            self.assertIn(".amdhsa_dx10_clamp", asm.mode_skip_reason({"dx10_clamp": 1, "denorm32": 3}, arch))
            # the denormal and rounding fields still assemble there
            self.assertIsNone(asm.mode_skip_reason({"denorm32": 3, "denorm16_64": 0, "round32": 1,
                                                    "round16_64": 2}, arch), arch)
            self.assertIsNone(asm.mode_skip_reason(None, arch))

    def test_rdna23_skip_reason(self):
        for arch in ("gfx1013", "gfx1030", "gfx1100", "gfx1150", "gfx1151"):
            self.assertIsNone(asm.rdna23_skip_reason(arch), arch)
        for arch in ("gfx1170", "gfx1200", "gfx1201"):
            self.assertIn(arch, asm.rdna23_skip_reason(arch))

    def test_skip_reasons(self):
        self.assertIsNone(asm.skip_reason(PROG, "gfx1150"))
        self.assertIn("s_ff0", asm.skip_reason(PROG + ["s_ff0_i32_b32 s4, s5"], "gfx1150"))
        self.assertIsNone(asm.skip_reason(PROG + ["s_ff0_i32_b32 s4, s5"], "gfx1030"))      # GFX10 still has it
        self.assertIn("two different literals", asm.skip_reason(["s_and_b32 s4, 0x100, 0x200"], "gfx1030"))
        # GFX12 keeps GFX11's removals and drops the SOPK compares, which GFX11 still has.
        cmpk = PROG + ["s_cmpk_lt_i32 s4, 0x12"]
        self.assertIsNone(asm.skip_reason(PROG, "gfx1201"))
        self.assertIsNone(asm.skip_reason(cmpk, "gfx1150"))
        self.assertIn("GFX12", asm.skip_reason(cmpk, "gfx1201"))
        self.assertIn("s_ff0", asm.skip_reason(PROG + ["s_ff0_i32_b64 s[4:5], s[6:7]"], "gfx1201"))
        self.assertEqual(len([op for op in asm.GFX12_REMOVED if op.startswith("s_cmpk_")]), 12)


class VcvarsScript(unittest.TestCase):
    """The .cmd that calls vcvars64.bat and cl holds the checkout, %TEMP% and Visual Studio paths. A character
    outside ASCII in any of them (a user name, the Turkish default folder name) used to end in UnicodeEncodeError,
    which is not the Unavailable that callers turn into a skip."""
    NAME = "Ayşe Ğİı klasör"      # Turkish letters: none of them is ASCII

    def build(self, tmp):
        work = os.path.join(tmp, self.NAME)
        os.makedirs(work)
        src = os.path.join(work, "hw_oracle.c")
        out = os.path.join(work, "hw_oracle.exe")
        vcvars = os.path.join(work, "vcvars64.bat")
        argv = runner.build_command("vcvars", vcvars, src, out, work)
        with open(os.path.join(work, argv[-1]), "rb") as f:         # the script runs with cwd=work
            return argv, f.read(), (src, out, vcvars)

    def test_paths_outside_ascii_are_written_as_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv, raw, paths = self.build(tmp)
        self.assertEqual(argv[:3], ["cmd.exe", "/d", "/c"])
        text = raw.decode("utf-8")                       # strict: every byte is UTF-8
        for path in paths:
            self.assertIn(path, text)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))   # cmd.exe would read a BOM as part of the first command

    def test_the_code_page_is_set_before_any_path_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            _argv, raw, _paths = self.build(tmp)
        lines = raw.decode("utf-8").split("\r\n")
        self.assertEqual(lines[0], "@echo off")
        self.assertEqual(lines[1], "chcp 65001 >nul")
        self.assertTrue(all(line.isascii() for line in lines[:2]))
        self.assertTrue(lines[2].startswith('set "PATH='))
        self.assertTrue(lines[3].startswith('call "'))

    def test_ascii_paths_give_the_same_commands_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("work", "a b", "a & b"):
                with self.subTest(name=name):
                    work = os.path.join(tmp, name)
                    os.makedirs(work)
                    argv = runner.build_command("vcvars", "vcvars64.bat", "hw_oracle.c", "hw_oracle.exe", work)
                    with open(os.path.join(work, argv[-1]), "rb") as f:
                        # UTF-8, not ASCII: the /Fo folder sits in %TEMP%, which can hold a non-ASCII user name.
                        lines = f.read().decode("utf-8").split("\r\n")
                    self.assertEqual(lines[3], 'call "vcvars64.bat" >nul || exit /b 1')
                    self.assertTrue(lines[4].startswith("cl /nologo /O1 hw_oracle.c /Fehw_oracle.exe "))
                    # list2cmdline quotes /Fo when the work folder holds a space or a cmd metacharacter such as &.
                    self.assertIn(f"/Fo{work}{os.sep}", lines[4].replace('"', ""))

    def test_the_direct_compilers_write_no_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, self.NAME)
            os.makedirs(work)
            self.assertEqual(runner.build_command("gcc", "gcc", "a.c", "a.exe", work),
                             ["gcc", "-O1", "a.c", "-o", "a.exe"])
            self.assertEqual(runner.build_command("msvc", "cl.exe", "a.c", "a.exe", work)[0], "cl.exe")
            self.assertEqual(os.listdir(work), [])

    @unittest.skipUnless(sys.platform == "win32", "runs the script through cmd.exe")
    def test_the_script_runs_from_any_work_folder(self):
        # A full path through `cmd.exe /c` loses its quotes when the folder holds & or ^, and a bare name is not
        # found with NoDefaultCurrentDirectoryInExePath set. A stub vcvars shows the script got that far.
        broken = []
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "stub_vcvars.bat")
            with open(stub, "wb") as f:
                f.write(b'@echo off\r\necho reached> "%LEONIDA_TEST_MARKER%"\r\nexit /b 0\r\n')
            for name in ("plain", "sp (x)", "a & b", "a&b", "x^y", self.NAME):
                work = os.path.join(tmp, name)
                os.makedirs(work)
                marker = os.path.join(work, "reached.txt")
                argv = runner.build_command("vcvars", stub, "hw_oracle.c", "hw_oracle.exe", work)
                env = dict(os.environ, LEONIDA_TEST_MARKER=marker, NoDefaultCurrentDirectoryInExePath="1")
                subprocess.run(argv, cwd=work, env=env, capture_output=True, timeout=60)   # cl itself is absent
                if not os.path.exists(marker):
                    broken.append(name)
        self.assertEqual(broken, [], "build_hw_oracle.cmd did not run from these work folders")


class Unavailable(unittest.TestCase):
    def test_missing_tools_are_reported_not_raised_as_crashes(self):
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_AMD_CLANG": str(TOOLS / "no-such-clang.exe")}):
            with self.assertRaises(runner.Unavailable):
                runner.find_clang()
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_HIP_DLL": str(TOOLS / "no-such.dll")}):
            with self.assertRaises(runner.Unavailable):
                runner.find_hip_dll()

    def test_host_compiler_choice(self):
        here = str(Path(__file__))   # any file that exists stands in for a compiler
        cl = str(Path(here).with_name("cl.exe"))
        which = {"gcc": None, "cl": None}
        real_isfile = os.path.isfile
        no_msys = lambda p: False if "msys64" in p else real_isfile(p)   # noqa: E731 (hide an installed MSYS2)
        with unittest.mock.patch.object(runner.shutil, "which", side_effect=lambda n: which.get(n)), \
                unittest.mock.patch.object(runner.os.path, "isfile", side_effect=no_msys), \
                unittest.mock.patch.object(runner, "find_vcvars", return_value=None), \
                unittest.mock.patch.dict(os.environ, {"LEONIDA_HOST_CC": ""}):
            with self.assertRaises(runner.Unavailable) as cm:
                runner.find_cc()
            self.assertIn("Visual Studio", str(cm.exception))
            which["gcc"] = here
            self.assertEqual(runner.find_cc(), ("gcc", here))                  # gcc stays first
            if os.name == "nt":
                which["gcc"], which["cl"] = None, here
                self.assertEqual(runner.find_cc(), ("msvc", here))             # a Developer shell's cl
                which["cl"] = None
                with unittest.mock.patch.object(runner, "find_vcvars", return_value=here):
                    self.assertEqual(runner.find_cc(), ("vcvars", here))       # VS found through vswhere
            with unittest.mock.patch.dict(os.environ, {"LEONIDA_HOST_CC": here}):
                self.assertEqual(runner.find_cc(), ("gcc", here))
        with unittest.mock.patch.object(runner.os.path, "isfile", return_value=True), \
                unittest.mock.patch.dict(os.environ, {"LEONIDA_HOST_CC": cl}):
            self.assertEqual(runner.find_cc(), ("msvc", cl))                   # the override may be cl.exe

    def test_host_compiler_commands(self):
        work = os.path.join(os.sep, "w")
        self.assertEqual(runner.build_command("gcc", "gcc", "a.c", "o.exe", work), ["gcc", "-O1", "a.c", "-o", "o.exe"])
        cl = runner.build_command("msvc", "cl.exe", "a.c", "o.exe", work)
        self.assertEqual(cl[:4], ["cl.exe", "/nologo", "/O1", "a.c"])
        self.assertIn("/Feo.exe", cl)
        self.assertIn("/Fo" + os.path.join(work, ""), cl)                      # objects stay out of the cwd
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            argv = runner.build_command("vcvars", r"C:\VS 2022\vcvars64.bat", "a.c", "o.exe", tmp)
            self.assertEqual(argv[:3], ["cmd.exe", "/d", "/c"])
            script = (Path(tmp) / argv[3]).read_text(encoding="utf-8")       # UTF-8; tmp may be non-ASCII
            self.assertIn('call "C:\\VS 2022\\vcvars64.bat"', script)          # quoted: the path has a space
            self.assertIn("cl /nologo /O1 a.c /Feo.exe", script)
            self.assertIn("Installer", script)                                 # vcvars needs vswhere on PATH

    def test_salu_diff_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "salu_diff.py"), "--oracle", "hw", "--programs", "3"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_an_unsupported_architecture_is_unavailable_not_a_traceback(self):
        # Oracle() checks the architecture right after finding the tools; stub those so this runs anywhere.
        with unittest.mock.patch.object(runner, "find_clang", return_value="clang"), \
                unittest.mock.patch.object(runner, "_find_lld", return_value="ld.lld"), \
                unittest.mock.patch.object(runner, "find_hip_dll", return_value="amdhip64_7.dll"):
            with self.assertRaises(runner.Unavailable) as cm:
                runner.Oracle(arch="gfx90a")
        self.assertIn("gfx90a", str(cm.exception))

    def test_unknown_oracle_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "salu_diff.py"), "--oracle", "gpu"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


class ClangOnPath(unittest.TestCase):
    """find_clang when the compiler comes from PATH: real files in a folder, and a stand-in for what each compiler
    prints for `--print-targets` and `--version`."""
    EXT = ".exe" if os.name == "nt" else ""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def tool(self, *parts):
        p = self.root.joinpath(*parts[:-1], parts[-1] + self.EXT)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return str(p)

    def tools(self, on_path, installed):
        """PATH holds on_path ({name: file}); the compiler `c` reports installed[c] as its InstalledDir."""
        def run(cmd, **kw):
            out = "amdgcn" if "--print-targets" in cmd else f"clang version 23.0.0\nInstalledDir: {installed[cmd[0]]}\n"
            return subprocess.CompletedProcess(cmd, 0, out, "")
        patches = [unittest.mock.patch.object(runner.shutil, "which", side_effect=lambda n: on_path.get(n)),
                   unittest.mock.patch.object(runner.subprocess, "run", side_effect=run),
                   unittest.mock.patch.dict(os.environ, {"LEONIDA_AMD_CLANG": "", "HIP_PATH": "", "ROCM_PATH": ""})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_a_launcher_on_path_stands_for_the_clang_it_is_installed_as(self):
        # The rocm-sdk wheels in a Python outside the checkout: `amdclang` in Scripts has the amdgcn target, while
        # clang, ld.lld and amdgpu-arch sit in the folder it names as InstalledDir.
        launcher = self.tool("Scripts", "amdclang")
        clang, lld = self.tool("llvm", "bin", "clang"), self.tool("llvm", "bin", "ld.lld")
        self.tools({"amdclang": launcher}, {launcher: str(self.root / "llvm" / "bin")})
        found = runner.find_clang()
        self.assertEqual(found, clang)
        self.assertEqual(runner._find_lld(found), lld)

    def test_a_clang_with_its_linker_beside_it_is_used_where_it_was_found(self):
        clang = self.tool("bin", "clang")
        self.tool("bin", "ld.lld")
        self.tool("other", "clang")
        self.tools({"clang": clang}, {clang: str(self.root / "other")})
        self.assertEqual(runner.find_clang(), clang)

    def test_a_launcher_that_names_a_folder_without_a_clang_is_kept(self):
        launcher = self.tool("Scripts", "amdclang")
        self.tools({"amdclang": launcher}, {launcher: str(self.root / "nothing-here")})
        found = runner.find_clang()
        self.assertEqual(found, launcher)
        with self.assertRaises(runner.Unavailable):
            runner._find_lld(found)


def relocatable(sections, symbols):
    """A little-endian ELF64 relocatable object: sections is [(name, bytes, address)], symbols [(name, section
    number from 1, value)]. Only what drm.text_section reads is filled in."""
    names = [""] + [s[0] for s in sections] + [".symtab", ".strtab", ".shstrtab"]
    shstr, name_at = b"", []
    for n in names:
        name_at.append(len(shstr))
        shstr += n.encode() + b"\0"
    strtab, symtab = b"\0", bytes(24)
    for name, shndx, value in symbols:
        symtab += struct.pack("<IBBHQQ", len(strtab), 0x12, 0, shndx, value, 0)
        strtab += name.encode() + b"\0"
    bodies = [b""] + [s[1] for s in sections] + [symtab, strtab, shstr]
    addresses = [0] + [s[2] for s in sections] + [0, 0, 0]
    types = [0] + [1] * len(sections) + [2, 3, 3]
    out, offsets = bytearray(64), []
    for body in bodies:
        offsets.append(len(out))
        out += body
    shoff = len(out)
    strtab_index = len(sections) + 2
    for i, body in enumerate(bodies):
        link = strtab_index if types[i] == 2 else 0
        out += struct.pack("<IIQQQQIIQQ", name_at[i], types[i], 0, addresses[i], offsets[i], len(body), link, 0, 1, 0)
    out[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", out, 0x28, shoff)
    struct.pack_into("<HHH", out, 0x3A, 64, len(bodies), len(bodies) - 1)
    return bytes(out)


class LinuxBackend(unittest.TestCase):
    """drm.py without a GPU."""

    def test_text_section_and_kernel_offsets(self):
        code = bytes(range(256)) * 3
        elf = relocatable([(".rodata", b"\xaa" * 64, 0), (".text", code, 0)],
                          [("k0", 2, 0), ("k1", 2, 256), ("k2", 2, 512), ("k0.kd", 1, 0)])
        text, symbols = drm.text_section(elf)
        self.assertEqual(text, code)
        self.assertEqual(symbols, {"k0": 0, "k1": 256, "k2": 512})   # the descriptor in .rodata is not a kernel
        # a linked object gives addresses: offsets are counted from where .text starts
        linked = relocatable([(".text", code, 0x1000)], [("k1", 1, 0x1100)])
        self.assertEqual(drm.text_section(linked)[1], {"k1": 256})
        with self.assertRaises(ValueError):
            drm.text_section(b"MZ" + bytes(100))
        with self.assertRaises(ValueError):
            drm.text_section(relocatable([(".data", b"x", 0)], []))

    @staticmethod
    def descriptor(rsrc1=0xE0AC0001, rsrc2=0x84, properties=0x0408, group=0, private=0, rsrc3=0):
        """A 64-byte kernel descriptor (the amdhsa layout): segment sizes first, RSRC3, RSRC1, RSRC2 and the code
        properties from byte 44."""
        return struct.pack("<II", group, private) + bytes(36) + struct.pack("<IIIH", rsrc3, rsrc1, rsrc2, properties) + bytes(6)

    def test_registers_come_from_the_kernel_descriptors(self):
        rodata = self.descriptor() + self.descriptor(rsrc1=0xE0AC3007, properties=0x0008)
        elf = relocatable([(".rodata", rodata, 0), (".text", bytes(512), 0)],
                          [("k0", 2, 0), ("k1", 2, 256), ("k0.kd", 1, 0), ("k1.kd", 1, 64)])
        # k0: a 32-lane kernel with the assembler's default float mode; k1: 64 lanes, f32 rounding towards zero
        self.assertEqual(drm.kernel_registers(elf), {"k0": (0xE0AC0001, 0x84, 32), "k1": (0xE0AC3007, 0x84, 64)})

    def test_a_descriptor_the_launcher_cannot_honour_is_refused(self):
        def registers(**fields):
            return drm.kernel_registers(relocatable([(".rodata", self.descriptor(**fields), 0), (".text", bytes(256), 0)],
                                                    [("k0", 2, 0), ("k0.kd", 1, 0)]))
        self.assertEqual(registers(), {"k0": (0xE0AC0001, 0x84, 32)})
        for fields in ({"properties": 0x040A},      # also wants the dispatch pointer
                       {"properties": 0x0400},      # no kernarg pointer
                       {"private": 64},             # scratch memory
                       {"rsrc3": 1},
                       {"rsrc2": 0x85},             # the private segment
                       {"rsrc2": 0x86},             # three user SGPRs
                       {"group": 65536 + 512}):     # more LDS than a workgroup gets
            with self.assertRaises(RuntimeError, msg=str(fields)):
                registers(**fields)
        with self.assertRaises(ValueError):          # a descriptor cut short
            drm.kernel_registers(relocatable([(".rodata", bytes(32), 0), (".text", bytes(256), 0)], [("k0.kd", 1, 0)]))

    def test_the_group_segment_becomes_the_lds_size_field(self):
        def registers(**fields):
            return drm.kernel_registers(relocatable([(".rodata", self.descriptor(**fields), 0), (".text", bytes(256), 0)],
                                                    [("k0", 2, 0), ("k0.kd", 1, 0)]))
        # The assembler leaves COMPUTE_PGM_RSRC2.LDS_SIZE (bits 15..23, 512-byte units) 0 and records the size in the
        # descriptor's group segment: measured on gfx1036, 4096 bytes of LDS give group 4096 and rsrc2 0x84.
        self.assertEqual(registers(group=4096), {"k0": (0xE0AC0001, 0x84 | 8 << 15, 32)})
        self.assertEqual(registers(group=4), {"k0": (0xE0AC0001, 0x84 | 1 << 15, 32)})    # rounded up to 512
        self.assertEqual(registers(group=65536), {"k0": (0xE0AC0001, 0x84 | 128 << 15, 32)})
        # A descriptor that already carries a size gets the group segment's, as the HIP runtime programs it.
        self.assertEqual(registers(rsrc2=0x84 | 1 << 15), {"k0": (0xE0AC0001, 0x84, 32)})

    def test_launcher_tokens_carry_the_offset_and_the_registers(self):
        offsets = {"k0": 0, "k1": 256}
        registers = {"k0": (0xE0AC0001, 0x84, 32), "k1": (0xE0AC3007, 0x84, 64)}
        self.assertEqual(drm.launcher_tokens(["k0", "k1"], offsets, registers),
                         ["k0@0/0xe0ac0001/0x84/32", "k1@256/0xe0ac3007/0x84/64"])
        self.assertEqual(drm.launcher_tokens(["k1:64:896"], offsets, registers),
                         ["k1@256/0xe0ac3007/0x84/64:64:896"])                        # name:threads:words
        with self.assertRaises(RuntimeError):
            drm.launcher_tokens(["k7"], offsets, registers)
        with self.assertRaises(RuntimeError):
            drm.launcher_tokens(["k0"], offsets, {})                                  # a kernel without a descriptor

    def test_a_node_must_be_named_and_must_not_drive_a_display(self):
        with tempfile.TemporaryDirectory() as root:
            sysfs = Path(root) / "sys"
            for card, node, connectors in (("card0", "renderD129", {"DP-1": "disconnected", "Writeback-1": "unknown"}),
                                           ("card1", "renderD128", {"DP-1": "connected", "HDMI-A-1": "disconnected"})):
                for name, status in connectors.items():
                    d = sysfs / node / "device" / "drm" / card / f"{card}-{name}"
                    d.mkdir(parents=True)
                    (d / "status").write_text(status + "\n", encoding="ascii")
            free, busy = Path(root) / "renderD129", Path(root) / "renderD128"
            free.touch()
            busy.touch()
            self.assertEqual(drm.connected_displays(str(free), str(sysfs)), [])
            self.assertEqual(drm.connected_displays(str(busy), str(sysfs)), ["card1-DP-1"])
            with self.assertRaisesRegex(runner.Unavailable, "LEONIDA_HW_DRM_NODE"):
                drm.find_node({}, str(sysfs))                                     # no default device
            with self.assertRaisesRegex(runner.Unavailable, "does not exist"):
                drm.find_node({"LEONIDA_HW_DRM_NODE": str(Path(root) / "renderD200")}, str(sysfs))
            self.assertEqual(drm.find_node({"LEONIDA_HW_DRM_NODE": str(free)}, str(sysfs)), str(free))
            with self.assertRaisesRegex(runner.Unavailable, "drives a display"):
                drm.find_node({"LEONIDA_HW_DRM_NODE": str(busy)}, str(sysfs))
            allowed = {"LEONIDA_HW_DRM_NODE": str(busy), "LEONIDA_HW_DRM_ALLOW_DISPLAY": "1"}
            self.assertEqual(drm.find_node(allowed, str(sysfs)), str(busy))

    def test_linux_gets_the_kernel_driver_backend(self):
        with unittest.mock.patch.object(sys, "platform", "linux"), \
                unittest.mock.patch.dict(os.environ, {"LEONIDA_HW_DRM_NODE": ""}):
            with self.assertRaisesRegex(runner.Unavailable, "LEONIDA_HW_DRM_NODE"):
                runner.open_oracle()


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_the_dll_override_reaches_the_launcher(self):
        if isinstance(self.oracle, drm.DrmOracle):
            self.skipTest("the Linux backend loads no HIP runtime")
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_HIP_DLL": str(Path(__file__))}):   # exists, not a DLL
            with self.assertRaises(runner.Unavailable):
                runner.open_oracle()

    def test_a_chunk_the_assembler_rejects_entirely_is_a_harness_error(self):
        with self.assertRaises(RuntimeError):
            self.oracle.run([["s_add_u32 s4, 0x100, 0x200"] + PROG[-4:]])

    def test_every_kernel_returns_its_own_words(self):
        # 300 kernels that differ only in what they store. A launcher that ran a neighbour's commands (the Linux
        # one did, for the second kernel of a process, while it rewrote one command buffer) fails here.
        programs = [[f"v_mov_b32 v{8 + j}, 0x{0x1000 * (i + 1) + j:x}" for j in range(4)] for i in range(300)]
        for i, words in enumerate(self.oracle.run(programs)):
            self.assertEqual(words, [0x1000 * (i + 1) + j for j in range(4)], f"kernel {i}")

    def test_the_float_mode_of_the_kernel_descriptor_reaches_the_hardware(self):
        # 1.0 + 1.5 * 2^-24 is 0x3f800001 when f32 rounds to nearest or up, 0x3f800000 when it rounds down or
        # towards zero. The HIP runtime starts a kernel in the mode its descriptor has; the Linux launcher must too.
        if not isinstance(self.oracle, drm.DrmOracle):
            self.skipTest("checks the Linux launcher's use of the descriptor")
        program = ["v_mov_b32 v1, 0x3f800000", "v_mov_b32 v2, 0x33c00000", "v_add_f32 v8, v1, v2",
                   "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"]
        got = []
        for mode in range(4):
            text, names, _ = asm.module_text([program], self.oracle.arch)
            marker = "  .amdhsa_system_vgpr_workitem_id 0\n"
            self.assertEqual(text.count(marker), 1)
            code, err = self.oracle._assemble(text.replace(marker, marker + f"  .amdhsa_float_round_mode_32 {mode}\n"))
            self.assertIsNotNone(code, err)
            rc, out, err = self.oracle._launch(code, names)
            self.assertEqual(rc, 0, err)
            got.append(int(out.splitlines()[-1].split()[1], 16))
        self.assertEqual(got, [0x3F800001, 0x3F800001, 0x3F800000, 0x3F800000])

    def test_f16_conversions_run_with_either_assembler_spelling(self):
        # 1.0 to f16 keeps the high half of the register; the low half back to f32 ignores it.
        program = ["v_mov_b32 v1, 0x3f800000", "v_mov_b32 v8, 0xabcd0000", "v_cvt_f16_f32 v8, v1",
                   "v_cvt_f32_f16 v9, v8", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"]
        self.assertEqual(self.oracle.run([program]), [[0xABCD3C00, 0x3F800000, 0, 0]])

    def test_a_kernel_gets_the_lds_its_descriptor_asks_for(self):
        # The group segment is the workgroup's LDS. A lane writes 0x1234 there, a barrier, and reads it back; the
        # Linux launcher has to program COMPUTE_PGM_RSRC2.LDS_SIZE from the descriptor for this to read back.
        kernel = asm.Kernel(["v_mov_b32 v1, 0x1234", "ds_write_b32 v0, v1", "s_barrier", "ds_read_b32 v8, v0",
                             "s_waitcnt lgkmcnt(0)", "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"],
                            wave=32, threads=1, words=4, lds=4096)
        self.assertEqual(self.oracle.run([kernel]), [[0x1234, 0, 0, 0]])

    def test_more_than_4096_words_of_output(self):
        # 1024 threads store 8 dwords each. The Linux launcher capped its buffer at 4096 words, which is fewer than
        # image_fuzz's 64x64 cases (16,384) and this one need.
        lines = ["v_lshlrev_b32 v2, 5, v0", "v_mov_b32 v1, 0xabcd0001"]
        lines += [f"global_store_b32 v2, v1, s[2:3] offset:{4 * j}" for j in range(8)]
        lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
        kernel = asm.Kernel(lines, wave=32, threads=1024, words=8192, raw=True)
        self.assertEqual(self.oracle.run([kernel]), [[0xABCD0001] * 8192])

    def test_runs_a_program_and_reports_a_rejected_one(self):
        bad = ["s_add_u32 s4, 0x100, 0x200"] + PROG[-4:]     # two literals: the assembler refuses it
        good, rejected = self.oracle.run([PROG, bad])
        self.assertEqual(good, [0x12345678, 7, 0x1234567F, 0])
        self.assertIsInstance(rejected, str)
        self.assertIn("rejected", rejected)


class KernelResources(unittest.TestCase):
    """Kernel(vgprs=, sgprs=): the register counts reach the descriptor and the metadata; no hardware needed."""

    def test_lds_reaches_the_descriptor_and_the_metadata(self):
        # The descriptor's group segment is where the Linux launcher reads the LDS size from (COMPUTE_PGM_RSRC2.
        # LDS_SIZE), so a wrapped kernel that uses LDS has to declare it there too, not only in the metadata.
        text, _, _ = asm.module_text([asm.Kernel(["s_endpgm"], lds=4096),
                                      asm.Kernel(["s_endpgm"], raw=True, lds=8192)], "gfx1150")
        self.assertEqual(text.count(".amdhsa_group_segment_fixed_size 4096"), 1)
        self.assertEqual(text.count(".amdhsa_group_segment_fixed_size 8192"), 1)
        self.assertEqual(text.count("    .group_segment_fixed_size: 4096"), 1)
        self.assertEqual(text.count("    .group_segment_fixed_size: 8192"), 1)

    def test_requested_registers_are_in_the_text(self):
        k = asm.Kernel(["s_endpgm"], wave=32, threads=64, words=8, raw=True, vgprs=40, sgprs=72)
        text, names, _ = asm.module_text([k, asm.Kernel(["s_endpgm"], raw=True)], "gfx1150")
        self.assertIn(".amdhsa_next_free_vgpr 40", text)
        self.assertIn(".amdhsa_next_free_sgpr 72", text)
        self.assertIn(".sgpr_count: 72", text)
        self.assertIn(".vgpr_count: 40", text)
        self.assertEqual(text.count(".amdhsa_next_free_vgpr"), 2)
        self.assertIn(".amdhsa_next_free_vgpr 32", text)      # the second kernel keeps the defaults
        self.assertIn(".amdhsa_next_free_sgpr 40", text)
        self.assertEqual(names, ["k0", "k1"])

    def test_registers_the_lines_name_are_allocated(self):
        self.assertEqual(asm.regs_used(["v_mov_b32 v200, s7", "s_mov_b64 s[40:41], -1", "v_mad_u64_u32 v[8:9], "
                                        "s20, v1, v1, v[2:3] ; v250 in a comment", "v_add_co_u32 v3, vcc_lo, v4, s0"]),
                         (201, 42))
        self.assertEqual(asm.regs_used(["s_endpgm"]), (0, 0))
        self.assertEqual(asm.register_counts(["v_mov_b32 v1, 0"], 16, 32), (16, 32))     # never lowered
        self.assertEqual(asm.register_counts(["v_mov_b32 v255, s103"], 16, 32), (256, 104))
        k = asm.raw_kernel_text("k0", ["v_mov_b32 v40, 0", "s_endpgm"], "gfx1036", vgprs=8, sgprs=16)
        self.assertIn(".amdhsa_next_free_vgpr 41", k)
        self.assertIn(".amdhsa_next_free_sgpr 16", k)

    def test_wrapped_kernels_take_the_wave_and_registers_of_the_kernel(self):
        # The wrapped kernel's descriptor was fixed at wave32, 16 VGPRs and 32 SGPRs: on gfx1036 v16 and up read
        # back 0 and Kernel(wave=64) ran as wave32.
        lines = ["v_mov_b32 v40, 1", "s_mov_b32 s60, 2"] + PROG[-4:]
        text, _, _ = asm.module_text([asm.Kernel(lines, wave=64)], "gfx1036", wave=64)
        self.assertIn(".amdhsa_wavefront_size32 0", text)
        self.assertIn(".amdhsa_next_free_vgpr 41", text)
        self.assertIn(".amdhsa_next_free_sgpr 61", text)
        self.assertIn(".vgpr_count: 41", text)          # the metadata says what the descriptor says
        self.assertIn(".sgpr_count: 61", text)
        text, _, _ = asm.module_text([asm.Kernel(PROG)], "gfx1036")
        self.assertIn(".amdhsa_wavefront_size32 1", text)
        self.assertIn(".amdhsa_next_free_vgpr 32", text)                # Kernel's default counts
        text, _, _ = asm.module_text([PROG], "gfx1036")
        self.assertIn(".amdhsa_next_free_vgpr 16", text)                # a plain program keeps the old ones
        self.assertIn(".amdhsa_next_free_sgpr 32", text)


class WrappedKernelsOnHardware(unittest.TestCase):
    """A wrapped kernel runs at the wave size it asks for and keeps every register it names."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_wave_size_and_high_registers(self):
        # Lane 40 of the lane ids of a full EXEC is 40 in a wave64 and lane 8 (40 mod 32) in a wave32.
        ids = ["s_mov_b64 exec, -1", "v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1",
               "v_readlane_b32 s4, v1, 40", "s_mov_b64 exec, 1"]
        regs = ["s_mov_b32 s90, 0x5a", "v_mov_b32 v16, 0x16", "v_mov_b32 v200, 0x200", "v_mov_b32 v9, s90",
                "v_mov_b32 v10, v16", "v_mov_b32 v11, v200", "v_mov_b32 v8, s4"]
        for wave, lane in ((32, 8), (64, 40)):
            with self.subTest(wave=wave):
                got = self.oracle.run([asm.Kernel(ids + regs, wave=wave)])
                self.assertEqual(got, [[lane, 0x5A, 0x16, 0x200]])


if __name__ == "__main__":
    unittest.main()

