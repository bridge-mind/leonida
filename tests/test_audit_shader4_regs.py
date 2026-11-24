"""Unit tests for scripts/audit_shader4_regs.py (standard library only, synthetic .s/.fssm/.dxbc)."""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import audit_shader4_regs as audit_mod  # noqa: E402


def make_dxbc(chunks):
    hdr_size = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(hdr_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    hdr = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, hdr_size + len(body), len(chunks))
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr) + bytes(body)


def make_ms_psv0(tx=128, ty=1, tz=1, maxv=64, maxp=64, topo=2):
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, maxv, maxp)
    r[24] = 13
    r[27] = topo
    struct.pack_into("<III", r, 36, tx, ty, tz)
    return b"PSV0", struct.pack("<I", len(r)) + bytes(r)


class AuditShader4RegsTests(unittest.TestCase):
    def test_inline_immediate_reads_destination_before_write(self):
        insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\ts_addk_i32 s3, 1",
            "\ts_addk_i32 s3, 2",
            "\ts_mulk_i32 s4, 2",
            "\ts_cmovk_i32 s5, 3",
            "\ts_movk_i32 s6, 0",
        )]
        self.assertEqual(audit_mod.rbw_registers(insns), ([3, 4, 5], [], ["s_addk_i32 s3, 1"]))

    def test_stores_atomics_dual_dest_valu_and_nsa_lists(self):
        # 1. tbuffer_store / s_store / s_buffer_store have no destination operand.
        store_insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\ttbuffer_store_format_xyzw v[0:1], v2, s[4:7], 0 idxen",
            "\ts_store_dword s8, s[10:11], 0x0",
            "\ts_buffer_store_dword s9, s[4:7], 0x0",
            "\tv_mov_b32 v3, v0",
            "\ts_mov_b32 s12, s8",
        )]
        self.assertEqual(audit_mod.rbw_registers(store_insns), ([4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2], []))

        # 2. Non-glc atomics do not write args[0]; glc global_atomic writes vdst at args[0] without reading it.
        atomic_insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\tbuffer_atomic_add v0, v1, s[4:7], 0 offen",
            "\tglobal_atomic_add v[2:3], v4, off",
            "\tglobal_atomic_add v5, v[6:7], v8, off glc",
            "\tv_mov_b32 v9, v0",
            "\tv_mov_b32 v10, v2",
            "\tv_mov_b32 v11, v5",
        )]
        self.assertEqual(audit_mod.rbw_registers(atomic_insns), ([4, 5, 6, 7], [0, 1, 2, 3, 4, 6, 7, 8], []))

        # 3. v_mad_u64_u32 / v_mad_i64_i32 / v_div_scale_f32 write both args[0] (VGPR) and args[1] (SGPR mask).
        dual_insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\tv_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[6:7]",
            "\ts_mov_b32 s8, s2",
        )]
        self.assertEqual(audit_mod.rbw_registers(dual_insns), ([], [4, 5, 6, 7], []))

        # 4. NSA vlist and slist operands report unread registers.
        nsa_insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\timage_sample v0, [v1, v3], s[4:11], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        )]
        self.assertEqual(audit_mod.rbw_registers(nsa_insns), (list(range(4, 16)), [1, 3], []))

    def test_direct_to_lds_load_and_accumulating_dot_forms(self):
        def rbw(*lines):
            return audit_mod.rbw_registers([audit_mod.shader_ir.parse_line("\t" + line) for line in lines])

        # A direct-to-LDS load writes LDS: its vdst is the LDS address, so a later read of it is read-before-write.
        self.assertEqual(rbw("buffer_load_dword v5, v1, s[4:7], 0 offen lds", "v_mov_b32 v6, v5"),
                         ([4, 5, 6, 7], [1, 5], []))
        self.assertEqual(rbw("buffer_load_dword lds:v5, v1, s[4:7], 0 offen", "v_mov_b32 v6, v5"),
                         ([4, 5, 6, 7], [1, 5], []))
        # The same load without the lds bit writes v5, so the later read is not one.
        self.assertEqual(rbw("buffer_load_dword v5, v1, s[4:7], 0 offen", "v_mov_b32 v6, v5"),
                         ([4, 5, 6, 7], [1], []))

        # The "c" dot forms accumulate into D; the plain forms overwrite it.
        for op, dst in (("v_dot2c_f32_f16", 5), ("v_dot2c_i32_i16", 5), ("v_dot2c_u32_u16", 5),
                        ("v_dot4c_i32_i8", 6), ("v_dot4c_u32_u8", 6),
                        ("v_dot8c_i32_i4", 7), ("v_dot8c_u32_u4", 7)):
            self.assertEqual(rbw(f"{op} v{dst}, v1, v2"), ([], [1, 2, dst], []), op)
        for op, dst in (("v_dot2_f32_f16", 5), ("v_dot4_i32_i8", 6)):
            self.assertEqual(rbw(f"{op} v{dst}, v1, v2"), ([], [1, 2], []), op)

    def test_non_returning_ds_ops_read_every_operand(self):
        def rbw(*lines):
            return audit_mod.rbw_registers([audit_mod.shader_ir.parse_line("\t" + line) for line in lines])

        # A later read of the LDS address register still counts: the non-returning op did not write it.
        self.assertEqual(rbw("ds_add_u32 v1, v2", "v_mov_b32 v3, v1"), ([], [1, 2], []))
        # Every non-_rtn DS atomic, the 64-bit and float forms, src2 forms and GWS ops write nothing.
        for line, reads in (
            ("ds_sub_u32 v1, v2 offset:16", [1, 2]),
            ("ds_rsub_u32 v1, v2", [1, 2]),
            ("ds_inc_u32 v1, v2", [1, 2]),
            ("ds_dec_u32 v1, v2", [1, 2]),
            ("ds_min_i32 v1, v2", [1, 2]),
            ("ds_max_u32 v1, v2", [1, 2]),
            ("ds_and_b32 v1, v2", [1, 2]),
            ("ds_or_b32 v1, v2", [1, 2]),
            ("ds_xor_b32 v1, v2", [1, 2]),
            ("ds_mskor_b32 v1, v2, v4", [1, 2, 4]),
            ("ds_cmpst_b32 v1, v2, v4", [1, 2, 4]),
            ("ds_cmpst_f32 v1, v2, v4", [1, 2, 4]),
            ("ds_add_f32 v1, v2", [1, 2]),
            ("ds_max_f32 v1, v2", [1, 2]),
            ("ds_add_u64 v1, v[2:3]", [1, 2, 3]),
            ("ds_cmpst_b64 v1, v[2:3], v[4:5]", [1, 2, 3, 4, 5]),
            ("ds_min_f64 v1, v[2:3]", [1, 2, 3]),
            ("ds_add_src2_u32 v1", [1]),
            ("ds_gws_init v2 gds", [2]),
            ("ds_gws_barrier v2 gds", [2]),
            ("ds_write_b32 v1, v2", [1, 2]),
            ("ds_write2_b32 v1, v2, v4 offset1:1", [1, 2, 4]),
        ):
            with self.subTest(line=line):
                self.assertEqual(rbw(line, "v_mov_b32 v8, v1", "v_mov_b32 v9, v2"),
                                 ([], sorted({1, 2} | set(reads)), []))
        # Returning forms keep args[0] (v6, or v[6:7]) as a destination that is written and not read.
        for line, reads in (
            ("ds_add_rtn_u32 v6, v1, v2", [1, 2]),
            ("ds_cmpst_rtn_b32 v6, v1, v2, v4", [1, 2, 4]),
            ("ds_wrxchg_rtn_b32 v6, v1, v2", [1, 2]),
            ("ds_wrxchg2_rtn_b32 v[6:7], v1, v2, v4", [1, 2, 4]),
            ("ds_append v6", []),
            ("ds_consume v6", []),
            ("ds_swizzle_b32 v6, v1 offset:swizzle(BROADCAST,2,0)", [1]),
            ("ds_permute_b32 v6, v1, v2", [1, 2]),
            ("ds_bpermute_b32 v6, v1, v2", [1, 2]),
            ("ds_read_b32 v6, v1", [1]),
            ("ds_read2_b32 v[6:7], v1 offset1:1", [1]),
            ("ds_ordered_count v6, v1 gds", [1]),
        ):
            with self.subTest(line=line):
                self.assertEqual(rbw(line, "v_mov_b32 v8, v6"), ([], reads, []))

    def test_vector_carry_in_and_out_are_distinct_in_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as disasm:
            for op in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
                path = os.path.join(disasm, f"cs_{op}.s")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"\t{op}_e64 v1, s3, v2, v4, s8\n"
                            "\ts_mov_b32 s9, s3\n\ts_endpgm\n")

            report = audit_mod.audit(disasm)["cs"]
            self.assertEqual(report["count"], 3)
            self.assertEqual(report["rbw_sgpr_counts"], {"s8": 3})
            self.assertEqual(report["rbw_vgpr_counts"], {"v2": 3, "v4": 3})
            self.assertEqual(report["s3_rbw_op_patterns"], {})

    def test_negative_limit_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            for name in ("a_cs", "b_cs"):
                with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8") as f:
                    f.write("\ts_endpgm\n")
            with open(os.path.join(disasm, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"a_cs": {"stage": "cs"}, "b_cs": {"stage": "cs"}}, f)

            out_json = os.path.join(tmp, "report.json")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    audit_mod.main([disasm, "--limit", "-1", "--json", out_json])
            self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(len(stderr.getvalue().splitlines()), 1)
            self.assertIn("--limit", stderr.getvalue())
            self.assertFalse(os.path.exists(out_json))

            with self.assertRaisesRegex(ValueError, "limit"):
                audit_mod.audit(disasm, limit=-1)
            for limit, count in ((None, 2), (0, 0), (1, 1)):
                self.assertEqual(audit_mod.audit(disasm, limit=limit)["cs"]["count"], count)

    def test_directory_named_like_shader_is_skipped_by_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            # A directory that sorts before the real shader must not consume the --limit slot.
            os.mkdir(os.path.join(disasm, "cs_a.s"))
            with open(os.path.join(disasm, "cs_b.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm\n")
            with open(os.path.join(disasm, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"cs_b": {"stage": "cs"}}, f)

            # Without scanning regular files only, the directory cs_a.s would be parsed,
            # raising IsADirectoryError or consuming the sole limit slot.
            self.assertEqual(audit_mod.audit(disasm, limit=1)["cs"]["count"], 1)

    def test_bad_input_and_output_paths_are_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            with open(os.path.join(disasm, "cs_sample.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm\n")
            plain_file = os.path.join(tmp, "not-a-folder.bin")
            with open(plain_file, "wb") as f:
                f.write(b"RPF9")
            missing = os.path.join(tmp, "missing")
            a_folder = os.path.join(tmp, "out-folder")
            os.mkdir(a_folder)

            def run(argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = audit_mod.main(argv)
                return rc, out.getvalue(), err.getvalue()

            for argv, needle in (
                ([missing], "no such folder"),
                ([plain_file], "not a folder"),
                ([disasm, "--fssm", missing], "--fssm"),
                ([disasm, "--dxbc", plain_file], "--dxbc"),
            ):
                rc, stdout, stderr = run(argv)
                self.assertEqual(rc, 2, argv)
                self.assertEqual(stdout, "", argv)
                self.assertEqual(len(stderr.splitlines()), 1, argv)
                self.assertIn(needle, stderr, argv)

            # The report's folder is created, so a finished audit is not lost.
            out_json = os.path.join(tmp, "nested", "deeper", "report.json")
            rc, stdout, stderr = run([disasm, "--json", out_json])
            self.assertEqual((rc, stderr), (0, ""))
            self.assertIn("wrote ", stdout)
            with open(out_json, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["cs"]["count"], 1)

            # A report path that cannot be written is one line, not a traceback.
            rc, stdout, stderr = run([disasm, "--json", a_folder])
            self.assertEqual(rc, 1)
            self.assertEqual(len(stderr.splitlines()), 1)
            self.assertIn("cannot write", stderr)
            self.assertTrue(os.path.isdir(a_folder))

    def make_temp_listing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        disasm = Path(tmp.name) / "disasm"
        disasm.mkdir()
        listing = disasm / "cs_sample.s"
        listing.write_bytes(b"\ts_endpgm\n")
        return listing

    def assert_json_refuses_listing(self, listing, output):
        original = listing.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = audit_mod.main([str(listing.parent), "--json", str(output)])
        self.assertEqual((rc, listing.read_bytes() == original), (2, True))
        self.assertIn("must not overwrite an input listing", err.getvalue())

    def test_json_cannot_overwrite_a_listing(self):
        listing = self.make_temp_listing()
        self.assert_json_refuses_listing(listing, listing)

    def test_json_cannot_overwrite_a_listing_via_equivalent_path(self):
        listing = self.make_temp_listing()
        disasm = listing.parent
        self.assert_json_refuses_listing(listing, disasm / ".." / disasm.name / listing.name)

    def test_json_cannot_overwrite_a_listing_via_hard_link(self):
        listing = self.make_temp_listing()
        output = listing.parent.parent / "alias.json"
        try:
            os.link(listing, output)
        except (AttributeError, NotImplementedError, OSError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assert_json_refuses_listing(listing, output)

    def test_json_can_replace_a_separate_report(self):
        listing = self.make_temp_listing()
        original = listing.read_bytes()
        output = listing.with_name("report.json")
        output.write_bytes(b"{}\n")

        self.assertEqual(audit_mod.main([str(listing.parent), "--json", str(output)]), 0)
        self.assertEqual(json.loads(output.read_bytes())["cs"]["count"], 1)
        self.assertEqual(listing.read_bytes(), original)

    def make_metadata_inputs(self, custom=False):
        disasm = self.make_temp_listing().parent
        root = disasm.parent
        fssm = root / ("custom-fssm" if custom else "fssm")
        dxbc = root / "custom-dxbc" if custom else root
        fssm.mkdir()
        dxbc.mkdir(exist_ok=True)
        (root / "index.json").write_bytes(b'{"shader_sample":{"stage":"ms"}}\n')
        (disasm / "index.json").write_bytes(b'{"shader_sample":{"stage":"cs"}}\n')
        (disasm / "shader_sample.s").write_bytes(b"\ts_endpgm\n")
        for stem in ("cs_sample", "shader_sample"):
            (fssm / (stem + ".fssm")).write_bytes(bytes(0x148))
            (dxbc / (stem + ".dxbc")).write_bytes(make_dxbc([make_ms_psv0()]))
        return disasm, fssm, dxbc

    def assert_json_refuses_input(self, disasm, source, output, *options):
        originals = {path: path.read_bytes() for path in disasm.parent.rglob("*") if path.is_file()}
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = audit_mod.main([str(disasm), *options, "--json", str(output)])
        changed = [str(path.relative_to(disasm.parent)) for path, data in originals.items()
                   if path.read_bytes() != data]
        self.assertEqual((rc, changed), (2, []))
        self.assertEqual(len(err.getvalue().splitlines()), 1)
        self.assertIn("--json", err.getvalue())
        self.assertIn("must not overwrite an input", err.getvalue())
        self.assertIn(source.name, err.getvalue())
        self.assertIn(output.name, err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_json_refuses_both_stage_indexes(self):
        for location in ("parent", "disasm"):
            with self.subTest(location=location):
                disasm, _, _ = self.make_metadata_inputs()
                source = (disasm.parent if location == "parent" else disasm) / "index.json"
                self.assert_json_refuses_input(disasm, source, source)

    def test_json_refuses_selected_metadata_in_default_and_custom_directories(self):
        for custom in (False, True):
            for name in ("cs_sample.fssm", "shader_sample.fssm", "shader_sample.dxbc"):
                with self.subTest(custom=custom, name=name):
                    disasm, fssm, dxbc = self.make_metadata_inputs(custom)
                    source = (fssm if name.endswith(".fssm") else dxbc) / name
                    options = ("--fssm", str(fssm), "--dxbc", str(dxbc)) if custom else ()
                    output = source.parent / ".." / source.parent.name / name if custom else source
                    self.assert_json_refuses_input(disasm, source, output, *options)

    def test_json_refuses_metadata_via_hard_link(self):
        disasm, _, dxbc = self.make_metadata_inputs()
        source = dxbc / "shader_sample.dxbc"
        output = disasm.parent / "alias.json"
        try:
            os.link(source, output)
        except (AttributeError, NotImplementedError, OSError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assert_json_refuses_input(disasm, source, output)

    def test_json_allows_metadata_not_selected_by_stage_or_limit(self):
        for name in ("cs_sample.dxbc", "shader_sample.fssm", "shader_sample.dxbc"):
            with self.subTest(name=name):
                disasm, fssm, dxbc = self.make_metadata_inputs()
                output = (fssm if name.endswith(".fssm") else dxbc) / name
                options = [] if name == "cs_sample.dxbc" else ["--limit", "1"]
                self.assertEqual(audit_mod.main([str(disasm), *options, "--json", str(output)]), 0)
                report = json.loads(output.read_bytes())
                self.assertEqual(report["cs"]["count"], 1)
                self.assertEqual(report["ms"]["count"], int(not options))

    def test_json_refuses_a_listing_even_when_limit_excludes_it(self):
        listing = self.make_temp_listing()
        self.assert_json_refuses_input(listing.parent, listing, listing, "--limit", "0")

    def test_synthetic_cs_ms_gs_audit_and_s3_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.mkdir(disasm)
            os.mkdir(fssm)

            # Synthetic CS: RSRC2 = 0x8088 (user=4, tgid=100, tg_size=0, tidig=0, scratch=0, lds=512B)
            with open(os.path.join(disasm, "cs_sample.s"), "w", encoding="utf-8") as f:
                f.write("; cs_sample\n"
                        "\ts_lshl_b32 s6, s4, 6 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_add_nc_u32_e32 v1, s6, v0 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(os.path.join(fssm, "cs_sample.fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))

            # Synthetic MS: reads s3 via s_bfe_u32 D, s3, 0x40018 and reads v5
            with open(os.path.join(disasm, "ms_sample.s"), "w", encoding="utf-8") as f:
                f.write("; ms_sample\n"
                        "\ts_bfe_u32 s4, s3, 0x40018 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_readfirstlane_b32 s5, v5 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_mov_b64 exec, -1 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(os.path.join(tmp, "ms_sample.dxbc"), "wb") as f:
                f.write(make_dxbc([make_ms_psv0(128, 1, 1, 64, 64, 2)]))

            # Synthetic GS: reads v0, v1 and s3
            with open(os.path.join(disasm, "gs_sample.s"), "w", encoding="utf-8") as f:
                f.write("; gs_sample\n"
                        "\ts_bfe_u32 s4, s3, 0x80008 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_mov_b32_e32 v2, v0 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_mov_b32_e32 v3, v1 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")

            out_json = os.path.join(tmp, "report.json")
            self.assertEqual(audit_mod.main([disasm, "--json", out_json]), 0)
            with open(out_json, encoding="utf-8") as f:
                rep = json.load(f)

            self.assertEqual(rep["cs"]["count"], 1)
            self.assertEqual(rep["cs"]["rbw_sgpr_counts"], {"s4": 1})
            self.assertEqual(rep["cs"]["rbw_vgpr_counts"], {"v0": 1})
            self.assertIn("user=4 tgid=100 tg_size=0 tidig=0 scratch=0 trap=0 lds=512B", rep["cs"]["rsrc2_combos"])

            self.assertEqual(rep["ms"]["count"], 1)
            self.assertEqual(rep["ms"]["wave64_heuristic"], 1)
            self.assertEqual(rep["ms"]["rbw_sgpr_counts"], {"s3": 1})
            self.assertEqual(rep["ms"]["rbw_vgpr_counts"], {"v5": 1})
            self.assertEqual(rep["ms"]["s3_rbw_op_patterns"], {"s_bfe_u32 D, s3, 0x40018": 1})
            self.assertIn("stage=13 topo=2 threads=128x1x1 maxv=64 maxp=64 payload=0B shared=0B",
                          rep["ms"]["psv0_combos"])

            self.assertEqual(rep["gs"]["count"], 1)
            self.assertEqual(rep["gs"]["rbw_vgpr_counts"], {"v0": 1, "v1": 1})
            self.assertEqual(rep["gs"]["s3_rbw_op_patterns"], {"s_bfe_u32 D, s3, 0x80008": 1})

    def test_buffer_load_lds_reads_vaddr_and_writes_no_vgpr(self):
        # The lds modifier sends the load straight to LDS: args[0] is vaddr (a source), not a destination.
        insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\tbuffer_load_dword v0, s[0:3], s4 offen lds",
            "\tv_mov_b32 v7, v0",
        )]
        s_rbw, v_rbw, _patterns = audit_mod.rbw_registers(insns)
        self.assertIn(0, v_rbw)          # vaddr read before write, as the repro requires

    def test_dot_accumulate_ops_read_their_destination(self):
        insns = [audit_mod.shader_ir.parse_line(line) for line in (
            "\tv_dot2c_f32_f16 v2, v3, v4",
            "\tv_dot4c_i32_i8 v5, v3, v4",
        )]
        _s_rbw, v_rbw, _patterns = audit_mod.rbw_registers(insns)
        self.assertEqual(v_rbw, [2, 3, 4, 5])


def setUpModule():

    # The tools' main() prints its report; keep it out of the test log (tests that check it redirect again).
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    unittest.addModuleCleanup(quiet.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
