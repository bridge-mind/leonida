"""tools/wave64_graphics_scan.py: classification, counting and privacy; no game data.

    python -I tests/test_wave64_graphics_scan.py -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fssm_header  # noqa: E402
import wave64_graphics_scan as scan  # noqa: E402


def fixture(name, lines):
    """A synthetic disassembly: the real files start with shader_disasm's '; name' header, then tab-prefixed asm."""
    body = "\n".join("\t" + ln if not ln.startswith(";") else ln for ln in lines)
    return name, "; %s  code 0x0..0x10 (header), gfx1013, 0 invalid words\n%s\n" % (name[:-2], body)


PS_W64_SENSITIVE = fixture("ps_w64a.s", [
    "v_mbcnt_hi_u32_b32 v1, v2, v3",
    "s_cbranch_execz 42",
    "v_add_f32_e32 v0, v1, v2",
])
PS_W64_EXEC_ONLY = fixture("ps_w64movexec.s", [
    # the review's case: no WAVE64_RE marker, but a _b64 op with an exec operand classifies wave64,
    # exactly as rdna2spirv.py's g.wave64 does
    "s_mov_b64 exec, s[0:1]",
    "s_cbranch_execz 42",
])
PS_W64_CLEAN = fixture("ps_w64clean.s", [
    # _saveexec_b64 marks the shader wave64, but mask arithmetic alone is per-lane and exact
    "s_and_saveexec_b64 s[0:1], s[2:3], s[4:5]",
    "v_add_f32_e32 v0, v1, v2",
])
PS_W64_BCNT_SPLIT = fixture("ps_w64bcnt.s", [
    "s_bcnt1_i32_b64 s0, exec",      # wave-mask path outright
    "s_bcnt1_i32_b64 s1, s[2:3]",    # register source: candidate, provenance unresolved
])
PS_W64_COMMENT = fixture("ps_w64comment.s", [
    "; v_readlane_b32 mentioned in a comment must not count",
    "s_wqm_b64 s[0:1], exec",        # WAVE64_RE marker only
    "v_add_f32_e32 v0, v1, v2",
])
PS_W32 = fixture("ps_w32a.s", ["v_add_f32_e32 v0, v1, v2"])
VS_W64 = fixture("vs_w64b.s", [
    "s_or_saveexec_b64 s[0:1], s[2:3], exec",
    "v_readlane_b32 v1, v2, s4",
    "v_readlane_b32 v3, v4, s5",
])
GS_W64 = fixture("gs_w64d.s", ["s_ff1_i32_b64 s0, s[2:3]"])
CS_W64 = fixture("cs_w64c.s", ["v_mbcnt_hi_u32_b32 v1, v2, v3"])  # compute: never counted
MS_W64 = fixture("ms_w64e.s", ["s_cbranch_execz 7"])              # mesh: wave64 regardless, never counted
VS_UNKNOWN = fixture("xx_w64f.s", ["v_readfirstlane_b32 s1, v1"])  # stage '?': skipped


class ScanTests(unittest.TestCase):
    def scan(self, *fixtures):
        with tempfile.TemporaryDirectory() as directory:
            for name, text in fixtures:
                with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
                    f.write(text)
            return scan.scan_directory(directory)

    def test_counts_wave64_graphics_only(self):
        report = self.scan(PS_W64_SENSITIVE, PS_W64_EXEC_ONLY, PS_W64_CLEAN, PS_W32, VS_W64, GS_W64,
                           CS_W64, MS_W64, VS_UNKNOWN)
        ps, vs, gs = report["per_stage"]["ps"], report["per_stage"]["vs"], report["per_stage"]["gs"]
        self.assertEqual(ps["shaders"], 4)
        self.assertEqual(ps["wave64"], 3)
        self.assertEqual(ps["with_selected_op"], 2)
        self.assertEqual(vs["wave64"], 1)
        self.assertEqual(gs["wave64"], 1)
        self.assertEqual(report["other_stages"], {"shaders": 3, "wave64": 2})  # cs + ms wave64, xx unknown

    def test_b64_exec_operand_classifies_wave64_without_a_marker(self):
        report = self.scan(PS_W64_EXEC_ONLY)
        b = report["per_stage"]["ps"]
        self.assertEqual((b["wave64"], b["with_selected_op"]), (1, 1))
        self.assertEqual(b["occurrences"], {"s_cbranch_execz": 1})  # s_mov_b64 exec is the classifier, not a hit

    def test_fssm_header_distinguishes_identical_markerless_shaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disasm = root / "disasm"
            fssm = root / "fssm"
            disasm.mkdir()
            fssm.mkdir()
            for size, flags in ((32, fssm_header.FLAG_WAVE32), (64, 0)):
                name = f"ps_header{size}"
                _, listing = fixture(name + ".s", ["v_readfirstlane_b32 s4, v1", "s_endpgm"])
                (disasm / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")
                header = bytearray(fssm_header.WAVE_FLAGS_OFFSET + 4)
                header[:4] = fssm_header.MAGIC
                struct.pack_into("<I", header, fssm_header.WAVE_FLAGS_OFFSET, flags)
                (fssm / (name + ".fssm")).write_bytes(header)

            report = scan.scan_directory(disasm)
            ps = report["per_stage"]["ps"]
            self.assertEqual((ps["shaders"], ps["wave64"], ps["with_selected_op"]), (2, 1, 1))
            self.assertEqual(ps["occurrences"], {"v_readfirstlane_b32": 1})

    def test_mask_word_source_split(self):
        report = self.scan(PS_W64_BCNT_SPLIT, GS_W64)
        ps = report["per_stage"]["ps"]
        self.assertEqual(ps["occurrences"], {"s_bcnt1_i32_b64": 2})
        self.assertEqual(ps["mask_word_source"]["exec_vcc"], {"s_bcnt1_i32_b64": 1})
        self.assertEqual(ps["mask_word_source"]["other"], {"s_bcnt1_i32_b64": 1})
        self.assertEqual(report["per_stage"]["gs"]["mask_word_source"]["other"], {"s_ff1_i32_b64": 1})

    def test_occurrence_counts_per_op(self):
        report = self.scan(PS_W64_SENSITIVE, VS_W64, GS_W64)
        self.assertEqual(report["per_stage"]["ps"]["occurrences"],
                         {"v_mbcnt_hi_u32_b32": 1, "s_cbranch_execz": 1})
        self.assertEqual(report["per_stage"]["vs"]["occurrences"], {"v_readlane_b32": 2})
        self.assertEqual(report["per_stage"]["gs"]["occurrences"], {"s_ff1_i32_b64": 1})

    def test_comments_and_operand_text_never_count(self):
        report = self.scan(PS_W64_COMMENT)
        b = report["per_stage"]["ps"]
        self.assertEqual((b["wave64"], b["with_selected_op"], b["occurrences"]), (1, 0, {}))

    def test_clean_wave64_shader_has_no_selected_op(self):
        report = self.scan(PS_W64_CLEAN)
        b = report["per_stage"]["ps"]
        self.assertEqual((b["wave64"], b["with_selected_op"], b["occurrences"]), (1, 0, {}))

    def test_stage_from_index_json_overrides_the_name(self):
        # stage_of reads <disasm>/../index.json (the carve index), like the translator does
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as disasm:
            with open(os.path.join(work, "index.json"), "w", encoding="utf-8") as f:
                json.dump([{"file": "ps_0001.dxbc", "stage": "ps"}], f)
            real = os.path.join(work, "dxbc", "disasm")
            os.makedirs(real)
            name, text = PS_W64_SENSITIVE
            with open(os.path.join(real, "ps_0001.s"), "w", encoding="utf-8") as f:
                f.write(text)  # name prefix says ps_ but the index is what counts
            report = scan.scan_directory(real)
            self.assertEqual(report["per_stage"]["ps"]["with_selected_op"], 1)

    def test_main_json_has_no_shader_names(self):
        # privacy rule: nothing from the game leaves the work folder, not even file names
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outdir:
            for name, text in (PS_W64_SENSITIVE, VS_W64):
                with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
                    f.write(text)
            out = os.path.join(outdir, "scan.json")
            report = scan.scan_directory(directory)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f)
            with open(out, encoding="utf-8") as f:
                blob = f.read()
            self.assertNotIn(".s", blob)
            self.assertNotIn("ps_w64a", blob)
            self.assertNotIn("comment", blob)
            self.assertIn("v_mbcnt_hi_u32_b32", blob)


class CliTests(unittest.TestCase):
    def test_scan_ignores_shader_named_directories(self):
        with tempfile.TemporaryDirectory() as src:
            os.mkdir(os.path.join(src, "cache.s"))
            name, text = PS_W64_SENSITIVE
            with open(os.path.join(src, name), "w", encoding="utf-8") as f:
                f.write(text)
            report = scan.scan_directory(src)
            self.assertEqual(report["per_stage"]["ps"]["shaders"], 1)

    def test_missing_source_directory_raises_error(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stderr(io.StringIO()):
                scan.main(["nonexistent_shaders_dir_xyz"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_out_is_directory_rejected(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([src, "--out", out_dir])
        self.assertNotEqual(cm.exception.code, 0)

    def test_out_parent_directory_missing_rejected(self):
        with tempfile.TemporaryDirectory() as src:
            missing_dir = os.path.join(src, "no_such_folder", "out.json")
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([src, "--out", missing_dir])
            self.assertNotEqual(cm.exception.code, 0)

    def test_out_same_as_source_rejected(self):
        with tempfile.TemporaryDirectory() as src:
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([src, "--out", src])
            self.assertNotEqual(cm.exception.code, 0)

    def test_out_overwriting_a_scanned_shader_rejected(self):
        # --out naming a *.s file the scan reads must not overwrite it; so must a hardlink alias that is
        # not itself a scanned listing (a .json name), which only the samefile alias check catches
        with tempfile.TemporaryDirectory() as src:
            name, text = PS_W64_SENSITIVE
            target = os.path.join(src, name)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([src, "--out", target])
            self.assertNotEqual(cm.exception.code, 0)
            alias = os.path.join(src, "same_file_alias.json")
            try:
                os.link(target, alias)  # hardlink: a different path for the same underlying scanned file
            except OSError:
                self.skipTest("hard links not supported on this filesystem")
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([src, "--out", alias])
            self.assertNotEqual(cm.exception.code, 0)
            with open(target, encoding="utf-8") as f:
                self.assertEqual(f.read(), text)  # the rejected run left the input bytes untouched

    def test_out_overwriting_the_carve_index_rejected(self):
        # stage_of reads <disasm>/../index.json; --out must not overwrite it (or an alias of it)
        with tempfile.TemporaryDirectory() as work:
            disasm = os.path.join(work, "dxbc", "disasm")
            os.makedirs(disasm)
            index = os.path.join(work, "dxbc", "index.json")
            with open(index, "w", encoding="utf-8") as f:
                json.dump([{"file": "ps_0001.dxbc", "stage": "ps"}], f)
            name, text = PS_W64_SENSITIVE
            with open(os.path.join(disasm, "ps_0001.s"), "w", encoding="utf-8") as f:
                f.write(text)
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([disasm, "--out", index])
            self.assertNotEqual(cm.exception.code, 0)
            alias = os.path.join(work, "same_index_alias.json")
            try:
                os.link(index, alias)  # an alias of the index, not of a scanned .s listing
            except OSError:
                self.skipTest("hard links not supported on this filesystem")
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stderr(io.StringIO()):
                    scan.main([disasm, "--out", alias])
            self.assertNotEqual(cm.exception.code, 0)

    def test_out_overwriting_a_wave_size_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disasm = root / "disasm"
            headers = root / "fssm"
            disasm.mkdir()
            headers.mkdir()
            name, listing = PS_W64_SENSITIVE
            (disasm / name).write_text(listing, encoding="utf-8", newline="\n")
            header = headers / (Path(name).stem + ".fssm")
            data = fssm_header.MAGIC + bytes(fssm_header.WAVE_FLAGS_OFFSET)
            header.write_bytes(data)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as cm:
                    scan.main([str(disasm), "--out", str(header)])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("must not overwrite", err.getvalue())
            self.assertEqual(header.read_bytes(), data)

    def test_uses_shared_same_report_path_and_catches_folding_volume_case_alias(self):
        import report_paths
        from unittest import mock
        self.assertIs(scan.same_report_path, report_paths.same_report_path)
        with tempfile.TemporaryDirectory() as work:
            disasm = os.path.join(work, "dxbc", "disasm")
            os.makedirs(disasm)
            index = os.path.join(work, "dxbc", "index.json")
            with open(index, "w", encoding="utf-8") as f:
                json.dump([{"file": "ps_0001.dxbc", "stage": "ps"}], f)
            name, text = PS_W64_SENSITIVE
            with open(os.path.join(disasm, "ps_0001.s"), "w", encoding="utf-8") as f:
                f.write(text)
            folded_index = os.path.join(work, "dxbc", "INDEX.JSON")
            with (mock.patch.object(report_paths.os.path, "realpath", side_effect=os.path.abspath),
                  mock.patch.object(report_paths, "_resolved_key", side_effect=str),
                  mock.patch.object(report_paths.os.path, "samefile", side_effect=OSError("not same")),
                  mock.patch.object(report_paths, "_same_resolved_name", return_value=True) as probe,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as cm:
                    scan.main([disasm, "--out", folded_index])
                self.assertEqual(cm.exception.code, 2)
            self.assertGreaterEqual(probe.call_count, 1)

    def test_cli_execution_with_out_json(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as outdir:
            for name, text in (PS_W64_SENSITIVE, VS_W64):
                with open(os.path.join(src, name), "w", encoding="utf-8") as f:
                    f.write(text)
            out = os.path.join(outdir, "report.json")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = scan.main([src, "--out", out])
            self.assertEqual(code, 0)
            self.assertIn("ps: 1 shaders, 1 wave64", stdout.getvalue())
            self.assertIn("wrote " + out, stdout.getvalue())
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("per_stage", saved)
            self.assertIn("other_stages", saved)
            self.assertEqual(saved["per_stage"]["ps"]["shaders"], 1)
            self.assertEqual(saved["per_stage"]["vs"]["shaders"], 1)
            with open(out, encoding="utf-8") as f:
                blob = f.read()
            self.assertNotIn(".s", blob)
            self.assertNotIn("ps_w64a", blob)
            self.assertIn("v_mbcnt_hi_u32_b32", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
