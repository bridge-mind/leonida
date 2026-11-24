"""Exercise the assets stage, existing C# catalogs and downstream texture lookup."""
import contextlib
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
import make_game
import ttpf_assets
import ttpf_mips
import world_build
from test_calibrated_texture import synthetic_profile
from test_ttpf_mips import build_bc1_bcpack_pool


class TextureAssets(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.game = self.root / "game"
        self.work = self.root / "work"
        audio = self.work / "assets/audio"
        audio.mkdir(parents=True)
        for marker in ("audio_manifest.json", "ogg.done", "xma.done"):
            (audio / marker).touch()
        self.pool = self.game / "data/pool.ttpf"
        self.pool.parent.mkdir(parents=True)
        self.folder = self.work / "assets/ttpf/pool"
        self.folder.mkdir(parents=True)
        (self.folder / make_game.DECODER_RECORD).write_text(make_game.decoder_digest(), encoding="utf-8")
        self.catalog = self.folder / "manifest.json"
        self.row = dict(Index=0, AssetKey="CAFEBABE12345678", SourceMips=5, BaseMipLevel=0,
                        Mips=0, File=None, DxgiFormat=71, Limitation="BCPack")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(make_game, "MODEL_ARCHIVES", ()))
        self.stack.enter_context(mock.patch.object(make_game, "log"))

    def build(self):
        make_game.assets(str(self.game), str(self.work), False, 1)

    def legacy(self, **changes):
        self.catalog.write_text(json.dumps({"Count": 1, "Textures": [self.row | changes]}), encoding="utf-8")

    def row_after_export(self):
        return json.loads(self.catalog.read_bytes())["Textures"][0]

    def test_partial_upgrade_failure_then_resume(self):
        _, linear, _ = build_bc1_bcpack_pool(self.pool)
        self.legacy(File="partial.dds", BaseMipLevel=2, Mips=3, Limitation=None)
        partial = self.folder / "partial.dds"
        partial.write_bytes(b"previous mip chain")
        prior = self.catalog.read_bytes()
        replace = os.replace

        def fail_catalog(src, dest):
            if dest == self.catalog:
                raise OSError("simulated write failure")
            replace(src, dest)

        with mock.patch.object(os, "replace", side_effect=fail_catalog), self.assertRaises(OSError):
            self.build()
        self.assertEqual(prior, self.catalog.read_bytes())
        self.assertEqual(partial.read_bytes(), b"previous mip chain")
        self.build()
        row = self.row_after_export()
        path = self.folder / row["File"]
        self.assertEqual((row["Mips"], row["BaseMipLevel"]), (5, 0))
        self.assertEqual(path.read_bytes()[148:], b"".join(linear.values()))
        self.assertEqual(world_build.index_ttpf_textures(self.work / "assets")[row["AssetKey"]], str(path))
        # The lower-mip file the row named is gone once the manifest names the full chain: it was kept through
        # the failed run above, and it is not stored a second time after the successful one.
        self.assertFalse(partial.exists())
        # decoder.sha256 is the decoder record assets() keeps in an exported texture folder (BUILD-1228)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir() if p.name != "decoder.sha256"),
                         sorted([row["File"], "manifest.json"]))
        with mock.patch.object(ttpf_assets.ttpf, "Pool", side_effect=AssertionError("unexpected reopen")):
            self.build()

    def upgrade(self, other_rows=(), file="partial.dds"):
        """A pool whose one BCPack texture has a lower-mip export in `file`; returns that file's path."""
        build_bc1_bcpack_pool(self.pool)
        first = self.row | dict(File=file, BaseMipLevel=2, Mips=3, Limitation=None)
        self.catalog.write_text(json.dumps({"Count": 1 + len(other_rows), "Textures": [first, *other_rows]}),
                                encoding="utf-8")
        old = self.folder / file
        old.write_bytes(b"previous mip chain")
        return old

    def test_superseded_file_another_row_names_is_kept(self):
        # Not a block-compressed format, so the pass leaves this row alone; it names the same file.
        other = dict(Index=1, AssetKey="0123456789ABCDEF", SourceMips=1, BaseMipLevel=0, Mips=1,
                     File="partial.dds", DxgiFormat=28, Limitation=None)
        old = self.upgrade(other_rows=[other])
        self.build()
        rows = json.loads(self.catalog.read_bytes())["Textures"]
        self.assertTrue(rows[0]["File"].startswith("000000_bcpack_"))
        self.assertEqual(rows[1]["File"], "partial.dds")
        self.assertEqual(old.read_bytes(), b"previous mip chain")

    def test_superseded_file_that_cannot_be_removed_does_not_fail_the_export(self):
        old = self.upgrade()
        unlink = Path.unlink

        def in_use(path, *args, **kwargs):
            if path.name == "partial.dds":
                raise PermissionError(13, "The process cannot access the file because it is being used")
            return unlink(path, *args, **kwargs)

        stderr = io.StringIO()
        with mock.patch.object(Path, "unlink", in_use), contextlib.redirect_stderr(stderr):
            self.build()
        self.assertTrue(self.row_after_export()["File"].startswith("000000_bcpack_"))
        self.assertTrue(old.exists())
        # Not hidden: a folder that stays double-sized has its reason on screen.
        self.assertIn("pool: 1 replaced DDS file(s) could not be removed", stderr.getvalue())
        # And not forever: the manifest lists it, so the next build removes it once nothing holds it, although
        # that build finds the pass stamped and writes nothing else.
        self.assertEqual(json.loads(self.catalog.read_bytes())[ttpf_assets.PENDING], ["partial.dds"])
        with mock.patch.object(ttpf_assets.ttpf, "Pool", side_effect=AssertionError("unexpected reopen")):
            self.build()
        self.assertFalse(old.exists())
        self.assertNotIn(ttpf_assets.PENDING, json.loads(self.catalog.read_bytes()))
        self.build()                                      # nothing left to do: no further write, no error
        self.assertTrue(self.row_after_export()["File"].startswith("000000_bcpack_"))

    def test_a_run_stopped_after_the_switch_is_finished_by_the_next(self):
        # The manifest that switches the rows already lists the file to remove, so a build stopped between that
        # write and the removal leaves nothing behind for good.
        old = self.upgrade()
        with mock.patch.object(ttpf_assets, "finish_replaced"):
            self.build()
        self.assertTrue(old.exists())
        self.assertEqual(json.loads(self.catalog.read_bytes())[ttpf_assets.PENDING], ["partial.dds"])
        self.build()
        self.assertFalse(old.exists())
        self.assertNotIn(ttpf_assets.PENDING, json.loads(self.catalog.read_bytes()))

    def complete_with_row_file(self, file):
        """Run the pass on a manifest whose row names `file`; returns the folder's file names afterwards."""
        self.upgrade()
        manifest = json.loads(self.catalog.read_bytes())
        manifest["Textures"][0]["File"] = file
        self.assertEqual(ttpf_assets.complete_pool(self.pool, self.folder, manifest, ("v1-strict", None)), (1, 0))
        return sorted(p.name for p in self.folder.iterdir())

    def test_a_row_that_named_the_new_file_in_other_letter_case_keeps_it(self):
        # On Windows and macOS that name is the file the pass has just published; removing it removed the export.
        names = self.complete_with_row_file("000000_BCPACK_V1-STRICT.DDS")
        row = self.row_after_export()
        self.assertEqual(row["File"], "000000_bcpack_v1-strict.dds")
        self.assertTrue((self.folder / row["File"]).is_file(), names)
        self.assertGreater((self.folder / row["File"]).stat().st_size, 148)

    def test_a_row_that_named_the_manifest_does_not_remove_it(self):
        names = self.complete_with_row_file("manifest.json")
        self.assertIn("manifest.json", names)
        self.assertTrue(self.row_after_export()["File"].startswith("000000_bcpack_"))

    def test_only_dds_files_are_removed(self):
        (self.folder / "notes.txt").write_text("mine", encoding="utf-8")
        names = self.complete_with_row_file("notes.txt")
        self.assertIn("notes.txt", names)

    def test_removal_looks_at_each_file_once_not_once_per_pair(self):
        # 50 rows and 50 replaced files: 100 files to look at, not 2,500 pairs. A pool completes most of its
        # textures in one pass, so a per-pair check takes minutes to hours on a real one.
        rows = [dict(File=f"{i:06d}_bcpack_v1-strict.dds") for i in range(50)]
        replaced = {f"{i:06d}.dds" for i in range(50)}
        for name in [row["File"] for row in rows] + sorted(replaced):
            (self.folder / name).write_bytes(b"dds")
        stat, calls = os.stat, []

        def counted(path, *args, **kwargs):
            calls.append(path)
            return stat(path, *args, **kwargs)

        with mock.patch.object(ttpf_assets.os, "stat", counted), \
                mock.patch.object(ttpf_assets.os.path, "samefile", side_effect=AssertionError("per-pair check")):
            self.assertEqual(ttpf_assets.remove_replaced(self.folder, replaced, rows), [])
        self.assertLessEqual(len(calls), 100, len(calls))
        # The replaced files are gone and the rows' files stay.
        self.assertEqual(sorted(p.name for p in self.folder.iterdir() if p.name != "decoder.sha256"),
                         sorted(row["File"] for row in rows))

    def test_only_a_file_in_the_pool_folder_is_removed(self):
        # A row is trusted for a file name, not for a path: nothing outside the pool's folder is touched.
        outside = self.folder.parent / "outside.dds"
        outside.write_bytes(b"not this pool's")
        self.upgrade()
        manifest = json.loads(self.catalog.read_bytes())
        manifest["Textures"][0]["File"] = "../outside.dds"
        written, blocked = ttpf_assets.complete_pool(self.pool, self.folder, manifest, ("v1-strict", None))
        self.assertEqual((written, blocked), (1, 0))
        self.assertTrue(self.row_after_export()["File"].startswith("000000_bcpack_"))
        self.assertEqual(outside.read_bytes(), b"not this pool's")

    def test_profile_added_after_strict_rejection(self):
        build_bc1_bcpack_pool(self.pool, ref_dists=[250] + [0] * 1023)
        self.legacy()
        self.build()
        row = self.row_after_export()
        self.assertIsNone(row["File"])
        self.assertIn("BCPackLimitation", row)
        try:
            from calibrated_texture import native
            native()
        except (ImportError, OSError) as exc:
            self.skipTest(str(exc))
        profile = self.work / "bcpack-calibration.json"
        config = synthetic_profile()
        profile.write_text(json.dumps(config), encoding="utf-8")
        self.build()
        row = self.row_after_export()
        original = self.folder / row["File"]
        self.assertEqual(original.read_bytes()[:4], b"DDS ")
        self.assertNotIn("BCPackLimitation", row)
        config["bc1"]["selectors"][0] = [mask ^ 1 for mask in config["bc1"]["selectors"][0]]
        profile.write_text(json.dumps(config), encoding="utf-8")
        self.build()
        updated = self.folder / self.row_after_export()["File"]
        self.assertNotEqual(updated, original)
        self.assertNotEqual(updated.read_bytes(), original.read_bytes())

    @unittest.skipUnless(os.environ.get("RPF9DUMP_DLL"), "requires the built rpf9dump")
    def test_new_base_and_pack_pools(self):
        dll = Path(os.environ["RPF9DUMP_DLL"]).resolve()
        exe = dll.with_suffix(".exe" if os.name == "nt" else "")
        pack = self.game / "pack/story/data/pool.ttpf"
        pack.parent.mkdir(parents=True)
        _, linear, _ = build_bc1_bcpack_pool(self.pool)
        pack.write_bytes(self.pool.read_bytes())
        with mock.patch.object(make_game, "rpf9dump", return_value=str(exe)):
            self.build()
        for folder in (self.folder, self.work / "assets/ttpf_pack/story/pool"):
            row = json.loads((folder / "manifest.json").read_bytes())["Textures"][0]
            self.assertEqual(row["Mips"], 5)
            self.assertEqual((folder / row["File"]).read_bytes()[148:], b"".join(linear.values()))

    def test_complete_pool_handles_unmapped_range_without_crashing(self):
        # If pool.codecs() raises ValueError (e.g., an unmapped hole inside a texture's span),
        # complete_pool must record BCPackLimitation on that row instead of aborting the pool.
        build_bc1_bcpack_pool(self.pool)
        self.legacy()
        with mock.patch.object(ttpf_assets.ttpf.Pool, "codecs",
                               side_effect=ValueError("texture range is not completely mapped")):
            self.build()
        row = self.row_after_export()
        self.assertIsNone(row["File"])
        self.assertEqual(row["BCPackLimitation"], "texture range is not completely mapped")


BC_FORMAT = next(iter(ttpf_mips.BC_FORMATS))
NOT_BC = "R8G8B8A8_UNORM"


def synthetic_row(**changes):
    out = {"DxgiFormat": BC_FORMAT, "File": None, "BaseMipLevel": 0, "Mips": 1, "SourceMips": 1}
    out.update(changes)
    return out


class PublishWriter(unittest.TestCase):
    """`publish` stages in the destination folder and must never leave the staging file behind."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def names(self):
        return sorted(p.name for p in self.dir.iterdir())

    def test_writes_the_bytes_and_leaves_no_staging_file(self):
        ttpf_assets.publish(self.dir / "manifest.json", b'{"Textures": []}')
        self.assertEqual((self.dir / "manifest.json").read_bytes(), b'{"Textures": []}')
        self.assertEqual(self.names(), ["manifest.json"])

    def test_replaces_the_previous_contents(self):
        target = self.dir / "manifest.json"
        target.write_bytes(b"old contents that are longer")
        ttpf_assets.publish(target, b"new")
        self.assertEqual(target.read_bytes(), b"new")
        self.assertEqual(self.names(), ["manifest.json"])

    def test_failed_replace_keeps_the_previous_bytes_and_no_temp(self):
        target = self.dir / "manifest.json"
        target.write_bytes(b"previous")
        with mock.patch.object(ttpf_assets.os, "replace", side_effect=OSError(5, "access denied")), \
                self.assertRaises(OSError):
            ttpf_assets.publish(target, b"replacement")
        self.assertEqual(target.read_bytes(), b"previous")
        self.assertEqual(self.names(), ["manifest.json"])

    @unittest.skipUnless(os.name == "posix", "POSIX file modes only")
    def test_preserves_the_destination_mode(self):
        target = self.dir / "manifest.json"
        target.write_bytes(b"previous")
        os.chmod(target, 0o640)
        ttpf_assets.publish(target, b"replacement")
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.names(), ["manifest.json"])


class CandidateRules(unittest.TestCase):
    def test_other_formats_are_never_candidates(self):
        self.assertFalse(ttpf_assets.candidate(synthetic_row(DxgiFormat=NOT_BC, File=None), None))

    def test_a_strict_decoder_claim_is_a_candidate_without_a_profile(self):
        self.assertTrue(ttpf_assets.candidate(synthetic_row(BCPackDecoder="bc1-strict"), None))

    def test_a_calibrated_decoder_claim_needs_a_profile(self):
        self.assertFalse(ttpf_assets.candidate(synthetic_row(BCPackDecoder="bc1-calibrated"), None))
        self.assertTrue(ttpf_assets.candidate(synthetic_row(BCPackDecoder="bc1-calibrated"), {"version": 1}))

    def test_a_missing_file_is_a_candidate(self):
        self.assertTrue(ttpf_assets.candidate(synthetic_row(File=None), None))

    def test_a_lower_mip_start_or_a_partial_chain_is_a_candidate(self):
        self.assertTrue(ttpf_assets.candidate(synthetic_row(File="000001.dds", BaseMipLevel=2), None))
        self.assertTrue(ttpf_assets.candidate(synthetic_row(File="000001.dds", Mips=2, SourceMips=5), None))

    def test_a_complete_chain_with_a_file_is_not_a_candidate(self):
        self.assertFalse(ttpf_assets.candidate(synthetic_row(File="000001.dds", Mips=5, SourceMips=5), None))


class ProfileRules(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)

    def write(self, text):
        (self.work / "bcpack-calibration.json").write_text(text, encoding="utf-8")

    def test_missing_calibration_is_the_strict_profile(self):
        self.assertEqual(ttpf_assets.load_profile(self.work), (ttpf_assets.strict_stamp(), None))

    def test_a_decoder_source_edit_changes_both_stamps_and_the_pass_is_done_again(self):
        # A pool stamped before a bcpack.py fix kept its limited rows for good: the strict stamp was a constant.
        tools = self.work / "tools"
        tools.mkdir()
        for name in ttpf_assets.DECODER_SOURCES:
            (tools / name).write_text(f"# {name}\n", encoding="utf-8")
        self.write(json.dumps({"version": 1}))
        with mock.patch.object(ttpf_assets, "__file__", str(tools / "ttpf_assets.py")), \
                mock.patch.dict(sys.modules, {"calibrated_texture": mock.Mock()}):
            old = ttpf_assets.strict_stamp(), ttpf_assets.load_profile(self.work)[0]
            with open(tools / "bcpack.py", "a", encoding="utf-8") as f:
                f.write("# fixed\n")
            new = ttpf_assets.strict_stamp(), ttpf_assets.load_profile(self.work)[0]
            self.assertTrue(new[0].endswith("-strict"))
            self.assertEqual(len(f"000000_bcpack_{new[0][:19]}.dds"), 37)
            (tools / "tex_headers.py").unlink()
            with self.assertRaises(FileNotFoundError):
                ttpf_assets.strict_stamp()
        self.assertNotEqual(old[0], new[0])
        self.assertNotEqual(old[1], new[1])
        manifest = {"BCPackPass": old[0], "Textures": [synthetic_row(BCPackDecoder=old[0])]}
        with mock.patch.object(ttpf_assets.ttpf, "Pool", side_effect=RuntimeError("opened")):
            with self.assertRaisesRegex(RuntimeError, "opened"):
                ttpf_assets.complete_pool("missing.ttpf", self.work, manifest, (new[0], None))

    def test_another_calibration_version_is_an_error(self):
        self.write(json.dumps({"version": 2}))
        with self.assertRaises(ValueError):
            ttpf_assets.load_profile(self.work)

    def test_a_non_object_calibration_is_an_error(self):
        self.write("[]")
        with self.assertRaises(ValueError):
            ttpf_assets.load_profile(self.work)

    def test_malformed_json_is_an_error(self):
        self.write("{")
        with self.assertRaises(ValueError):
            ttpf_assets.load_profile(self.work)

    def test_malformed_json_names_the_profile(self):
        self.write("{")
        with self.assertRaises(ValueError) as caught:
            ttpf_assets.load_profile(self.work)
        self.assertIn("bcpack-calibration.json", str(caught.exception))


class CompletePoolEarlyReturn(unittest.TestCase):
    def test_a_matching_pass_stamp_returns_before_opening_the_pool(self):
        manifest = {"BCPackPass": "v1-deadbeef", "Textures": [synthetic_row()]}
        with mock.patch.object(ttpf_assets, "publish") as publish, \
                mock.patch.object(ttpf_assets.ttpf, "Pool") as pool:
            self.assertEqual(ttpf_assets.complete_pool("missing.rpf", "/tmp/out", manifest, ("v1-deadbeef", None)),
                             (0, 0))
        publish.assert_not_called()
        pool.assert_not_called()

    def test_no_candidates_returns_before_opening_the_pool(self):
        manifest = {"BCPackPass": "v1-new",
                    "Textures": [synthetic_row(File="000001.dds", Mips=5, SourceMips=5)]}
        with mock.patch.object(ttpf_assets.ttpf, "Pool") as pool:
            self.assertEqual(ttpf_assets.complete_pool("missing.rpf", "/tmp/out", manifest, ("v1-other", None)),
                             (0, 0))
        pool.assert_not_called()


    def test_a_pool_hashes_the_decoder_sources_once_for_all_its_strict_fallback_rows(self):
        # With a profile, each row the calibrated decoder can't take falls back to strict decoding; hashing the
        # sources per row read them 32 times over for 32 rows. A new pass computes the stamp again.
        pool = mock.Mock(chunks=[(0,)], virtual_size=64)
        pool.descriptor.return_value = {"base": 0, "dimension": 1, "format": 71}
        pool.span.return_value = 8
        pool.codecs.return_value = ttpf_assets.bcpack.SUPPORTED_REBUILD_CODECS
        calibrated = mock.Mock()
        calibrated.LocalDecoder.return_value.formats = set()
        with tempfile.TemporaryDirectory() as out, ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, calibrated_texture=calibrated))
            stack.enter_context(mock.patch.object(ttpf_assets.ttpf, "Pool", return_value=pool))
            stack.enter_context(mock.patch.object(ttpf_assets.ttpf, "analyze", side_effect=ValueError("limited")))
            stamps = stack.enter_context(mock.patch.object(ttpf_assets, "strict_stamp",
                                                           wraps=ttpf_assets.strict_stamp))
            for passes in (1, 2):
                manifest = {"Textures": [synthetic_row(Index=i) for i in range(32)]}
                self.assertEqual(ttpf_assets.complete_pool("synthetic", out, manifest, ("v1-profile", {})), (0, 32))
                self.assertEqual(stamps.call_count, passes)


if __name__ == "__main__":
    unittest.main()
