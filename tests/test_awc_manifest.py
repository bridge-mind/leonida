"""Tests for tools/awc_manifest.py, built from synthetic AWC data only (no game files).

    python -I tests/test_awc_manifest.py
"""
import contextlib, csv, io, json, os, struct, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "awc_manifest.py")
sys.path.insert(0, os.path.dirname(TOOL))
import awc_manifest  # noqa: E402
import awc_names  # noqa: E402

HEADER = ["bank", "stream", "codec", "codec_name", "channels", "sample_rate", "declared_samples",
          "duration_s", "streamed_music", "data_offset", "data_size", "bank_size", "notes"]


def fmt(samples, rate, codec, channels):
    block = bytearray(32)
    struct.pack_into("<I", block, 0, samples)
    struct.pack_into("<H", block, 8, rate)
    block[28], block[29] = codec, channels
    return bytes(block)


def music_blob(ids, samples, rate, toc=0xF8):
    """(descriptor, data) for one block of Opus channels. toc is the packet TOC byte."""
    chunk = 2048 * (1 + len(ids))
    desc = bytearray(12 + 16 * len(ids))
    struct.pack_into("<III", desc, 0, 1, chunk, len(ids))
    for i, cid in enumerate(ids):
        base = 12 + 16 * i
        struct.pack_into("<II", desc, base, cid, samples)
        struct.pack_into("<H", desc, base + 10, rate)
        desc[base + 12] = 0x0B
    data = bytearray(chunk)                          # aligned record table, then channel blocks
    for i in range(len(ids)):
        struct.pack_into("<I", data, 24 * i + 4, 1)   # subblock count; skip stays 0
        struct.pack_into("<H", data, 2048 + 2048 * i, 1)
        data[2050 + 2048 * i] = toc                   # CELT 20 ms, one frame (stereo when bit 2 is set)
    return bytes(desc), bytes(data)


def build_bank(entries):
    """entries: [(id, [(tag type, blob), ...])]; fmt/data blobs follow the header."""
    ntags = sum(len(tags) for _, tags in entries)
    header = 16 + len(entries) * 4 + ntags * 8
    body, placed = bytearray(), []
    for sid, tags in entries:
        recs = []
        for typ, blob in tags:
            recs.append((typ, header + len(body), len(blob)))
            body += blob
        placed.append((sid, recs))
    out = bytearray(b"ADAT" + struct.pack("<III", 0, len(entries), header))
    out += b"".join(struct.pack("<I", (sid & 0x1FFFFFFF) | (len(recs) << 29))
                    for sid, recs in placed)
    out += b"".join(struct.pack("<Q", (typ << 56) | (size << 28) | at)
                    for _, recs in placed for typ, at, size in recs)
    if len(out) != header:
        raise AssertionError("bad synthetic header")
    return bytes(out + body)


def entries(with_music=True):
    opus = bytearray(2048)
    struct.pack_into("<H", opus, 0, 1)
    opus[2] = 0xF8
    out = [(0x10, [(0x76, fmt(4800, 48000, 0x00, 1)), (0x55, bytes(9600))]),
           (0x20, [(0x76, fmt(4410, 44100, 0x02, 2)), (0x55, bytes(35280))]),
           (0x30, [(0x76, fmt(96000, 48000, 0x0B, 2)), (0x55, bytes(opus))])]
    if with_music:
        desc, data = music_blob([0x40, 0x41], 48000, 48000)
        out.append((0x50, [(0x48, desc), (0x55, data)]))
    return out


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_bank(self, name="alpha.awc", with_music=True):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(build_bank(entries(with_music)))
        return path

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True,
                              text=True, errors="replace", timeout=60)

    def test_csv_rows_codecs_and_bad_bank_skip(self):
        bank = self.write_bank()
        with open(os.path.join(self.tmp.name, "bad.awc"), "wb") as f:
            f.write(b"XATD" + b"\x00" * 64)
        p = self.run_tool(self.tmp.name)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("warning", p.stderr.lower())
        self.assertIn("bad.awc", p.stderr)
        self.assertIn("1 banks, 5 streams, 2 streamed-music channels", p.stderr)
        reader = csv.DictReader(io.StringIO(p.stdout))
        self.assertEqual(reader.fieldnames, HEADER)
        rows = {r["stream"]: r for r in reader}
        self.assertEqual(sorted(rows), ["00000010", "00000020", "00000030", "00000040", "00000041"])
        self.assertEqual({r["bank"] for r in rows.values()}, {"alpha.awc"})
        self.assertEqual(int(rows["00000010"]["bank_size"]), os.path.getsize(bank))
        for sid, codec, name in [("00000010", "0x00", "PCM16"),
                                 ("00000020", "0x02", "PCM float32"),
                                 ("00000030", "0x0B", "Opus")]:
            self.assertEqual((rows[sid]["codec"], rows[sid]["codec_name"]), (codec, name))
        for sid, channels, rate, samples, seconds in [("00000010", 1, 48000, 4800, 0.1),
                                                      ("00000020", 2, 44100, 4410, 0.1),
                                                      ("00000030", 2, 48000, 96000, 1.0),
                                                      ("00000040", 1, 48000, 48000, 1.0)]:
            row = rows[sid]
            self.assertEqual((row["channels"], row["sample_rate"], row["declared_samples"]),
                             (str(channels), str(rate), str(samples)))
            self.assertAlmostEqual(float(row["duration_s"]), seconds, places=6)
        self.assertEqual({s: rows[s]["streamed_music"] for s in ("00000010", "00000040", "00000041")},
                         {"00000010": "no", "00000040": "yes", "00000041": "yes"})
        self.assertEqual((int(rows["00000010"]["data_size"]), int(rows["00000040"]["data_size"])),
                         (9600, 6144))

    def test_plain_opus_duration_is_48khz_frames_per_channel(self):
        # The tag's rate is not the unit of the count: 4800 frames at a 32 kHz tag is 0.1 s
        # at 48 kHz, and a stereo count of 9600 is 4800 frames per channel.
        opus = bytearray(2048)
        struct.pack_into("<H", opus, 0, 1)
        opus[2] = 0xF8
        bank = build_bank([(1, [(0x76, fmt(4800, 32000, 0x0B, 1)), (0x55, bytes(opus))]),
                           (2, [(0x76, fmt(9600, 48000, 0x0B, 2)), (0x55, bytes(opus))])])
        path = os.path.join(self.tmp.name, "opus_rates.awc")
        with open(path, "wb") as f:
            f.write(bank)
        p = self.run_tool(path)
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = {r["stream"]: r for r in csv.DictReader(io.StringIO(p.stdout))}
        self.assertAlmostEqual(float(rows["00000001"]["duration_s"]), 0.1, places=6)
        self.assertAlmostEqual(float(rows["00000002"]["duration_s"]), 0.1, places=6)

    def test_streams_with_the_same_id_are_both_listed(self):
        path = os.path.join(self.tmp.name, "dup.awc")
        with open(path, "wb") as f:
            f.write(build_bank([(0x123, [(0x76, fmt(2, 48000, 0x00, 1)), (0x55, bytes(4))]),
                                (0x123, [(0x76, fmt(4, 48000, 0x00, 1)), (0x55, bytes(8))])]))
        p = self.run_tool(path)
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
        self.assertEqual([(r["stream"], r["declared_samples"]) for r in rows],
                         [("00000123", "2"), ("00000123", "4")])
        self.assertNotIn("00000123~2", rows[0]["notes"])
        self.assertIn("00000123~2", rows[1]["notes"])

    def test_inventory_can_name_plain_and_music_streams_without_banks(self):
        root = Path(self.tmp.name)
        names = ["zz_rootau_abcd_01", "zz_rootau_abcd_02"]
        plain_id, music_id = [awc_names.joaat(name) & awc_names.M29 for name in names]
        desc, audio = music_blob([music_id], 960, 48000)
        bank = root / "banks" / "nested" / "one.AWC"
        bank.parent.mkdir(parents=True)
        bank.write_bytes(build_bank([(plain_id, [(0x76, fmt(2, 48000, 0, 1)), (0x55, bytes(4))]),
                                     (0x50, [(0x48, desc), (0x55, audio)])]))
        inventory = root / "inventory.json"
        result = self.run_tool(root / "banks", "--out", inventory)
        self.assertEqual(result.returncode, 0, result.stderr)
        bank.unlink()

        words, out = root / "words.txt", root / "names.csv"
        words.write_text("\n".join(names), encoding="utf-8")
        result = subprocess.run([sys.executable, "-I", awc_names.__file__, inventory, "--words", words, "--out", out],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with out.open(encoding="utf-8", newline="") as fh:
            self.assertCountEqual([(row["bank"], row["name"]) for row in csv.DictReader(fh)],
                                  [("nested/one", name) for name in names])

    def test_json_out_is_inferred_from_extension(self):
        bank = self.write_bank(with_music=False)
        out = os.path.join(self.tmp.name, "nested", "manifest.json")
        p = self.run_tool(bank, "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertIn("1 banks, 3 streams, 0 streamed-music channels", p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual((doc["bank_count"], doc["stream_count"]), (1, 3))
        self.assertEqual(doc["streamed_music_channel_count"], 0)
        self.assertEqual([r["stream"] for r in doc["streams"]], ["00000010", "00000020", "00000030"])
        self.assertIs(doc["streams"][0]["streamed_music"], False)
        self.assertAlmostEqual(doc["streams"][0]["duration_s"], 0.1, places=6)

    def assert_input_refused(self, src, out, bank):
        with open(bank, "rb") as f:
            original = f.read()
        p = self.run_tool(src, "--out", out)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("would overwrite the input", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stdout, "")
        with open(bank, "rb") as f:
            self.assertEqual(f.read(), original)

    def test_report_cannot_replace_the_input_bank(self):
        bank = self.write_bank(with_music=False)
        self.assert_input_refused(bank, bank, bank)

    def test_report_cannot_replace_a_nested_bank(self):
        os.mkdir(os.path.join(self.tmp.name, "nested"))
        bank = self.write_bank(os.path.join("nested", "alpha.AWC"), with_music=False)
        self.assert_input_refused(self.tmp.name, bank, bank)

    def test_report_cannot_replace_a_normalized_input_path(self):
        bank = self.write_bank(with_music=False)
        os.mkdir(os.path.join(self.tmp.name, "nested"))
        out = os.path.join(self.tmp.name, "nested", "..", "alpha.awc")
        self.assert_input_refused(bank, out, bank)

    def test_report_cannot_replace_a_hard_link_to_the_input(self):
        bank = self.write_bank(with_music=False)
        out = os.path.join(self.tmp.name, "manifest.json")
        try:
            os.link(bank, out)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assert_input_refused(self.tmp.name, out, bank)
        self.assertTrue(os.path.samefile(bank, out))

    def test_report_cannot_replace_a_symlink_to_the_input(self):
        bank = self.write_bank(with_music=False)
        out = os.path.join(self.tmp.name, "manifest.json")
        try:
            os.symlink(bank, out)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks unavailable: {e}")
        self.assert_input_refused(bank, out, bank)
        self.assertTrue(os.path.islink(out))

    def test_report_cannot_replace_the_target_of_a_scanned_symlink(self):
        bank = self.write_bank(with_music=False)
        src = os.path.join(self.tmp.name, "banks")
        os.mkdir(src)
        try:
            os.symlink(bank, os.path.join(src, "linked.awc"))
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks unavailable: {e}")
        self.assert_input_refused(src, bank, bank)

    def test_malformed_input_is_protected_before_any_bank_is_read(self):
        bank = os.path.join(self.tmp.name, "bad.awc")
        with open(bank, "wb") as f:
            f.write(b"not an AWC")
        with mock.patch.object(awc_manifest, "_bank_rows", side_effect=AssertionError("bank read")):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                result = awc_manifest.main([self.tmp.name, "--out", bank])
        self.assertEqual(result, 2)
        self.assertIn("would overwrite the input", err.getvalue())
        with open(bank, "rb") as f:
            self.assertEqual(f.read(), b"not an AWC")

    def test_existing_reports_can_be_replaced_without_changing_the_bank(self):
        bank = self.write_bank(with_music=False)
        with open(bank, "rb") as f:
            original = f.read()
        for ext in ("csv", "json"):
            with self.subTest(format=ext):
                out = os.path.join(self.tmp.name, "manifest." + ext)
                with open(out, "w", encoding="utf-8") as f:
                    f.write("previous report\n")
                p = self.run_tool(self.tmp.name, "--out", out)
                self.assertEqual(p.returncode, 0, p.stderr)
                with open(out, encoding="utf-8") as f:
                    if ext == "json":
                        self.assertEqual(json.load(f)["stream_count"], 3)
                    else:
                        self.assertEqual(len(list(csv.DictReader(f))), 3)
                with open(bank, "rb") as f:
                    self.assertEqual(f.read(), original)

    def test_invalid_banks_are_not_counted(self):
        for name in ("bad1.awc", "bad2.awc"):
            with open(os.path.join(self.tmp.name, name), "wb") as f:
                f.write(b"not an AWC")
        p = self.run_tool(self.tmp.name, "--format", "json")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("warning", p.stderr.lower())
        self.assertIn("bad1.awc", p.stderr)
        self.assertIn("bad2.awc", p.stderr)
        self.assertIn("0 banks, 0 streams, 0 streamed-music channels", p.stderr)
        doc = json.loads(p.stdout)
        self.assertEqual((doc["bank_count"], doc["stream_count"]), (0, 0))
        self.assertEqual(doc["streams"], [])

    def test_incomplete_bank_discovery_preserves_the_previous_manifest(self):
        root = self.tmp.name
        blocked = os.path.join(root, "blocked")
        out = os.path.join(root, "manifest.json")
        previous = b'{"previous": true}\n'
        with open(out, "wb") as f:
            f.write(previous)

        def walk(path, onerror=None):
            if onerror:
                onerror(PermissionError(13, "permission denied", blocked))
            return iter(())

        with mock.patch.object(awc_manifest.os, "walk", walk), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(awc_manifest.main([root, "--out", out]), 1)
        self.assertIn(blocked, err.getvalue())
        with open(out, "rb") as f:
            self.assertEqual(f.read(), previous)

    def test_unreadable_bank_preserves_the_previous_manifest(self):
        good = os.path.join(self.tmp.name, "good.awc")
        blocked = os.path.join(self.tmp.name, "unreadable.awc")
        self.write_bank(good, with_music=False)
        self.write_bank(blocked, with_music=False)
        out = os.path.join(self.tmp.name, "manifest.json")
        previous = b'{"previous": true}\n'
        with open(out, "wb") as f:
            f.write(previous)
        read_bank = awc_manifest._bank_rows

        def read_with_error(path, bank):
            if path == blocked:
                raise PermissionError(13, "permission denied", path)
            return read_bank(path, bank)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(awc_manifest, "_bank_rows", side_effect=read_with_error), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(awc_manifest.main([self.tmp.name, "--out", out]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("cannot read", stderr.getvalue())
        self.assertIn(blocked, stderr.getvalue())
        with open(out, "rb") as f:
            self.assertEqual(f.read(), previous)

    def test_failed_write_keeps_the_previous_manifest(self):
        out = os.path.join(self.tmp.name, "manifest.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write('{"previous": true}\n')

        def full_disk(_doc, f, **_kwargs):
            f.write('{"bank_count": ')
            raise OSError(28, "No space left on device")

        with mock.patch.object(awc_manifest.json, "dump", side_effect=full_disk):
            with self.assertRaises(OSError):
                awc_manifest.write_manifest([], 0, 0, "json", out)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"previous": true}\n')
        self.assertEqual(os.listdir(self.tmp.name), ["manifest.json"])   # no temporary file left

    def test_failed_replace_of_read_only_manifest_removes_writable_temp(self):
        out = os.path.join(self.tmp.name, "manifest.json")
        previous = '{"previous": true}\n'
        with open(out, "w", encoding="utf-8") as f:
            f.write(previous)
        os.chmod(out, 0o444)
        self.addCleanup(lambda: os.path.exists(out) and os.chmod(out, 0o600))
        made_writable = []
        real_chmod = os.chmod

        def chmod(path, mode, *args, **kwargs):
            if isinstance(path, str) and os.path.basename(path).startswith(".awc_manifest-") and mode == 0o600:
                made_writable.append(path)
            return real_chmod(path, mode, *args, **kwargs)

        with mock.patch.object(awc_manifest.os, "replace", side_effect=OSError("replace denied")), \
                mock.patch.object(awc_manifest.os, "chmod", chmod):
            with self.assertRaises(OSError):
                awc_manifest.write_manifest([], 0, 0, "json", out)
        self.assertEqual(len(made_writable), 1)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), previous)
        self.assertEqual(os.listdir(self.tmp.name), ["manifest.json"])

    def test_recursive_report_distinguishes_banks_with_the_same_basename(self):
        for folder in ("left", "right"):
            os.mkdir(os.path.join(self.tmp.name, folder))
            self.write_bank(os.path.join(folder, "shared.awc"), with_music=False)
        p = self.run_tool(self.tmp.name, "--format", "json")
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = json.loads(p.stdout)["streams"]
        self.assertEqual([row["bank"] for row in rows], ["left/shared.awc"] * 3 + ["right/shared.awc"] * 3)

    def test_stereo_streamed_music_uses_packet_channels_and_per_channel_samples(self):
        desc, data = music_blob([0x40], 1920, 48000, toc=0xFC)   # 2 x 960 interleaved samples
        path = os.path.join(self.tmp.name, "stereo.awc")
        with open(path, "wb") as f:
            f.write(build_bank([(0x50, [(0x48, desc), (0x55, data)])]))
        p = self.run_tool(path)
        self.assertEqual(p.returncode, 0, p.stderr)
        rows = {r["stream"]: r for r in csv.DictReader(io.StringIO(p.stdout))}
        row = rows["00000040"]
        self.assertEqual((row["channels"], row["declared_samples"]), ("2", "960"))
        self.assertAlmostEqual(float(row["duration_s"]), 0.02, places=6)
        self.assertEqual(row["streamed_music"], "yes")


    def test_a_single_named_file_that_is_not_a_bank_is_refused(self):
        path = os.path.join(self.tmp.name, "notes.bin")
        with open(path, "wb") as f:
            f.write(b"not an AWC bank")
        out = os.path.join(self.tmp.name, "manifest.json")
        p = self.run_tool(path, "--out", out)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("holds no AWC bank", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertFalse(os.path.exists(out))

    def test_a_scanned_directory_with_no_bank_still_reports_empty(self):
        # The refusal is for one named file: an empty folder scan is a valid "0 banks" answer.
        os.mkdir(os.path.join(self.tmp.name, "banks"))
        with open(os.path.join(self.tmp.name, "banks", "notes.bin"), "wb") as f:
            f.write(b"not an AWC bank")
        p = self.run_tool(os.path.join(self.tmp.name, "banks"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0 banks, 0 streams, 0 streamed-music channels", p.stderr)
        self.assertEqual(p.stdout.splitlines()[0], ",".join(HEADER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
