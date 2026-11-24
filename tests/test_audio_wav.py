"""Check Ogg Opus end trimming, chained audio, and WAV output paths.

    python -I tests/test_audio_wav.py

The decode tests use synthetic audio and skip when ffmpeg is not on PATH.
"""
import contextlib
import errno
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_wav.py")
FFMPEG = shutil.which("ffmpeg")

sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import audio_wav  # noqa: E402
import awc_opus  # noqa: E402
import winbatch  # noqa: E402


class DecodeQueueTest(unittest.TestCase):
    def test_failure_summary_counts_all_but_prints_only_ten_first_lines(self):
        items = [(str(i), str(i)) for i in range(13)]
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(audio_wav, "collect", return_value=(items, 0)), \
                mock.patch.object(audio_wav, "plan_outputs", return_value=(items, 0, 0)), \
                mock.patch.object(audio_wav, "find_ffmpeg", return_value="unused"), \
                mock.patch.object(audio_wav, "decode", return_value=(False, "\n  first line\nextra detail")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            status = audio_wav.main([root, "--jobs", "2"])
        self.assertEqual(status, 1)
        self.assertIn("13 failed", out.getvalue())
        self.assertEqual(out.getvalue().count(": first line"), 10)
        self.assertNotIn("extra detail", out.getvalue())
        self.assertIn("and 3 more failures", out.getvalue())

    def test_submission_stays_within_worker_window(self):
        workers, consumed = 2, 0
        expected = {str(i) for i in range(1000)}
        received = set()

        def jobs():
            for submitted, name in enumerate(sorted(expected), 1):
                self.assertLessEqual(submitted - consumed, workers)
                yield name, name, name + ".wav"

        with audio_wav.concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, \
                mock.patch.object(audio_wav, "decode", new=lambda _, src, dst: (src, dst)):
            for rel, future in audio_wav._decode_jobs(pool, "unused", jobs(), workers):
                self.assertEqual(future.result(), (rel, rel + ".wav"))
                consumed += 1
                received.add(rel)
        self.assertEqual(received, expected)
        self.assertEqual(consumed, len(expected))


@unittest.skipUnless(FFMPEG, "ffmpeg is not on PATH")
class AudioWavTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.src = os.path.join(self.dir, "in", "one.ogg")
        os.makedirs(os.path.dirname(self.src))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=48000:duration=1",
                        "-ac", "1", "-c:a", "libopus", self.src], stdin=subprocess.DEVNULL, check=True)
        self.out = os.path.join(self.dir, "wav")
        self.dst = os.path.join(self.out, "one.wav")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_writes_exact_16_bit_mono_wav(self):
        p = self.run_tool(self.src, "--out", self.out, "--jobs", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(self.dst, "rb") as f:
            header = f.read(12)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:], b"WAVE")
        with wave.open(self.dst, "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 48000)
            self.assertEqual(w.getnframes(), 48000)

    def test_repackaged_stream_keeps_its_declared_end(self):
        # The first data page contains 50 silence packets (48,000 samples). The declared end falls before it.
        declared = 47877
        awc_opus.write_ogg_opus(self.src, [bytes.fromhex("f8fffe")] * 51, 1, 42, declared)
        self.assertEqual(audio_wav.ogg_output_samples(self.src), declared)
        p = self.run_tool(self.src, "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with wave.open(self.dst, "rb") as w:
            self.assertEqual(w.getnframes(), declared)

    def test_chained_links_match_direct_ffmpeg_decode(self):
        for name, tones in [("unequal", [(440, 100), (880, 200)]), ("repeated", [(440, 100)] * 3)]:
            with self.subTest(name=name):
                links = []
                for serial, (hz, ms) in enumerate(tones, 1):
                    tone = f"sine=frequency={hz}:sample_rate=48000:duration={ms / 1000}"
                    link = subprocess.check_output([FFMPEG, "-v", "error", "-f", "lavfi", "-i", tone,
                                                    "-c:a", "libopus", "-fflags", "+bitexact", "-serial_offset",
                                                    str(serial), "-f", "ogg", "pipe:1"], stdin=subprocess.DEVNULL)
                    self.assertEqual(int.from_bytes(link[14:18], "little"), serial)
                    links.append(link)
                src = Path(self.dir, name + ".ogg")
                src.write_bytes(b"".join(links))
                reference = os.path.join(self.dir, "reference.wav")
                subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-c:a", "pcm_s16le", reference],
                               stdin=subprocess.DEVNULL, check=True, capture_output=True)
                with wave.open(reference, "rb") as w:
                    self.assertEqual(w.getnframes(), sum(ms for _, ms in tones) * 48)
                    pcm = w.readframes(w.getnframes())
                p = self.run_tool(src)
                self.assertEqual(p.returncode, 0, p.stderr)
                with wave.open(str(src.with_suffix(".wav")), "rb") as w:
                    self.assertEqual(w.getnframes() * 2, len(pcm))
                    self.assertTrue(w.readframes(w.getnframes()) == pcm, "decoded PCM differs")

    def test_second_run_skips_the_existing_wav(self):
        self.assertEqual(self.run_tool(self.src, "--out", self.out).returncode, 0)
        p = self.run_tool(self.src, "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0 WAV written", p.stdout)
        self.assertIn("1 already present", p.stdout)

    def test_decode_preserves_an_existing_wav_at_the_old_temporary_name(self):
        old_tmp = self.dst + ".part.wav"
        os.makedirs(self.out)
        with open(old_tmp, "wb") as f:
            f.write(b"previous output")
        # This input would resume from old_tmp while the other input is decoded.
        shutil.copyfile(self.src, os.path.join(os.path.dirname(self.src), "one.wav.part.opus"))
        p = self.run_tool(os.path.dirname(self.src), "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 WAV written, 1 already present", p.stdout)
        with open(old_tmp, "rb") as f:
            self.assertEqual(f.read(), b"previous output")
        self.assertEqual(sorted(os.listdir(self.out)), ["one.wav", "one.wav.part.wav"])

    def test_repair_audio_matches_pcm_with_different_tags_and_rejects_other_audio(self):
        self.assertEqual(self.run_tool(self.src, "--out", self.out).returncode, 0)
        tagged = os.path.join(self.dir, "tagged.wav")
        silent = os.path.join(self.dir, "silent.wav")
        subprocess.run([FFMPEG, "-v", "error", "-i", self.dst, "-c:a", "copy", "-metadata", "comment=other", tagged],
                       stdin=subprocess.DEVNULL, check=True)
        subprocess.run([FFMPEG, "-v", "error", "-i", self.dst, "-af", "volume=0", silent],
                       stdin=subprocess.DEVNULL, check=True)
        expected = audio_wav.audio_digest(FFMPEG, self.src, source=True)
        self.assertIsNotNone(expected)
        self.assertEqual(audio_wav.audio_digest(FFMPEG, tagged, source=False), expected)
        self.assertNotEqual(audio_wav.audio_digest(FFMPEG, silent, source=False), expected)

    def test_readonly_shared_and_foreign_repair_wavs_survive_resume(self):
        src_dir = os.path.dirname(self.src)
        first, second = (os.path.join(src_dir, n) for n in ("Track.ogg", "track.opus"))
        os.rename(self.src, first)
        os.makedirs(self.out)
        with open(os.path.join(self.out, "CaseProbe"), "wb"):
            pass
        if not os.path.exists(os.path.join(self.out, "caseprobe")):
            second = os.path.join(src_dir, "Track.opus")  # a lexical collision on a case-sensitive test volume
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=1",
                        "-c:a", "libopus", second], stdin=subprocess.DEVNULL, check=True)
        shared = os.path.join(self.out, "Track.wav")
        occupied = os.path.join(self.out, os.path.basename(second) + ".wav")
        for name in (shared, occupied):
            with wave.open(name, "wb") as w:
                w.setparams((1, 2, 48000, 48000, "NONE", "not compressed"))
                w.writeframes(bytes(96000))
        with open(shared, "rb") as f:
            previous = f.read()
        try:
            for name in (shared, occupied):
                os.chmod(name, 0o444)
            p = self.run_tool(src_dir, "--out", self.out, "--jobs", "2")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("2 WAV written", p.stdout)
            for src, name in ((first, "Track.ogg.wav"), (second, os.path.basename(second) + "~2.wav")):
                self.assertEqual(audio_wav.audio_digest(FFMPEG, os.path.join(self.out, name), source=False),
                                 audio_wav.audio_digest(FFMPEG, src, source=True))
            p = self.run_tool(src_dir, "--out", self.out, "--jobs", "1")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("0 WAV written, 2 already present", p.stdout)
            for name in (shared, occupied):
                with open(name, "rb") as f:
                    self.assertEqual(f.read(), previous)
        finally:
            for name in (shared, occupied):
                os.chmod(name, 0o644)

    def test_new_source_cannot_resume_from_a_previous_inputs_repair_name(self):
        src = os.path.join(os.path.dirname(self.src), "Track.ogg.ogg")
        os.rename(self.src, src)
        os.makedirs(self.out)
        old = os.path.join(self.out, "Track.ogg.wav")
        with wave.open(old, "wb") as w:
            w.setparams((1, 2, 48000, 48000, "NONE", "not compressed"))
            w.writeframes(bytes(96000))                  # right frame count, wrong recording
        with open(old, "rb") as f:
            previous = f.read()
        p = self.run_tool(src, "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 WAV written", p.stdout)
        fixed = os.path.join(self.out, "Track.ogg~2.wav")
        self.assertEqual(audio_wav.audio_digest(FFMPEG, fixed, source=False),
                         audio_wav.audio_digest(FFMPEG, src, source=True))
        p = self.run_tool(os.path.dirname(src), "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0 WAV written, 1 already present", p.stdout)
        with open(old, "rb") as f:
            self.assertEqual(f.read(), previous)

    def test_clean_case_pair_has_two_outputs_and_a_stable_second_run(self):
        folder = os.path.dirname(self.src)
        shutil.copyfile(self.src, os.path.join(folder, "One.opus"))
        first = self.run_tool(folder, "--out", self.out, "--jobs", "2")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("2 WAV written", first.stdout)
        files = [os.path.join(self.out, name) for name in os.listdir(self.out)]
        self.assertEqual(len(files), 2)
        times = [os.stat(path).st_mtime_ns for path in files]
        second = self.run_tool(folder, "--out", self.out, "--jobs", "1")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("0 WAV written, 2 already present", second.stdout)
        self.assertEqual([os.stat(path).st_mtime_ns for path in files], times)

    def test_unlabelled_quad_pcm_matches_without_guessing_a_speaker_layout(self):
        src, dst = (os.path.join(self.dir, name) for name in ("quad.ogg", "quad.wav"))
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=quad", "-t", "0.12",
                        "-c:a", "libopus", "-mapping_family", "255", src], stdin=subprocess.DEVNULL, check=True)
        with open(src, "rb") as f:
            header = f.read(512)
        self.assertEqual(header[header.index(b"OpusHead") + 18], 255)
        self.assertTrue(audio_wav.decode(FFMPEG, src, dst)[0])
        expected = audio_wav.audio_digest(FFMPEG, src, source=True)
        self.assertIsNotNone(expected)
        self.assertEqual(audio_wav.audio_digest(FFMPEG, dst, source=False), expected)

    def test_explicit_speaker_layouts_still_distinguish_the_same_pcm(self):
        signatures = []
        for layout in ("5.1", "5.1(side)"):
            path = os.path.join(self.dir, layout + ".wav")
            subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=" + layout, "-t", "0.1",
                            "-c:a", "pcm_s16le", path], stdin=subprocess.DEVNULL, check=True)
            signatures.append(audio_wav.audio_digest(FFMPEG, path, source=False))
        self.assertNotIn(None, signatures)
        self.assertNotEqual(*signatures)

    def test_long_repaired_names_resume_from_a_readonly_directory(self):
        if os.name == "nt":
            self.skipTest("uses the POSIX lexical naming path")
        with open(os.path.join(self.dir, "CaseProbe"), "wb"):
            pass
        if not os.path.exists(os.path.join(self.dir, "caseprobe")):
            self.skipTest("requires a case-insensitive output volume")
        folder = os.path.dirname(self.src)
        first = os.path.join(folder, "A" * 249 + ".ogg")
        os.rename(self.src, first)
        shutil.copyfile(first, os.path.join(folder, "a" * 249 + ".opus"))
        result = self.run_tool(folder, "--out", self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 WAV written", result.stdout)
        os.chmod(self.out, 0o555)
        try:
            result = self.run_tool(folder, "--out", self.out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0 WAV written, 2 already present", result.stdout)
        finally:
            os.chmod(self.out, 0o755)

    def test_missing_ffmpeg_is_an_explanation_not_a_traceback(self):
        p = self.run_tool(self.src, "--out", self.out, "--ffmpeg", os.path.join(self.dir, "no-ffmpeg"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ffmpeg", p.stderr.lower())
        self.assertNotIn("Traceback", p.stderr)


class CliArgumentTest(unittest.TestCase):
    def test_rejects_nonpositive_jobs(self):
        with tempfile.TemporaryDirectory() as source:
            for jobs in ("0", "-1"):
                with self.subTest(jobs=jobs):
                    result = subprocess.run(
                        [sys.executable, "-I", TOOL, source, "--jobs", jobs],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--jobs must be at least 1", result.stderr)


class OpusInputSelectionTest(unittest.TestCase):
    def test_collect_requires_an_opus_id_packet(self) -> None:
        opus_id = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 312, 48000, 0, 0)
        opus_bos = awc_opus.ogg_page(10, 0, 0, [opus_id], 2)
        vorbis_id = b"\x01vorbis" + struct.pack("<IBIIII2B", 0, 1, 48000, 0, 0, 0, 0xB8, 1)
        vorbis_bos = awc_opus.ogg_page(11, 0, 0, [vorbis_id], 2)
        comment = b"COMMENT=OpusHead"
        vorbis_comment = b"\x03vorbis" + struct.pack("<III", 0, 1, len(comment)) + comment + b"\x01"
        vorbis_tags = awc_opus.ogg_page(11, 1, 0, [vorbis_comment], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "opus.ogg").write_bytes(opus_bos)
            (root / "vorbis.ogg").write_bytes(vorbis_bos + vorbis_tags)
            (root / "incomplete.opus").write_bytes(opus_bos[:-1])

            found, skipped = audio_wav.collect(directory)

        self.assertEqual([stem for _, stem in found], ["opus"])
        self.assertEqual(skipped, 2)

    def test_plan_outputs_does_not_reserve_rejected_wav_slots_for_later_inputs(self) -> None:
        from collections import Counter
        from unittest.mock import patch

        src = os.path.join("fake_src")
        out = os.path.join("fake_out")
        p0 = os.path.join(src, "a.ogg.opus")
        p1 = os.path.join(src, "a.ogg~2.opus")
        items = [(p0, "a.ogg"), (p1, "a.ogg~2")]
        src_dig = {p0: "D0", p1: "D1"}
        wav_dig = {
            os.path.join(out, "a.ogg.wav"): "DX",
            os.path.join(out, "a.ogg~2.wav"): "DY",
            os.path.join(out, "a.ogg~3.wav"): "D1",
            os.path.join(out, "a.ogg~4.wav"): "D0",
        }
        dst_calls = Counter()

        def fake_digest(_ff, p, source=False):
            if source:
                return src_dig[p]
            dst_calls[p] += 1
            return wav_dig.get(p)

        with patch.object(audio_wav, "audio_digest", side_effect=fake_digest), \
                patch("os.path.lexists", side_effect=lambda p: p in wav_dig), \
                patch("os.path.isfile", side_effect=lambda p: p in wav_dig):
            planned, renamed, moved = audio_wav.plan_outputs(items, src, out, "ffmpeg")
        self.assertEqual([(os.path.basename(p), stem) for p, stem in planned],
                         [("a.ogg.opus", "a.ogg~4"), ("a.ogg~2.opus", "a.ogg~3")])
        self.assertEqual((renamed, moved), (0, 2))
        self.assertEqual(max(dst_calls.values()), 1)


class GranuleTest(unittest.TestCase):
    def test_trim_requires_identifiable_matching_endpoints(self):
        # A 240-channel family-255 header occupies two laces. Every output maps to the same mono stream.
        head = b"OpusHead" + struct.pack("<BBHIhBBB", 1, 240, 312, 48000, 0, 255, 1, 0) + bytes(240)
        bos = awc_opus.ogg_page(10, 0, 0, [head], 2)
        eos = awc_opus.ogg_page(10, 2, 1272, [b"\xf8\xff\xfe"], 4)
        cases = [
            ("matching", bos, eos, 960),
            ("different serial", bos, awc_opus.ogg_page(11, 2, 1272, [b"x"], 4), 0),
            ("no EOS", bos, awc_opus.ogg_page(10, 2, 1272, [b"x"], 0), 0),
            ("no BOS", awc_opus.ogg_page(10, 0, 0, [head], 0), eos, 0),
            ("embedded signature", awc_opus.ogg_page(10, 0, 0, [b"x" + head], 2), eos, 0),
            ("short header", awc_opus.ogg_page(10, 0, 0, [head[:12]], 2), eos, 0),
            ("split across packets", awc_opus.ogg_page(10, 0, 0, [head[:12], head[12:]], 2), eos, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sample.ogg")
            for name, first, last, expected in cases:
                with self.subTest(name=name):
                    path.write_bytes(first + last)
                    self.assertEqual(audio_wav.ogg_output_samples(path), expected)

    def test_walk_rejects_same_serial_chaining_multiplexing_and_trailing_data(self):
        head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 312, 48000, 0, 0)
        bos = awc_opus.ogg_page(10, 0, 0, [head], 2)
        tags = awc_opus.ogg_page(10, 1, 0, [b"OpusTags\x00\x00\x00\x00\x00\x00\x00\x00"], 0)
        mid = awc_opus.ogg_page(10, 2, 1272, [b"\xf8\xff\xfe"], 0)
        eos = awc_opus.ogg_page(10, 3, 2232, [b"\xf8\xff\xfe"], 4)
        # Second link reusing the same serial number (valid per RFC 3533 section 4).
        link2 = (
            awc_opus.ogg_page(10, 0, 0, [head], 2)
            + awc_opus.ogg_page(10, 1, 0, [b"OpusTags\x00\x00\x00\x00\x00\x00\x00\x00"], 0)
            + awc_opus.ogg_page(10, 2, 1272, [b"\xf8\xff\xfe"], 4)
        )
        cases = [
            ("same-serial chained", bos + tags + mid + eos + link2, 0),
            ("same-serial chained without first EOS", bos + tags + mid + link2, 0),
            ("trailing garbage after EOS", bos + tags + eos + b"trailing", 0),
            (
                "embedded OggS in intermediate payload",
                bos + tags + awc_opus.ogg_page(10, 2, 1272, [b"xxOggS\x00\x04yy"], 0) + eos,
                1920,
            ),
            (
                "continuation flags on intermediate and EOS",
                bos
                + tags
                + awc_opus.ogg_page(10, 2, 1272, [b"\xf8\xff\xfe"], 1)
                + awc_opus.ogg_page(10, 3, 2232, [b"\xf8\xff\xfe"], 5),
                1920,
            ),
            (
                "multiplexed different serial mid-stream",
                bos + tags + awc_opus.ogg_page(11, 0, 0, [b"other"], 0) + eos,
                0,
            ),
            ("truncated page at EOF", bos + tags + eos[:-1], 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sample.ogg")
            for name, data, expected in cases:
                with self.subTest(name=name):
                    path.write_bytes(data)
                    self.assertEqual(audio_wav.ogg_output_samples(path), expected)


# Stands in for ffmpeg: writes the input's file name into the output, so each WAV shows which file it came from.
FAKE_FFMPEG = """import os, sys
args = sys.argv[1:]
src = args[args.index("-i") + 1]
if "copy" in args:
    with open(src, "rb") as f:
        data = f.read()
else:
    data = b"FAKE:" + os.path.basename(src).encode()
if args[-1] == "-":
    sys.stdout.buffer.write(b"RIFF" + bytes(4) + b"WAVE" + data)
else:
    with open(args[-1], "wb") as f:
        f.write(data)
"""

# FAKE_FFMPEG that also logs what it could read from its stdin, one line per run, to $FAKE_FFMPEG_STDIN_LOG.
STDIN_LOGGING_FFMPEG = FAKE_FFMPEG + """
with open(os.environ["FAKE_FFMPEG_STDIN_LOG"], "a", encoding="utf-8") as log:
    log.write(repr(sys.stdin.buffer.read() if sys.stdin else b"") + "\\n")
"""


def write_fake_ffmpeg(directory, source=FAKE_FFMPEG):
    """Write source as a Python script and return an ffmpeg command for --ffmpeg that runs it."""
    script = os.path.join(directory, "fake_ffmpeg.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(source)
    if os.name == "nt":
        ffmpeg = winbatch.write_python_cmd(os.path.join(directory, "ffmpeg.cmd"), sys.executable, script)
    else:
        ffmpeg = os.path.join(directory, "ffmpeg")
        with open(ffmpeg, "w", encoding="utf-8") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        os.chmod(ffmpeg, 0o755)
    return ffmpeg


def write_fake_opus_inputs(directory, names=("x.ogg", "x.opus", "y.ogg")):
    """Write only the Opus ID page because the fake ffmpeg never parses audio."""
    os.makedirs(directory)
    head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 312, 48000, 0, 0)
    page = awc_opus.ogg_page(10, 0, 0, [head], 2)
    for name in names:
        with open(os.path.join(directory, name), "wb") as f:
            f.write(page)


class SameStemTest(unittest.TestCase):
    """x.ogg and x.opus both decoded to x.wav: two jobs raced on one file and one result was lost."""

    def test_inputs_that_share_a_stem_keep_their_extension(self):
        items = [("d/x.ogg", "x"), ("d/x.opus", "x"), ("d/y.ogg", "y")]
        self.assertEqual(audio_wav.output_names(items),
                         ([("d/x.ogg", "x.ogg"), ("d/x.opus", "x.opus"), ("d/y.ogg", "y")], 2))

    def test_unique_names_are_unchanged(self):
        items = [("a/x.ogg", os.path.join("a", "x")), ("b/x.opus", os.path.join("b", "x")), ("y.opus", "y")]
        self.assertEqual(audio_wav.output_names(items), (items, 0))

    def test_a_name_that_still_clashes_gets_a_counter(self):
        # x.ogg keeps its extension (it clashes with x.opus) and then meets x.ogg.ogg's plain stem.
        items = [("x.ogg", "x"), ("x.ogg.ogg", "x.ogg"), ("x.opus", "x")]
        named, _ = audio_wav.output_names(items)
        self.assertEqual(named, [("x.ogg", "x.ogg"), ("x.ogg.ogg", "x.ogg~2"), ("x.opus", "x.opus")])

    def test_names_that_differ_only_in_case_clash_where_the_file_system_ignores_case(self):
        items = [("x.ogg", "x"), ("X.OPUS", "X")]
        named, renamed = audio_wav.output_names(items)
        if os.path.normcase("X") == os.path.normcase("x"):
            self.assertEqual((named, renamed), ([("x.ogg", "x.ogg"), ("X.OPUS", "X.OPUS")], 2))
        else:
            self.assertEqual((named, renamed), (items, 0))

    def test_output_names_follow_linked_directories_and_counter_collisions(self):
        with tempfile.TemporaryDirectory() as out:
            first = os.path.join(out, "first")
            os.mkdir(first)
            try:
                os.symlink(first, os.path.join(out, "second"), target_is_directory=True)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"directory symlinks unavailable: {e}")

            # The worker will create child/ later. Both logical paths lead to one physical WAV.
            a = os.path.join("first", "child", "track")
            b = os.path.join("second", "child", "track")
            c = os.path.join("independent", "child", "track")
            items = [(a + ".ogg", a), (b + ".ogg", b), (c + ".opus", c)]
            expected = [(a + ".ogg", a + ".ogg"), (b + ".ogg", b + ".ogg~2"), (c + ".opus", c)]
            self.assertEqual(audio_wav.output_names(items, out), (expected, 2))
            self.assertFalse(os.path.exists(os.path.join(first, "child")))
            os.mkdir(os.path.join(first, "child"))
            self.assertEqual(audio_wav.output_names(items, out), (expected, 2))

    def test_case_collisions_in_leaves_and_missing_parents_follow_the_output_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in")
            os.mkdir(os.path.join(tmp, "Probe"))
            insensitive = os.path.isdir(os.path.join(tmp, "probe"))
            for a, b in (("Track", "track"), (os.path.join("A", "x"), os.path.join("a", "x"))):
                with self.subTest(a=a, b=b):
                    out = os.path.join(tmp, "out")
                    items = [(os.path.join(src, a + ".ogg"), a), (os.path.join(src, b + ".opus"), b)]
                    names, lexical, native = audio_wav.plan_outputs(items, src, out, None)
                    self.assertEqual([stem for _, stem in names], [a + ".ogg", b + ".opus"] if insensitive else [a, b])
                    self.assertEqual(lexical + native, 2 if insensitive else 0)
                    self.assertFalse(os.path.exists(out))

    def test_unicode_spellings_follow_native_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = "caf\u00e9", "cafe\u0301"
            with open(os.path.join(tmp, a), "wb"):
                pass
            equivalent = os.path.exists(os.path.join(tmp, b))
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            items = [(os.path.join(src, a + ".ogg"), a), (os.path.join(src, b + ".opus"), b)]
            names, lexical, native = audio_wav.plan_outputs(items, src, out, None)
            self.assertEqual([stem for _, stem in names], [a + ".ogg", b + ".opus"] if equivalent else [a, b])
            self.assertEqual(lexical + native, 2 if equivalent else 0)
            self.assertFalse(os.path.exists(out))

    def test_distinct_readonly_hard_links_keep_their_own_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.mkdir(out)
            a, b = os.path.join(out, "Track.wav"), os.path.join(out, "track.wav")
            with open(a, "wb"):
                pass
            if os.path.exists(b):
                self.skipTest("requires two case-distinct directory entries")
            try:
                os.link(a, b)
            except OSError as e:
                self.skipTest(f"hard links unavailable: {e}")
            items = [(os.path.join(src, "Track.ogg"), "Track"), (os.path.join(src, "track.opus"), "track")]
            os.chmod(out, 0o555)
            try:
                self.assertEqual(audio_wav.plan_outputs(items, src, out, None), (items, 0, 0))
            finally:
                os.chmod(out, 0o755)

    def test_canonical_alias_with_a_case_distinct_hard_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.mkdir(out)
            a, b = "\u00c9", "E\u0301"
            old = os.path.join(out, a + ".wav")
            with open(old, "wb") as f:
                f.write(b"older WAV")
            lower = os.path.join(out, a.lower() + ".wav")
            if not os.path.exists(os.path.join(out, b + ".wav")) or os.path.exists(lower):
                self.skipTest("requires canonical equivalence with distinct letter case")
            try:
                os.link(old, lower)
            except OSError as e:
                self.skipTest(f"hard links unavailable: {e}")
            items = [(os.path.join(src, a + ".ogg"), a), (os.path.join(src, b + ".opus"), b)]
            named, _, repaired = audio_wav.plan_outputs(items, src, out, None)
            self.assertEqual(repaired, 2)
            self.assertEqual([stem for _, stem in named], [a + ".ogg", b + ".opus"])
            with open(old, "rb") as f:
                self.assertEqual(f.read(), b"older WAV")

    def test_readonly_repaired_pair_and_distinct_plain_wav_in_one_fold_group(self):
        with tempfile.TemporaryDirectory() as out:
            a, b, c = "\u00c9", "E\u0301", "\u00e9"
            for name in (a + ".ogg.wav", a + ".opus.wav", c + ".wav"):
                with open(os.path.join(out, name), "wb"):
                    pass
            if os.path.exists(os.path.join(out, a + ".wav")) or not os.path.exists(os.path.join(out, b + ".ogg.wav")):
                self.skipTest("requires canonical equivalence with distinct letter case")
            items = [(a + ".ogg", a), (b + ".opus", b), (c + ".ogg", c)]
            uncertain = set()
            os.chmod(out, 0o555)
            try:
                repaired = audio_wav.native_collisions([0, 1, 2], items, lambda stem: os.path.join(out, stem), uncertain)
                self.assertEqual(repaired, {0, 1})
                self.assertFalse(uncertain)
            finally:
                os.chmod(out, 0o755)

    @unittest.skipUnless(os.name == "nt", "checks the Windows normcase mapping")
    def test_broad_unicode_filter_includes_windows_lexical_matches(self):
        for n in range(0x10000):
            c = chr(n)
            self.assertEqual(audio_wav.fold_name(c), audio_wav.fold_name(os.path.normcase(c)), hex(n))


class FakeFfmpegDecodeTest(unittest.TestCase):
    def test_every_input_gets_its_own_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in")
            out = os.path.join(tmp, "out")
            write_fake_opus_inputs(src)
            ffmpeg = write_fake_ffmpeg(tmp)

            run = [sys.executable, "-I", TOOL, src, "--out", out, "--jobs", "3", "--ffmpeg", ffmpeg]
            p = subprocess.run(run, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("3 WAV written", p.stdout)
            self.assertIn("2 file(s) share a name", p.stdout)
            got = {}
            for name in sorted(os.listdir(out)):
                with open(os.path.join(out, name), "rb") as f:
                    got[name] = f.read()
            self.assertEqual(got, {"x.ogg.wav": b"FAKE:x.ogg", "x.opus.wav": b"FAKE:x.opus", "y.wav": b"FAKE:y.ogg"})

            p = subprocess.run(run, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("0 WAV written, 3 already present", p.stdout)

    def test_directory_at_wav_destination_counts_as_failed_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in")
            out = os.path.join(tmp, "out")
            blocker = os.path.join(out, "blocked.wav")
            write_fake_opus_inputs(src, ("blocked.ogg", "existing.ogg", "new.ogg"))
            os.makedirs(blocker)
            marker = os.path.join(blocker, "keep")
            with open(marker, "wb") as f:
                f.write(b"keep the directory")
            with open(os.path.join(out, "existing.wav"), "wb") as f:
                f.write(b"existing audio")

            run = [sys.executable, "-I", TOOL, src, "--out", out, "--jobs", "1",
                   "--ffmpeg", write_fake_ffmpeg(tmp)]
            result = subprocess.run(run, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("1 WAV written, 1 already present, 1 failed", result.stdout)
            self.assertIn("FAILED blocked: output path is not a file", result.stdout)
            for filename, expected in ((marker, b"keep the directory"),
                                       (os.path.join(out, "existing.wav"), b"existing audio"),
                                       (os.path.join(out, "new.wav"), b"FAKE:new.ogg")):
                with open(filename, "rb") as f:
                    self.assertEqual(f.read(), expected)

    def test_ffmpeg_gets_no_stdin(self):
        # ffmpeg reads stdin for key commands. Given the caller's stdin, it stops a run in the background of a
        # terminal ("Stopped (tty output)") and swallows input piped to the caller, such as a shell loop's file list.
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in")
            write_fake_opus_inputs(src)
            log = os.path.join(tmp, "stdin.log")
            env = dict(os.environ, FAKE_FFMPEG_STDIN_LOG=log)
            run = [sys.executable, "-I", TOOL, src, "--out", os.path.join(tmp, "out"), "--jobs", "1",
                   "--ffmpeg", write_fake_ffmpeg(tmp, STDIN_LOGGING_FFMPEG)]
            # The first run decodes; the second checks the WAVs of the inputs that share a name (audio_digest).
            for step, summary in (("decode", "3 WAV written"), ("resume check", "0 WAV written, 3 already present")):
                with self.subTest(step):
                    if os.path.exists(log):
                        os.remove(log)
                    p = subprocess.run(run, input="tests/test_next.py\n", capture_output=True, text=True, env=env)
                    self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                    self.assertIn(summary, p.stdout)
                    with open(log, encoding="utf-8") as f:
                        reads = f.read().splitlines()
                    self.assertTrue(reads, "the fake ffmpeg never ran")
                    self.assertEqual(reads, ["b''"] * len(reads))


class AudioWavCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.src_dir = os.path.join(self.dir, "in")
        os.makedirs(self.src_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_rejects_jobs_below_one(self):
        for val in ("0", "-1", "-5"):
            with self.subTest(val=val):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    audio_wav.main([self.src_dir, "--jobs", val])
                self.assertEqual(cm.exception.code, 2)
                self.assertIn("--jobs must be at least 1", err.getvalue())

    def test_cli_subprocess_rejects_jobs_below_one(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.src_dir, "--jobs", "0"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--jobs must be at least 1", p.stderr)

    def test_cli_rejects_out_file(self):
        out_file = os.path.join(self.dir, "out_file")
        with open(out_file, "wb") as f:
            f.write(b"content")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            audio_wav.main([self.src_dir, "--out", out_file])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("is a file, not a folder", err.getvalue())

    def test_cli_subprocess_rejects_out_file(self):
        out_file = os.path.join(self.dir, "out_file")
        with open(out_file, "wb") as f:
            f.write(b"content")
        p = subprocess.run([sys.executable, "-I", TOOL, self.src_dir, "--out", out_file],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("is a file, not a folder", p.stderr)

    def test_cli_rejects_missing_source(self):
        missing = os.path.join(self.dir, "nonexistent")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = audio_wav.main([missing])
        self.assertEqual(code, 2)
        self.assertIn(f"source not found: {missing}", err.getvalue())

    def test_cli_reports_unreadable_source_directory(self):
        bank = os.path.join(self.src_dir, "bank")
        os.mkdir(bank)
        scandir = os.scandir

        def list_directory(path):
            if path == bank:
                raise PermissionError(errno.EACCES, "permission denied", path)
            return scandir(path)

        err = io.StringIO()
        with mock.patch.object(os, "scandir", list_directory), contextlib.redirect_stderr(err):
            code = audio_wav.main([self.src_dir])
        self.assertEqual(code, 2)
        self.assertIn(bank, err.getvalue())

    def test_cli_rejects_non_opus_file_extension(self):
        txt_file = os.path.join(self.dir, "test.txt")
        with open(txt_file, "wb") as f:
            f.write(b"hello")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = audio_wav.main([txt_file])
        self.assertEqual(code, 2)
        self.assertIn("is not an .ogg/.opus file", err.getvalue())

    def test_cli_rejects_file_not_opus(self):
        bogus_ogg = os.path.join(self.dir, "test.ogg")
        with open(bogus_ogg, "wb") as f:
            f.write(b"NotOpusData" * 10)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = audio_wav.main([bogus_ogg])
        self.assertEqual(code, 2)
        self.assertIn("is not Ogg Opus", err.getvalue())


if __name__ == "__main__":
    unittest.main()
