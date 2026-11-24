"""Exit codes of a built rpf9dump. Set RPF9DUMP_DLL to its Release DLL to run it."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib


def awc_rpf(path):
    # "9FPR" archive holding one AWC bank: one nonstreamed XMA2 (codec 5) mono stream, id 0x1234, 64 data bytes.
    hlen = 16 + 4 + 2 * 8
    tag = lambda t, size, rel: struct.pack("<Q", (t << 56) | (size << 28) | rel)
    fmt = bytearray(32)
    struct.pack_into("<I", fmt, 0, 1000)
    struct.pack_into("<H", fmt, 8, 48000)
    fmt[28], fmt[29] = 5, 1
    bank = (b"ADAT" + struct.pack("<IIII", 0xFF000001, 1, hlen, (2 << 29) | 0x1234)
            + tag(0x76, 32, hlen) + tag(0x55, 64, hlen + 32) + bytes(fmt) + b"\x11" * 64)
    path.write_bytes(b"9FPR" + bytes(0x1000 - 4) + bank + bytes(0x1000))


def texture_dictionary_rpf(path):
    # "9FPR" archive holding one recoverable texture dictionary: a zlib system block at 0x1010 (16 zero
    # bytes into a 4 KiB page, where the recovery scan looks) inflating to a 1-texture dictionary
    # (R8_UNORM 4x4, one mip, inline name "probe"), then zero padding to the next 16-byte boundary and a
    # zlib graphics block inflating to the 4096 bytes the descriptor allocates. System pointers use base
    # 0x50000000 and graphics pointers 0x60000000, matching RageTextureDecoder's pointer checks. The
    # resource page-count block sits at 0x48, inside the free 16 bytes below the descriptor at 0x50: it
    # must not overlap the descriptor, whose width field at t+0x38 the decoder validates against the
    # allocation ("Descriptor allocation disagrees with AMD mip layout").
    t = 0x50                                    # the one texture descriptor, 0x70 bytes
    system = bytearray(0xD0)
    system[0x08:0x10] = struct.pack("<Q", 0x50000000 + 0x40)          # -> resource page counts (at 0x48)
    system[0x20:0x28] = struct.pack("<Q", 0x50000000 + 0xB0)          # -> name hashes
    system[0x28:0x2A] = struct.pack("<H", 1)                          # dictionary count
    system[0x30:0x38] = struct.pack("<Q", 0x50000000 + 0xB8)          # -> texture pointer array
    system[0x38:0x3A] = struct.pack("<H", 1)                          # dictionary count
    system[0x3A:0x3C] = struct.pack("<H", 1)                          # >= count
    system[t + 0x08:t + 0x0C] = struct.pack("<I", 4096)               # allocation in elements
    system[t + 0x0C:t + 0x0E] = struct.pack("<H", 1)                  # element bytes (R8_UNORM)
    system[t + 0x18:t + 0x20] = struct.pack("<Q", 0x60000000)         # graphics pointer, offset 0
    system[t + 0x28:t + 0x30] = struct.pack("<Q", 0x50000000 + t + 0x50)
    system[t + 0x30:t + 0x38] = struct.pack("<Q", 0x50000000 + 0xC0)  # -> inline-name descriptor
    system[t + 0x38:t + 0x3A] = struct.pack("<H", 4)                  # width
    system[t + 0x3A:t + 0x3C] = struct.pack("<H", 4)                  # height
    system[t + 0x3C:t + 0x3E] = struct.pack("<H", 1)                  # depth
    system[t + 0x3E:t + 0x43] = bytes([1, 61, 5, 0, 1])               # 2D, R8_UNORM, tag 5, -, 1 mip
    system[0x48:0x4A] = bytes([1, 1])                                 # system/graphics resource pages
    system[0xB0:0xB4] = struct.pack("<I", 0x11111111)                 # name hash
    system[0xB8:0xC0] = struct.pack("<Q", 0x50000000 + t)             # pointer to the texture descriptor
    system[0xC0:0xC8] = struct.pack("<Q", 0)                          # descriptor: inline name follows
    system[0xC8:0xCE] = b"probe\x00"
    system_z = zlib.compress(bytes(system))
    payload = system_z + b"\x00" * (-len(system_z) % 16) + zlib.compress(bytes(4096))
    path.write_bytes(b"9FPR" + struct.pack("<I", 0x1000) + bytes(0x1010 - 8) + payload)


def run(*args):
    return subprocess.run(["dotnet", os.environ["RPF9DUMP_DLL"], *map(str, args)], capture_output=True, text=True)


@unittest.skipUnless(os.environ.get("RPF9DUMP_DLL"), "requires a built rpf9dump")
class Rpf9dumpExitCodeTest(unittest.TestCase):
    def test_missing_or_directory_source_exits_2_before_the_output_folder_is_created(self):
        # A mistyped source used to create <out>/<name> and then die with an unhandled FileNotFoundException
        # from FileInfo.Length. Both mistakes are usage errors (exit 2) and must leave the output alone.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, src in (("missing", root / "missing.rpf"), ("directory", root)):
                with self.subTest(source=label):
                    out = root / f"out-{label}"
                    done = run("ttpf", src, out)
                    self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                    self.assertIn("source is not a file", done.stderr)
                    self.assertNotIn("Unhandled exception", done.stderr)
                    self.assertFalse(out.exists(), f"{label} source left {out} behind")

    def test_failed_xma_write_exits_nonzero_and_rerun_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "audio.rpf"
            awc_rpf(src)
            wav = root / "out" / "audio" / "xma" / "audio_0001_at_0000000000001000" / "00001234.xma.wav"
            blocker = Path(str(wav) + ".partial")
            blocker.mkdir(parents=True)          # the write of this one stream fails
            failed = run("xma", src, root / "out")
            self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
            self.assertFalse(wav.exists())
            blocker.rmdir()
            retry = run("xma", src, root / "out")
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertEqual(wav.read_bytes()[:4], b"RIFF")
            self.assertEqual(wav.read_bytes()[-64:], b"\x11" * 64)

    def test_failed_texture_write_exits_nonzero_and_withholds_manifest(self):
        # A per-item write failure must fail the run like ttpf and xma do: make_game resumes the
        # textures/models stages on the mere existence of manifest.json, so a manifest published after a
        # failed DDS write would mark the stage finished with the file missing forever.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "textures.rpf"
            texture_dictionary_rpf(src)
            sub = root / "out" / "textures" / "textures"
            dds = sub / "dictionary_at_0000000000001010" / "000_probe.dds"
            blocker = Path(str(dds) + ".partial")
            blocker.mkdir(parents=True)          # the write of this one DDS fails
            failed = run("textures", src, root / "out")
            self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
            self.assertFalse(dds.exists(), failed.stdout)          # the blocked write must not have succeeded
            self.assertFalse((sub / "manifest.json").exists(), failed.stdout)
            self.assertFalse((sub / "manifest.json.partial").exists())
            blocker.rmdir()
            retry = run("textures", src, root / "out")
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertTrue((sub / "manifest.json").exists())
            self.assertEqual(dds.read_bytes()[:4], b"DDS ")

    def test_unknown_mode_exits_nonzero_before_the_source_or_output_is_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "package.rpf"     # never created: the mode is rejected before the source is opened
            out = root / "out"
            bad = run("invalidmode", src, out)
            self.assertNotEqual(bad.returncode, 0, bad.stdout + bad.stderr)
            self.assertIn("unknown mode 'invalidmode'", bad.stdout + bad.stderr)
            self.assertNotIn("ok=0 fail=0 skip=0", bad.stdout, "an unknown mode still reported a finished run")
            self.assertFalse(out.exists(), "an unknown mode created the output folder")
            self.assertEqual([p.name for p in root.iterdir()], [], "an unknown mode touched the work folder")

    def test_usage_lists_every_mode(self):
        short = run()
        self.assertEqual(short.returncode, 2, short.stdout + short.stderr)
        self.assertIn("usage: rpf9dump", short.stdout)
        for mode in ("ttpf", "models", "textures", "audio", "xma"):
            self.assertIn(mode, short.stdout, f"usage does not list {mode}")

    def test_start_past_eof_exits_2_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "tiny.rpf"
            src.write_bytes(bytes(16))
            for mode in ("models", "textures"):
                with self.subTest(mode=mode):
                    out = root / f"out-{mode}"
                    done = run(mode, src, out, "--start", "1024")
                    self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                    self.assertIn("--start", done.stderr)
                    self.assertNotIn("Unhandled exception", done.stderr)
                    self.assertFalse(out.exists(), f"{mode} left {out} behind on --start past EOF")

    def test_start_plus_length_overflow_is_clamped_and_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "tiny.rpf"
            src.write_bytes(bytes(16))
            for mode in ("models", "textures"):
                with self.subTest(mode=mode):
                    out = root / f"out-{mode}"
                    done = run(mode, src, out, "--start", "8", "--length", str((1 << 63) - 1))
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertNotIn("Unhandled exception", done.stderr)
                    manifest = json.loads((out / "tiny" / mode / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["Count"], 0)
                    self.assertEqual(manifest["Scan"]["Start"], 8)
                    self.assertEqual(manifest["Scan"]["Length"], 8)

    def test_short_source_under_64_bytes_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for size in (0, 16):
                src = root / f"tiny_{size}.rpf"
                src.write_bytes(bytes(size))
                for mode in ("models", "textures"):
                    with self.subTest(size=size, mode=mode):
                        out = root / f"out-{mode}-{size}"
                        done = run(mode, src, out)
                        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                        self.assertNotIn("Unhandled exception", done.stderr)
                        manifest = json.loads((out / f"tiny_{size}" / mode / "manifest.json").read_text(encoding="utf-8"))
                        self.assertEqual(manifest["Count"], 0)


if __name__ == "__main__":
    unittest.main()
